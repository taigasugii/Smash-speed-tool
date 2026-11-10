"""離れた端末から使うときの入口（Tailscale の端末だけ・合言葉）のテスト。"""

import io

import pytest

from smash_speed.webapp import remote
from smash_speed.webapp.server import REMOTE, app

PASS = "correct-horse-battery"
TAILNET_IP = "100.101.102.103"


@pytest.fixture
def client(tmp_path):
    cfg = tmp_path / "remote_access.json"
    remote.set_passphrase(PASS, cfg)
    old = (REMOTE.path, REMOTE.enabled)
    REMOTE.path, REMOTE.enabled, REMOTE.fails = cfg, True, {}
    try:
        yield app.test_client()
    finally:
        REMOTE.path, REMOTE.enabled = old
        REMOTE.fails = {}


def _get(c, url, ip):
    return c.get(url, environ_base={"REMOTE_ADDR": ip})


def test_this_pc_needs_no_passphrase(client):
    r = _get(client, "/api/state", "127.0.0.1")
    assert r.status_code == 200 and r.get_json()["remote"] is False


def test_other_networks_are_refused_even_with_passphrase(client):
    for ip in ("192.168.1.20", "8.8.8.8", "10.0.0.5"):
        r = client.post("/login", data={"passphrase": PASS}, environ_base={"REMOTE_ADDR": ip})
        assert r.status_code == 403
        assert _get(client, "/api/state", ip).status_code == 403


def test_tailnet_device_must_log_in(client):
    assert _get(client, "/api/state", TAILNET_IP).status_code == 401
    page = _get(client, "/", TAILNET_IP)
    assert page.status_code == 302 and page.headers["Location"].endswith("/login")
    assert _get(client, "/login", TAILNET_IP).status_code == 200
    bad = client.post("/login", data={"passphrase": "wrong-passphrase"}, environ_base={"REMOTE_ADDR": TAILNET_IP})
    assert bad.status_code == 401
    good = client.post("/login", data={"passphrase": PASS}, environ_base={"REMOTE_ADDR": TAILNET_IP})
    assert good.status_code == 302
    r = _get(client, "/api/state", TAILNET_IP)
    assert r.status_code == 200 and r.get_json()["remote"] is True


def test_every_route_is_behind_the_guard(client):
    """あとから足した画面の API（ショットの一覧・フォーム・関節・レポートなど）も含め、すべての道が入口の制限を通る。"""
    public = set(remote.PUBLIC_PATHS)
    n = 0
    for rule in app.url_map.iter_rules():
        url = rule.rule.replace("<path:name>", "core.js").replace("<kind>", "shots").replace("<tid>", "x")
        assert "<" not in url, f"テストが知らない引数つきの道: {rule.rule}"
        if url in public:
            continue
        for method in sorted(rule.methods - {"HEAD", "OPTIONS"}):
            outside = client.open(url, method=method, environ_base={"REMOTE_ADDR": "192.168.1.20"})
            assert outside.status_code == 403, (method, url)
            tail = client.open(url, method=method, environ_base={"REMOTE_ADDR": TAILNET_IP})
            assert tail.status_code in (401, 302), (method, url, tail.status_code)
            if tail.status_code == 302:
                assert tail.headers["Location"].endswith("/login"), (method, url)
            n += 1
    assert n > 60   # 道が読めていること（今は 70 ほど）


def test_repeated_wrong_passphrase_locks_out(client):
    for _ in range(remote.MAX_FAILS):
        client.post("/login", data={"passphrase": "nope-nope-nope"}, environ_base={"REMOTE_ADDR": TAILNET_IP})
    r = client.post("/login", data={"passphrase": PASS}, environ_base={"REMOTE_ADDR": TAILNET_IP})
    assert r.status_code == 429


def test_changing_passphrase_logs_everyone_out(client):
    client.post("/login", data={"passphrase": PASS}, environ_base={"REMOTE_ADDR": TAILNET_IP})
    assert _get(client, "/api/state", TAILNET_IP).status_code == 200
    remote.set_passphrase("another-passphrase", REMOTE.path)
    assert _get(client, "/api/state", TAILNET_IP).status_code == 401


def test_remote_device_cannot_shut_down_or_open_desktop_dialogs(client):
    client.post("/login", data={"passphrase": PASS}, environ_base={"REMOTE_ADDR": TAILNET_IP})
    for url in ("/api/shutdown", "/api/pick", "/api/open"):
        r = client.post(url, json={}, environ_base={"REMOTE_ADDR": TAILNET_IP})
        assert r.status_code == 400 and r.get_json()["ok"] is False


def test_upload_saves_video_and_rejects_other_files(client, tmp_path, monkeypatch):
    from smash_speed.webapp import server

    monkeypatch.setattr(server, "UPLOAD_DIR", tmp_path / "uploads")
    client.post("/login", data={"passphrase": PASS}, environ_base={"REMOTE_ADDR": TAILNET_IP})
    env = {"REMOTE_ADDR": TAILNET_IP}
    r = client.post("/api/upload", data={"file": (io.BytesIO(b"fake"), "smash.MOV")}, environ_base=env,
                    content_type="multipart/form-data")
    assert r.get_json()["ok"] and (tmp_path / "uploads" / "smash.MOV").read_bytes() == b"fake"
    r2 = client.post("/api/upload", data={"file": (io.BytesIO(b"x"), "smash.MOV")}, environ_base=env,
                     content_type="multipart/form-data")
    assert r2.get_json()["path"].endswith("smash_1.MOV")
    bad = client.post("/api/upload", data={"file": (io.BytesIO(b"x"), "evil.exe")}, environ_base=env,
                      content_type="multipart/form-data")
    assert bad.get_json()["ok"] is False


def test_short_passphrase_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        remote.set_passphrase("short", tmp_path / "x.json")


def test_tailnet_address_ranges():
    assert remote.is_tailnet("100.64.0.1") and remote.is_tailnet("100.127.255.254")
    assert not remote.is_tailnet("100.128.0.1") and not remote.is_tailnet("192.168.0.2")
    assert remote.is_tailnet("fd7a:115c:a1e0::1") and remote.is_tailnet("::ffff:100.100.1.1")
    assert remote.is_local("127.0.0.1") and remote.is_local("::1") and not remote.is_local("100.100.1.1")


def test_launch_arguments():
    """起動の引数：``--port 8768`` と ``--port=8768`` のどちらも読み、知らない引数・``--help`` では起動しない。"""
    from smash_speed.webapp import parse_args

    a = parse_args(["--remote", "--no-browser", "--port", "8768"])
    assert a.remote and a.no_browser and a.port == 8768 and not a.set_passphrase
    assert parse_args(["--port=8766"]).port == 8766
    assert parse_args([]).port is None and not parse_args([]).remote
    for bad in (["--help"], ["--prot=1"], ["remote"]):
        with pytest.raises(SystemExit):
            parse_args(bad)
