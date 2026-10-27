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
