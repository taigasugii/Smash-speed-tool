"""まとめて解析（video_queue.py・GUI の /api/queue。ROADMAP v2 の L3）のテスト。"""

import json
import sys
import time
from pathlib import Path

import pytest

from smash_speed import video_queue as vq


def _videos(d: Path, n: int = 3) -> list[Path]:
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(n):
        p = d / f"drill_{i}.mp4"
        p.write_bytes(b"0" * 1000)
        t = time.time() - 100 + i                     # 古い順に並ぶように
        import os
        os.utime(p, (t, t))
        out.append(p)
    (d / "memo.txt").write_text("動画ではない", encoding="utf-8")
    return out


def test_list_videos_and_commands(tmp_path):
    vids = _videos(tmp_path / "videos")
    out = tmp_path / "out"
    # 1 本目は基礎打ちの解析が済み、2 本目はコート合わせがある
    (vq.drill_dir(out, "drill_0")).mkdir(parents=True)
    (vq.drill_dir(out, "drill_0") / vq.DRILL_DONE).write_text("{}", encoding="utf-8")
    vq.calib_path(out, "drill_1").parent.mkdir(parents=True, exist_ok=True)
    vq.calib_path(out, "drill_1").write_text("{}", encoding="utf-8")
    items = vq.list_videos(tmp_path / "videos", out)
    assert [it.name for it in items] == ["drill_0.mp4", "drill_1.mp4", "drill_2.mp4"]
    assert [it.todo for it in items] == [[], ["drill", "shots"], ["drill"]]
    assert items[0].done and not items[1].done
    cli = lambda *a: ["py", "-m", "smash_speed.cli", *[str(x) for x in a]]   # noqa: E731
    cmds = vq.commands(items[1], out, cli, python="py")
    assert cmds[0][:3] == ["py", "-m", "smash_speed.drill"] and "--baseline-detector" in cmds[0]
    assert cmds[0][cmds[0].index("--out") + 1] == str(vq.drill_dir(out, "drill_1"))
    assert cmds[1][3] == "shots" and str(vq.calib_path(out, "drill_1")) in cmds[1]
    assert "--baseline-detector" not in vq.commands(items[2], out, cli, model=True, python="py")[0]
    assert vq.list_videos(tmp_path / "nothing", out) == []
    assert vids


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from smash_speed.webapp import server

    monkeypatch.setattr(server, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(server, "S", server.Session())
    server.S.state = {"out_root": str(tmp_path / "out"), "videos_dir": str(tmp_path / "videos")}
    server.TASKS.clear()
    return server.app.test_client()


def _fake_commands(item, out_root, cli, model=False, python=None):
    """本当の解析の代わりに、少し待ってから終わりのファイルを書くコマンド。"""
    d = vq.drill_dir(out_root, Path(item.path).stem)
    code = (f"import time, pathlib; time.sleep(1.5); p = pathlib.Path({str(d)!r}); p.mkdir(parents=True, exist_ok=True); "
            f"(p / {vq.DRILL_DONE!r}).write_text('{{}}'); print('done', {item.name!r})")
    return [[sys.executable, "-c", code]]


def _wait(client, tid, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = client.get(f"/api/task/{tid}?since=0").get_json()
        if r["status"] != "running":
            return r
        time.sleep(0.2)
    raise AssertionError("終わりませんでした")


def test_queue_runs_in_order_stops_and_resumes(client, tmp_path, monkeypatch):
    """3 本を順に解析し、途中で止め、もう一度押すと残りだけを続ける。"""
    monkeypatch.setattr(vq, "commands", _fake_commands)
    vids = _videos(tmp_path / "videos")
    r = client.get("/api/queue").get_json()
    assert r["ok"] and [it["name"] for it in r["items"]] == ["drill_0.mp4", "drill_1.mp4", "drill_2.mp4"]
    assert not any(it["done"] for it in r["items"])
    paths = [str(p) for p in vids]
    tid = client.post("/api/run/queue", json={"paths": paths}).get_json()["task"]
    # 1 本目が終わったら止める
    t0 = time.time()
    while not (vq.drill_dir(tmp_path / "out", "drill_0") / vq.DRILL_DONE).exists():
        assert time.time() - t0 < 30
        time.sleep(0.1)
    client.post(f"/api/task/{tid}/cancel")
    r = _wait(client, tid)
    assert r["status"] == "cancelled"
    done = [it["done"] for it in client.get("/api/queue").get_json()["items"]]
    assert done[0] and not done[2]
    # 続きから：終わった動画はとばす
    tid = client.post("/api/run/queue", json={"paths": paths}).get_json()["task"]
    r = _wait(client, tid)
    assert r["status"] == "done"
    names = [ln[2:] for ln in r["lines"] if ln.startswith("▶ ")]
    assert names == [n for n, d in zip(["drill_0.mp4", "drill_1.mp4", "drill_2.mp4"], done) if not d]
    assert all(it["done"] for it in client.get("/api/queue").get_json()["items"])
    # 全部終わっていれば、始めない
    assert not client.post("/api/run/queue", json={"paths": paths}).get_json()["ok"]


def test_queue_keeps_going_after_a_failure(client, tmp_path, monkeypatch):
    """1 本がうまくいかなくても、次の動画へ進む（最後は「失敗」として知らせる）。"""
    def cmds(item, out_root, cli, model=False, python=None):
        if item.name == "drill_0.mp4":
            return [[sys.executable, "-c", "import sys; sys.exit(3)"]]
        return _fake_commands(item, out_root, cli)

    monkeypatch.setattr(vq, "commands", cmds)
    vids = _videos(tmp_path / "videos", 2)
    tid = client.post("/api/run/queue", json={"paths": [str(p) for p in vids]}).get_json()["task"]
    r = _wait(client, tid)
    assert r["status"] == "failed"
    items = client.get("/api/queue").get_json()["items"]
    assert [it["done"] for it in items] == [False, True]


def test_queue_without_folder(client, tmp_path):
    from smash_speed.webapp import server

    server.S.state["videos_dir"] = str(tmp_path / "missing")
    r = client.get("/api/queue").get_json()
    assert r["ok"] and r["items"] == [] and r["error"]
    assert json.dumps(r)


def test_home_status_follows_the_first_run_order(tmp_path, monkeypatch):
    """ホームの「この動画の進み具合」：動画 → コート合わせ（基礎打ちの解析は先にできる）→ ショットの一覧 → 見る（ROADMAP v2 の L2）。"""
    from smash_speed.config import load_config
    from smash_speed.validate import make_synthetic_session
    from smash_speed.webapp import server

    cfg = load_config()
    cfg["synthetic"]["camera"].update(width=640, height=360, f_px=500.0)
    d = tmp_path / "web"
    make_synthetic_session(cfg, d, speed=60.0, fps=60.0)
    monkeypatch.setattr(server, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(server, "S", server.Session())
    out = tmp_path / "out"
    server.S.state = {"out_root": str(out)}
    server.TASKS.clear()
    client = server.app.test_client()
    assert client.get("/api/home").get_json()["next"] == "video"
    video = d / "synthetic.mp4"
    client.post("/api/video", json={"path": str(video)})
    r = client.get("/api/home").get_json()
    assert r["video"] == "synthetic.mp4" and not r["calibrated"] and r["next"] == "calib" and not r["drill"]["done"]
    # 基礎打ちの解析は、コート合わせの前にできる（本当の解析の代わりに、終わりのファイルを書くコマンド）
    monkeypatch.setattr(vq, "commands", lambda item, root, cli, model=False, python=None: [[sys.executable, "-c",
        f"import json, pathlib; p = pathlib.Path({str(vq.drill_dir(root, 'synthetic'))!r}); p.mkdir(parents=True); "
        f"(p / {vq.DRILL_DONE!r}).write_text(json.dumps({{'n_hits': 12, 'hits_per_min': 40.0}}))"]])
    tid = client.post("/api/run/drill", json={}).get_json()["task"]
    _wait(client, tid)
    r = client.get("/api/home").get_json()
    assert r["drill"]["done"] and r["drill"]["n_hits"] == 12 and r["next"] == "calib"
    # コート合わせ → 次はショットの一覧 → できたら見る
    client.post("/api/calib/load", json={"path": str(d / "calibration.json")})
    assert client.get("/api/home").get_json()["next"] == "shots"
    sd = out / "synthetic_shots"
    sd.mkdir(parents=True)
    (sd / "shot_list.json").write_text(json.dumps({"video": str(video), "n_hits": 5, "n_rallies": 1}), encoding="utf-8")
    r = client.get("/api/home").get_json()
    assert r["shots"]["done"] and r["shots"]["n_shots"] == 5 and r["next"] == "view"


def test_home_drill_highlights(tmp_path, monkeypatch):
    """ホームの基礎打ちの行の「見どころの動画を作る」：解析の前は断り、書き出した後は「見どころを開く」の場所を返す（K2）。"""
    from smash_speed.config import load_config
    from smash_speed.validate import make_synthetic_session
    from smash_speed.webapp import server

    cfg = load_config()
    cfg["synthetic"]["camera"].update(width=640, height=360, f_px=500.0)
    d = tmp_path / "web"
    make_synthetic_session(cfg, d, speed=60.0, fps=60.0)
    monkeypatch.setattr(server, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(server, "S", server.Session())
    out = tmp_path / "out"
    server.S.state = {"out_root": str(out)}
    server.TASKS.clear()
    client = server.app.test_client()
    client.post("/api/video", json={"path": str(d / "synthetic.mp4")})
    assert not client.post("/api/run/drill_highlights", json={}).get_json()["ok"]
    dd = vq.drill_dir(out, "synthetic")
    (dd / "highlights").mkdir(parents=True)
    (dd / vq.DRILL_DONE).write_text(json.dumps({"n_hits": 3}), encoding="utf-8")
    (dd / "highlights" / "highlights.json").write_text(json.dumps({"joined": "highlights.mp4", "clips": [{}, {}, {}]}),
                                                     encoding="utf-8")
    r = client.get("/api/home").get_json()
    assert r["drill"]["n_highlights"] == 3 and r["drill"]["highlights"].endswith("highlights.mp4")
