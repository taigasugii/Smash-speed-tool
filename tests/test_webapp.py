"""GUI（ローカル Web アプリ）のサーバー側のテスト。"""

import json
import time

import numpy as np
import pytest

from smash_speed.config import load_config
from smash_speed.validate import make_synthetic_session


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    cfg = load_config()
    cfg["synthetic"]["camera"].update(width=960, height=540, f_px=750.0)
    d = tmp_path_factory.mktemp("web")
    info = make_synthetic_session(cfg, d, speed=75.0, fps=120.0)
    return d, info


@pytest.fixture()
def client(tmp_path, monkeypatch, session):
    from smash_speed.webapp import server

    d, _ = session
    monkeypatch.setattr(server, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(server, "S", server.Session())
    server.S.state = {"out_root": str(d / "out")}
    return server.app.test_client()


def test_open_video_and_frames(client, session):
    d, _ = session
    r = client.post("/api/video", json={"path": str(d / "synthetic.mp4")}).get_json()
    assert r["ok"] and r["meta"]["width"] == 960 and len(r["meta"]["timestamps"]) == r["meta"]["n_frames"]
    img = client.get("/api/frame?i=10&w=480")
    assert img.status_code == 200 and img.data[:2] == b"\xff\xd8"  # JPEG
    lm = client.get("/api/landmarks").get_json()
    assert len(lm["landmarks"]) >= 23 and lm["segments"]


def test_calibration_solve_save_and_floor(client, session, tmp_path):
    d, info = session
    client.post("/api/video", json={"path": str(d / "synthetic.mp4")})
    pts = {p["name"]: p["image_px"] for p in json.loads((d / "calibration.json").read_text(encoding="utf-8"))["points"]}
    few = dict(list(pts.items())[:3])
    assert not client.post("/api/calib/solve", json={"points": few}).get_json()["ok"]  # 4 点未満
    r = client.post("/api/calib/solve", json={"points": pts}).get_json()
    assert r["ok"] and r["rmse"] < 1.5 and r["overlay"] and set(pts) <= set(r["predicted"])
    path = tmp_path / "calib.json"
    s = client.post("/api/calib/save", json={"points": pts, "path": str(path), "view": {"theta": 0}}).get_json()
    assert s["ok"] and path.exists() and s["calib"]["view"] == {"theta": 0}
    # 打つ人の足元を投影した画像座標 → 床の座標に戻る
    X = [info["hitter_xy"][0], info["hitter_xy"][1], 0.0]
    uv = client.post("/api/project", json={"points": [X]}).get_json()["points"][0]
    f = client.post("/api/floor", json={"u": uv[0], "v": uv[1]}).get_json()
    assert f["x"] == pytest.approx(X[0], abs=0.02) and f["y"] == pytest.approx(X[1], abs=0.02)
    assert not client.post("/api/floor", json={"u": 480, "v": 0}).get_json()["ok"]  # 地平線より上


def test_analyze_task_and_result(client, session):
    d, info = session
    client.post("/api/video", json={"path": str(d / "synthetic.mp4")})
    client.post("/api/state", json={"calib": str(d / "calibration.json"), "calib_video": str(d / "synthetic.mp4"),
                                    "start": 0.0, "end": info["t_land_video_s"] + 0.1, "shot": "web_test",
                                    "hitter": {"x": info["hitter_xy"][0], "y": info["hitter_xy"][1]},
                                    "landing": {"x": info["landing_xy"][0], "y": info["landing_xy"][1]},
                                    "audio": str(d / "synthetic.wav"), "overlay": False})
    client.post("/api/calib/load", json={})
    r = client.post("/api/run/analyze", json={}).get_json()
    assert r["ok"]
    tid = r["task"]
    assert not client.post("/api/run/analyze", json={}).get_json()["ok"]  # 実行中は二重に起動しない
    t0 = time.time()
    while time.time() - t0 < 300:
        st = client.get(f"/api/task/{tid}").get_json()
        if st["status"] != "running":
            break
        time.sleep(0.5)
    assert st["status"] == "done", "\n".join(client.get(f"/api/task/{tid}").get_json()["lines"])
    res = client.get("/api/result").get_json()
    assert res["ok"] and res["row"]["speed_initial_mps"] == pytest.approx(75.0, rel=0.05)
    assert "image_fit" in res["files"]
    lst = client.get("/api/results").get_json()["rows"]
    assert any(row["shot"] == "web_test" for row in lst)
    img = client.get("/file", query_string={"path": res["files"]["image_fit"]})
    assert img.status_code == 200


def test_labels_roundtrip_and_file_guard(client, session, tmp_path):
    d, _ = session
    client.post("/api/video", json={"path": str(d / "synthetic.mp4")})
    rows = [{"frame": 20, "x": 100.0, "y": 50.0, "visible": 1, "event": "hit", "source": "manual"},
            {"frame": 21, "x": None, "y": None, "visible": 0, "event": "", "source": "manual"},
            {"frame": 22, "x": None, "y": None, "visible": -1, "event": "", "source": ""}]
    out = tmp_path / "labels"
    r = client.post("/api/labels", json={"dir": str(out), "rows": rows, "start": 0.1, "end": 0.2}).get_json()
    assert r["ok"] and (out / "synthetic_ball.csv").exists()
    back = client.get("/api/labels", query_string={"dir": str(out)}).get_json()["rows"]
    assert [b["visible"] for b in back] == [1, 0, -1] and back[0]["event"] == "hit"
    assert client.get("/file", query_string={"path": "C:/Windows/win.ini"}).status_code == 404


def _wait(client, tid, limit=600):
    t0 = time.time()
    while time.time() - t0 < limit:
        st = client.get(f"/api/task/{tid}").get_json()
        if st["status"] != "running":
            return st
        time.sleep(0.5)
    raise TimeoutError


def test_fastest_smash_flow(client, session):
    """いちばん速いスマッシュを探す：候補探し → 打つ人・着地点の確認 → 詳しい解析 → 速い順。"""
    d, info = session
    client.post("/api/video", json={"path": str(d / "synthetic.mp4")})
    client.post("/api/state", json={"calib": str(d / "calibration.json"), "calib_video": str(d / "synthetic.mp4"),
                                    "audio": str(d / "synthetic.wav"), "overlay": False})
    client.post("/api/calib/load", json={})
    assert not client.post("/api/run/batch_analyze", json={}).get_json()["ok"]  # 先に候補を探す必要がある
    r = client.post("/api/run/batch_find", json={"select": 8}).get_json()
    assert r["ok"]
    st = _wait(client, r["task"])
    assert st["status"] == "done", "\n".join(client.get(f"/api/task/{r['task']}").get_json()["lines"])
    b = client.get("/api/batch").get_json()
    ok = [c for c in b["candidates"] if c["status"] == "ok"]
    assert len(ok) == 1 and ok[0]["thumb_path"]
    c = ok[0]
    u = client.post("/api/batch/update", json={"id": c["id"], "selected": True, "confirmed": True,
                                               "hitter": {"x": info["hitter_xy"][0], "y": info["hitter_xy"][1]},
                                               "landing": {"x": info["landing_xy"][0], "y": info["landing_xy"][1]}})
    assert u.get_json()["ok"]
    r = client.post("/api/run/batch_analyze", json={}).get_json()
    st = _wait(client, r["task"])
    assert st["status"] == "done", "\n".join(client.get(f"/api/task/{r['task']}").get_json()["lines"])
    c2 = [x for x in client.get("/api/batch").get_json()["candidates"] if x["id"] == c["id"]][0]
    assert c2["confirmed"] and c2["result"]["kmh"] == pytest.approx(75.0 * 3.6, rel=0.05)
    res = client.get("/api/result", query_string={"dir": c2["result"]["dir"]}).get_json()
    assert res["ok"] and res["row"]["speed_initial_mps"] == pytest.approx(75.0, rel=0.05)


def test_shot_list_flow(client, tmp_path):
    """ショットの一覧：裏で ``shots`` を動かし、種類・初速つきのショットを返す。区間を決めて解析し直せる。"""
    from smash_speed.physics import PhysicsParams
    from smash_speed.synth_rally import make_rally, rally_camera, render_rally_video

    cfg = load_config()
    phys = PhysicsParams.from_config(cfg)
    cam = rally_camera(cfg)
    rally = make_rally(np.random.default_rng(6), phys, n_shots=4)
    d = tmp_path / "rally"
    d.mkdir()
    render_rally_video(rally, cam, cfg, phys, d / "rally.mp4", fps=30.0, pre_s=0.3, audio_path=d / "rally.wav")
    (d / "calibration.json").write_text(json.dumps({"camera": cam.to_dict()}), encoding="utf-8")
    truth = rally.to_shots(0.3 - rally.t_start, phys=phys)

    assert client.get("/api/shots").get_json()["shots"] == []  # まだ解析していない
    client.post("/api/video", json={"path": str(d / "rally.mp4")})
    assert not client.post("/api/run/shots", json={}).get_json()["ok"]  # コートを合わせる前は始めない
    client.post("/api/state", json={"calib": str(d / "calibration.json"), "audio": str(d / "rally.wav")})
    r = client.post("/api/run/shots", json={}).get_json()
    assert r["ok"]
    st = _wait(client, r["task"])
    lines = client.get(f"/api/task/{r['task']}").get_json()["lines"]
    assert st["status"] == "done", "\n".join(lines)
    assert any("検出中" in s for s in lines)  # 進み具合が流れてくる
    g = client.get("/api/shots").get_json()
    assert g["ok"] and g["dir"].endswith("rally_shots") and g["files"]["shots"].endswith("shots.csv")
    assert len(g["shots"]) == len(truth) and g["info"]["n_hits"] == len(truth)
    assert [s["shot_type"] for s in g["shots"]] == [s.shot_type for s in truth]
    assert all(s["label"] == g["labels"][s["shot_type"]] for s in g["shots"])
    assert sum(v["n"] for v in g["info"]["by_type"].values()) == len(truth)
    for s, t in zip(g["shots"], truth):
        if t.shot_type != "serve":
            assert s["speed_kmh"] == pytest.approx(t.speed_kmh, rel=0.12)
    assert client.get("/file", query_string={"path": g["files"]["shots"]}).status_code == 200
    # 区間を決めて解析し直す（最初の 2 本の後から）
    t0 = (truth[1].t_hit_s + truth[2].t_hit_s) / 2
    r = client.post("/api/run/shots", json={"start": t0, "end": 100.0}).get_json()
    assert _wait(client, r["task"])["status"] == "done"
    g2 = client.get("/api/shots").get_json()
    assert g2["info"]["start_s"] == pytest.approx(t0, abs=1e-3)
    assert 0 < len(g2["shots"]) < len(truth) and all(s["t_hit_s"] >= t0 for s in g2["shots"])
