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


def test_shot_labels_api(client, tmp_path):
    """種類の正解ラベル（B6b）：保存すると一覧に付き、解析し直した一覧（時刻が少しずれる）にも付き直す。"""
    from smash_speed.shots import Shot, write_shots

    out = tmp_path / "IMG_1_shots"
    out.mkdir()
    shots = [Shot(t_hit_s=1.0, side="h", shot_type="clear", speed_kmh=150.0),
             Shot(t_hit_s=2.0, side="o", shot_type="smash", speed_kmh=300.0),
             Shot(t_hit_s=3.0, side="h", shot_type="unknown")]
    write_shots(out / "shots.csv", shots)
    (out / "shot_list.json").write_text(json.dumps({"video": r"C:\v\IMG_1.MOV", "by_type": {}}), encoding="utf-8")
    client.post("/api/state", json={"shots_dir": str(out), "out_root": str(tmp_path / "out")})

    g = client.get("/api/shots").get_json()
    assert [s["truth"] for s in g["shots"]] == [None, None, None]
    assert g["label_stats"]["all"]["n"] == 0 and g["label_stats"]["all"]["rule_accuracy"] is None
    assert not client.post("/api/shot_label", json={"i": 9, "label": "drop"}).get_json()["ok"]
    assert not client.post("/api/shot_label", json={"i": 0, "label": "kick"}).get_json()["ok"]
    assert not client.post("/api/shot_label", json={"i": 0, "label": "unknown"}).get_json()["ok"]
    r = client.post("/api/shot_label", json={"i": 0, "label": "drop"}).get_json()
    assert r["ok"] and r["truth"] == "drop"
    r = client.post("/api/shot_label", json={"i": 1, "label": "smash"}).get_json()
    st = r["label_stats"]
    assert st["video"]["n"] == st["all"]["n"] == 2 and st["all"]["rule_accuracy"] == pytest.approx(0.5)
    assert (tmp_path / "out" / "shot_labels.csv").exists()  # 出力の場所に 1 つ
    assert [s["truth"] for s in client.get("/api/shots").get_json()["shots"]] == ["drop", "smash", None]
    # 空にすると消える
    r = client.post("/api/shot_label", json={"i": 0, "label": ""}).get_json()
    assert r["ok"] and r["truth"] is None and r["label_stats"]["all"]["n"] == 1
    # 解析し直して時刻が少しずれても付き直す。別の動画の一覧には付かないが、全部の本数には入る
    write_shots(out / "shots.csv", [Shot(t_hit_s=1.02, shot_type="clear"), Shot(t_hit_s=2.03, shot_type="drive")])
    g = client.get("/api/shots").get_json()
    assert [s["truth"] for s in g["shots"]] == [None, "smash"]
    (out / "shot_list.json").write_text(json.dumps({"video": "/x/IMG_2.mp4", "by_type": {}}), encoding="utf-8")
    g = client.get("/api/shots").get_json()
    assert [s["truth"] for s in g["shots"]] == [None, None]
    assert g["label_stats"]["video"]["n"] == 0 and g["label_stats"]["all"]["n"] == 1


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
    # 1 本の場面と軌道（B5c）：軌道は打点から始まり、画像の上の点はシャトルの検出の近くを通る
    assert not client.get("/api/shot_view", query_string={"i": 99}).get_json()["ok"]
    for k, s in enumerate(g["shots"]):
        v = client.get("/api/shot_view", query_string={"i": k}).get_json()
        assert v["ok"] and v["court"] and v["image_size"]
        assert v["t_hit_s"] < v["t_end_s"] and v["frame"] >= 0
        # 場面を前後に送れるコマ（B5c2）：打つ少し前から次の打球まで、時刻の順。見えたコマは検出と同じ点
        fr = v["frames"]
        assert v["frame"] in [f[0] for f in fr] and np.all(np.diff([f[1] for f in fr]) > 0)
        assert fr[0][1] < v["t_hit_s"] and fr[0][1] >= v["t_hit_s"] - 0.2 - 1e-6 and fr[-1][1] <= v["t_end_s"] + 0.05 + 1e-6
        seen = {f[1]: (f[2], f[3]) for f in fr if f[2] is not None}
        assert all(seen[t] == (x, y) for t, x, y in v["detections"])
        if s["speed_kmh"] is None:
            continue
        assert v["path"][0] == pytest.approx(v["hit"], abs=1e-3) and len(v["path"]) == len(v["path_px"]) == len(v["path_t"])
        assert v["path_t"][0] == pytest.approx(v["t_hit_s"], abs=1e-3) and np.all(np.diff(v["path_t"]) >= 0)
        if k + 1 < len(g["shots"]):
            assert v["t_end_s"] <= g["shots"][k + 1]["t_hit_s"] + 1e-6
        assert len(v["detections"]) >= 3
        px = np.array([p for p in v["path_px"] if p is not None])
        near = [np.min(np.hypot(px[:, 0] - x, px[:, 1] - y)) for _, x, y in v["detections"]]
        assert np.median(near) < 15
    # 区間を決めて解析し直す（最初の 2 本の後から）
    t0 = (truth[1].t_hit_s + truth[2].t_hit_s) / 2
    r = client.post("/api/run/shots", json={"start": t0, "end": 100.0}).get_json()
    assert _wait(client, r["task"])["status"] == "done"
    g2 = client.get("/api/shots").get_json()
    assert g2["info"]["start_s"] == pytest.approx(t0, abs=1e-3)
    assert 0 < len(g2["shots"]) < len(truth) and all(s["t_hit_s"] >= t0 for s in g2["shots"])
