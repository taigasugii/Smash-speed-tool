"""フォームを比べる画面（E4）のサーバー側のテスト。"""

import numpy as np
import pytest

from smash_speed.form import FormOptions, analyze
from tests.test_form import INFO, SWINGS, SessionModel, _frames


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from smash_speed.webapp import server

    monkeypatch.setattr(server, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(server, "S", server.Session())
    server.S.state = {"out_root": str(tmp_path / "out")}
    return server.app.test_client()


def _analyze(out, video="clear_01.mp4"):
    analyze(video, out, FormOptions(), frames=_frames, info=INFO, onsets=[(c + 0.0204, 100.0) for c in SWINGS],
            model=SessionModel(SWINGS), progress=lambda s: None)


def test_form_list_and_pose(client, tmp_path):
    assert client.get("/api/form").get_json()["shots"] == []
    _analyze(tmp_path / "out" / "clear_01_form")
    _analyze(tmp_path / "out" / "clear_02_form", "clear_02.mp4")
    r = client.get("/api/form").get_json()
    assert r["ok"] and len(r["shots"]) == 6
    s = r["shots"][1]
    assert s["id"] == "clear_01_001000" and s["video"] == "clear_01.mp4" and s["date"]
    assert s["features"]["elbow_angle_deg"] == pytest.approx(140.0, abs=3.0)
    assert r["info"][0][0] == "wrist_height_ratio" and len(r["info"][0]) == 3
    p = client.get("/api/form/pose", query_string={"dir": s["dir"], "id": s["id"]}).get_json()
    assert p["ok"] and len(p["t"]) == len(p["xy"]) and len(p["xy"][0]) == 17
    i0 = int(np.argmin(np.abs(np.array(p["t"]))))
    assert p["series"]["elbow"][i0] == pytest.approx(140.0, abs=3.0)
    assert p["height"] > 3 and ["left_shoulder", "right_shoulder"] in [list(e) for e in p["skeleton"]]
    # 出力の場所の外のファイルは読まない
    bad = client.get("/api/form/pose", query_string={"dir": s["dir"], "id": "../../x"})
    assert bad.status_code == 400


def test_form_run_needs_a_video(client):
    r = client.post("/api/run/form", json={})
    assert r.status_code == 400 and "動画" in r.get_json()["error"]
