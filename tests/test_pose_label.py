"""関節・打点を付ける画面（pose_label.py・webapp/label_api.py）のテスト。合成の棒人間の動画を使う。"""

import numpy as np
import pytest

from smash_speed import pose_label as PL
from smash_speed.form import FormOptions, analyze
from smash_speed.pose import KEYPOINTS, PoseSequence, label_videos, load_labels, main as pose_main, pck
from smash_speed.pose_prep import SPEED_OF_SOUND_MPS
from smash_speed.synth_pose import SwingParams, render_swing_video, side_camera
from tests.test_form import INFO, SWINGS, SessionModel, _frames

TS = np.arange(600) / 60.0


def test_plan_frames_around_the_impact():
    plan = PL.plan_frames(5.0, TS)
    frames = [p["frame"] for p in plan]
    assert len(plan) == 10 and frames == sorted(frames)
    assert frames[4:7] == [299, 300, 301] and plan[5]["phase"] == "打った瞬間"
    assert frames[0] == 300 - 48 and frames[-1] == 327
    # 動画の始まりに近いと、前のコマは動画の外なので除く
    near = PL.plan_frames(0.3, TS)
    assert [p["phase"] for p in near][:2] == ["振り出し", "振り出し"] and near[0]["frame"] >= 0
    assert PL.plan_frames(1.0, []) == []


def _truth_joints(seq, k):
    xy = seq.xy[k].copy()
    vis = np.full(17, 2)
    vis[KEYPOINTS.index("left_wrist")] = 1      # 隠れている（推測）
    vis[KEYPOINTS.index("left_ear")] = 0        # 写っていない
    xy[KEYPOINTS.index("left_ear")] = np.nan
    return xy, vis


def test_save_merge_and_read_with_pck(tmp_path):
    cam = side_camera()
    from smash_speed.synth_pose import project_swing, swing_times
    truth = project_swing(SwingParams(), cam, swing_times(60.0), t_offset=2.0)
    d = tmp_path / "pose_labels"
    for k in (10, 50, 60):
        xy, vis = _truth_joints(truth, k)
        pr = PL.save_frame(d, "a.mp4", k, truth.t[k], xy, vis, "mediapipe")
    assert pr == {"frames": 3, "usable": 3, "target": 200, "per_video": {"a.mp4": 3}}
    # 同じコマは置き換える・別の動画は足す
    xy, vis = _truth_joints(truth, 50)
    PL.save_frame(d, "a.mp4", 50, truth.t[50], xy + 1.0, vis)
    pr = PL.save_frame(d, "b.mp4", 50, truth.t[50], xy, np.where(np.arange(17) == 11, 0, vis))
    assert pr["frames"] == 4 and pr["usable"] == 3 and pr["per_video"] == {"a.mp4": 3, "b.mp4": 1}
    assert PL.read_prefill(d / PL.PREFILL_FILE) == {("a.mp4", 10): "mediapipe", ("a.mp4", 50): "",
                                                     ("a.mp4", 60): "mediapipe", ("b.mp4", 50): ""}
    data = PL.read_joints(d / PL.JOINTS_FILE)
    assert data[("a.mp4", 50)]["xy"][0] == pytest.approx(truth.xy[50, 0] + 1.0, abs=0.01)
    # pose.load_labels でそのまま読める（複数の動画なら動画を選ぶ）
    assert label_videos(d / PL.JOINTS_FILE) == ["a.mp4", "b.mp4"]
    with pytest.raises(ValueError, match="複数の動画"):
        load_labels(d / PL.JOINTS_FILE)
    lab = load_labels(d / PL.JOINTS_FILE, "a.mp4")
    assert list(lab.frames) == [10, 50, 60] and lab.video == "a.mp4"
    assert lab.visible[0, KEYPOINTS.index("left_wrist")] == 1 and lab.visible[0, KEYPOINTS.index("left_ear")] == 0
    assert lab.t == pytest.approx(truth.t[[10, 50, 60]], abs=1e-5)
    r = pck(truth, lab)
    assert r["pck"] == 1.0 and r["n_frames"] == 3 and r["n"] == 3 * 15
    # 消す
    pr = PL.delete_frame(d, "a.mp4", 50)
    assert pr["frames"] == 3 and ("a.mp4", 50) not in PL.read_joints(d / PL.JOINTS_FILE)
    assert ("a.mp4", 50) not in PL.read_prefill(d / PL.PREFILL_FILE)


def test_pck_cli_picks_the_video_of_the_estimate(tmp_path, capsys):
    from smash_speed.synth_pose import project_swing, swing_times
    truth = project_swing(SwingParams(), side_camera(), swing_times(60.0))
    d = tmp_path / "pose_labels"
    for k in (20, 40):
        PL.save_frame(d, "a.mp4", k, truth.t[k], truth.xy[k], np.full(17, 2))
        PL.save_frame(d, "b.mp4", k, truth.t[k], truth.xy[k] + 50, np.full(17, 2))
    truth.meta["video"] = "a.mp4"
    truth.save(tmp_path / "a.npz")
    assert pose_main(["pck", str(d / PL.JOINTS_FILE), str(tmp_path / "a.npz")]) == 0
    assert "PCK@0.2 1.000" in capsys.readouterr().out
    assert pose_main(["pck", str(d / PL.JOINTS_FILE), str(tmp_path / "a.npz"), "--video", "b.mp4"]) == 0
    assert "PCK@0.2 0.000" in capsys.readouterr().out
    truth.meta.pop("video")
    truth.save(tmp_path / "x.npz")
    assert pose_main(["pck", str(d / PL.JOINTS_FILE), str(tmp_path / "x.npz")]) == 2


def test_check_joints():
    j = [None] * 17
    j[0] = [10, 20, 2]
    j[5] = [1, 2, 0]
    xy, vis = PL.check_joints(j)
    assert vis[0] == 2 and vis[5] == 0 and np.isnan(xy[5]).all() and xy[0].tolist() == [10, 20]
    with pytest.raises(ValueError):
        PL.check_joints([None] * 16)
    with pytest.raises(ValueError):
        PL.check_joints([[1, 2, 3]] + [None] * 16)


def test_impacts_add_replace_delete(tmp_path):
    d = tmp_path / "pose_labels"
    PL.add_impact(d, "a.mp4", 300, 5.0)
    PL.add_impact(d, "a.mp4", 600, 10.0)
    PL.add_impact(d, "b.mp4", 300, 5.0)
    rows = PL.add_impact(d, "a.mp4", 302, 5.0333)       # 同じスイングの付け直し
    assert [(r["frame"], r["t_s"]) for r in rows] == [(302, pytest.approx(5.0333)), (600, 10.0)]
    assert len(PL.read_impacts(d / PL.IMPACTS_FILE)) == 3
    assert [r["frame"] for r in PL.delete_impact(d, "a.mp4", 600)] == [302]
    head = (d / PL.IMPACTS_FILE).read_text(encoding="utf-8-sig").splitlines()[0]
    assert head == ",".join(PL.IMPACT_COLUMNS)


def test_impact_report_against_the_form_analysis(tmp_path, capsys):
    out = tmp_path / "out"
    delay = 7.0 / SPEED_OF_SOUND_MPS
    analyze("clear_01.mp4", out / "clear_01_form", FormOptions(), frames=_frames, info=INFO,
            onsets=[(c + delay, 100.0) for c in SWINGS], model=SessionModel(SWINGS), progress=lambda s: None)
    d = out / PL.LABELS_DIR
    for c in SWINGS:
        PL.add_impact(d, "clear_01.mp4", int(round(c * 60)), c + 1 / 60)   # 手は 1 コマ遅く付けた
    PL.add_impact(d, "clear_01.mp4", 1080, 18.0)                           # 解析にないスイング
    r = PL.impact_report(PL.read_impacts(d / PL.IMPACTS_FILE), out)
    assert r["n"] == 3 and r["unmatched"] == 1 and r["methods"] == {"audio": 3}
    assert r["mean_abs_s"] == pytest.approx(1 / 60, abs=1e-4) and r["median_s"] == pytest.approx(-1 / 60, abs=1e-4)
    assert r["mean_abs_frames"] == pytest.approx(1.0, abs=0.01)
    # 手首の速さの最大は打点の約 0.03 秒前（合成のスイング）。手の打点は 1 コマ（0.017 秒）遅い
    assert r["n_wrist"] == 3 and r["wrist_offset_s"] == pytest.approx(0.03 + 1 / 60, abs=0.01)
    assert PL.main(["impacts", str(d / PL.IMPACTS_FILE), str(out)]) == 0
    text = capsys.readouterr().out
    assert "組にできた 3" in text and "wrist_offset_s" in text


# --- 画面の API ------------------------------------------------------------------

@pytest.fixture(scope="module")
def videos(tmp_path_factory):
    root = tmp_path_factory.mktemp("videos")
    cam = side_camera(distance_m=4.0, width=640, height=360)
    swings = {1.6: SwingParams(), 4.0: SwingParams()}
    truth = render_swing_video(root / "clear_01.mp4", swings, cam, audio=True)
    (root / "sub").mkdir()
    render_swing_video(root / "sub" / "clear_02.mp4", {1.6: SwingParams()}, cam)
    (root / "notes.txt").write_text("x")
    secret = tmp_path_factory.mktemp("outside") / "secret.mp4"
    secret.write_bytes((root / "clear_01.mp4").read_bytes())
    return root, truth, swings, secret


@pytest.fixture()
def client(tmp_path, monkeypatch, videos):
    from smash_speed.webapp import label_api, server

    monkeypatch.setattr(server, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(server, "S", server.Session())
    monkeypatch.setattr(label_api, "_ONSETS", {})
    server.S.state = {"out_root": str(tmp_path / "out"), "label_videos_dir": str(videos[0])}
    return server.app.test_client()


def test_api_lists_videos_and_serves_frames_only_inside_the_folder(client, videos):
    _, truth, _, secret = videos
    r = client.get("/api/poselab/videos").get_json()
    assert r["ok"] and r["root_from"] == "setting" and r["progress"]["frames"] == 0
    assert [v["rel"] for v in r["videos"]] == ["clear_01.mp4", "sub/clear_02.mp4"]
    m = client.get("/api/poselab/video", query_string={"video": "clear_01.mp4"}).get_json()
    assert m["ok"] and m["meta"]["n_frames"] == len(truth) and m["meta"]["width"] == 640 and m["swings"] == []
    img = client.get("/api/poselab/frame", query_string={"video": "sub/clear_02.mp4", "i": 30})
    assert img.status_code == 200 and img.mimetype == "image/jpeg" and len(img.data) > 1000
    for bad in ("../" + secret.parent.name + "/secret.mp4", str(secret), "notes.txt", "none.mp4", ""):
        assert client.get("/api/poselab/frame", query_string={"video": bad, "i": 0}).status_code == 404
        assert client.get("/api/poselab/video", query_string={"video": bad}).status_code == 400
        assert client.post("/api/poselab/joints", json={"video": bad, "frame": 0, "joints": [None] * 17}).status_code == 400


def test_api_joints_roundtrip(client, videos, tmp_path):
    _, truth, _, _ = videos
    k = 96
    joints = [[round(float(x), 2), round(float(y), 2), 2] for x, y in truth.xy[k]]
    joints[3] = None
    joints[9][2] = 1
    r = client.post("/api/poselab/joints", json={"video": "clear_01.mp4", "frame": k, "joints": joints,
                                                 "prefill": "form"}).get_json()
    assert r["ok"] and r["progress"]["frames"] == 1 and r["progress"]["usable"] == 1
    got = client.get("/api/poselab/joints", query_string={"video": "clear_01.mp4", "frame": k}).get_json()
    assert got["prefill"] == "form" and got["joints"][3] is None and got["joints"][9][2] == 1
    assert got["joints"][0][:2] == pytest.approx(joints[0][:2], abs=0.01)
    lab = load_labels(tmp_path / "out" / PL.LABELS_DIR / PL.JOINTS_FILE)
    assert lab.t[0] == pytest.approx(truth.t[k], abs=1e-4) and pck(truth, lab)["pck"] == 1.0
    m = client.get("/api/poselab/video", query_string={"video": "clear_01.mp4"}).get_json()
    assert m["done"] == {str(k): {"n": 16, "prefill": "form"}}
    assert client.post("/api/poselab/joints", json={"video": "clear_01.mp4", "frame": 10 ** 6,
                                                    "joints": joints}).status_code == 400
    assert client.post("/api/poselab/joints", json={"video": "clear_01.mp4", "frame": k,
                                                    "joints": joints[:5]}).status_code == 400
    r = client.post("/api/poselab/joints/delete", json={"video": "clear_01.mp4", "frame": k}).get_json()
    assert r["ok"] and r["progress"]["frames"] == 0
    assert client.get("/api/poselab/joints", query_string={"video": "clear_01.mp4", "frame": k}).get_json()["joints"] is None


def test_api_impacts_onsets_and_swings(client, videos):
    _, _, swings, _ = videos
    r = client.post("/api/poselab/onsets", json={"video": "clear_01.mp4"}).get_json()
    assert r["ok"] and r["n"] == 2 and [s["source"] for s in r["swings"]] == ["打った音", "打った音"]
    assert [s["t"] for s in r["swings"]] == pytest.approx(sorted(swings), abs=0.01)
    assert [p["frame"] for p in r["swings"][0]["plan"]][4:7] == [95, 96, 97]
    r = client.post("/api/poselab/impacts", json={"video": "clear_01.mp4", "frame": 97}).get_json()
    assert r["ok"] and r["impacts"] == [{"video": "clear_01.mp4", "frame": 97, "t_s": pytest.approx(97 / 60, abs=1e-4)}]
    # 手で付けた打点が、近くの音より先に来る
    assert [s["source"] for s in r["swings"]] == ["手で付けた打点", "打った音"]
    r = client.post("/api/poselab/impacts", json={"video": "clear_01.mp4", "frame": 96}).get_json()
    assert [i["frame"] for i in r["impacts"]] == [96]
    assert client.get("/api/poselab/videos").get_json()["videos"][0]["impacts"] == 1
    r = client.post("/api/poselab/impacts/delete", json={"video": "clear_01.mp4", "frame": 96}).get_json()
    assert r["impacts"] == []


def test_api_prefill_from_the_form_result_and_a_model(client, videos, tmp_path, monkeypatch):
    from smash_speed.webapp import label_api

    _, truth, _, _ = videos
    # フォームの解析の結果（推定したままの姿勢）にあるコマは、それを下書きにする
    raw = PoseSequence(t=truth.t[60:130], xy=truth.xy[60:130], score=np.full((70, 17), 0.9), width=640, height=360,
                       fps=60.0, model="fake")
    form = tmp_path / "out" / "clear_01_form" / "form"
    raw.save(form / "raw" / "clear_01_000160.npz")
    (form / "form.json").write_text('{"video": "C:/videos/clear_01.mp4"}', encoding="utf-8")
    (form / "form_features.csv").write_text("shot_id,video,t_hit_s,t_impact_s\nclear_01_000160,clear_01.mp4,1.6,1.6\n",
                                            encoding="utf-8")
    r = client.get("/api/poselab/prefill", query_string={"video": "clear_01.mp4", "frame": 96, "model": "form"}).get_json()
    assert r["ok"] and len(r["people"]) == 1
    assert np.array(r["people"][0]["xy"], float) == pytest.approx(truth.xy[96], abs=0.1)
    r = client.get("/api/poselab/prefill", query_string={"video": "clear_01.mp4", "frame": 200, "model": "form"}).get_json()
    assert r["ok"] and r["people"] == []
    m = client.get("/api/poselab/video", query_string={"video": "clear_01.mp4"}).get_json()
    assert m["has_form"] and m["swings"][0]["source"] == "フォームの解析"
    # モデル：大きく写っている人から並べる
    small = (truth.xy[96] - truth.xy[96].mean(0)) * 0.3 + 100
    monkeypatch.setattr(label_api, "available_models", lambda: {"mediapipe": True, "rtmpose": False})
    monkeypatch.setattr(label_api, "_MODELS", {"mediapipe": lambda img: [(small, np.full(17, 0.9)),
                                                                         (truth.xy[96], np.full(17, 0.8))]})
    r = client.get("/api/poselab/prefill", query_string={"video": "clear_01.mp4", "frame": 96,
                                                         "model": "mediapipe"}).get_json()
    assert r["ok"] and len(r["people"]) == 2 and r["people"][0]["score"][0] == 0.8
    bad = client.get("/api/poselab/prefill", query_string={"video": "clear_01.mp4", "frame": 96, "model": "rtmpose"})
    assert bad.status_code == 400


def test_videos_root_falls_back_to_the_agent_setting_and_the_open_video(tmp_path, monkeypatch, videos):
    from smash_speed.webapp import label_api

    monkeypatch.setenv("SMASH_AGENT_DIR", str(tmp_path / "agent"))
    assert label_api.videos_root({}) == (None, "")
    assert label_api.videos_root({"video": str(videos[0] / "sub" / "clear_02.mp4")}) == (videos[0] / "sub", "video")
    (tmp_path / "agent" / ".agent").mkdir(parents=True)
    (tmp_path / "agent" / ".agent" / "config.json").write_text(f'{{"videos_dir": "{videos[0].as_posix()}"}}')
    assert label_api.videos_root({"video": "x"}) == (videos[0], "agent")
