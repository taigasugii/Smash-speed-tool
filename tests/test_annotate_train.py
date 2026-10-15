"""Phase 5 / 6：アノテーション、疑似低 fps データ、学習データの書き出し、検出の評価、TrackNet ラッパー。"""

import json

import numpy as np
import pandas as pd
import pytest

from smash_speed import synth
from smash_speed.annotate import AnnotationState, normalize_labels, save_labels
from smash_speed.config import load_config
from smash_speed.detection.tracknet import convert_tracknet_csv, tracknet_command
from smash_speed.physics import PhysicsParams
from smash_speed.train.dataset import assign_splits, augment_clip, export_tracknet_dataset
from smash_speed.train.finetune import build_commands, parse_log
from smash_speed.train.lowfps import lowfps_plan, make_lowfps
from smash_speed.train.metrics import detection_metrics
from smash_speed.video_io import probe


@pytest.fixture(scope="module")
def video240(tmp_path_factory):
    """240fps の合成動画と、その正解ラベル（annotate の形式）。"""
    cfg = load_config()
    cfg["synthetic"]["camera"].update(width=640, height=360, f_px=500.0)
    phys = PhysicsParams.from_config(cfg)
    cam = synth.default_camera(cfg)
    shot = synth.make_shot(70.0, cfg, phys)
    d = tmp_path_factory.mktemp("v240")
    truth = synth.render_video(shot, cam, cfg, phys, d / "rally1.mp4", fps=240.0, radius_px=2)
    truth["event"] = ""
    truth.loc[truth.visible.idxmax(), "event"] = "hit"
    truth.to_csv(d / "labels.csv", index=False)
    return cfg, d, truth


def test_annotation_state_and_interpolation(tmp_path):
    st = AnnotationState(np.arange(10, 20), np.arange(10) / 240)
    st.set_point(100.0, 50.0)          # frame 10
    st.go(3)                           # frame 14
    st.set_point(140.0, 90.0)
    assert st.cur == 5
    st.cur = 2
    assert st.suggestion() == pytest.approx((120.0, 70.0))
    assert st.interpolate_gaps() == 3
    st.cur = 6
    st.set_invisible()
    st.cur = 0
    st.toggle_event("hit")
    st.cur = 1
    st.toggle_event("hit")             # 付け直すと移動する
    assert list(st.event).count("hit") == 1 and st.event[1] == "hit"

    meta = type("M", (), {"path": "x.mp4", "capture_fps": 240.0})()
    save_labels(st, tmp_path, "rally", meta, None, None)
    lab = pd.read_csv(tmp_path / "labels.csv")
    tn = pd.read_csv(tmp_path / "rally_ball.csv")
    assert list(tn.columns) == ["Frame", "Visibility", "X", "Y"]
    assert tn.Frame.iloc[0] == 0 and tn.Visibility.sum() == 5
    assert (lab.source == "interp").sum() == 3
    meta_json = json.loads((tmp_path / "annotation_meta.json").read_text(encoding="utf-8"))
    assert "hit" in meta_json["events_s"]


def test_load_initial_from_detections():
    st = AnnotationState(np.arange(5), np.arange(5) / 30)
    det = pd.DataFrame({"frame": [1, 2], "timestamp": [1 / 30, 2 / 30], "x_px": [10.0, np.nan],
                        "y_px": [5.0, np.nan], "visible": [1, 0]})
    assert st.load_initial(det) == 1
    assert st.source[1] == "init"
    tn = pd.DataFrame({"Frame": [0, 1], "Visibility": [0, 1], "X": [0, 12.5], "Y": [0, 3.0]})
    assert normalize_labels(tn).x_px.isna().iloc[0]


def test_lowfps_plan():
    groups, step, n_avg = lowfps_plan(80, 240, 30, blur=True)
    assert step == 8 and n_avg == 8 and len(groups) == 10
    groups, _, n_avg = lowfps_plan(80, 240, 30, n_avg=2, offset=3)
    assert n_avg == 2 and list(groups[0]) == [3, 4]
    with pytest.raises(ValueError):
        lowfps_plan(80, 240, 70)


def test_make_lowfps_labels_and_video(video240, tmp_path):
    cfg, d, truth = video240
    meta = probe(d / "rally1.mp4")
    info = make_lowfps(meta, d / "labels.csv", 30, tmp_path, blur=True)
    low = pd.read_csv(info["labels"])
    m2 = probe(info["video"])
    assert m2.fps_meta == pytest.approx(30, rel=1e-3)
    assert m2.n_frames == len(low)
    assert info["exposure_s"] == pytest.approx(8 / 240)
    # 時刻のずれ：出力フレーム 0 は元の 0〜7 フレームの平均時刻
    assert info["time_offset_s"] == pytest.approx(3.5 / 240, abs=1e-4)
    # 正解位置は平均したフレームの見えている位置の平均
    k = int(low.visible.idxmax()) + 1
    src = truth.iloc[8 * k:8 * k + 8]
    assert low.x_px[k] == pytest.approx(src.x_px[src.visible == 1].mean())
    assert (low.event == "hit").sum() == 1


def test_detection_metrics():
    gt = pd.DataFrame({"frame": range(6), "x_px": [0, 10, 20, np.nan, 40, 50.0],
                       "y_px": [0, 0, 0, np.nan, 0, 0.0], "visible": [1, 1, 1, 0, 1, 1]})
    pr = pd.DataFrame({"frame": range(6), "x_px": [1, 10, 30, 5, np.nan, 50.0],
                       "y_px": [0, 0, 0, 5, np.nan, 0.0], "visible": [1, 1, 1, 1, 0, 1]})
    m = detection_metrics(pr, gt, tolerance_px=4)
    assert (m["tp"], m["fp_far"], m["fp_invisible"], m["fn"], m["tn"]) == (3, 1, 1, 1, 0)
    assert m["precision"] == pytest.approx(3 / 5) and m["recall"] == pytest.approx(3 / 4)


def test_group_split_never_mixes_sessions():
    idx = pd.DataFrame({"video": [f"v{i}.mp4" for i in range(12)],
                        "session": ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E", "F", "F"]})
    sp = assign_splits(idx, seed=1)
    for s, g in idx.assign(split=sp).groupby("session"):
        assert g.split.nunique() == 1
    assert set(sp) == {"train", "val", "test"}


def test_augment_clip_keeps_labels_consistent():
    rng = np.random.default_rng(0)
    f = np.zeros((100, 200, 3), np.uint8)
    f[50, 100] = 255
    frames, xy = augment_clip([f], np.array([[100.0, 50.0]]), rng,
                              {"brightness": 0, "contrast": 0, "noise_sd": 0, "scale": 0.2, "shift_px": 10, "blur_max": 0})
    y, x = np.unravel_index(np.argmax(frames[0].max(axis=2)), frames[0].shape[:2])
    assert np.hypot(x - xy[0, 0], y - xy[0, 1]) < 1.5


def test_export_tracknet_dataset(video240, tmp_path):
    cfg, d, _ = video240
    idx = pd.DataFrame({"video": ["rally1.mp4"] * 3, "labels": ["labels.csv"] * 3, "session": ["a", "b", "c"]})
    idx.to_csv(d / "index.csv", index=False)
    cfg = {**cfg, "train": {"augment_copies": 1, "augment": {}}}
    n = export_tracknet_dataset(d / "index.csv", tmp_path / "ds", cfg, width=320)
    assert n >= 3
    csvs = list((tmp_path / "ds").glob("*/match*/csv/*_ball.csv"))
    assert csvs and list(pd.read_csv(csvs[0]).columns) == ["Frame", "Visibility", "X", "Y"]
    assert list((tmp_path / "ds" / "test").glob("match*/corrected_csv/*_ball.csv"))
    split = json.loads((tmp_path / "ds" / "split.json").read_text(encoding="utf-8"))
    assert {r["split"] for r in split} <= {"train", "test"}


def test_tracknet_wrapper_conversion(tmp_path):
    cfg = load_config()
    with pytest.raises(ValueError):
        tracknet_command(cfg, tmp_path / "v.mp4", tmp_path)
    (tmp_path / "TrackNetV3").mkdir()
    (tmp_path / "TrackNetV3" / "predict.py").write_text("")
    cfg["detection"]["tracknet"].update(repo_dir=str(tmp_path / "TrackNetV3"), tracknet_ckpt="a.pt")
    cmd = tracknet_command(cfg, tmp_path / "v.mp4", tmp_path / "out")
    assert cmd[1] == "predict.py" and "--tracknet_file" in cmd and "--inpaintnet_file" not in cmd
    pd.DataFrame({"Frame": [0, 1, 2], "Visibility": [0, 1, 1], "X": [0, 100, 110], "Y": [0, 50, 52]}
                 ).to_csv(tmp_path / "v_ball.csv", index=False)
    meta = type("M", (), {"timestamps": np.arange(3) / 30})()
    det = convert_tracknet_csv(tmp_path / "v_ball.csv", meta, frame_offset=100, time_offset=5.0)
    assert det.frame.tolist() == [100, 101, 102]
    assert det.timestamp.iloc[1] == pytest.approx(5.0 + 1 / 30)
    assert np.isnan(det.x_px.iloc[0]) and det.x_px.iloc[2] == 110


def test_finetune_helpers():
    cfg = load_config("configs/train.yaml")
    cmds, repo, save = build_commands(cfg)
    assert cmds[-1][1] == "train.py" and "--resume_training" in cmds[-1]
    s = parse_log("Epoch 1 train_loss: 0.12 val_acc=0.81\nEpoch 2 train_loss: 0.08 val_acc=0.85")
    assert s["train_loss"] == [0.12, 0.08] and s["val_acc"] == [0.81, 0.85]
