"""基礎打ちの検出と打球（drill.py。コート合わせなし）のテスト。"""

import json

import numpy as np
import pandas as pd
import pytest

from smash_speed.config import load_config
from smash_speed.drill import (DETECT_INFO_FILE, HITS_FILE, DrillDetectOptions, detect, drill_hits, drill_tracks,
                               hits_table, image_track_table, synthetic_hits_benchmark)
from smash_speed.hits import Hit
from smash_speed.synth_clutter import ClutterOptions
from smash_speed.synth_drill import DrillOptions

CFG = load_config()


def _track(tid: int, t0: float, n: int, x0: float, vx: float, y0: float = 500.0, fps: float = 30.0) -> pd.DataFrame:
    k = np.arange(n)
    return pd.DataFrame({"frame": (t0 * fps + k).astype(int), "timestamp": t0 + k / fps, "x_px": x0 + vx * k / fps,
                         "y_px": y0 + 0.0 * k, "visible": 1, "confidence": 1.0, "track_id": tid})


def test_image_filter_drops_still_and_short_tracks():
    det = pd.concat([_track(0, 0.0, 12, 1000.0, 1200.0),          # 飛んでいる（画像の幅 4000 の 0.3 倍/s）
                     _track(1, 0.0, 12, 2000.0, 0.0, 800.0),      # 止まっている点
                     _track(2, 1.0, 3, 1000.0, 1200.0)])           # 短い
    tab = image_track_table(det, 4000)
    keep = dict(zip(tab.track_id, tab.keep))
    assert keep == {0: True, 1: False, 2: False}


def test_overlapping_tracks_pick_the_longer_one():
    det = pd.concat([_track(0, 0.0, 20, 1000.0, 1200.0), _track(1, 0.2, 8, 3000.0, -1200.0)])
    out, tab = drill_tracks(det, 4000)
    assert tab.selected.tolist() == [True, False]
    assert set(out.track_id[out.visible > 0]) == {0}
    assert len(out) == det.frame.nunique()


def test_hits_table_columns():
    t = hits_table([Hit(1.5, method="audio+track", score=2.0)])
    assert t.t.tolist() == [1.5] and "method" in t and "confidence" in t
    assert hits_table([]).empty


@pytest.fixture(scope="module")
def bench():
    small = dict(n_streaks=2, streak_hits=(4, 8), pause_s=(3.0, 5.0))
    return {name: synthetic_hits_benchmark(("drive", "clear"), n_scenes=1, opts=DrillOptions(**small, clutter=c),
                                           cfg=CFG)
            for name, c in (("clean", ClutterOptions.clean()), ("default", ClutterOptions(neighbor=False)))}


def test_synthetic_drill_hits_are_found(bench):
    # 打球音ありで、ネットの近くの斜めのカメラ（30fps）でも打球を見つける（完了の条件：再現率 0.9・適合率 0.85）
    r = bench["clean"][True]
    assert r["tp"] + r["fn"] >= 10
    assert r["recall"] >= 0.9 and r["precision"] >= 0.85
    assert r["mean_abs_dt_s"] < 0.05


def test_synthetic_drill_hits_with_clutter(bench):
    # 足音・ラケット・頭の検出と、強さの重なる雑音の音を足した条件（1 場面だけなので余裕を持たせる）
    r = bench["default"][True]
    assert r["recall"] >= 0.75 and r["precision"] >= 0.7


def test_synthetic_drill_hits_without_audio_still_works(bench):
    r = bench["clean"][False]
    assert r["recall"] >= 0.35 and r["precision"] >= 0.5


def test_drill_hits_have_no_side_without_calibration():
    det = pd.concat([_track(0, 0.0, 12, 1000.0, 1200.0)])
    assert all(h.side == "" for h in drill_hits(det, 30.0, None, (3840, 2160)))


def test_detect_writes_outputs_from_a_detections_file(tmp_path, monkeypatch):
    import smash_speed.drill as d
    from smash_speed.video_io import VideoMeta

    ts = np.arange(300) / 30.0
    meta = VideoMeta(path=str(tmp_path / "v.mp4"), width=3840, height=2160, fps_meta=30.0, capture_fps=30.0,
                     time_scale=1.0, n_frames=300, duration_s=10.0, rotation_deg=0, is_vfr=False, has_audio=False,
                     timestamps=ts)
    monkeypatch.setattr(d, "probe", lambda *a, **k: meta)
    # 左から右へ飛んで、2 秒で折り返すシャトルの検出
    a, b = _track(0, 0.0, 30, 500.0, 1500.0), _track(0, 1.1, 30, 2000.0, -1500.0)
    p = tmp_path / "det.csv"
    pd.concat([a, b])[["frame", "x_px", "y_px", "visible", "track_id"]].to_csv(p, index=False)
    info = detect(DrillDetectOptions(tmp_path / "v.mp4", tmp_path / "out", use_audio=False, detections=p), CFG,
                  progress=lambda m: None)
    assert (tmp_path / "out" / HITS_FILE).exists()
    saved = json.loads((tmp_path / "out" / DETECT_INFO_FILE).read_text(encoding="utf-8"))
    assert saved["n_hits"] == info["n_hits"] >= 1 and saved["detector"] == "detections"


def test_detect_tells_how_to_export_the_model_when_it_cannot_run(tmp_path, monkeypatch):
    """既定のモデルを動かせない PC で今の検出を使うときは、ONNX に書き出す手順を出して記録する（K1）。"""
    import smash_speed.drill as d
    from smash_speed.video_io import VideoMeta

    ts = np.arange(300) / 30.0
    meta = VideoMeta(path=str(tmp_path / "v.mp4"), width=3840, height=2160, fps_meta=30.0, capture_fps=30.0,
                     time_scale=1.0, n_frames=300, duration_s=10.0, rotation_deg=0, is_vfr=False, has_audio=False,
                     timestamps=ts)
    monkeypatch.setattr(d, "probe", lambda *a, **k: meta)
    det = pd.concat([_track(0, 0.0, 30, 500.0, 1500.0), _track(1, 1.1, 30, 2000.0, -1500.0)])
    monkeypatch.setattr(d, "detect_all", lambda *a, **k: det)
    monkeypatch.setattr(d, "tracknet_hint", lambda cfg: "model_ft1.onnx を作ってください")
    lines = []
    info = detect(DrillDetectOptions(tmp_path / "v.mp4", tmp_path / "out", use_audio=False), CFG, progress=lines.append)
    assert info["detector"] == "baseline" and info["detector_hint"] == "model_ft1.onnx を作ってください"
    assert any("model_ft1.onnx" in m for m in lines)
    monkeypatch.setattr(d, "tracknet_hint", lambda cfg: None)
    assert detect(DrillDetectOptions(tmp_path / "v.mp4", tmp_path / "out2", use_audio=False), CFG,
                  progress=lambda m: None)["detector_hint"] is None
