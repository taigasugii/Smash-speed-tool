"""Phase 4：合成動画で ``analyze`` を通しで動かすテスト。"""

import json

import pandas as pd
import pytest

from smash_speed.config import load_config
from smash_speed.pipeline import AnalyzeOptions, analyze
from smash_speed.validate import make_synthetic_session


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    cfg = load_config()
    cfg["synthetic"]["camera"].update(width=960, height=540, f_px=750.0)
    d = tmp_path_factory.mktemp("session")
    info = make_synthetic_session(cfg, d, speed=75.0, fps=120.0)
    return cfg, d, info


@pytest.mark.parametrize("use_audio", [True, False])
def test_analyze_end_to_end(session, use_audio):
    cfg, d, info = session
    out = d / f"out_{use_audio}"
    o = AnalyzeOptions(video=d / "synthetic.mp4", calib=d / "calibration.json", out_dir=out,
                       audio_path=(d / "synthetic.wav") if use_audio else None,
                       hitter_xy=tuple(info["hitter_xy"]), landing_xy=tuple(info["landing_xy"]),
                       overlay=use_audio)
    row = analyze(o, cfg)
    assert row["speed_initial_mps"] == pytest.approx(75.0, rel=0.05)
    assert row["impact_method"] == ("audio" if use_audio else "hitter_position")
    for f in ("config_used.yaml", "video_meta.json", "detections.csv", "events.json", "plane.json",
              "fit_result.json", "result.csv", "speed_time.png", "trajectory.png", "image_fit.png"):
        assert (out / f).exists(), f
    if use_audio:
        assert (out / "overlay.mp4").stat().st_size > 0
    res = pd.read_csv(out / "result.csv")
    assert res.loc[0, "speed_initial_kmh"] == pytest.approx(row["speed_initial_mps"] * 3.6)
    ev = json.loads((out / "events.json").read_text(encoding="utf-8"))
    assert ev["impact"]["t"] == pytest.approx(info["t_impact_video_s"], abs=0.006)


def test_analyze_reuses_cached_detections(session):
    cfg, d, info = session
    out = d / "out_True"
    before = (out / "detections.csv").stat().st_mtime
    o = AnalyzeOptions(video=d / "synthetic.mp4", calib=d / "calibration.json", out_dir=out,
                       audio_path=d / "synthetic.wav", hitter_xy=tuple(info["hitter_xy"]),
                       landing_xy=tuple(info["landing_xy"]), overlay=False)
    analyze(o, cfg)
    assert (out / "detections.csv").stat().st_mtime == before
