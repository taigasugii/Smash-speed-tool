"""Phase 7 / 8：精度検証と2台カメラモード。"""

import numpy as np
import pandas as pd
import pytest

from smash_speed import synth
from smash_speed.calibration import calibrate, landmark_table, save_calibration
from smash_speed.camera import look_at
from smash_speed.config import load_config
from smash_speed.physics import PhysicsParams
from smash_speed.stereo import StereoOptions, analyze_stereo, audio_offset
from smash_speed.validate import apply_correction, error_stats, fit_linear_correction, synthetic_study


def test_error_stats_and_correction():
    ref = np.array([200.0, 250.0, 300.0, 350.0])
    est = 0.8 * ref + 10
    s = error_stats(ref, est)
    assert s["bias"] == pytest.approx(np.mean(est - ref))
    assert s["mape"] > 0
    c = fit_linear_correction(est, ref)
    np.testing.assert_allclose(apply_correction(est, c), ref, atol=1e-9)


def test_synthetic_study_outputs(tmp_path):
    cfg = load_config()
    df = synthetic_study(cfg, tmp_path, n_shots=6, fps_list=[30, 240], seed=0)
    for f in ("results.csv", "summary.csv", "correction.csv", "mape_vs_fps.png", "bland_altman.png", "study.json"):
        assert (tmp_path / f).exists(), f
    summ = pd.read_csv(tmp_path / "summary.csv")
    phys240 = summ[(summ.method == "physics") & (summ.impact == "audio") & (summ.fps == 240)].iloc[0]
    naive30 = summ[(summ.method == "naive") & (summ.fps == 30)].iloc[0]
    assert phys240["mape"] < 5
    assert naive30["mape"] > phys240["mape"]  # 低 fps の差分は物理モデルより悪い
    assert set(df.method) == {"physics", "naive"}


def test_audio_offset_recovers_shift():
    rate = 48000
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.002, 3 * rate)
    for t in (0.5, 1.3, 2.1):
        i = int(t * rate)
        base[i:i + 150] += 0.5 * np.exp(-np.arange(150) / 25) * rng.normal(0, 1, 150)
    shift = 0.23741
    k = int(round(shift * rate))
    x2 = np.r_[rng.normal(0, 0.002, k), base][: len(base)]
    off, sharp = audio_offset(base, x2, rate)
    assert off == pytest.approx(shift, abs=1 / rate)
    assert sharp > 1.0


def test_stereo_end_to_end(tmp_path):
    cfg = load_config()
    cfg["synthetic"]["camera"].update(width=960, height=540, f_px=750.0)
    phys = PhysicsParams.from_config(cfg)
    cam1 = synth.default_camera(cfg)
    cam2 = look_at(np.array([-9.5, 5.0, 3.5]), np.array([0.0, 0.0, 1.0]), 700.0, 960, 540)
    shot = synth.make_shot(78.0, cfg, phys)
    table = landmark_table(cfg)
    rng = np.random.default_rng(0)
    for k, cam in enumerate((cam1, cam2), start=1):
        synth.render_video(shot, cam, cfg, phys, tmp_path / f"v{k}.mp4", fps=120.0, radius_px=3,
                           pre_s=0.2 + 0.05 * (k - 1), audio_path=tmp_path / f"a{k}.wav", seed=k)
        names = [n for n, X in table.items() if np.isfinite(cam.project(X[None])).all()
                 and 0 <= cam.project(X[None])[0, 0] < 960 and 0 <= cam.project(X[None])[0, 1] < 540]
        W = np.array([table[n] for n in names])
        res = calibrate(names, W, cam.project(W) + rng.normal(0, 0.5, (len(W), 2)), 960, 540)
        save_calibration(res, tmp_path / f"c{k}.json")
    o = StereoOptions(videos=[tmp_path / "v1.mp4", tmp_path / "v2.mp4"],
                      calibs=[tmp_path / "c1.json", tmp_path / "c2.json"], out_dir=tmp_path / "out",
                      audio_paths=[tmp_path / "a1.wav", tmp_path / "a2.wav"])
    row = analyze_stereo(o, cfg)
    assert row["sync_offset_s"] == pytest.approx(0.05, abs=1e-3)  # 2台目は 50 ms 遅れて録画開始
    assert row["speed_initial_mps"] == pytest.approx(78.0, rel=0.05)
