"""Phase 4：イベント検出とベースライン検出のテスト。"""

import shutil

import numpy as np
import pytest

from smash_speed import events, synth
from smash_speed.config import load_config
from smash_speed.detection.baseline import BaselineParams, detect
from smash_speed.geometry import FlightPlane
from smash_speed.physics import PhysicsParams
from smash_speed.video_io import load_wav, probe


def test_audio_onset_subframe_accuracy():
    rate = 48000
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.002, rate)
    t_hit = 0.53217
    i = int(round(t_hit * rate))
    click = np.exp(-np.arange(200) / 30.0) * rng.normal(0, 1, 200)
    x[i:i + 200] += 0.5 * click
    x[int(0.2 * rate):int(0.25 * rate)] += 0.01 * np.sin(np.arange(int(0.05 * rate)) * 2 * np.pi * 200 / rate)  # 低い音
    ons = events.audio_onsets(x, rate)
    assert len(ons) == 1
    # 零位相フィルタと包絡線の平滑化で 0.5 ms 程度早めに出る（既定の不確かさ 2 ms に含まれる）
    assert ons[0][0] == pytest.approx(t_hit, abs=1e-3)
    imp = events.impact_from_audio(ons, t_first_detection=t_hit + 0.01)
    assert imp is not None and imp.method == "audio"
    assert events.impact_from_audio(ons, t_first_detection=t_hit + 0.5) is None


def test_first_frame_uniform_prior():
    imp = events.impact_from_first_frame(1.0, 1 / 30)
    assert imp.t == pytest.approx(1.0 - 1 / 60)
    assert imp.sigma == pytest.approx((1 / 30) / np.sqrt(12))
    narrowed = events.impact_from_first_frame(1.0, 1 / 30, t_last_before=0.99)
    assert narrowed.t == pytest.approx(0.995)


def test_crossing_of_incoming_and_outgoing():
    t_imp = 0.4123
    t_in = np.arange(0.2, 0.40, 1 / 60)
    uv_in = np.column_stack([500 + 300 * (t_in - t_imp), 300 - 200 * (t_in - t_imp)])
    out = lambda t: np.array([500 + 3000 * (t - t_imp), 300 + 800 * (t - t_imp)])  # noqa: E731
    imp = events.impact_from_crossing(t_in, uv_in, out, 0.39, 0.43)
    assert imp.t == pytest.approx(t_imp, abs=2e-4)


def test_hitter_position_method():
    cfg = load_config()
    phys = PhysicsParams.from_config(cfg)
    shot = synth.make_shot(80.0, cfg, phys)
    from smash_speed.physics import simulate
    pos = lambda t: simulate(shot.p0, shot.v0, 0.0, np.array([t]), phys)[0][0]  # noqa: E731
    hitter = FlightPlane(shot.plane.origin - 0.3 * shot.plane.direction, shot.plane.direction)
    imp = events.impact_from_hitter_position(pos, hitter, -0.05, 0.05, contact_offset_m=0.3)
    assert imp.t == pytest.approx(0.0, abs=1e-5)
    assert 0.003 < imp.sigma < 0.008  # 0.4 m / 約 80 m/s


def test_landing_from_detections():
    t = np.arange(10) / 30
    uv = np.column_stack([np.r_[np.arange(6) * 50.0, [250.0] * 4], np.zeros(10)])
    assert events.landing_from_detections(t, uv) == pytest.approx(t[5])


@pytest.mark.skipif(shutil.which("ffprobe") is None and not __import__("os").environ.get("LOCALAPPDATA"),
                    reason="ffmpeg がありません")
def test_baseline_detection_on_synthetic_video(tmp_path):
    cfg = load_config()
    cfg["synthetic"]["camera"].update(width=960, height=540, f_px=750.0)
    phys = PhysicsParams.from_config(cfg)
    cam = synth.default_camera(cfg)
    shot = synth.make_shot(70.0, cfg, phys)
    truth = synth.render_video(shot, cam, cfg, phys, tmp_path / "v.mp4", fps=120.0, radius_px=3,
                               audio_path=tmp_path / "a.wav")
    meta = probe(tmp_path / "v.mp4")
    det = detect(meta, params=BaselineParams.from_config(cfg))
    both = (det.visible == 1) & (truth.visible == 1)
    err = np.hypot(det.x_px - truth.x_px, det.y_px - truth.y_px)[both]
    recall = both.sum() / (truth.visible == 1).sum()
    assert recall > 0.8
    assert np.median(err) < 1.5
    # 音声から打点時刻（動画の 0.2 s）
    x, rate = load_wav(tmp_path / "a.wav")
    ons = events.audio_onsets(x, rate)
    t_first = det[det.visible == 1].timestamp.min()
    delay = float(np.linalg.norm(cam.center - shot.p0)) / 343.0  # 打球音はカメラまでの距離 / 音速だけ遅れる
    assert events.impact_from_audio(ons, t_first) is None  # 補正しないと最初の検出より後になる
    imp = events.impact_from_audio(ons, t_first, av_offset_s=-delay)
    assert imp.t == pytest.approx(0.2, abs=1e-3)
