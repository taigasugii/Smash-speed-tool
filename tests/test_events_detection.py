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


def _cands(rows):
    import pandas as pd

    df = pd.DataFrame(rows, columns=["frame", "timestamp", "x", "y", "strength"])
    return df.assign(cid=np.arange(len(df)), area=20.0, length=10.0, angle=0.0, brightness=200.0)


def test_linking_keeps_strongly_decelerating_shuttle_against_clutter():
    """遠ざかるシャトル（画面上で 1 コマごとに約半分に減速）を、まわりの雑音・別の動きに取られずにつなぐ。

    実際の動画（30fps・カメラが打つ人の後ろ）で起きた失敗の再現：
    - 打つ直前の腕の動きの点が、シャトルの最初の点を先に取る
    - 予想位置のすぐ近くの小さな雑音が、少し離れた本物の残像より選ばれる
    - ゆっくり動く遠くの人の軌道に、途中からシャトルがつながる
    """
    from smash_speed.detection.baseline import BaselineParams, _dedupe_overlapping, link_tracks, merge_tracks, \
        split_at_jumps, track_score

    p = BaselineParams()
    fps = 30.0
    shuttle = [(480, 145), (570, 207), (618, 240), (645, 262), (665, 276), (679, 288), (691, 299), (700, 307)]
    rows = []
    for k, (x, y) in enumerate(shuttle):
        f = 10 + k
        rows.append((f, f / fps, x, y, 5000.0 + 300 * (-1) ** k))
    rows.append((9, 9 / fps, 340, 27, 4000.0))                       # 腕の動き（シャトルの直前）
    rows.append((12, 12 / fps, 645, 270, 120.0))                      # 予想位置の近くの小さな雑音
    for k in range(20):                                               # 遠くでゆっくり動く人
        rows.append((4 + k, (4 + k) / fps, 600 + 3 * k, 280 + 0.3 * k, 3000.0))
    for f in range(4, 24):                                            # 動かない雑音
        rows.append((f, f / fps, 100.0, 400.0, 2500.0))
    tracks = split_at_jumps(merge_tracks(link_tracks(_cands(rows).sort_values("frame"), p), p))
    tracks = _dedupe_overlapping([t for t in tracks if len(t.t) >= p.min_track_len])
    # シャトルの軌道が途切れず・余計な点なしで候補に残り、打った前後（9〜12 コマ目）に始まる軌道の中で最もよい
    # （遠くの人のような長い軌道との選び分けは、解析で打球音・打つ人の位置を使って行う）
    near = [t for t in tracks if 9 <= t.frames[0] <= 12]
    best = max(near, key=track_score)
    assert best.frames == list(range(10, 18))
    assert np.allclose(np.array(best.xy), np.array(shuttle, float))


def test_remove_outliers_keeps_early_points_of_decelerating_track():
    from smash_speed.detection.baseline import remove_outliers

    t = np.arange(8) / 30.0
    x = 700 - 220 * 0.55 ** np.arange(8) * 1.0
    steps = np.r_[0, 220 * 0.55 ** np.arange(7)]
    xy = np.column_stack([np.cumsum(steps), 0.4 * np.cumsum(steps)])
    keep = remove_outliers(t, xy, 4.0)
    assert keep[:3].all()
