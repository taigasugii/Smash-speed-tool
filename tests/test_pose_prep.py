"""姿勢の前処理（pose_prep.py）のテスト。合成の骨格（synth_pose）で正解と比べる。"""

import numpy as np
import pytest

from smash_speed.pose import KP
from smash_speed.pose_prep import (
    SPEED_OF_SOUND_MPS, blur_faces, calibrate_wrist_offset, facing_sign, find_impact, impact_errors, mirror,
    normalize, prepare_shot, shot_window,
)
from smash_speed.synth_pose import SwingParams, project_swing, side_camera, swing_times, wrist_speed_peak_time

P = SwingParams()


def _seq(fps=60.0, t_offset=10.0, **kw):
    """インパクトが動画の 10 秒にあるスイング（前 1.5 秒・後 0.8 秒）。"""
    return project_swing(P, side_camera(**kw), swing_times(fps, 1.5, 0.8), t_offset=t_offset)


def test_shot_window():
    assert shot_window(10.0) == (9.0, 10.5)


def test_impact_from_wrist_and_audio():
    seq = _seq(fps=60.0)
    imp = find_impact(seq)
    # 手首の速さの最大（正解は 3 次元で −0.0325 秒。画像の上でもほぼ同じ時刻）
    assert imp.method == "wrist"
    assert imp.t_wrist - 10.0 == pytest.approx(wrist_speed_peak_time(P), abs=1 / 60)
    # 音があれば、手首の最大の近くの音から、届くまでの遅れを引いた時刻
    delay = 7.0 / SPEED_OF_SOUND_MPS
    imp = find_impact(seq, audio_times=[8.0, 10.0 + delay, 12.0])
    assert imp.method == "audio" and imp.t == pytest.approx(10.0, abs=1e-9)
    # 遠い音（相手の打球）は使わない
    assert find_impact(seq, audio_times=[9.5, 10.6]).method == "wrist"
    # 手で付けた打点から、手首の最大との差を決めて足す
    off = calibrate_wrist_offset([imp.t_wrist], [10.0])
    assert find_impact(seq, wrist_offset_s=off).t == pytest.approx(10.0, abs=1e-9)


def test_impact_errors_in_frames():
    r = impact_errors([10.01, None, 20.0], [10.0, 15.0, 20.02], fps=100)
    assert r["n"] == 2 and r["errors_s"][1] is None
    assert r["mean_abs_s"] == pytest.approx(0.015)
    assert r["mean_abs_frames"] == pytest.approx(1.5)
    assert impact_errors([None], [1.0]) == {"errors_s": [None], "n": 0}


def test_mirror_is_its_own_inverse_and_swaps_sides():
    seq = _seq()
    m = mirror(seq)
    assert m.meta["mirrored"] is True
    assert np.allclose(m.joint("left_wrist")[:, 0], seq.width - seq.joint("right_wrist")[:, 0])
    assert facing_sign(seq) == 1.0 and facing_sign(m) == -1.0
    back = mirror(m)
    assert np.allclose(back.xy, seq.xy) and back.meta["mirrored"] is False


def test_left_handed_is_the_same_as_right_handed():
    """左利き（ラケットの腕の側 = 左から撮る）の画像は、右利きの画像の左右の反転。反転すると同じ値になる。"""
    seq = _seq()
    lefty = mirror(seq)
    a, ia = prepare_shot(seq, 10.0)
    b, ib = prepare_shot(lefty, 10.0, left_handed=True)
    assert ia.t == pytest.approx(ib.t)
    assert np.allclose(a.xy, b.xy, equal_nan=True)
    assert b.meta["left_handed"] is True


def test_normalize_origin_scale_and_direction():
    seq = _seq()
    n = normalize(seq, t_impact=10.0)
    hip = (n.joint("left_hip") + n.joint("right_hip")) / 2
    assert np.allclose(hip, 0.0, atol=1e-9)
    sh = (n.joint("left_shoulder") + n.joint("right_shoulder")) / 2
    assert np.median(np.linalg.norm(sh, axis=1)) == pytest.approx(1.0)
    assert (sh[:, 1] > 0.9).all()                       # 上が +y
    assert n.hip[-1, 0] > 0.5                           # 前（+x）へ踏み込んで戻る
    assert n.t[0] == pytest.approx(-1.5) and n.facing == 1.0
    # 距離が変わっても（写る大きさが変わっても）ほとんど同じ
    far = normalize(_seq(distance_m=8.0), t_impact=10.0)
    assert np.nanmax(np.abs(far.xy - n.xy)) < 0.05
    # 画像の左へ打つ向き（反対側から撮った）でも、+x が打つ向き
    flipped = seq.copy(xy=seq.xy * [-1, 1] + [seq.width, 0])
    nf = normalize(flipped, t_impact=10.0)
    assert nf.facing == -1.0 and np.allclose(nf.hip, n.hip)


def test_prepare_shot_window_and_save(tmp_path):
    n, imp = prepare_shot(_seq(), 10.05)
    assert imp.method == "wrist" and abs(imp.t - 10.0) < 0.05
    assert n.t[0] >= -1.0 - 1e-9 and n.t[-1] <= 0.5 + 1e-9 and len(n.t) >= 85
    n.save(tmp_path / "n.npz")
    from smash_speed.pose_prep import NormalizedPose
    back = NormalizedPose.load(tmp_path / "n.npz")
    assert np.allclose(back.xy, n.xy) and back.torso_px == n.torso_px


def test_normalize_without_body_fails():
    seq = _seq()
    seq.xy[:, KP["left_hip"]] = np.nan
    with pytest.raises(ValueError, match="肩と腰"):
        normalize(seq)


def test_blur_faces_only_blurs_the_face():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    seq = _seq()
    k = 90
    out = blur_faces(img, seq.xy[k], seq.score[k])
    nose = seq.xy[k, KP["nose"]].astype(int)
    face = (slice(nose[1] - 5, nose[1] + 5), slice(nose[0] - 5, nose[0] + 5))
    assert out[face].std() < img[face].std() * 0.5
    assert np.array_equal(out[:100, :100], img[:100, :100])
    assert np.array_equal(blur_faces(img, seq.xy[k], np.zeros(17)), img)   # 顔が見つからなければそのまま
