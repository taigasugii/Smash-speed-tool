"""フォームの特徴量（pose_features.py、E2・E3）のテスト。合成の骨格の正解（決めた角度・タイミング）と比べる。"""

import numpy as np
import pytest

from smash_speed.pose import KP
from smash_speed.pose_features import FEATURE_INFO, FEATURE_NAMES, FormFeatures, angle_deg, body_height, form_features
from smash_speed.pose_prep import mirror, normalize, prepare_shot
from smash_speed.synth_pose import SwingParams, angles, joints_3d, project_swing, side_camera, swing_times, vary

P = SwingParams()


def _features(p=P, fps=60.0, noise_px=0.0, seed=0, **cam):
    seq = project_swing(p, side_camera(**cam), swing_times(fps, 1.5, 0.8), t_offset=10.0, noise_px=noise_px,
                        rng=np.random.default_rng(seed))
    return form_features(normalize(seq.slice(9.0, 10.5), 10.0))


def _truth_at_impact(p=P):
    X = joints_3d(p, np.array([0.0]))[0]
    hip = (X[KP["left_hip"]] + X[KP["right_hip"]]) / 2
    knee = {s: float(angle_deg(X[KP[f"{s}_hip"]], X[KP[f"{s}_knee"]], X[KP[f"{s}_ankle"]])) for s in ("left", "right")}
    return {"wrist_height_ratio": X[KP["right_wrist"], 2] / p.height_m,
            "wrist_forward_ratio": (X[KP["right_wrist"], 0] - hip[0]) / p.height_m,
            "front_knee": knee["left"], "back_knee": knee["right"]}


@pytest.mark.parametrize("fps", [60.0, 120.0])
def test_impact_features_match_truth(fps):
    f = _features(fps=fps)
    truth = _truth_at_impact()
    assert f.body_height_torso * 0.288 == pytest.approx(1.0, abs=0.02)   # 身長の見積もり
    assert f.wrist_height_ratio == pytest.approx(truth["wrist_height_ratio"], abs=0.02)
    assert f.wrist_forward_ratio == pytest.approx(truth["wrist_forward_ratio"], abs=0.02)
    assert f.elbow_angle_deg == pytest.approx(P.elbow_impact_deg, abs=2.0)
    assert f.shoulder_angle_deg == pytest.approx(P.arm_impact_deg, abs=3.0)
    assert f.front_knee_angle_deg == pytest.approx(truth["front_knee"], abs=2.0)
    assert f.back_knee_angle_deg == pytest.approx(truth["back_knee"], abs=2.0)
    # 打点では胸が正面近くまで回っている（肩幅がほぼ 0）
    a = angles(P, np.array([0.0]))
    assert f.trunk_turn_impact_deg == pytest.approx(90 + a["trunk"][0], abs=5.0)
    assert f.shoulder_width_impact < 0.15


def test_speeds_and_times_match_truth():
    f = _features(fps=120.0)
    t = np.arange(-0.6, 0.6, 0.0005)
    X = joints_3d(P, t)
    w = np.linalg.norm(np.gradient(X[:, KP["right_wrist"]][:, [0, 2]], t, axis=0), axis=1)
    # 手首の速さ（身長/秒）：コマの間を飛ばす分だけ少なめに出る（120fps で 1 割ほど）
    assert f.wrist_speed_max == pytest.approx(w.max() / P.height_m, rel=0.15)
    # 捻り（腰の回旋 − 胸の回旋）の最大とその時刻
    a = angles(P, t)
    twist = a["pelvis"] - a["trunk"]
    assert f.twist_max_deg == pytest.approx(twist.max(), abs=3.0)
    assert f.t_twist_max == pytest.approx(t[np.argmax(twist)], abs=0.01)
    assert f.twist_release_dps > 100
    # 反対の腕の手首を下へ引く速さの最大
    vz = -np.gradient(X[:, KP["left_wrist"], 2], t)
    assert f.t_free_arm_pull == pytest.approx(t[np.argmax(vz)], abs=0.01)
    # 打った後に戻る速さ（ロジスティック関数の傾きの最大 = 距離 ÷ 4τ）
    assert f.recovery_speed == pytest.approx(P.recover_m / (4 * P.tau_step) / P.height_m, rel=0.05)


@pytest.mark.parametrize("fps", [60.0, 120.0])
def test_kinetic_chain_times(fps):
    f = _features(fps=fps)
    assert f.t_pelvis_peak == pytest.approx(P.t_pelvis, abs=0.01)
    assert f.t_trunk_peak == pytest.approx(P.t_trunk, abs=0.01)
    assert f.t_elbow_peak == pytest.approx(P.t_elbow, abs=0.01)
    from smash_speed.synth_pose import wrist_speed_peak_time
    assert f.t_wrist_peak == pytest.approx(wrist_speed_peak_time(P), abs=0.01)
    assert f.chain_order == "pelvis>trunk>elbow>wrist" and f.chain_in_order == 1.0
    assert f.lag_pelvis_trunk == pytest.approx(P.t_trunk - P.t_pelvis, abs=0.015)
    # 体重移動：踏み込みの量・速さの最大の時刻・沈み込み
    t = np.arange(-1.0, 0.0001, 0.001)
    hx = angles(P, t)["hip_x"]
    assert f.step_forward_ratio == pytest.approx((hx[-1] - hx.min()) / P.height_m, abs=0.01)
    assert f.t_step_peak == pytest.approx(P.t_step, abs=0.01)
    assert f.hip_drop_ratio == pytest.approx(P.dip_m / P.height_m, abs=0.01)


def test_changes_in_the_swing_show_up():
    """角度・タイミングを変えたスイングでは、その特徴量だけが変わる。"""
    base = _features()
    bent = _features(vary(P, elbow_impact_deg=135.0))
    assert bent.elbow_angle_deg == pytest.approx(135.0, abs=2.5)
    assert bent.shoulder_angle_deg == pytest.approx(base.shoulder_angle_deg, abs=1.0)
    # 腰が肩より遅れて回ると、順番が崩れる
    late = _features(vary(P, t_pelvis=-0.06))
    assert late.chain_in_order == 0.0 and late.chain_order.startswith("trunk")
    assert late.twist_max_deg < base.twist_max_deg
    # 踏み込みが大きいと、移動量が大きい
    big = _features(vary(P, step_m=0.7))
    assert big.step_forward_ratio > base.step_forward_ratio * 1.5
    # 背の高さが違っても、身長比の値はほとんど同じ
    tall = _features(vary(P, height_m=1.85))
    for k in ("wrist_height_ratio", "wrist_forward_ratio", "step_forward_ratio"):
        assert getattr(tall, k) == pytest.approx(getattr(base, k), abs=0.02)


def test_robust_to_small_noise_and_camera_distance():
    """関節に 1 px の揺れがあっても・カメラが 6〜8 m のどこでも、角度と順番はほぼ同じ。"""
    base = _features()
    for kw in ({"noise_px": 1.0, "seed": 3}, {"distance_m": 6.0}, {"distance_m": 8.0, "height_m": 1.5}):
        f = _features(**kw)
        assert f.elbow_angle_deg == pytest.approx(base.elbow_angle_deg, abs=4.0)
        assert f.wrist_height_ratio == pytest.approx(base.wrist_height_ratio, abs=0.03)
        assert f.chain_in_order == 1.0


def test_left_handed_gives_the_same_features():
    seq = project_swing(P, side_camera(), swing_times(60.0, 1.5, 0.8), t_offset=10.0)
    a, _ = prepare_shot(seq, 10.0)
    b, _ = prepare_shot(mirror(seq), 10.0, left_handed=True)
    fa, fb = form_features(a).to_dict(), form_features(b).to_dict()
    for k, v in fa.items():
        if isinstance(v, float):
            assert fb[k] == pytest.approx(v, abs=1e-6), k


def test_missing_joints_give_nan_not_errors():
    seq = project_swing(P, side_camera(), swing_times(60.0, 1.0, 0.5))
    seq.xy[:, KP["left_hip"]] = np.nan
    seq.xy[:, KP["left_knee"]] = np.nan
    seq.xy[:, KP["left_shoulder"], :] = seq.xy[:, KP["right_shoulder"], :]  # 正規化はできる
    n = normalize(seq.copy(xy=np.where(np.isnan(seq.xy), seq.xy[:, [KP["right_hip"]]], seq.xy)))
    n.xy[:, KP["left_knee"]] = np.nan
    n.xy[:, KP["left_hip"]] = np.nan
    f = form_features(n)
    assert np.isnan(f.front_knee_angle_deg) and np.isnan(f.hip_width_impact)
    assert np.isfinite(f.elbow_angle_deg)
    d = f.to_dict()
    assert d["front_knee_angle_deg"] is None and set(d) == set(FEATURE_NAMES)
    assert set(FEATURE_INFO) <= set(FEATURE_NAMES)
    assert isinstance(FormFeatures().to_dict()["chain_order"], str)
    assert body_height(n) > 0
