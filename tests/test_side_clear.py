"""側方から撮ったクリアの打球の結果（side_clear.py、E6）のテスト。"""

import numpy as np
import pytest

from smash_speed.physics import PhysicsParams, simulate, simulate_until_ground
from smash_speed.pose_prep import mirror, normalize
from smash_speed.side_clear import SideCase, SideSetup, fit_clear, setup_from_pose, side_benchmark
from smash_speed.synth_pose import SwingParams, joints_3d, project_swing, side_camera, swing_times

PHYS = PhysicsParams()


def test_setup_camera_matches_the_true_camera_on_the_plane():
    """体から決めた縮尺・床・打つ人の位置だけで、打つ人の面の点が本当のカメラと同じ所に写る。"""
    cam = side_camera(distance_m=7.0, height_m=1.3, target_x_m=2.0)
    ground = cam.project(np.array([0.0, 0.0, 0.0]))[0]
    st = SideSetup(px_per_m=cam.f / 7.0, ground_v_px=ground[1], hitter_u_px=ground[0], f_px=cam.f)
    pts = np.array([[0.0, 0.0, 2.5], [3.0, 0.0, 4.0], [-1.0, 0.0, 0.5]])
    assert np.allclose(st.camera().project(pts), cam.project(pts), atol=1e-6)
    # 打つ向きが画像の左なら、画像を反転してから使う
    left = SideSetup(px_per_m=st.px_per_m, ground_v_px=st.ground_v_px, hitter_u_px=1920 - st.hitter_u_px, f_px=cam.f,
                     facing=-1.0)
    uv = cam.project(pts) * [-1, 1] + [1920, 0]
    assert np.allclose(left.camera().project(pts), left.to_right(uv), atol=1e-6)


def test_fit_clear_recovers_a_clear_seen_long_enough():
    """カメラの画角が広く 120fps で長く見えていれば、初速・打ち出し角・最高点・落ちた所はほぼ正しい。"""
    p0, v0 = np.array([0.0, 0.0, 2.6]), 30.0 * np.array([np.cos(np.radians(40)), 0.0, np.sin(np.radians(40))])
    cam = side_camera(distance_m=8.0, height_m=1.5, target_x_m=4.0, hfov_deg=100.0)
    t = np.arange(0.0, 0.8, 1 / 120)
    pos, _ = simulate(p0, v0, 0.0, t, PHYS)
    uv = cam.project(pos) + np.random.default_rng(0).normal(0, 0.5, (len(t), 2))
    g = cam.project(np.zeros(3))[0]
    st = SideSetup(px_per_m=cam.f / 8.0, ground_v_px=g[1], hitter_u_px=g[0], f_px=cam.f)
    r = fit_clear(t, uv, 0.0, st.camera(), PHYS, hitter_from_net_m=5.5)
    tr = simulate_until_ground(p0, v0, 0.0, PHYS)
    assert r.speed_kmh == pytest.approx(108.0, rel=0.02)
    assert r.elevation_deg == pytest.approx(40.0, abs=1.0)
    assert r.apex_height_m == pytest.approx(tr["pos"][:, 2].max(), abs=0.1)
    assert r.land_from_hitter_m == pytest.approx(tr["pos"][-1, 0], abs=0.3)
    assert r.land_depth_m == pytest.approx(r.land_from_hitter_m - 5.5)
    assert r.n_obs == len(t) and r.rms_px < 1.5
    assert fit_clear(t[:3], uv[:3], 0.0, st.camera(), PHYS) is None


def test_setup_from_pose_gives_the_scale_and_the_floor():
    p = SwingParams()
    cam = side_camera()
    seq = project_swing(p, cam, swing_times(60.0, 1.5, 0.8))
    for s, facing in ((seq, 1.0), (mirror(seq), -1.0)):
        n = normalize(s.slice(-1.0, 0.5), 0.0)
        st = setup_from_pose(n, p.height_m, 1920, 1080)
        assert st.px_per_m == pytest.approx(cam.f / 7.0, rel=0.02) and st.facing == facing
        X = joints_3d(p, np.array([0.0]))[0]
        foot = cam.project(np.array([(X[11, 0] + X[12, 0]) / 2, 0.0, 0.0]))[0]
        assert st.ground_v_px == pytest.approx(foot[1], abs=3.0)
        u = foot[0] if facing > 0 else 1920 - foot[0]
        assert st.hitter_u_px == pytest.approx(u, abs=1.0)
    n.meta.pop("raw_hip_px")
    with pytest.raises(ValueError):
        setup_from_pose(n, 1.7, 1920, 1080)


def test_side_benchmark_shows_the_effect_of_the_setup():
    """合成のクリア：基準の置き方で動き、打点が体よりカメラ側にあると初速が多めに出る。"""
    base = side_benchmark(8, SideCase(fps=120.0), seed=1)
    assert base["n"] >= 6 and base["speed_mape"] < 15 and base["in_frame_s"] < 0.3
    near = side_benchmark(8, SideCase(fps=120.0, contact_offset_m=0.5), seed=1)
    assert near["speed_kmh"]["bias"] > base["speed_kmh"]["bias"]
    tall = side_benchmark(8, SideCase(fps=120.0, width=1080, height=1920, hfov_deg=42.6, lead_m=1.0), seed=1)
    assert tall["in_frame_s"] > base["in_frame_s"]
