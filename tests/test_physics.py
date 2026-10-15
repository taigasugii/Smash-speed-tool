"""Phase 1：物理モデルとフィッティングを合成データで検証する。"""

import numpy as np
import pytest

from smash_speed import synth
from smash_speed.config import load_config
from smash_speed.physics import (
    FitOptions, Observations, PhysicsParams, finite_difference_speeds, fit_trajectory,
    simulate, speed_metrics,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def phys(cfg):
    return PhysicsParams.from_config(cfg)


@pytest.fixture(scope="module")
def camera(cfg):
    return synth.default_camera(cfg)


def test_simulate_matches_analytic_without_gravity():
    """g → 0 の水平1次元では v(t) = v0 / (1 + k v0 t) になる。"""
    phys = PhysicsParams(g=1e-12, terminal_velocity=1e-6 * np.sqrt(1 / 0.2185))
    phys.g, k = 9.81, 0.2185
    phys.terminal_velocity = np.sqrt(phys.g / k)
    # 重力の影響を消すため鉛直成分を見ずに、水平方向の速さを解析解と比べる（短時間）
    t = np.linspace(0, 0.02, 5)
    _, vel = simulate(np.zeros(3), np.array([70.0, 0, 0]), 0.0, t, phys)
    v_analytic = 70.0 / (1 + k * 70.0 * t)
    np.testing.assert_allclose(vel[:, 0], v_analytic, rtol=2e-3)


def test_backward_integration_is_consistent(phys):
    p0, v0 = np.array([-4.0, 0.0, 2.5]), np.array([60.0, 5.0, -10.0])
    t = np.array([-0.01, 0.0, 0.05, 0.1])
    pos, vel = simulate(p0, v0, 0.0, t, phys)
    pos2, vel2 = simulate(pos[2], vel[2], 0.05, t, phys)
    np.testing.assert_allclose(pos, pos2, atol=1e-7)
    np.testing.assert_allclose(vel, vel2, atol=1e-6)


@pytest.mark.parametrize("fps", [30, 60, 120, 240])
@pytest.mark.parametrize("speed", [50.0, 75.0, 100.0])
def test_initial_speed_recovered_within_few_percent(cfg, phys, camera, fps, speed):
    """既知の打点時刻・飛行平面のもとで、初速が誤差 3% 以内で復元できる。"""
    rng = np.random.default_rng(int(fps * 1000 + speed))
    shot = synth.make_shot(speed, cfg, phys)
    df = synth.observe(shot, camera, fps, phys, rng, noise_px=1.5, dropout=0.1)
    d = df[df.visible == 1]
    obs = Observations(camera, d.timestamp.values, d[["x_px", "y_px"]].values)
    fit = fit_trajectory(obs, shot.t_impact, "plane", shot.plane, phys)
    assert fit.success
    assert abs(fit.speed0 - speed) / speed < 0.03
    lo, hi = fit.speed0_ci
    assert lo < fit.speed0 < hi


def test_huber_loss_resists_outliers(cfg, phys, camera):
    rng = np.random.default_rng(3)
    shot = synth.make_shot(80.0, cfg, phys)
    df = synth.observe(shot, camera, 60, phys, rng, noise_px=1.0, dropout=0.0)
    uv = df[["x_px", "y_px"]].values.copy()
    uv[[3, 9, 15]] += np.array([60.0, -40.0])  # 誤検出
    obs = Observations(camera, df.timestamp.values, uv)
    fit = fit_trajectory(obs, shot.t_impact, "plane", shot.plane, phys)
    assert abs(fit.speed0 - 80.0) / 80.0 < 0.03


def test_bootstrap_ci_and_timing_uncertainty(cfg, phys, camera):
    rng = np.random.default_rng(4)
    shot = synth.make_shot(70.0, cfg, phys)
    df = synth.observe(shot, camera, 60, phys, rng, noise_px=1.5, dropout=0.1)
    d = df[df.visible == 1]
    obs = Observations(camera, d.timestamp.values, d[["x_px", "y_px"]].values)
    opts = FitOptions(ci_method="bootstrap", bootstrap_n=30, t_ref_sigma=0.002, seed=0)
    fit = fit_trajectory(obs, shot.t_impact, "plane", shot.plane, phys, opts)
    lo, hi = fit.speed0_ci
    assert lo < 70.0 < hi
    # 70 m/s では d|v|/dt ≈ -1000 m/s^2 なので、2 ms の打点誤差で約 2 m/s
    assert 1.5 < fit.speed0_sd_timing < 3.0


def test_terminal_velocity_can_be_estimated(cfg, camera):
    true = PhysicsParams(terminal_velocity=6.9)
    rng = np.random.default_rng(5)
    shot = synth.make_shot(70.0, cfg, true)
    df = synth.observe(shot, camera, 240, true, rng, noise_px=0.5, dropout=0.0)
    obs = Observations(camera, df.timestamp.values, df[["x_px", "y_px"]].values)
    fit = fit_trajectory(obs, shot.t_impact, "plane", shot.plane, PhysicsParams(terminal_velocity=6.5),
                         opts=FitOptions(estimate_terminal_velocity=True))
    assert abs(fit.terminal_velocity - 6.9) < 0.15
    assert abs(fit.speed0 - 70.0) < 2.0


def test_speed_metrics_and_naive_speed_underestimate(cfg, phys, camera):
    rng = np.random.default_rng(6)
    shot = synth.make_shot(80.0, cfg, phys)
    df = synth.observe(shot, camera, 30, phys, rng, noise_px=0.0, dropout=0.0)
    obs = Observations(camera, df.timestamp.values, df[["x_px", "y_px"]].values)
    fit = fit_trajectory(obs, shot.t_impact, "plane", shot.plane, phys)
    m = speed_metrics(fit, phys)
    assert m["speed_initial_kmh"] == pytest.approx(m["speed_initial_mps"] * 3.6)
    assert m["flight_time_s"] == pytest.approx(shot.t_land - shot.t_impact, abs=2e-3)
    assert m["speed_mean_path_mps"] >= m["speed_mean_straight_mps"]
    assert 0 < m["deceleration_ratio"] < 1
    # 強い減速のため、30fps のフレーム間差分は初速を大きく過小評価する
    fd = finite_difference_speeds(df.timestamp.values, df[["x_m", "y_m", "z_m"]].values)
    assert fd["first_mps"] < 0.8 * 80.0
