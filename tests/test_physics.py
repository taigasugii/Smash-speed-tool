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


def test_rhs_matches_the_numpy_formula_bit_for_bit():
    """Python の数で計算する運動方程式（F3b で速くした）は、numpy の式と 1 ビットも違わない。"""
    from smash_speed.physics import _rhs

    rng = np.random.default_rng(0)
    g, k = 9.80665, 9.80665 / 6.8 ** 2
    for _ in range(3000):
        st = rng.normal(0.0, 1.0, 6) * 10 ** rng.uniform(-3, 3, 6)
        v = st[3:]
        a = -k * np.sqrt(v @ v) * v
        a[2] -= g
        assert np.array_equal(_rhs(0.0, st, g, k), np.concatenate([v, a]))


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


def test_too_early_hit_time_stops_at_the_speed_ceiling(cfg, phys, camera):
    """打った時刻が早すぎると合う初速がなくなる（前へ戻すと速さが無限大になる）。上限の目安の所で止まり、迷わない。"""
    import smash_speed.physics as physics

    rng = np.random.default_rng(8)
    shot = synth.make_shot(90.0, cfg, phys)
    df = synth.observe(shot, camera, 30, phys, rng, noise_px=1.0, dropout=0.0)
    d = df[df.timestamp > shot.t_impact + 0.01]
    obs = Observations(camera, d.timestamp.values, d[["x_px", "y_px"]].values)
    calls = []
    sim = physics.simulate
    try:
        physics.simulate = lambda p0, v0, *a: (calls.append(float(np.linalg.norm(v0))), sim(p0, v0, *a))[1]
        fit = fit_trajectory(obs, shot.t_impact - 0.08, "plane", shot.plane, phys, FitOptions(ci_method="none"))
        n_ceiling = len(calls)
        calls.clear()
        free = fit_trajectory(obs, shot.t_impact - 0.08, "plane", shot.plane, phys,
                              FitOptions(ci_method="none", max_speed_mps=None, ftol=1e-8, xtol=1e-8,
                                         max_nfev_per_param=200))
    finally:
        physics.simulate = sim
    assert physics.MAX_FIT_SPEED_MPS < fit.speed0 < physics.MAX_FIT_SPEED_MPS + 15.0
    assert free.speed0 > 2 * physics.MAX_FIT_SPEED_MPS       # 上限がないと、積分できる限界の近くまで上がる
    assert n_ceiling < len(calls)
    # 打った時刻が合っていれば、上限は結果を変えない
    a = fit_trajectory(obs, shot.t_impact, "plane", shot.plane, phys, FitOptions(ci_method="none"))
    b = fit_trajectory(obs, shot.t_impact, "plane", shot.plane, phys, FitOptions(ci_method="none", max_speed_mps=None))
    assert abs(a.speed0 - b.speed0) < 1e-3 * b.speed0


def test_start_faster_than_the_ceiling_is_slowed_down(phys):
    from smash_speed.geometry import FlightPlane
    from smash_speed.physics import MAX_FIT_SPEED_MPS, _Model

    m = _Model("plane", FlightPlane(np.zeros(3), np.array([1.0, 0.0, 0.0])), False, phys.terminal_velocity,
               max_speed=MAX_FIT_SPEED_MPS)
    th = m.clip_speed(np.array([-5.0, 2.5, 900.0, -300.0]))
    assert np.hypot(th[2], th[3]) == pytest.approx(MAX_FIT_SPEED_MPS)
    assert th[3] / th[2] == pytest.approx(-300.0 / 900.0)
    np.testing.assert_array_equal(m.clip_speed(np.array([-5.0, 2.5, 60.0, -10.0])), [-5.0, 2.5, 60.0, -10.0])
    assert m.speed_residual(np.array([-5.0, 2.5, 60.0, -10.0]))[0] == 0.0
    assert m.speed_residual(np.array([-5.0, 2.5, 200.0, 0.0]))[0] > 0.0


def test_quadratic_priors_are_not_weakened_by_huber_loss(cfg, phys, camera):
    """打点の位置の目安は、Huber 損失をかけても 2 乗の罰のまま効く（目安から離れた打点を引き戻す）。"""
    from smash_speed.geometry import FlightPlane
    from smash_speed.physics import _Model

    m = _Model("plane", FlightPlane(np.zeros(3), np.array([1.0, 0.0, 0.0])), False, phys.terminal_velocity)
    m.huber_px = 2.0
    r = np.array([-9.0, -2.0, 0.5, 2.0, 7.0])
    stretched = np.abs(m._quadratic(r))
    huber = np.where(stretched <= 2.0, 0.5 * stretched**2, 2.0 * (stretched - 1.0))
    np.testing.assert_allclose(huber, 0.5 * r**2)

    rng = np.random.default_rng(5)
    shot = synth.make_shot(70.0, cfg, phys)
    df = synth.observe(shot, camera, 30, phys, rng, noise_px=1.0, dropout=0.0)
    d = df[df.visible == 1].iloc[1:7]            # 打った直後を除いた 6 コマ：奥行きが決まりにくい
    obs = Observations(camera, d.timestamp.values, d[["x_px", "y_px"]].values)
    prior = shot.p0 + 1.5 * shot.plane.direction      # 目安を飛ぶ向きに 1.5 m ずらす
    gaps = []
    for quad in (False, True):
        f = fit_trajectory(obs, shot.t_impact, "plane", shot.plane, phys,
                           FitOptions(ci_method="none", p0_prior_m=prior, p0_prior_sigma_m=(0.3, 0.3),
                                      quadratic_priors=quad))
        gaps.append(float(np.linalg.norm(f.p0 - prior)))
    assert gaps[1] <= gaps[0] + 1e-6


def test_fd_steps_have_a_floor():
    """数値微分の幅は diff_step × max(|x|, 下限)。下限なしなら scipy の least_squares と同じ幅（ROADMAP v2 の I3）。"""
    from scipy.optimize._numdiff import _compute_absolute_step

    from smash_speed.physics import fd_steps

    x = np.array([0.0, -0.0, 1e-14, -1e-14, 0.5, -3.0, 2.0e5])
    np.testing.assert_array_equal(fd_steps(1e-6, x, None), _compute_absolute_step(1e-6, x, x, "2-point"))
    np.testing.assert_array_equal(fd_steps(1e-6, x, 1.0),
                                  1e-6 * np.where(x >= 0, 1, -1) * np.maximum(np.abs(x), 1.0))


def test_fd_jacobian_without_floor_is_the_same_as_least_squares(cfg, phys, camera):
    """``_FdJacobian`` は下限なしなら、scipy の数値微分（"2-point"）と 1 ビットも同じ解になる。"""
    from scipy.optimize import least_squares

    from smash_speed.physics import _FdJacobian, _Model, _residuals

    shot = synth.make_shot(70.0, cfg, phys)
    df = synth.observe(shot, camera, 60, phys, np.random.default_rng(2), noise_px=1.0, dropout=0.1)
    d = df[df.visible == 1]
    obs = [Observations(camera, d.timestamp.values, d[["x_px", "y_px"]].values)]
    model = _Model("plane", shot.plane, False, phys.terminal_velocity, refine_plane=True)
    x0 = model.pack(shot.p0 + 0.3, shot.v0 * 0.9, phys.terminal_velocity)
    args = (model, obs, phys, shot.t_impact)
    kw = dict(loss="huber", f_scale=2.0, x_scale="jac", max_nfev=300)
    a = least_squares(_residuals, x0, args=args, diff_step=1e-6, **kw)
    fd = _FdJacobian(_residuals, 1e-6, None)
    b = least_squares(fd.residuals, x0, jac=fd.jacobian, args=args, **kw)
    np.testing.assert_array_equal(a.x, b.x)
    assert a.nfev == b.nfev


def test_tiny_plane_correction_is_not_stuck(cfg, phys, camera):
    """平面の向きの補正がごく小さい（0 でない）値から始まっても動いて、合う平面が見つかる（ROADMAP v2 の I3）。

    幅が「値 × 1e-6」だけだと、10⁻¹⁴ rad を動かしても残差が変わらず、ヤコビアンの列が 0 になって止まる。
    """
    from smash_speed.geometry import FlightPlane

    shot = synth.make_shot(80.0, cfg, phys)
    df = synth.observe(shot, camera, 60, phys, np.random.default_rng(0), noise_px=0.5, dropout=0.0)
    d = df[df.visible == 1]
    obs = Observations(camera, d.timestamp.values, d[["x_px", "y_px"]].values)
    a, dd = np.deg2rad(4.0), shot.plane.direction
    wrong = FlightPlane(shot.plane.origin + 0.3 * shot.plane.normal,
                        np.array([np.cos(a) * dd[0] - np.sin(a) * dd[1], np.sin(a) * dd[0] + np.cos(a) * dd[1], 0.0]))
    rms, speed = {}, {}
    for floor in (None, 1.0):
        f0 = fit_trajectory(obs, shot.t_impact, "plane", wrong, phys,
                            FitOptions(ci_method="none", diff_step_floor=floor))
        f = fit_trajectory(obs, shot.t_impact, "plane", wrong, phys,
                           FitOptions(refine_plane=True, ci_method="none", diff_step_floor=floor),
                           np.r_[f0.theta[:4], 1e-14, 1e-14])
        rms[floor], speed[floor] = f.rms_px, f.speed0
    assert rms[None] > 10.0          # 下限なし：平面が動かず、平面を固定したときと同じずれのまま
    assert rms[1.0] < 1.0
    assert abs(speed[1.0] - 80.0) < 2.0
