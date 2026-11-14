"""まとめて解く当てはめ（fit_batch.py・fit_gpu.py）のテスト。"""

import numpy as np
import pytest
from scipy.optimize import least_squares

from smash_speed import fit_batch as fb
from smash_speed.config import load_config
from smash_speed.geometry import FlightPlane
from smash_speed.physics import FitOptions, Observations, PhysicsParams, fit_trajectory, simulate
from smash_speed.shot_fit import _fit_map, fit_shots, grid_fit, grid_planes, loose_fit
from smash_speed.synth_rally import make_rally, observe_rally, rally_camera

PHYS = PhysicsParams()


@pytest.fixture(scope="module")
def cam():
    return rally_camera(load_config())


@pytest.fixture(scope="module")
def rally(cam):
    """30 fps の合成ラリー 4 本と、ショットごとの観測。"""
    from smash_speed.shot_fit import split_observations

    r = make_rally(np.random.default_rng(4), PHYS, n_shots=4)
    det = observe_rally(r, cam, 30.0, PHYS, np.random.default_rng(0))
    return r, det, split_observations(det, cam, [s.t_hit for s in r.segments])


# =============================================================================
# 積分
# =============================================================================
def test_trajectories_match_simulate():
    """決まった刻みの RK4 とエルミート補間は、``physics.simulate``（DOP853）と 5 µm 以内で合う。傾きは数値微分と合う。"""
    rng = np.random.default_rng(0)
    n = 60
    speed = rng.uniform(1, 170, n)
    ang = rng.uniform(-1.2, 1.4, n)
    v0 = np.column_stack([speed * np.cos(ang), speed * np.sin(ang)])
    tau = np.sort(rng.uniform(0.005, 4.0, (n, 20)), axis=1)
    tau[:20] *= 0.1
    tau[5, 12:] = np.nan                                      # 使わない所は NaN
    tr = fb.trajectories(v0, np.full(n, PHYS.k), np.full(n, PHYS.g), tau)
    for b in range(n):
        ok = np.isfinite(tau[b])
        start = np.array([v0[b, 0], 0.0, v0[b, 1]])
        p, _ = simulate(np.zeros(3), start, 0.0, tau[b, ok], PHYS)
        assert np.abs(tr[b, ok, 0, :] - p[:, [0, 2]]).max() < 5e-6
        for j, axis in enumerate((0, 2)):
            h = np.zeros(3)
            h[axis] = 1e-4 * max(1.0, abs(start[axis]))
            d = (simulate(np.zeros(3), start + h, 0.0, tau[b, ok], PHYS)[0]
                 - simulate(np.zeros(3), start - h, 0.0, tau[b, ok], PHYS)[0])[:, [0, 2]] / (2 * h[axis])
            assert np.abs(tr[b, ok, 1 + j, :] - d).max() < 1e-4 * (1 + np.abs(d).max())


def test_time_grid_is_a_prefix_and_serve_merges_requests():
    """刻みはどこまで作っても前の方が同じ。いくつかの要求をまとめて積分しても、別々に積分したときと同じ。"""
    short, long = fb.time_grid(0.3), fb.time_grid(3.0)
    assert np.array_equal(long[:len(short)], short) and short[-1] >= 0.3 and long[-1] >= 3.0
    assert np.diff(long).min() >= fb.STEP_MIN_S * 0.999 and np.diff(long).max() <= fb.STEP_MAX_S * 1.001
    rng = np.random.default_rng(1)
    reqs = []
    for n, m in ((3, 5), (7, 11), (1, 2)):
        reqs.append((rng.uniform(-30, 60, (n, 2)), np.full(n, PHYS.k), np.full(n, PHYS.g),
                     np.sort(rng.uniform(0.01, 2.5, (n, m)), axis=1)))
    merged = fb.serve(reqs)
    for r, out in zip(reqs, merged):
        assert np.allclose(out, fb.trajectories(*r), rtol=1e-12, atol=1e-12)


# =============================================================================
# 信頼領域法
# =============================================================================
class _Curve:
    """y = a·exp(−b·t) + c の当てはめ（積分はいらないので、形だけの要求を出す）。"""

    def __init__(self, t, y):
        self.t, self.y = t, y

    def request(self, i, x):
        return np.ones((len(i), 2)), np.full(len(i), PHYS.k), np.full(len(i), PHYS.g), np.full((len(i), 1), 0.01)

    def residuals(self, i, x, tr):
        e = np.exp(-x[:, 1:2] * self.t[i])
        f = x[:, 0:1] * e + x[:, 2:3] - self.y[i]
        return f, np.stack([e, -x[:, 0:1] * self.t[i] * e, np.ones_like(e)], axis=2)


@pytest.mark.parametrize("loss", ["linear", "huber", "soft_l1"])
def test_trf_follows_scipy_least_squares(loss):
    """まとめて解く信頼領域法は、同じ式のヤコビアンを渡した ``scipy.optimize.least_squares`` と同じ道すじをたどる
    （同じ解・同じ評価の回数・同じ終わり方）。評価の回数の上限で止まる問題も同じ。"""
    rng = np.random.default_rng(3)
    n, m = 12, 25
    t = np.tile(np.linspace(0, 4, m), (n, 1))
    truth = np.column_stack([rng.uniform(1, 5, n), rng.uniform(0.3, 2, n), rng.uniform(-1, 1, n)])
    y = truth[:, 0:1] * np.exp(-truth[:, 1:2] * t) + truth[:, 2:3] + rng.normal(0, 0.05, (n, m))
    y[:, 3] += 3.0                                             # 外れ値
    x0 = truth * rng.uniform(0.3, 2.5, (n, 3)) + 0.1
    max_nfev = np.where(np.arange(n) % 4 == 0, 4, 200)         # 4 個に 1 個は、すぐ上限で止める
    ev = _Curve(t, y)
    ones = np.ones(n)
    sol = fb.run(fb._trf(ev, x0, np.full(n, m), loss, 0.2 * ones, 1e-8 * ones, 1e-8 * ones, 1e-8, max_nfev))
    for b in range(n):
        one = np.array([b])
        ref = least_squares(lambda x: ev.residuals(one, x[None], None)[0][0], x0[b],
                            jac=lambda x: ev.residuals(one, x[None], None)[1][0], x_scale="jac", loss=loss,
                            f_scale=0.2, max_nfev=int(max_nfev[b]))
        assert sol["status"][b] == ref.status and sol["nfev"][b] == ref.nfev
        assert np.allclose(sol["x"][b], ref.x, rtol=1e-9, atol=1e-12)
        assert np.isclose(sol["cost"][b], ref.cost, rtol=1e-10)
        assert np.allclose(sol["fun"][b], ref.fun, rtol=1e-9, atol=1e-12)
        # 外れ値の行の重み（√ε の大きさ）は、丸めの打ち消しで決まるので、行の大きさでは比べない
        assert np.allclose(sol["jac"][b], ref.jac, rtol=1e-8, atol=1e-7)


# =============================================================================
# 当てはめ
# =============================================================================
def _close(a, b, rel=2e-4, pos=2e-3):
    assert (a is None) == (b is None)
    if a is None:
        return
    assert a.success == b.success
    assert abs(b.speed0 / a.speed0 - 1) < rel
    assert np.linalg.norm(b.p0 - a.p0) < pos
    assert abs(b.rms_px - a.rms_px) < 1e-3 * (1 + a.rms_px)
    assert b.n_obs == a.n_obs and b.mode == a.mode and b.terminal_velocity == a.terminal_velocity


def test_fit_trajectories_match_fit_trajectory(rally, cam):
    """当てはめの形（平面の補正のあり・なし、打点の目安、初期値の与え方、信頼区間）ごとに、1 回ずつの当てはめと合う。"""
    r, _, obs = rally
    segs = r.segments
    jobs = []
    for k, s in enumerate(segs):
        a = s.p0
        b = segs[k + 1].p0 if k + 1 < len(segs) else a + np.array([4.0 * (1 if s.side == "h" else -1), 0.5, 0.0])
        plane = FlightPlane.from_points(a, b)
        tilted = FlightPlane(plane.origin + 0.2 * plane.normal,
                             plane.direction + 0.05 * plane.normal)       # 少しずれた平面から
        jobs.append(fb.FitJob(obs[k], s.t_hit, plane, PHYS, FitOptions()))
        jobs.append(fb.FitJob(obs[k], s.t_hit, tilted, PHYS, FitOptions(refine_plane=True)))
        jobs.append(fb.FitJob(obs[k], s.t_hit, tilted, PHYS,
                              FitOptions(refine_plane=True, p0_prior_m=a + 0.1, quadratic_priors=True,
                                         init_state=(a + 0.05, s.v0 * 0.9), ci_method="none")))
        jobs.append(fb.FitJob(obs[k], s.t_hit, plane, PHYS, FitOptions(p0_prior_m=a, t_ref_sigma=0.002),
                              theta0=np.r_[0.1, a[2], s.v0 @ plane.direction, s.v0[2]]))
        jobs.append(fb.FitJob(obs[k], s.t_hit, plane, PHYS, FitOptions(loss="linear", max_speed_mps=None)))
    got = fb.fit_trajectories(jobs)
    for job, g in zip(jobs, got):
        ref = fit_trajectory(job.obs, job.t_ref, "plane", job.plane, job.phys, job.opts, theta0=job.theta0)
        _close(ref, g)
        assert g.message == ref.message and g.ci_method == ref.ci_method
        assert np.allclose(g.plane.direction, ref.plane.direction, atol=1e-4)
        assert np.allclose(g.plane.origin, ref.plane.origin, atol=2e-3)
        if ref.speed0_ci is not None:
            assert np.allclose(g.speed0_ci, ref.speed0_ci, rtol=2e-3)
            assert np.isclose(g.speed0_sd_timing, ref.speed0_sd_timing, rtol=1e-3)
            assert np.allclose(g.cov_theta, ref.cov_theta, rtol=5e-3, atol=1e-9)


def test_jobs_that_cannot_be_fitted_or_batched(rally, cam):
    """コマが足りない当てはめは None（``fit_trajectory`` は ``ValueError``）。まとめて解けない当てはめ（終端速度の推定・
    打点より前の観測）は 1 本ずつ解いた結果そのもの。"""
    r, _, obs = rally
    s = r.segments[0]
    plane = FlightPlane.from_points(s.p0, r.segments[1].p0)
    few = Observations(cam, obs[0].t[:2], obs[0].uv[:2])
    vt = FitOptions(estimate_terminal_velocity=True, ci_method="none")
    early = s.t_hit + 0.2
    jobs = [fb.FitJob(few, s.t_hit, plane, PHYS, FitOptions()), fb.FitJob(obs[0], s.t_hit, plane, PHYS, vt),
            fb.FitJob(obs[0], early, plane, PHYS, FitOptions(ci_method="none"))]
    assert not fb.supported(jobs[1]) and not fb.supported(jobs[2]) and fb.supported(jobs[0])
    got = fb.fit_trajectories(jobs)
    assert got[0] is None
    with pytest.raises(ValueError):
        fit_trajectory(few, s.t_hit, "plane", plane, PHYS, FitOptions())
    for job, g in zip(jobs[1:], got[1:]):
        ref = fit_trajectory(job.obs, job.t_ref, "plane", job.plane, job.phys, job.opts)
        assert np.array_equal(g.theta, ref.theta)


def test_grid_and_loose_fits_match_one_by_one(rally, cam):
    """``fit_shots`` の最初の段階（格子の平面・平面を大きく動かす当てはめ）を、まとめて解いても 1 回ずつと合う。
    格子は 1 回ずつの方が粗い積分（RK45）なので、合い方は少しゆるい。"""
    r, _, obs = rally
    opts = FitOptions(refine_plane=True, quadratic_priors=True)
    args = [(obs[k], s.t_hit, p, PHYS, opts) for k, s in enumerate(r.segments[:2]) for p in grid_planes(s.side)]
    one = _fit_map(grid_fit, args, 1, "cpu")
    many = _fit_map(grid_fit, args, 1, "batch")
    n_close = 0
    for a, b in zip(one, many):
        assert (a is None) == (b is None)
        # 合わない平面の当てはめ（ずれが何十 px）は道すじが不安定なので、合った平面だけを比べる
        if a is not None and a.rms_px < 5:
            _close(a, b, rel=2e-3, pos=2e-2)
            n_close += 1
    assert n_close >= 2
    best = [min((f for f in one[i:i + 9] if f is not None), key=lambda f: f.rms_px) for i in (0, 9)]
    args = [(obs[k], r.segments[k].t_hit, best[k], PHYS, opts, None, (0.5, 2.0)) for k in range(2)] \
        + [(obs[0], r.segments[0].t_hit, None, PHYS, opts, None, (0.5, 2.0))]
    one, many = _fit_map(loose_fit, args, 1, "cpu"), _fit_map(loose_fit, args, 1, "batch")
    assert one[2] is None and many[2] is None
    for a, b in zip(one[:2], many[:2]):
        _close(a, b)


def test_fit_shots_batch_backend_matches_cpu(cam):
    """ショットごとの当てはめ全体（格子 → 打点をつなぐ → 信頼区間）を、まとめて解いても 1 回ずつと合う。"""
    r = make_rally(np.random.default_rng(11), PHYS, n_shots=5)
    det = observe_rally(r, cam, 60.0, PHYS, np.random.default_rng(2))
    t_hits = [s.t_hit + 0.004 for s in r.segments]
    sides = [s.side for s in r.segments]
    opts = FitOptions(refine_plane=True)
    one = fit_shots(det, t_hits, cam, PHYS, sides, opts)
    many = fit_shots(det, t_hits, cam, PHYS, sides, opts, backend="batch")
    assert len(one) == len(many)
    for a, b in zip(one, many):
        assert abs(b.t_hit - a.t_hit) < 2e-4 and b.n_obs == a.n_obs
        _close(a.fit, b.fit, rel=2e-3, pos=1e-2)
        assert (a.fit.speed0_ci is None) == (b.fit.speed0_ci is None)
    with pytest.raises(ValueError):
        fit_shots(det, t_hits, cam, PHYS, sides, opts, backend="nope")


def test_solve_together_gives_each_task_its_own_results(rally, cam):
    """いくつかの計算（ラリー）を同時に進めても、それぞれの結果は 1 つずつ解いたときと同じ。"""
    r, _, obs = rally
    segs = r.segments
    lists = []
    for k in (0, 1, 2):
        plane = FlightPlane.from_points(segs[k].p0, segs[k + 1].p0)
        lists.append([fb.FitJob(obs[k], segs[k].t_hit, plane, PHYS, FitOptions(refine_plane=bool(j), ci_method="none"))
                      for j in range(2)])

    def task(jobs):
        first = yield jobs[:1]
        second = yield jobs[1:]
        empty = yield []
        return first + second + empty

    got = fb.solve_together([task(jobs) for jobs in lists])
    for jobs, res in zip(lists, got):
        alone = fb.fit_trajectories(jobs)
        assert len(res) == 2
        for a, b in zip(alone, res):
            assert np.allclose(b.theta, a.theta, rtol=1e-9, atol=1e-12)


def test_finite_difference_emulation_keeps_tiny_plane_corrections_frozen(rally, cam, monkeypatch):
    """1 回ずつの当てはめの数値微分は、ごく小さい（0 でない）パラメータでは観測が動かず、傾きが 0 になる。
    それを再現する（``EMULATE_FD``）と、その列の観測の行は 0、平面の目安の行は残る。"""
    r, _, obs = rally
    s = r.segments[0]
    plane = FlightPlane.from_points(s.p0, r.segments[1].p0)
    job = fb.FitJob(obs[0], s.t_hit, plane, PHYS, FitOptions(refine_plane=True, ci_method="none"),
                    theta0=np.r_[0.0, s.p0[2], s.v0 @ plane.direction, s.v0[2], 1e-13, 0.0])
    seen = []
    orig = fb._Batch.residuals

    def spy(self, i, theta, tr):
        f, jac = orig(self, i, theta, tr)
        if not seen:
            seen.append(jac[0].copy())
        return f, jac

    monkeypatch.setattr(fb._Batch, "residuals", spy)
    fb.fit_trajectories([job])
    n = 2 * len(obs[0].t)
    assert np.all(seen[0][:n, 4] == 0) and seen[0][n, 4] > 0       # dψ = 1e-13：観測の行は 0、目安の行は 1/σ
    assert np.any(seen[0][:n, 5] != 0)                             # dn = 0：数値微分は既定の幅で動かすので生きている
    seen.clear()
    monkeypatch.setattr(fb, "EMULATE_FD", False)
    fb.fit_trajectories([job])
    assert np.any(seen[0][:n, 4] != 0)


# =============================================================================
# GPU（onnxruntime）
# =============================================================================
def test_gpu_backend_falls_back_to_numpy(monkeypatch):
    """GPU が使えない PC では、積分は NumPy のまま（当てはめは止まらない）。"""
    from smash_speed import fit_gpu

    monkeypatch.setattr(fit_gpu, "available", lambda: False)
    fn, name = fb.get_integrator("dml")
    assert name == "numpy" and fn is fb.integrate
    assert fb.get_integrator(None)[1] == "numpy" and fb.get_integrator("numpy")[1] == "numpy"


def test_onnx_graph_matches_numpy_integration():
    """ONNX のグラフにした RK4 は、CPU の倍精度なら NumPy と丸めの範囲で同じ。DirectML（GPU・単精度）は 0.1 mm 以内。"""
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from smash_speed import fit_gpu

    rng = np.random.default_rng(5)
    n = 9
    speed = rng.uniform(5, 170, n)
    ang = rng.uniform(-1.2, 1.4, n)
    v0 = np.column_stack([speed * np.cos(ang), speed * np.sin(ang)])
    k, g = np.full(n, PHYS.k), np.full(n, PHYS.g)
    grid = fb.time_grid(0.6)
    pos, vel = fb.integrate(v0, k, g, grid)
    p_cpu, v_cpu = fit_gpu.integrate_cpu(v0, k, g, grid)
    assert np.abs(p_cpu - pos).max() < 1e-11 and np.abs(v_cpu - vel).max() < 1e-9
    import onnxruntime as ort

    if "DmlExecutionProvider" not in ort.get_available_providers() or not fit_gpu.available():
        pytest.skip("DirectML（GPU）がない")
    p_gpu, _ = fit_gpu.integrate(v0, k, g, grid)
    assert np.abs(p_gpu - pos).max() < 1e-4
