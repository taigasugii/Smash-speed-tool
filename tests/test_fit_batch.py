"""当てはめをまとめて解く計算（fit_batch.py。ROADMAP F3e）のテスト。"""

import threading

import numpy as np
import pandas as pd
import pytest

from smash_speed import fit_batch
from smash_speed.config import load_config
from smash_speed.fit_batch import BatchGate, fit_trajectories, gate_session, simulate_batch, solve_batch
from smash_speed.geometry import FlightPlane
from smash_speed.hits import find_hits
from smash_speed.physics import FitOptions, Observations, PhysicsParams, fit_plan, fit_trajectory, simulate
from smash_speed.shot_fit import fit_shots, grid_planes, split_observations
from smash_speed.shot_list import analyze_hits
from smash_speed.synth_rally import make_rally, observe_rally, rally_camera

CFG = load_config()
PHYS = PhysicsParams.from_config(CFG)
CAM = rally_camera(CFG)


@pytest.fixture(scope="module")
def rally():
    """30 fps の合成ラリー（5 本）と、ショットごとの観測。"""
    r = make_rally(np.random.default_rng(3), PHYS, n_shots=5)
    det = observe_rally(r, CAM, 30.0, PHYS, np.random.default_rng(1))
    obs = split_observations(det, CAM, [s.t_hit for s in r.segments], 0.01, None)
    return r, det, obs


def _jobs(rally, opts):
    r, _, obs = rally
    return [(o, s.t_hit, "plane", p, PHYS, opts, None)
            for s, o in zip(r.segments, obs) if len(o.t) >= 5 for p in grid_planes(s.side)[:3]]


def test_integration_matches_simulate():
    """2 次元の RK4（幾何級数の刻み）とエルミート補間の位置は、``physics.simulate``（DOP853）と 0.01 mm 以内。"""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(100):
        v0 = rng.normal(0, 1, 3)
        v0 *= rng.uniform(2, 170) / np.linalg.norm(v0)
        p0 = rng.uniform(-3, 3, 3)
        t = np.sort(rng.uniform(0.002, 2.5, 40))
        worst = max(worst, np.abs(simulate(p0, v0, 0.0, t, PHYS)[0]
                                  - simulate_batch(p0, v0, 0.0, t, PHYS.g, PHYS.terminal_velocity)).max())
    assert worst < 1e-5


def test_batch_fits_match_cpu_fits(rally):
    """まとめた当てはめは、1 つずつの ``fit_trajectory`` と初速 0.001 km/h・打点 0.1 mm 以内で同じ
    （平面の補正・打点の目安・信頼区間つき）。"""
    r, _, obs = rally
    opts = FitOptions(refine_plane=True, quadratic_priors=True, ci_method="jacobian", t_ref_sigma=0.002)
    jobs = _jobs(rally, opts)
    jobs += [(o, s.t_hit, "plane", FlightPlane.from_points(s.p0, s.p0 + s.v0 * 0.1), PHYS,
              FitOptions(refine_plane=True, quadratic_priors=True, p0_prior_m=s.p0 + 0.2, p0_prior_sigma_m=(0.5, 0.5),
                         init_state=(s.p0, s.v0), ci_method="jacobian"), None)
             for s, o in zip(r.segments, obs) if len(o.t) >= 5]
    got = fit_trajectories(jobs)
    assert len(got) == len(jobs)
    for job, b in zip(jobs, got):
        a = fit_trajectory(*job)
        assert isinstance(b, type(a))
        assert abs(a.speed0 - b.speed0) * 3.6 < 1e-3
        assert np.abs(a.p0 - b.p0).max() < 1e-4
        assert abs(a.rms_px - b.rms_px) < 1e-4
        assert a.success == b.success
        assert abs(a.speed0_ci[0] - b.speed0_ci[0]) * 3.6 < 1e-2


def test_batch_reports_the_same_errors(rally):
    """1 つずつの当てはめが例外になる当てはめ（観測が少なすぎる）は、まとめても同じ例外を返す。"""
    _, _, obs = rally
    o = obs[0]
    few = Observations(o.camera, o.t[:2], o.uv[:2])
    plane = FlightPlane(np.zeros(3), np.array([1.0, 0.0, 0.0]))
    good = (o, float(o.t[0]) - 0.02, "plane", plane, PHYS, FitOptions(ci_method="none"), None)
    out = fit_trajectories([(few, float(o.t[0]) - 0.02, "plane", plane, PHYS, FitOptions(), None), good])
    with pytest.raises(ValueError) as e:
        fit_trajectory(few, float(o.t[0]) - 0.02, "plane", plane, PHYS, FitOptions())
    assert isinstance(out[0], ValueError) and str(out[0]) == str(e.value)
    assert out[1].success


def test_result_does_not_depend_on_the_batch(rally):
    """1 つの当てはめの結果は、一緒に解いたほかの当てはめによらない（1 ビットも違わない）。"""
    jobs = _jobs(rally, FitOptions(refine_plane=True, ci_method="none"))
    together = fit_trajectories(jobs)
    for k in (0, len(jobs) // 2, len(jobs) - 1):
        alone = fit_trajectories([jobs[k]])[0]
        assert np.array_equal(alone.theta, together[k].theta)
        assert alone.rms_px == together[k].rms_px


def test_solve_batch_matches_least_squares(rally):
    """``solve_batch`` の最小 2 乗は、scipy の ``least_squares`` と同じ点に止まる（同じ手順の信頼領域法）。"""
    r, _, obs = rally
    s, o = r.segments[1], obs[1]
    plan = fit_plan(o, s.t_hit, "plane", FlightPlane.from_points(s.p0, s.p0 + s.v0), PHYS, FitOptions())
    reqs = next(plan)                      # 3 次元点への当てはめ（第 1 段階）
    a, b = reqs[0].solve_cpu(), solve_batch(reqs)[0]
    assert b.status == a.status and b.nfev == a.nfev
    assert np.allclose(a.x, b.x, rtol=1e-6, atol=1e-8)
    assert abs(a.cost - b.cost) <= 1e-6 * max(a.cost, 1e-12)


def test_fit_shots_batch_matches_cpu(rally):
    """``fit_shots`` の当てはめをまとめて解いても、1 つずつと同じ初速・打点・打った時刻になる。"""
    r, det, _ = rally
    t_hits = [s.t_hit + 0.004 for s in r.segments]
    sides = [s.side for s in r.segments]
    opts = FitOptions(refine_plane=True, ci_method="jacobian")
    cpu = fit_shots(det, t_hits, CAM, PHYS, sides, opts)
    bat = fit_shots(det, t_hits, CAM, PHYS, sides, opts, backend="batch")
    for a, b in zip(cpu, bat):
        assert (a.fit is None) == (b.fit is None)
        assert abs(a.t_hit - b.t_hit) < 1e-6
        if a.fit is not None:
            assert abs(a.speed_kmh - b.speed_kmh) < 1e-2
            assert np.abs(a.fit.p0 - b.fit.p0).max() < 1e-3


def test_analyze_hits_batch_is_the_same_for_any_number_of_rally_threads():
    """まとめて解くとき、同時に進めるラリーの数によらず同じショットになり、1 つずつの当てはめとも合う。"""
    rng = np.random.default_rng(5)
    rallies = [make_rally(rng, PHYS, n_shots=3, t_start=0.5)]
    rallies.append(make_rally(rng, PHYS, n_shots=3, server_side="o", t_start=rallies[0].t_end + 5.0))
    rallies.append(make_rally(rng, PHYS, n_shots=2, t_start=rallies[1].t_end + 5.0))
    det = pd.concat([observe_rally(r, CAM, 30.0, PHYS, np.random.default_rng(k)) for k, r in enumerate(rallies)],
                    ignore_index=True)
    det["frame"] = np.arange(len(det))
    hits = find_hits(det, 30.0, None, camera=CAM)
    kw = dict(ci=False, progress=lambda m: None)
    one = analyze_hits(det, hits, CAM, PHYS, CFG, rally_threads=1, fit_backend="batch", **kw)
    two = analyze_hits(det, hits, CAM, PHYS, CFG, rally_threads=2, fit_backend="batch", **kw)
    many = analyze_hits(det, hits, CAM, PHYS, CFG, fit_backend="batch", **kw)
    cpu = analyze_hits(det, hits, CAM, PHYS, CFG, fit_backend="cpu", **kw)
    key = [(s.t_hit_s, s.shot_type, s.speed_kmh, s.hit_x_m) for s in one]
    assert key == [(s.t_hit_s, s.shot_type, s.speed_kmh, s.hit_x_m) for s in two]
    assert key == [(s.t_hit_s, s.shot_type, s.speed_kmh, s.hit_x_m) for s in many]
    assert [s.shot_type for s in one] == [s.shot_type for s in cpu]
    for a, b in zip(one, cpu):
        assert abs(a.t_hit_s - b.t_hit_s) < 1e-6
        assert (a.speed_kmh is None) == (b.speed_kmh is None)
        if a.speed_kmh is not None:
            assert abs(a.speed_kmh - b.speed_kmh) < 1e-2


def test_gate_runs_waiting_threads_together(rally, monkeypatch):
    """``BatchGate`` は、仕事をしているスレッドが全部待ったら 1 回でまとめて解き、それぞれに結果を返す。
    途中で例外で終わったスレッドがあっても、ほかのスレッドは止まらない。"""
    jobs = _jobs(rally, FitOptions(ci_method="none"))[:4]
    calls = []
    real = fit_batch._fit_now
    monkeypatch.setattr(fit_batch, "_fit_now", lambda js: calls.append(len(js)) or real(js))
    out: dict[int, object] = {}
    with gate_session(3, 3) as gate:
        def work(k: int) -> None:
            gate.enter()
            try:
                if k == 2:
                    raise RuntimeError("途中で止まった")
                out[k] = fit_trajectories(jobs[2 * k:2 * k + 2])
            except RuntimeError:
                pass
            finally:
                gate.leave()

        ts = [threading.Thread(target=work, args=(k,)) for k in range(3)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(60)
    assert calls == [4]
    alone = fit_trajectories(jobs)
    assert [f.rms_px for f in out[0] + out[1]] == [f.rms_px for f in alone]
    assert isinstance(gate, BatchGate) and not fit_batch._GATE
