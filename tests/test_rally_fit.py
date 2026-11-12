"""ラリー全体をまとめた当てはめ（rally_fit.py）のテスト。"""

import numpy as np
import pandas as pd
import pytest

from smash_speed.config import load_config
from smash_speed.physics import MPS_TO_KMH, FitOptions, PhysicsParams, simulate
from smash_speed.rally_fit import (JointOptions, VtEstimate, _Problem, _Shot, combine_vt, fit_rallies, integrate,
                                   interpolate, joint_fit, time_grid)
from smash_speed.shot_fit import fit_shots, joint_job, shot_fits_to_shots
from smash_speed.synth_rally import corner_rally_camera, make_rally, observe_rally, rally_camera

PHYS = PhysicsParams()
OPTS = FitOptions(refine_plane=True)


@pytest.fixture(scope="module")
def corner_fit():
    """角のカメラ・30 fps の合成ラリー（8 本まで）の、ショットごとの当てはめ。"""
    cam = corner_rally_camera()
    rally = make_rally(np.random.default_rng(11), PHYS, n_shots=8)
    det = observe_rally(rally, cam, 30.0, PHYS, np.random.default_rng(2))
    segs = rally.segments
    fits = fit_shots(det, [s.t_hit + 0.006 for s in segs], cam, PHYS, [s.side for s in segs], OPTS)
    return cam, rally, det, fits


def _errors(segs, fits):
    return np.array([abs(sf.speed_kmh / (s.speed0 * MPS_TO_KMH) - 1) for s, sf in zip(segs, fits)])


def test_integrate_matches_simulate():
    """まとめて進める RK4 と補間が、``physics.simulate``（DOP853）と同じ軌道になる（打つ前の時刻も）。"""
    o = JointOptions()
    v0 = np.array([[95.0, 5.0, -12.0], [22.0, -3.0, 18.0], [3.0, 1.0, 4.0]])
    vt = np.array([6.7, 6.3, 7.1])
    grid, i0 = time_grid(-0.02, 2.2, o)
    assert grid[i0] == 0.0 and np.all(np.diff(grid) > 0)
    y = integrate(v0, vt, grid, i0, PHYS.g)
    s = np.array([-0.015, 0.0, 0.0133, 0.1, 0.299, 0.31, 0.777, 2.1])
    for b in range(3):
        pos, vel = interpolate(grid, y[:, b:b + 1], s)
        ref_p, ref_v = simulate(np.zeros(3), v0[b], 0.0, s, PhysicsParams(PHYS.g, vt[b]))
        assert np.abs(pos[:, 0] - ref_p).max() < 1e-5
        assert np.abs(vel[:, 0] - ref_v).max() < 1e-3
    # 刻みの外は NaN
    assert np.isnan(interpolate(grid, y[:, :1], np.array([grid[-1] + 0.1]))[0]).all()


def test_jacobian_matches_finite_differences():
    """残差のヤコビアン（つなぎ目・足元・打球音・終端速度を含む）が、数値微分と合う。"""
    cam = rally_camera(load_config())
    rally = make_rally(np.random.default_rng(3), PHYS, n_shots=3)
    det = observe_rally(rally, cam, 60.0, PHYS, np.random.default_rng(0), noise_px=0.0, dropout=0.0)
    segs = rally.segments
    shots, x = [], []
    for k, s in enumerate(segs):
        m = (det.shot_index == k) & (det.visible > 0)
        t, uv = det.timestamp[m].to_numpy(float)[1:], det.loc[m, ["x_px", "y_px"]].to_numpy(float)[1:]
        dist = float(np.linalg.norm(cam.center - s.p0))
        shots.append(_Shot(t, uv, s.side, k > 0, s.p0[:2] + 0.2 if k != 1 else None, s.p0 + 0.1,
                           s.t_hit + dist / 343.0 if k == 0 else None, s.t_hit + 0.004, -np.inf, np.inf, k != 1))
        x += [s.t_hit + 0.003, *(s.p0 + [0.3, -0.2, 0.1]), *(s.v0 * 1.03)]
    x = np.array(x + [np.log(6.5)])
    prob = _Problem(shots, cam, PHYS, JointOptions(), 2.0, 2.0, 0.02, 343.0, True, 170.0)
    assert prob.est and prob.n_par == 22
    r, jac, info = prob.evaluate(x)
    assert np.isfinite(r).all() and np.isfinite(info["gaps"][1:]).all()
    num = np.empty_like(jac)
    for j in range(len(x)):
        h = 1e-6 * max(1.0, abs(x[j]))
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        num[:, j] = (prob.evaluate(xp)[0] - prob.evaluate(xm)[0]) / (2 * h)
    scale = np.abs(num).max(axis=0)
    assert (scale > 0).all()
    assert (np.abs(jac - num).max(axis=0) / scale).max() < 2e-3


def test_joint_fit_links_shots_and_improves_short_ones(corner_fit):
    """角のカメラでは短いショットの奥行きが 1 本では決まらない。つなぐと、打点が合い、初速の誤差が減る。"""
    cam, rally, det, fits = corner_fit
    segs = rally.segments
    res = joint_fit(fits, det, cam, PHYS, OPTS, JointOptions(), estimate_vt=False)
    assert all(res.joint) and all(res.links[1:]) and not res.cut
    assert np.nanmax(res.gaps_m[1:]) < 0.1
    before, after = _errors(segs, fits), _errors(segs, res.fits)
    assert after.mean() < 0.7 * before.mean()
    assert after.max() < 0.15
    hit_before = np.array([np.linalg.norm(sf.fit.p0 - s.p0) for s, sf in zip(segs, fits)])
    hit_after = np.array([np.linalg.norm(sf.fit.p0 - s.p0) for s, sf in zip(segs, res.fits)])
    assert hit_after.mean() < 0.6 * hit_before.mean()
    for i, sf in enumerate(res.fits):
        # 打った時刻・飛行の終わりは、となりのショットと同じ時刻
        if i + 1 < len(res.fits):
            assert sf.t_end == res.fits[i + 1].t_hit
        assert sf.fit.message == "joint" and sf.fit.speed0_ci is not None
        assert sf.fit.speed0_ci[0] < sf.fit.speed0 < sf.fit.speed0_ci[1]
    # 元の当てはめは書き換えない
    assert all(sf.fit.message != "joint" for sf in fits)
    shots = shot_fits_to_shots(res.fits, PHYS)
    assert all("ラリーでつないだ" in s.note for s in shots)


def test_fit_shots_joint_option_gives_the_same_result(corner_fit):
    """``fit_shots(joint=…)`` は、ショットごとの当てはめの後に ``fit_rallies`` を呼んだものと同じ。"""
    cam, rally, det, fits = corner_fit
    segs = rally.segments
    o = JointOptions(estimate_vt=False)
    res, rec = fit_rallies([joint_job(fits, det, cam)], PHYS, OPTS, o)
    assert rec["used"] is False and rec["default_mps"] == PHYS.terminal_velocity
    direct = fit_shots(det, [s.t_hit + 0.006 for s in segs], cam, PHYS, [s.side for s in segs], OPTS, joint=o)
    assert [sf.speed_kmh for sf in direct] == pytest.approx([sf.speed_kmh for sf in res[0].fits], rel=1e-6)


def test_a_wrong_track_is_cut_and_keeps_its_own_fit(corner_fit):
    """1 本のショットの検出が別の物（まっすぐ動く点）にすり替わっても、そのつなぎ目だけを切って、ほかはつなぐ。"""
    cam, rally, det, _ = corner_fit
    segs = rally.segments
    bad = 4
    det = det.copy()
    m = (det.shot_index == bad) & (det.visible > 0)
    n = int(m.sum())
    det.loc[m, "x_px"] = np.linspace(300.0, 900.0, n)
    det.loc[m, "y_px"] = np.linspace(900.0, 200.0, n)
    fits = fit_shots(det, [s.t_hit + 0.006 for s in segs], cam, PHYS, [s.side for s in segs], OPTS)
    res = joint_fit(fits, det, cam, PHYS, OPTS, JointOptions(), estimate_vt=False)
    assert not res.joint[bad] and res.fits[bad] is fits[bad]
    assert not res.links[bad] and not res.links[bad + 1]
    assert any(c[0] == "link" for c in res.cut)
    good = [i for i in range(len(segs)) if abs(i - bad) > 1]
    assert all(res.joint[i] for i in good if i not in (0, len(segs) - 1))
    err = _errors([segs[i] for i in good], [res.fits[i] for i in good])
    assert err.max() < 0.15


def test_a_missed_hit_does_not_spoil_the_rally(corner_fit):
    """打球を 1 本見落としても（2 本の飛行が 1 本のショットになる）、ほかのショットの初速は崩れない。"""
    cam, rally, det, fits = corner_fit
    segs = rally.segments
    miss = 3
    keep = [i for i in range(len(segs)) if i != miss]
    part = fit_shots(det, [segs[i].t_hit + 0.006 for i in keep], cam, PHYS, [segs[i].side for i in keep], OPTS)
    res = joint_fit(part, det, cam, PHYS, OPTS, JointOptions(), estimate_vt=False)
    far = [k for k, i in enumerate(keep) if abs(i - miss) > 1]
    after = _errors([segs[keep[k]] for k in far], [res.fits[k] for k in far])
    before = _errors([segs[keep[k]] for k in far], [part[k] for k in far])
    assert after.max() < max(0.15, 1.2 * before.max())
    assert after.mean() <= before.mean() + 0.01


def test_combine_vt_needs_enough_evidence():
    """終端速度は、長いショットが足りて不確かさが小さいときだけ使う。そうでなければ設定の値のまま。"""
    o = JointOptions()
    vt, rec = combine_vt([None, None], 6.7, o)
    assert vt == 6.7 and not rec["used"] and rec["value_mps"] is None
    vt, rec = combine_vt([VtEstimate(6.3, 0.05, 2)], 6.7, o)       # 長いショットが少ない
    assert vt == 6.7 and not rec["used"] and rec["value_mps"] == 6.3
    vt, rec = combine_vt([VtEstimate(6.3, 0.4, 5)], 6.7, o)        # 不確かさが大きい
    assert vt == 6.7 and not rec["used"]
    vt, rec = combine_vt([VtEstimate(6.2, 0.2, 5), VtEstimate(7.4, 0.2, 5)], 6.7, o)   # ラリーどうしでばらつく
    assert vt == 6.7 and not rec["used"] and rec["sigma_mps"] > 0.3
    vt, rec = combine_vt([VtEstimate(6.28, 0.06, 3), VtEstimate(6.35, 0.08, 2), None], 6.7, o)
    assert rec["used"] and 6.28 < vt < 6.35 and rec["n_flights"] == 5 and rec["sigma_mps"] <= 0.06
    vt, rec = combine_vt([VtEstimate(4.2, 0.05, 6)], 6.7, o)       # ありえない値
    assert vt == 6.7 and not rec["used"]


def test_terminal_velocity_from_long_flights():
    """本当の終端速度が設定（6.7 m/s）と違うシャトル（6.3 m/s）でも、長いショットから求めて初速のかたよりを減らす。"""
    cam = rally_camera(load_config())
    true_phys = PhysicsParams(PHYS.g, 6.3)
    rng = np.random.default_rng(4)
    jobs, truth = [], []
    for r in range(3):
        rally = make_rally(rng, true_phys, n_shots=7, serve="serve_long")
        det = observe_rally(rally, cam, 30.0, true_phys, np.random.default_rng(r))
        segs = rally.segments
        fits = fit_shots(det, [s.t_hit for s in segs], cam, PHYS, [s.side for s in segs], OPTS)
        jobs.append(joint_job(fits, det, cam))
        truth.append(segs)
    res, rec = fit_rallies(jobs, PHYS, OPTS, JointOptions())
    assert rec["used"] and abs(rec["value_mps"] - 6.3) < 0.15 and rec["n_flights"] >= 3
    before = np.concatenate([_errors(s, j["fits"]) for s, j in zip(truth, jobs)])
    after = np.concatenate([_errors(s, r.fits) for s, r in zip(truth, res)])
    assert after.mean() < 0.6 * before.mean()
    for r in res:
        assert all(abs(sf.fit.terminal_velocity - rec["value_mps"]) < 1e-3 for sf in r.fits if sf.fit is not None)


def test_shot_list_joint_option():
    """ショットの一覧（``analyze_hits``）で、まとめた当てはめを選ぶと、記録（つないだ数・終端速度）が残る。"""
    from smash_speed.hits import Hit
    from smash_speed.shot_list import analyze_hits

    cfg = load_config()
    cam = rally_camera(cfg)
    rng = np.random.default_rng(5)
    rallies = [make_rally(rng, PHYS, n_shots=4, t_start=0.5)]
    rallies.append(make_rally(rng, PHYS, n_shots=4, server_side="o", t_start=rallies[0].t_end + 5.0))
    det = pd.concat([observe_rally(r, cam, 30.0, PHYS, np.random.default_rng(k)) for k, r in enumerate(rallies)],
                    ignore_index=True)
    det["frame"] = np.arange(len(det))
    segs = rallies[0].segments + rallies[1].segments
    hits = [Hit(s.t_hit, s.side, "track") for s in segs]
    msgs: list[str] = []
    info: dict = {}
    base = analyze_hits(det, hits, cam, PHYS, cfg, ci=False, progress=lambda m: None)
    shots = analyze_hits(det, hits, cam, PHYS, cfg, ci=False, progress=msgs.append, joint=JointOptions(),
                         joint_info=info)
    assert len(shots) == len(base) == len(segs)
    assert info["n_joint"] >= len(segs) - 2 and info["n_links"] >= len(segs) - 3
    assert info["terminal_velocity"]["default_mps"] == PHYS.terminal_velocity
    assert any("つないで" in m for m in msgs)
    assert sum("ラリーでつないだ" in s.note for s in shots) == info["n_joint"]
    err = [abs(e.speed_kmh / (s.speed0 * MPS_TO_KMH) - 1) for e, s in zip(shots, segs) if e.speed_kmh]
    err0 = [abs(e.speed_kmh / (s.speed0 * MPS_TO_KMH) - 1) for e, s in zip(base, segs) if e.speed_kmh]
    assert len(err) >= len(err0) and np.mean(err) < np.mean(err0) + 0.01
