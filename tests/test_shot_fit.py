"""ショットごとの軌道と初速（shot_fit.py）のテスト。"""

import numpy as np
import pandas as pd
import pytest

from smash_speed.config import load_config
from smash_speed.physics import MPS_TO_KMH, FitOptions, PhysicsParams
from smash_speed.shot_fit import (audio_hit_times, fit_shots, joint_hit_points, refine_hit_times, shot_fits_to_shots,
                                  split_observations)
from smash_speed.shots import read_shots, write_shots
from smash_speed.synth_rally import make_rally, observe_rally, rally_camera

PHYS = PhysicsParams()


@pytest.fixture(scope="module")
def cam():
    return rally_camera(load_config())


@pytest.fixture(scope="module")
def rally_fit(cam):
    """60 fps の合成ラリー（6 本まで）を、正解から 6 ms ずらした打球の時刻で当てはめた結果。"""
    rally = make_rally(np.random.default_rng(11), PHYS, n_shots=6)
    det = observe_rally(rally, cam, 60.0, PHYS, np.random.default_rng(2))
    t_hits = [s.t_hit + 0.006 for s in rally.segments]
    fits = fit_shots(det, t_hits, cam, PHYS, [s.side for s in rally.segments], FitOptions(refine_plane=True))
    return rally, det, fits


def test_speed_of_every_shot_type(rally_fit):
    """山なり・遅いショットも、初速が正解に近い（打球の時刻のずれは前後の軌道が交わる時刻で直す）。"""
    rally, _, fits = rally_fit
    assert len(fits) == len(rally.segments)
    errs = []
    for s, sf in zip(rally.segments, fits):
        assert sf.fit is not None, sf.message
        errs.append(abs(sf.speed_kmh / (s.speed0 * MPS_TO_KMH) - 1))
    assert np.mean(errs) < 0.05
    assert max(errs) < 0.12
    # 2 本目からは、時刻を直して 2 ms 以内に戻る（1 本目のサーブは前の軌道がないので直せない）。
    # 遅いショット（36 km/h 未満）は交わる時刻がはっきりしないが、時刻のずれが初速にほとんど効かない
    for s, sf in zip(rally.segments[1:], fits[1:]):
        assert abs(sf.t_hit - s.t_hit) < (0.002 if s.speed0 > 10 else 0.006)


def test_hit_points_and_confidence_interval(rally_fit):
    """つないだ打点が正解の近くにあり、初速の信頼区間が出る。"""
    rally, _, fits = rally_fit
    for s, sf in zip(rally.segments, fits):
        assert np.linalg.norm(sf.fit.p0 - s.p0) < 0.8
        assert sf.fit.speed0_ci is not None
        lo, hi = sf.fit.speed0_ci
        assert lo < sf.fit.speed0 < hi


def test_shots_csv_from_fits(rally_fit, tmp_path):
    """当てはめの結果を shots.csv の形で書いて読める。最後のショットの着地点は正解の近く。"""
    rally, _, fits = rally_fit
    shots = shot_fits_to_shots(fits, PHYS)
    path = write_shots(tmp_path / "shots.csv", shots)
    back = read_shots(path)
    assert len(back) == len(rally.segments)
    for s, b in zip(rally.segments, back):
        assert b.side == s.side
        assert b.speed_ci_low_kmh < b.speed_kmh < b.speed_ci_high_kmh
        assert 0.0 <= b.confidence <= 1.0
    last = back[-1]
    assert np.hypot(last.land_x_m - rally.segments[-1].landing_xy[0],
                    last.land_y_m - rally.segments[-1].landing_xy[1]) < 0.5


def test_refine_hit_times_moves_toward_crossing(rally_fit, cam):
    """当てはめた前後の軌道から、ずらした時刻をもとに戻す（遅いショットは交わる時刻がはっきりしないので除く）。"""
    rally, _, fits = rally_fit
    for sf in fits:
        sf.t_hit += 0.01
        sf.t_end += 0.01
    try:
        t_new = refine_hit_times(fits, cam, PHYS)
    finally:
        for sf in fits:
            sf.t_hit -= 0.01
            sf.t_end -= 0.01
    fast = [(s, t) for s, t in zip(rally.segments[1:], t_new[1:]) if s.speed0 > 10]
    assert len(fast) >= 2
    for s, t in fast:
        assert abs(t - s.t_hit) < 0.003


def test_audio_hit_times_subtract_the_sound_delay(rally_fit, cam):
    """打球音の時刻から、打点からカメラまでの音の遅れを引いた時刻にする。交わる時刻と近ければそのまま。"""
    rally, _, fits = rally_fit
    segs = rally.segments
    t_audio = [s.t_hit + float(np.linalg.norm(cam.center - s.p0)) / 343.0 for s in segs]
    # 交わる時刻が 10 ms ずれていたら、打球音の時刻に直す（打点の位置の誤差 0.8 m は 2.3 ms）
    out = audio_hit_times(fits, t_audio, cam, PHYS, [s.t_hit + 0.01 for s in segs])
    for s, t in zip(segs, out):
        assert abs(t - s.t_hit) < 0.003
    # 交わる時刻と 4 ms 以内で合っていれば、交わる時刻のまま。打球音のないショットもそのまま
    near = [s.t_hit + 0.001 for s in segs]
    assert audio_hit_times(fits, t_audio, cam, PHYS, near) == near
    far = [s.t_hit + 0.01 for s in segs]
    assert audio_hit_times(fits, [None] * len(segs), cam, PHYS, far) == far


def test_fit_shots_with_audio_times_recover_a_late_hit_time(cam):
    """打球の時刻が 1 コマ遅れていても（実動画で打点の前後の検出が抜けた所）、打球音があれば初速が合う。"""
    rally = make_rally(np.random.default_rng(5), PHYS, n_shots=3)
    det = observe_rally(rally, cam, 30.0, PHYS, np.random.default_rng(3))
    segs = rally.segments
    t_late = [s.t_hit + 0.02 for s in segs]
    t_audio = [s.t_hit + float(np.linalg.norm(cam.center - s.p0)) / 343.0 for s in segs]
    opts = FitOptions(refine_plane=True, ci_method="none")
    fits = fit_shots(det, t_late, cam, PHYS, [s.side for s in segs], opts, refine_times=False, t_audio=t_audio)
    for s, sf in zip(segs, fits):
        assert abs(sf.t_hit - s.t_hit) < 0.003
        assert abs(sf.speed_kmh / (s.speed0 * MPS_TO_KMH) - 1) < 0.1


def test_split_observations_and_too_few_frames(cam):
    """打球の時刻で観測を分け、打球の前後 margin は使わない。コマが足りない飛行は当てはめない。"""
    t = np.arange(0.0, 1.0, 0.01)
    det = pd.DataFrame({"timestamp": t, "x_px": 100 + 50 * t, "y_px": 500 - 10 * t, "visible": 1})
    det.loc[5, "visible"] = 0
    obs = split_observations(det, cam, [0.0, 0.5], margin_s=0.02)
    assert obs[0].t.min() > 0.02 and obs[0].t.max() <= 0.48 + 1e-9
    assert obs[1].t.min() > 0.52 and len(obs[1].t) == 47
    assert len(obs[0].t) == 45  # 0.03〜0.48 の 46 コマから見えない 1 コマを除く
    fits = fit_shots(det, [0.0, 0.95], cam, PHYS, min_obs=5, n_rounds=0, opts=FitOptions(ci_method="none"))
    assert fits[1].fit is None and "コマ" in fits[1].message
    assert shot_fits_to_shots(fits[1:], PHYS)[0].speed_kmh is None


def test_parallel_fits_give_the_same_result(cam):
    """ショットごとの当てはめを別々のプロセスで並べても、1 つずつ計算したときと同じ結果になる。"""
    rally = make_rally(np.random.default_rng(4), PHYS, n_shots=3)
    det = observe_rally(rally, cam, 30.0, PHYS, np.random.default_rng(0))
    t_hits = [s.t_hit for s in rally.segments]
    sides = [s.side for s in rally.segments]
    opts = FitOptions(refine_plane=True)
    one = fit_shots(det, t_hits, cam, PHYS, sides, opts)
    two = fit_shots(det, t_hits, cam, PHYS, sides, opts, workers=2)
    assert [f.speed_kmh for f in two] == [f.speed_kmh for f in one]
    assert all(f.fit is not None and f.fit.speed0_ci is not None for f in two)
    # 打点をつないだ後もずれが大きいショットは、信頼区間つきの当てはめ直しをしない
    loose = fit_shots(det, t_hits, cam, PHYS, sides, opts, max_rms_px=0.0)
    assert all(f.fit is not None and f.fit.speed0_ci is None for f in loose)


def test_joint_hit_points_use_hitter_feet(rally_fit):
    """足元から離れた打点は捨て、どちらも捨てたら足元の真上にする。足元に近い打点はそのまま（ROADMAP D3c）。"""
    rally, _, fits = rally_fit
    plain = joint_hit_points(fits, PHYS)
    near = joint_hit_points(fits, PHYS, feet=[None if p is None else p[:2] + 0.3 for p in plain[:-1]])
    for a, b in zip(plain[:-1], near[:-1]):
        assert a is not None and np.allclose(a, b)
    far = [s.p0[:2] + np.array([0.0, 6.0]) for s in rally.segments]
    moved = joint_hit_points(fits, PHYS, feet=far)
    for p, f in zip(moved[:-1], far):
        assert np.allclose(p[:2], f) and 0.3 <= p[2] <= 3.5


def test_hitter_feet_keep_speeds_and_pull_hit_points(cam):
    """打つ人の足元を渡しても初速は正解に近いまま。打点は足元の近くに出る。"""
    rally = make_rally(np.random.default_rng(4), PHYS, n_shots=5)
    det = observe_rally(rally, cam, 30.0, PHYS, np.random.default_rng(0))
    segs = rally.segments
    feet = [s.p0[:2] + np.random.default_rng(k).normal(0.0, 0.3, 2) for k, s in enumerate(segs)]
    fits = fit_shots(det, [s.t_hit for s in segs], cam, PHYS, [s.side for s in segs],
                     FitOptions(refine_plane=True, ci_method="none"), hitter_xy=feet)
    errs = [abs(sf.speed_kmh / (s.speed0 * MPS_TO_KMH) - 1) for s, sf in zip(segs, fits) if sf.fit is not None]
    assert len(errs) >= len(segs) - 1 and np.mean(errs) < 0.05
    gaps = [np.hypot(*(sf.fit.p0[:2] - f)) for sf, f in zip(fits, feet) if sf.fit is not None]
    assert np.median(gaps) < 1.0


def test_worker_pool_is_reused_with_single_thread_blas(monkeypatch):
    """``worker_pool`` の中では同じプロセスの組を使い回し、子のプロセスの行列計算は 1 スレッドにする（ROADMAP F3）。"""
    import os

    from smash_speed import shot_fit

    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    args = [("OPENBLAS_NUM_THREADS",)] * 3
    with shot_fit.worker_pool(2):
        ex = shot_fit._shared_pool[0]
        assert shot_fit._parallel_map(os.getenv, args, 2) == ["1"] * 3
        assert shot_fit._parallel_map(os.getenv, args, 2) == ["1"] * 3
        assert shot_fit._shared_pool[0] is ex
    assert not shot_fit._shared_pool
    # 親のプロセスの設定は元に戻す
    assert os.getenv("OPENBLAS_NUM_THREADS") is None
    # 1 つずつの計算は親のプロセスで行う
    assert shot_fit._parallel_map(os.getenv, args, 1) == [None] * 3


def test_broken_pool_falls_back_one_by_one(monkeypatch):
    """``worker_pool`` の組が壊れたら、新しい組を作らず 1 つずつ計算する（ラリーを並べて動かしているときのため。F3b）。"""
    import os
    from concurrent.futures.process import BrokenProcessPool

    from smash_speed import shot_fit

    class Broken:
        def map(self, *a):
            raise BrokenProcessPool("test")

    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    monkeypatch.setattr(shot_fit, "_shared_pool", [Broken(), 2])
    assert shot_fit.pool_active()
    args = [("OPENBLAS_NUM_THREADS",)] * 3
    assert shot_fit._parallel_map(os.getenv, args, 2) == [None] * 3   # 親のプロセスで計算した
    assert not shot_fit.pool_active() and shot_fit._shared_pool[0] is None
    assert shot_fit._parallel_map(os.getenv, args, 2) == [None] * 3


def test_parallel_grid_fits_match_initial_fit(cam):
    """格子の平面ごとに並べた当てはめ（``fit_shots`` の 1 段目）は、``initial_fit`` を 1 本ずつ動かしたときと同じ（F3b）。"""
    from smash_speed.shot_fit import initial_fit

    rally = make_rally(np.random.default_rng(4), PHYS, n_shots=3)
    det = observe_rally(rally, cam, 30.0, PHYS, np.random.default_rng(0))
    segs = rally.segments
    t_hits = [s.t_hit for s in segs]
    feet = [None, segs[1].p0[:2] + 0.2, None]
    opts = FitOptions(refine_plane=True, ci_method="none")
    fits = fit_shots(det, t_hits, cam, PHYS, [s.side for s in segs], opts, n_rounds=0, refine_times=False,
                     hitter_xy=feet, workers=2)
    for k, s in enumerate(segs):
        # ``fit_shots`` は打点の位置の目安を 2 乗のまま効かせる（``quadratic_priors``）ので、同じ設定で比べる
        one = initial_fit(fits[k].obs, fits[k].t_hit, s.side, PHYS,
                          FitOptions(refine_plane=True, ci_method="none", quadratic_priors=True), feet[k])
        assert np.array_equal(fits[k].fit.theta, one.theta)
        assert np.array_equal(fits[k].fit.plane.direction, one.plane.direction)


def test_foot_hit_times_move_a_late_hit_back_to_the_hitter(rally_fit):
    """打球音のないショットの遅れた打球の時刻を、軌道が打つ人の足元にいちばん近づく時刻へ戻す。"""
    from smash_speed.shot_fit import foot_hit_times

    rally, _, fits = rally_fit
    fast = [i for i, s in enumerate(rally.segments) if s.speed0 > 25 and fits[i].fit is not None]
    assert fast
    feet = [s.p0[:2] for s in rally.segments]
    late = [sf.t_hit + (0.09 if i in fast else 0.0) for i, sf in enumerate(fits)]     # 25 m/s でも 2 m 先
    none = [None] * len(fits)
    new = foot_hit_times(fits, feet, none, PHYS, late)
    for i, s in enumerate(rally.segments):
        if i in fast:
            assert abs(new[i] - s.t_hit) < 0.02
            assert abs(late[i] - s.t_hit) > 0.08
        else:
            assert new[i] == late[i]
    # 打球音のあるショット・足元のないショット・打点が足元に近いショットは動かさない
    assert foot_hit_times(fits, feet, [0.0] * len(fits), PHYS, late) == late
    assert foot_hit_times(fits, none, none, PHYS, late) == late
    on_time = [sf.t_hit for sf in fits]
    assert foot_hit_times(fits, feet, none, PHYS, on_time) == on_time

