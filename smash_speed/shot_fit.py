"""ショットごとの軌道と初速（スマッシュ以外の山なり・遅いショットも）。

``hits.find_hits`` で見つけた打球の時刻で、動画全体のシャトルの検出結果を 1 本ずつの飛行に分け、
それぞれに物理モデル（``physics.fit_trajectory`` の plane モード）を当てはめる。

カメラ 1 台では奥行きが決まらないので、飛行平面（鉛直な面）を次の順に決める。

1. 平面の当たりをつける（``initial_fit``）：コートの x 軸の向き（打った側から相手側へ）に近い平面を、
   ネットを横切る位置 y と向き（yaw）の粗い格子で試し、画像上のずれがいちばん小さいものから、平面の向き・位置も
   大きく動かせるようにして当てはめ直す。重力と空気抵抗の大きさは決まっているので、長く飛ぶショット
   （クリア・ロブ）は画像上の曲がり方から奥行きが決まる。短い・遅いショット（ネット・ショートサーブ）は決まりにくい。
2. 打点をつなぐ（``joint_hit_points``）：打点 i は「ショット i − 1 の終わり」と「ショット i の始まり」の
   同じ点なので、両方の当てはめから打点の位置を出し、決まりやすい方（長く飛んだ方）を重く平均する。
3. 当てはめ直す：飛行平面を「打点 → 次の打点（最後のショットは着地点）」にし、打点の位置を事前分布として、
   平面の小さな補正（``physics.plane_sigma_deg``・``plane_sigma_m``）つきで当てはめ直す。2〜3 を繰り返す。

初速は打点の時刻の速さなので、速いショットほど時刻のずれが効く（スマッシュ 100 m/s では 1 ms で約 2 m/s 減る）。
打球の時刻（``Hit.t``）は、打球音なら届くまでの遅れ、軌道の向きの変化ならコマの間隔の分ずれるので、
2〜3 の前に、前のショットと次のショットの軌道が画像上で交わる時刻に直す（``refine_hit_times``）。

``python -m smash_speed.shot_fit --bench`` で、合成ラリー（``synth_rally``）での種類ごとの初速の誤差を出す。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from smash_speed.camera import Camera
from smash_speed.geometry import FlightPlane
from smash_speed.physics import (MPS_TO_KMH, FitOptions, FitResult, Observations, PhysicsParams, fit_trajectory,
                                 simulate, simulate_until_ground)
from smash_speed.shots import Shot, side_of

# 平面の当たりをつける格子（ネットを横切る位置 [m] と、コートの x 軸からの向き [deg]）
GRID_Y_M = (-2.0, 0.0, 2.0)
GRID_YAW_DEG = (-20.0, 0.0, 20.0)


@dataclass
class ShotFit:
    """1 本のショットの当てはめ結果。

    Attributes:
        t_hit: 打った時刻（当てはめの基準の時刻）[s]
        side: 打った側 h | o | ""
        t_end: 飛行の終わり（次の打球の時刻。最後のショットは最後の検出の時刻）[s]
        n_obs: 当てはめに使ったコマの数
        fit: 当てはめの結果（コマが足りない・計算できなければ None）
        hit_point: つないだ打点の位置 [m]（なければ None）
        message: 当てはめられなかった理由など
        obs: 当てはめに使った観測
    """

    t_hit: float
    side: str
    t_end: float
    n_obs: int = 0
    fit: FitResult | None = None
    hit_point: np.ndarray | None = None
    message: str = ""
    obs: Observations | None = field(default=None, repr=False)

    @property
    def speed_kmh(self) -> float | None:
        return None if self.fit is None else self.fit.speed0 * MPS_TO_KMH

    @property
    def duration(self) -> float:
        return self.t_end - self.t_hit


def _phys_fast(phys: PhysicsParams) -> PhysicsParams:
    """格子で当たりをつけるときの積分は粗くてよい（最後は ``phys`` で当てはめ直す）。"""
    return PhysicsParams(phys.g, phys.terminal_velocity, "RK45", 1e-7, 1e-9)


def split_observations(det: pd.DataFrame, camera: Camera, t_hits: list[float], margin_s: float = 0.01,
                       t_last: float | None = None) -> list[Observations]:
    """検出結果を、打球から次の打球までの飛行ごとの観測に分ける。

    打球の時刻の前後 ``margin_s`` [s] は、どちらの飛行か紛らわしいので使わない。
    最後の飛行は ``t_last``（省略時は最後の検出）まで。
    """
    vis = det[(det.visible > 0) & det.x_px.notna() & det.y_px.notna()]
    t, uv = vis.timestamp.to_numpy(float), vis[["x_px", "y_px"]].to_numpy(float)
    out = []
    for i, t0 in enumerate(t_hits):
        t1 = t_hits[i + 1] - margin_s if i + 1 < len(t_hits) else (t_last if t_last is not None else np.inf)
        m = (t > t0 + margin_s) & (t <= t1)
        out.append(Observations(camera, t[m], uv[m]))
    return out


def _x_signs(side: str) -> list[float]:
    """飛ぶ向きのコートの x の符号の候補（打った側が分からなければ両方）。"""
    if side in ("h", "o"):
        return [1.0 if side == "h" else -1.0]
    return [1.0, -1.0]


def initial_fit(obs: Observations, t_hit: float, side: str, phys: PhysicsParams, opts: FitOptions
                ) -> FitResult | None:
    """平面の格子から当たりをつけ、平面の向き・位置を大きく動かせるようにして当てはめる（``ShotFit`` の 1）。"""
    fast = _phys_fast(phys)
    coarse = FitOptions(loss=opts.loss, loss_scale_px=opts.loss_scale_px, ci_method="none", diff_step=1e-5)
    best: FitResult | None = None
    for sx in _x_signs(side):
        for y in GRID_Y_M:
            for yaw in np.deg2rad(GRID_YAW_DEG):
                plane = FlightPlane(np.array([0.0, y, 0.0]), sx * np.array([np.cos(yaw), np.sin(yaw), 0.0]))
                try:
                    f = fit_trajectory(obs, t_hit, "plane", plane, fast, coarse)
                except (ValueError, RuntimeError):
                    continue
                if np.isfinite(f.rms_px) and (best is None or f.rms_px < best.rms_px):
                    best = f
    if best is None:
        return None
    loose = FitOptions(loss=opts.loss, loss_scale_px=opts.loss_scale_px, ci_method="none", refine_plane=True,
                       plane_sigma_deg=30.0, plane_sigma_m=3.0)
    try:
        return fit_trajectory(obs, t_hit, "plane", best.plane, phys, loose,
                              theta0=np.r_[best.theta[:4], 0.0, 0.0])
    except (ValueError, RuntimeError):
        return best


def _state_at(f: FitResult, t: float, phys: PhysicsParams) -> tuple[np.ndarray, np.ndarray] | None:
    """当てはめた軌道の時刻 ``t`` [s] の位置 [m]・速度 [m/s]（計算できなければ None）。"""
    p, v = simulate(f.p0, f.v0, f.t_ref, np.array([t]), f.phys(phys))
    return (p[0], v[0]) if np.isfinite(p).all() and np.isfinite(v).all() else None


def _end_point(sf: ShotFit, phys: PhysicsParams) -> np.ndarray | None:
    """飛行の終わり（次の打点の時刻の位置。最後のショットは床に着いた所）[m]。"""
    f = sf.fit
    if f is None:
        return None
    st = _state_at(f, sf.t_end, phys)
    if st is None or st[0][2] < 0:
        tr = simulate_until_ground(f.p0, f.v0, f.t_ref, f.phys(phys), t_max=max(sf.duration + 1.0, 1.0), n_dense=2)
        return tr["pos"][-1]
    return st[0]


def joint_hit_points(fits: list[ShotFit], phys: PhysicsParams, max_jump_m: float = 3.0) -> list[np.ndarray | None]:
    """打点 i を、ショット i の始まりとショット i − 1 の終わりから出す（``ShotFit`` の 2）。

    重みは飛行の時間の 2 乗（画像上の曲がり方 = 奥行きの決まりやすさは時間の 2 乗に比例する）。
    2 つが ``max_jump_m`` 以上離れていたら（どちらかが外れている）、長く飛んだ方だけを使う。
    最後の要素は、最後のショットの着地点（床の上）。
    """
    starts = []
    for sf in fits:
        st = None if sf.fit is None else _state_at(sf.fit, sf.t_hit, phys)
        starts.append(None if st is None else st[0])
    ends = [_end_point(sf, phys) for sf in fits]
    out: list[np.ndarray | None] = []
    for i, sf in enumerate(fits):
        cands = []
        if starts[i] is not None:
            cands.append((starts[i], sf.duration ** 2))
        if i > 0 and ends[i - 1] is not None and fits[i - 1].t_end >= sf.t_hit - 0.05:
            cands.append((ends[i - 1], fits[i - 1].duration ** 2))
        if not cands:
            out.append(None)
            continue
        if len(cands) == 2 and np.linalg.norm(cands[0][0] - cands[1][0]) > max_jump_m:
            cands = [max(cands, key=lambda c: c[1])]
        w = np.array([c[1] for c in cands])
        out.append(np.sum([c[0] * wi for c, wi in zip(cands, w)], axis=0) / w.sum())
    last = ends[-1] if fits else None
    out.append(None if last is None else np.r_[last[:2], 0.0])
    return out


def crossing_time(prev: FitResult, nxt: FitResult, camera: Camera, phys: PhysicsParams, lo: float, hi: float,
                  step_s: float = 2.5e-4) -> tuple[float, float] | None:
    """飛んできた軌道（``prev`` を先へ）と打ち返した軌道（``nxt`` を前へ）が画像上でいちばん近づく時刻 [s] と、
    そのときの距離 [px]。"""
    if hi <= lo:
        return None
    t = np.arange(lo, hi + step_s / 2, step_s)
    a = camera.project(simulate(prev.p0, prev.v0, prev.t_ref, t, prev.phys(phys))[0])
    b = camera.project(simulate(nxt.p0, nxt.v0, nxt.t_ref, t, nxt.phys(phys))[0])
    gap = np.linalg.norm(a - b, axis=1)
    if not np.isfinite(gap).any():
        return None
    k = int(np.nanargmin(gap))
    tk = float(t[k])
    if 0 < k < len(t) - 1 and np.isfinite(gap[k - 1:k + 2]).all():  # 放物線で刻みより細かく
        g0, g1, g2 = gap[k - 1:k + 2]
        den = g0 - 2 * g1 + g2
        if den > 0:
            tk += float(np.clip(0.5 * (g0 - g2) / den, -1, 1)) * step_s
    return tk, float(gap[k])


def refine_hit_times(fits: list[ShotFit], camera: Camera, phys: PhysicsParams, max_shift_s: float = 0.03,
                     max_gap_px: float = 6.0) -> list[float]:
    """打点の時刻を、前のショットと次のショットの軌道が画像上で交わる時刻に直す。

    初速は打点の時刻の速さなので、速いショットは時刻の数 ms のずれが大きく効く。
    交わる所が ``max_gap_px``（と当てはめのずれ）より離れていたら直さない。
    """
    out = [sf.t_hit for sf in fits]
    for i in range(1, len(fits)):
        prev, sf = fits[i - 1], fits[i]
        if prev.fit is None or sf.fit is None or prev.t_end < sf.t_hit - 1e-9:
            continue
        lo, hi = sf.t_hit - max_shift_s, sf.t_hit + max_shift_s
        if prev.obs is not None and len(prev.obs.t):
            lo = max(lo, float(prev.obs.t[-1]) - 0.02)
        if sf.obs is not None and len(sf.obs.t):
            hi = min(hi, float(sf.obs.t[0]) + 0.02)
        c = crossing_time(prev.fit, sf.fit, camera, phys, lo, hi)
        if c is not None and c[1] <= max(max_gap_px, 3 * (prev.fit.rms_px + sf.fit.rms_px)):
            out[i] = c[0]
    return out


def audio_hit_times(fits: list[ShotFit], t_audio: list[float | None], camera: Camera, phys: PhysicsParams,
                    times: list[float], speed_of_sound_mps: float = 343.0, tol_s: float = 0.004) -> list[float]:
    """打球音のあるショットの打点の時刻を「打球音の時刻 − 打点からカメラ（マイク）までの距離 ÷ 音速」にする
    （スマッシュの解析 ``pipeline`` と同じ）。

    ``times``（前後の軌道が交わる時刻）がこれと ``tol_s`` 以内なら ``times`` のまま（当てはめた軌道どうしで
    矛盾がなく、合成ラリーではこの方が初速が合う）。それより離れていたら、交わる時刻の方が外れている
    （実動画の 30 fps では、打球の瞬間はラケット・体に隠れ、ぶれて検出が抜けるので、交わりが 5〜11 ms ずれる。
    打点の位置の 1 m の誤差は音の時刻で 3 ms）。

    打点の位置は、つないだ打点（なければ当てはめた軌道の始まり）。コートから遠すぎる打点（当てはめの失敗）なら
    ``times`` のまま。
    """
    out = list(times)
    for i, (sf, ta) in enumerate(zip(fits, t_audio)):
        if ta is None or sf.fit is None:
            continue
        p = sf.hit_point
        if p is None:
            st = _state_at(sf.fit, sf.t_hit, phys)
            p = None if st is None else st[0]
        if p is None or abs(p[0]) > 9.0 or abs(p[1]) > 6.0:
            continue
        t = float(ta) - float(np.linalg.norm(camera.center - p)) / speed_of_sound_mps
        if abs(times[i] - t) > tol_s:
            out[i] = t
    return out


def _parallel_map(fn: Callable[..., Any], args: list[tuple[Any, ...]], workers: int) -> list[Any]:
    """``fn(*a)`` を ``args`` の順に計算する。``workers`` が 2 以上なら別々のプロセスで並べて計算する（結果は同じ）。

    プロセスを起動できなかったとき（Windows で、``if __name__ == "__main__":`` のないスクリプトから呼んだ、など）は
    1 つずつ計算する。
    """
    workers = min(workers, len(args))
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        from concurrent.futures.process import BrokenProcessPool

        try:
            with ProcessPoolExecutor(workers) as ex:
                return list(ex.map(fn, *zip(*args)))
        except (BrokenProcessPool, OSError):
            pass
    return [fn(*a) for a in args]


def _try_fit(obs: Observations, t_hit: float, plane: FlightPlane, phys: PhysicsParams, opts: FitOptions,
             theta0: np.ndarray | None) -> FitResult | None:
    """plane モードで当てはめる（計算できなければ None）。"""
    try:
        return fit_trajectory(obs, t_hit, "plane", plane, phys, opts, theta0=theta0)
    except (ValueError, RuntimeError):
        return None


def fit_shots(det: pd.DataFrame, t_hits: list[float], camera: Camera, phys: PhysicsParams,
              sides: list[str] | None = None, opts: FitOptions | None = None, n_rounds: int = 2,
              min_obs: int = 5, margin_s: float = 0.01, hit_sigma_m: tuple[float, float] = (0.5, 0.5),
              t_last: float | None = None, refine_times: bool = True, workers: int = 1,
              max_rms_px: float = np.inf, t_audio: list[float | None] | None = None,
              speed_of_sound_mps: float = 343.0, max_kmh: float = np.inf,
              bad_max_nfev_per_param: int = 30) -> list[ShotFit]:
    """打球の時刻 ``t_hits`` [s] で分けた飛行ごとに、軌道と初速を当てはめる。

    Args:
        det: 動画全体のシャトルの検出結果（列 ``timestamp, x_px, y_px, visible``）
        t_hits: 打球の時刻（``Hit.t``）。時刻の順
        sides: 打った側（``Hit.side``）。分からなければ両方の向きを試す
        opts: 当てはめの設定（``FitOptions.from_config``。信頼区間の出し方・平面の補正の大きさ）
        n_rounds: 打点をつないで当てはめ直す回数
        min_obs: これより少ないコマの飛行は当てはめない
        hit_sigma_m: つないだ打点の位置の不確かさ（水平, 高さ）[m]
        t_last: 最後の飛行の終わり [s]（省略時は最後の検出）
        refine_times: 打点の時刻を、前後の軌道が交わる時刻に直す（``refine_hit_times``）
        max_rms_px: 打点をつないだ後も画像の上のずれ（RMS）がこれを超えるショットは、信頼区間つきの当てはめ直しを
            しない（実動画で、別の物の検出が混ざった飛行。失敗として捨てるので、時間をかけない）[px]
        workers: ショットごとの当てはめ（それぞれの段階の中では互いに関係しない）を、この数のプロセスで並べて動かす
        t_audio: ショットごとの打球音の時刻（``Hit.t_audio``。音が届くまでの遅れを引く前。なければ None）。
            あるショットは、打点の時刻を前後の軌道が交わる時刻ではなく、打点からの音の遅れを引いた時刻にする
            （``audio_hit_times``）
        speed_of_sound_mps: 音速 [m/s]
        max_kmh: これより速い初速は、当てはめに失敗したとみなす（``bad_max_nfev_per_param``）[km/h]
        bad_max_nfev_per_param: ずれが ``max_rms_px`` を超えた・初速が ``max_kmh`` を超えたショットの当てはめ直しの、
            最適化の評価の回数の上限
            （パラメータ 1 個あたり。ふつうは 200）。打点をつなぐと救われるショットもあるので飛ばさないが、
            救われないショットは積分が重く、実動画で 1 本の当てはめ直しに 20〜50 秒かかっていた

    Returns:
        ``t_hits`` と同じ順の ``ShotFit``（``t_hit`` は直した時刻）
    """
    opts = opts or FitOptions(refine_plane=True)
    t_hits = [float(t) for t in t_hits]
    sides = list(sides) if sides is not None else [""] * len(t_hits)
    vis_t = det.timestamp[(det.visible > 0) & det.x_px.notna()].to_numpy(float)
    if t_last is None:
        t_last = float(vis_t.max()) if len(vis_t) else (t_hits[-1] if t_hits else 0.0)
    fits = [ShotFit(t, sides[i], t) for i, t in enumerate(t_hits)]

    def split(times: list[float]) -> None:
        for i, o in enumerate(split_observations(det, camera, times, margin_s, t_last)):
            sf = fits[i]
            sf.t_hit, sf.obs, sf.n_obs = times[i], o, len(o.t)
            sf.t_end = times[i + 1] if i + 1 < len(times) else max(t_last, times[i])

    split(t_hits)
    todo = [i for i, sf in enumerate(fits) if sf.n_obs >= min_obs]
    for sf in fits:
        if sf.n_obs < min_obs:
            sf.message = f"打った後のシャトルが {sf.n_obs} コマしか見えていません"
    first = _parallel_map(initial_fit, [(fits[i].obs, fits[i].t_hit, sides[i], phys, opts) for i in todo], workers)
    for i, f in zip(todo, first):
        fits[i].fit = f
        if f is None:
            fits[i].message = "軌道を当てはめられませんでした"

    t_audio = list(t_audio) if t_audio is not None else [None] * len(t_hits)
    for _ in range(n_rounds):
        times = refine_hit_times(fits, camera, phys) if refine_times else [sf.t_hit for sf in fits]
        if any(t is not None for t in t_audio):
            times = audio_hit_times(fits, t_audio, camera, phys, times, speed_of_sound_mps)
        if refine_times or any(t is not None for t in t_audio):
            split(times)
        pts = joint_hit_points(fits, phys)
        jobs, args = [], []
        for i, sf in enumerate(fits):
            a, b = pts[i], pts[i + 1]
            if (sf.fit is None or a is None or b is None or np.linalg.norm((b - a)[:2]) < 0.3
                    or sf.n_obs < min_obs):
                continue
            fo = FitOptions(**{**opts.__dict__, "p0_prior_m": a, "p0_prior_sigma_m": hit_sigma_m,
                               "refine_plane": opts.refine_plane and sf.n_obs > 5,
                               "init_state": _state_at(sf.fit, sf.t_hit, phys), "ci_method": "none",
                               **({"max_nfev_per_param": bad_max_nfev_per_param}
                                  if not (sf.fit.rms_px <= max_rms_px and sf.fit.speed0 * MPS_TO_KMH <= max_kmh)
                                  else {})})
            jobs.append((i, a))
            args.append((sf.obs, sf.t_hit, FlightPlane.from_points(a, b), phys, fo, None))
        for (i, a), f in zip(jobs, _parallel_map(_try_fit, args, workers)):
            if f is not None and f.success and np.isfinite(f.speed0):
                fits[i].fit, fits[i].hit_point = f, a

    # 最後に、平面を決めたまま信頼区間つきで当てはめ直す
    jobs, args = [], []
    for i, sf in enumerate(fits):
        if (sf.fit is None or sf.fit.plane is None or opts.ci_method == "none"
                or abs(sf.fit.t_ref - sf.t_hit) > 1e-9 or sf.n_obs < min_obs or not sf.fit.rms_px <= max_rms_px):
            continue
        fo = FitOptions(**{**opts.__dict__, "p0_prior_m": sf.hit_point, "p0_prior_sigma_m": hit_sigma_m,
                           "refine_plane": False, "init_state": None})
        jobs.append(i)
        args.append((sf.obs, sf.t_hit, sf.fit.plane, phys, fo, sf.fit.theta[:4] if sf.fit.mode == "plane" else None))
    for i, f in zip(jobs, _parallel_map(_try_fit, args, workers)):
        if f is not None:
            fits[i].fit = f
    return fits


def shot_fits_to_shots(fits: list[ShotFit], phys: PhysicsParams, source: str = "shot_fit") -> list[Shot]:
    """当てはめの結果を ``shots.csv`` の形（``Shot``）にする。種類はまだ分からない（unknown）。"""
    out = []
    for sf in fits:
        f = sf.fit
        if f is None:
            out.append(Shot(t_hit_s=sf.t_hit, side=sf.side, source=source, note=sf.message))
            continue
        kmh = f.speed0 * MPS_TO_KMH
        lo, hi = (None, None) if f.speed0_ci is None else (f.speed0_ci[0] * MPS_TO_KMH, f.speed0_ci[1] * MPS_TO_KMH)
        tr = simulate_until_ground(f.p0, f.v0, f.t_ref, f.phys(phys), n_dense=2)
        t_land = float(tr["t_land"]) if np.isfinite(tr["t_land"]) else None
        conf = None if lo is None else float(np.clip(1.0 - (hi - lo) / kmh, 0.0, 1.0))
        out.append(Shot(
            t_hit_s=sf.t_hit, side=sf.side or side_of(float(f.p0[0])), speed_kmh=kmh,
            speed_ci_low_kmh=lo, speed_ci_high_kmh=hi,
            hit_x_m=float(f.p0[0]), hit_y_m=float(f.p0[1]), hit_z_m=float(f.p0[2]),
            land_x_m=float(tr["pos"][-1, 0]) if t_land else None, land_y_m=float(tr["pos"][-1, 1]) if t_land else None,
            t_land_s=t_land, v0_x_mps=float(f.v0[0]), v0_y_mps=float(f.v0[1]), v0_z_mps=float(f.v0[2]),
            terminal_velocity_mps=f.terminal_velocity, confidence=conf, source=source,
            note=f"{sf.n_obs} コマ・ずれ {f.rms_px:.1f} px"))
    return out


# =============================================================================
# 合成ラリーでの測り方
# =============================================================================
def synthetic_benchmark(n_rallies: int = 10, fps: float = 120.0, seed: int = 0, hits: str = "true",
                        noise_px: float = 1.5, dropout: float = 0.1, camera: Camera | None = None,
                        cfg: dict[str, Any] | None = None, verbose: bool = False,
                        keep_shots: bool = False) -> dict[str, Any]:
    """合成ラリーで、ショットの種類ごとの初速の誤差を測る。

    Args:
        hits: 打球の時刻と打った側の決め方。"true" は正解、"track" は ``find_hits``（軌道だけ）、
            "audio" は ``find_hits``（打つたびの打球音の時刻を、カメラまでの音の遅れつきで渡す）。
            見つからなかった打球・正解のない打球は数えない（打球の見つけ方の精度は ``hits.synthetic_benchmark``）
        keep_shots: ``rows`` に、推定したショット（``Shot``。``index_in_rally`` は見つけた打球の順）の列 ``shot`` を足す

    Returns:
        ``rows``（1 ショット 1 行の DataFrame：種類・正解と推定の初速・誤差 [%]）と、
        ``by_type``（種類ごとの本数・当てはめられなかった本数・平均絶対誤差 [%]・誤差の中央値 [%]）、
        ``mape``（全体の平均絶対誤差 [%]）
    """
    from smash_speed.config import load_config
    from smash_speed.hits import find_hits
    from smash_speed.synth_rally import SHOT_PROFILES, make_rally, observe_rally, rally_camera

    cfg = cfg or load_config()
    phys = PhysicsParams.from_config(cfg)
    camera = camera or rally_camera(cfg)
    opts = FitOptions.from_config(cfg, ci_method="none")
    rally_rng = np.random.default_rng([seed, 1])
    rows = []
    for r in range(n_rallies):
        rally = make_rally(rally_rng, phys, n_shots=int(rally_rng.integers(4, 13)),
                           server_side="h" if r % 2 == 0 else "o")
        det = observe_rally(rally, camera, fps, phys, np.random.default_rng([seed, 2, r]), noise_px=noise_px,
                            dropout=dropout)
        segs = rally.segments
        if hits == "true":
            t_hits, sides, pairs = [s.t_hit for s in segs], [s.side for s in segs], list(range(len(segs)))
            t_audio = None
        else:
            onsets = None
            if hits == "audio":
                onsets = [(s.t_hit + float(np.linalg.norm(camera.center - s.p0)) / 343.0, 30.0) for s in segs]
            found = find_hits(det, fps, onsets, camera=camera)
            t_hits, sides, t_audio = [h.t for h in found], [h.side for h in found], [h.t_audio for h in found]
            pairs = [None] * len(found)  # 見つけた打球ごとの正解のショットの番号（50 ms 以内。なければ数えない）
            for i, h in enumerate(found):
                d = [abs(h.t - s.t_hit) for s in segs]
                j = int(np.argmin(d))
                if d[j] <= 0.05 and j not in pairs:
                    pairs[i] = j
        fits = fit_shots(det, t_hits, camera, phys, sides, opts, t_audio=t_audio)
        shots = shot_fits_to_shots(fits, phys) if keep_shots else [None] * len(fits)
        for i, (sf, j) in enumerate(zip(fits, pairs)):
            if j is None:
                continue
            s = segs[j]
            if shots[i] is not None:
                shots[i].rally_id, shots[i].index_in_rally = r, i + 1
            est = sf.speed_kmh
            true = s.speed0 * MPS_TO_KMH
            rows.append({"rally": r, "index": j, "profile": s.profile, "shot_type": SHOT_PROFILES[s.profile].shot_type,
                         "true_kmh": true, "est_kmh": est, "n_obs": sf.n_obs,
                         "dt_hit_ms": 1000 * (sf.t_hit - s.t_hit),
                         "err_pct": None if est is None else 100 * (est / true - 1),
                         **({"shot": shots[i]} if keep_shots else {})})
            if verbose and (est is None or abs(est / true - 1) > 0.1):
                print(f"ラリー {r} の {j + 1} 本目 {s.profile}：正解 {true:.1f} km/h、推定 "
                      f"{'-' if est is None else f'{est:.1f}'} km/h（{sf.n_obs} コマ）{sf.message}")
    df = pd.DataFrame(rows)
    by_type = {}
    for name, g in df.groupby("shot_type"):
        e = g.err_pct.dropna().abs()
        by_type[name] = {"n": int(len(g)), "n_failed": int(g.err_pct.isna().sum()),
                         "mape": float(e.mean()) if len(e) else None,
                         "median_abs_pct": float(e.median()) if len(e) else None,
                         "p90_abs_pct": float(e.quantile(0.9)) if len(e) else None}
    e = df.err_pct.dropna().abs() if len(df) else pd.Series(dtype=float)
    return {"n_rallies": n_rallies, "fps": fps, "hits": hits, "rows": df, "by_type": by_type,
            "mape": float(e.mean()) if len(e) else None, "n_failed": int(df.err_pct.isna().sum()) if len(df) else 0}


def main(argv: list[str] | None = None) -> None:
    import argparse
    import time

    from smash_speed.shots import SHOT_TYPE_LABELS

    ap = argparse.ArgumentParser(description="合成ラリーで、ショットの種類ごとの初速の誤差を測る")
    ap.add_argument("--bench", action="store_true", help="合成ラリーで測る")
    ap.add_argument("--rallies", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=float, nargs="+", default=[30.0, 120.0])
    ap.add_argument("--hits", nargs="+", default=["true", "audio", "track"], choices=["true", "audio", "track"])
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    for fps in a.fps:
        for h in a.hits:
            t0 = time.time()
            r = synthetic_benchmark(a.rallies, fps, a.seed, h, verbose=a.verbose)
            parts = [f"{SHOT_TYPE_LABELS[k]} {v['mape']:.1f}%（{v['n']} 本"
                     + (f"・失敗 {v['n_failed']}" if v["n_failed"] else "") + "）"
                     for k, v in r["by_type"].items() if v["mape"] is not None]
            print(f"{fps:5.0f} fps 打球の時刻 {h}：全体 {r['mape']:.2f}%、" + "、".join(parts)
                  + f"（{time.time() - t0:.0f} 秒）")


if __name__ == "__main__":
    import sys

    if (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    main()
