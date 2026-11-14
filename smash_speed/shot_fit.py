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

打つ人の足元（``hitter.hitter_at``。選手の追跡から）が分かっていれば、打点の手がかりにする（ROADMAP D3c）。
1 では平面を足元を通るものから探し、足元を打点の水平の位置の目安にする。2 では足元から ``max_hitter_gap_m`` より
離れた打点を捨て、どちらも捨てたら足元の真上を打点にする（実動画の 30 fps では、1 の当てはめが奥行きを
取り違えて打点がコートの外 30 m に出ることがある）。

``python -m smash_speed.shot_fit --bench`` で、合成ラリー（``synth_rally``）での種類ごとの初速の誤差を出す。
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, Iterator

import numpy as np
import pandas as pd

from smash_speed.camera import Camera
from smash_speed.fit_batch import FitJob, solve_together
from smash_speed.geometry import FlightPlane
from smash_speed.physics import (MAX_FIT_SPEED_MPS, MPS_TO_KMH, FitOptions, FitResult, Observations, PhysicsParams, fit_trajectory,
                                 simulate, simulate_until_ground)
from smash_speed.shots import Shot, side_of

# 平面の当たりをつける格子（ネットを横切る位置 [m] と、コートの x 軸からの向き [deg]）
GRID_Y_M = (-2.0, 0.0, 2.0)
GRID_YAW_DEG = (-20.0, 0.0, 20.0)
# 最初の当てはめが打った側から見てありえないときに試す、広い向きの格子 [deg]
WIDE_YAW_DEG = (-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0)


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


def plausible_start(p0: np.ndarray, v0: np.ndarray, side: str, max_yaw_deg: float = 75.0,
                    net_margin_m: float = 1.0) -> bool:
    """打点 ``p0`` [m] と打ち出しの速度 ``v0`` [m/s] が、打った側 ``side`` から見てありえるか。

    打点が自分の側（ネットから ``net_margin_m`` まではみ出してよい）・コートの近くにあり、水平の向きが相手側へ
    （コートの x 軸から ``max_yaw_deg`` 以内）。打った側が分からなければ、コートの近くにあるかだけを見る。
    角・後ろから撮ると飛ぶ向きが奥行きに近く、短いショットの当てはめが「ネットの反対側から逆向きに飛ぶ」軌道に
    落ちることがあるので、それを見分ける。
    """
    p0, v0 = np.asarray(p0, float), np.asarray(v0, float)
    if not (np.isfinite(p0).all() and np.isfinite(v0).all()):
        return False
    if abs(p0[0]) > 8.0 or abs(p0[1]) > 4.0 or not -0.2 < p0[2] < 4.5:
        return False
    if side not in ("h", "o"):
        return True
    sx = 1.0 if side == "h" else -1.0
    vh = float(np.hypot(v0[0], v0[1]))
    return sx * p0[0] < net_margin_m and vh > 0 and sx * v0[0] > vh * np.cos(np.deg2rad(max_yaw_deg))


def _plausible_fit(f: FitResult | None, side: str) -> bool:
    return f is not None and plausible_start(f.p0, f.v0, side)


def initial_fit(obs: Observations, t_hit: float, side: str, phys: PhysicsParams, opts: FitOptions,
                hitter_xy: np.ndarray | None = None, hitter_sigma_m: tuple[float, float] = (0.5, 2.0)
                ) -> FitResult | None:
    """平面の格子から当たりをつけ、平面の向き・位置を大きく動かせるようにして当てはめる（``ShotFit`` の 1）。

    ``hitter_xy``（打つ人の足元 [m]）があれば、格子の平面を足元を通るものにし、当てはめ直しで足元を
    打点の水平の位置の目安（不確かさ ``hitter_sigma_m``（水平, 高さ））にする。
    いちばん合う格子から動かした結果が、打った側から見てありえない（``plausible_start``）ときは、広い向きの格子から
    当てはめ直す（``wide_fit``）。
    """
    grid = [grid_fit(obs, t_hit, plane, phys, opts) for plane in grid_planes(side, hitter_xy)]
    first = loose_fit(obs, t_hit, best_grid_fit(grid), phys, opts, hitter_xy, hitter_sigma_m)
    if first is None or _plausible_fit(first, side):
        return first
    return wide_fit(obs, t_hit, side, first, phys, opts, hitter_xy)


def grid_planes(side: str, hitter_xy: np.ndarray | None = None,
                yaw_deg: tuple[float, ...] = GRID_YAW_DEG) -> list[FlightPlane]:
    """``initial_fit`` で試す平面の格子（試す順。足元があれば足元を通るものだけ）。"""
    origins = ([np.array([0.0, y, 0.0]) for y in GRID_Y_M] if hitter_xy is None
               else [np.r_[np.asarray(hitter_xy, float)[:2], 0.0]])
    return [FlightPlane(origin, sx * np.array([np.cos(yaw), np.sin(yaw), 0.0]))
            for sx in _x_signs(side) for origin in origins for yaw in np.deg2rad(yaw_deg)]


def _grid_job(obs: Observations, t_hit: float, plane: FlightPlane, phys: PhysicsParams, opts: FitOptions) -> FitJob:
    """``grid_fit`` の当てはめの引数。"""
    coarse = FitOptions(loss=opts.loss, loss_scale_px=opts.loss_scale_px, ci_method="none", diff_step=1e-5)
    return FitJob(obs, t_hit, plane, _phys_fast(phys), coarse)


def _run(job: FitJob) -> FitResult | None:
    """当てはめ 1 回（計算できなければ None）。"""
    try:
        return fit_trajectory(job.obs, job.t_ref, "plane", job.plane, job.phys, job.opts, theta0=job.theta0)
    except (ValueError, RuntimeError):
        return None


def grid_fit(obs: Observations, t_hit: float, plane: FlightPlane, phys: PhysicsParams, opts: FitOptions
             ) -> FitResult | None:
    """格子の平面 1 つで、粗い積分で当てはめる（計算できなければ None）。"""
    return _run(_grid_job(obs, t_hit, plane, phys, opts))


def best_grid_fit(grid: list[FitResult | None]) -> FitResult | None:
    """格子の当てはめのうち、画像の上のずれがいちばん小さいもの（同じなら先のもの）。"""
    best: FitResult | None = None
    for f in grid:
        if f is not None and np.isfinite(f.rms_px) and (best is None or f.rms_px < best.rms_px):
            best = f
    return best


def _loose_job(obs: Observations, t_hit: float, best: FitResult, phys: PhysicsParams, opts: FitOptions,
               hitter_xy: np.ndarray | None = None, hitter_sigma_m: tuple[float, float] = (0.5, 2.0)) -> FitJob:
    """``loose_fit`` の当てはめの引数。"""
    loose = FitOptions(loss=opts.loss, loss_scale_px=opts.loss_scale_px, ci_method="none", refine_plane=True,
                       plane_sigma_deg=30.0, plane_sigma_m=3.0, quadratic_priors=opts.quadratic_priors)
    if hitter_xy is not None:
        loose.p0_prior_m = np.r_[np.asarray(hitter_xy, float)[:2], np.clip(best.p0[2], 0.3, 3.5)]
        loose.p0_prior_sigma_m = hitter_sigma_m
    return FitJob(obs, t_hit, best.plane, phys, loose, np.r_[best.theta[:4], 0.0, 0.0])


def loose_fit(obs: Observations, t_hit: float, best: FitResult | None, phys: PhysicsParams, opts: FitOptions,
              hitter_xy: np.ndarray | None = None, hitter_sigma_m: tuple[float, float] = (0.5, 2.0)
              ) -> FitResult | None:
    """格子でいちばん合った ``best`` から、平面の向き・位置を大きく動かせるようにして当てはめ直す（``initial_fit`` の後半）。"""
    if best is None:
        return None
    f = _run(_loose_job(obs, t_hit, best, phys, opts, hitter_xy, hitter_sigma_m))
    return best if f is None else f


def _wide_candidates(grid: list[FitResult | None], side: str) -> list[FitResult]:
    """``wide_fit`` の広い格子の当てはめのうち、ありえるものを、ずれの小さい順に 3 つ。"""
    wide = [f for f in grid if f is not None and np.isfinite(f.rms_px) and _plausible_fit(f, side)]
    wide.sort(key=lambda f: f.rms_px)
    return wide[:3]


def _tight_job(obs: Observations, t_hit: float, g: FitResult, phys: PhysicsParams, opts: FitOptions) -> FitJob:
    """``wide_fit`` の、平面を小さく動かす当てはめの引数。"""
    tight = FitOptions(loss=opts.loss, loss_scale_px=opts.loss_scale_px, ci_method="none", refine_plane=True,
                       plane_sigma_deg=10.0, plane_sigma_m=1.0)
    return FitJob(obs, t_hit, g.plane, phys, tight, np.r_[g.theta[:4], 0.0, 0.0])


def _wide_choice(cands: list[FitResult], tight: list[FitResult | None], side: str, first: FitResult) -> FitResult:
    """``wide_fit`` の結果：小さく動かした当てはめ（計算できなければ格子のまま）のうち、ありえる中でいちばん合うもの。"""
    ok = [f for f in (g if t is None else t for g, t in zip(cands, tight)) if _plausible_fit(f, side)]
    return min(ok, key=lambda f: f.rms_px) if ok else first


def wide_fit(obs: Observations, t_hit: float, side: str, first: FitResult, phys: PhysicsParams, opts: FitOptions,
             hitter_xy: np.ndarray | None = None) -> FitResult:
    """``loose_fit`` の結果 ``first`` が打った側から見てありえないときの当てはめ直し（``initial_fit`` の最後）。

    ありえる向きの平面を広い格子（``WIDE_YAW_DEG``）で探し、ずれの小さいものから 3 つ、平面を小さく動かして当てはめ、
    ありえる中でいちばん合うものにする（大きく動かすと、ありえない向きに戻ってしまう）。どれもありえなければ ``first``。
    """
    cands = _wide_candidates([grid_fit(obs, t_hit, p, phys, opts) for p in grid_planes(side, hitter_xy, WIDE_YAW_DEG)],
                             side)
    return _wide_choice(cands, [_run(_tight_job(obs, t_hit, g, phys, opts)) for g in cands], side, first)


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


def hit_point_options(fits: list[ShotFit], phys: PhysicsParams, max_jump_m: float = 3.0,
                      feet: list[np.ndarray | None] | None = None, max_foot_m: float = 2.5,
                      foot_height_m: float = 2.0) -> list[list[np.ndarray]]:
    """打点 i の候補（ショット i の始まりとショット i − 1 の終わりから。最初の候補が ``joint_hit_points`` の値）。

    ``feet``（ショットごとの打つ人の足元 [m]。なければ None）があれば、足元から水平に ``max_foot_m`` より離れた
    ものを先に捨て、どちらも残らなければ足元の真上（高さは捨てたものの中央値を 0.3〜3.5 m にしたもの、
    なければ ``foot_height_m``）の 1 つにする。
    打った側から見てありえない所（相手側・コートから遠い）の候補は、ありえる候補があれば使わない。
    2 つが ``max_jump_m`` 以上離れていたら（どちらかが外れている）、長く飛んだ方を最初に、もう片方を 2 番目に置く
    （``fit_shots`` が両方で当てはめ直し、画像上のずれの小さい方を使う）。それ以外は、飛行の時間の 2 乗の重みで
    平均した 1 つ（画像上の曲がり方 = 奥行きの決まりやすさは時間の 2 乗に比例する）。
    最後の要素は、最後のショットの着地点（床の上）の 1 つ（なければ空）。
    """
    starts = []
    for sf in fits:
        st = None if sf.fit is None else _state_at(sf.fit, sf.t_hit, phys)
        starts.append(None if st is None else st[0])
    ends = [_end_point(sf, phys) for sf in fits]
    out: list[list[np.ndarray]] = []
    for i, sf in enumerate(fits):
        cands = []
        if starts[i] is not None:
            cands.append((starts[i], sf.duration ** 2))
        if i > 0 and ends[i - 1] is not None and fits[i - 1].t_end >= sf.t_hit - 0.05:
            cands.append((ends[i - 1], fits[i - 1].duration ** 2))
        foot = None if feet is None else feet[i]
        if foot is not None:
            near = [c for c in cands if np.hypot(*(c[0][:2] - foot[:2])) <= max_foot_m]
            if not near:
                z = float(np.clip(np.median([c[0][2] for c in cands]), 0.3, 3.5)) if cands else foot_height_m
                out.append([np.r_[foot[:2], z]])
                continue
            cands = near
        good = [c for c in cands if _plausible_point(c[0], sf.side)]
        if good:
            cands = good
        if not cands:
            out.append([])
            continue
        if len(cands) == 2 and np.linalg.norm(cands[0][0] - cands[1][0]) > max_jump_m:
            cands.sort(key=lambda c: -c[1])
            out.append([c[0] for c in cands])
            continue
        w = np.array([c[1] for c in cands])
        out.append([np.sum([c[0] * wi for c, wi in zip(cands, w)], axis=0) / w.sum()])
    last = ends[-1] if fits else None
    out.append([] if last is None else [np.r_[last[:2], 0.0]])
    return out


def _plausible_point(p: np.ndarray, side: str, net_margin_m: float = 1.0) -> bool:
    """打点の位置だけで ``plausible_start`` を見る（自分の側・コートの近く）。"""
    p = np.asarray(p, float)
    if not np.isfinite(p).all() or abs(p[0]) > 8.0 or abs(p[1]) > 4.0 or not -0.2 < p[2] < 4.5:
        return False
    return side not in ("h", "o") or (1.0 if side == "h" else -1.0) * p[0] < net_margin_m


def joint_hit_points(fits: list[ShotFit], phys: PhysicsParams, max_jump_m: float = 3.0,
                     feet: list[np.ndarray | None] | None = None, max_foot_m: float = 2.5,
                     foot_height_m: float = 2.0) -> list[np.ndarray | None]:
    """打点 i を、ショット i の始まりとショット i − 1 の終わりから出す（``ShotFit`` の 2）。

    ``hit_point_options`` の最初の候補（なければ None）。最後の要素は、最後のショットの着地点（床の上）。
    """
    return [o[0] if o else None
            for o in hit_point_options(fits, phys, max_jump_m, feet, max_foot_m, foot_height_m)]


def serve_hit_point(a: np.ndarray, f: FitResult, side: str, depth_m: tuple[float, float],
                    min_dir_x: float = 0.3) -> np.ndarray:
    """サーブの打点 ``a`` [m] を、飛行平面に沿って、深さの目安 ``depth_m``（ネットからの距離の平均, 標準偏差）と
    重みつきで平均した位置にする。

    当てはめの打点の、飛行平面に沿った位置の不確かさは、当てはめの共分散（plane モードの θ の 1 つ目）から出す。
    横から撮れば小さいので、ほとんど動かない。平面がコートの x 軸から大きく外れている・打った側が分からない・
    共分散がないときは ``a`` のまま。
    """
    a = np.asarray(a, float)
    if side not in ("h", "o") or f.plane is None or f.mode != "plane" or f.cov_theta is None:
        return a
    d = f.plane.direction
    var_fit = float(f.cov_theta[0, 0])
    if abs(d[0]) < min_dir_x or not np.isfinite(var_fit) or var_fit <= 0:
        return a
    x_prior = (-1.0 if side == "h" else 1.0) * depth_m[0]
    lam_prior = (x_prior - a[0]) / d[0]
    var_prior = (depth_m[1] / abs(d[0])) ** 2
    lam = lam_prior * var_fit / (var_fit + var_prior)
    return a + lam * np.array([d[0], d[1], 0.0])


def choose_hit_points(fits: list[ShotFit], pts: list[list[np.ndarray]],
                      res: dict[tuple[int, int, int], FitResult], opts: FitOptions | None = None,
                      implausible_cost: float = 1e6) -> list[int]:
    """打点の候補（``hit_point_options``）が 2 つある所で、どちらを使うかを決める（候補の番号のリスト）。

    打点 i は、ショット i − 1 の終わりとショット i の始まりの両方に効くので、候補 c ごとに
    「ショット i − 1 を終わり c で当てはめ直したずれ」と「ショット i を始まり c で当てはめ直したずれ」
    （どちらも画像上のずれの 2 乗の和に、平面を「打点 → 次の打点」から動かした分の罰を足したもの。相手の端の
    候補は、いちばん合うもの）を足し、小さい方を選ぶ。平面を動かせるので、外れた終わりでも長いショットは
    画像上では合ってしまい、ずれは平面を動かした分に出る。
    1 本のショットのずれだけでは、短いショットの奥行きのあいまいさ（角・後ろのカメラ）で、外れた候補の方が
    わずかに合うことがある。長く飛んだ前のショットの終わりは、外れた候補ではずれが大きく増える。
    打った側から見てありえない当てはめは ``implausible_cost`` を足す。比べられなければ最初の候補。

    Args:
        res: (ショットの番号, 始まりの候補の番号, 終わりの候補の番号) → 当てはめ直した結果
    """
    opts = opts or FitOptions(refine_plane=True)
    sig = np.array([np.deg2rad(opts.plane_sigma_deg), opts.plane_sigma_m])

    def cost(f: FitResult, side: str) -> float:
        c = f.rms_px ** 2 * f.n_obs
        if f.mode == "plane" and len(f.theta) >= 6 and opts.refine_plane:
            # 事前分布の 1σ は、当てはめと同じく画像上の 1 px（ずれが大きければその大きさ）に当たる
            w = max(opts.loss_scale_px, f.rms_px)
            c += float(np.sum((w * f.theta[4:6] / sig) ** 2))
        return c + (0.0 if _plausible_fit(f, side) else implausible_cost)

    best: dict[tuple[int, str, int], float] = {}  # (ショット, "a" | "b", 候補) → いちばん小さいずれ
    for (i, ka, kb), f in res.items():
        c = cost(f, fits[i].side)
        for key in ((i, "a", ka), (i, "b", kb)):
            best[key] = min(best.get(key, np.inf), c)
    chosen = [0] * len(pts)
    for i, opts_i in enumerate(pts):
        if len(opts_i) < 2:
            continue
        totals = []
        for k in range(len(opts_i)):
            parts = [best.get((i, "a", k)) if i < len(fits) else None, best.get((i - 1, "b", k)) if i > 0 else None]
            parts = [x for x in parts if x is not None]
            totals.append(sum(parts) if parts else np.inf)
        # 片方の候補でしか当てはめられていないショットがあると比べられないので、両方の候補で同じショットが
        # 当てはめられているときだけ比べる
        same = all((best.get((i, "a", k)) is None) == (best.get((i, "a", 0)) is None)
                   and (best.get((i - 1, "b", k)) is None) == (best.get((i - 1, "b", 0)) is None)
                   for k in range(len(opts_i)))
        if same and np.isfinite(min(totals)):
            chosen[i] = int(np.argmin(totals))
    return chosen


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


def foot_hit_times(fits: list[ShotFit], feet: list[np.ndarray | None], t_audio: list[float | None],
                   phys: PhysicsParams, times: list[float], reach_m: float = 1.5, max_shift_s: float = 0.1,
                   step_s: float = 0.002, max_speed_mps: float = MAX_FIT_SPEED_MPS) -> list[float]:
    """打球音のないショットの打った時刻を、当てはめた軌道が打つ人の足元に水平にいちばん近づく時刻へ戻す。

    軌道の向きの変化から決めた打球の時刻は、打った後に最初に見えたコマに寄って遅れやすい（実動画の 30 fps で 70 ms）。
    遅れた時刻の「打点」は飛んだ先にあり（速いショットは 2 m）、初速はその時刻の速さなので大きく下がる。
    当てはめた軌道を前へ戻すと打つ人の所を通るので、打点が足元から ``reach_m`` より離れているときだけ、
    ``max_shift_s`` 前までの間でいちばん足元に近い時刻にする（そこでも ``reach_m`` より離れていれば、足元か軌道が
    合っていないので直さない）。前へ戻すと速さは急に上がるので、
    ``max_speed_mps`` を超える所までは戻さない。前のショットの最後の観測より前にも戻さない。
    """
    out = list(times)
    for i, sf in enumerate(fits):
        if feet[i] is None or t_audio[i] is not None or sf.fit is None:
            continue
        lo = times[i] - max_shift_s
        prev = fits[i - 1] if i > 0 else None
        if prev is not None and prev.obs is not None and len(prev.obs.t):
            lo = max(lo, float(prev.obs.t[-1]))
        t = np.arange(times[i], lo - step_s / 2, -step_s)
        if len(t) < 2:
            continue
        pos, vel = simulate(sf.fit.p0, sf.fit.v0, sf.fit.t_ref, t, sf.fit.phys(phys))
        ok = np.isfinite(pos).all(axis=1) & (np.linalg.norm(vel, axis=1) <= max_speed_mps)
        gap = np.where(ok, np.hypot(pos[:, 0] - feet[i][0], pos[:, 1] - feet[i][1]), np.inf)
        if not np.isfinite(gap[0]) or gap[0] <= reach_m:
            continue
        k = int(np.argmin(gap))
        if gap[k] <= reach_m and gap[k] < gap[0] - 0.3:
            out[i] = float(t[k])
    return out


# 並べて動かすプロセスでは、numpy・scipy の行列計算（OpenBLAS）を 1 スレッドにする。プロセスの数だけで並べているので
# スレッドを増やしても速くならず、28 スレッドの PC では読み込むだけで 1 プロセス 1.5 GB を確保する（1 スレッドなら 0.1 GB。
# 15 プロセスで 23 GB になり、ほかのアプリが動いていると確保に失敗して 1 つずつの計算に戻っていた）
_WORKER_ENV = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
_shared_pool: list[Any] = []     # ``worker_pool`` の中で使い回すプロセスの組（[組, プロセスの数]。壊れたら組は None）


@contextmanager
def _worker_env() -> Iterator[None]:
    """この中で起動したプロセスに ``_WORKER_ENV`` を渡す（すでに決めてある値はそのまま）。"""
    old = {k: os.environ.get(k) for k in _WORKER_ENV}
    os.environ.update({k: v for k, v in _WORKER_ENV.items() if old[k] is None})
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)


@contextmanager
def worker_pool(workers: int) -> Iterator[None]:
    """この中の ``_parallel_map``（``fit_shots`` の当てはめ）で、同じプロセスの組を使い回す。

    Windows ではプロセスの起動に 1 組 4〜5 秒（15 プロセス）かかり、``fit_shots`` は 1 回に 4 組を起動するので、
    ラリーごとに呼ぶショットの一覧では起動の時間が積み重なる（ROADMAP F3）。
    """
    if workers <= 1 or _shared_pool:
        yield
        return
    from concurrent.futures import ProcessPoolExecutor

    with _worker_env():
        ex = ProcessPoolExecutor(workers)
        _shared_pool[:] = [ex, workers]
        try:
            yield
        finally:
            _shared_pool.clear()
            ex.shutdown(cancel_futures=True)


def pool_active() -> bool:
    """``worker_pool`` のプロセスの組が使える（壊れていない）か。"""
    return bool(_shared_pool) and _shared_pool[0] is not None


def _parallel_map(fn: Callable[..., Any], args: list[tuple[Any, ...]], workers: int) -> list[Any]:
    """``fn(*a)`` を ``args`` の順に計算する。``workers`` が 2 以上なら別々のプロセスで並べて計算する（結果は同じ）。

    ``worker_pool`` の中なら、そのプロセスの組を使う。プロセスを起動できなかったとき（Windows で、
    ``if __name__ == "__main__":`` のないスクリプトから呼んだ、など）は 1 つずつ計算する。
    """
    workers = min(workers, len(args))
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        from concurrent.futures.process import BrokenProcessPool

        try:
            if _shared_pool:
                # 壊れた組は使わず 1 つずつ計算する（ラリーを並べて動かしているとき、それぞれが新しい組を作らないように）
                if _shared_pool[0] is not None:
                    with _worker_env():
                        return list(_shared_pool[0].map(fn, *zip(*args)))
            else:
                with _worker_env(), ProcessPoolExecutor(workers) as ex:
                    return list(ex.map(fn, *zip(*args)))
        except (BrokenProcessPool, OSError):
            if _shared_pool:
                _shared_pool[0] = None     # ``worker_pool`` の終わりで片付ける
    return [fn(*a) for a in args]


def _try_fit(obs: Observations, t_hit: float, plane: FlightPlane, phys: PhysicsParams, opts: FitOptions,
             theta0: np.ndarray | None) -> FitResult | None:
    """plane モードで当てはめる（計算できなければ None）。"""
    return _run(FitJob(obs, t_hit, plane, phys, opts, theta0))


# ``fit_shots`` の当てはめの計算の仕方（``backend``）
FIT_BACKENDS = ("cpu", "batch", "gpu")
FIT_BACKEND_ENV = "SMASH_SPEED_FIT_BACKEND"     # 既定の計算の仕方を変える環境変数（GUI・チェックで試すとき）
# ``fit_shots`` の中身：段階ごとに (関数, 引数のリスト) を出して、その結果のリストを受け取りながら進む
FitSteps = Generator[tuple[Callable[..., Any], list[tuple[Any, ...]]], list[Any], Any]
FitJobs = Generator[list[FitJob], list[FitResult | None], Any]


def default_backend(backend: str | None = None) -> str:
    """当てはめの計算の仕方。``backend`` が None なら、環境変数 ``SMASH_SPEED_FIT_BACKEND``（なければ "cpu"）。"""
    backend = backend or os.environ.get(FIT_BACKEND_ENV) or "cpu"
    if backend not in FIT_BACKENDS:
        raise ValueError(f"当てはめの計算の仕方は {FIT_BACKENDS} のどれかです：{backend}")
    return backend


def _engine(backend: str) -> str:
    """``backend`` の、軌道の積分の計算の仕方（``fit_batch.get_integrator``）。"""
    if backend not in FIT_BACKENDS or backend == "cpu":
        raise ValueError(f"まとめて解く当てはめの計算の仕方は {FIT_BACKENDS[1:]} のどれかです：{backend}")
    return "dml" if backend == "gpu" else "numpy"


def _fit_jobs(fn: Callable[..., Any], args: list[tuple[Any, ...]]) -> FitJobs:
    """``[fn(*a) for a in args]``（``fn`` は ``grid_fit``・``loose_fit``・``wide_fit``・``_try_fit``）を、当てはめを
    1 回ずつ解く代わりに、当てはめの引数のリストを出して結果を受け取る形で計算する（``fit_batch.solve_together``）。"""
    if fn is grid_fit:
        return (yield [_grid_job(*a) for a in args])
    if fn is _try_fit:
        return (yield [FitJob(*a) for a in args])
    if fn is loose_fit:
        todo = [k for k, a in enumerate(args) if a[2] is not None]
        out: list[Any] = [None] * len(args)
        for k, f in zip(todo, (yield [_loose_job(*args[k]) for k in todo])):
            out[k] = args[k][2] if f is None else f
        return out
    if fn is wide_fit:
        planes = [grid_planes(a[2], a[6], WIDE_YAW_DEG) for a in args]
        grid = iter((yield [_grid_job(a[0], a[1], p, a[4], a[5]) for a, ps in zip(args, planes) for p in ps]))
        cands = [_wide_candidates([next(grid) for _ in ps], a[2]) for a, ps in zip(args, planes)]
        tight = iter((yield [_tight_job(a[0], a[1], g, a[4], a[5]) for a, cs in zip(args, cands) for g in cs]))
        return [_wide_choice(cs, [next(tight) for _ in cs], a[2], a[3]) for a, cs in zip(args, cands)]
    raise ValueError(f"まとめて解けない当てはめです：{fn.__name__}")


def fit_jobs(steps: FitSteps) -> FitJobs:
    """``fit_shots_steps`` の計算を、当てはめの引数のリストを出して結果を受け取る形にする（まとめて解く用）。"""
    res: list[Any] | None = None
    while True:
        try:
            fn, args = next(steps) if res is None else steps.send(res)
        except StopIteration as e:
            return e.value
        res = yield from _fit_jobs(fn, args)


def _fit_map(fn: Callable[..., Any], args: list[tuple[Any, ...]], workers: int, backend: str = "cpu") -> list[Any]:
    """``fn``（``grid_fit``・``loose_fit``・``wide_fit``・``_try_fit``）を ``args`` の順に計算する。

    ``backend`` が "cpu" なら 1 回ずつ（``_parallel_map``。``workers`` 個のプロセスで並べる）。"batch"・"gpu" なら
    全部の当てはめをまとめて解く（``fit_batch``。"gpu" は軌道の積分を GPU で。使えなければ "batch" と同じ）。
    """
    if backend == "cpu" or not args:
        return _parallel_map(fn, args, workers)
    return solve_together([_fit_jobs(fn, args)], _engine(backend))[0]


def fit_shots(det: pd.DataFrame, t_hits: list[float], camera: Camera, phys: PhysicsParams,
              sides: list[str] | None = None, opts: FitOptions | None = None, n_rounds: int = 2,
              min_obs: int = 5, margin_s: float = 0.01, hit_sigma_m: tuple[float, float] = (0.5, 0.5),
              t_last: float | None = None, refine_times: bool = True, workers: int = 1,
              max_rms_px: float = np.inf, t_audio: list[float | None] | None = None,
              speed_of_sound_mps: float = 343.0, max_kmh: float = np.inf,
              bad_max_nfev_per_param: int = 30, hitter_xy: list[Any] | None = None,
              hitter_sigma_m: tuple[float, float] = (0.5, 2.0), max_hitter_gap_m: float = 2.5,
              serve_depth_m: tuple[float, float] | None = None, hitter_reach_m: float | None = 1.5,
              joint: Any = None, backend: str | None = None) -> list[ShotFit]:
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
        hitter_xy: ショットごとの打つ人の床の上の足元 [m]（``hitter.hitter_at``。分からなければ None）。
            打点の手がかりにする（``initial_fit``・``joint_hit_points``）
        hitter_sigma_m: 足元を打点の目安にするときの不確かさ（水平, 高さ）[m]。高さは分からないので大きくする
            （水平は IMG_4397 80〜100 s で 1.0 m より 0.5 m の方が足元から 2.5 m 以上離れた打点が少なく、14 → 12 → 8 本）
        max_hitter_gap_m: 足元からこれより離れた打点は外れとみなす [m]
        hitter_reach_m: 足元から打点まで水平に届く距離 [m]。打球音のないショットの打点がこれより離れていたら、
            打った時刻を、当てはめた軌道が足元にいちばん近づく時刻へ戻す（``foot_hit_times``。None なら直さない）
        serve_depth_m: 最初のショットをサーブとみなすときの、打点のネットからの深さの目安（平均, 標準偏差）[m]。
            サーブは前の軌道がないので、角・後ろのカメラでは打点の奥行きが決まらず、初速が 15〜40% ずれる。
            最初の当てはめの打点を、飛行平面に沿ってこの目安と重みつきで平均する（``serve_hit_point``）。
            None なら使わない（ラリーの最初の打球が見つかっていない動画では、サーブではないので）。
            最初のショットの打つ人の足元（``hitter_xy``）があれば、足元の方を使う
        joint: 与えると（``rally_fit.JointOptions``）、最後にラリー全体をまとめて当てはめ直す（``rally_fit.fit_rallies``。
            となりのショットと打点・打った時刻をつなぎ、長いショットから終端速度も求める）。つながらなかった
            ショットは、ショットごとの当てはめのまま。動画に何ラリーもあるときは、ラリーごとにこの関数を
            ``joint=None`` で呼んでから ``fit_rallies`` をまとめて呼ぶ（終端速度を動画全体で 1 つにする）
        backend: 当てはめの計算の仕方。"cpu" は 1 回ずつ（``workers`` 個のプロセスで並べる）、"batch" は段階ごとに
            全部の当てはめをまとめて解く（``fit_batch``。プロセスは使わない）、"gpu" は "batch" の軌道の積分を GPU で
            （使えなければ "batch" と同じ）。結果は、最適化の終わりの判定の幅の中で同じ（``docs/speed.md``）。
            None なら ``default_backend``（環境変数 ``SMASH_SPEED_FIT_BACKEND``。なければ "cpu"）

    Returns:
        ``t_hits`` と同じ順の ``ShotFit``（``t_hit`` は直した時刻）
    """
    steps = fit_shots_steps(det, t_hits, camera, phys, sides, opts, n_rounds, min_obs, margin_s, hit_sigma_m, t_last,
                            refine_times, max_rms_px, t_audio, speed_of_sound_mps, max_kmh, bad_max_nfev_per_param,
                            hitter_xy, hitter_sigma_m, max_hitter_gap_m, serve_depth_m, hitter_reach_m)
    backend = default_backend(backend)
    if backend == "cpu":
        res: list[Any] | None = None
        while True:
            try:
                fn, args = next(steps) if res is None else steps.send(res)
            except StopIteration as e:
                fits = e.value
                break
            res = _parallel_map(fn, args, workers)
    else:
        fits = solve_together([fit_jobs(steps)], _engine(backend))[0]
    if joint is not None:
        from smash_speed.rally_fit import fit_rallies

        opts = FitOptions(**{**(opts or FitOptions(refine_plane=True)).__dict__, "quadratic_priors": True})
        if t_last is None:
            vis_t = det.timestamp[(det.visible > 0) & det.x_px.notna()].to_numpy(float)
            t_last = float(vis_t.max()) if len(vis_t) else (float(t_hits[-1]) if len(t_hits) else 0.0)
        res, _ = fit_rallies([joint_job(fits, det, camera, hitter_xy, t_audio, margin_s, t_last, speed_of_sound_mps,
                                        max_rms_px)], phys, opts, joint)
        fits = res[0].fits
    return fits


def fit_shots_steps(det: pd.DataFrame, t_hits: list[float], camera: Camera, phys: PhysicsParams,
                    sides: list[str] | None = None, opts: FitOptions | None = None, n_rounds: int = 2,
                    min_obs: int = 5, margin_s: float = 0.01, hit_sigma_m: tuple[float, float] = (0.5, 0.5),
                    t_last: float | None = None, refine_times: bool = True, max_rms_px: float = np.inf,
                    t_audio: list[float | None] | None = None, speed_of_sound_mps: float = 343.0,
                    max_kmh: float = np.inf, bad_max_nfev_per_param: int = 30, hitter_xy: list[Any] | None = None,
                    hitter_sigma_m: tuple[float, float] = (0.5, 2.0), max_hitter_gap_m: float = 2.5,
                    serve_depth_m: tuple[float, float] | None = None, hitter_reach_m: float | None = 1.5) -> FitSteps:
    """``fit_shots`` の中身（引数は同じ。ラリーをまとめた当てはめ直しの前まで）。

    段階ごとに (関数, 引数のリスト) を ``yield`` で出し、その結果のリスト（``[関数(*a) for a in 引数のリスト]``）を
    受け取って進む。1 つの段階の中の当てはめは互いに関係しないので、呼ぶ側が並べて（``_parallel_map``）、または
    まとめて（``fit_jobs``・``fit_batch.solve_together``）計算する。戻り値は ``t_hits`` と同じ順の ``ShotFit``。
    """
    opts = opts or FitOptions(refine_plane=True)
    opts = FitOptions(**{**opts.__dict__, "quadratic_priors": True})
    t_hits = [float(t) for t in t_hits]
    sides = list(sides) if sides is not None else [""] * len(t_hits)
    vis_t = det.timestamp[(det.visible > 0) & det.x_px.notna()].to_numpy(float)
    if t_last is None:
        t_last = float(vis_t.max()) if len(vis_t) else (t_hits[-1] if t_hits else 0.0)
    fits = [ShotFit(t, sides[i], t) for i, t in enumerate(t_hits)]
    feet = [None if h is None or not np.isfinite(np.asarray(h, float)[:2]).all() else np.asarray(h, float)[:2]
            for h in (hitter_xy if hitter_xy is not None else [None] * len(t_hits))]

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
    # ``initial_fit`` を、格子の平面ごと（互いに関係しない）と当てはめ直しに分けて並べる。1 本の格子は 9〜18 回の
    # 当てはめで、実動画では 1 本に 1 分かかることがあり、ほかのプロセスが空いたまま待っていた（ROADMAP F3b）
    planes = {i: grid_planes(sides[i], feet[i]) for i in todo}
    grid_jobs = [(i, p) for i in todo for p in planes[i]]
    grid = yield grid_fit, [(fits[i].obs, fits[i].t_hit, p, phys, opts) for i, p in grid_jobs]
    best = {i: best_grid_fit([f for (j, _), f in zip(grid_jobs, grid) if j == i]) for i in todo}
    first = yield loose_fit, [(fits[i].obs, fits[i].t_hit, best[i], phys, opts, feet[i], hitter_sigma_m)
                              for i in todo]
    # 打った側から見てありえない結果だけ、広い向きの格子から当てはめ直す（``initial_fit`` の最後。ROADMAP B3b）
    retry = [k for k, (i, f) in enumerate(zip(todo, first)) if f is not None and not _plausible_fit(f, sides[i])]
    wide = yield wide_fit, [(fits[todo[k]].obs, fits[todo[k]].t_hit, sides[todo[k]], first[k], phys, opts,
                             feet[todo[k]]) for k in retry]
    for k, f in zip(retry, wide):
        first[k] = f
    for i, f in zip(todo, first):
        fits[i].fit = f
        if f is None:
            fits[i].message = "軌道を当てはめられませんでした"

    t_audio = list(t_audio) if t_audio is not None else [None] * len(t_hits)
    for _ in range(n_rounds):
        times = refine_hit_times(fits, camera, phys) if refine_times else [sf.t_hit for sf in fits]
        if any(t is not None for t in t_audio):
            times = audio_hit_times(fits, t_audio, camera, phys, times, speed_of_sound_mps)
        moved = False
        if hitter_reach_m is not None and any(f is not None for f in feet):
            new = foot_hit_times(fits, feet, t_audio, phys, times, hitter_reach_m)
            moved, times = new != times, new
        if refine_times or moved or any(t is not None for t in t_audio):
            split(times)
        pts = hit_point_options(fits, phys, feet=feet if any(f is not None for f in feet) else None,
                                max_foot_m=max_hitter_gap_m)
        if serve_depth_m is not None and _ == 0 and fits and fits[0].fit is not None and feet[0] is None:
            # 1 回目だけ（2 回目からは、目安に寄せた打点を事前分布にして当てはめ直した始まりなので、重ねて寄せない）
            pts[0] = [serve_hit_point(a, fits[0].fit, fits[0].side, serve_depth_m) for a in pts[0]]
        jobs, args = [], []
        for i, sf in enumerate(fits):
            if sf.fit is None or sf.n_obs < min_obs:
                continue
            fo = None
            for ka, a in enumerate(pts[i]):
                # 足元の真上にした打点（``hit_point_options`` が足元の水平の位置をそのまま入れる）は、不確かさを大きくする
                on_foot = feet[i] is not None and np.hypot(*(a[:2] - feet[i])) < 1e-9
                for kb, b in enumerate(pts[i + 1]):
                    if np.linalg.norm((b - a)[:2]) < 0.3:
                        continue
                    fo = fo or FitOptions(**{
                        **opts.__dict__, "p0_prior_sigma_m": hit_sigma_m,
                        "refine_plane": opts.refine_plane and sf.n_obs > 5,
                        "init_state": _state_at(sf.fit, sf.t_hit, phys), "ci_method": "none",
                        **({"max_nfev_per_param": bad_max_nfev_per_param}
                           if not (sf.fit.rms_px <= max_rms_px and sf.fit.speed0 * MPS_TO_KMH <= max_kmh)
                           else {})})
                    jobs.append((i, ka, kb))
                    args.append((sf.obs, sf.t_hit, FlightPlane.from_points(a, b), phys,
                                 FitOptions(**{**fo.__dict__, "p0_prior_m": a,
                                               "p0_prior_sigma_m": hitter_sigma_m if on_foot else hit_sigma_m}),
                                 None))
        res = {job: f for job, f in zip(jobs, (yield _try_fit, args))
               if f is not None and f.success and np.isfinite(f.speed0)}
        chosen = choose_hit_points(fits, pts, res, opts)
        for i, sf in enumerate(fits):
            got = [(k, f) for k, f in res.items() if k[0] == i]
            if not got:
                continue
            # 選んだ打点の組の当てはめ（なければ、ありえる中で画像上のずれの小さいもの）
            k, f = min(got, key=lambda kf: (kf[0][1:] != (chosen[i], chosen[i + 1]),
                                            not _plausible_fit(kf[1], sf.side), kf[1].rms_px))
            sf.fit, sf.hit_point = f, pts[i][k[1]]

    # 最後に、平面を決めたまま信頼区間つきで当てはめ直す
    jobs, args = [], []
    for i, sf in enumerate(fits):
        if (sf.fit is None or sf.fit.plane is None or opts.ci_method == "none"
                or abs(sf.fit.t_ref - sf.t_hit) > 1e-9 or sf.n_obs < min_obs or not sf.fit.rms_px <= max_rms_px):
            continue
        on_foot = (feet[i] is not None and sf.hit_point is not None
                   and np.hypot(*(sf.hit_point[:2] - feet[i])) < 1e-9)
        fo = FitOptions(**{**opts.__dict__, "p0_prior_m": sf.hit_point,
                           "p0_prior_sigma_m": hitter_sigma_m if on_foot else hit_sigma_m,
                           "refine_plane": False, "init_state": None})
        jobs.append(i)
        args.append((sf.obs, sf.t_hit, sf.fit.plane, phys, fo, sf.fit.theta[:4] if sf.fit.mode == "plane" else None))
    for i, f in zip(jobs, (yield _try_fit, args)):
        if f is not None:
            fits[i].fit = f
    return fits


def joint_job(fits: list[ShotFit], det: pd.DataFrame, camera: Camera, feet: list[Any] | None = None,
              t_audio: list[float | None] | None = None, margin_s: float = 0.01, t_last: float | None = None,
              speed_of_sound_mps: float = 343.0, max_rms_px: float = np.inf) -> dict[str, Any]:
    """``rally_fit.fit_rallies`` に渡す 1 ラリー分の引数（``fit_shots`` の結果と、そのときの引数から）。"""
    return {"fits": fits, "det": det, "camera": camera, "feet": feet, "t_audio": t_audio, "margin_s": margin_s,
            "t_last": t_last, "speed_of_sound_mps": speed_of_sound_mps, "max_rms_px": max_rms_px}


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
            note=f"{sf.n_obs} コマ・ずれ {f.rms_px:.1f} px" + ("・ラリーでつないだ" if f.message == "joint" else "")))
    return out


# =============================================================================
# 合成ラリーでの測り方
# =============================================================================
def synthetic_benchmark(n_rallies: int = 10, fps: float = 120.0, seed: int = 0, hits: str = "true",
                        noise_px: float = 1.5, dropout: float = 0.1, camera: Camera | None = None,
                        cfg: dict[str, Any] | None = None, verbose: bool = False,
                        keep_shots: bool = False, feet_noise_m: float | None = None,
                        serve_depth_m: tuple[float, float] | None = None, joint: Any = None,
                        true_vt: float | None = None, workers: int = 1) -> dict[str, Any]:
    """合成ラリーで、ショットの種類ごとの初速の誤差を測る。

    Args:
        hits: 打球の時刻と打った側の決め方。"true" は正解、"track" は ``find_hits``（軌道だけ）、
            "audio" は ``find_hits``（打つたびの打球音の時刻を、カメラまでの音の遅れつきで渡す）。
            見つからなかった打球・正解のない打球は数えない（打球の見つけ方の精度は ``hits.synthetic_benchmark``）
        keep_shots: ``rows`` に、推定したショット（``Shot``。``index_in_rally`` は見つけた打球の順）の列 ``shot`` を足す
        feet_noise_m: 与えると、本当の打点の真下からこの大きさ（水平の各向きの標準偏差）[m] だけずらした位置を
            打つ人の足元として渡す（``fit_shots`` の ``hitter_xy``。足元の手がかりの効き目を見る）
        joint: 与えると（``rally_fit.JointOptions``）、ショットごとの当てはめの後に、全部のラリーを 1 本の動画として
            まとめて当てはめ直す（``rally_fit.fit_rallies``）。``rows`` の ``base_kmh``・``base_err_pct`` がショットごと、
            ``est_kmh``・``err_pct`` がまとめた後の値。戻り値の ``vt`` が終端速度の記録
        true_vt: 合成ラリーを作る・観測するときの本当の終端速度 [m/s]（省略時は設定の値と同じ。当てはめは設定の値から）
        workers: ショットごとの当てはめを並べるプロセスの数（``fit_shots``）

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
    true_phys = phys if true_vt is None else PhysicsParams(phys.g, true_vt, phys.method, phys.rtol, phys.atol)
    rally_rng = np.random.default_rng([seed, 1])
    done = []
    for r in range(n_rallies):
        rally = make_rally(rally_rng, true_phys, n_shots=int(rally_rng.integers(4, 13)),
                           server_side="h" if r % 2 == 0 else "o")
        det = observe_rally(rally, camera, fps, true_phys, np.random.default_rng([seed, 2, r]), noise_px=noise_px,
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
        feet = None
        if feet_noise_m is not None:
            foot_rng = np.random.default_rng([seed, 3, r])
            feet = [None if j is None else segs[j].p0[:2] + foot_rng.normal(0.0, feet_noise_m, 2) for j in pairs]
        fits = fit_shots(det, t_hits, camera, phys, sides, opts, t_audio=t_audio, hitter_xy=feet, workers=workers,
                         serve_depth_m=serve_depth_m if pairs and pairs[0] == 0 else None)
        done.append((segs, pairs, fits, joint_job(fits, det, camera, feet, t_audio)))
    vt_rec, joined = None, [None] * len(done)
    if joint is not None:
        from smash_speed.rally_fit import fit_rallies

        joined, vt_rec = fit_rallies([d[3] for d in done], phys,
                                     FitOptions(**{**opts.__dict__, "quadratic_priors": True}), joint)
    rows = []
    for r, (segs, pairs, base, _) in enumerate(done):
        fits = base if joined[r] is None else joined[r].fits
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
            if joint is not None:
                b = base[i].speed_kmh
                rows[-1].update(base_kmh=b, base_err_pct=None if b is None else 100 * (b / true - 1),
                                joint=bool(joined[r].joint[i]), base_dt_hit_ms=1000 * (base[i].t_hit - s.t_hit),
                                hit_err_m=None if sf.fit is None else float(np.linalg.norm(sf.fit.p0 - s.p0)),
                                base_hit_err_m=None if base[i].fit is None
                                else float(np.linalg.norm(base[i].fit.p0 - s.p0)))
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
    out = {"n_rallies": n_rallies, "fps": fps, "hits": hits, "rows": df, "by_type": by_type,
           "mape": float(e.mean()) if len(e) else None, "n_failed": int(df.err_pct.isna().sum()) if len(df) else 0}
    if joint is not None:
        b = df.base_err_pct.dropna().abs() if len(df) else pd.Series(dtype=float)
        out.update(vt=vt_rec, base_mape=float(b.mean()) if len(b) else None,
                   n_joint=int(df.joint.sum()) if len(df) else 0)
    return out


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
    ap.add_argument("--feet-noise", type=float, default=None,
                    help="本当の打点の真下からこの大きさ [m] ずらした位置を、打つ人の足元として渡す")
    ap.add_argument("--camera", default="side", choices=["side", "corner", "low"],
                    help="合成のカメラの位置（side：横・遠く、corner：角の後ろ・低い、low：横・低い）")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    from smash_speed.config import load_config
    from smash_speed.synth_rally import named_rally_camera

    cfg = load_config()
    camera = named_rally_camera(a.camera, cfg)
    for fps in a.fps:
        for h in a.hits:
            t0 = time.time()
            r = synthetic_benchmark(a.rallies, fps, a.seed, h, camera=camera, cfg=cfg, verbose=a.verbose,
                                    feet_noise_m=a.feet_noise)
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
