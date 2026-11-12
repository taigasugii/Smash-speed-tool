"""2 台のスマホの動画から、ラリーのすべてのショットの 3 次元の軌道と「正しい初速」を出す。

スピードガンがなくても、1 台のカメラの解析（奥行きが決まりにくい）の答え合わせに使える値を作る。

流れ（``analyze_stereo_rally``）：

1. **時刻合わせ**：音（打球音・手をたたく音）で大まかに合わせ、映像（2 台の視線のすき間）で細かく合わせる
   （``smash_speed.stereo_sync``）。
2. **キャリブレーション**：2 本の動画をそれぞれ今までどおりコートに合わせる（同じコートの座標になる）。
3. **打球**：補間して三角測量した 3 次元の点の、速度が急に変わる所（``find_hits_3d``）。
4. **軌道**：打球から次の打球までの飛行ごとに、空気抵抗つきの運動方程式を **連続した時間の中で** 2 台の画像の位置へ
   一緒に当てはめる（``solve_session``）。2 台のコマは同じ瞬間ではないので、コマの組を三角測量するのではなく、
   それぞれのコマの時刻で軌道を評価して画像の上のずれを小さくする。外れ値に強い損失（Huber）で当てはめ、
   ずれの大きいコマは外して当て直す。
5. **打った時刻**：前のショットの軌道と次のショットの軌道が 3 次元で交わる時刻（速いショットほど鋭く決まる）。
   ラリーの最初のショットは打球音（打点から 2 台への音の遅れを引く）。
6. **書き出し**：``stereo_truth.json``（ショットごとの初速 ± σ・打点・打ち出し角・軌道の初期値。
   ``agent/checks/real_videos_truth.json`` と同じ考え方）・``shots.csv``・``trajectories.csv``。
7. **答え合わせ**（``compare_shots``）：2 本のどちらかを 1 台で解析した ``shots.csv`` と比べる。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from smash_speed.camera import Camera
from smash_speed.physics import (MAX_FIT_SPEED_MPS, MPS_TO_KMH, FitOptions, FitResult, Observations, PhysicsParams, fit_trajectory,
                                 initial_guess, simulate, simulate_until_ground)
from smash_speed.shots import Shot, side_of
from smash_speed.stereo_sync import (SPEED_OF_SOUND_MPS, RayPairs, SyncModel, SyncReport, TrackInterpolator,
                                     ray_gap, visible_points)

TRUTH_FILE = "stereo_truth.json"
TRAJECTORY_FILE = "trajectories.csv"
COMPARE_FILE = "stereo_compare.csv"


@dataclass
class TruthOptions:
    """当てはめの設定。時刻 [s]・長さ [m]・画像 [px]。"""

    min_obs_each: int = 3            # 飛行ごとに、それぞれのカメラで要るコマの数
    min_obs: int = 8                 # 2 台を合わせて要るコマの数
    margin_s: float = 0.004          # 打球の時刻の前後これだけは、どちらの飛行か紛らわしいので使わない
    rally_gap_s: float = 3.0         # 打球の間がこれより空いたら別のラリー
    tail_s: float = 4.0              # ラリーの最後のショットを追う長さ
    loss_scale_px: float = 2.0       # Huber 損失の切り替え点
    reject_k: float = 3.5            # ずれが「ふつうのずれ × これ」を超えたコマを外す
    reject_min_px: float = 4.0       # ただしこのずれより小さいコマは外さない
    reject_rounds: int = 2
    time_rounds: int = 4             # 打った時刻を直して当て直す回数
    cross_half_s: float = 0.15       # 前後の軌道の交わる時刻を探す幅（最初の回。次からは 1/3）
    cross_max_gap_m: float = 0.5     # 交わる所のすき間がこれより大きければ、交わる時刻を使わない
    cross_floor_m: float = 0.03      # 交わる位置の不確かさの下限（シャトルの長さほど）
    merge_gap_m: float = 0.3         # 打球の前後で位置・速度がつながっていれば、打球ではなかったとして 1 つの飛行にする
    merge_dv_frac: float = 0.1
    audio_match_s: float = 0.1       # 打球音と打球を同じとみなす時刻の差
    prev_gap_s: float = 0.4          # 前の飛行の最後のコマからこれ以内の打球は、前の軌道と交わる時刻を探す
    audio_sigma_s: float = 0.005     # 打球音だけで決めた打った時刻の不確かさ（スマホの音と映像のずれが分からないとき）
    # 終端速度（空気抵抗の強さ）の決め方：session = 長い飛行から 1 本ずつ出した値の中央値をすべてのショットに使う
    # （シャトルごとに違い、高いロブ・クリアの初速に大きく効くので）| fixed = 設定の値のまま
    terminal_velocity: str = "session"
    vt_min_duration_s: float = 0.5   # 終端速度を出すのに使う飛行の長さ・コマの数の下限
    vt_min_obs: int = 40
    vt_min_shots: int = 3            # 長い飛行がこれより少なければ、設定の値のまま
    vt_range_mps: tuple[float, float] = (4.5, 9.0)   # ありえる終端速度の範囲（外の値は使わない）
    terminal_velocity_sigma: float = 0.4   # 設定の値のまま使うときの終端速度の不確かさ [m/s]（6.7 ± 0.4）
    fit_window_s: float | None = None      # 最後の当てはめに使う、打ってからの長さ（None なら飛行の終わりまで）
    max_rms_px: float = 6.0          # 画像の上のずれ（RMS）がこれを超えたショットは「確かでない」
    max_kmh: float = 600.0
    min_hit_z_m: float = 0.1         # 打点がこれより低い「ショット」は使わない（床で跳ねた・転がったシャトル）
    min_hit_interval_s: float = 0.03  # 直した打った時刻がこれより近い 2 つは同じ打球（片方を消す）
    start_audio_match_s: float = 0.3  # ラリーの最初のショットの打球音を探す幅（前の軌道がなく、見え始めが遅れるので広く）
    start_track_sigma_s: float = 0.05  # 打球音もないラリーの最初のショットの、打った時刻の不確かさ
    rolling_shutter_s: tuple[float, float] = (0.0, 0.0)   # 画像の上の行から下の行までの読み出しの時間（分かれば）
    uncertainty: bool = True         # 終端速度・時刻合わせの不確かさも当て直して出す（当てはめが 4 回増える）
    speed_of_sound_mps: float = SPEED_OF_SOUND_MPS


@dataclass
class StereoShot:
    """2 台から出した 1 本のショット。時刻は 1 台目の動画の時刻 [s]。"""

    t_hit: float
    t_end: float = float("nan")
    fit: FitResult | None = None
    n_obs: tuple[int, int] = (0, 0)
    rms_px: tuple[float, float] = (float("nan"), float("nan"))
    n_rejected: int = 0
    hit_method: str = "track"        # crossing | audio | track
    t_hit_sigma: float = float("nan")
    t_hit_audio: float | None = None  # 打球音から出した打った時刻（比べる用）
    cross_gap_m: float | None = None
    sigma_parts: dict[str, float] = field(default_factory=dict)   # 初速の不確かさの内訳 [km/h]
    naive_kmh: dict[str, float] = field(default_factory=dict)     # 素朴な三角測量の速さ（比べる用）
    message: str = ""
    obs: list[Observations] = field(default_factory=list, repr=False)

    @property
    def kmh(self) -> float | None:
        return None if self.fit is None else self.fit.speed0 * MPS_TO_KMH

    @property
    def sigma_kmh(self) -> float | None:
        if self.fit is None or not self.sigma_parts:
            return None
        return float(np.sqrt(sum(v * v for v in self.sigma_parts.values())))

    @property
    def sigma_same_time_kmh(self) -> float | None:
        if self.fit is None or not self.sigma_parts:
            return None
        return float(np.sqrt(sum(v * v for k, v in self.sigma_parts.items() if k != "hit_time")))

    def ok(self, o: TruthOptions) -> bool:
        """正解として使えるか（当てはめが合っていて、ありえる速さ）。"""
        f = self.fit
        return bool(f is not None and np.isfinite(f.rms_px) and max(self.rms_px) <= o.max_rms_px
                    and f.speed0 * MPS_TO_KMH <= o.max_kmh and f.p0[2] > o.min_hit_z_m)


# =============================================================================
# 飛行ごとの当てはめ
# =============================================================================
def _residual_px(f: FitResult, ob: Observations, phys: PhysicsParams) -> np.ndarray:
    """それぞれのコマの、画像の上のずれの大きさ [px]。"""
    pos, _ = f.evaluate(ob.t, phys)
    return np.linalg.norm(ob.camera.project(pos) - ob.uv, axis=1)


def triangulated_points(obs: list[Observations]) -> tuple[np.ndarray, np.ndarray]:
    """同じ時刻へ補間した 2 台の視線の交わる点（時刻 [s]、位置 (N, 3) [m]）。コマの間隔の短い方を補間する。"""
    a, b = obs
    if len(a.t) < 1 or len(b.t) < 2:
        return np.empty(0), np.empty((0, 3))
    step = [np.median(np.diff(o.t)) if len(o.t) > 1 else np.inf for o in obs]
    if step[0] < step[1] and len(a.t) >= 2:
        uv = TrackInterpolator(a.t, a.uv)(b.t)
        ok = np.isfinite(uv).all(axis=1)
        t, g, X = b.t[ok], *ray_gap(a.camera, uv[ok], b.camera, b.uv[ok])
    else:
        uv = TrackInterpolator(b.t, b.uv)(a.t)
        ok = np.isfinite(uv).all(axis=1)
        t, g, X = a.t[ok], *ray_gap(a.camera, a.uv[ok], b.camera, uv[ok])
    good = np.isfinite(g)
    return t[good], X[good]


def nearest_points(obs: list[Observations]) -> tuple[np.ndarray, np.ndarray]:
    """時刻のいちばん近いコマどうしを、同じ瞬間とみなして三角測量した点（素朴なやり方。比べる用）。"""
    a, b = obs
    if len(a.t) == 0 or len(b.t) == 0:
        return np.empty(0), np.empty((0, 3))
    i = np.clip(np.searchsorted(b.t, a.t), 1, max(len(b.t) - 1, 1))
    j = np.where(np.abs(b.t[i - 1] - a.t) <= np.abs(b.t[np.minimum(i, len(b.t) - 1)] - a.t), i - 1,
                 np.minimum(i, len(b.t) - 1))
    step = max(np.median(np.diff(b.t)) if len(b.t) > 1 else 0.0, np.median(np.diff(a.t)) if len(a.t) > 1 else 0.0)
    ok = np.abs(b.t[j] - a.t) <= max(step, 1e-3) / 2 + 1e-9
    g, X = ray_gap(a.camera, a.uv[ok], b.camera, b.uv[j[ok]])
    good = np.isfinite(g)
    return a.t[ok][good], X[good]


def _first_speed_kmh(t: np.ndarray, X: np.ndarray) -> float | None:
    """3 次元の点の最初の 2 点の間の速さ [km/h]（素朴な「初速」）。"""
    if len(t) < 2 or t[1] <= t[0]:
        return None
    return float(np.linalg.norm(X[1] - X[0]) / (t[1] - t[0]) * MPS_TO_KMH)


def fit_flight(obs: list[Observations], t_ref: float, phys: PhysicsParams, o: TruthOptions,
               t_ref_sigma: float = 0.0, theta0: np.ndarray | None = None, reject: bool = True,
               ci: bool = True, estimate_vt: bool = False
               ) -> tuple[FitResult | None, list[Observations], int, str]:
    """1 つの飛行の 2 台の観測に、運動方程式を一緒に当てはめる。

    床に着いた後のコマと、ずれの大きいコマ（別の物の検出）を外して当て直す。

    Returns:
        (当てはめ（できなければ None）, 使った観測, 外したコマの数, できなかった理由)
    """
    n0 = sum(len(ob.t) for ob in obs)

    def enough(oo: list[Observations]) -> bool:
        return min(len(ob.t) for ob in oo) >= o.min_obs_each and sum(len(ob.t) for ob in oo) >= o.min_obs

    if not enough(obs):
        only = [k + 1 for k, ob in enumerate(obs) if len(ob.t) >= o.min_obs_each]
        return None, obs, 0, ("両方のカメラに写っているコマが足りません" if len(only) != 1
                              else f"カメラ {only[0]} にしか写っていません")
    fo = FitOptions(loss="huber", loss_scale_px=o.loss_scale_px, ci_method="jacobian" if ci else "none",
                    t_ref_sigma=t_ref_sigma, estimate_terminal_velocity=estimate_vt)
    if theta0 is not None:
        theta0 = np.asarray(theta0, float)[:6]
        if estimate_vt:
            theta0 = np.r_[theta0, np.log(phys.terminal_velocity)]

    def solve(oo: list[Observations], th: np.ndarray | None) -> FitResult | None:
        try:
            return fit_trajectory(oo, t_ref, "free", phys=phys, opts=fo, theta0=th)
        except ValueError:
            pass
        # 同じ瞬間に近いコマの組がない（fps が大きく違う・片方がまばら）ときは、補間した三角測量の点から始める
        t3, X3 = triangulated_points(oo)
        if len(t3) < 3:
            return None
        fo2 = FitOptions(**{**fo.__dict__, "init_state": initial_guess(t3, X3, t_ref)})
        try:
            return fit_trajectory(oo, t_ref, "free", phys=phys, opts=fo2)
        except ValueError:
            return None

    f = solve(obs, theta0)
    if f is None:
        return None, obs, 0, "当てはめを始められませんでした"
    for _ in range(o.reject_rounds if reject else 0):
        keep = []
        t_land = float(simulate_until_ground(f.p0, f.v0, f.t_ref, f.phys(phys), n_dense=2)["t_land"])
        for ob in obs:
            r = _residual_px(f, ob, phys)
            thr = max(o.reject_k * 1.4826 * float(np.median(r)), o.reject_min_px) if len(r) else np.inf
            m = np.isfinite(r) & (r <= thr)
            if np.isfinite(t_land):
                m &= ob.t <= t_land - 0.005
            keep.append(m)
        if all(m.all() for m in keep):
            break
        new = [ob.subset(m) for ob, m in zip(obs, keep)]
        if not enough(new):
            break
        f2 = solve(new, f.theta if not estimate_vt else np.r_[f.theta[:6], np.log(f.terminal_velocity)])
        if f2 is None:
            break
        obs, f = new, f2
    return f, obs, n0 - sum(len(ob.t) for ob in obs), ""


def crossing_3d(prev: FitResult, nxt: FitResult, phys: PhysicsParams, t0: float, half_s: float = 0.05
                ) -> tuple[float, float, float] | None:
    """前の飛行の軌道と次の飛行の軌道が、3 次元でいちばん近づく時刻 [s]・そのすき間 [m]・速度の差 [m/s]。

    次の飛行を時刻の前へ戻すと速さが急に上がる（空気抵抗が速さの 2 乗に比例）ので、速さが上限の目安
    （``MAX_FIT_SPEED_MPS``）になる所より前は見ない。
    """
    back = 0.95 * (1.0 / max(nxt.speed0, 1.0) - 1.0 / MAX_FIT_SPEED_MPS) / nxt.phys(phys).k
    lo = max(t0 - half_s, nxt.t_ref - max(back, 0.0), prev.t_ref + 1e-4)
    hi = t0 + half_s
    if hi <= lo:
        return None

    def gap(t: float) -> float:
        a, _ = prev.evaluate(np.array([t]), phys)
        b, _ = nxt.evaluate(np.array([t]), phys)
        d = float(np.linalg.norm(a[0] - b[0]))
        return d if np.isfinite(d) else 1e3

    ts = np.linspace(lo, hi, max(41, int((hi - lo) / 0.0025) + 1))
    pa, _ = prev.evaluate(ts, phys)
    pb, _ = nxt.evaluate(ts, phys)
    d = np.linalg.norm(pa - pb, axis=1)
    d = np.where(np.isfinite(d), d, 1e3)
    i = int(np.argmin(d))
    res = minimize_scalar(gap, bounds=(ts[max(i - 1, 0)], ts[min(i + 1, len(ts) - 1)]), method="bounded",
                          options={"xatol": 1e-5})
    t = float(res.x)
    _, va = prev.evaluate(np.array([t]), phys)
    _, vb = nxt.evaluate(np.array([t]), phys)
    dv = float(np.linalg.norm(va[0] - vb[0]))
    if not np.isfinite(dv) or dv <= 0:
        return None
    return t, float(res.fun), dv


# =============================================================================
# セッション全体
# =============================================================================
def camera_observations(dets: list[pd.DataFrame], cams: list[Camera], sync: SyncModel,
                        rolling_shutter_s: tuple[float, float] = (0.0, 0.0)) -> list[tuple[np.ndarray, np.ndarray]]:
    """2 台の見えているコマの (1 台目の時刻 [s], 画像の位置 [px])。

    読み出しの時間（ローリングシャッター）が分かっていれば、画像の行に応じてコマの時刻を直す（真ん中の行が 0）。
    """
    out = []
    for k, (det, cam) in enumerate(zip(dets, cams)):
        t, uv = visible_points(det)
        if rolling_shutter_s[k]:
            t = t + (uv[:, 1] / cam.height - 0.5) * rolling_shutter_s[k]
        out.append((t if k == 0 else sync.to_cam1(t), uv))
    return out


def _window_slopes(t: np.ndarray, X: np.ndarray, i0: np.ndarray, i1: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """点の番号 [i0, i1) の窓ごとに、時刻の 1 次式を当てはめた (時刻の平均, 傾き = 速度 (N, 3), 点の数)。

    窓の数だけ当てはめを繰り返さずに、累積和でまとめて計算する。
    """
    tt = (t - t[0])[:, None]

    def cs(a: np.ndarray) -> np.ndarray:
        return np.vstack([np.zeros((1, a.shape[1])), np.cumsum(a, axis=0)])

    St, Stt, Sx, Stx = cs(tt), cs(tt * tt), cs(X), cs(tt * X)
    n = np.maximum(i1 - i0, 1)[:, None].astype(float)
    st, stt, sx, stx = St[i1] - St[i0], Stt[i1] - Stt[i0], Sx[i1] - Sx[i0], Stx[i1] - Stx[i0]
    den = stt - st * st / n
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (stx - st * sx / n) / den
    slope[(den <= 1e-12)[:, 0]] = np.nan
    return (st / n)[:, 0] + t[0], slope, (i1 - i0)


def _propagate_velocity(v: np.ndarray, dt: np.ndarray, phys: PhysicsParams, n_steps: int = 8) -> np.ndarray:
    """速度 (N, 3) を、重力と空気抵抗だけで ``dt`` [s] 進めた速度（4 次のルンゲ・クッタ。まとめて計算する）。"""
    g = np.array([0.0, 0.0, phys.g])

    def f(u: np.ndarray) -> np.ndarray:
        return -phys.k * np.linalg.norm(u, axis=1, keepdims=True) * u - g

    h = (dt / n_steps)[:, None]
    v = v.copy()
    for _ in range(n_steps):
        k1 = f(v)
        k2 = f(v + 0.5 * h * k1)
        k3 = f(v + 0.5 * h * k2)
        k4 = f(v + h * k3)
        v = v + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return v


def find_hits_3d(t: np.ndarray, X: np.ndarray, phys: PhysicsParams | None = None, window_s: float = 0.08,
                 min_dv_mps: float = 4.0, rel_dv: float = 0.3, min_interval_s: float = 0.15,
                 gap_s: float = 0.3) -> list[float]:
    """三角測量した 3 次元の点（``RayPairs.points``）から、打球のだいたいの時刻を見つける。

    画像の上では、近づく・遠ざかるシャトルの見かけの加速が打球に見えることがあるが、3 次元では打球だけが速度を
    急に変える。点の間ごとに、前の ``window_s`` の点と後の ``window_s`` の点の速度（1 次式の傾き）を出し、前の速度を
    重力と空気抵抗だけで後の窓の時刻まで進めたものと比べる。食い違いが ``min_dv_mps`` + 速さの ``rel_dv`` 倍を
    超える所を打球とする（打ち返すと向きが変わるので、食い違いは速さと同じくらい大きい。``min_interval_s`` の中では
    いちばん大きい所だけ）。``gap_s`` より長く見えなかった後の最初の点は、その直前に打ったとする。
    時刻は 1〜2 コマの粗さでよい（``solve_session`` が軌道の交わる時刻に直し、打球ではなかった区切りはつなぐ）。
    """
    phys = phys or PhysicsParams()
    n = len(t)
    if n < 2:
        return []
    dt = np.diff(t)
    step = float(np.median(dt))
    w = max(window_s, 3.2 * step)
    run_start = np.r_[0, np.flatnonzero(dt > gap_s) + 1]          # 続けて見えている区間の始まりの番号
    run_of = np.searchsorted(run_start, np.arange(n), side="right") - 1
    first = run_start[run_of]
    last = np.r_[run_start[1:], n][run_of]                        # 区間の終わり（その番号は含まない）
    starts = [float(t[i] - step / 2) for i in run_start]

    i = np.arange(1, n)                                           # 点 i−1 と点 i の間
    # 窓は時間で決めるが、点がまばらな所（低い fps・見落とし）では 3 点入るまで広げる
    b0 = np.maximum(np.minimum(np.searchsorted(t, t[i - 1] - w, side="left"), i - 3), first[i])
    a1 = np.minimum(np.maximum(np.searchsorted(t, t[i] + w, side="right"), i + 3), last[i])
    tb, vb, nb = _window_slopes(t, X, b0, i)
    ta, va, na = _window_slopes(t, X, i, a1)
    ok = (run_of[i] == run_of[i - 1]) & (nb >= 3) & (na >= 3) & np.isfinite(vb).all(axis=1) & np.isfinite(va).all(axis=1)
    if not ok.any():
        return sorted(starts)
    i, tb, vb, ta, va = i[ok], tb[ok], vb[ok], ta[ok], va[ok]
    speed = np.minimum(np.linalg.norm(vb, axis=1), MAX_FIT_SPEED_MPS)
    vb = vb * (speed / np.maximum(np.linalg.norm(vb, axis=1), 1e-9))[:, None]
    pred = _propagate_velocity(vb, ta - tb, phys)
    score = np.linalg.norm(va - pred, axis=1) / (
        min_dv_mps + rel_dv * np.maximum(np.linalg.norm(pred, axis=1), np.linalg.norm(va, axis=1)))
    t_mid = (t[i - 1] + t[i]) / 2
    # 近い候補は、食い違いのいちばん大きい所だけ残す
    keep: list[float] = []
    for j in np.argsort(-score):
        if score[j] <= 1.0:
            break
        if all(abs(t_mid[j] - k) >= min_interval_s for k in keep):
            keep.append(float(t_mid[j]))
    keep = [k for k in keep if all(not (-step <= k - s0 < min_interval_s) for s0 in starts)]
    return sorted(starts + keep)


def _audio_estimates(p0: np.ndarray, t_hit: float, cams: list[Camera], onsets: list[np.ndarray] | None,
                     o: TruthOptions, match_s: float | None = None) -> list[float | None]:
    """打点 ``p0`` から出た音がそれぞれのカメラに届いた時刻に近い打球音から、打った時刻を出す（なければ None）。"""
    match_s = o.audio_match_s if match_s is None else match_s
    out: list[float | None] = []
    for k, cam in enumerate(cams):
        ons = None if onsets is None else onsets[k]
        if ons is None or len(ons) == 0:
            out.append(None)
            continue
        delay = float(np.linalg.norm(cam.center - p0)) / o.speed_of_sound_mps
        i = int(np.argmin(np.abs(ons - (t_hit + delay))))
        out.append(float(ons[i] - delay) if abs(ons[i] - (t_hit + delay)) <= match_s else None)
    return out


def solve_session(dets: list[pd.DataFrame], cams: list[Camera], sync: SyncModel, hit_times: list[float],
                  phys: PhysicsParams, o: TruthOptions | None = None,
                  onsets: list[list[float] | None] | None = None, sync_sigma_s: float = 0.0,
                  progress: Callable[[str], None] | None = None) -> tuple[list[StereoShot], dict[str, Any]]:
    """打球の時刻で分けた飛行ごとに 2 台へ一緒に当てはめ、打った時刻を直し、不確かさを出す。

    Args:
        dets: 2 台の検出結果（列 ``timestamp, x_px, y_px, visible``。時刻はそれぞれの動画の時刻）
        hit_times: 打球のだいたいの時刻（1 台目の時刻。1 コマほどずれていてよい）
        onsets: カメラごとの打球音の時刻（それぞれの動画の時刻。なければ None）
        sync_sigma_s: 時刻合わせの不確かさ（初速の不確かさに入れる）

    Returns:
        (ショットの列, 情報。``av_offset_ms`` はカメラごとの「打球音から出した時刻 − 軌道の交わる時刻」の中央値)
    """
    o = o or TruthOptions()
    say = progress or (lambda m: None)
    raw = camera_observations(dets, cams, sync, o.rolling_shutter_s)
    ons = None
    if onsets is not None:
        ons = [None if x is None else np.sort(np.asarray(x, float) if k == 0 else sync.to_cam1(np.asarray(x, float)))
               for k, x in enumerate(onsets)]
    times = sorted(float(t) for t in hit_times)

    def window(i: int, ts: list[float], margin: float) -> list[Observations]:
        t0 = ts[i] + margin
        nxt = ts[i + 1] if i + 1 < len(ts) and ts[i + 1] - ts[i] <= o.rally_gap_s else None
        t1 = nxt - margin if nxt is not None else ts[i] + o.tail_s
        return [Observations(cam, t[(t > t0) & (t <= t1)], uv[(t > t0) & (t <= t1)])
                for cam, (t, uv) in zip(cams, raw)]

    def session_vt(ss: list[StereoShot]) -> tuple[float, float, int] | None:
        """長い飛行から 1 本ずつ出した終端速度の (中央値, その不確かさ, 本数)。足りなければ None。"""
        vts = []
        for s in ss:
            if (s.fit is None or sum(s.n_obs) < o.vt_min_obs or s.t_end - s.t_hit < o.vt_min_duration_s
                    or not np.isfinite(s.fit.rms_px) or s.fit.rms_px > o.max_rms_px):
                continue
            g, *_ = fit_flight(s.obs, s.t_hit, phys, o, theta0=s.fit.theta, reject=False, ci=False, estimate_vt=True)
            if g is not None and o.vt_range_mps[0] <= g.terminal_velocity <= o.vt_range_mps[1]:
                vts.append(g.terminal_velocity)
        if len(vts) < o.vt_min_shots:
            return None
        v = np.array(vts)
        sd = 1.4826 * float(np.median(np.abs(v - np.median(v))))
        return float(np.median(v)), float(max(1.2533 * sd / np.sqrt(len(v)), 0.05)), len(v)

    def fit_all(ts: list[float], margin: float, prev: list[StereoShot] | None = None) -> list[StereoShot]:
        out = []
        for i, t in enumerate(ts):
            obs = window(i, ts, margin)
            old = prev[i] if prev is not None else None
            th = old.fit.theta if old is not None and old.fit is not None and abs(old.fit.t_ref - t) < 1e-9 else None
            f, used, n_rej, msg = fit_flight(obs, t, phys, o, theta0=th, ci=False)
            end = max((float(ob.t.max()) for ob in used if len(ob.t)), default=t)
            s = StereoShot(t, end, f, tuple(len(ob.t) for ob in used), n_rejected=n_rej, message=msg, obs=used)
            if old is not None:
                s.hit_method, s.t_hit_sigma, s.t_hit_audio, s.cross_gap_m = (
                    old.hit_method, old.t_hit_sigma, old.t_hit_audio, old.cross_gap_m)
            out.append(s)
        return out

    shots = fit_all(times, 0.01)
    say(f"[進行] 飛行 {len(shots)} 本のうち {sum(s.fit is not None for s in shots)} 本を当てはめました")

    # --- 打球ではなかった区切り（前後で位置も速度もつながっている）を消す
    drop = set()
    for i in range(1, len(shots)):
        a, b = shots[i - 1].fit, shots[i].fit
        if a is None or b is None or times[i] - times[i - 1] > o.rally_gap_s:
            continue
        (pa, va), (pb, vb) = a.evaluate(np.array([times[i]]), phys), b.evaluate(np.array([times[i]]), phys)
        dv, v = float(np.linalg.norm(va[0] - vb[0])), float(np.linalg.norm(vb[0]))
        if np.linalg.norm(pa[0] - pb[0]) < o.merge_gap_m and dv < max(2.0, o.merge_dv_frac * v):
            drop.add(i)
    if drop:
        times = [t for i, t in enumerate(times) if i not in drop]
        shots = fit_all(times, 0.01)
        say(f"[進行] 打球ではなかった区切り {len(drop)} 個をつなぎました")

    # --- 当てはめのできない短い区切り（余分に見つけた打球のすぐ後が本当の打球）は、後ろの打球を消してつなぐ
    short = {i + 1 for i in range(len(shots) - 1)
             if shots[i].fit is None and 0 < times[i + 1] - times[i] < 0.12 and shots[i + 1].fit is not None}
    if short:
        times = [t for i, t in enumerate(times) if i not in short]
        shots = fit_all(times, 0.01)
        drop |= short

    # --- 終端速度を長い飛行から決める（前後の軌道の交わる時刻にも効くので、打った時刻を直す前に）
    vt_info: dict[str, Any] = {"terminal_velocity_mps": phys.terminal_velocity, "source": "config",
                               "sigma_mps": o.terminal_velocity_sigma}

    def update_vt() -> None:
        nonlocal phys
        r = session_vt(shots) if o.terminal_velocity == "session" else None
        if r is not None:
            phys = PhysicsParams(phys.g, r[0], phys.method, phys.rtol, phys.atol)
            vt_info.update({"terminal_velocity_mps": round(r[0], 3), "source": "session", "sigma_mps": round(r[1], 3),
                            "n_flights": r[2]})

    update_vt()
    if vt_info["source"] == "session":
        shots = fit_all(times, 0.01)
        say(f"[進行] 終端速度 {vt_info['terminal_velocity_mps']:.2f} m/s（長い飛行 {vt_info['n_flights']} 本から）")

    # --- 打った時刻を直す（前後の軌道が 3 次元で交わる時刻 → 打球音 → そのまま）
    step = min(float(np.median(np.diff(t))) if len(t) > 1 else 1 / 30 for t, _ in raw)
    av_bias, av_sd = [0.0, 0.0], [o.audio_sigma_s, o.audio_sigma_s]
    rnd = 0
    n_same = 0
    while rnd < o.time_rounds:
        diffs: list[list[float]] = [[], []]
        cross: list[tuple[float, float, float] | None] = [None] * len(shots)
        audio: list[list[float | None]] = [[None, None] for _ in shots]
        # 前の飛行がこの打球の近くまで見えていれば「前の軌道がある」（ラリーの最初のショットにはない）
        has_prev = [i > 0 and shots[i - 1].fit is not None and times[i] - shots[i - 1].t_end <= o.prev_gap_s
                    for i in range(len(shots))]
        for i, s in enumerate(shots):
            if s.fit is None:
                continue
            if has_prev[i]:
                c = crossing_3d(shots[i - 1].fit, s.fit, phys, times[i], o.cross_half_s / (1 if rnd == 0 else 3))
                if c is not None and c[1] <= o.cross_max_gap_m:
                    cross[i] = c
            audio[i] = _audio_estimates(s.fit.p0, times[i], cams, ons, o,
                                        None if has_prev[i] else o.start_audio_match_s)
            if cross[i] is not None:
                for k in range(2):
                    if audio[i][k] is not None:
                        diffs[k].append(audio[i][k] - cross[i][0])
        # スマホごとの「音と映像のずれ」：交わる時刻のあるショットの打球音との差の中央値
        for k in range(2):
            if len(diffs[k]) >= 5:
                d = np.array(diffs[k])
                av_bias[k] = float(np.median(d))
                av_sd[k] = float(max(1.4826 * np.median(np.abs(d - av_bias[k])), 0.001))
        new_times = list(times)
        for i, s in enumerate(shots):
            if s.fit is None:
                continue
            est = [(a - av_bias[k], av_sd[k]) for k, a in enumerate(audio[i]) if a is not None]
            t_audio = sig_audio = None
            if est:
                w = np.array([1 / e[1] ** 2 for e in est])
                t_audio = float(np.sum(w * np.array([e[0] for e in est])) / w.sum())
                spread = float(np.ptp([e[0] for e in est])) / 2 if len(est) > 1 else 0.0
                sig_audio = float(max(1 / np.sqrt(w.sum()), spread))
            s.t_hit_audio = t_audio
            if cross[i] is not None:
                t_c, gap, dv = cross[i]
                sig_c = float(np.hypot(gap, o.cross_floor_m) / dv)
                if sig_audio is None or sig_c <= sig_audio:
                    new_times[i], s.hit_method, s.t_hit_sigma, s.cross_gap_m = t_c, "crossing", sig_c, gap
                    continue
            if t_audio is not None:
                new_times[i], s.hit_method, s.t_hit_sigma = t_audio, "audio", sig_audio
            else:
                # 手がかりがなければ、見つけた時刻のまま（1 コマほどの幅。ラリーの最初は見え始めが遅れるのでもっと広い）
                s.hit_method, s.t_hit_sigma = "track", (step if has_prev[i] else max(step, o.start_track_sigma_s))
        # 直した時刻が重なった 2 つは同じ打球（見つけた区切りが 1 つ余分だった）。当てはめのない方・コマの少ない方を消す
        same = [i for i in range(1, len(new_times)) if new_times[i] - new_times[i - 1] < o.min_hit_interval_s]
        if same:
            i = same[0]
            weak = i - 1 if (shots[i - 1].fit is None or sum(shots[i - 1].n_obs) < sum(shots[i].n_obs)) else i
            times = [t for j, t in enumerate(new_times) if j != weak]
            shots = fit_all(times, 0.01)
            n_same += 1
            if n_same <= len(hit_times):
                continue                     # 同じ回をやり直す
        moved = max((abs(a - b) for a, b in zip(new_times, times)), default=0.0) if not same else 1.0
        if not same:
            times = new_times
            shots = fit_all(times, o.margin_s, shots)
        rnd += 1
        if moved < 2e-4:
            break
    say("[進行] 打った時刻を直しました")

    # --- 最後の当てはめ（信頼区間つき）と不確かさ
    update_vt()
    for i, s in enumerate(shots):
        if s.fit is None:
            continue
        obs = window(i, times, o.margin_s)
        if o.fit_window_s is not None:
            obs = [ob.subset(ob.t <= times[i] + o.fit_window_s) for ob in obs]
            if min(len(ob.t) for ob in obs) < o.min_obs_each or sum(len(ob.t) for ob in obs) < o.min_obs:
                obs = window(i, times, o.margin_s)
        f, used, n_rej, msg = fit_flight(obs, times[i], phys, o, t_ref_sigma=s.t_hit_sigma, theta0=s.fit.theta)
        if f is None:
            s.fit, s.message = None, msg
            continue
        s.fit, s.obs, s.n_rejected, s.n_obs = f, used, n_rej, tuple(len(ob.t) for ob in used)
        s.t_end = max(float(ob.t.max()) for ob in used)
        s.rms_px = tuple(float(np.sqrt(np.mean(_residual_px(f, ob, phys) ** 2))) for ob in used)
        parts = {"fit": (f.speed0_sd_fit or 0.0) * MPS_TO_KMH, "hit_time": (f.speed0_sd_timing or 0.0) * MPS_TO_KMH}
        if o.uncertainty:
            def refit(ph: PhysicsParams, shift: float = 0.0) -> float | None:
                oo = [used[0], Observations(used[1].camera, used[1].t + shift, used[1].uv)]
                g, *_ = fit_flight(oo, times[i], ph, o, theta0=f.theta, reject=False, ci=False)
                return None if g is None else g.speed0 * MPS_TO_KMH

            vt = phys.terminal_velocity
            v = [refit(PhysicsParams(phys.g, vt + d, phys.method, phys.rtol, phys.atol))
                 for d in (-vt_info["sigma_mps"], vt_info["sigma_mps"])]
            if None not in v:
                parts["terminal_velocity"] = abs(v[1] - v[0]) / 2
            if sync_sigma_s > 0:
                v = [refit(phys, d) for d in (-sync_sigma_s, sync_sigma_s)]
                if None not in v:
                    parts["sync"] = abs(v[1] - v[0]) / 2
        s.sigma_parts = {k: float(v) for k, v in parts.items()}
        # 比べる用：素朴な三角測量（いちばん近いコマどうし・時刻で補間）の最初の 2 点の間の速さ
        for name, fn in (("nearest", nearest_points), ("interpolated", triangulated_points)):
            v0 = _first_speed_kmh(*fn(used))
            if v0 is not None:
                s.naive_kmh[name] = v0
    info = {"av_offset_ms": [round(b * 1000, 2) for b in av_bias],
            "av_offset_sd_ms": [round(b * 1000, 2) for b in av_sd], "n_merged": len(drop),
            "terminal_velocity": vt_info}
    return shots, info


# =============================================================================
# 書き出し
# =============================================================================
def speed_at(entry: dict[str, Any], t: float) -> float:
    """正解のショット（``stereo_truth.json`` の 1 件）の軌道の、時刻 ``t`` [s]（1 台目の時刻）の速さ [km/h]。"""
    st = entry["state"]
    _, v = simulate(np.asarray(st["p0_m"]), np.asarray(st["v0_mps"]), st["t_s"], np.array([t]),
                    PhysicsParams(terminal_velocity=st["terminal_velocity_mps"]))
    return float(np.linalg.norm(v[0]) * MPS_TO_KMH)


def shot_entry(i: int, s: StereoShot, sync: SyncModel, o: TruthOptions) -> dict[str, Any]:
    """``stereo_truth.json`` の 1 件。"""
    f = s.fit
    d: dict[str, Any] = {"id": i, "t_hit_s": round(s.t_hit, 4), "t_hit_cam2_s": round(float(sync.to_cam2(s.t_hit)), 4),
                         "ok": s.ok(o), "hit_time_method": s.hit_method, "n_points": list(s.n_obs)}
    if f is None:
        d["note"] = s.message
        return d
    r3 = lambda v: [round(float(x), 4) for x in v]  # noqa: E731
    d.update({
        "t_hit_sigma_s": round(float(s.t_hit_sigma), 5), "kmh": round(s.kmh, 1),
        "sigma_kmh": None if s.sigma_kmh is None else round(s.sigma_kmh, 1),
        "sigma_parts_kmh": {k: round(v, 1) for k, v in s.sigma_parts.items()},
        "sigma_same_time_kmh": None if s.sigma_same_time_kmh is None else round(s.sigma_same_time_kmh, 1),
        "side": side_of(float(f.p0[0])), "hit_point_m": [round(float(x), 2) for x in f.p0],
        "launch_angle_deg": round(float(np.degrees(np.arctan2(f.v0[2], np.linalg.norm(f.v0[:2])))), 1),
        "direction_deg": round(float(np.degrees(np.arctan2(f.v0[1], f.v0[0]))), 1),
        "state": {"t_s": round(s.t_hit, 5), "p0_m": r3(f.p0), "v0_mps": r3(f.v0),
                  "terminal_velocity_mps": f.terminal_velocity},
        "t_end_s": round(s.t_end, 4), "rms_px": [round(v, 2) for v in s.rms_px], "n_rejected": s.n_rejected,
        "t_hit_audio_s": None if s.t_hit_audio is None else round(s.t_hit_audio, 4),
        "cross_gap_cm": None if s.cross_gap_m is None else round(s.cross_gap_m * 100, 1),
        "naive_first_kmh": {k: round(v, 1) for k, v in s.naive_kmh.items()},
    })
    if not d["ok"]:
        d["note"] = "画像とのずれが大きい・ありえない速さ（打球の見落としや別の物の検出が混ざった飛行）"
    return d


def build_truth(shots: list[StereoShot], sync: SyncReport, o: TruthOptions, videos: list[str],
                fps: list[float], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """``stereo_truth.json`` の中身（``agent/checks/real_videos_truth.json`` と同じ考え方の、2 台からの正解）。"""
    return {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "tool": "smash_speed/stereo_truth.py",
        "videos": videos, "fps": [round(float(f), 3) for f in fps],
        "time_base": "t_hit_s・state は 1 台目の動画の時刻。2 台目の時刻 = offset_s + (1 + drift) × 1 台目の時刻",
        "sigma": "1σ（当てはめ・打った時刻・終端速度・時刻合わせの 2 乗和。キャリブレーションの誤差は入っていない）",
        "sync": sync.to_dict(), **(extra or {}),
        "shots": [shot_entry(i + 1, s, sync.model, o) for i, s in enumerate(shots)],
    }


def truth_to_shots(truth: dict[str, Any], phys: PhysicsParams | None = None, classify: bool = True) -> list[Shot]:
    """正解を ``shots.csv`` の形（``Shot``）にする（GUI・ほかの道具で読めるように）。確かでないショットは速さなし。"""
    out = []
    for e in truth["shots"]:
        if "state" not in e or not e.get("ok"):
            out.append(Shot(t_hit_s=e["t_hit_s"], source="stereo", ref=str(e["id"]), note=e.get("note", "")))
            continue
        st, sg = e["state"], e.get("sigma_kmh") or 0.0
        ph = PhysicsParams(terminal_velocity=st["terminal_velocity_mps"])
        tr = simulate_until_ground(np.asarray(st["p0_m"]), np.asarray(st["v0_mps"]), st["t_s"], ph, n_dense=2)
        landed = bool(np.isfinite(tr["t_land"]))
        out.append(Shot(
            t_hit_s=e["t_hit_s"], side=e["side"], speed_kmh=e["kmh"], speed_ci_low_kmh=e["kmh"] - 1.96 * sg,
            speed_ci_high_kmh=e["kmh"] + 1.96 * sg, hit_x_m=st["p0_m"][0], hit_y_m=st["p0_m"][1],
            hit_z_m=st["p0_m"][2], land_x_m=float(tr["pos"][-1, 0]) if landed else None,
            land_y_m=float(tr["pos"][-1, 1]) if landed else None, t_land_s=float(tr["t_land"]) if landed else None,
            v0_x_mps=st["v0_mps"][0], v0_y_mps=st["v0_mps"][1], v0_z_mps=st["v0_mps"][2],
            terminal_velocity_mps=st["terminal_velocity_mps"],
            confidence=float(np.clip(1.0 - 3.92 * sg / max(e["kmh"], 1e-6), 0.0, 1.0)), source="stereo",
            ref=str(e["id"]), note=f"2 台 {e['n_points'][0]}+{e['n_points'][1]} コマ・± {sg:.1f} km/h"))
    if classify:
        from smash_speed.shot_types import classify_shots
        from smash_speed.shots import make_rallies
        make_rallies(out)
        try:
            classify_shots(out, phys)
        except Exception:  # noqa: BLE001  種類が付かなくても正解は書き出す
            pass
    return out


def trajectory_table(truth: dict[str, Any], n: int = 60) -> pd.DataFrame:
    """確かなショットの軌道（打った時刻から飛行の終わりまで ``n`` 点）。列 ``id, t_s, x_m, y_m, z_m, speed_kmh``。"""
    rows = []
    for e in truth["shots"]:
        if "state" not in e or not e.get("ok"):
            continue
        st = e["state"]
        t = np.linspace(st["t_s"], max(e["t_end_s"], st["t_s"] + 1e-3), n)
        pos, vel = simulate(np.asarray(st["p0_m"]), np.asarray(st["v0_mps"]), st["t_s"], t,
                            PhysicsParams(terminal_velocity=st["terminal_velocity_mps"]))
        for tt, p, v in zip(t, pos, vel):
            rows.append({"id": e["id"], "t_s": round(float(tt), 5), "x_m": round(float(p[0]), 4),
                         "y_m": round(float(p[1]), 4), "z_m": round(float(p[2]), 4),
                         "speed_kmh": round(float(np.linalg.norm(v)) * MPS_TO_KMH, 2)})
    return pd.DataFrame(rows, columns=["id", "t_s", "x_m", "y_m", "z_m", "speed_kmh"])


def write_truth(out: Path, truth: dict[str, Any], phys: PhysicsParams | None = None) -> None:
    """``stereo_truth.json``・``shots.csv``・``trajectories.csv`` を書く。"""
    from smash_speed.shots import SHOTS_FILE, write_shots

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / TRUTH_FILE).write_text(json.dumps(truth, ensure_ascii=False, indent=1), encoding="utf-8")
    write_shots(out / SHOTS_FILE, truth_to_shots(truth, phys))
    trajectory_table(truth).to_csv(out / TRAJECTORY_FILE, index=False)


# =============================================================================
# 1 台の解析との答え合わせ
# =============================================================================
def compare_shots(truth: dict[str, Any], shots: list[Shot], camera: int = 1, max_dt_s: float = 0.3
                  ) -> pd.DataFrame:
    """1 台で解析したショット（``shots.csv``）を、2 台からの正解と比べる。

    時刻の差 ``max_dt_s`` 以内で近い順に 1 対 1 で組にする（``agent/truth/README.md`` と同じ数え方）。

    Args:
        camera: 解析したのが 1 台目の動画か 2 台目の動画か（1 | 2）。2 なら時刻を 2 台目の時刻に直して比べる

    Returns:
        正解 1 本につき 1 行。``kmh`` が空なら、1 台の解析にそのショットがない・初速が出ていない
    """
    sm = truth["sync"]["model"]
    sync = SyncModel(sm["offset_s"], sm["drift_ppm"] * 1e-6)
    refs = [e for e in truth["shots"] if e.get("ok")]
    key = "t_hit_s" if camera == 1 else "t_hit_cam2_s"
    cand = sorted(((abs(s.t_hit_s - e[key]), i, j) for i, e in enumerate(refs) for j, s in enumerate(shots)
                   if abs(s.t_hit_s - e[key]) <= max_dt_s))
    pair: dict[int, int] = {}
    used: set[int] = set()
    for _, i, j in cand:
        if i not in pair and j not in used:
            pair[i] = j
            used.add(j)
    rows = []
    for i, e in enumerate(refs):
        row: dict[str, Any] = {"id": e["id"], "t_hit_s": e[key], "true_kmh": e["kmh"],
                               "true_sigma_kmh": e["sigma_kmh"], "side": e["side"]}
        s = shots[pair[i]] if i in pair else None
        if s is not None:
            row["shot_type"] = s.shot_type
            row["dt_hit_ms"] = round((s.t_hit_s - e[key]) * 1000, 1)
        if s is not None and s.speed_kmh is not None:
            t1 = s.t_hit_s if camera == 1 else float(sync.to_cam1(s.t_hit_s))
            same = speed_at(e, t1)
            row.update({"kmh": round(s.speed_kmh, 1), "err_kmh": round(s.speed_kmh - e["kmh"], 1),
                        "err_pct": round(100 * (s.speed_kmh - e["kmh"]) / e["kmh"], 1),
                        "z": None if not e["sigma_kmh"] else round((s.speed_kmh - e["kmh"]) / e["sigma_kmh"], 2),
                        "true_kmh_at_t": round(same, 1) if np.isfinite(same) else None})
            if np.isfinite(same):
                row["err_same_t_pct"] = round(100 * (s.speed_kmh - same) / same, 1)
            if s.hit_point is not None:
                row["hit_point_err_m"] = round(float(np.linalg.norm(s.hit_point - np.asarray(e["hit_point_m"]))), 2)
        rows.append(row)
    cols = ["id", "t_hit_s", "side", "shot_type", "true_kmh", "true_sigma_kmh", "kmh", "err_kmh", "err_pct", "z",
            "dt_hit_ms", "true_kmh_at_t", "err_same_t_pct", "hit_point_err_m"]
    return pd.DataFrame(rows, columns=cols)


def compare_summary(df: pd.DataFrame, n_single: int | None = None) -> dict[str, Any]:
    """``compare_shots`` の表のまとめ（誤差は 1 台の解析 − 正解）。"""
    got = df[df.kmh.notna()]
    e = got.err_pct.astype(float)
    same = got.err_same_t_pct.dropna().astype(float)
    hp = got.hit_point_err_m.dropna().astype(float)
    out: dict[str, Any] = {"n_truth": int(len(df)), "n_compared": int(len(got)), "n_single": n_single}
    if len(got):
        out.update({"mape_pct": round(float(e.abs().mean()), 2), "median_abs_pct": round(float(e.abs().median()), 2),
                    "bias_pct": round(float(e.mean()), 2), "p90_abs_pct": round(float(e.abs().quantile(0.9)), 2),
                    "median_abs_dt_ms": round(float(got.dt_hit_ms.abs().median()), 1)})
    if len(same):
        out["median_abs_same_t_pct"] = round(float(same.abs().median()), 2)
    if len(hp):
        out["median_hit_point_err_m"] = round(float(hp.median()), 2)
    return out


# =============================================================================
# 動画 2 本から
# =============================================================================
@dataclass
class StereoRallyOptions:
    videos: list[Path]
    calibs: list[Path]
    out_dir: Path
    detections: list[Path] | None = None       # 自前の検出の代わりに使う検出 CSV（1 台ずつ）
    tracknet: Path | None = None                # 学習したモデル（あれば両方の動画をこれで検出する）
    capture_fps: list[float | None] | None = None
    start_s: float | None = None                # 解析する区間（1 台目の時刻）
    end_s: float | None = None
    offset_s: float | None = None               # 2 台目の時刻 − 1 台目の時刻（省略時は音から。指定しても映像で細かく合わせる）
    sync_search_s: float = 0.08                 # 映像で時刻を合わせるときに探す幅
    refine_sync: bool = True                    # 映像で時刻を細かく合わせる
    audio_paths: list[Path] | None = None       # 動画の音声の代わりに使う WAV（1 台ずつ）
    use_audio: bool = True
    hit_times: list[float] | None = None        # 打球の時刻を手で渡す（1 台目の時刻。省略時は軌道と打球音から見つける）
    truth: TruthOptions = field(default_factory=TruthOptions)


def _load_audio(k: int, video: Path, meta: Any, o: StereoRallyOptions, out: Path, cfg: dict[str, Any]
                ) -> tuple[np.ndarray, int] | None:
    from smash_speed.video_io import extract_audio, load_wav

    if not o.use_audio:
        return None
    if o.audio_paths:
        return load_wav(o.audio_paths[k])
    if not meta.has_audio:
        return None
    return load_wav(extract_audio(video, out / f"audio{k + 1}.wav", cfg=cfg))


def analyze_stereo_rally(o: StereoRallyOptions, cfg: dict[str, Any], progress: Callable[[str], None] = print
                         ) -> dict[str, Any]:
    """2 本の動画から、すべてのショットの正しい初速と軌道を出して ``<出力先>/`` に書く。``stereo_truth.json`` の中身を返す。"""
    from smash_speed import events
    from smash_speed.calibration.io import load_calibration
    from smash_speed.errors import AnalysisError
    from smash_speed.pipeline import _camera_for_video
    from smash_speed.shot_list import (ShotListOptions, detect_all, load_detections, one_per_frame, shuttle_tracks,
                                       tracknet_detections)
    from smash_speed.stereo_sync import audio_coarse_offset, audio_pairs, fit_audio_sync, sharp_envelope, video_sync
    from smash_speed.video_io import probe

    out = Path(o.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    phys = PhysicsParams.from_config(cfg)
    cap = o.capture_fps or [None, None]
    metas = [probe(v, c, cfg) for v, c in zip(o.videos, cap)]
    cams = [_camera_for_video(load_calibration(c)[0], m) for c, m in zip(o.calibs, metas)]
    ev = cfg.get("events", {})
    warnings: list[str] = []

    # --- 音：大まかなずれと、打球音の候補
    audio = [_load_audio(k, v, m, o, out, cfg) for k, (v, m) in enumerate(zip(o.videos, metas))]
    onsets: list[list[tuple[float, float]] | None] = [None, None]
    for k, a in enumerate(audio):
        if a is not None:
            onsets[k] = [(t * metas[k].time_scale, s) for t, s in events.audio_onsets(
                a[0], a[1], ev.get("audio_highpass_hz", 2000.0), k_mad=ev.get("audio_k_mad", 10.0))]
    audio_info: dict[str, Any] = {}
    pairs: list[tuple[float, float, float]] = []
    sync0 = None if o.offset_s is None else SyncModel(o.offset_s)
    if audio[0] is not None and audio[1] is not None and metas[0].time_scale == metas[1].time_scale == 1.0:
        if audio[0][1] != audio[1][1]:
            raise AnalysisError("2 本の動画の音声のサンプリング周波数が違います。", ["--offset で時刻のずれを指定してください。"])
        rate = audio[0][1]
        e1, e2 = sharp_envelope(audio[0][0], rate), sharp_envelope(audio[1][0], rate)
        coarse, sharp = (o.offset_s, None) if o.offset_s is not None else audio_coarse_offset(e1, e2, rate)
        pairs = audio_pairs(e1, e2, rate, onsets[0] or [], coarse)
        a_model, a_sd, n = fit_audio_sync(pairs)
        audio_info = {"coarse_offset_s": round(float(coarse), 4), "coarse_sharpness": None if sharp is None else round(sharp, 2),
                      "n_sounds": n}
        if sync0 is None:
            sync0 = a_model if n >= 3 else SyncModel(coarse)
            progress(f"[進行] 音から出した時刻のずれ（2 台目 − 1 台目）{sync0.offset_s:.3f} s（音 {n} 個）")
        del e1, e2
    if sync0 is None:
        raise AnalysisError("音声がないので、時刻のずれを決められません。",
                            ["--offset で 2 台目の時刻 − 1 台目の時刻（秒）を指定してください（±0.05 秒ほど合っていれば、"
                             "あとは映像で合わせます）。"])

    # --- 検出（2 台目は時刻をずらした区間）
    dets = []
    for k, (v, m, cam) in enumerate(zip(o.videos, metas, cams)):
        s0 = o.start_s if k == 0 or o.start_s is None else float(sync0.to_cam2(o.start_s)) - 0.5
        s1 = o.end_s if k == 0 or o.end_s is None else float(sync0.to_cam2(o.end_s)) + 0.5
        d = out / f"cam{k + 1}"
        d.mkdir(exist_ok=True)
        progress(f"[進行] {k + 1} 台目の動画でシャトルを探しています")
        if o.detections:
            det = load_detections(o.detections[k], m, s0, s1)
        elif o.tracknet is not None:
            so = ShotListOptions(video=Path(v), calib=Path(o.calibs[k]), out_dir=d, start_s=s0, end_s=s1,
                                 tracknet=o.tracknet)
            det = load_detections(tracknet_detections(so, d, progress), m, s0, s1)
        else:
            det = detect_all(m, cfg, s0, s1, exclusive=False, progress=progress)
        # 外から渡した検出が 1 コマ 1 点（軌道の番号なし）なら、そのまま使う
        if "track_id" in det and det.loc[det.visible > 0, "track_id"].nunique() > 1:
            det, _ = shuttle_tracks(det, cam, cfg)
        else:
            det = one_per_frame(det)
        det.to_csv(d / "detections.csv", index=False)
        if int(det.visible.sum()) < 5:
            raise AnalysisError(f"{k + 1} 台目の動画でシャトルがほとんど見つかりませんでした。",
                                ["コートとシャトルが写っている動画か、区間が正しいかを確かめてください。"])
        dets.append(det)

    # --- 映像で時刻を細かく合わせる
    if o.refine_sync:
        rep = video_sync(dets, cams, sync0, o.sync_search_s)
        if rep.method == "none":
            rep = SyncReport(sync0, "audio" if o.offset_s is None else "manual", sigma_s=0.002, warnings=rep.warnings)
    else:
        rep = SyncReport(sync0, "audio" if o.offset_s is None else "manual", sigma_s=0.002)
    sync = rep.model
    progress(f"[進行] 時刻のずれ（2 台目 − 1 台目）{sync.offset_s:.4f} s、時計の進み方の違い {sync.drift * 1e6:.0f} ppm"
             + (f"、2 台の視線のすき間 {rep.gap_cm:.1f} cm" if rep.gap_cm is not None else ""))

    # --- 打球を見つける（三角測量した 3 次元の点の、速度が急に変わる所）
    if o.hit_times is not None:
        hit_times = sorted(o.hit_times)
    else:
        hit_times = find_hits_3d(*RayPairs(dets, cams).points(sync), phys)
    progress(f"[進行] 打球 {len(hit_times)} 本")
    if not hit_times:
        raise AnalysisError("打球が見つかりませんでした。", ["区間・検出を確かめてください。"])

    ons_t = [None if x is None else [t for t, _ in x] for x in onsets]
    shots, info = solve_session(dets, cams, sync, hit_times, phys, o.truth, ons_t, rep.sigma_s, progress)

    # --- 確かめ：音から出したずれ（打点から 2 台への音の遅れを引いた後）と、映像のずれの差
    if pairs:
        good = [s for s in shots if s.fit is not None and s.ok(o.truth)]
        dd = np.full(len(pairs), np.nan)
        for i, (t1, _, _) in enumerate(pairs):
            near = [s for s in good if abs(t1 - s.t_hit - np.linalg.norm(cams[0].center - s.fit.p0)
                                           / o.truth.speed_of_sound_mps) <= o.truth.audio_match_s]
            if near:
                p0 = near[0].fit.p0
                dd[i] = (np.linalg.norm(cams[1].center - p0) - np.linalg.norm(cams[0].center - p0)) / o.truth.speed_of_sound_mps
        a_model, a_sd, n = fit_audio_sync(pairs, dd)
        if n:
            tm = float(np.mean([p[0] for p, d in zip(pairs, dd) if np.isfinite(d)]))
            diff = float(a_model.to_cam2(tm) - sync.to_cam2(tm))
            audio_info.update({"offset_s": round(a_model.offset_s, 5), "drift_ppm": round(a_model.drift * 1e6, 1),
                               "n_hit_sounds": n, "residual_sd_ms": None if not np.isfinite(a_sd) else round(a_sd * 1000, 3),
                               "audio_minus_video_ms": round(diff * 1000, 2)})
            if abs(diff) > 0.02:
                warnings.append(f"音から出した時刻のずれと映像から出したずれが {diff * 1000:.0f} ms 違います。"
                                "どちらかのスマホの音と映像がずれているか、時刻合わせが外れています。")
    rep.audio = audio_info
    rep.warnings += warnings
    truth = build_truth(shots, rep, o.truth, [Path(v).name for v in o.videos], [m.capture_fps for m in metas],
                        {"calibs": [Path(c).name for c in o.calibs], "av_offset_ms": info["av_offset_ms"],
                         "n_hits_merged": info["n_merged"], "terminal_velocity": info["terminal_velocity"]})
    write_truth(out, truth, phys)
    n_ok = sum(e.get("ok", False) for e in truth["shots"])
    progress(f"[進行] 完了：{len(shots)} 本のうち、正しい初速が出たもの {n_ok} 本")
    for w in rep.warnings:
        progress(f"[注意] {w}")
    return truth


def print_truth(truth: dict[str, Any], top: int = 12) -> None:
    """結果を画面に出す（時刻合わせの確かめと、ショットの表）。"""
    sy = truth["sync"]
    m = sy["model"]
    print(f"時刻のずれ（2 台目 − 1 台目）{m['offset_s']:.4f} s・時計の進み方の違い {m['drift_ppm']:.0f} ppm"
          f"（{sy['method']}、± {sy['sigma_s'] * 1000:.2f} ms）")
    if sy.get("gap_cm") is not None:
        print(f"2 台の視線のすき間 {sy['gap_cm']:.1f} cm（中央値。大きいとキャリブレーションが合っていない）")
    a = sy.get("audio") or {}
    if a.get("audio_minus_video_ms") is not None:
        print(f"音から出したずれ − 映像から出したずれ {a['audio_minus_video_ms']:+.2f} ms"
              f"（打球音 {a['n_hit_sounds']} 個、ばらつき {a.get('residual_sd_ms')} ms）")
    ok = [e for e in truth["shots"] if e.get("ok")]
    print(f"ショット {len(truth['shots'])} 本のうち、正しい初速が出たもの {len(ok)} 本")
    for e in sorted(ok, key=lambda e: -e["kmh"])[:top]:
        print(f"  #{e['id']:<3d} {e['t_hit_s']:8.3f} s  {e['kmh']:6.1f} ± {e['sigma_kmh'] or 0:4.1f} km/h  "
              f"打点 ({e['hit_point_m'][0]:+.2f}, {e['hit_point_m'][1]:+.2f}, {e['hit_point_m'][2]:.2f}) m  "
              f"打ち出し角 {e['launch_angle_deg']:+.0f}°  コマ {e['n_points'][0]}+{e['n_points'][1]}")
