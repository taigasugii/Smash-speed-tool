"""ラリー全体をまとめて当てはめる（``shot_fit.fit_shots`` のショットごとの当てはめの後の仕上げ）。

カメラ 1 台では、短いショット・カメラへ向かうショットの奥行きが 1 本だけでは決まらない。ラリーの中では、
ショット k の軌道の終わりはショット k + 1 の打点と同じ点・同じ時刻なので、ラリーのすべてのショットを 1 つの
最小 2 乗問題にして、となりのショットに奥行きと打った時刻を決めてもらう。

決める量は、ショットごとの「打った時刻 τ・打点 P（3 次元）・打ち出しの速度 V（3 次元）」と、動画全体で 1 つの
シャトルの終端速度 v_t。ずれ（残差）は次のとおり。

- 画像の上のずれ [px]：ショット k の観測と、(τ_k, P_k, V_k) から運動方程式で進めた軌道の投影。
- つなぎ目：ショット k − 1 の軌道の時刻 τ_k の位置が P_k と同じ（``link_sigma_m``）。
- 打つ人の足元：P_k の水平の位置が足元の近く（ラケットの届く範囲。``feet_sigma_m``）。
- 打った時刻：打球音があれば「音の時刻 − 打点からカメラまでの音の遅れ」、なければ軌道から決めた時刻の近く。
  前のショットの最後の観測より後・自分の最初の観測より前。
- 打点の高さ・コートの中・打った側・初速の上限（``physics.MAX_FIT_SPEED_MPS``）。

間違った打球（見落とし・余分な打球）・別の物の軌道が 1 本あってもラリー全体が崩れないよう、つなぎ目と軌道から
決めた時刻は外れ値に強い損失（Huber）のままにし、解いた後に

- つなぎ目のすき間が ``max_link_gap_m`` を超えた
- つないだことでショットの画像の上のずれが大きくなった（``max_rms_ratio``）
- 打点が足元から ``max_feet_gap_m`` より離れた

ものを切って解き直す（``max_rounds`` 回まで）。当てはめる前にも、前後の軌道が画像の上で交わらないつなぎ目は
切っておく。前後どちらともつながらなかったショットは、ショットごとの当てはめの結果のまま。

終端速度は、長く飛んだショット（クリア・ロブ。``long_flight_s``・``long_min_obs``）だけが共通の v_t を使う形で
一緒に解き、ラリーごとの推定値と不確かさを出す（``estimate_vt=True``）。動画全体の値は ``combine_vt`` で
まとめ、不確かさが大きい・長いショットが少ないときは設定の値のままにする。

軌道の積分は、ラリーの全ショットと、ヤコビアン用に速度・終端速度を少し動かしたものを、決まった刻みの
RK4 でまとめて進める（``integrate``。``physics.simulate`` との差は 0.1 mm 以下）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from smash_speed.camera import Camera
from smash_speed.geometry import FlightPlane
from smash_speed.physics import MAX_FIT_SPEED_MPS, SPEED_PENALTY_SIGMA_MPS, FitOptions, FitResult, PhysicsParams

_DV = 1e-3        # ヤコビアン用に速度を動かす大きさ [m/s]
_DLV = 1e-4       # 同・log(v_t)
_EPS_M = 1e-4     # 投影のヤコビアン用に位置を動かす大きさ [m]
_BAD_PX = 1e3     # 計算できなかった残差の罰（``physics._residuals`` と同じ）
_N_PRIOR = 14     # ショットごとの、画像の上のずれ以外の残差の数


@dataclass
class JointOptions:
    """ラリー全体の当てはめの設定。"""

    link_sigma_m: float = 0.15        # つなぎ目（前のショットの終わりと打点）のすき間の不確かさ [m]
    max_link_gap_m: float = 0.6       # 解いた後のすき間がこれを超えたつなぎ目は切る [m]
    max_cross_frac: float = 0.006     # 当てはめる前に、前後の軌道が画像の上でこれ（画像の幅の割合）より近づかない
                                      # つなぎ目は切る（1920 px で 12 px。``shot_fit.refine_hit_times`` は 6 px）
    cross_window_s: float = 0.1       # そのとき近づく所を探す、打った時刻の前後の幅 [s]
    feet_sigma_m: float = 0.5         # 打つ人の足元から打点までの水平の距離の不確かさ [m]
    max_feet_gap_m: float = 2.5       # 解いた後の打点が足元からこれより離れたら、その足元は使わない [m]
    audio_sigma_s: float = 0.003      # 打球音から決めた打った時刻の不確かさ [s]
    track_sigma_s: float | None = None  # 軌道から決めた打った時刻の不確かさ [s]（None ならコマの間隔）
    order_sigma_s: float = 0.003      # 打った時刻が前後の観測の順を越えたときの罰の幅 [s]
    hit_z_m: tuple[float, float] = (0.0, 3.6)   # 打点の高さの範囲 [m]
    max_rms_ratio: float = 1.5        # つないだ後の画像の上のずれが「ショットごとのずれ × これ + rms_slack_px」を
    rms_slack_px: float = 1.0         # 超えたショットは、つなぎ目を切る（最後まで超えていれば元の当てはめに戻す）
    max_rounds: int = 4               # 切って解き直す回数の上限
    n_splits: int = 2                 # 直した打った時刻で観測を分け直して解く回数
    max_nfev: int = 60                # 1 回の最適化の評価の回数の上限
    step_fast_s: float = 0.002        # 積分の刻み [s]（打ってから ``fast_s`` まで。速い間は空気抵抗が大きい）
    fast_s: float = 0.3
    step_slow_s: float = 0.008        # 積分の刻み [s]（その後）
    max_flight_s: float = 4.5         # これより後の観測は使わない [s]
    # --- 終端速度 ---
    estimate_vt: bool = True          # 長く飛んだショットから終端速度を一緒に求める
    long_flight_s: float = 0.6        # 長く飛んだショットとみなす飛行の時間 [s]
    long_min_obs: int = 12            # 同・見えたコマの数
    vt_prior_sigma: float = 0.15      # 設定の値からの log(v_t) の不確かさ（弱い目安。±1 m/s）
    vt_min_flights: int = 3           # 動画全体で、長く飛んだショットがこれより少なければ設定の値のまま
    vt_max_sigma_mps: float = 0.15    # 動画全体の推定の不確かさがこれより大きければ設定の値のまま [m/s]
    vt_range_mps: tuple[float, float] = (5.0, 8.5)  # これを外れた推定は使わない（羽根・ナイロンのシャトルは 6〜7.5）


@dataclass
class VtEstimate:
    """終端速度の推定。``value``・``sigma`` [m/s]、``n_flights`` は使った長いショットの数。"""

    value: float
    sigma: float
    n_flights: int

    def to_dict(self) -> dict[str, Any]:
        return {"value_mps": round(self.value, 3), "sigma_mps": round(self.sigma, 3), "n_flights": self.n_flights}


@dataclass
class JointResult:
    """``joint_fit`` の結果。

    Attributes:
        fits: 入力と同じ順の ``ShotFit``（まとめて当てはめたショットは置き換えたもの）
        joint: ショットごとに、まとめた当てはめの結果を使ったか
        links: つなぎ目 k（ショット k − 1 → k。k = 0 は使わない）を使ったか
        gaps_m: つなぎ目のすき間 [m]（使わなかった所は NaN）
        cut: 切ったつなぎ目・足元の記録（``(種類, ショットの番号, 理由)``）
        vt: 終端速度の推定（求めなかった・長いショットがなければ None）
        terminal_velocity: 最後の当てはめに使った終端速度 [m/s]
        n_solves: 最適化を解いた回数
    """

    fits: list[Any]
    joint: list[bool]
    links: list[bool]
    gaps_m: list[float]
    cut: list[tuple[str, int, str]] = field(default_factory=list)
    vt: VtEstimate | None = None
    terminal_velocity: float = 0.0
    n_solves: int = 0


# =============================================================================
# 積分
# =============================================================================
def time_grid(s_min: float, s_max: float, o: JointOptions) -> tuple[np.ndarray, int]:
    """打った時刻からの時間 [s] の刻み（0 を含む。0 の番号も返す）。同じ時間の刻みは、端によらず同じ。"""
    n_neg = int(np.ceil(max(-s_min, 0.0) / o.step_fast_s)) + 1
    n_fast = int(round(o.fast_s / o.step_fast_s))
    n_slow = int(np.ceil(max(s_max - o.fast_s, 0.0) / o.step_slow_s)) + 1
    grid = np.concatenate([-o.step_fast_s * np.arange(n_neg, 0, -1), o.step_fast_s * np.arange(n_fast + 1),
                           o.fast_s + o.step_slow_s * np.arange(1, n_slow + 1)])
    return grid, n_neg


def _rhs(y: np.ndarray, k: np.ndarray, g: float) -> np.ndarray:
    v = y[:, 3:]
    out = np.empty_like(y)
    out[:, :3] = v
    out[:, 3:] = (-k * np.sqrt(np.einsum("ij,ij->i", v, v)))[:, None] * v
    out[:, 5] -= g
    return out


def integrate(v0: np.ndarray, vt: np.ndarray, grid: np.ndarray, i0: int, g: float) -> np.ndarray:
    """原点を速度 ``v0`` (B, 3) [m/s] で出た B 本の軌道を、``grid`` [s] の時刻で進める（RK4。0 より前は逆向き）。

    Returns:
        (len(grid), B, 6) の位置 [m]・速度 [m/s]。前へ戻して速さが発散した所は NaN・inf
    """
    v0 = np.asarray(v0, float)
    k = g / np.asarray(vt, float) ** 2
    out = np.empty((len(grid), len(v0), 6))
    y0 = np.concatenate([np.zeros_like(v0), v0], axis=1)
    out[i0] = y0
    with np.errstate(over="ignore", invalid="ignore"):
        for step in (1, -1):
            y = y0
            i = i0
            while 0 <= i + step < len(grid):
                h = grid[i + step] - grid[i]
                k1 = _rhs(y, k, g)
                k2 = _rhs(y + 0.5 * h * k1, k, g)
                k3 = _rhs(y + 0.5 * h * k2, k, g)
                k4 = _rhs(y + h * k3, k, g)
                y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                i += step
                out[i] = y
    return out


def interpolate(grid: np.ndarray, y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``integrate`` の結果 ``y`` (len(grid), B, 6) を、時間 ``s`` (M,) [s] で補間する（3 次のエルミート補間）。

    Returns:
        (位置 (M, B, 3), 速度 (M, B, 3))。速度は補間した位置の時間微分。刻みの外は NaN
    """
    s = np.asarray(s, float)
    i = np.clip(np.searchsorted(grid, s, side="right") - 1, 0, len(grid) - 2)
    h = (grid[i + 1] - grid[i])[:, None, None]
    u = ((s - grid[i]) / (grid[i + 1] - grid[i]))[:, None, None]
    p0, p1, v0, v1 = y[i, :, :3], y[i + 1, :, :3], y[i, :, 3:], y[i + 1, :, 3:]
    u2, u3 = u * u, u * u * u
    pos = (2 * u3 - 3 * u2 + 1) * p0 + (u3 - 2 * u2 + u) * h * v0 + (-2 * u3 + 3 * u2) * p1 + (u3 - u2) * h * v1
    vel = (6 * u2 - 6 * u) / h * (p0 - p1) + (3 * u2 - 4 * u + 1) * v0 + (3 * u2 - 2 * u) * v1
    bad = (s < grid[0] - 1e-12) | (s > grid[-1] + 1e-12)
    if bad.any():
        pos[bad], vel[bad] = np.nan, np.nan
    return pos, vel


# =============================================================================
# 最小 2 乗問題
# =============================================================================
@dataclass
class _Shot:
    """問題の中のショット 1 本分のデータ。"""

    t: np.ndarray                     # 観測の時刻
    uv: np.ndarray                    # 観測の画像の位置
    side: str
    link: bool                        # 前のショット（問題の中の 1 つ前）とつなぐか
    foot: np.ndarray | None           # 打つ人の足元
    t_audio: float | None             # 打球音の時刻
    t_track: float                    # 軌道から決めた打った時刻（打球音がないときの目安）
    t_lo: float                       # 打った時刻の下限（前のショットの最後の観測）
    t_hi: float                       # 同・上限（自分の最初の観測）
    vt_free: bool = False             # 共通の終端速度を使うか


class _Problem:
    """ラリー（つながったショットの組）の残差とヤコビアン。パラメータはショットごとに (τ, P, V) の 7 個と、
    あれば最後に log(v_t)。"""

    def __init__(self, shots: list[_Shot], camera: Camera, phys: PhysicsParams, o: JointOptions, weight_px: float,
                 huber_px: float | None, track_sigma_s: float, speed_of_sound: float, estimate_vt: bool,
                 max_speed: float | None):
        self.shots, self.camera, self.g, self.vt0, self.o = shots, camera, phys.g, phys.terminal_velocity, o
        self.w, self.huber = weight_px, huber_px
        self.track_sigma, self.c_sound, self.max_speed = track_sigma_s, speed_of_sound, max_speed
        self.n = len(shots)
        self.est = estimate_vt and any(s.vt_free for s in shots)
        self.n_par = 7 * self.n + int(self.est)
        self.nv = 4 + int(self.est)
        self.rows = np.cumsum([0] + [2 * len(s.t) + _N_PRIOR for s in shots])
        self.n_res = int(self.rows[-1]) + int(self.est)
        self.center = camera.center
        self._key: bytes | None = None
        self._val: tuple[np.ndarray, np.ndarray, dict[str, Any]] | None = None

    # ------------------------------------------------------------------ 残差の形
    def _quad(self, r: np.ndarray, jac: np.ndarray) -> None:
        """Huber 損失をかけても 2 乗のまま効くように残差 ``r`` を引き伸ばす（``physics._Model._quadratic``）。
        ``jac`` の同じ行にも傾きを掛ける。どちらも書き換える。"""
        d = self.huber
        if d is None:
            return
        a = np.abs(r)
        big = a > d
        if big.any():
            jac[big] *= (a[big] / d)[:, None]
            r[big] = np.sign(r[big]) * (a[big] * a[big] / (2 * d) + d / 2)

    def unpack(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        q = x[:7 * self.n].reshape(self.n, 7)
        vt = np.full(self.n, self.vt0)
        if self.est:
            vt[[s.vt_free for s in self.shots]] = float(np.exp(x[-1]))
        return q[:, 0], q[:, 1:4], q[:, 4:7], vt

    def evaluate(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """残差・ヤコビアンと、ショットごとの手がかり（画像の上のずれ・つなぎ目のすき間・足元までの距離）。"""
        key = x.tobytes()
        if key == self._key and self._val is not None:
            return self._val
        n, nv, o, w = self.n, self.nv, self.o, self.w
        tau, P, V, vt = self.unpack(x)
        vb, vtb = np.repeat(V, nv, axis=0), np.repeat(vt, nv)
        for j in range(3):
            vb[j + 1::nv, j] += _DV
        if self.est:
            vtb[4::nv] *= np.exp(_DLV)
        # ショット k の軌道を見る時間：自分の観測と、次のショットの打った時刻（つなぎ目）
        times = []
        for k, s in enumerate(self.shots):
            t = s.t
            if k + 1 < n and self.shots[k + 1].link:
                t = np.r_[t, tau[k + 1]]
            times.append(t - tau[k])
        s_min = min(float(t.min()) if len(t) else 0.0 for t in times)
        s_max = max(float(t.max()) if len(t) else 0.0 for t in times)
        grid, i0 = time_grid(max(s_min, -0.1), min(s_max, o.max_flight_s), o)
        y = integrate(vb, vtb, grid, i0, self.g)

        r = np.zeros(self.n_res)
        jac = np.zeros((self.n_res, self.n_par))
        rel, vel = [], []
        for k in range(n):
            p, v = interpolate(grid, y[:, k * nv:(k + 1) * nv], times[k])
            rel.append(p)
            vel.append(v[:, 0])
        # 投影と、その位置についての傾き（すべてのショットをまとめて）
        m_obs = [len(s.t) for s in self.shots]
        X = np.concatenate([P[k] + rel[k][:m_obs[k], 0] for k in range(n)])
        uv = self.camera.project(X)
        A = np.stack([(self.camera.project(X + _EPS_M * e) - uv) / _EPS_M for e in np.eye(3)], axis=2)
        rms, gaps, feet_gap = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
        at = 0
        for k, s in enumerate(self.shots):
            m, c, row = m_obs[k], 7 * k, int(self.rows[k])
            pr, vk = rel[k], vel[k]
            dV = (pr[:, 1:4] - pr[:, :1]) / _DV            # (M, 3 つの速度の成分, 3)
            dlv = (pr[:, 4] - pr[:, 0]) / _DLV if (self.est and s.vt_free) else None
            # --- 画像の上のずれ
            a = A[at:at + m]
            e = uv[at:at + m] - s.uv
            ok = np.isfinite(e).all(axis=1) & np.isfinite(a).all(axis=(1, 2))
            a = np.where(ok[:, None, None], a, 0.0)
            rr = slice(row, row + 2 * m)
            r[rr] = np.where(ok[:, None], e, _BAD_PX).ravel()
            if ok.any():
                rms[k] = float(np.sqrt(np.mean(e[ok] ** 2)))
            jac[rr, c] = -np.einsum("mij,mj->mi", a, np.nan_to_num(vk[:m])).ravel()
            jac[rr, c + 1:c + 4] = a.reshape(2 * m, 3)
            jac[rr, c + 4:c + 7] = np.einsum("mij,mvj->miv", a, np.nan_to_num(dV[:m])).reshape(2 * m, 3)
            if dlv is not None:
                jac[rr, -1] = np.einsum("mij,mj->mi", a, np.nan_to_num(dlv[:m])).ravel()
            at += m
            row += 2 * m
            # --- つなぎ目（前のショットの終わり = 打点）。外れ値に強い損失のまま
            if s.link:
                wl = w / o.link_sigma_m
                pe, ve = rel[k - 1][-1], vel[k - 1][-1]
                gap = P[k - 1] + pe[0] - P[k]
                if np.isfinite(gap).all() and np.isfinite(ve).all():
                    gaps[k] = float(np.linalg.norm(gap))
                    lr, cp = slice(row, row + 3), 7 * (k - 1)
                    r[lr] = wl * gap
                    jac[lr, cp] = -wl * ve
                    jac[lr, cp + 1:cp + 4] = wl * np.eye(3)
                    jac[lr, cp + 4:cp + 7] = wl * ((pe[1:4] - pe[:1]) / _DV).T
                    if self.est and self.shots[k - 1].vt_free:
                        jac[lr, -1] = wl * (pe[4] - pe[0]) / _DLV
                    jac[lr, c] = wl * ve
                    jac[lr, c + 1:c + 4] = -wl * np.eye(3)
                else:
                    r[row:row + 3] = _BAD_PX
                    gaps[k] = np.inf
            row += 3
            # ここから下の目安・範囲の罰は 2 乗のまま効かせる
            q0 = row
            # --- 打つ人の足元
            if s.foot is not None:
                wf = w / o.feet_sigma_m
                d = P[k, :2] - s.foot
                feet_gap[k] = float(np.hypot(*d))
                r[row:row + 2] = wf * d
                jac[row, c + 1], jac[row + 1, c + 2] = wf, wf
            row += 2
            # --- 打った時刻
            if s.t_audio is not None:
                wa = w / o.audio_sigma_s
                to_cam = P[k] - self.center
                dist = float(np.linalg.norm(to_cam))
                r[row] = wa * (tau[k] - s.t_audio + dist / self.c_sound)
                jac[row, c] = wa
                jac[row, c + 1:c + 4] = wa * to_cam / max(dist, 1e-9) / self.c_sound
            row += 1
            for sign, lim in ((-1.0, s.t_lo), (1.0, s.t_hi)):
                over = sign * (tau[k] - lim)
                if np.isfinite(lim) and over > 0:
                    r[row] = w * over / o.order_sigma_s
                    jac[row, c] = sign * w / o.order_sigma_s
                row += 1
            # --- 打点の高さ・コートの中・打った側
            sx = {"h": 1.0, "o": -1.0}.get(s.side, 0.0)
            x_lo, x_hi = (-8.0, 8.0) if sx == 0 else ((-8.0, 1.0) if sx > 0 else (-1.0, 8.0))
            for ax, lo, hi, sig in ((2, o.hit_z_m[0], o.hit_z_m[1], 0.2), (0, x_lo, x_hi, 0.5), (1, -4.5, 4.5, 0.5)):
                val = P[k, ax]
                if val < lo or val > hi:
                    r[row] = w * (val - (lo if val < lo else hi)) / sig
                    jac[row, c + 1 + ax] = w / sig
                row += 1
            # --- 初速の上限・相手側へ飛ぶ
            speed = float(np.linalg.norm(V[k]))
            if self.max_speed is not None and speed > self.max_speed:
                ws = np.sqrt(2.0 * max(m, 1)) / SPEED_PENALTY_SIGMA_MPS
                r[row] = ws * (speed - self.max_speed)
                jac[row, c + 4:c + 7] = ws * V[k] / speed
            row += 1
            if sx != 0 and sx * V[k, 0] < 0:
                r[row] = w * sx * V[k, 0]
                jac[row, c + 4] = w * sx
            row += 1
            self._quad(r[q0:row], jac[q0:row])
            # --- 軌道から決めた打った時刻（打球音がないとき。外れ値に強い損失のまま）
            if s.t_audio is None:
                r[row] = w * (tau[k] - s.t_track) / self.track_sigma
                jac[row, c] = w / self.track_sigma
            row += 1
            assert row == self.rows[k + 1]
        if self.est:
            r[-1] = w * (x[-1] - np.log(self.vt0)) / o.vt_prior_sigma
            jac[-1, -1] = w / o.vt_prior_sigma
        self._key, self._val = key, (r, jac, {"rms": rms, "gaps": gaps, "feet_gap": feet_gap})
        return self._val

    def solve(self, x0: np.ndarray) -> Any:
        return least_squares(lambda x: self.evaluate(x)[0], x0, jac=lambda x: self.evaluate(x)[1],
                             loss="huber" if self.huber is not None else "linear",
                             f_scale=self.huber if self.huber is not None else 1.0, x_scale="jac",
                             max_nfev=self.o.max_nfev, ftol=1e-6, xtol=1e-6)


# =============================================================================
# ラリーの当てはめ
# =============================================================================
def crossing_gap_px(prev: FitResult, nxt: FitResult, camera: Camera, phys: PhysicsParams, t_hit: float,
                    window_s: float) -> float:
    """前のショットを先へ・次のショットを前へ進めた軌道が、打った時刻の前後で画像の上でいちばん近づく距離 [px]。"""
    from smash_speed.shot_fit import crossing_time

    c = crossing_time(prev, nxt, camera, phys, t_hit - window_s, t_hit + window_s, step_s=2e-3)
    return np.inf if c is None else c[1]


def _is_long(sf: Any, o: JointOptions) -> bool:
    return sf.duration >= o.long_flight_s and sf.n_obs >= o.long_min_obs


def joint_fit(fits: list[Any], det: pd.DataFrame, camera: Camera, phys: PhysicsParams,
              opts: FitOptions | None = None, o: JointOptions | None = None, feet: list[Any] | None = None,
              t_audio: list[float | None] | None = None, margin_s: float = 0.01, t_last: float | None = None,
              speed_of_sound_mps: float = 343.0, max_rms_px: float = np.inf, estimate_vt: bool | None = None,
              links: list[bool] | None = None, base: list[Any] | None = None) -> JointResult:
    """ショットごとの当てはめ ``fits``（``shot_fit.fit_shots`` の結果。時刻の順の 1 ラリー分）から始めて、
    ラリー全体をまとめて当てはめ直す。

    Args:
        det: ラリーの検出結果（``fit_shots`` に渡したもの）
        phys: 運動方程式の設定（終端速度は、求めないとき・長くないショットに使う値）
        opts: 当てはめの設定（損失・信頼区間の出し方）
        feet: ショットごとの打つ人の足元 [m]（なければ None）
        t_audio: ショットごとの打球音の時刻（音の遅れを引く前。なければ None）
        max_rms_px: ショットごとの当てはめのずれがこれを超えたショット（別の物の軌道）は、まとめない
        estimate_vt: 終端速度を一緒に求めるか（None なら ``o.estimate_vt``）。求めたときの ``fits`` は、長いショットだけが
            推定した値を使っているので、動画全体の値を決めてから ``estimate_vt=False`` で解き直す
        links: 前の ``joint_fit`` の ``JointResult.links``（切ったつなぎ目を引き継ぐ）
        base: 元のショットごとの当てはめ（ずれを比べる相手・戻す先。None なら ``fits``）

    Returns:
        ``JointResult``
    """
    from smash_speed.shot_fit import ShotFit, _state_at, split_observations

    o = o or JointOptions()
    opts = opts or FitOptions(refine_plane=True)
    base = base if base is not None else fits
    est = o.estimate_vt if estimate_vt is None else estimate_vt
    n = len(fits)
    feet = list(feet) if feet is not None else [None] * n
    feet = [None if f is None or not np.isfinite(np.asarray(f, float)[:2]).all() else np.asarray(f, float)[:2]
            for f in feet]
    t_audio = list(t_audio) if t_audio is not None else [None] * n
    out = JointResult(list(fits), [False] * n, [False] * n, [np.nan] * n, terminal_velocity=phys.terminal_velocity)
    vis = det[(det.visible > 0) & det.x_px.notna() & det.y_px.notna()]
    vis_t = vis.timestamp.to_numpy(float)
    if t_last is None:
        t_last = float(vis_t.max()) if len(vis_t) else 0.0
    dt_frame = float(np.median(np.diff(np.unique(det.timestamp.to_numpy(float))))) if len(det) > 1 else 1 / 30
    track_sigma = o.track_sigma_s if o.track_sigma_s is not None else max(dt_frame, 0.004)

    # はじめの値：打った時刻の位置・速度
    state: dict[int, np.ndarray] = {}
    for i, sf in enumerate(fits):
        if sf.fit is None or not np.isfinite(base[i].fit.rms_px if base[i].fit is not None else np.nan):
            continue
        if not base[i].fit.rms_px <= max_rms_px:
            continue
        st = _state_at(sf.fit, sf.t_hit, phys)
        if st is not None and np.linalg.norm(st[1]) > 0:
            state[i] = np.r_[sf.t_hit, st[0], st[1]]
    ref_rms = {i: float(base[i].fit.rms_px) for i in state}
    use_link = [False] * n
    for i in range(1, n):
        if i in state and i - 1 in state and (links is None or links[i]):
            if links is not None:
                use_link[i] = True
                continue
            gap = crossing_gap_px(fits[i - 1].fit, fits[i].fit, camera, phys, fits[i].t_hit, o.cross_window_s)
            lim = max(o.max_cross_frac * camera.width, 3 * (ref_rms[i - 1] + ref_rms[i]))
            use_link[i] = gap <= lim
            if not use_link[i]:
                out.cut.append(("link", i, f"画像の上で {gap:.0f} px 離れている"))
    use_feet = [f is not None for f in feet]
    huber = opts.loss_scale_px if opts.loss == "huber" else None
    weight = max(opts.loss_scale_px, float(np.median(list(ref_rms.values()))) if ref_rms else 0.0)
    times = [sf.t_hit for sf in fits]
    t_track = list(times)
    lv = np.log(phys.terminal_velocity)
    sol = prob = active = None
    info: dict[str, Any] = {}

    def build() -> tuple[_Problem, list[int]] | None:
        """いまのつなぎ目で、どちらかの側がつながっているショットだけの問題を作る。"""
        act = [i for i in state if use_link[i] or (i + 1 < n and use_link[i + 1])]
        if not act:
            return None
        obs = split_observations(det, camera, times, margin_s, t_last)
        shots = []
        for i in act:
            ob = obs[i]
            keep = ob.t <= times[i] + o.max_flight_s
            prev_t = obs[i - 1].t if i > 0 else np.empty(0)
            shots.append(_Shot(ob.t[keep], ob.uv[keep], fits[i].side, use_link[i],
                               feet[i] if use_feet[i] else None, t_audio[i], t_track[i],
                               float(prev_t[-1]) if len(prev_t) else -np.inf,
                               float(ob.t[0]) if len(ob.t) else np.inf,
                               est and _is_long(fits[i], o)))
        return _Problem(shots, camera, phys, o, weight, huber, track_sigma, speed_of_sound_mps, est,
                        opts.max_speed_mps), act

    for split in range(max(o.n_splits, 1)):
        for _ in range(max(o.max_rounds, 1)):
            built = build()
            if built is None:
                sol = None
                break
            prob, active = built
            x0 = np.concatenate([state[i] for i in active] + ([[lv]] if prob.est else []))
            sol = prob.solve(x0)
            out.n_solves += 1
            _, _, info = prob.evaluate(sol.x)
            for k, i in enumerate(active):
                state[i] = sol.x[7 * k:7 * k + 7].copy()
            if prob.est:
                lv = float(sol.x[-1])
            # 切るもの：すき間の大きいつなぎ目 → ずれが大きくなったショットのつなぎ目 → 足元から離れた打点
            changed = False
            gaps = {i: info["gaps"][k] for k, i in enumerate(active) if use_link[i]}
            big = [i for i, gp in gaps.items() if gp > o.max_link_gap_m]
            if big:
                worst = max(big, key=lambda i: gaps[i])
                for i in big:
                    if i == worst or gaps[i] > 2 * o.max_link_gap_m:
                        use_link[i] = False
                        out.cut.append(("link", i, f"すき間 {gaps[i]:.2f} m"))
                changed = True
            else:
                for k, i in enumerate(active):
                    if info["rms"][k] <= o.max_rms_ratio * ref_rms[i] + o.rms_slack_px:
                        continue
                    near = [j for j in (i, i + 1) if j < n and use_link[j]]
                    if near:
                        j = max(near, key=lambda j: gaps.get(j, 0.0))
                        use_link[j] = False
                        out.cut.append(("link", j, f"ショット {i} のずれ {info['rms'][k]:.1f} px"
                                                   f"（元は {ref_rms[i]:.1f} px）"))
                        changed = True
            for k, i in enumerate(active):
                if use_feet[i] and info["feet_gap"][k] > o.max_feet_gap_m:
                    use_feet[i] = False
                    out.cut.append(("feet", i, f"打点が足元から {info['feet_gap'][k]:.1f} m"))
                    changed = True
            if not changed:
                break
        if sol is None or split + 1 >= o.n_splits:
            break
        # 直した打った時刻で観測を分け直す（時刻が動いていなければ終わり）
        new = list(times)
        for i in active:
            new[i] = float(state[i][0])
        if max(abs(a - b) for a, b in zip(new, times)) < 1e-4:
            break
        times = new
    out.links = list(use_link)
    if sol is None or prob is None or active is None:
        return out

    # --- 結果をショットごとの形に戻す
    r, jac, info = prob.evaluate(sol.x)
    cov = None
    if opts.ci_method != "none" or prob.est:
        dof = max(prob.n_res - prob.n_par, 1)
        try:
            cov = np.linalg.pinv(sol.jac.T @ sol.jac) * (2.0 * sol.cost / dof)
        except np.linalg.LinAlgError:
            cov = None
    if prob.est:
        n_long = sum(s.vt_free for s in prob.shots)
        sd = float(np.sqrt(max(cov[-1, -1], 0.0))) if cov is not None else np.inf
        out.vt = VtEstimate(float(np.exp(lv)), float(np.exp(lv)) * sd, n_long)
    from scipy.stats import norm

    zq = norm.ppf(0.5 + opts.ci_level / 2)
    _, _, _, vts = prob.unpack(sol.x)
    for k, i in enumerate(active):
        new = list(times)
        tau, P, V = float(state[i][0]), state[i][1:4].copy(), state[i][4:7].copy()
        rms, m = float(info["rms"][k]), len(prob.shots[k].t)
        speed = float(np.linalg.norm(V))
        linked = use_link[i] or (i + 1 < n and use_link[i + 1])
        if not (linked and np.isfinite(rms) and np.isfinite(speed) and speed > 0
                and rms <= o.max_rms_ratio * ref_rms[i] + o.rms_slack_px):
            continue
        if use_link[i]:
            out.gaps_m[i] = float(info["gaps"][k])
        horiz = V[:2] if np.hypot(*V[:2]) > 1e-9 else np.array([1.0, 0.0])
        f = FitResult(mode="plane", t_ref=tau, p0=P, v0=V, terminal_velocity=float(vts[k]),
                      theta=np.array([0.0, P[2], float(np.hypot(*V[:2])), V[2]]), cov_theta=None, rms_px=rms,
                      n_obs=m, success=bool(sol.success), message="joint",
                      plane=FlightPlane(np.r_[P[:2], 0.0], np.r_[horiz, 0.0]))
        if opts.ci_method != "none" and cov is not None:
            gr = V / speed
            sd = float(np.sqrt(max(gr @ cov[7 * k + 4:7 * k + 7, 7 * k + 4:7 * k + 7] @ gr, 0.0)))
            f.speed0_sd_fit, f.speed0_sd_timing = sd, 0.0
            if 2 * m - 7 >= 3 and sd < speed:
                f.speed0_ci, f.ci_method = (speed - zq * sd, speed + zq * sd), "jacobian"
            else:
                f.ci_method = "jacobian(unreliable)"
        old = fits[i]
        out.fits[i] = ShotFit(tau, old.side, old.t_end, m, f, P, old.message, None)
        out.joint[i] = True
    # 打った時刻・飛行の終わり・観測を、決まった時刻でそろえる
    final_t = [out.fits[i].t_hit for i in range(n)]
    obs = split_observations(det, camera, final_t, margin_s, t_last)
    for i in range(n):
        sf = out.fits[i]
        if not out.joint[i] and not (i + 1 < n and out.joint[i + 1]):
            continue
        t_end = final_t[i + 1] if i + 1 < n else max(t_last, final_t[i])
        if out.joint[i]:
            sf.t_end, sf.obs = t_end, obs[i]
        else:
            out.fits[i] = ShotFit(sf.t_hit, sf.side, t_end, sf.n_obs, sf.fit, sf.hit_point, sf.message, sf.obs)
    return out


def combine_vt(estimates: list[VtEstimate | None], default: float, o: JointOptions | None = None
               ) -> tuple[float, dict[str, Any]]:
    """ラリーごとの終端速度の推定を、動画全体の 1 つの値にまとめる。

    不確かさの 2 乗の逆数で重みをつけた平均。不確かさは、当てはめからの値と、ラリーどうしのばらつきからの値の
    大きい方。長いショットが ``vt_min_flights`` 本より少ない・不確かさが ``vt_max_sigma_mps`` より大きい・
    ``vt_range_mps`` の外なら、``default``（設定の値）のままにする。

    Returns:
        (使う終端速度 [m/s], 記録用の辞書：``used``（推定を使ったか）・``value_mps``・``sigma_mps``・``n_flights``・
        ``n_rallies``・``default_mps``・``reason``)
    """
    o = o or JointOptions()
    ok = [e for e in estimates if e is not None and np.isfinite(e.value) and np.isfinite(e.sigma) and e.sigma > 0]
    rec: dict[str, Any] = {"used": False, "default_mps": default, "value_mps": None, "sigma_mps": None,
                           "n_flights": int(sum(e.n_flights for e in ok)), "n_rallies": len(ok), "reason": ""}
    if not ok:
        rec["reason"] = "長く飛んだショットがない"
        return default, rec
    wts = np.array([1.0 / e.sigma ** 2 for e in ok])
    vals = np.array([e.value for e in ok])
    mean = float(np.sum(wts * vals) / wts.sum())
    sigma = float(1.0 / np.sqrt(wts.sum()))
    if len(ok) > 1:
        scatter = float(np.sqrt(np.sum(wts * (vals - mean) ** 2) / wts.sum() / (len(ok) - 1)))
        sigma = max(sigma, scatter)
    rec["value_mps"], rec["sigma_mps"] = round(mean, 3), round(sigma, 3)
    if rec["n_flights"] < o.vt_min_flights:
        rec["reason"] = f"長く飛んだショットが {rec['n_flights']} 本（{o.vt_min_flights} 本以上で使う）"
    elif sigma > o.vt_max_sigma_mps:
        rec["reason"] = f"不確かさ ±{sigma:.2f} m/s が大きい（±{o.vt_max_sigma_mps} m/s 以下で使う）"
    elif not o.vt_range_mps[0] <= mean <= o.vt_range_mps[1]:
        rec["reason"] = f"推定 {mean:.2f} m/s がありえる範囲の外"
    else:
        rec["used"] = True
        return mean, rec
    return default, rec
