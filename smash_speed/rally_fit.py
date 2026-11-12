"""ラリー全体をまとめて当てはめる（``shot_fit.fit_shots`` のショットごとの当てはめの後の仕上げ）。

カメラ 1 台では、短いショット・カメラへ向かうショットの奥行きが 1 本だけでは決まらない。ラリーの中では、
ショット k の軌道の終わりはショット k + 1 の打点と同じ点・同じ時刻なので、ラリーのすべてのショットを 1 つの
最小 2 乗問題にして、となりのショットに奥行きと打った時刻を決めてもらう。

決める量は、ショットごとの「打った時刻 τ・打点 P（3 次元）・打ち出しの速度 V（3 次元）」（と、求めるときは
長いショットごとの終端速度 v_t）。ずれ（残差）は次のとおり。

- 画像の上のずれ [px]：ショット k の観測と、(τ_k, P_k, V_k) から運動方程式で進めた軌道の投影。
- つなぎ目：ショット k − 1 の軌道の時刻 τ_k の位置が P_k と同じ（``link_sigma_m``）。
- 打つ人の足元：P_k の水平の位置が足元の近く（ラケットの届く範囲。``feet_sigma_m``）。
- 打った時刻：打球音があれば「音の時刻 − 打点からカメラまでの音の遅れ」、なければ軌道から決めた時刻の近く。
  前のショットの最後の観測より後・自分の最初の観測より前。
- 打点の高さ・コートの中・打った側・初速の上限（``physics.MAX_FIT_SPEED_MPS``）。

目安の 1σ は、画像の上のずれの損失の切り替え点（``FitOptions.loss_scale_px``。2 px）と同じ重み。ショットごとの当てはめの
ずれがそれより大きいショット（実動画の高いロブは、1 本だけで当てはめても 10 px 以上ずれることがある）は、画像の上の
ずれを「切り替え点 ÷ そのずれ」倍に軽くして、合わない長いショットが、となりの短いショットを引きずらないようにする。

間違った打球（見落とし・余分な打球）・別の物の軌道が 1 本あってもラリー全体が崩れないよう、つなぎ目と軌道から
決めた時刻は外れ値に強い損失（Huber）のままにし、解いた後に

- つなぎ目のすき間が ``max_link_gap_m`` を超えた
- つないだことでショットの画像の上のずれが大きくなった（``max_rms_ratio``）
- つないだことで打点が打つ人の足元から届かない所（``reach_m``）へ動いた

つなぎ目を切って解き直す（``max_rounds`` 回まで）。当てはめる前にも、前後の軌道が画像の上で交わらないつなぎ目は
切っておく。前後どちらともつながらなかったショットは、ショットごとの当てはめの結果のまま。

終端速度は、長く飛んだショット（クリア・ロブ。``long_flight_s``・``long_min_obs``）に 1 本ずつ自分の v_t を持たせて
一緒に解く（``estimate_vt=True``）。動画全体の値は、その中央値（``combine_vt``）。不確かさは 1 本ずつの値の
ばらつきから出すので、モデルに合わない動画（実動画の高いロブは画像の上で 10 px 以上ずれることがある）では
大きくなる。不確かさが大きい・長いショットが少ないときは設定の値のままにする。

軌道の積分は、つながったショットの組と、ヤコビアン用に速度・終端速度を少し動かしたものを、決まった刻みの
RK4 でまとめて進める（``integrate``。``physics.simulate`` との差は 0.1 mm 以下）。つなぎ目を切ると組が分かれるので、
変わった組だけを解き直す。
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
_N_PRIOR = 17     # ショットごとの、画像の上のずれ以外の残差の数


@dataclass
class JointOptions:
    """ラリー全体の当てはめの設定。"""

    link_sigma_m: float = 0.15        # つなぎ目（前のショットの終わりと打点）のすき間の不確かさ [m]
    firm_links: bool = True           # つなぎ目を 2 乗のまま効かせる（偽なら外れ値に強い損失で弱める）
    max_link_gap_m: float = 0.6       # 解いた後のすき間がこれを超えたつなぎ目は切る [m]
    max_cross_frac: float = 0.03      # 当てはめる前に、前後の軌道が画像の上でこれ（画像の幅の割合）より近づかない
                                      # つなぎ目は切る（1920 px で 58 px。はっきり別の物だけ。速いショットは 1 本ずつの
                                      # 当てはめのずれで 12 px ほど離れることがある。あとは解いてから切る）
    cross_window_s: float = 0.1       # そのとき近づく所を探す、打った時刻の前後の幅 [s]
    feet_sigma_m: float = 0.5         # 打つ人の足元から打点までの水平の距離の不確かさ [m]
    anchor_sigma_m: float = 0.5       # 前のショットとつながらないショットの打点を、ショットごとの当てはめのつないだ打点
                                      # （``ShotFit.hit_point``）の近くに留める不確かさ [m]
    max_feet_gap_m: float = 2.5       # ショットごとの当てはめの打点が足元からこれより離れていたら、その足元は使わない [m]
    reach_m: float = 1.5              # 足元から打点まで水平に届く距離 [m]。つないだことで打点がこれより離れ、ショットごとの
    reach_slack_m: float = 0.5        # 当てはめより ``reach_slack_m`` 以上遠くなったら、足元を信じてつなぎ目を切る
    audio_sigma_s: float = 0.003      # 打球音から決めた打った時刻の不確かさ [s]
    track_sigma_s: float | None = None  # 軌道から決めた打った時刻の不確かさ [s]（None ならコマの間隔）
    order_sigma_s: float = 0.003      # 打った時刻が前後の観測の順を越えたときの罰の幅 [s]
    hit_z_m: tuple[float, float] = (0.0, 3.6)   # 打点の高さの範囲 [m]
    max_shot_rms_frac: float = 0.01   # ショットごとの当てはめのずれがこれ（画像の幅の割合）を超えたショットはまとめない
                                      # （別の物の軌道。ショットの一覧が「軌道が検出と合わない」とする大きさと同じ）
    min_obs: int = 5                  # これより少ないコマのショットはまとめない（つなぎ目だけで決まってしまう）
    max_rms_ratio: float = 1.5        # つないだ後の画像の上のずれが「ショットごとのずれ × これ + rms_slack_px」を
    rms_slack_px: float = 1.0         # 超えたショットは、つなぎ目を切る（最後まで超えていれば元の当てはめに戻す）
    max_rounds: int = 6               # 切って解き直す回数の上限
    n_splits: int = 2                 # 直した打った時刻で観測を分け直して解く回数
    max_nfev: int = 60                # 1 回の最適化の評価の回数の上限
    step_fast_s: float = 0.002        # 積分の刻み [s]（打ってから ``fast_s`` まで。速い間は空気抵抗が大きい）
    fast_s: float = 0.3
    step_slow_s: float = 0.008        # 積分の刻み [s]（その後）
    max_flight_s: float = 3.5         # 打ってからこれより後の観測は使わない [s]（床に落ちた後の検出）
    # --- 終端速度 ---
    estimate_vt: bool = True          # 長く飛んだショットから終端速度を一緒に求める
    long_flight_s: float = 0.6        # 長く飛んだショットとみなす飛行の時間 [s]
    long_min_obs: int = 12            # 同・見えたコマの数
    vt_max_rms_frac: float = 0.002    # ショットごとの当てはめのずれがこれ（画像の幅の割合。と損失の切り替え点の 2 倍の
                                      # 大きい方）を超えた長いショットは、終端速度を求めるのに使わない
    vt_flight_sigma_mps: float = 0.2  # 当てはめからの不確かさがこれより大きい 1 本の推定（軌道から終端速度が決まって
                                      # いない。低いカメラの山なりのショットなど）は数えない [m/s]
    vt_prior_sigma: float = 0.15      # 設定の値からの log(v_t) の不確かさ（弱い目安。±1 m/s）
    vt_min_flights: int = 3           # 動画全体で、長く飛んだショットがこれより少なければ設定の値のまま
    vt_sys_sigma_mps: float = 0.1     # カメラ 1 台の終端速度の、ばらつきに出ない不確かさ [m/s]（コート合わせの高さ方向の
                                      # 縮尺が 4% ずれると、どのショットの終端速度も同じ向きに 2% = 0.13 m/s ずれる）
    vt_max_sigma_mps: float = 0.15    # 動画全体の推定の不確かさがこれより大きければ設定の値のまま [m/s]
    vt_refit_mps: float = 0.05        # 決めた終端速度が設定の値とこれ以上違えば、つなぎ目を見直して解き直す [m/s]
    vt_range_mps: tuple[float, float] = (5.0, 8.5)  # これを外れた推定は使わない（羽根・ナイロンのシャトルは 6〜7.5）


@dataclass
class VtEstimate:
    """長いショット 1 本からの終端速度の推定。``value``・``sigma``（当てはめからの不確かさ）[m/s]、``shot`` はショットの番号。"""

    value: float
    sigma: float
    shot: int = -1

    def to_dict(self) -> dict[str, Any]:
        return {"value_mps": round(self.value, 3), "sigma_mps": round(self.sigma, 3), "shot": self.shot}


@dataclass
class JointResult:
    """``joint_fit`` の結果。

    Attributes:
        fits: 入力と同じ順の ``ShotFit``（まとめて当てはめたショットは置き換えたもの）
        joint: ショットごとに、まとめた当てはめの結果を使ったか
        links: つなぎ目 k（ショット k − 1 → k。k = 0 は使わない）を使ったか
        gaps_m: つなぎ目のすき間 [m]（使わなかった所は NaN）
        cut: 切ったつなぎ目（link）・まとめなかったショット（shot）の記録（``(種類, ショットの番号, 理由)``）
        vt: 長いショット 1 本ずつの終端速度の推定（求めなかった・長いショットがなければ空）
        terminal_velocity: 最後の当てはめに使った終端速度 [m/s]
        n_solves: 最適化を解いた回数
    """

    fits: list[Any]
    joint: list[bool]
    links: list[bool]
    gaps_m: list[float]
    cut: list[tuple[str, int, str]] = field(default_factory=list)
    vt: list[VtEstimate] = field(default_factory=list)
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
    with np.errstate(invalid="ignore"):    # 前へ戻して発散した所（inf）は NaN になる
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
    anchor: np.ndarray | None         # 前のショットとつながらないときの、打点の目安
    t_audio: float | None             # 打球音の時刻
    t_track: float                    # 軌道から決めた打った時刻（打球音がないときの目安）
    t_lo: float                       # 打った時刻の下限（前のショットの最後の観測）
    t_hi: float                       # 同・上限（自分の最初の観測）
    vt_free: bool = False             # 共通の終端速度を使うか
    scale: float = 1.0                # 画像の上のずれに掛ける重み（ショットごとの当てはめのずれが大きいショットは軽くする）


class _Problem:
    """ラリー（つながったショットの組）の残差とヤコビアン。パラメータはショットごとに (τ, P, V) の 7 個と、
    その後ろに、終端速度を求めるショットごとの log(v_t)。"""

    def __init__(self, shots: list[_Shot], camera: Camera, phys: PhysicsParams, o: JointOptions, weight_px: float,
                 huber_px: float | None, track_sigma_s: float, speed_of_sound: float, estimate_vt: bool,
                 max_speed: float | None):
        self.shots, self.camera, self.g, self.vt0, self.o = shots, camera, phys.g, phys.terminal_velocity, o
        self.w, self.huber = weight_px, huber_px
        self.track_sigma, self.c_sound, self.max_speed = track_sigma_s, speed_of_sound, max_speed
        self.n = len(shots)
        self.free = [k for k, s in enumerate(shots) if estimate_vt and s.vt_free]
        self.est = bool(self.free)
        self.lv_col = {k: 7 * self.n + j for j, k in enumerate(self.free)}   # ショット → log(v_t) の列
        self.n_par = 7 * self.n + len(self.free)
        self.nv = 4 + int(self.est)
        self.rows = np.cumsum([0] + [2 * len(s.t) + _N_PRIOR for s in shots])
        self.n_res = int(self.rows[-1]) + len(self.free)
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
            vt[self.free] = np.exp(x[7 * self.n:])
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
            dlv = (pr[:, 4] - pr[:, 0]) / _DLV if k in self.lv_col else None
            # --- 画像の上のずれ
            a = A[at:at + m]
            e = uv[at:at + m] - s.uv
            ok = np.isfinite(e).all(axis=1) & np.isfinite(a).all(axis=(1, 2))
            a = np.where(ok[:, None, None], a, 0.0)
            a = a * s.scale
            rr = slice(row, row + 2 * m)
            r[rr] = np.where(ok[:, None], e * s.scale, _BAD_PX).ravel()
            if ok.any():
                rms[k] = float(np.sqrt(np.mean(e[ok] ** 2)))
            jac[rr, c] = -np.einsum("mij,mj->mi", a, np.nan_to_num(vk[:m])).ravel()
            jac[rr, c + 1:c + 4] = a.reshape(2 * m, 3)
            jac[rr, c + 4:c + 7] = np.einsum("mij,mvj->miv", a, np.nan_to_num(dV[:m])).reshape(2 * m, 3)
            if dlv is not None:
                jac[rr, self.lv_col[k]] = np.einsum("mij,mj->mi", a, np.nan_to_num(dlv[:m])).ravel()
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
                    if k - 1 in self.lv_col:
                        jac[lr, self.lv_col[k - 1]] = wl * (pe[4] - pe[0]) / _DLV
                    jac[lr, c] = wl * ve
                    jac[lr, c + 1:c + 4] = -wl * np.eye(3)
                    if o.firm_links:
                        self._quad(r[lr], jac[lr])
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
            # --- 前とつながらないショットの打点の目安
            if s.anchor is not None and not s.link:
                wa = w / o.anchor_sigma_m
                r[row:row + 3] = wa * (P[k] - s.anchor)
                jac[row:row + 3, c + 1:c + 4] = wa * np.eye(3)
            row += 3
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
        for j, k in enumerate(self.free):
            row = int(self.rows[-1]) + j
            r[row] = w * (x[self.lv_col[k]] - np.log(self.vt0)) / o.vt_prior_sigma
            jac[row, self.lv_col[k]] = w / o.vt_prior_sigma
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


def _is_long(sf: Any, o: JointOptions, rms_px: float = 0.0, max_rms_px: float = np.inf) -> bool:
    """終端速度を求めるのに使う長いショットか（長く飛び、コマが多く、ショットごとの当てはめが合っている）。"""
    return sf.duration >= o.long_flight_s and sf.n_obs >= o.long_min_obs and rms_px <= max_rms_px


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
    from smash_speed.shot_fit import ShotFit, _plausible_point, _state_at, split_observations

    o = o or JointOptions()
    opts = opts or FitOptions(refine_plane=True)
    base = base if base is not None else fits
    est = o.estimate_vt if estimate_vt is None else estimate_vt
    n = len(fits)
    feet = list(feet) if feet is not None else [None] * n
    feet = [None if f is None or not np.isfinite(np.asarray(f, float)[:2]).all() else np.asarray(f, float)[:2]
            for f in feet]
    t_audio = list(t_audio) if t_audio is not None else [None] * n
    out = JointResult(list(base), [False] * n, [False] * n, [np.nan] * n, terminal_velocity=phys.terminal_velocity)
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
        if sf.n_obs < o.min_obs:
            continue
        if not base[i].fit.rms_px <= min(max_rms_px, o.max_shot_rms_frac * camera.width):
            out.cut.append(("shot", i, f"ずれ {base[i].fit.rms_px:.0f} px"))
            continue
        st = _state_at(sf.fit, sf.t_hit, phys)
        if st is not None and np.linalg.norm(st[1]) > 0:
            state[i] = np.r_[sf.t_hit, st[0], st[1]]
    # 同じ側が続けて打っているショットは、間の打球を見落としている（2 本の飛行が 1 本になっている）ので、まとめない
    for i in range(n - 1):
        if i in state and fits[i].side in ("h", "o") and fits[i].side == fits[i + 1].side:
            del state[i]
            out.cut.append(("shot", i, "次の打球も同じ側"))
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
    # ショットごとの当てはめの打点が足元から遠すぎるショット（足元か軌道のどちらかが違う）は、足元を使わない
    base_feet_gap = {}
    for i in state:
        st = _state_at(base[i].fit, base[i].t_hit, phys) if feet[i] is not None else None
        if st is not None:
            base_feet_gap[i] = float(np.hypot(*(st[0][:2] - feet[i])))
    use_feet = [i in base_feet_gap and base_feet_gap[i] <= o.max_feet_gap_m for i in range(n)]
    huber = opts.loss_scale_px if opts.loss == "huber" else None
    weight = opts.loss_scale_px
    times = [sf.t_hit for sf in fits]
    t_track = [sf.t_hit for sf in base]
    anchors = [None if sf.hit_point is None or not np.isfinite(sf.hit_point).all() else np.asarray(sf.hit_point, float)
               for sf in base]
    lv: dict[int, float] = {}     # ショット → log(v_t)（終端速度を求めるショット）
    vt_rms = max(2 * opts.loss_scale_px, o.vt_max_rms_frac * camera.width)
    long = {i for i in state if est and _is_long(base[i], o, ref_rms[i], vt_rms)}
    solved: dict[tuple[Any, ...], tuple[_Problem, Any, dict[str, Any]]] = {}   # 解いた組（つながったショットの並び）
    chains: list[list[int]] = []
    rms_now: dict[int, float] = {}
    gaps: dict[int, float] = {}
    feet_now: dict[int, float] = {}

    def linked(i: int) -> bool:
        return use_link[i] or (i + 1 < n and use_link[i + 1])

    def solve_chains(obs: list[Any]) -> None:
        """いまのつなぎ目でつながったショットの組ごとに解く（組どうしは関係しないので、変わった組だけを解き直す）。
        終端速度を求めるときは、つながっていない長いショットも 1 本だけの組として解く。"""
        chains.clear()
        for i in sorted(state):
            if not (linked(i) or i in long):
                continue
            if chains and use_link[i] and chains[-1][-1] == i - 1:
                chains[-1].append(i)
            else:
                chains.append([i])
        for ch in chains:
            key = (tuple(ch), tuple(use_feet[i] for i in ch))
            if key not in solved:
                shots = []
                for i in ch:
                    ob = obs[i]
                    keep = ob.t <= times[i] + o.max_flight_s
                    prev_t = obs[i - 1].t if i > 0 else np.empty(0)
                    shots.append(_Shot(ob.t[keep], ob.uv[keep], fits[i].side, use_link[i] and i != ch[0],
                                       feet[i] if use_feet[i] else None, anchors[i], t_audio[i], t_track[i],
                                       float(prev_t[-1]) if len(prev_t) else -np.inf,
                                       float(ob.t[0]) if len(ob.t) else np.inf, i in long,
                                       min(1.0, opts.loss_scale_px / max(ref_rms[i], 1e-9))))
                prob = _Problem(shots, camera, phys, o, weight, huber, track_sigma, speed_of_sound_mps, est,
                                opts.max_speed_mps)
                x0 = np.concatenate([state[i] for i in ch]
                                    + [[lv.get(ch[k], np.log(phys.terminal_velocity)) for k in prob.free]])
                sol = prob.solve(x0)
                out.n_solves += 1
                solved[key] = (prob, sol, prob.evaluate(sol.x)[2])
            prob, sol, info = solved[key]
            for k, i in enumerate(ch):
                state[i] = sol.x[7 * k:7 * k + 7].copy()
                rms_now[i], feet_now[i] = float(info["rms"][k]), float(info["feet_gap"][k])
                if use_link[i] and k > 0:
                    gaps[i] = float(info["gaps"][k])
            for k in prob.free:
                lv[ch[k]] = float(sol.x[prob.lv_col[k]])

    def cut_links() -> bool:
        """切るもの：すき間の大きいつなぎ目 → ずれが大きくなったショットのつなぎ目 → 打点が足元から離れたショットのつなぎ目。"""
        act = [i for ch in chains for i in ch]
        big = [i for i in act if use_link[i] and gaps.get(i, 0.0) > o.max_link_gap_m]
        if big:
            for i in big:
                use_link[i] = False
                out.cut.append(("link", i, f"すき間 {gaps[i]:.2f} m"))
            return True
        changed = False
        for why in ("rms", "feet"):
            for i in act:
                if why == "rms":
                    if rms_now[i] <= o.max_rms_ratio * ref_rms[i] + o.rms_slack_px:
                        continue
                    text = f"ショット {i} のずれ {rms_now[i]:.1f} px（元は {ref_rms[i]:.1f} px）"
                else:
                    if not (use_feet[i] and feet_now[i] > max(o.reach_m, base_feet_gap[i] + o.reach_slack_m)):
                        continue
                    text = f"ショット {i} の打点が足元から {feet_now[i]:.1f} m（元は {base_feet_gap[i]:.1f} m）"
                near = [j for j in (i, i + 1) if j < n and use_link[j]]
                if near:
                    j = max(near, key=lambda j: gaps.get(j, 0.0))
                    use_link[j] = False
                    out.cut.append(("link", j, text))
                    changed = True
            if changed:
                return True
        return False

    for split in range(max(o.n_splits, 1)):
        obs = split_observations(det, camera, times, margin_s, t_last)
        solved.clear()
        for _ in range(max(o.max_rounds, 1)):
            gaps.clear()
            solve_chains(obs)
            if not chains or not cut_links():
                break
        if not chains or split + 1 >= o.n_splits:
            break
        # 直した打った時刻で観測を分け直す（時刻が動いていなければ終わり）
        moved = [float(state[i][0]) if any(i in ch for ch in chains) else t for i, t in enumerate(times)]
        if max(abs(a - b) for a, b in zip(moved, times)) < 1e-4:
            break
        times = moved
    out.links = list(use_link)

    # --- 結果をショットごとの形に戻す
    from scipy.stats import norm

    zq = norm.ppf(0.5 + opts.ci_level / 2)
    for ch in chains:
        prob, sol, info = solved[(tuple(ch), tuple(use_feet[i] for i in ch))]
        cov = None
        if opts.ci_method != "none" or prob.est:
            dof = max(prob.n_res - prob.n_par, 1)
            try:
                cov = np.linalg.pinv(sol.jac.T @ sol.jac) * (2.0 * sol.cost / dof)
            except np.linalg.LinAlgError:
                cov = None
        for k in prob.free:
            c = prob.lv_col[k]
            sd = float(np.sqrt(max(cov[c, c], 0.0))) if cov is not None else np.inf
            out.vt.append(VtEstimate(float(np.exp(sol.x[c])), float(np.exp(sol.x[c])) * sd, ch[k]))
        vts = prob.unpack(sol.x)[3]
        for k, i in enumerate(ch):
            tau, P, V = float(state[i][0]), state[i][1:4].copy(), state[i][4:7].copy()
            rms, m = rms_now[i], len(prob.shots[k].t)
            speed = float(np.linalg.norm(V))
            if not (linked(i) and np.isfinite(rms) and np.isfinite(speed) and speed > 0
                    and rms <= o.max_rms_ratio * ref_rms[i] + o.rms_slack_px and _plausible_point(P, fits[i].side)):
                continue
            if use_link[i]:
                out.gaps_m[i] = gaps.get(i, np.nan)
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


def combine_vt(estimates: list[VtEstimate], default: float, o: JointOptions | None = None
               ) -> tuple[float, dict[str, Any]]:
    """長いショット 1 本ずつの終端速度の推定を、動画全体の 1 つの値にまとめる。

    値は中央値。不確かさは、1 本ずつの値のばらつき（中央値からの差の中央値）から出した中央値の標準誤差と、
    当てはめからの不確かさの大きい方に、ばらつきに出ない分（``vt_sys_sigma_mps``）を 2 乗和で足したもの。
    当てはめからの不確かさが ``vt_flight_sigma_mps`` より大きい推定（軌道から終端速度が決まっていない）は数えない。数えた本数が ``vt_min_flights`` より少ない・不確かさが
    ``vt_max_sigma_mps`` より大きい・``vt_range_mps`` の外なら、``default``（設定の値）のままにする。

    Returns:
        (使う終端速度 [m/s], 記録用の辞書：``used``（推定を使ったか）・``value_mps``・``sigma_mps``・``n_flights``・
        ``default_mps``・``reason``)
    """
    o = o or JointOptions()
    ok = [e for e in estimates if e is not None and np.isfinite(e.value) and np.isfinite(e.sigma)
          and 0 < e.sigma <= o.vt_flight_sigma_mps]
    rec: dict[str, Any] = {"used": False, "default_mps": default, "value_mps": None, "sigma_mps": None,
                           "n_flights": len(ok), "reason": ""}
    if not ok:
        rec["reason"] = "終端速度が決まるほど長く飛んだショットがない"
        return default, rec
    vals = np.array([e.value for e in ok])
    med = float(np.median(vals))
    formal = float(1.0 / np.sqrt(np.sum([1.0 / e.sigma ** 2 for e in ok])))
    # 正規分布なら 標準偏差 ≈ 1.4826 × MAD、中央値の標準誤差 ≈ 1.2533 × 標準偏差 / √n
    scatter = 1.2533 * 1.4826 * float(np.median(np.abs(vals - med))) / np.sqrt(len(ok)) if len(ok) > 1 else np.inf
    sigma = float(np.hypot(max(formal, scatter), o.vt_sys_sigma_mps))
    rec["value_mps"], rec["sigma_mps"] = round(med, 3), round(float(sigma), 3) if np.isfinite(sigma) else None
    if len(ok) < o.vt_min_flights:
        rec["reason"] = f"長く飛んだショットが {len(ok)} 本（{o.vt_min_flights} 本以上で使う）"
    elif sigma > o.vt_max_sigma_mps:
        rec["reason"] = f"不確かさ ±{sigma:.2f} m/s が大きい（±{o.vt_max_sigma_mps} m/s 以下で使う）"
    elif not o.vt_range_mps[0] <= med <= o.vt_range_mps[1]:
        rec["reason"] = f"推定 {med:.2f} m/s がありえる範囲の外"
    else:
        rec["used"] = True
        return med, rec
    return default, rec


def refit_with_vt(fits: list[Any], skip: list[bool], phys: PhysicsParams, opts: FitOptions) -> None:
    """まとめなかったショット（``skip`` が偽で当てはめのあるもの）を、いまの飛行平面と打点の目安から、終端速度 ``phys`` で
    当てはめ直す（``fits`` を書き換える）。動画全体の終端速度が設定の値と違うと決まったときに、すべてのショットを
    同じ終端速度にそろえる。当てはめ直せなければ元のまま。"""
    from smash_speed.physics import fit_trajectory

    for sf, done in zip(fits, skip):
        f = sf.fit
        if (done or f is None or f.plane is None or f.mode != "plane" or sf.obs is None
                or abs(f.terminal_velocity - phys.terminal_velocity) < 1e-6):
            continue
        fo = FitOptions(**{**opts.__dict__, "refine_plane": opts.refine_plane and sf.n_obs > 5, "init_state": None,
                           "p0_prior_m": sf.hit_point, "quadratic_priors": True})
        try:
            new = fit_trajectory(sf.obs, f.t_ref, "plane", f.plane, phys, fo,
                                 theta0=np.r_[f.theta[:4], [0.0, 0.0] if fo.refine_plane else []])
        except (ValueError, RuntimeError):
            continue
        if new.success and np.isfinite(new.speed0):
            sf.fit = new


def fit_rallies(rallies: list[dict[str, Any]], phys: PhysicsParams, opts: FitOptions | None = None,
                o: JointOptions | None = None, mapper: Any = None) -> tuple[list[JointResult], dict[str, Any]]:
    """動画（セッション）のすべてのラリーをまとめて当てはめ、終端速度を動画全体で 1 つ決める。

    1. ラリーごとに、長いショット 1 本ずつの終端速度を一緒に求めながら当てはめる（``joint_fit``）。
    2. 1 本ずつの推定を 1 つにまとめる（``combine_vt``。足りなければ設定の値のまま）。
    3. 決めた終端速度で、すべてのショットを当てはめ直す（1 の結果から始めるので速い）。

    ``o.estimate_vt`` が偽なら、設定の終端速度で 1 回だけ当てはめる。

    Args:
        rallies: ラリーごとの ``joint_fit`` の引数の辞書（``fits``・``det``・``camera`` と、あれば ``feet``・``t_audio``・
            ``margin_s``・``t_last``・``speed_of_sound_mps``・``max_rms_px``）
        mapper: ``mapper(fn, 引数のリスト)`` でラリーごとの計算を並べる関数（None なら 1 つずつ）

    Returns:
        (ラリーごとの ``JointResult``, 終端速度の記録（``combine_vt``）)
    """
    o = o or JointOptions()
    opts = opts or FitOptions(refine_plane=True)
    run = mapper or (lambda fn, args: [fn(*a) for a in args])
    first = run(_joint_job, [(r, phys, opts, o, o.estimate_vt, None, None) for r in rallies])
    if not o.estimate_vt:
        return first, {"used": False, "default_mps": phys.terminal_velocity, "value_mps": None, "sigma_mps": None,
                       "n_flights": 0, "reason": "求めていない"}
    vt, rec = combine_vt([e for r in first for e in r.vt], phys.terminal_velocity, o)
    phys2 = PhysicsParams(phys.g, vt, phys.method, phys.rtol, phys.atol)
    o2, keep_links = o, True
    if rec["used"] and abs(vt - phys.terminal_velocity) > o.vt_refit_mps:
        # 終端速度が変わると、設定の値で切れたつなぎ目がつながることがあるので、つなぎ目を見直す。つながらない
        # ショットの打点の目安（設定の終端速度での当てはめの打点）もずれているので、ゆるめる
        o2 = JointOptions(**{**o.__dict__, "anchor_sigma_m": 4 * o.anchor_sigma_m})
        keep_links = False
    second = run(_joint_job, [(r, phys2, opts, o2, False, f.links if keep_links else None, f.fits)
                              for r, f in zip(rallies, first)])
    for res, f in zip(second, first):
        res.vt, res.cut, res.n_solves = f.vt, f.cut + res.cut, f.n_solves + res.n_solves
        if rec["used"]:
            refit_with_vt(res.fits, res.joint, phys2, opts)
    return second, rec


def _joint_job(rally: dict[str, Any], phys: PhysicsParams, opts: FitOptions, o: JointOptions, estimate_vt: bool,
               links: list[bool] | None, start: list[Any] | None) -> JointResult:
    """``fit_rallies`` の 1 ラリー分（``start`` があれば、そこから始めて元の当てはめ ``rally["fits"]`` と比べる）。"""
    kw = {k: v for k, v in rally.items() if k != "fits"}
    return joint_fit(start if start is not None else rally["fits"], phys=phys, opts=opts, o=o,
                     estimate_vt=estimate_vt, links=links, base=rally["fits"], **kw)
