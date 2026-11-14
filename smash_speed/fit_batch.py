"""たくさんの軌道の当てはめを、まとめて解く（ROADMAP F3e）。

ショットの一覧の当てはめ（``shot_fit.fit_shots``）は、段階ごとに互いに関係しない当てはめ（``physics.fit_trajectory``）を
何百回も行う。1 回ずつ解くと、ほとんどの時間が scipy の積分（``solve_ivp``）と最小 2 乗（``least_squares``）の
Python の手間になる。ここでは、同じ段階の当てはめをまとめて、

1. 当てはめの手順は ``physics.fit_plan`` をそのまま使う（初期値・事前分布・初速の上限の罰・信頼区間は同じ）。
2. 最小 2 乗は、scipy の ``least_squares``（``method="trf"``・``x_scale="jac"``・数値微分のヤコビアン）と同じ手順を
   問題ごとに進める（``_trf``。信頼領域・終わりの判定・頑健な損失は scipy の関数をそのまま呼ぶ）。
3. 残差の計算（運動方程式の積分・投影）だけを、全部の問題の全部の試す点について 1 回の配列の計算にまとめる
   （``_Evaluator``）。

積分は、空気抵抗と重力だけなので、水平の速さの向きは変わらない。軌道は「水平の向きに進んだ距離 H・高さ Z」の
2 次元で解き（状態は H, Z, U = dH/dt, W = dZ/dt）、打点の位置・水平の向き・飛行平面の補正は後から足す。
数値微分で打点の位置・平面の補正を少し動かした点は、同じ 2 次元の軌道を使い回す（積分するのは速度・終端速度を
動かした点だけ）。

時間の刻みは打った時刻からの時間 s の幾何級数（``time_grid``。刻み h = max(0.5 ms, 0.04 s)、上限 20 ms）。
空気抵抗が速さの 2 乗に比例するので、打ってから s 秒後の速さは初速によらず 1/(k s) 以下で、速さが変わる時間の
目安 1/(k v) は s 以上になる。刻みを s に比例させると、どの初速でも 1 刻みの誤差（RK4）は同じくらいに小さい。
観測の時刻の位置は、刻みの両端の位置・速度からの 3 次のエルミート補間。``physics.simulate``（DOP853、
相対誤差 1e-10）との差は ``tests/test_fit_batch.py`` で確かめている。

1 つの問題の結果は、一緒に解いたほかの問題によらない（配列の計算は要素ごとの四則演算と平方根で、
足し合わせる計算は問題ごとに分けてある）。

GPU について：この計算は 1 刻みあたり 50 回ほどの配列の演算を 200 刻みほど続けるもので、1 回の演算は数千〜数万の
要素しかない。DirectML（onnxruntime・torch）では 1 回の演算の起動に数十 µs かかり、配列が 10 万要素を超えないと
CPU の numpy より速くならない（``docs/speed.md`` の F3e）。そのため numpy で計算する。
"""

from __future__ import annotations

import threading
from typing import Any, Generator, Sequence

import numpy as np
from numpy.linalg import norm, svd

from smash_speed.physics import MAX_SIM_SPEED_MPS, SPEED_PENALTY_SIGMA_MPS, FitResult, LsqRequest, fit_plan

try:  # scipy の least_squares（trf）と同じ手順にするため、中の関数をそのまま使う
    from scipy.optimize import OptimizeResult
    from scipy.optimize._lsq.common import (check_termination, compute_grad, compute_jac_scale, evaluate_quadratic,
                                            scale_for_robust_loss_function, solve_lsq_trust_region, update_tr_radius)
    from scipy.optimize._lsq.least_squares import TERMINATION_MESSAGES, construct_loss_function
    from scipy.optimize._numdiff import _compute_absolute_step
    AVAILABLE = True
except ImportError:  # scipy の中の名前が変わったら、1 つずつの当てはめ（CPU）に戻す
    AVAILABLE = False

STEP_MIN_S = 5e-4      # 打った直後の刻み [s]
STEP_GROWTH = 0.04     # 刻み ÷ 打ってからの時間
STEP_MAX_S = 0.02      # 刻みの上限 [s]（遅くなった後は重力の時間 v_t / g ≈ 0.7 s で変わる）
_GRID: list[np.ndarray] = []


def time_grid(s_max: float) -> np.ndarray:
    """打った時刻からの時間 [s] の刻み（0 から、``s_max`` 以上の所まで）。どの問題も同じ刻みを使う。"""
    if not _GRID or _GRID[0][-1] < s_max:
        end = max(30.0, 2 * s_max)
        s = [0.0]
        while s[-1] < end:
            s.append(s[-1] + min(max(STEP_MIN_S, STEP_GROWTH * s[-1]), STEP_MAX_S))
        _GRID[:] = [np.asarray(s)]
    g = _GRID[0]
    return g[:int(np.searchsorted(g, s_max, side="left")) + 1]


def _rhs(y: np.ndarray, negk: np.ndarray, g: float) -> np.ndarray:
    """2 次元の状態 (4, B) = (H, Z, U, W) の時間微分。``negk`` = −g / v_t²。"""
    d = np.empty_like(y)
    d[:2] = y[2:]
    sp = negk * np.sqrt(y[2] * y[2] + y[3] * y[3])
    np.multiply(sp, y[2], out=d[2])
    np.multiply(sp, y[3], out=d[3])
    d[3] -= g
    return d


def integrate(u0: np.ndarray, w0: np.ndarray, negk: np.ndarray, g: float, s_max: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """B 本の 2 次元の軌道（打点で H = Z = 0、水平の速さ ``u0``・上向きの速さ ``w0`` [m/s]）を、``time_grid`` の刻みで
    それぞれ ``s_max`` [s] まで RK4 で進める。

    Returns:
        (刻み (N,), 刻みごとの状態 (N, 4, B)（``order`` の順）, ``order``（終わりの遅い順の番号）)。
        軌道 ``order[b]`` の刻み j の状態は ``hist[j, :, b]``（``s_max`` より先の刻みは 0 のまま）
    """
    order = np.argsort(-s_max, kind="stable")
    sm = s_max[order]
    grid = time_grid(float(sm[0]) if len(sm) else 0.0)
    n_nodes = len(grid)
    n_act = np.searchsorted(-sm, -grid[:-1], side="left")  # 刻み j で進める軌道の数（s_max > grid[j]）
    hist = np.zeros((n_nodes, 4, len(sm)))
    y = np.zeros((4, len(sm)))
    y[2], y[3] = u0[order], w0[order]
    nk = negk[order]
    hist[0] = y
    with np.errstate(over="ignore", invalid="ignore"):
        for j in range(n_nodes - 1):
            nb = int(n_act[j])
            if nb == 0:
                break
            h = float(grid[j + 1] - grid[j])
            yy, kk = y[:, :nb], nk[:nb]
            d1 = _rhs(yy, kk, g)
            d2 = _rhs(yy + (0.5 * h) * d1, kk, g)
            d3 = _rhs(yy + (0.5 * h) * d2, kk, g)
            d4 = _rhs(yy + h * d3, kk, g)
            yy = yy + (h / 6.0) * (d1 + 2.0 * (d2 + d3) + d4)
            y[:, :nb] = yy
            hist[j + 1, :, :nb] = yy
    return grid, hist, order


def interpolate(grid: np.ndarray, hist: np.ndarray, b: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``integrate`` の軌道 ``b``（``hist`` の列の番号）の時間 ``s`` [s] の (H, Z) [m]（3 次のエルミート補間）。"""
    if len(grid) < 2:
        return np.zeros(len(s)), np.zeros(len(s))
    j = np.clip(np.searchsorted(grid, s, side="left") - 1, 0, len(grid) - 2)
    h = grid[j + 1] - grid[j]
    u = (s - grid[j]) / h
    a, c = hist[j, :, b], hist[j + 1, :, b]      # (P, 4)
    u2 = u * u
    u3 = u2 * u
    w00, w10, w01, w11 = 2 * u3 - 3 * u2 + 1, (u3 - 2 * u2 + u) * h, -2 * u3 + 3 * u2, (u3 - u2) * h
    with np.errstate(invalid="ignore"):
        H = w00 * a[:, 0] + w10 * a[:, 2] + w01 * c[:, 0] + w11 * c[:, 2]
        Z = w00 * a[:, 1] + w10 * a[:, 3] + w01 * c[:, 1] + w11 * c[:, 3]
    return H, Z


def simulate_batch(p0: np.ndarray, v0: np.ndarray, t_ref: float, times: np.ndarray, g: float, vt: float
                   ) -> np.ndarray:
    """``physics.simulate`` と同じ軌道の位置 (N, 3) [m] を、この計算（``integrate``・``interpolate``）で出す（確かめ用）。
    ``times`` は ``t_ref`` 以後。"""
    p0, v0 = np.asarray(p0, float), np.asarray(v0, float)
    s = np.asarray(times, float) - t_ref
    u = float(np.hypot(v0[0], v0[1]))
    e = np.array([v0[0], v0[1], 0.0]) / u if u > 0 else np.zeros(3)
    grid, hist, _ = integrate(np.array([u]), np.array([v0[2]]), np.array([-g / vt ** 2]), g,
                              np.array([max(float(s.max()), 0.0)]))
    H, Z = interpolate(grid, hist, np.zeros(len(s), int), s)
    return p0 + H[:, None] * e + Z[:, None] * np.array([0.0, 0.0, 1.0])


# =============================================================================
# 残差
# =============================================================================
class _Evaluator:
    """``LsqRequest`` の組の残差を、たくさんの試す点（パラメータ）についてまとめて計算する。"""

    def __init__(self, reqs: Sequence[LsqRequest]):
        self.reqs = list(reqs)
        R = len(self.reqs)
        s_list, tgt_img, tgt_pts, cam_list = [], [], [], []
        self.n_obs = np.zeros(R, int)
        self.kind_img = np.zeros(R, bool)
        self.cameras: list[Any] = []
        cam_ids: dict[int, int] = {}
        for r, q in enumerate(self.reqs):
            if q.kind == "image":
                self.kind_img[r] = True
                ts, cs = [], []
                for o in q.data:
                    ts.append(o.t)
                    tgt_img.append(o.uv)
                    cid = cam_ids.setdefault(id(o.camera), len(self.cameras))
                    if cid == len(self.cameras):
                        self.cameras.append(o.camera)
                    cs.append(np.full(len(o.t), cid))
                t = np.concatenate(ts)
                cam_list.append(np.concatenate(cs))
                tgt_pts.append(np.zeros((len(t), 3)))
            else:
                t, X = q.data
                t = np.asarray(t, float)
                tgt_pts.append(np.asarray(X, float))
                tgt_img.append(np.zeros((len(t), 2)))
                cam_list.append(np.full(len(t), -1))
            s_list.append(t - q.t_ref)
            self.n_obs[r] = len(t)
        self.obs_off = np.concatenate([[0], np.cumsum(self.n_obs)])
        self.s = np.concatenate(s_list) if s_list else np.zeros(0)
        self.uv = np.vstack(tgt_img) if tgt_img else np.zeros((0, 2))
        self.X = np.vstack(tgt_pts) if tgt_pts else np.zeros((0, 3))
        self.cam = np.concatenate(cam_list) if cam_list else np.zeros(0, int)
        self.s_max = np.array([max(float(self.s[self.obs_off[r]:self.obs_off[r + 1]].max(initial=0.0)), 0.0)
                               for r in range(R)])
        # 問題ごとの、パラメータから位置・速度への変換の値
        self.n_kin = np.array([q.model.n_kin for q in self.reqs])
        self.n_plane = np.array([q.model.n_plane for q in self.reqs])
        self.plane_mode = np.array([q.model.mode == "plane" for q in self.reqs])
        self.vt_idx = np.array([q.model.n_kin + q.model.n_plane if q.model.estimate_vt else -1 for q in self.reqs])
        self.vt = np.array([q.model.vt for q in self.reqs], float)
        self.g = np.array([q.base.g for q in self.reqs], float)
        org, drc, nrm = np.zeros((R, 3)), np.zeros((R, 3)), np.zeros((R, 3))
        for r, q in enumerate(self.reqs):
            if q.model.plane is not None:
                org[r], drc[r], nrm[r] = q.model.plane.origin, q.model.plane.direction, q.model.plane.normal
        self.origin, self.direction, self.normal = org, drc, nrm
        self.m_obs = np.where(self.kind_img, 2, 3) * self.n_obs
        self.m = np.array([self.m_obs[r] + self._n_prior(q) for r, q in enumerate(self.reqs)])

    @staticmethod
    def _n_prior(q: LsqRequest) -> int:
        md = q.model
        if q.kind == "points":
            return int(md.max_speed is not None)
        return 2 * (md.n_plane > 0) + 3 * (md.p0_prior is not None) + int(md.max_speed is not None)

    def evaluate(self, rid: np.ndarray, theta: list[np.ndarray]) -> list[np.ndarray]:
        """問題 ``rid[i]`` の試す点 ``theta[i]``（(n_i, パラメータの数)）の残差（(n_i, 残差の数)）のリスト。"""
        rid = np.asarray(rid, int)
        nv = np.array([len(t) for t in theta], int)
        vr = np.repeat(rid, nv)                                   # 点ごとの問題の番号
        V = int(nv.sum())
        th = np.zeros((V, 7))
        i = 0
        for t in theta:
            th[i:i + len(t), :t.shape[1]] = t
            i += len(t)
        # ---- パラメータ → 打点の位置 p0、水平の向き e、水平・上向きの速さ、終端速度
        pm = self.plane_mode[vr]
        has_pl = self.n_plane[vr] > 0
        dpsi = np.where(pm & has_pl, th[:, 4], 0.0)
        dn = np.where(pm & has_pl, th[:, 5], 0.0)
        d0 = self.direction[vr]
        c, sn = np.cos(dpsi), np.sin(dpsi)
        rx, ry = c * d0[:, 0] - sn * d0[:, 1], sn * d0[:, 0] + c * d0[:, 1]
        rn = np.hypot(rx, ry)
        with np.errstate(invalid="ignore", divide="ignore"):
            ex_pl = np.where(has_pl, rx / rn, d0[:, 0])
            ey_pl = np.where(has_pl, ry / rn, d0[:, 1])
        org = self.origin[vr] + dn[:, None] * self.normal[vr]
        p0 = np.empty((V, 3))
        p0[:, 0] = np.where(pm, org[:, 0] + th[:, 0] * ex_pl, th[:, 0])
        p0[:, 1] = np.where(pm, org[:, 1] + th[:, 0] * ey_pl, th[:, 1])
        p0[:, 2] = np.where(pm, th[:, 1], th[:, 2])
        vx_f, vy_f = th[:, 3], th[:, 4]
        uf = np.hypot(vx_f, vy_f)
        with np.errstate(invalid="ignore", divide="ignore"):
            ex = np.where(pm, ex_pl, np.where(uf > 0, vx_f / uf, 0.0))
            ey = np.where(pm, ey_pl, np.where(uf > 0, vy_f / uf, 0.0))
        u0 = np.where(pm, th[:, 2], uf)
        w0 = np.where(pm, th[:, 3], th[:, 5])
        vti = self.vt_idx[vr]
        vt = np.where(vti >= 0, np.exp(th[np.arange(V), np.maximum(vti, 0)]), self.vt[vr])
        g = self.g[vr]
        negk = -g / vt ** 2
        v0 = np.stack([u0 * ex, u0 * ey, w0], axis=1)     # 世界座標の初速（plane モードは vs·d + vz·ez）
        speed = np.sqrt(v0[:, 0] ** 2 + v0[:, 1] ** 2 + v0[:, 2] ** 2)
        ok = speed <= MAX_SIM_SPEED_MPS                    # 速すぎる初速は積分しない（``physics.simulate`` と同じ）
        ok &= np.isfinite(p0).all(axis=1) & np.isfinite(negk)
        # ---- 積分する軌道：問題の最初の点と (u0, w0, k) が同じ点は、最初の点の軌道を使い回す
        start = np.repeat(np.concatenate([[0], np.cumsum(nv)[:-1]]), nv)
        same = (u0 == u0[start]) & (w0 == w0[start]) & (negk == negk[start]) & ok[start]
        traj = np.where(same, start, np.arange(V))
        uniq = np.flatnonzero((traj == np.arange(V)) & ok)
        gs = np.unique(g)
        if len(gs) > 1:
            raise ValueError("重力加速度の違う問題はまとめて解けません")
        grid, hist, order = integrate(u0[uniq], w0[uniq], negk[uniq], float(gs[0]) if len(gs) else 9.81,
                                      self.s_max[vr[uniq]])
        col = np.full(V, -1)
        col[uniq[order]] = np.arange(len(uniq))
        tcol = col[traj]
        # ---- 観測の時刻の位置
        npair = self.n_obs[vr]
        P = int(npair.sum())
        pv = np.repeat(np.arange(V), npair)                     # 組ごとの点の番号
        pstart = np.repeat(np.concatenate([[0], np.cumsum(npair)[:-1]]), npair)
        pobs = self.obs_off[vr][pv] + (np.arange(P) - pstart)   # 組ごとの観測の番号
        s = self.s[pobs]
        good = tcol[pv] >= 0
        H, Z = interpolate(grid, hist, np.maximum(tcol[pv], 0), s)
        pos = p0[pv] + H[:, None] * np.stack([ex, ey, np.zeros(V)], axis=1)[pv]
        pos[:, 2] += Z
        pos[~good] = np.nan
        # ---- 残差（画像の上 [px] または 3 次元 [m]。計算できない所は 1e3 の罰）
        img = self.kind_img[vr][pv]
        res_img = np.full((P, 2), np.nan)
        for cid, cam in enumerate(self.cameras):
            sel = img & (self.cam[pobs] == cid)
            if sel.any():
                res_img[sel] = _project(cam, pos[sel]) - self.uv[pobs[sel]]
        res_pts = pos - self.X[pobs]
        res_img = np.where(np.isfinite(res_img), res_img, 1e3)
        res_pts = np.where(np.isfinite(res_pts), res_pts, 1e3)
        # ---- 事前分布・初速の上限の罰（``physics._Model.prior_residuals``・``speed_residual``）
        out, i, k = [], 0, 0
        for r, n_i in zip(rid, nv):
            q = self.reqs[r]
            md = q.model
            n_o = int(self.n_obs[r])
            F = np.empty((n_i, int(self.m[r])))
            mo = int(self.m_obs[r])
            if self.kind_img[r]:
                F[:, :mo] = res_img[k:k + n_i * n_o].reshape(n_i, mo)
            else:
                F[:, :mo] = res_pts[k:k + n_i * n_o].reshape(n_i, mo)
            col_ = mo
            sl = slice(i, i + n_i)
            if q.kind == "image":
                if md.n_plane:
                    F[:, col_:col_ + 2] = th[sl, md.n_kin:md.n_kin + 2] / md.plane_sigma
                    col_ += 2
                if md.p0_prior is not None:
                    F[:, col_:col_ + 3] = _quadratic(md.prior_px * (p0[sl] - md.p0_prior) / md.p0_prior_sigma,
                                                     md.huber_px)
                    col_ += 3
                if md.max_speed is not None:
                    F[:, col_] = _quadratic(md.speed_weight * np.maximum(speed[sl] - md.max_speed, 0.0)
                                            / SPEED_PENALTY_SIGMA_MPS, md.huber_px)
            elif md.max_speed is not None:
                F[:, col_] = md.speed_weight * np.maximum(speed[sl] - md.max_speed, 0.0) / SPEED_PENALTY_SIGMA_MPS
            out.append(F)
            i += n_i
            k += n_i * n_o
        return out


def _quadratic(r: np.ndarray, d: float | None) -> np.ndarray:
    """``physics._Model._quadratic`` と同じ（Huber 損失の後で罰が r² / 2 になるように引き伸ばす）。"""
    if d is None:
        return r
    a = np.abs(r)
    return np.where(a > d, np.sign(r) * (a * a / (2 * d) + d / 2), r)


def _project(cam: Any, X: np.ndarray) -> np.ndarray:
    """``Camera.project`` と同じ投影（要素ごとの計算で書く。行列の掛け算は配列の長さで丸めが変わることがある）。"""
    R, t = cam.R, cam.tvec
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    xc = R[0, 0] * x + R[0, 1] * y + R[0, 2] * z + t[0]
    yc = R[1, 0] * x + R[1, 1] * y + R[1, 2] * z + t[1]
    zc = R[2, 0] * x + R[2, 1] * y + R[2, 2] * z + t[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        xn, yn = xc / zc, yc / zc
    r2 = xn * xn + yn * yn
    fac = 1.0 + cam.k1 * r2 + cam.k2 * r2 * r2
    uv = np.stack([xn * fac * cam.f + cam.cx, yn * fac * cam.f + cam.cy], axis=1)
    uv[~(zc > 0)] = np.nan
    return uv


# =============================================================================
# 最小 2 乗（scipy の least_squares の trf と同じ手順）
# =============================================================================
def _trf(x0: np.ndarray, f0: np.ndarray, J0: np.ndarray, kw: dict[str, Any]
         ) -> Generator[np.ndarray, tuple[np.ndarray, np.ndarray], Any]:
    """``scipy.optimize._lsq.trf.trf_no_bounds``（``tr_solver="exact"``・``x_scale="jac"``）と同じ手順。

    試す点 x を出し、そこでの残差とヤコビアン（数値微分）を受け取って進む。scipy は受け入れた点でだけヤコビアンを
    計算するが、ここでは残差と一緒に先に計算しておく（受け入れなかった点のヤコビアンは捨てる）。
    """
    ftol, xtol, gtol = kw.get("ftol", 1e-8), kw.get("xtol", 1e-8), kw.get("gtol", 1e-8)
    max_nfev = kw.get("max_nfev")
    m, n = J0.shape
    loss_function = construct_loss_function(m, kw.get("loss", "linear"), kw.get("f_scale", 1.0))
    x = x0.copy()
    f = f0
    f_true = f.copy()
    nfev = 1
    J = J0
    njev = 1
    if loss_function is not None:
        rho = loss_function(f)
        cost = 0.5 * np.sum(rho[0])
        J, f = scale_for_robust_loss_function(J, f, rho)
    else:
        cost = 0.5 * np.dot(f, f)
    g = compute_grad(J, f)
    scale, scale_inv = compute_jac_scale(J)
    Delta = norm(x0 * scale_inv)
    if Delta == 0:
        Delta = 1.0
    if max_nfev is None:
        max_nfev = x0.size * 100
    alpha = 0.0
    termination_status = None
    while True:
        g_norm = norm(g, ord=np.inf)
        if g_norm < gtol:
            termination_status = 1
        if termination_status is not None or nfev == max_nfev:
            break
        d = scale
        g_h = d * g
        J_h = J * d
        U, s, V = svd(J_h, full_matrices=False)
        V = V.T
        uf = U.T.dot(f)
        actual_reduction = -1
        while actual_reduction <= 0 and nfev < max_nfev:
            step_h, alpha, _ = solve_lsq_trust_region(n, m, uf, s, V, Delta, initial_alpha=alpha)
            predicted_reduction = -evaluate_quadratic(J_h, g_h, step_h)
            step = d * step_h
            x_new = x + step
            f_new, J_new = yield x_new
            nfev += 1
            step_h_norm = norm(step_h)
            if not np.all(np.isfinite(f_new)):
                Delta = 0.25 * step_h_norm
                continue
            if loss_function is not None:
                cost_new = loss_function(f_new, cost_only=True)
            else:
                cost_new = 0.5 * np.dot(f_new, f_new)
            actual_reduction = cost - cost_new
            Delta_new, ratio = update_tr_radius(Delta, actual_reduction, predicted_reduction, step_h_norm,
                                                step_h_norm > 0.95 * Delta)
            step_norm = norm(step)
            termination_status = check_termination(actual_reduction, cost, step_norm, norm(x), ratio, ftol, xtol)
            if termination_status is not None:
                break
            alpha *= Delta / Delta_new
            Delta = Delta_new
        if actual_reduction > 0:
            x = x_new
            f = f_new
            f_true = f.copy()
            cost = cost_new
            J = J_new
            njev += 1
            if loss_function is not None:
                rho = loss_function(f)
                J, f = scale_for_robust_loss_function(J, f, rho)
            g = compute_grad(J, f)
            scale, scale_inv = compute_jac_scale(J, scale_inv)
    if termination_status is None:
        termination_status = 0
    return OptimizeResult(x=x, cost=cost, fun=f_true, jac=J, grad=g, optimality=g_norm,
                          active_mask=np.zeros_like(x), nfev=nfev, njev=njev, status=termination_status,
                          message=TERMINATION_MESSAGES[termination_status], success=termination_status > 0)


MAX_PAIRS = 1_500_000  # 1 回の残差の計算で扱う（試す点, 観測）の組の数の上限（メモリ。1 組あたり 0.3 kB ほど）


def _chunks(sizes: list[int], limit: int) -> list[tuple[int, int]]:
    """大きさ ``sizes`` の並びを、合計が ``limit`` を超えない区切り (始め, 終わり) に分ける（1 つで超えるものは 1 つで）。"""
    out, lo, tot = [], 0, 0
    for i, s in enumerate(sizes):
        if i > lo and tot + s > limit:
            out.append((lo, i))
            lo, tot = i, 0
        tot += s
    if lo < len(sizes):
        out.append((lo, len(sizes)))
    return out


def _batchable(q: LsqRequest) -> bool:
    kw = q.kwargs
    return (kw.get("x_scale") == "jac" and kw.get("method", "trf") == "trf" and kw.get("jac", "2-point") == "2-point"
            and kw.get("bounds") is None and kw.get("diff_step") is not None)


def solve_batch(reqs: Sequence[LsqRequest]) -> list[Any]:
    """最小 2 乗問題 ``reqs`` をまとめて解く。``LsqRequest.solve_cpu``（``least_squares``）と同じ形の結果、解けなければ
    その例外のリスト。打った時刻より前の観測がある問題・``least_squares`` の別の設定の問題は 1 つずつ解く。"""
    out: list[Any] = [None] * len(reqs)
    idx = []
    for i, q in enumerate(reqs):
        s_min = min((float(np.min(o.t)) for o in q.data if len(o.t)), default=q.t_ref) if q.kind == "image" \
            else float(np.min(q.data[0], initial=q.t_ref))
        if AVAILABLE and _batchable(q) and s_min >= q.t_ref:
            idx.append(i)
        else:
            try:
                out[i] = q.solve_cpu()
            except Exception as e:  # noqa: BLE001（呼ぶ側で 1 つずつの当てはめと同じに扱う）
                out[i] = e
    if not idx:
        return out
    sub = [reqs[i] for i in idx]
    ev = _Evaluator(sub)
    x_try = [np.asarray(q.x0, float).copy() for q in sub]
    gens: list[Any] = [None] * len(sub)
    active = list(range(len(sub)))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        while active:
            thetas, dxs = [], []
            for a in active:
                x = x_try[a]
                h = _compute_absolute_step(sub[a].kwargs["diff_step"], x, x, "2-point")
                X = np.repeat(x[None], len(x) + 1, axis=0)
                X[1 + np.arange(len(x)), np.arange(len(x))] = x + h
                thetas.append(X)
                dxs.append((x + h) - x)
            F = []
            for lo, hi in _chunks([len(t) * int(ev.n_obs[a]) for a, t in zip(active, thetas)], MAX_PAIRS):
                F += ev.evaluate(np.asarray(active[lo:hi]), thetas[lo:hi])
            nxt = []
            for a, Fa, dx in zip(active, F, dxs):
                f = Fa[0].copy()
                J = ((Fa[1:] - Fa[0]) / dx[:, None]).T
                try:
                    if gens[a] is None:
                        if not np.all(np.isfinite(f)):
                            raise ValueError("Residuals are not finite in the initial point.")
                        gens[a] = _trf(x_try[a], f, J, sub[a].kwargs)
                        x_try[a] = next(gens[a])
                    else:
                        x_try[a] = gens[a].send((f, J))
                    nxt.append(a)
                except StopIteration as stop:
                    out[idx[a]] = stop.value
                except Exception as e:  # noqa: BLE001
                    out[idx[a]] = e
            active = nxt
    return out


# =============================================================================
# 当てはめ（physics.fit_trajectory）をまとめて
# =============================================================================
def fit_trajectories(jobs: Sequence[tuple]) -> list[FitResult | Exception]:
    """``physics.fit_trajectory(*job)`` を全部の ``jobs`` についてまとめて計算する（結果、計算できなければ例外のリスト）。

    ``BatchGate`` の中のスレッドから呼ぶと、ほかのスレッドの当てはめと一緒に解く。
    """
    gate = _GATE[0] if _GATE else None
    if gate is not None and gate.member():
        return gate.run(list(jobs))
    return _fit_now(list(jobs))


def _fit_now(jobs: list[tuple]) -> list[FitResult | Exception]:
    out: list[Any] = [None] * len(jobs)
    pending: dict[int, tuple[Any, list[LsqRequest]]] = {}
    for i, job in enumerate(jobs):
        plan = fit_plan(*job)
        try:
            pending[i] = (plan, next(plan))
        except StopIteration as stop:
            out[i] = stop.value
        except Exception as e:  # noqa: BLE001
            out[i] = e
    while pending:
        flat = [(i, r) for i, (_, reqs) in pending.items() for r in reqs]
        sols = solve_batch([r for _, r in flat])
        got: dict[int, list[Any]] = {i: [] for i in pending}
        for (i, _), s in zip(flat, sols):
            got[i].append(s)
        nxt = {}
        for i, (plan, _) in pending.items():
            err = next((s for s in got[i] if isinstance(s, BaseException)), None)
            try:
                if err is not None:
                    plan.close()
                    raise err
                nxt[i] = (plan, plan.send(got[i]))
            except StopIteration as stop:
                out[i] = stop.value
            except Exception as e:  # noqa: BLE001
                out[i] = e
        pending = nxt
    return out


class BatchGate:
    """いくつかのスレッド（ラリーごとの当てはめ）の ``fit_trajectories`` を、まとめて 1 回で解く。

    ``n_tasks`` 個の仕事（ラリー）を ``n_threads`` 個のスレッドで進める。仕事の始めに ``enter``、終わりに ``leave`` を呼ぶ。
    始まるはずの仕事が全部 ``enter`` し、``enter`` した仕事が全部 ``fit_trajectories`` で待つか ``leave`` したら、
    待っている当てはめをまとめて計算する（最後に来たスレッドが計算する）。結果は 1 つずつ解いたときと同じ。
    """

    def __init__(self, n_threads: int = 1, n_tasks: int = 1) -> None:
        self._cv = threading.Condition()
        self._members: set[int] = set()
        self._waiting: list[dict[str, Any]] = []
        self._expected = max(0, min(n_threads, n_tasks))   # まだ ``enter`` していない、すぐに始まるはずの仕事
        self._unstarted = max(0, n_tasks)

    def member(self) -> bool:
        with self._cv:
            return threading.get_ident() in self._members

    def enter(self) -> None:
        with self._cv:
            self._members.add(threading.get_ident())
            self._expected = max(0, self._expected - 1)
            self._unstarted = max(0, self._unstarted - 1)

    def leave(self) -> None:
        with self._cv:
            self._members.discard(threading.get_ident())
            if self._unstarted > 0:
                self._expected += 1      # 空いたスレッドが次の仕事を始める
            self._cv.notify_all()
        self._maybe_run()

    def run(self, jobs: list[tuple]) -> list[Any]:
        slot: dict[str, Any] = {"jobs": jobs, "out": None}
        with self._cv:
            self._waiting.append(slot)
        self._maybe_run()
        with self._cv:
            while slot["out"] is None:
                self._cv.wait()
        if isinstance(slot["out"], BaseException):
            raise slot["out"]
        return slot["out"]

    def _maybe_run(self) -> None:
        with self._cv:
            if not self._waiting or self._expected > 0 or len(self._waiting) < len(self._members):
                return
            slots, self._waiting = self._waiting, []
        try:
            res = _fit_now([j for s in slots for j in s["jobs"]])
            k = 0
            for s in slots:
                s["result"] = res[k:k + len(s["jobs"])]
                k += len(s["jobs"])
        except BaseException as e:  # noqa: BLE001
            for s in slots:
                s["result"] = e
        with self._cv:
            for s in slots:
                s["out"] = s.pop("result")
            self._cv.notify_all()


_GATE: list[BatchGate] = []


class gate_session:
    """この中で ``BatchGate`` を使う（``with gate_session(n_threads, n_tasks) as gate:``。仕事ごとに ``gate.enter()``・
    ``leave()``）。"""

    def __init__(self, n_threads: int = 1, n_tasks: int = 1) -> None:
        self.n_threads, self.n_tasks = n_threads, n_tasks

    def __enter__(self) -> BatchGate:
        _GATE[:] = [BatchGate(self.n_threads, self.n_tasks)]
        return _GATE[0]

    def __exit__(self, *exc: Any) -> None:
        _GATE.clear()
