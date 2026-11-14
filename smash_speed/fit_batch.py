"""軌道の当てはめを、何本もまとめて解く（``physics.fit_trajectory`` の plane モードと同じ問題・同じ手順）。

ショットの一覧では、1 本のショットに 10〜30 回の当てはめ（平面の格子・打点をつないだ当てはめ直し・信頼区間）があり、
1 回ずつ別のプロセスで ``scipy`` の積分（``solve_ivp``）と最適化（``least_squares``）を動かすと、時間の 9 割が積分
（そのうち 8 割は数値微分のヤコビアンのための積分）になる（``docs/speed.md``）。ここでは、同じ計算を
「何百もの当てはめ × 配列の演算」にする。

- **積分**：打点を原点、飛行平面の中の 2 次元（平面に沿った向き s・高さ z）の軌道を、すべての当てはめで同じ刻みの
  RK4 で進める（``integrate``）。初速 (vs, vz) についての傾き（変分方程式）も一緒に進めるので、ヤコビアンは数値微分ではなく
  式から出る。観測の時刻の位置は、刻みの間を 3 次のエルミート補間で出す。
  刻みは、打った直後は細かく、後は打ってからの時間に比例して広げる（空気抵抗が速さの 2 乗に比例するので、打ってから
  τ 秒後の速さは初速によらず 1/(kτ) 以下。``time_grid``）。``physics.simulate``（DOP853）との差は 3 µm 以下
  （初速 170 m/s まで。画像の上で 0.001 px ほど）。
- **最適化**：``scipy.optimize.least_squares``（``method="trf"``・範囲なし・``x_scale="jac"``）と同じ信頼領域法を、当てはめごとに
  状態を持たせてまとめて進める（``_trf``）。同じ損失（Huber・soft_l1）・同じ終わりの判定・同じ評価の回数の数え方で、
  同じ残差とヤコビアンを渡せば scipy と同じ道すじをたどる（テストで確かめている）。
- **手順**：``fit_trajectory`` と同じ（3 次元点への当てはめで初期値 → 初期値ごとに解いてずれの小さい方 → 打点の目安の重みを
  直して解き直す → 共分散と信頼区間）。
- **まとめ方**：当てはめの時間のほとんどは「ほかが終わった後に残った数本の、何百回もの繰り返し」で、NumPy では 1 回の積分の
  手間が本数にあまりよらない。そこで、最適化は積分を自分ではせず、積分の要求を ``yield`` で外へ出し（``Steps``）、
  同時に進んでいるすべての計算（ラリー・当てはめの形）の要求を 1 回の積分にまとめる（``serve``・``together``・``run``）。

1 回ずつの当てはめと違うのは、積分の誤差（上のとおり）と、ヤコビアン（数値微分 → 式）だけ。最適化の途中の道すじは
少し変わるので、結果は丸めではなく「終わりの判定の幅」（``FitOptions.ftol`` = 1e-5）の中で一致する。道すじが不安定な
当てはめ（合わない平面・別の物の軌道で、ずれが何十 px もあるもの）は結果が変わることがある（``docs/speed.md`` の比べた結果）。

積分は ``backend`` で差し替えられる（``numpy``：CPU の倍精度。``dml``：onnxruntime-directml で GPU の単精度。
``fit_gpu``）。GPU は 1 回の積分の手間が大きく、この問題では速くならない（``docs/gpu.md``）。まとめて解けない当てはめ
（終端速度の推定・ブートストラップ・打点より前の観測）は、1 本ずつ ``fit_trajectory`` で解く。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generator

import numpy as np

from smash_speed.camera import Camera
from smash_speed.geometry import FlightPlane
from smash_speed.physics import (MAX_SIM_SPEED_MPS, SPEED_PENALTY_SIGMA_MPS, FitOptions, FitResult, Observations,
                                 PhysicsParams, fit_trajectory)

EPS = float(np.finfo(float).eps)
_BAD = 1e3                      # 計算できなかった残差の罰（``physics._residuals`` と同じ）
_MESSAGES = {                   # ``scipy.optimize.least_squares`` の終わり方の文
    0: "The maximum number of function evaluations is exceeded.",
    1: "`gtol` termination condition is satisfied.",
    2: "`ftol` termination condition is satisfied.",
    3: "`xtol` termination condition is satisfied.",
    4: "Both `ftol` and `xtol` termination conditions are satisfied.",
}
# 積分の刻み [s]：打った直後は STEP_MIN_S、その後は「打ってからの時間 × STEP_GROWTH」、STEP_MAX_S まで
STEP_MIN_S = 5e-4
STEP_GROWTH = 0.05
STEP_MAX_S = 0.04
CHUNK = 4096                    # 1 回の積分でまとめる当てはめの数の上限（メモリ：1 本 × 1 刻みで 96 バイト）
_TINY = 1e-300
# 1 回ずつの当てはめ（``fit_trajectory``）のヤコビアンは、パラメータを「その値 × ``FitOptions.diff_step``」だけ動かした
# 数値微分で、値がごく小さい（0 ではない）と動かす幅が倍精度の丸めより小さくなり、観測の行の傾きがちょうど 0 になる。
# 平面の向きの補正 dψ は 0 から始まり、最初の歩が小さいとそうなって、その後は動かなくなる（``docs/speed.md``）。
# 同じ結果にするために、これを再現する。偽にすると、式どおりの傾きで解く（合わなくなる当てはめが出る）
EMULATE_FD = True
_FD_DEAD = 1e-16                # 動かす幅がこれより小さいパラメータは、数値微分では観測が動かない


@dataclass
class FitJob:
    """``fit_trajectory(obs, t_ref, "plane", plane, phys, opts, theta0)`` 1 回分の引数。"""

    obs: Observations
    t_ref: float
    plane: FlightPlane
    phys: PhysicsParams
    opts: FitOptions
    theta0: np.ndarray | None = None


# =============================================================================
# 積分
# =============================================================================
_grid_cache: list[np.ndarray] = [np.zeros(1)]


def time_grid(t_end: float) -> np.ndarray:
    """打ってからの時間 [s] の刻み（0 から ``t_end`` を越える所まで）。どこまで作っても、前の方は同じ。"""
    g = _grid_cache[0]
    if g[-1] < t_end:
        nodes = g.tolist()
        while nodes[-1] < t_end:
            nodes.append(nodes[-1] + min(max(STEP_MIN_S, STEP_GROWTH * nodes[-1]), STEP_MAX_S))
        g = _grid_cache[0] = np.array(nodes)
    n = int(np.searchsorted(g, t_end, side="left")) + 1
    return g[:max(n, 2)]


def _acc(w: np.ndarray, ikg: np.ndarray) -> np.ndarray:
    """速度と、その初速 (vs, vz) についての傾き ``w`` (3, B)（複素数。実部が平面に沿った向き、虚部が上向き）の時間微分。

    行 0 が k × 速度、行 1・2 が速度の vs0・vz0 についての傾き。速度に抵抗係数 k をかけておくと、運動方程式は
    d(kv)/dt = −|kv|·kv − k·g·ez で、k をかける演算がいらない。2 次元のベクトルを複素数 1 つで持つのは、
    NumPy を呼ぶ回数を減らすため（残った数本の当てはめの繰り返しでは、時間のほとんどが呼び出しの手間）。
    """
    z = w[0]
    q = np.abs(z)
    a = q * w
    np.maximum(q, _TINY, out=q)
    # 速さの傾き dq = (v · dv) / q
    c = (w[1:] * np.conj(z)).real
    c /= q
    a[1:] += c * z
    np.negative(a, out=a)
    a[0] -= ikg
    return a


def integrate(v0: np.ndarray, k: np.ndarray, g: np.ndarray, grid: np.ndarray, n_steps: np.ndarray | None = None
              ) -> tuple[np.ndarray, np.ndarray]:
    """原点を速度 ``v0`` (B, 2) [m/s]（平面に沿った向き・上向き）で出た B 本の軌道を、``grid`` [s] の時刻で進める（RK4）。

    Args:
        k: 抵抗係数 g / v_t² (B,) [1/m]
        g: 重力加速度 (B,) [m/s²]
        n_steps: 軌道ごとの、進める刻みの数（省略時は全部）。**大きい順に並んでいること**（長く飛んだ順）。
            それより先の値は使えない

    Returns:
        (位置 (len(grid), 3, B) [m], 速度 (len(grid), 3, B) [m/s])。複素数（実部が s、虚部が z）で、2 つ目の軸は
        [値, vs0 についての傾き, vz0 についての傾き]
    """
    b, n = len(v0), len(grid) - 1
    pos = np.empty((n + 1, 3, b), complex)
    vel = np.empty((n + 1, 3, b), complex)
    p, w = pos[0], vel[0]
    p[:] = 0.0
    w[0] = (v0[:, 0] + 1j * v0[:, 1]) * k
    w[1], w[2] = 1.0, 1j
    ikg = 1j * k * g
    # 刻み i を進める軌道の本数（n_steps > i の本数）
    count = np.full(n, b) if n_steps is None else np.searchsorted(-n_steps, -np.arange(n), side="left")
    h_all = np.diff(grid).tolist()
    m = b
    tmp = np.empty((3, b), complex)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for i, h in enumerate(h_all):
            m_new = int(count[i])
            if m_new < m:
                if m_new == 0:
                    break
                m = m_new
                p, w, ikg, tmp = p[:, :m], w[:, :m], ikg[:m], tmp[:, :m]
            a1 = _acc(w, ikg)
            np.multiply(a1, 0.5 * h, out=tmp)
            tmp += w
            a2 = _acc(tmp, ikg)
            np.multiply(a2, 0.5 * h, out=tmp)
            tmp += w
            a3 = _acc(tmp, ikg)
            np.multiply(a3, h, out=tmp)
            tmp += w
            a4 = _acc(tmp, ikg)
            a2 += a3
            a1 += a2
            # 位置：p + h·w + (h²/6)(a1 + a2 + a3)、速度：w + (h/6)(a1 + 2 a2 + 2 a3 + a4)
            p_new, w_new = pos[i + 1, :, :m], vel[i + 1, :, :m]
            np.multiply(w, h, out=p_new)
            p_new += p
            np.multiply(a1, h * h / 6.0, out=tmp)
            p_new += tmp
            a1 += a2
            a1 += a4
            np.multiply(a1, h / 6.0, out=w_new)
            w_new += w
            p, w = p_new, w_new
        # k をかけて進めた値を戻す（傾きの行は、初速にも k がかかっているので、そのまま）
        pos[:, 0] /= k
        vel[:, 0] /= k
    return pos, vel


def interpolate(grid: np.ndarray, pos: np.ndarray, vel: np.ndarray, tau: np.ndarray, cols: np.ndarray | None = None
                ) -> np.ndarray:
    """``integrate`` の結果を、軌道ごとの時間 ``tau`` (B, M) [s] で補間した位置 (B, M, 3, 2)（3 次のエルミート補間）。

    ``cols`` は、``tau`` の行ごとの、``pos``・``vel`` の中の軌道の番号（省略時は 0, 1, 2, …）。
    最後の 2 つの軸は、(値・vs0 についての傾き・vz0 についての傾き) × (s, z)。
    """
    i = np.clip(np.searchsorted(grid, tau, side="right") - 1, 0, len(grid) - 2)
    col = (np.arange(tau.shape[0]) if cols is None else cols)[:, None]
    h = (grid[i + 1] - grid[i])[:, :, None]
    u = ((tau - grid[i]) / (grid[i + 1] - grid[i]))[:, :, None]
    u2 = u * u
    u3 = u2 * u
    with np.errstate(invalid="ignore", over="ignore"):
        out = ((2 * u3 - 3 * u2 + 1) * pos[i, :, col] + (u3 - 2 * u2 + u) * h * vel[i, :, col]
               + (3 * u2 - 2 * u3) * pos[i + 1, :, col] + (u3 - u2) * h * vel[i + 1, :, col])
    return np.ascontiguousarray(out).view(float).reshape(tau.shape + (3, 2))


Integrator = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
# 積分の要求：(初速 (B, 2), 抵抗係数 (B,), 重力加速度 (B,), 位置を見る時間 (B, M)（NaN は使わない所）)
Request = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
# 積分の要求を出して、その結果（``interpolate`` の位置 (B, M, 3, 2)）を受け取りながら進む計算
Steps = Generator[list[Request], list[np.ndarray], Any]


def get_integrator(backend: str | None) -> tuple[Integrator, str]:
    """積分の計算の仕方 ``backend``（"numpy"・"dml"）と、実際に使う名前。

    "dml" が使えない PC（onnxruntime-directml がない・GPU で動かない）では、警告を 1 回出して "numpy" にする
    （理由は ``fit_gpu.why_not``）。
    """
    if backend == "dml":
        from smash_speed import fit_gpu

        if fit_gpu.available():
            return fit_gpu.integrate, "dml"
    return integrate, "numpy"


def serve(requests: list[Request], integrator: Integrator = integrate) -> list[np.ndarray]:
    """いくつかの積分の要求を、1 回の積分にまとめて計算する。"""
    if not requests:
        return []
    sizes = [len(r[0]) for r in requests]
    v0, k, g = (np.concatenate([r[j] for r in requests]) for j in range(3))
    with np.errstate(invalid="ignore"):
        t_max = np.concatenate([np.max(np.where(np.isfinite(r[3]), r[3], 0.0), axis=1, initial=0.0)
                                for r in requests])
    owner = np.repeat(np.arange(len(requests)), sizes)
    local = np.concatenate([np.arange(n) for n in sizes])
    out = [np.empty(r[3].shape + (3, 2)) for r in requests]
    order = np.argsort(-t_max, kind="stable")
    for at in range(0, len(order), CHUNK):
        idx = order[at:at + CHUNK]
        grid = time_grid(float(t_max[idx[0]]))
        n_steps = np.maximum(np.searchsorted(grid, t_max[idx], side="left"), 1)
        pos, vel = integrator(v0[idx], k[idx], g[idx], grid, n_steps)
        own = owner[idx]
        for r in np.unique(own):
            cols = np.nonzero(own == r)[0]
            rows = local[idx[cols]]
            tau = requests[r][3][rows]
            out[r][rows] = interpolate(grid, pos, vel, np.where(np.isfinite(tau), tau, 0.0), cols)
    return out


def trajectories(v0: np.ndarray, k: np.ndarray, g: np.ndarray, tau: np.ndarray, integrator: Integrator = integrate
                 ) -> np.ndarray:
    """速度 ``v0`` (B, 2) で原点を出た軌道の、時間 ``tau`` (B, M) [s]（NaN は使わない所）の位置と傾き (B, M, 3, 2)。"""
    return serve([(v0, k, g, tau)], integrator)[0]


def together(tasks: list[Steps]) -> Steps:
    """いくつかの計算を、積分の要求をまとめながら同時に進める。戻り値は、それぞれの戻り値のリスト。"""
    out: list[Any] = [None] * len(tasks)
    pending: dict[int, list[Request]] = {}
    for j, task in enumerate(tasks):
        try:
            pending[j] = next(task)
        except StopIteration as e:
            out[j] = e.value
    while pending:
        order = list(pending)
        res = yield [r for j in order for r in pending[j]]
        at = 0
        for j in order:
            n = len(pending[j])
            try:
                pending[j] = tasks[j].send(res[at:at + n])
            except StopIteration as e:
                out[j] = e.value
                del pending[j]
            at += n
    return out


def run(task: Steps, backend: str | None = None) -> Any:
    """積分の要求を出しながら進む計算 ``task`` を、最後まで動かして戻り値を返す。"""
    integrator, _ = get_integrator(backend)
    try:
        req = next(task)
        while True:
            req = task.send(serve(req, integrator))
    except StopIteration as e:
        return e.value


# =============================================================================
# 信頼領域法（scipy の trf_no_bounds・tr_solver="exact"・x_scale="jac" と同じ）
# =============================================================================
def _rho(f: np.ndarray, f_scale: np.ndarray, loss: str, cost_only: bool = False) -> Any:
    """損失 ρ（``scipy.optimize._lsq.least_squares`` と同じ式）。``f`` (B, M)、``f_scale`` (B,)。

    Returns:
        ``cost_only`` ならコスト (B,)。そうでなければ (コスト, ρ', ρ'')（どれも f_scale の 2 乗をかけ直した後の値）
    """
    s2 = (f_scale * f_scale)[:, None]
    z = f * f / s2
    with np.errstate(divide="ignore", invalid="ignore"):
        if loss == "huber":
            big = z > 1
            r0 = np.where(big, 2 * np.sqrt(z) - 1, z)
            if cost_only:
                return 0.5 * s2[:, 0] * r0.sum(axis=1)
            r1 = np.where(big, z ** -0.5, 1.0)
            r2 = np.where(big, -0.5 * z ** -1.5, 0.0)
        elif loss == "soft_l1":
            t = 1 + z
            r0 = 2 * (np.sqrt(t) - 1)
            if cost_only:
                return 0.5 * s2[:, 0] * r0.sum(axis=1)
            r1 = t ** -0.5
            r2 = -0.5 * t ** -1.5
        else:
            raise ValueError(f"まとめて解けない損失です：{loss}")
    return 0.5 * (r0 * s2).sum(axis=1), r1, r2 / s2


def _scaled(f: np.ndarray, jac: np.ndarray, f_scale: np.ndarray, loss: str
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """残差とヤコビアンに損失の重みをかける（``scale_for_robust_loss_function``）。(コスト, 残差, ヤコビアン)。"""
    if loss == "linear":
        return 0.5 * np.einsum("bm,bm->b", f, f), f, jac
    cost, r1, r2 = _rho(f, f_scale, loss)
    js = r1 + 2 * r2 * f * f
    js = np.sqrt(np.where(js < EPS, EPS, js))
    return cost, f * r1 / js, jac * js[:, :, None]


def _cost(f: np.ndarray, f_scale: np.ndarray, loss: str) -> np.ndarray:
    return 0.5 * np.einsum("bm,bm->b", f, f) if loss == "linear" else _rho(f, f_scale, loss, cost_only=True)


def _tr_step(uf: np.ndarray, s: np.ndarray, v: np.ndarray, delta: np.ndarray, alpha0: np.ndarray, m: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray]:
    """信頼領域の中の最小 2 乗の 1 歩（``solve_lsq_trust_region``）。(歩 (B, n), LM のパラメータ α (B,))。"""
    n = s.shape[1]
    suf = s * uf
    full = (m >= n) & (s[:, -1] > EPS * m * s[:, 0])
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        p_gn = -np.einsum("bij,bj->bi", v, uf / s)
        gn = full & (np.linalg.norm(p_gn, axis=1) <= delta)
        a_up = np.linalg.norm(suf, axis=1) / delta

        def phi_and_derivative(alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            denom = s * s + alpha[:, None]
            p_norm = np.linalg.norm(suf / denom, axis=1)
            return p_norm - delta, -np.sum(suf * suf / denom ** 3, axis=1) / p_norm

        phi, phi_prime = phi_and_derivative(np.zeros_like(delta))
        a_lo = np.where(full, -phi / phi_prime, 0.0)
        restart = np.maximum(0.001 * a_up, np.sqrt(a_lo * a_up))
        alpha = np.where(~full & (alpha0 == 0), restart, alpha0)
        run = ~gn
        for _ in range(10):
            if not run.any():
                break
            alpha = np.where(run & ((alpha < a_lo) | (alpha > a_up)), np.maximum(0.001 * a_up, np.sqrt(a_lo * a_up)),
                             alpha)
            phi, phi_prime = phi_and_derivative(alpha)
            a_up = np.where(run & (phi < 0), alpha, a_up)
            ratio = phi / phi_prime
            a_lo = np.where(run, np.maximum(a_lo, alpha - ratio), a_lo)
            alpha = np.where(run, alpha - (phi + delta) * ratio / delta, alpha)
            run = run & ~(np.abs(phi) < 0.01 * delta)
        p = -np.einsum("bij,bj->bi", v, suf / (s * s + alpha[:, None]))
        p = p * (delta / np.linalg.norm(p, axis=1))[:, None]
    return np.where(gn[:, None], p_gn, p), np.where(gn, 0.0, alpha)


def _trf(ev: Any, x0: np.ndarray, m: np.ndarray, loss: str, f_scale: np.ndarray, ftol: np.ndarray, xtol: np.ndarray,
         gtol: float, max_nfev: np.ndarray, rows: np.ndarray | None = None) -> Steps:
    """B 個の最小 2 乗問題を、``least_squares(method="trf", x_scale="jac", loss=loss)`` と同じ手順でまとめて解く。

    残差の評価は 2 つに分かれる：``ev.request(番号, x)`` が軌道の積分の要求を作り、積分の結果 ``tr`` を受け取って
    ``ev.residuals(番号, x, tr)`` が (残差 (b, M), ヤコビアン (b, M, n)) を返す（使わない行は 0）。積分はここではせず、
    要求を ``yield`` で外へ出す（ほかの計算の要求とまとめて 1 回で積分するため。``together``・``run``）。

    Args:
        x0: 初期値 (B, n)
        m: 問題ごとの本当の残差の数 (B,)
        max_nfev: 問題ごとの評価の回数の上限 (B,)
        rows: ``ev`` に渡す問題の番号（省略時は 0, 1, 2, …）

    Returns:
        ``x``・``cost``・``fun``（損失をかける前の残差）・``jac``（損失の重みをかけたヤコビアン）・``status``
        （scipy と同じ。初期値で残差が計算できなければ −1）・``nfev``
    """
    b_all, n = x0.shape
    every = np.arange(b_all)
    rows = every if rows is None else rows
    x = np.array(x0, float)
    f_true, jac = ev.residuals(rows, x, (yield [ev.request(rows, x)])[0])
    status = np.full(b_all, -9)
    status[~(np.isfinite(f_true).all(axis=1) & np.isfinite(x).all(axis=1))] = -1
    f_true = np.where(status[:, None] == -1, 0.0, f_true)
    jac = np.where(status[:, None, None] == -1, 0.0, jac)
    nfev = np.ones(b_all, int)
    cost, f, jac = _scaled(f_true, jac, f_scale, loss)
    grad = np.einsum("bmi,bm->bi", jac, f)
    scale_inv = np.sqrt(np.einsum("bmi,bmi->bi", jac, jac))
    scale_inv[scale_inv == 0] = 1.0
    delta = np.linalg.norm(x * scale_inv, axis=1)
    delta[delta == 0] = 1.0
    alpha = np.zeros(b_all)
    sv = np.zeros((b_all, n))
    vv = np.zeros((b_all, n, n))
    uf = np.zeros((b_all, n))
    outer = status == -9            # 次に、新しいヤコビアンで分解し直す問題
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        while True:
            io = every[outer & (status == -9)]
            if len(io):
                status[io[np.abs(grad[io]).max(axis=1) < gtol]] = 1
                io = io[status[io] == -9]
                jh = jac[io] * (1.0 / scale_inv[io])[:, None, :]
                u_, s_, vt_ = np.linalg.svd(jh, full_matrices=False)
                sv[io], vv[io] = s_, np.swapaxes(vt_, 1, 2)
                uf[io] = np.einsum("bmi,bm->bi", u_, f[io])
                outer[io] = False
            status[(status == -9) & (nfev >= max_nfev)] = 0
            ia = every[status == -9]
            if not len(ia):
                break
            d = 1.0 / scale_inv[ia]
            step_h, a_new = _tr_step(uf[ia], sv[ia], vv[ia], delta[ia], alpha[ia], m[ia])
            step = d * step_h
            js = np.einsum("bmi,bi->bm", jac[ia], step)
            predicted = -(0.5 * np.einsum("bm,bm->b", js, js) + np.einsum("bi,bi->b", step_h, d * grad[ia]))
            x_new = x[ia] + step
            f_new, jac_new = ev.residuals(rows[ia], x_new, (yield [ev.request(rows[ia], x_new)])[0])
            nfev[ia] += 1
            step_h_norm = np.linalg.norm(step_h, axis=1)
            finite = np.isfinite(f_new).all(axis=1)
            cost_new = _cost(np.where(finite[:, None], f_new, 0.0), f_scale[ia], loss)
            actual = np.where(finite, cost[ia] - cost_new, -1.0)
            ratio = np.where(predicted > 0, actual / predicted, np.where((predicted == 0) & (actual == 0), 1.0, 0.0))
            delta_new = np.where(ratio < 0.25, 0.25 * step_h_norm,
                                 np.where((ratio > 0.75) & (step_h_norm > 0.95 * delta[ia]), 2.0 * delta[ia],
                                          delta[ia]))
            step_norm = np.linalg.norm(step, axis=1)
            x_norm = np.linalg.norm(x[ia], axis=1)
            ft = (actual < ftol[ia] * cost[ia]) & (ratio > 0.25)
            xt = step_norm < xtol[ia] * (xtol[ia] + x_norm)
            term = np.where(ft & xt, 4, np.where(ft, 2, np.where(xt, 3, -9)))
            term = np.where(finite, term, -9)
            alpha[ia] = np.where(finite, a_new * delta[ia] / delta_new, a_new)
            delta[ia] = np.where(finite, delta_new, 0.25 * step_h_norm)
            acc = actual > 0
            k = ia[acc]
            if len(k):
                x[k], f_true[k] = x_new[acc], f_new[acc]
                cost[k], f[k], jac[k] = _scaled(f_new[acc], jac_new[acc], f_scale[k], loss)
                grad[k] = np.einsum("bmi,bm->bi", jac[k], f[k])
                scale_inv[k] = np.maximum(np.sqrt(np.einsum("bmi,bmi->bi", jac[k], jac[k])), scale_inv[k])
                outer[k] = True
            status[ia] = term
            # 終わった問題のうち、最後の歩を受け入れたものは、新しい傾きが小さければ gtol で終わったことになる
            done = ia[acc & (term != -9)]
            status[done[np.abs(grad[done]).max(axis=1) < gtol]] = 1
    return {"x": x, "cost": cost, "fun": f_true, "jac": jac, "status": status, "nfev": nfev}


# =============================================================================
# 残差（plane モード）
# =============================================================================
def _quadratic(r: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``physics._Model._quadratic`` と、その傾き。``d``（切り替え点。NaN なら何もしない）は ``r`` に合わせて広がる形。"""
    a = np.abs(r)
    with np.errstate(invalid="ignore"):
        big = a > d
    dd = np.where(big, d, 1.0)
    return np.where(big, np.sign(r) * (a * a / (2 * dd) + dd / 2), r), np.where(big, a / dd, 1.0)


def _fd_dead(theta: np.ndarray, diff_step: np.ndarray) -> np.ndarray:
    """``theta`` (b, n) のうち、1 回ずつの当てはめの数値微分では観測が動かないパラメータ（``EMULATE_FD``）。"""
    a = np.abs(theta)
    return (a > 0) & (a * diff_step[:, None] < _FD_DEAD)


def _request(theta: np.ndarray, k: np.ndarray, g: np.ndarray, tau: np.ndarray) -> Request:
    """パラメータ ``theta``（3・4 番目が初速 vs, vz）の軌道の積分の要求。積分しない速さ（``MAX_SIM_SPEED_MPS`` より上・
    NaN）は、結果を使わないので、決まった速度に置き換える。"""
    v0 = theta[:, 2:4]
    with np.errstate(invalid="ignore"):
        ok = np.hypot(v0[:, 0], v0[:, 1]) <= MAX_SIM_SPEED_MPS
    return np.where(ok[:, None], v0, 1.0), k, g, tau


class _CameraArrays:
    """カメラの投影と、その位置についての傾き。"""

    def __init__(self, camera: Camera):
        self.rot, self.tvec = camera.R, camera.tvec
        self.f, self.c = float(camera.f), np.array([camera.cx, camera.cy], float)
        self.k1, self.k2 = float(camera.k1), float(camera.k2)

    def project(self, xw: np.ndarray, dxw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """位置 ``xw`` (B, M, 3) [m] と、その傾き ``dxw`` (B, M, n, 3) の投影。

        Returns:
            (画像の位置 (B, M, 2) [px]。カメラの後ろは NaN、その傾き (B, M, 2, n)（カメラの後ろは 0）)
        """
        xc = xw @ self.rot.T + self.tvec
        dxc = dxw @ self.rot.T
        z = xc[..., 2]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            xn = xc[..., :2] / z[..., None]
            r2 = np.sum(xn * xn, axis=-1)
            dist = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
            uv = xn * (dist * self.f)[..., None] + self.c
            # d(xn·dist) = dist·dxn + xn·(2k1 + 4k2 r2)(xn·dxn)、dxn = (dxc_xy − xn·dxc_z) / z
            dxn = (dxc[..., :2] - xn[..., None, :] * dxc[..., 2:]) / z[..., None, None]
            radial = (2 * self.k1 + 4 * self.k2 * r2)[..., None] * np.einsum("bmj,bmnj->bmn", xn, dxn)
            duv = self.f * (dist[..., None, None] * dxn + radial[..., None] * xn[..., None, :])
        behind = ~(z > 0)
        uv[behind] = np.nan
        duv[behind] = 0.0
        return uv, np.swapaxes(duv, 2, 3)


class _Batch:
    """同じカメラ・同じパラメータの数の当てはめ B 個の残差とヤコビアン。

    パラメータは (s0, z0, vs, vz)（``n_plane`` = 2 なら、後ろに平面の向き dψ・位置 dn）。残差の並びは
    ``physics._residuals`` と同じ：[観測 2 × M 個（足りない分は 0）, 平面の目安 2, 打点の目安 3, 初速の上限 1]。
    """

    def __init__(self, cam: _CameraArrays, n_plane: int, tau: np.ndarray, uv: np.ndarray, origin: np.ndarray,
                 direction: np.ndarray, k: np.ndarray, g: np.ndarray, plane_sigma: np.ndarray,
                 p0_prior: np.ndarray, p0_sigma: np.ndarray, prior_px: np.ndarray, max_speed: np.ndarray,
                 speed_weight: np.ndarray, huber_px: np.ndarray, diff_step: np.ndarray):
        self.cam, self.n_plane, self.n = cam, n_plane, 4 + n_plane
        self.diff_step = diff_step                            # (B,)
        self.tau, self.uv = tau, uv                           # (B, M)・(B, M, 2)。使わない所は NaN
        self.valid = np.isfinite(tau)
        self.origin, self.direction = origin, direction       # (B, 3)
        self.normal = np.stack([-direction[:, 1], direction[:, 0], np.zeros(len(tau))], axis=1)
        self.k, self.g = k, g
        self.plane_sigma = plane_sigma                        # (B, 2)
        self.p0_prior, self.p0_sigma, self.prior_px = p0_prior, p0_sigma, prior_px   # (B, 3)（なければ NaN）・(B,)
        self.max_speed, self.speed_weight, self.huber_px = max_speed, speed_weight, huber_px   # (B,)（なければ NaN）
        self.n_rows = 2 * tau.shape[1] + 6
        self.m = (2 * self.valid.sum(axis=1) + n_plane + 3 * np.isfinite(p0_prior[:, 0])
                  + np.isfinite(max_speed)).astype(int)

    def frame(self, i: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """パラメータの平面の (向き, 向きの dψ についての傾き, 原点)。"""
        d = self.direction[i]
        if not self.n_plane:
            return d, np.zeros_like(d), self.origin[i]
        c, s = np.cos(theta[:, 4]), np.sin(theta[:, 4])
        rot = np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1], np.zeros(len(i))], axis=1)
        drot = np.stack([-rot[:, 1], rot[:, 0], np.zeros(len(i))], axis=1)
        return rot, drot, self.origin[i] + theta[:, 5:6] * self.normal[i]

    def start(self, i: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """打点 p0 (b, 3) [m] と初速 v0 (b, 3) [m/s]（``_Model.unpack``）。"""
        rot, _, org = self.frame(i, theta)
        p0 = org + theta[:, 0:1] * rot
        p0[:, 2] += theta[:, 1]
        v0 = theta[:, 2:3] * rot
        v0[:, 2] += theta[:, 3]
        return p0, v0

    def request(self, i: np.ndarray, theta: np.ndarray) -> Request:
        """問題 ``i`` をパラメータ ``theta`` で評価するのに要る、軌道の積分の要求。"""
        return _request(theta, self.k[i], self.g[i], self.tau[i])

    def residuals(self, i: np.ndarray, theta: np.ndarray, tr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """残差 (b, 行) とヤコビアン (b, 行, n)。``tr`` は ``request`` の積分の結果。"""
        b, m_obs, n = len(i), self.tau.shape[1], self.n
        f = np.zeros((b, self.n_rows))
        jac = np.zeros((b, self.n_rows, n))
        rot, drot, org = self.frame(i, theta)
        speed = np.hypot(theta[:, 2], theta[:, 3])
        ok = speed <= MAX_SIM_SPEED_MPS
        valid = self.valid[i]
        # --- 画像の上のずれ
        s_rel, z_rel = tr[..., 0, 0], tr[..., 0, 1]
        xw = (org + theta[:, 0:1] * rot)[:, None, :] + s_rel[..., None] * rot[:, None, :]
        xw[..., 2] += theta[:, 1:2] + z_rel
        dxw = np.zeros((b, m_obs, n, 3))
        dxw[:, :, 0, :] = rot[:, None, :]
        dxw[:, :, 1, 2] = 1.0
        dxw[:, :, 2, :] = tr[..., 1, 0, None] * rot[:, None, :]
        dxw[:, :, 2, 2] += tr[..., 1, 1]
        dxw[:, :, 3, :] = tr[..., 2, 0, None] * rot[:, None, :]
        dxw[:, :, 3, 2] += tr[..., 2, 1]
        if self.n_plane:
            dxw[:, :, 4, :] = (theta[:, 0:1] + s_rel)[..., None] * drot[:, None, :]
            dxw[:, :, 5, :] = self.normal[i][:, None, :]
        uv, duv = self.cam.project(xw, dxw)
        r = uv - self.uv[i]
        good = np.isfinite(r) & ok[:, None, None]
        r = np.where(good, r, _BAD)
        duv = np.where(good[..., None], duv, 0.0)
        f[:, :2 * m_obs] = np.where(valid[..., None], r, 0.0).reshape(b, 2 * m_obs)
        jac[:, :2 * m_obs] = np.where(valid[..., None, None], duv, 0.0).reshape(b, 2 * m_obs, n)
        row = 2 * m_obs
        # --- 平面の補正の目安
        if self.n_plane:
            sig = self.plane_sigma[i]
            f[:, row:row + 2] = theta[:, 4:6] / sig
            jac[:, row, 4], jac[:, row + 1, 5] = 1.0 / sig[:, 0], 1.0 / sig[:, 1]
        row += 2
        # --- 打点の位置の目安
        prior = self.p0_prior[i]
        has = np.isfinite(prior[:, 0])
        if has.any():
            w = (self.prior_px[i][:, None] / self.p0_sigma[i])
            p0 = org + theta[:, 0:1] * rot
            p0[:, 2] += theta[:, 1]
            e, de = _quadratic(np.where(has[:, None], w * (p0 - prior), 0.0), self.huber_px[i][:, None])
            f[:, row:row + 3] = e
            w = np.where(has[:, None], w * de, 0.0)
            jac[:, row:row + 3, 0] = w * rot
            jac[:, row + 2, 1] = w[:, 2]
            if self.n_plane:
                jac[:, row:row + 3, 4] = w * theta[:, 0:1] * drot
                jac[:, row:row + 3, 5] = w * self.normal[i]
        row += 3
        # --- 初速の上限
        vmax = self.max_speed[i]
        with np.errstate(invalid="ignore"):
            over = speed - vmax
            e = np.where(over > 0, self.speed_weight[i] * over / SPEED_PENALTY_SIGMA_MPS, 0.0)
            e = np.where(np.isfinite(vmax) & ~np.isfinite(speed), np.nan, e)   # 初期値が NaN なら当てはめられない
        e, de = _quadratic(e, self.huber_px[i])
        f[:, row] = e
        with np.errstate(invalid="ignore", divide="ignore"):
            w = np.where(over > 0, de * self.speed_weight[i] / SPEED_PENALTY_SIGMA_MPS / speed, 0.0)
        jac[:, row, 2], jac[:, row, 3] = w * theta[:, 2], w * theta[:, 3]
        if EMULATE_FD:
            dead = _fd_dead(theta, self.diff_step[i])
            if dead.any():
                keep = jac[:, 2 * m_obs:2 * m_obs + 2, :].copy()     # 平面の目安の行は、値そのものなので傾きが出る
                jac[np.broadcast_to(dead[:, None, :], jac.shape)] = 0.0
                jac[:, 2 * m_obs:2 * m_obs + 2, :] = keep
        return f, jac


class _Stage1:
    """初期値の計算の第 1 段階：視線と平面の交点（3 次元点）に運動方程式を当てはめる（``physics._residuals_3d``）。

    パラメータは (s0, z0, vs, vz)。残差 [m] は [点ごとに x・y・z（足りない分は 0）, 初速の上限 1]。
    """

    def __init__(self, tau: np.ndarray, sz: np.ndarray, direction: np.ndarray, k: np.ndarray, g: np.ndarray,
                 max_speed: np.ndarray, speed_weight: np.ndarray, diff_step: np.ndarray):
        self.diff_step = diff_step
        self.tau, self.sz, self.direction = tau, sz, direction
        self.valid = np.isfinite(tau)
        self.k, self.g, self.max_speed, self.speed_weight = k, g, max_speed, speed_weight
        self.m = (3 * self.valid.sum(axis=1) + np.isfinite(max_speed)).astype(int)

    def request(self, i: np.ndarray, theta: np.ndarray) -> Request:
        return _request(theta, self.k[i], self.g[i], self.tau[i])

    def residuals(self, i: np.ndarray, theta: np.ndarray, tr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        b, m_obs = len(i), self.tau.shape[1]
        f = np.zeros((b, 3 * m_obs + 1))
        jac = np.zeros((b, 3 * m_obs + 1, 4))
        speed = np.hypot(theta[:, 2], theta[:, 3])
        ok = (speed <= MAX_SIM_SPEED_MPS)[:, None]
        valid = self.valid[i]
        d = self.direction[i][:, None, :2]
        ds = theta[:, 0:1] + tr[..., 0, 0] - self.sz[i][..., 0]
        dz = theta[:, 1:2] + tr[..., 0, 1] - self.sz[i][..., 1]
        r = np.concatenate([ds[..., None] * d, dz[..., None]], axis=2)             # (b, M, 3)
        dr = np.zeros((b, m_obs, 3, 4))
        dr[:, :, :2, 0] = d
        dr[:, :, 2, 1] = 1.0
        dr[:, :, :2, 2] = tr[..., 1, 0, None] * d
        dr[:, :, 2, 2] = tr[..., 1, 1]
        dr[:, :, :2, 3] = tr[..., 2, 0, None] * d
        dr[:, :, 2, 3] = tr[..., 2, 1]
        good = np.isfinite(r) & ok[:, :, None]
        r = np.where(good, r, _BAD)
        dr = np.where(good[..., None], dr, 0.0)
        f[:, :3 * m_obs] = np.where(valid[..., None], r, 0.0).reshape(b, 3 * m_obs)
        jac[:, :3 * m_obs] = np.where(valid[..., None, None], dr, 0.0).reshape(b, 3 * m_obs, 4)
        vmax = self.max_speed[i]
        with np.errstate(invalid="ignore", divide="ignore"):
            over = speed - vmax
            f[:, -1] = np.where(over > 0, self.speed_weight[i] * over / SPEED_PENALTY_SIGMA_MPS, 0.0)
            w = np.where(over > 0, self.speed_weight[i] / SPEED_PENALTY_SIGMA_MPS / speed, 0.0)
        jac[:, -1, 2], jac[:, -1, 3] = w * theta[:, 2], w * theta[:, 3]
        if EMULATE_FD:
            dead = _fd_dead(theta, self.diff_step[i])
            if dead.any():
                jac[np.broadcast_to(dead[:, None, :], jac.shape)] = 0.0
        return f, jac


# =============================================================================
# 当てはめ（``fit_trajectory`` と同じ手順を、まとめて）
# =============================================================================
def supported(job: FitJob) -> bool:
    """この当てはめを、まとめて解けるか（解けなければ ``fit_trajectory`` で 1 本ずつ解く）。"""
    o = job.opts
    return (job.plane is not None and not o.estimate_terminal_velocity and o.ci_method in ("jacobian", "none")
            and o.loss in ("linear", "huber", "soft_l1") and len(job.obs.t) > 0
            and bool((job.obs.t >= job.t_ref).all()))


def _one(job: FitJob) -> FitResult | None:
    try:
        return fit_trajectory(job.obs, job.t_ref, "plane", job.plane, job.phys, job.opts, theta0=job.theta0)
    except (ValueError, RuntimeError):
        return None


def _pad(rows: list[np.ndarray], width: int, tail: tuple[int, ...] = ()) -> np.ndarray:
    out = np.full((len(rows), width) + tail, np.nan)
    for j, r in enumerate(rows):
        out[j, :len(r)] = r
    return out


def _clip_speed(theta: np.ndarray, max_speed: np.ndarray) -> np.ndarray:
    """``_Model.clip_speed``：上限より速い初期値を、向きはそのままに上限まで遅くする。"""
    theta = np.array(theta, float)
    v = np.hypot(theta[:, 2], theta[:, 3])
    with np.errstate(invalid="ignore", divide="ignore"):
        fac = np.where(v > max_speed, max_speed / v, 1.0)
    theta[:, 2:4] *= fac[:, None]
    return theta


def fit_steps(jobs: list[FitJob]) -> Steps:
    """``fit_trajectory``（plane モード）を ``jobs`` の数だけまとめて解く計算（``run``・``together`` で動かす）。

    戻り値は ``jobs`` と同じ順の結果。``fit_trajectory`` が ``ValueError`` を出す当てはめ（コマが足りない、など）は None。
    """
    out: list[FitResult | None] = [None] * len(jobs)
    groups: dict[tuple[Any, ...], list[int]] = {}
    for j, job in enumerate(jobs):
        if supported(job):
            n_plane = 2 if job.opts.refine_plane else 0
            groups.setdefault((id(job.obs.camera), n_plane, job.opts.loss), []).append(j)
        else:
            out[j] = _one(job)
    keys = list(groups)
    res = yield from together([_fit_group([jobs[j] for j in groups[key]], key[1], key[2]) for key in keys])
    for key, part in zip(keys, res):
        for j, f in zip(groups[key], part):
            out[j] = f
    return out


def fit_trajectories(jobs: list[FitJob], backend: str | None = None) -> list[FitResult | None]:
    """``fit_trajectory``（plane モード）を ``jobs`` の数だけ、まとめて解く（``backend`` は ``get_integrator``）。

    Returns:
        ``jobs`` と同じ順の結果。``fit_trajectory`` が ``ValueError`` を出す当てはめ（コマが足りない、など）は None
    """
    return run(fit_steps(jobs), backend)


def solve_together(tasks: list[Generator[list[FitJob], list[FitResult | None], Any]], backend: str | None = None
                   ) -> list[Any]:
    """「当てはめのリストを出して、その結果を受け取りながら進む計算」をいくつか、同時に進める。

    ラリーごとの当てはめ（``shot_fit.fit_shots`` の中身）は、段階ごとに何百もの当てはめを出す。ラリーどうしは
    関係しないので、すべてのラリーの当てはめの積分を 1 回にまとめる（当てはめの時間は「残った数本の繰り返し」が
    ほとんどで、1 回の積分の手間は本数にあまりよらないため）。戻り値は、それぞれの計算の戻り値のリスト。
    """
    def steps(task: Generator[list[FitJob], list[FitResult | None], Any]) -> Steps:
        res: list[FitResult | None] | None = None
        while True:
            try:
                jobs = task.send(res) if res is not None else next(task)
            except StopIteration as e:
                return e.value
            res = yield from fit_steps(jobs)

    return run(together([steps(t) for t in tasks]), backend)


def _fit_group(jobs: list[FitJob], n_plane: int, loss: str) -> Steps:
    from scipy.stats import norm

    camera = jobs[0].obs.camera
    cam = _CameraArrays(camera)
    n_par = 4 + n_plane
    out: list[FitResult | None] = [None] * len(jobs)
    # --- 当てはめられる問題と、その初期値
    rays: dict[int, np.ndarray] = {}      # 観測ごとの視線の向き（同じ観測を何度も使うので、1 回だけ計算する）
    center = camera.center
    keep: list[int] = []                  # 当てはめる問題（jobs の番号）
    theta_given: dict[int, np.ndarray] = {}
    stage1: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []   # (jobs の番号, τ, 平面の中の位置, 初期値)
    for j, job in enumerate(jobs):
        o, n_obs = job.opts, len(job.obs.t)
        n_prior = 3 if (o.prior_counts and o.p0_prior_m is not None) else 0
        if 2 * n_obs + n_prior <= n_par:
            continue
        pl = job.plane
        if job.theta0 is not None:
            theta_given[j] = np.asarray(job.theta0, float)
            keep.append(j)
            continue
        if o.init_state is not None:
            p_init, v_init = (np.asarray(x, float) for x in o.init_state)
            theta_given[j] = np.r_[(p_init - pl.origin) @ pl.direction, p_init[2], v_init @ pl.direction, v_init[2],
                                   np.zeros(n_plane)]
        # 視線と平面の交点（``geometry.intersect_plane``）を、平面の中の位置 (s, z) で
        if id(job.obs) not in rays:
            rays[id(job.obs)] = camera.rays(job.obs.uv)[1]
        d = rays[id(job.obs)]
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = d @ pl.normal
            lam = ((pl.origin - center) @ pl.normal) / denom
            pts = center + lam[:, None] * d
        good = ~((np.abs(denom) < 1e-9) | (lam <= 0)) & np.isfinite(pts).all(axis=1)
        order = np.argsort(job.obs.t[good])
        t3 = job.obs.t[good][order]
        sz = np.column_stack([(pts[good][order] - pl.origin) @ pl.direction, pts[good][order][:, 2]])
        if len(t3) >= 3:
            # ``physics.initial_guess``：序盤の点に 2 次式を当てはめて、打点の時刻へ外挿する
            n = min(len(t3), 6)
            coef = np.polyfit(t3[:n] - job.t_ref, sz[:n], 2 if n >= 4 else 1)
            stage1.append((j, t3 - job.t_ref, sz, np.r_[coef[-1], coef[-2]]))
            keep.append(j)
        elif j in theta_given:
            keep.append(j)
    if not keep:
        return out

    def arr(fn: Callable[[FitJob], Any], which: list[int]) -> np.ndarray:
        return np.array([fn(jobs[j]) for j in which], float)

    def max_speed_of(job: FitJob) -> float:
        return np.nan if job.opts.max_speed_mps is None else job.opts.max_speed_mps

    # --- 第 1 段階（3 次元点への当てはめ）
    start1: dict[int, np.ndarray] = {}
    failed: set[int] = set()
    if stage1:
        which = [s[0] for s in stage1]
        width = max(len(s[1]) for s in stage1)
        n3 = np.array([len(s[1]) for s in stage1])
        vmax = arr(max_speed_of, which)
        prob = _Stage1(_pad([s[1] for s in stage1], width), _pad([s[2] for s in stage1], width, (2,)),
                       arr(lambda q: q.plane.direction, which), arr(lambda q: q.phys.k, which),
                       arr(lambda q: q.phys.g, which), vmax, 0.1 * np.sqrt(3.0 * n3),
                       arr(lambda q: q.opts.diff_step, which))
        ones = np.ones(len(which))
        sol = yield from _trf(prob, _clip_speed(np.array([s[3] for s in stage1]), vmax), prob.m, "soft_l1", 0.1 * ones,
                   1e-8 * ones, 1e-8 * ones, 1e-8, arr(lambda q: q.opts.max_nfev_per_param * n_par, which).astype(int))
        for j, x, st in zip(which, sol["x"], sol["status"]):
            if st == -1:
                failed.add(j)
            start1[j] = np.r_[x, np.zeros(n_plane)]
    keep = [j for j in keep if j not in failed]
    if not keep:
        return out

    # --- 初期値ごとに解く
    pos = {j: a for a, j in enumerate(keep)}
    width = max(len(jobs[j].obs.t) for j in keep)
    n_res = np.array([2 * len(jobs[j].obs.t) for j in keep])
    vmax = arr(max_speed_of, keep)
    quad = arr(lambda q: q.opts.loss_scale_px if (q.opts.quadratic_priors and q.opts.loss == "huber") else np.nan, keep)
    prior = np.array([np.full(3, np.nan) if jobs[j].opts.p0_prior_m is None
                      else np.asarray(jobs[j].opts.p0_prior_m, float) for j in keep])
    sig = arr(lambda q: (q.opts.p0_prior_sigma_m[0], q.opts.p0_prior_sigma_m[0], q.opts.p0_prior_sigma_m[1]), keep)
    batch = _Batch(cam, n_plane, _pad([jobs[j].obs.t - jobs[j].t_ref for j in keep], width),
                   _pad([jobs[j].obs.uv for j in keep], width, (2,)), arr(lambda q: q.plane.origin, keep),
                   arr(lambda q: q.plane.direction, keep), arr(lambda q: q.phys.k, keep),
                   arr(lambda q: q.phys.g, keep),
                   arr(lambda q: (np.deg2rad(q.opts.plane_sigma_deg), q.opts.plane_sigma_m), keep), prior, sig,
                   np.ones(len(keep)), vmax, np.sqrt(n_res.astype(float)), quad, arr(lambda q: q.opts.diff_step, keep))
    f_scale = arr(lambda q: q.opts.loss_scale_px, keep)
    ftol, xtol = arr(lambda q: q.opts.ftol, keep), arr(lambda q: q.opts.xtol, keep)
    max_nfev = arr(lambda q: q.opts.max_nfev_per_param * n_par, keep).astype(int)

    def solve(rows: np.ndarray, x0: np.ndarray) -> Steps:
        """``keep`` の中の番号 ``rows`` の問題を、初期値 ``x0`` から解く。"""
        return _trf(batch, _clip_speed(x0, vmax[rows]), batch.m[rows], loss, f_scale[rows], ftol[rows], xtol[rows],
                    1e-8, max_nfev[rows], rows)

    rows, x0 = [], []
    for j in keep:                       # ``fit_trajectory`` と同じ順：与えた初期値（探索の結果）→ 第 1 段階
        for th in ([theta_given[j]] if j in theta_given else []) + ([start1[j]] if j in start1 else []):
            rows.append(pos[j])
            x0.append(th)
    rows = np.array(rows)
    sol = yield from solve(rows, np.array(x0))
    best: dict[int, int] = {}            # keep の中の番号 → sol の中の番号（ずれの小さい方。同じなら先のもの）
    bad = set(rows[sol["status"] == -1].tolist())
    for a, r in enumerate(rows.tolist()):
        if r not in bad and (r not in best or sol["cost"][a] < sol["cost"][best[r]]):
            best[r] = a
    if not best:
        return out
    rows = np.array(sorted(best))
    pick = np.array([best[r] for r in rows])
    sol = {k: v[pick] for k, v in sol.items()}
    # --- 打点の目安の 1σ を、画像の上のずれの大きさと同じ重みにして解き直す
    again = np.isfinite(prior[rows, 0])
    if again.any():
        m_obs = 2 * width
        rms0 = np.sqrt(np.sum(sol["fun"][again, :m_obs] ** 2, axis=1) / n_res[rows[again]])
        batch.prior_px[rows[again]] = np.maximum(f_scale[rows[again]], rms0)
        new = yield from solve(rows[again], sol["x"][again])
        for key, v in new.items():
            sol[key][again] = v
    # --- 結果
    p0, v0 = batch.start(rows, sol["x"])
    rot, _, org = batch.frame(rows, sol["x"])
    for a, r in enumerate(rows.tolist()):
        j = keep[r]
        if sol["status"][a] == -1:
            continue
        job, o, x = jobs[j], jobs[j].opts, sol["x"][a]
        n = int(n_res[r])
        rms = float(np.sqrt(np.sum(sol["fun"][a, :2 * width] ** 2) / n))
        cov = None
        dof = max(n - n_par, 1)
        jac = sol["jac"][a]
        try:
            cov = np.linalg.inv(jac.T @ jac) * (2.0 * sol["cost"][a] / dof)
            if not np.isfinite(cov).all():
                cov = None
        except np.linalg.LinAlgError:
            pass
        status = int(sol["status"][a])
        vt = job.phys.terminal_velocity
        res = FitResult(mode="plane", t_ref=job.t_ref, p0=p0[a].copy(), v0=v0[a].copy(), terminal_velocity=vt,
                        theta=x.copy(), cov_theta=cov, rms_px=rms, n_obs=n // 2, success=status > 0,
                        message=_MESSAGES[status],
                        plane=FlightPlane(org[a], rot[a]) if n_plane else job.plane)
        speed = res.speed0
        # 打点時刻の不確かさによる初速のばらつき（d|v|/dt × σ_t）
        rate = -job.phys.k * speed * speed - job.phys.g * v0[a, 2] / speed if speed > 0 else np.nan
        res.speed0_sd_timing = abs(rate) * o.t_ref_sigma
        if o.ci_method == "jacobian" and cov is not None:
            grad = np.zeros(n_par)
            grad[2:4] = x[2:4] / speed
            sd_fit = float(np.sqrt(max(grad @ cov @ grad, 0.0)))
            sd = np.hypot(sd_fit, res.speed0_sd_timing)
            res.speed0_sd_fit = sd_fit
            if dof >= 3 and sd < speed:
                zq = norm.ppf(0.5 + o.ci_level / 2)
                res.speed0_ci = (speed - zq * sd, speed + zq * sd)
                res.ci_method = "jacobian"
            else:
                res.ci_method = "jacobian(unreliable)"
        out[j] = res
    return out
