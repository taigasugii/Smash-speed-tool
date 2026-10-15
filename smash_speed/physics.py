"""物理モデル：空気抵抗つき運動方程式、再投影誤差による軌道フィッティング、初速の信頼区間。

運動方程式（速度の2乗に比例する空気抵抗）::

    dv/dt = -g ez - (g / v_t^2) |v| v

g: 重力加速度 [m/s^2]、v_t: シャトルの終端速度 [m/s]。

フィッティングは **画像上の再投影誤差 [px]** を Huber 損失で最小化する（観測誤差は画像上で生じるため）。
推定するパラメータは打点時刻 t_ref [s] における位置と速度で、

- ``plane`` モード: 飛行平面内の (s0, z0, vs0, vz0) の4つ（カメラ1台 + 平面拘束）
- ``free`` モード: (x0, y0, z0, vx0, vy0, vz0) の6つ（カメラ2台以上）

必要に応じて log(v_t) を加える。抵抗は等方的なので、平面内に初速を持つ軌道は平面から出ない。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from smash_speed.camera import Camera
from smash_speed.geometry import FlightPlane, intersect_plane, triangulate

MPS_TO_KMH = 3.6


# =============================================================================
# 運動方程式
# =============================================================================
def _rhs(t: float, state: np.ndarray, g: float, k: float) -> np.ndarray:
    """状態 (x, y, z, vx, vy, vz) の時間微分。k = g / v_t^2 [1/m]。"""
    v = state[3:]
    speed = np.sqrt(v @ v)
    a = -k * speed * v
    a[2] -= g
    return np.concatenate([v, a])


def _ground_event(t: float, state: np.ndarray, g: float, k: float) -> float:
    return state[2]


_ground_event.terminal = True
_ground_event.direction = -1


@dataclass
class PhysicsParams:
    """運動方程式と数値積分の設定。"""

    g: float = 9.81                 # [m/s^2]
    terminal_velocity: float = 6.7  # [m/s]
    method: str = "DOP853"
    rtol: float = 1e-10
    atol: float = 1e-12

    @property
    def k(self) -> float:
        """抵抗係数 g / v_t^2 [1/m]。"""
        return self.g / self.terminal_velocity**2

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "PhysicsParams":
        p = cfg["physics"]
        return cls(g=p["g_mps2"], terminal_velocity=p["terminal_velocity_mps"],
                   method=p.get("integrator", "DOP853"), rtol=p["rtol"], atol=p["atol"])


def simulate(p0: np.ndarray, v0: np.ndarray, t_ref: float, times: np.ndarray,
             phys: PhysicsParams) -> tuple[np.ndarray, np.ndarray]:
    """時刻 ``t_ref`` [s] で位置 p0 [m]・速度 v0 [m/s] の軌道を ``times`` [s] で評価する。

    ``t_ref`` より前の時刻は逆向きに積分する。

    Returns:
        (positions (N, 3) [m], velocities (N, 3) [m/s])
    """
    times = np.asarray(times, float)
    out = np.empty((len(times), 6))
    y0 = np.concatenate([p0, v0])
    args = (phys.g, phys.k)
    for mask, sign in ((times >= t_ref, 1.0), (times < t_ref, -1.0)):
        if not mask.any():
            continue
        # solve_ivp の t_eval は積分の向きに狭義単調でなければならないので、重複を除いて並べる
        uniq, inverse = np.unique(sign * times[mask], return_inverse=True)
        tt = sign * uniq
        if tt[-1] == t_ref:
            out[mask] = y0
            continue
        sol = solve_ivp(_rhs, (t_ref, tt[-1]), y0, method=phys.method, t_eval=tt,
                        rtol=phys.rtol, atol=phys.atol, args=args)
        y = np.asarray(sol.y)
        if not sol.success or y.ndim != 2 or y.shape[1] != len(tt):
            # 最適化の途中で極端なパラメータを試したときなど。NaN は残差で罰則として扱われる
            out[mask] = np.nan
            continue
        out[mask] = y.T[inverse]
    return out[:, :3], out[:, 3:]


def simulate_until_ground(p0: np.ndarray, v0: np.ndarray, t_ref: float, phys: PhysicsParams,
                          t_max: float = 5.0, n_dense: int = 2000) -> dict[str, np.ndarray]:
    """床面 (z = 0) に着くまで積分し、細かい時刻刻みの軌道を返す。

    Returns:
        ``t`` [s], ``pos`` (N, 3) [m], ``vel`` (N, 3) [m/s], ``t_land`` [s]（着地しなければ NaN）
    """
    y0 = np.concatenate([p0, v0])
    sol = solve_ivp(_rhs, (t_ref, t_ref + t_max), y0, method=phys.method, dense_output=True,
                    events=_ground_event, rtol=phys.rtol, atol=phys.atol, args=(phys.g, phys.k))
    t_land = sol.t_events[0][0] if len(sol.t_events[0]) else np.nan
    t_end = t_land if np.isfinite(t_land) else sol.t[-1]
    t = np.linspace(t_ref, t_end, n_dense)
    y = sol.sol(t).T
    return {"t": t, "pos": y[:, :3], "vel": y[:, 3:], "t_land": np.asarray(t_land)}


# =============================================================================
# 観測とフィッティング
# =============================================================================
@dataclass
class Observations:
    """1台のカメラによるシャトルの観測。

    Attributes:
        camera: 観測したカメラ
        t: 観測時刻 (N,) [s]（露光の中央時刻）
        uv: 画像座標 (N, 2) [px]
    """

    camera: Camera
    t: np.ndarray
    uv: np.ndarray

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, float)
        self.uv = np.asarray(self.uv, float).reshape(-1, 2)
        ok = np.isfinite(self.t) & np.isfinite(self.uv).all(axis=1)
        self.t, self.uv = self.t[ok], self.uv[ok]

    def subset(self, idx: np.ndarray) -> "Observations":
        return Observations(self.camera, self.t[idx], self.uv[idx])


@dataclass
class FitResult:
    """軌道フィッティングの結果。速度 [m/s]、位置 [m]、時刻 [s]。"""

    mode: str
    t_ref: float
    p0: np.ndarray
    v0: np.ndarray
    terminal_velocity: float
    theta: np.ndarray
    cov_theta: np.ndarray | None
    rms_px: float
    n_obs: int
    success: bool
    message: str
    plane: FlightPlane | None = None
    speed0_ci: tuple[float, float] | None = None
    ci_method: str | None = None
    speed0_sd_fit: float | None = None
    speed0_sd_timing: float | None = None
    bootstrap_speeds: np.ndarray | None = field(default=None, repr=False)

    @property
    def speed0(self) -> float:
        """打点時刻の速さ（初速）[m/s]。"""
        return float(np.linalg.norm(self.v0))

    def phys(self, base: PhysicsParams) -> PhysicsParams:
        """推定した v_t を反映した PhysicsParams を返す。"""
        return PhysicsParams(base.g, self.terminal_velocity, base.method, base.rtol, base.atol)

    def evaluate(self, times: np.ndarray, base: PhysicsParams) -> tuple[np.ndarray, np.ndarray]:
        """当てはめた軌道の位置 [m] と速度 [m/s] を ``times`` で評価する。"""
        return simulate(self.p0, self.v0, self.t_ref, times, self.phys(base))

    def to_dict(self) -> dict[str, Any]:
        d = {
            "mode": self.mode, "t_ref_s": self.t_ref, "p0_m": self.p0.tolist(),
            "v0_mps": self.v0.tolist(), "speed0_mps": self.speed0,
            "speed0_kmh": self.speed0 * MPS_TO_KMH,
            "terminal_velocity_mps": self.terminal_velocity,
            "rms_px": self.rms_px, "n_obs": self.n_obs, "success": self.success,
            "message": self.message, "ci_method": self.ci_method,
            "speed0_ci_mps": list(self.speed0_ci) if self.speed0_ci else None,
            "speed0_sd_fit_mps": self.speed0_sd_fit, "speed0_sd_timing_mps": self.speed0_sd_timing,
            "theta": self.theta.tolist(),
            "cov_theta": self.cov_theta.tolist() if self.cov_theta is not None else None,
        }
        if self.plane is not None:
            d["plane"] = self.plane.to_dict()
        return d


@dataclass
class FitOptions:
    """フィッティングの設定。"""

    loss: str = "huber"
    loss_scale_px: float = 2.0       # Huber 損失の切り替え点 [px]
    estimate_terminal_velocity: bool = False
    t_ref_sigma: float = 0.0         # 打点時刻の不確かさ（標準偏差）[s]。信頼区間に加算する
    ci_level: float = 0.95
    ci_method: str = "jacobian"      # "jacobian" | "bootstrap" | "none"
    bootstrap_n: int = 200
    diff_step: float = 1e-6
    seed: int = 0
    refine_plane: bool = False       # 飛行平面の向き・位置もデータから補正する（plane モード）
    plane_sigma_deg: float = 5.0     # 平面の向きの事前の不確かさ [deg]
    plane_sigma_m: float = 0.5       # 平面の位置（法線方向）の事前の不確かさ [m]

    @classmethod
    def from_config(cls, cfg: dict[str, Any], **overrides: Any) -> "FitOptions":
        p = cfg["physics"]
        opts = cls(loss=p["loss"], loss_scale_px=p["loss_scale_px"],
                   estimate_terminal_velocity=p["estimate_terminal_velocity"],
                   ci_level=p["ci_level"], ci_method=p.get("ci_method", "jacobian"),
                   bootstrap_n=p["bootstrap_n"], seed=cfg.get("seed", 0),
                   refine_plane=p.get("refine_plane", False),
                   plane_sigma_deg=p.get("plane_sigma_deg", 5.0),
                   plane_sigma_m=p.get("plane_sigma_m", 0.5))
        for k, v in overrides.items():
            setattr(opts, k, v)
        return opts


class _Model:
    """パラメータベクトル θ ⇔ (p0, v0, v_t) の変換。"""

    def __init__(self, mode: str, plane: FlightPlane | None, estimate_vt: bool, vt: float,
                 refine_plane: bool = False, plane_sigma_rad: float = 0.1, plane_sigma_m: float = 0.5):
        if mode == "plane" and plane is None:
            raise ValueError("plane モードには飛行平面が必要です")
        self.mode, self.plane, self.estimate_vt, self.vt = mode, plane, estimate_vt, vt
        self.n_kin = 4 if mode == "plane" else 6
        self.n_plane = 2 if (mode == "plane" and refine_plane) else 0
        self.plane_sigma = np.array([plane_sigma_rad, plane_sigma_m])

    def plane_at(self, theta: np.ndarray) -> FlightPlane | None:
        """θ に含まれる平面の補正（向き dψ [rad]・法線方向の平行移動 dn [m]）を反映した平面。"""
        if self.plane is None or self.n_plane == 0:
            return self.plane
        dpsi, dn = theta[self.n_kin:self.n_kin + 2]
        d = self.plane.direction
        c, s = np.cos(dpsi), np.sin(dpsi)
        rot = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], 0.0])
        return FlightPlane(self.plane.origin + dn * self.plane.normal, rot)

    def unpack(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        if self.mode == "plane":
            plane = self.plane_at(theta)
            s0, z0, vs, vz = theta[:4]
            p0 = plane.to_world(np.array([s0]), np.array([z0]))[0]
            v0 = vs * plane.direction + vz * np.array([0.0, 0.0, 1.0])
        else:
            p0, v0 = theta[:3].copy(), theta[3:6].copy()
        vt = float(np.exp(theta[self.n_kin + self.n_plane])) if self.estimate_vt else self.vt
        return p0, v0, vt

    def prior_residuals(self, theta: np.ndarray) -> np.ndarray:
        """平面の補正に対する事前分布の残差（標準偏差で割った値。1σ ≈ 1 px の残差と同じ重み）。"""
        if self.n_plane == 0:
            return np.empty(0)
        return theta[self.n_kin:self.n_kin + 2] / self.plane_sigma

    def pack(self, p0: np.ndarray, v0: np.ndarray, vt: float) -> np.ndarray:
        if self.mode == "plane":
            s0, z0 = self.plane.to_plane(p0[None])[0]
            kin = [s0, z0, v0 @ self.plane.direction, v0[2]]
        else:
            kin = list(p0) + list(v0)
        kin += [0.0] * self.n_plane
        if self.estimate_vt:
            kin.append(np.log(vt))
        return np.asarray(kin, float)


def _points_3d(obs: list[Observations], mode: str, plane: FlightPlane | None
               ) -> tuple[np.ndarray, np.ndarray]:
    """初期値計算用に観測から3次元点を作る（平面交点 or 同時刻の三角測量）。"""
    if mode == "plane":
        X = np.vstack([intersect_plane(o.camera, o.uv, plane) for o in obs])
        t = np.concatenate([o.t for o in obs])
    else:
        # 2台の観測を時刻で合わせる（最も近い時刻、半フレーム以内）
        a, b = obs[0], obs[1]
        dt = np.median(np.diff(b.t)) if len(b.t) > 1 else 1.0
        idx = np.clip(np.searchsorted(b.t, a.t), 0, len(b.t) - 1)
        idx2 = np.clip(idx - 1, 0, len(b.t) - 1)
        pick = np.where(np.abs(b.t[idx] - a.t) < np.abs(b.t[idx2] - a.t), idx, idx2)
        ok = np.abs(b.t[pick] - a.t) <= dt / 2
        X = triangulate([a.camera, b.camera], [a.uv[ok], b.uv[pick[ok]]])
        t = a.t[ok]
    good = np.isfinite(X).all(axis=1)
    order = np.argsort(t[good])
    return t[good][order], X[good][order]


def initial_guess(t: np.ndarray, X: np.ndarray, t_ref: float, n_early: int = 6
                  ) -> tuple[np.ndarray, np.ndarray]:
    """3次元点から打点時刻 t_ref の位置 [m]・速度 [m/s] の初期値を作る。

    序盤の点に2次多項式を当てはめて t_ref に外挿する（抵抗を無視するので速度は過小になりやすい）。
    """
    n = min(len(t), max(n_early, 3))
    tt, XX = t[:n] - t_ref, X[:n]
    deg = 2 if n >= 4 else 1
    coef = [np.polyfit(tt, XX[:, j], deg) for j in range(3)]
    p0 = np.array([np.polyval(c, 0.0) for c in coef])
    v0 = np.array([np.polyval(np.polyder(c), 0.0) for c in coef])
    return p0, v0


def _residuals(theta: np.ndarray, model: _Model, obs: list[Observations], base: PhysicsParams,
               t_ref: float) -> np.ndarray:
    p0, v0, vt = model.unpack(theta)
    phys = PhysicsParams(base.g, vt, base.method, base.rtol, base.atol)
    t_all = np.concatenate([o.t for o in obs])
    pos, _ = simulate(p0, v0, t_ref, t_all, phys)
    res, i = [], 0
    for o in obs:
        n = len(o.t)
        uv = o.camera.project(pos[i:i + n])
        r = (uv - o.uv).ravel()
        res.append(np.where(np.isfinite(r), r, 1e3))  # カメラの後ろに出た場合の罰則 [px]
        i += n
    res.append(model.prior_residuals(theta))
    return np.concatenate(res)


def _residuals_3d(theta: np.ndarray, model: _Model, t: np.ndarray, X: np.ndarray,
                  base: PhysicsParams, t_ref: float) -> np.ndarray:
    """3次元点に対する残差 [m]（初期値計算の第1段階）。"""
    p0, v0, vt = model.unpack(theta)
    phys = PhysicsParams(base.g, vt, base.method, base.rtol, base.atol)
    pos, _ = simulate(p0, v0, t_ref, t, phys)
    return (pos - X).ravel()


def _speed_gradient(theta: np.ndarray, model: _Model) -> np.ndarray:
    """初速 |v0| の θ に関する勾配（数値微分）。"""
    g = np.zeros_like(theta)
    for j in range(len(theta)):
        h = 1e-6 * max(1.0, abs(theta[j]))
        tp, tm = theta.copy(), theta.copy()
        tp[j] += h
        tm[j] -= h
        g[j] = (np.linalg.norm(model.unpack(tp)[1]) - np.linalg.norm(model.unpack(tm)[1])) / (2 * h)
    return g


def _speed_rate(p0: np.ndarray, v0: np.ndarray, phys: PhysicsParams) -> float:
    """初速の時間変化率 d|v|/dt [m/s^2]（打点時刻の誤差を信頼区間に加えるため）。"""
    a = _rhs(0.0, np.concatenate([p0, v0]), phys.g, phys.k)[3:]
    return float(a @ v0 / np.linalg.norm(v0))


def fit_trajectory(obs: Observations | list[Observations], t_ref: float, mode: str = "plane",
                   plane: FlightPlane | None = None, phys: PhysicsParams | None = None,
                   opts: FitOptions | None = None, theta0: np.ndarray | None = None
                   ) -> FitResult:
    """観測にシャトルの運動方程式を当てはめ、打点時刻 t_ref の位置・速度を推定する。

    Args:
        obs: 観測（カメラ1台なら plane モード、2台以上なら free モード）
        t_ref: 打点時刻 [s]。初速はこの時刻の速さとして定義する
        mode: "plane" または "free"
        plane: plane モードの飛行平面
        phys: 運動方程式の設定
        opts: フィッティングの設定
        theta0: 初期値（省略時は観測から自動計算）

    Returns:
        FitResult（信頼区間は ``opts.ci_method`` に従う）
    """
    obs = [obs] if isinstance(obs, Observations) else list(obs)
    phys = phys or PhysicsParams()
    opts = opts or FitOptions()
    if mode == "free" and len(obs) < 2:
        raise ValueError("free モードには2台以上のカメラの観測が必要です")
    model = _Model(mode, plane, opts.estimate_terminal_velocity, phys.terminal_velocity,
                   opts.refine_plane, np.deg2rad(opts.plane_sigma_deg), opts.plane_sigma_m)
    n_par = model.n_kin + model.n_plane + int(opts.estimate_terminal_velocity)
    n_res = 2 * sum(len(o.t) for o in obs)
    if n_res <= n_par:
        raise ValueError(f"観測点が少なすぎます（残差 {n_res} 個 ≤ パラメータ {n_par} 個）")

    if theta0 is None:
        t3, X3 = _points_3d(obs, mode, plane)
        if len(t3) < 3:
            raise ValueError("初期値の計算に使える3次元点が3点未満です")
        p0, v0 = initial_guess(t3, X3, t_ref)
        # 第1段階：3次元点に運動方程式を当てはめる（多項式外挿は強い減速を表せないため）。平面は固定
        m1 = _Model(mode, plane, opts.estimate_terminal_velocity, phys.terminal_velocity)
        stage1 = least_squares(_residuals_3d, m1.pack(p0, v0, phys.terminal_velocity),
                               args=(m1, t3, X3, phys, t_ref), loss="soft_l1", f_scale=0.1,
                               diff_step=opts.diff_step, x_scale="jac", max_nfev=200 * n_par)
        theta0 = np.insert(stage1.x, m1.n_kin, np.zeros(model.n_plane))

    def solve(o: list[Observations], th0: np.ndarray):
        return least_squares(_residuals, th0, args=(model, o, phys, t_ref), loss=opts.loss,
                             f_scale=opts.loss_scale_px, diff_step=opts.diff_step,
                             x_scale="jac", max_nfev=200 * n_par)

    sol = solve(obs, theta0)
    p0, v0, vt = model.unpack(sol.x)
    r = sol.fun[:n_res]
    rms = float(np.sqrt(np.mean(r**2)))

    # ヤコビアンからの共分散（Gauss–Newton 近似）
    cov = None
    dof = max(n_res - n_par, 1)
    try:
        J = sol.jac
        s2 = 2.0 * sol.cost / dof
        cov = np.linalg.inv(J.T @ J) * s2
        if not np.isfinite(cov).all():
            cov = None
    except np.linalg.LinAlgError:
        pass

    res = FitResult(mode=mode, t_ref=t_ref, p0=p0, v0=v0, terminal_velocity=vt, theta=sol.x,
                    cov_theta=cov, rms_px=rms, n_obs=n_res // 2, success=bool(sol.success),
                    message=str(sol.message), plane=model.plane_at(sol.x))

    # 打点時刻の不確かさによる初速のばらつき（d|v|/dt × σ_t）
    phys_fit = res.phys(phys)
    sd_timing = abs(_speed_rate(p0, v0, phys_fit)) * opts.t_ref_sigma
    res.speed0_sd_timing = sd_timing

    from scipy.stats import norm
    zq = norm.ppf(0.5 + opts.ci_level / 2)
    if opts.ci_method == "jacobian" and cov is not None:
        g = _speed_gradient(sol.x, model)
        sd_fit = float(np.sqrt(max(g @ cov @ g, 0.0)))
        sd = np.hypot(sd_fit, sd_timing)
        res.speed0_sd_fit = sd_fit
        if dof >= 3 and sd < res.speed0:
            res.speed0_ci = (res.speed0 - zq * sd, res.speed0 + zq * sd)
            res.ci_method = "jacobian"
        else:  # 自由度が小さい・不確かさが初速と同程度以上なら区間は意味をなさない
            res.ci_method = "jacobian(unreliable)"
    elif opts.ci_method == "bootstrap":
        rng = np.random.default_rng(opts.seed)
        speeds = []
        for _ in range(opts.bootstrap_n):
            sub = [o.subset(rng.integers(0, len(o.t), len(o.t))) for o in obs]
            if 2 * sum(len(o.t) for o in sub) <= n_par:
                continue
            b = solve(sub, sol.x)
            speeds.append(np.linalg.norm(model.unpack(b.x)[1]))
        speeds = np.asarray(speeds)
        if sd_timing > 0:  # 打点時刻の誤差を独立な正規誤差として加える
            speeds = speeds + rng.normal(0.0, sd_timing, len(speeds))
        lo, hi = np.quantile(speeds, [0.5 - opts.ci_level / 2, 0.5 + opts.ci_level / 2])
        res.speed0_ci = (float(lo), float(hi))
        res.speed0_sd_fit = float(np.std(speeds))
        res.bootstrap_speeds = speeds
        res.ci_method = "bootstrap"
    return res


# =============================================================================
# 速度指標
# =============================================================================
def speed_metrics(fit: FitResult, base: PhysicsParams, t_land: float | None = None,
                  x_arrival: float | None = None) -> dict[str, float]:
    """当てはめた軌道から速度指標を計算する。

    Args:
        fit: フィッティングの結果
        base: 運動方程式の設定（v_t はフィッティング結果のものを使う）
        t_land: 着地時刻 [s]。省略時は軌道が z = 0 に達する時刻
        x_arrival: 到達時速度を評価するコート上の x 座標 [m]（省略時は着地直前）

    Returns:
        各指標（``*_mps`` [m/s] と ``*_kmh`` [km/h]）、飛行時間 [s]、距離 [m]
    """
    phys = fit.phys(base)
    traj = simulate_until_ground(fit.p0, fit.v0, fit.t_ref, phys)
    t, pos, vel = traj["t"], traj["pos"], traj["vel"]
    if t_land is not None and np.isfinite(t_land) and t_land < t[-1]:
        keep = t <= t_land
        t, pos, vel = t[keep], pos[keep], vel[keep]
    t_end = float(t[-1])
    flight = t_end - fit.t_ref
    seg = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    path = float(seg.sum())
    straight = float(np.linalg.norm(pos[-1] - pos[0]))
    speed = np.linalg.norm(vel, axis=1)
    if x_arrival is not None and (pos[:, 0] >= x_arrival).any():
        i = int(np.argmax(pos[:, 0] >= x_arrival))
        v_arr = float(speed[i])
    else:
        v_arr = float(speed[-1])
    out = {
        "speed_initial": fit.speed0,
        "speed_mean_path": path / flight if flight > 0 else np.nan,
        "speed_mean_straight": straight / flight if flight > 0 else np.nan,
        "speed_arrival": v_arr,
    }
    metrics: dict[str, float] = {}
    for k, v in out.items():
        metrics[f"{k}_mps"] = v
        metrics[f"{k}_kmh"] = v * MPS_TO_KMH
    metrics["deceleration_ratio"] = v_arr / fit.speed0
    metrics["flight_time_s"] = flight
    metrics["path_length_m"] = path
    metrics["straight_distance_m"] = straight
    metrics["t_land_s"] = t_end
    return metrics


def finite_difference_speeds(t: np.ndarray, X: np.ndarray) -> dict[str, Any]:
    """フレーム間差分による素朴な速さ（物理モデルとの比較用）。

    Args:
        t: 時刻 (N,) [s]
        X: 3次元位置 (N, 3) [m]（NaN は除外）

    Returns:
        ``t_mid`` (M,) [s]、``speed`` (M,) [m/s]、``first_mps``（最初の区間）、``mean_mps``（全区間の平均）
    """
    ok = np.isfinite(X).all(axis=1) & np.isfinite(t)
    t, X = np.asarray(t)[ok], np.asarray(X)[ok]
    order = np.argsort(t)
    t, X = t[order], X[order]
    if len(t) < 2:
        return {"t_mid": np.array([]), "speed": np.array([]), "first_mps": np.nan, "mean_mps": np.nan}
    dt = np.diff(t)
    sp = np.linalg.norm(np.diff(X, axis=0), axis=1) / dt
    total = np.linalg.norm(np.diff(X, axis=0), axis=1).sum() / (t[-1] - t[0])
    return {"t_mid": (t[1:] + t[:-1]) / 2, "speed": sp, "first_mps": float(sp[0]),
            "mean_mps": float(total)}
