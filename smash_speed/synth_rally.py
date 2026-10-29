"""すべての種類のショットと、1 ラリー分の連続したショットの合成データ（分類・速度の正解用）。

ショットの種類ごとに「打点の高さ・打つ位置の深さ・狙う着地点の深さ・初速・打ち出し角」の範囲
（``SHOT_PROFILES``）を決め、範囲から引いた値で、狙った着地点に落ちる軌道を物理モデルで作る。
初速か打ち出し角の片方を引き、もう片方は着地点に届くように二分法で求める。ネットを越えない・
求めた値が範囲の外になるものは引き直す。

ラリーは、前のショットの軌道が相手側で「次のショットの打点の高さ」の範囲まで下りてきた所を次の打点にして、
ショットの種類の移り変わり（``NEXT_TYPES``）に沿ってつなぐ。最後のショットは床に落ちる。

深さはネットからの距離 [m]（コートの座標の |x|）。種類の名前は ``smash_speed.shots.SHOT_TYPES`` と同じ。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from smash_speed.camera import Camera, look_at
from smash_speed.court import draw_court
from smash_speed.physics import MPS_TO_KMH, PhysicsParams, simulate, simulate_until_ground
from smash_speed.shots import Rally, Shot, side_of
from smash_speed.synth import write_click_audio

NET_HEIGHT_M = 1.55       # ネットの高さ（端の高さ。中央は 1.524）[m]
NET_MARGIN_M = 0.05       # ネットの上をこれだけ空けて越える [m]
HALF_LENGTH_M = 6.70      # ネット〜バックバウンダリーライン [m]
HALF_WIDTH_M = 2.40       # 狙う着地点・打点の y の範囲（シングルスのサイドラインの少し内側）[m]


@dataclass(frozen=True)
class ShotProfile:
    """ショットの種類ごとの範囲。``solve`` が "speed" なら打ち出し角を引いて初速を求め、"elevation" なら逆。

    求めた方の値が範囲の外なら、そのショットは作り直す。
    """

    shot_type: str
    solve: str                          # speed | elevation
    speed_mps: tuple[float, float]      # 初速
    elevation_deg: tuple[float, float]  # 打ち出し角（上向きが正）
    hit_z_m: tuple[float, float]        # 打点の高さ
    hit_depth_m: tuple[float, float]    # 打つ位置の深さ（ネットから）
    land_depth_m: tuple[float, float]   # 狙う着地点の深さ（相手側のネットから）
    max_apex_m: float = 20.0            # 軌道のいちばん高い所の上限（ドライブ・プッシュは低い）


# ShuttleSet の分類を ROADMAP B4 の 8 種類にまとめたもの。サーブはショート・ロングの 2 通り
SHOT_PROFILES: dict[str, ShotProfile] = {
    "serve_short": ShotProfile("serve", "speed", (4.0, 20.0), (0.0, 25.0), (0.9, 1.1), (2.0, 2.8), (2.0, 2.8)),
    "serve_long": ShotProfile("serve", "speed", (15.0, 70.0), (35.0, 60.0), (0.9, 1.1), (2.0, 3.0), (5.4, 6.6)),
    "smash": ShotProfile("smash", "elevation", (40.0, 110.0), (-35.0, -2.0), (2.3, 3.1), (2.0, 6.3), (1.0, 5.5)),
    "clear": ShotProfile("clear", "speed", (15.0, 90.0), (15.0, 50.0), (2.0, 3.0), (3.5, 6.6), (5.0, 6.6)),
    "drop": ShotProfile("drop", "speed", (5.0, 40.0), (-15.0, 15.0), (2.0, 3.0), (3.0, 6.6), (0.5, 2.5)),
    "drive": ShotProfile("drive", "elevation", (20.0, 45.0), (-10.0, 20.0), (0.8, 1.8), (1.5, 5.5), (3.0, 6.6),
                         max_apex_m=2.8),
    "lob": ShotProfile("lob", "speed", (10.0, 80.0), (35.0, 65.0), (0.2, 1.2), (0.3, 5.5), (5.0, 6.6)),
    "net": ShotProfile("net", "speed", (1.0, 15.0), (0.0, 60.0), (0.2, 1.4), (0.3, 3.0), (0.2, 1.5),
                       max_apex_m=2.3),
    "push": ShotProfile("push", "elevation", (8.0, 22.0), (-15.0, 15.0), (1.2, 2.0), (0.5, 2.8), (2.0, 4.5),
                        max_apex_m=2.3),
}

# 次に打たれやすいショット（前のショット → 返し方）。重みは引く割合
NEXT_TYPES: dict[str, dict[str, float]] = {
    "serve_short": {"net": 3, "push": 2, "lob": 2, "drive": 1},
    "serve_long": {"clear": 2, "smash": 2, "drop": 2},
    "smash": {"lob": 3, "drive": 2, "net": 2},
    "clear": {"clear": 2, "smash": 3, "drop": 3},
    "drop": {"net": 3, "lob": 3, "push": 1},
    "drive": {"drive": 3, "push": 1, "drop": 1, "clear": 1},
    "lob": {"smash": 4, "clear": 2, "drop": 2},
    "net": {"net": 3, "lob": 3, "push": 2},
    "push": {"drive": 2, "lob": 2, "net": 1, "smash": 1},
}


@dataclass
class FlightSegment:
    """ラリーの中の 1 本のショットの正解。時刻 [s]、位置 [m]、速度 [m/s]。

    Attributes:
        profile: ``SHOT_PROFILES`` のキー（serve_short など）
        t_hit: 打った時刻
        p0, v0: 打点と、打点での速度
        t_end: 飛行が終わった時刻（相手が打った時刻、最後のショットは着地した時刻）
        t_land: 誰も打たなかったら床に着いた時刻
        landing_xy: 誰も打たなかったら床に着いた位置
        intercepted: 相手が打ち返したか（False なら床に落ちた）
    """

    profile: str
    t_hit: float
    p0: np.ndarray
    v0: np.ndarray
    t_end: float
    t_land: float
    landing_xy: np.ndarray
    intercepted: bool

    @property
    def shot_type(self) -> str:
        return SHOT_PROFILES[self.profile].shot_type

    @property
    def speed0(self) -> float:
        return float(np.linalg.norm(self.v0))

    @property
    def elevation_deg(self) -> float:
        return float(np.degrees(np.arctan2(self.v0[2], np.linalg.norm(self.v0[:2]))))

    @property
    def side(self) -> str:
        return side_of(float(self.p0[0]))


@dataclass
class SyntheticRally:
    """合成ラリー 1 つの正解（ショットの並び）。"""

    segments: list[FlightSegment] = field(default_factory=list)
    end: str = "in"   # 終わり方：in（相手のコートの中）| out（外）| net（ネットを越えずに自分の側）

    @property
    def t_start(self) -> float:
        return self.segments[0].t_hit

    @property
    def t_end(self) -> float:
        return self.segments[-1].t_end

    def to_shots(self, time_offset: float = 0.0, rally_id: int = 1,
                 phys: PhysicsParams | None = None) -> list[Shot]:
        """``smash_speed.shots.Shot`` の形の正解。時刻には ``time_offset`` を足す（動画の時刻に合わせる）。

        着地点・着地時刻は「誰も打たなかったら床に着いた」所（相手が打ったショットも埋める）。
        """
        vt = (phys or PhysicsParams()).terminal_velocity
        out = []
        for i, s in enumerate(self.segments, start=1):
            kmh = s.speed0 * MPS_TO_KMH
            out.append(Shot(
                t_hit_s=s.t_hit + time_offset, side=s.side, shot_type=s.shot_type, speed_kmh=kmh,
                speed_ci_low_kmh=kmh, speed_ci_high_kmh=kmh,
                hit_x_m=float(s.p0[0]), hit_y_m=float(s.p0[1]), hit_z_m=float(s.p0[2]),
                land_x_m=float(s.landing_xy[0]), land_y_m=float(s.landing_xy[1]), t_land_s=s.t_land + time_offset,
                v0_x_mps=float(s.v0[0]), v0_y_mps=float(s.v0[1]), v0_z_mps=float(s.v0[2]),
                terminal_velocity_mps=vt, confidence=1.0, rally_id=rally_id, index_in_rally=i,
                source="synthetic", ref=s.profile, note="" if s.intercepted else "床に落ちた"))
        return out

    def to_rally(self, time_offset: float = 0.0, rally_id: int = 1) -> Rally:
        """``smash_speed.shots.Rally`` の形の正解。得点は、中に落ちれば最後に打った側、外・ネットなら相手。"""
        last = self.segments[-1].side
        winner = last if self.end == "in" else ("o" if last == "h" else "h")
        return Rally(rally_id=rally_id, t_start_s=self.t_start + time_offset, t_end_s=self.t_end + time_offset,
                     n_shots=len(self.segments), winner_side=winner, end_reason=self.end)


# =============================================================================
# 1 本のショット
# =============================================================================
def _solver_phys(phys: PhysicsParams) -> PhysicsParams:
    """着地点を探すときの積分は粗くてよい（できた軌道は ``phys`` で積分し直す）。"""
    return PhysicsParams(phys.g, phys.terminal_velocity, "RK45", 1e-6, 1e-8)


def _velocity(direction: np.ndarray, speed: float, elevation_deg: float) -> np.ndarray:
    el = np.deg2rad(elevation_deg)
    return speed * (np.cos(el) * np.r_[direction, 0.0] + np.sin(el) * np.array([0.0, 0.0, 1.0]))


def _range(p0: np.ndarray, direction: np.ndarray, speed: float, elevation_deg: float,
           phys: PhysicsParams) -> float:
    """打点から床に着くまでの水平距離 [m]（着かなければ inf）。"""
    tr = simulate_until_ground(p0, _velocity(direction, speed, elevation_deg), 0.0, phys, t_max=8.0, n_dense=2)
    if not np.isfinite(tr["t_land"]):
        return np.inf
    return float(np.linalg.norm(tr["pos"][-1, :2] - p0[:2]))


def _bisect(f, lo: float, hi: float, target: float, n_iter: int = 24) -> float | None:
    """単調に増える f で f(x) = target となる x を [lo, hi] から探す。範囲の外なら None。"""
    f_lo, f_hi = f(lo), f(hi)
    if not f_lo <= target <= f_hi:
        return None
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def aim_shot(profile: str, p0: np.ndarray, target_xy: np.ndarray, rng: np.random.Generator,
             phys: PhysicsParams, n_tries: int = 6) -> np.ndarray | None:
    """打点 ``p0`` から ``target_xy`` に落ちる、種類 ``profile`` の初速度 [m/s] を作る。

    範囲から初速か打ち出し角を引き、もう片方を求める（``n_tries`` 回まで引き直す）。
    範囲に収まらない・ネットを越えない・高すぎるときは None。
    """
    pr = SHOT_PROFILES[profile]
    d = np.asarray(target_xy, float) - p0[:2]
    dist = float(np.linalg.norm(d))
    if dist < 0.5:
        return None
    d /= dist
    sp = _solver_phys(phys)
    for _ in range(n_tries):
        if pr.solve == "speed":
            el = rng.uniform(*pr.elevation_deg)
            speed = _bisect(lambda s: _range(p0, d, s, el, sp), *pr.speed_mps, dist)
        else:
            speed = rng.uniform(*pr.speed_mps)
            el = _bisect(lambda e: _range(p0, d, speed, e, sp), *pr.elevation_deg, dist)
        if speed is None or el is None:
            continue
        v0 = _velocity(d, speed, el)
        if _clears_net(p0, v0, pr.max_apex_m, sp):
            return v0
    return None


def _clears_net(p0: np.ndarray, v0: np.ndarray, max_apex_m: float, phys: PhysicsParams) -> bool:
    """ネットの真上でネットより ``NET_MARGIN_M`` 以上高く、軌道の最高点が ``max_apex_m`` 以下か。"""
    tr = simulate_until_ground(p0, v0, 0.0, phys, t_max=8.0, n_dense=600)
    x, z = tr["pos"][:, 0], tr["pos"][:, 2]
    if z.max() > max_apex_m:
        return False
    cross = np.nonzero(np.sign(x[:-1]) != np.sign(x[1:]))[0]
    if not len(cross):
        return False
    i = cross[0]
    w = x[i] / (x[i] - x[i + 1])
    return bool(z[i] + w * (z[i + 1] - z[i]) >= NET_HEIGHT_M + NET_MARGIN_M)


def sample_target(profile: str, hitter_sign: float, rng: np.random.Generator) -> np.ndarray:
    """種類 ``profile`` の狙う着地点（相手側）を引く。"""
    pr = SHOT_PROFILES[profile]
    return np.array([-hitter_sign * rng.uniform(*pr.land_depth_m), rng.uniform(-HALF_WIDTH_M, HALF_WIDTH_M)])


def make_typed_shot(profile: str, rng: np.random.Generator, phys: PhysicsParams, side: str = "h",
                    p0: np.ndarray | None = None, t_hit: float = 0.0,
                    max_tries: int = 200) -> FlightSegment:
    """種類 ``profile`` の合成ショットを 1 本作る（床に落ちるまで）。

    ``p0`` を省略すると、打点の高さ・深さを範囲から引く（``side`` の側から打つ）。
    """
    pr = SHOT_PROFILES[profile]
    sign = -1.0 if side == "h" else 1.0
    for _ in range(max_tries):
        if p0 is None:
            p = np.array([sign * rng.uniform(*pr.hit_depth_m), rng.uniform(-HALF_WIDTH_M, HALF_WIDTH_M),
                          rng.uniform(*pr.hit_z_m)])
        else:
            p = np.asarray(p0, float)
        v0 = aim_shot(profile, p, sample_target(profile, sign, rng), rng, phys)
        if v0 is not None:
            return _segment(profile, p, v0, t_hit, phys)
    raise RuntimeError(f"{profile} のショットを作れませんでした（打点 {p0}）")


def _segment(profile: str, p0: np.ndarray, v0: np.ndarray, t_hit: float, phys: PhysicsParams) -> FlightSegment:
    tr = simulate_until_ground(p0, v0, t_hit, phys, t_max=8.0)
    t_land = float(tr["t_land"])
    return FlightSegment(profile, t_hit, p0, v0, t_land, t_land, tr["pos"][-1, :2].copy(), False)


# =============================================================================
# ラリー
# =============================================================================
def intercept(seg: FlightSegment, hit_z_m: tuple[float, float], rng: np.random.Generator, phys: PhysicsParams,
              min_flight_s: float = 0.25) -> tuple[float, np.ndarray] | None:
    """ショット ``seg`` を相手が打ち返す時刻と位置。

    相手側のコートの上で下りてきていて、高さが ``hit_z_m`` の範囲にある間の時刻を一様に引く。
    そういう所がなければ None。
    """
    tr = simulate_until_ground(seg.p0, seg.v0, seg.t_hit, phys, t_max=8.0, n_dense=2000)
    t, pos, vel = tr["t"], tr["pos"], tr["vel"]
    ok = ((np.sign(pos[:, 0]) == -np.sign(seg.p0[0])) & (np.abs(pos[:, 0]) <= HALF_LENGTH_M)
          & (np.abs(pos[:, 1]) <= HALF_WIDTH_M + 0.6) & (vel[:, 2] < 0) & (t - seg.t_hit >= min_flight_s)
          & (pos[:, 2] >= hit_z_m[0]) & (pos[:, 2] <= hit_z_m[1]))
    idx = np.nonzero(ok)[0]
    if not len(idx):
        return None
    i = int(rng.choice(idx))
    return float(t[i]), pos[i].copy()


def make_rally(rng: np.random.Generator, phys: PhysicsParams, n_shots: int = 8, server_side: str = "h",
               serve: str | None = None, t_start: float = 0.0, max_tries: int = 12, end: str = "in") -> SyntheticRally:
    """サーブから始まる最大 ``n_shots`` 本の合成ラリーを作る。

    次のショットは ``NEXT_TYPES`` から重みで引き、前の軌道が相手側で下りてきて、その種類の打点の高さの
    範囲にある間に打つ（``intercept``。打つ位置の深さが範囲から 0.5 m 以上外れるものは引き直す）。
    つなげなくなったら、そこで最後のショットが床に落ちて終わる。

    Args:
        serve: serve_short | serve_long（省略時はどちらかを引く）
        end: 最後のショットの落ち方。"in" は相手のコートの中（今までどおり）、"out" は線の外、
            "net" はネットを越えずに自分の側（``fault_shot``）。外・ネットにできなければ "in" のまま
    """
    serve = serve or ("serve_short" if rng.uniform() < 0.6 else "serve_long")
    segs = [make_typed_shot(serve, rng, phys, side=server_side, t_hit=t_start)]
    while len(segs) < n_shots:
        prev = segs[-1]
        nxt = _continue(prev, rng, phys, max_tries)
        if nxt is None:
            break
        t_hit, p, profile, v0 = nxt
        prev.t_end, prev.intercepted = t_hit, True
        segs.append(_segment(profile, p, v0, t_hit, phys))
    if end != "in":
        last = segs[-1]
        v0 = fault_shot(end, last.profile, last.p0, rng, phys)
        if v0 is None:
            end = "in"
        else:
            segs[-1] = _segment(last.profile, last.p0, v0, last.t_hit, phys)
    return SyntheticRally(segs, end)


def fault_shot(end: str, profile: str, p0: np.ndarray, rng: np.random.Generator, phys: PhysicsParams,
               n_tries: int = 12) -> np.ndarray | None:
    """打点 ``p0`` から、外（"out"）かネットの手前（"net"）に落ちる初速度。作れなければ None。

    out：相手側の線の外（バックバウンダリーラインの 0.15〜0.9 m 先か、サイドラインの 0.15〜0.8 m 外）を狙う。
    打ち出し角は種類の範囲から引き、初速は 1〜120 m/s から求める（ネットを越えること）。
    net：自分の側のネットの手前 0.1〜0.8 m を狙う遅いショット（ネットに掛かって落ちたものの代わり。網は物理モデルにない）。
    """
    pr = SHOT_PROFILES[profile]
    sign = np.sign(p0[0])
    sp = _solver_phys(phys)
    for _ in range(n_tries):
        if end == "out":
            if rng.uniform() < 0.5:
                target = np.array([-sign * (HALF_LENGTH_M + rng.uniform(0.15, 0.9)), rng.uniform(-2.0, 2.0)])
            else:
                target = np.array([-sign * rng.uniform(1.0, HALF_LENGTH_M - 0.3),
                                   rng.choice([-1, 1]) * (3.05 + rng.uniform(0.15, 0.8))])
            el = rng.uniform(*pr.elevation_deg)
        else:
            if abs(p0[0]) < 0.4:
                return None
            target = np.array([sign * rng.uniform(0.1, min(0.8, abs(p0[0]) - 0.2)), p0[1] + rng.uniform(-0.5, 0.5)])
            el = rng.uniform(-10.0, 30.0)
        d = target - p0[:2]
        dist = float(np.linalg.norm(d))
        d /= dist
        speed = _bisect(lambda v: _range(p0, d, v, el, sp), 1.0, 120.0, dist)
        if speed is None:
            continue
        v0 = _velocity(d, speed, el)
        if end == "out" and not _clears_net(p0, v0, max(pr.max_apex_m, 4.0), sp):
            continue
        return v0
    return None


def _continue(prev: FlightSegment, rng: np.random.Generator, phys: PhysicsParams,
              max_tries: int) -> tuple[float, np.ndarray, str, np.ndarray] | None:
    """前のショットへの返球（打った時刻・打点・種類・初速度）。作れなければ None。"""
    choices = NEXT_TYPES[prev.profile]
    names = list(choices)
    weights = np.array([choices[n] for n in names], float)
    for _ in range(max_tries):
        profile = names[rng.choice(len(names), p=weights / weights.sum())]
        pr = SHOT_PROFILES[profile]
        hit = intercept(prev, pr.hit_z_m, rng, phys)
        if hit is None:
            continue
        t_hit, p = hit
        lo, hi = pr.hit_depth_m
        if not lo - 0.5 <= abs(p[0]) <= hi + 0.5:
            continue
        v0 = aim_shot(profile, p, sample_target(profile, np.sign(p[0]), rng), rng, phys)
        if v0 is not None:
            return t_hit, p, profile, v0
    return None


def rally_positions(rally: SyntheticRally, times: np.ndarray, phys: PhysicsParams
                    ) -> tuple[np.ndarray, np.ndarray]:
    """各時刻のシャトルの位置 [m] と、何本目のショットか（0 から。飛んでいなければ位置は NaN・番号は -1）。"""
    times = np.asarray(times, float)
    pos = np.full((len(times), 3), np.nan)
    which = np.full(len(times), -1)
    for i, s in enumerate(rally.segments):
        m = (times >= s.t_hit) & (times < s.t_end) if s.intercepted else (times >= s.t_hit) & (times <= s.t_end)
        if m.any():
            pos[m] = simulate(s.p0, s.v0, s.t_hit, times[m], phys)[0]
            which[m] = i
    return pos, which


def make_session(rng: np.random.Generator, phys: PhysicsParams, n_rallies: int = 6,
                 n_shots: tuple[int, int] = (2, 12), rest_s: tuple[float, float] = (1.0, 8.0),
                 t_start: float = 0.5, ends: dict[str, float] | None = None) -> list[SyntheticRally]:
    """続けて打った ``n_rallies`` 個の合成ラリー（1 本の動画の中の練習・試合）。

    ラリーの本数は ``n_shots`` の範囲から引く（つなげなくなればそれより短い）。前のラリーのシャトルが
    床に落ちてから ``rest_s`` の範囲の間だけ休んで、次のラリーのサーブを打つ。サーブは前のラリーで
    得点した側が打つ。``ends`` は終わり方の割合（例 ``{"in": 0.5, "out": 0.3, "net": 0.2}``。省略時はすべて in）。
    """
    out: list[SyntheticRally] = []
    t, server = t_start, "h"
    for _ in range(n_rallies):
        n = int(rng.integers(n_shots[0], n_shots[1] + 1))
        end = "in" if not ends else str(rng.choice(list(ends), p=np.array(list(ends.values())) / sum(ends.values())))
        r = make_rally(rng, phys, n_shots=n, server_side=server, t_start=t, end=end)
        out.append(r)
        server = r.to_rally().winner_side
        t = r.t_end + float(rng.uniform(*rest_s))
    return out


def observe_session(rallies: list[SyntheticRally], camera: Camera, fps: float, phys: PhysicsParams,
                    rng: np.random.Generator, noise_px: float = 1.5, dropout: float = 0.1,
                    post_s: float = 0.5) -> pd.DataFrame:
    """続けて打ったラリー（``make_session``）を、休みの間も含めて fps [Hz] のカメラで観測した検出結果。

    休みの間（シャトルが床にある・拾っている）は見えない（``visible`` = 0）とする。列は ``observe_rally`` と同じに、
    何個目のラリーか（``rally_index``、0 から。休みの間は -1）を足したもの。``shot_index`` はラリーの中の番号。
    """
    dt = 1.0 / fps
    times = np.arange(rallies[0].t_start + rng.uniform(0.0, 1.0) * dt, rallies[-1].t_end + post_s, dt)
    pos = np.full((len(times), 3), np.nan)
    which = np.full(len(times), -1)
    rally_index = np.full(len(times), -1)
    for k, r in enumerate(rallies):
        p, w = rally_positions(r, times, phys)
        m = w >= 0
        pos[m], which[m], rally_index[m] = p[m], w[m], k
    uv = camera.project(np.nan_to_num(pos)) + rng.normal(0.0, noise_px, (len(times), 2))
    inside = (np.isfinite(pos).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < camera.width)
              & (uv[:, 1] >= 0) & (uv[:, 1] < camera.height))
    visible = inside & (rng.uniform(size=len(times)) >= dropout)
    return pd.DataFrame({
        "frame": np.arange(len(times)), "timestamp": times,
        "x_px": np.where(visible, uv[:, 0], np.nan), "y_px": np.where(visible, uv[:, 1], np.nan),
        "visible": visible.astype(int), "confidence": visible.astype(float),
        "x_m": pos[:, 0], "y_m": pos[:, 1], "z_m": pos[:, 2], "shot_index": which, "rally_index": rally_index,
    })


def rally_camera(cfg: dict[str, Any]) -> Camera:
    """コート全体と高い軌道が入る、ラリー用の仮想カメラ（コートの側方・遠く）。"""
    c = cfg["synthetic"]["rally_camera"]
    return look_at(np.array(c["eye_m"]), np.array(c["target_m"]), c["f_px"], c["width"], c["height"])


def observe_rally(rally: SyntheticRally, camera: Camera, fps: float, phys: PhysicsParams,
                  rng: np.random.Generator, noise_px: float = 1.5, dropout: float = 0.1,
                  phase: float | None = None) -> pd.DataFrame:
    """合成ラリーを fps [Hz] のカメラで観測した検出結果（瞬間露光）。

    Returns:
        列 ``frame, timestamp, x_px, y_px, visible, confidence, x_m, y_m, z_m, shot_index``
        （後ろ 4 つは正解。shot_index は何本目のショットか、0 から）
    """
    dt = 1.0 / fps
    phase = rng.uniform(0.0, 1.0) if phase is None else phase
    times = np.arange(rally.t_start + phase * dt, rally.t_end, dt)
    pos, which = rally_positions(rally, times, phys)
    uv = camera.project(np.nan_to_num(pos)) + rng.normal(0.0, noise_px, (len(times), 2))
    inside = (np.isfinite(pos).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < camera.width)
              & (uv[:, 1] >= 0) & (uv[:, 1] < camera.height))
    visible = inside & (rng.uniform(size=len(times)) >= dropout)
    return pd.DataFrame({
        "frame": np.arange(len(times)), "timestamp": times,
        "x_px": np.where(visible, uv[:, 0], np.nan), "y_px": np.where(visible, uv[:, 1], np.nan),
        "visible": visible.astype(int), "confidence": visible.astype(float),
        "x_m": pos[:, 0], "y_m": pos[:, 1], "z_m": pos[:, 2], "shot_index": which,
    })


def render_rally_video(rally: SyntheticRally, camera: Camera, cfg: dict[str, Any], phys: PhysicsParams,
                       path: str | Path, fps: float = 120.0, pre_s: float = 0.3, post_s: float = 0.3,
                       radius_px: int = 4, audio_path: str | Path | None = None, audio_rate: int = 48000,
                       seed: int = 0, speed_of_sound: float | None = 343.0) -> pd.DataFrame:
    """白い背景にコートと小さな点（シャトル）が動く、1 ラリー分の合成動画を書き出す（瞬間露光）。

    動画の時刻 0 は ``rally.t_start - pre_s``（正解のショットは ``rally.to_shots(pre_s - rally.t_start)``）。
    ``audio_path`` を与えると、打つたびにクリック音を入れた WAV も書き出す
    （打点からカメラまでの距離 / 音速だけ遅れる）。

    Returns:
        各フレームの正解（``frame, timestamp, x_px, y_px, visible, shot_index``。timestamp は動画の時刻 [s]）
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = camera.width, camera.height
    base = np.full((h, w, 3), 255, np.uint8)
    draw_court(base, camera, cfg, color_line=(170, 170, 170), color_net=(120, 120, 120), thickness=2)
    t0 = rally.t_start - pre_s
    n_frames = int(np.ceil((rally.t_end - t0 + post_s) * fps))
    t_vid = np.arange(n_frames) / fps
    pos, which = rally_positions(rally, t0 + t_vid, phys)
    uv = camera.project(np.nan_to_num(pos))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    rows = []
    for k in range(n_frames):
        frame = base.copy()
        vis = bool(which[k] >= 0 and 0 <= uv[k, 0] < w and 0 <= uv[k, 1] < h)
        if vis:
            cv2.circle(frame, (int(round(uv[k, 0])), int(round(uv[k, 1]))), radius_px, (30, 30, 30), -1, cv2.LINE_AA)
        rows.append({"frame": k, "timestamp": t_vid[k], "x_px": uv[k, 0] if vis else np.nan,
                     "y_px": uv[k, 1] if vis else np.nan, "visible": int(vis), "shot_index": int(which[k])})
        writer.write(frame)
    writer.release()

    if audio_path is not None:
        clicks = []
        for s in rally.segments:
            delay = 0.0 if speed_of_sound is None else float(np.linalg.norm(camera.center - s.p0)) / speed_of_sound
            clicks.append((s.t_hit - t0 + delay, 0.6 if s.shot_type in ("smash", "drive", "clear") else 0.35))
        write_click_audio(audio_path, n_frames / fps, clicks, audio_rate, seed)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    import argparse

    from smash_speed.config import load_config
    from smash_speed.shots import RALLIES_FILE, SHOTS_FILE, write_rallies, write_shots

    ap = argparse.ArgumentParser(description="1 ラリー分の合成動画と、ショットの正解（shots.csv）を書き出す")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shots", type=int, default=10, help="ショットの数の上限")
    ap.add_argument("--fps", type=float, default=120.0)
    ap.add_argument("--config", type=Path, default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    phys = PhysicsParams.from_config(cfg)
    rally = make_rally(np.random.default_rng(a.seed), phys, n_shots=a.shots)
    pre_s = 0.3
    truth = render_rally_video(rally, rally_camera(cfg), cfg, phys, a.out_dir / "rally.mp4", fps=a.fps, pre_s=pre_s,
                               audio_path=a.out_dir / "rally.wav", seed=a.seed)
    truth.to_csv(a.out_dir / "truth_frames.csv", index=False)
    offset = pre_s - rally.t_start
    write_shots(a.out_dir / SHOTS_FILE, rally.to_shots(offset, phys=phys))
    write_rallies(a.out_dir / RALLIES_FILE, [rally.to_rally(offset)])
    kinds = "・".join(SHOT_PROFILES[s.profile].shot_type for s in rally.segments)
    print(f"{len(rally.segments)} 本（{kinds}）を {a.out_dir} に書き出しました")


if __name__ == "__main__":
    main()
