"""続けて打った合成ラリー（``synth_rally.make_session``）に、選手の足の動きと、ラリーの間の「打球ではない打球」を足す（ROADMAP C の「選手の動き」）。

ラリーの間には、ラリーではないのにシャトルが打たれることがある。

- 拾って返す（``pickup``）：床に落ちたシャトルを、落ちた側の選手が拾って、相手へゆっくり打ち返す。
  打つ人は歩いてシャトルの所へ行き、拾ってすぐ打つ。相手は歩いて受け取りに行く。外に落ちたシャトルなら、
  拾う人はコートの外にいる。ときどき、拾わない方の選手もコートの外へ歩いていく（タオル・水）。
- となりのコートのシャトル（``neighbor``）：となりのコート（y の向きに ``NEIGHBOR_Y_M`` だけずれた所）で打たれたシャトル。
  選手は 2 人ともサーブの位置に止まって待っている。

どちらも打球音があり、シャトルも見えるので、打球として見つかり、短い（1 本の）ラリーになる。

``session_feet`` が、選手の足元の正解（側 → (n, 2) [m]）を作る。ラリーの中では、打つ選手は打点の近くへ素早く動き
（躍度最小。3 m で最高 5 m/s ほど）、打った後はホーム（自分の半面の中ほど）へ戻る。ラリーの間は歩いて
（最高 1.5 m/s ほど）次のサーブの位置へ行き、サーブの前は止まって待つ。サーブの位置は、打つ側がサーブの
打点の近く、受ける側が対角のサービスコート。``synth_players.observe_players`` で ``players.csv`` の形にする。

既存の関数（``make_session``・``observe_session``・``footwork_path``）は変えずに、組み合わせて使う。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from smash_speed.camera import Camera
from smash_speed.physics import PhysicsParams
from smash_speed.synth_players import _min_jerk, observe_players
from smash_speed.synth_rally import (FlightSegment, SyntheticRally, _bisect, _range, _segment, _solver_phys,
                                     _velocity, make_session, make_typed_shot, observe_session)

NEIGHBOR_Y_M = 8.0     # となりのコートの中心までの距離 [m]（コートの幅 6.1 m ＋ 間 2 m ほど）
PLAY_PACE_MPS = 4.5    # ラリーの中の動きの平均の速さの目安 [m/s]（最高はその 1.9 倍ほど。時間が足りなければ遅れて着く）
WALK_PACE_MPS = 0.9    # 歩く平均の速さ [m/s]（最高 1.6 m/s ほど）
MOVE_BASE_S = 0.25     # 1 回の移動に、距離によらずかかる時間 [s]
HOME_X_M = 3.4         # ホームの位置のネットからの距離 [m]
REACH_M = 0.6          # 打点と足元のずれの大きさ（標準偏差）[m]


@dataclass
class ExtraShot:
    """ラリーの間の、ラリーではない打球の正解。"""

    seg: FlightSegment
    kind: str              # pickup | neighbor
    after_rally: int       # 何個目のラリーの後か（0 から）
    picker_side: str = ""  # 拾って返した選手の側（pickup だけ）

    def as_rally(self) -> SyntheticRally:
        return SyntheticRally([self.seg], "in")


@dataclass
class _Waypoint:
    t: float               # この時刻までに着く（着けなければ遅れて着く）
    xy: np.ndarray
    pace: float            # 平均の速さ [m/s]
    on_time: bool = False  # 遅れそうなら速く動いて、時刻どおりに着く（サーブの位置。最短 0.5 秒）


def _sx(side: str) -> float:
    return -1.0 if side == "h" else 1.0


def pickup_shot(land_xy: np.ndarray, t_hit: float, rng: np.random.Generator, phys: PhysicsParams,
                n_tries: int = 12) -> FlightSegment | None:
    """床の ``land_xy`` から拾って、相手の側の中ほどへゆっくり打ち返す（下から、高さ 0.7〜1.0 m）。作れなければ None。"""
    sign = 1.0 if land_xy[0] >= 0 else -1.0
    p0 = np.array([land_xy[0], land_xy[1], rng.uniform(0.7, 1.0)])
    sp = _solver_phys(phys)
    for _ in range(n_tries):
        target = np.array([-sign * rng.uniform(2.5, 4.5), rng.uniform(-1.5, 1.5)])
        el = rng.uniform(30.0, 50.0)
        d = target - p0[:2]
        dist = float(np.linalg.norm(d))
        d /= dist
        speed = _bisect(lambda v: _range(p0, d, v, el, sp), 1.0, 40.0, dist)
        if speed is not None:
            return _segment("lob", p0, _velocity(d, speed, el), t_hit, phys)
    return None


def neighbor_shot(t_hit: float, rng: np.random.Generator, phys: PhysicsParams) -> FlightSegment:
    """となりのコートで打たれた 1 本（自分のコートの合成ショットを y の向きにずらしたもの）。"""
    seg = make_typed_shot(str(rng.choice(["drive", "push", "drop", "clear"])), rng, phys,
                          side=str(rng.choice(["h", "o"])), t_hit=t_hit)
    shift = np.array([0.0, NEIGHBOR_Y_M, 0.0])
    return replace(seg, p0=seg.p0 + shift, landing_xy=seg.landing_xy + shift[:2])


def serve_positions(serve: FlightSegment, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """サーブを打つ側（サーブの打点の近く）と受ける側（対角のサービスコート）の足元の位置。"""
    server = serve.side
    other = "o" if server == "h" else "h"
    feet = serve.p0[:2] + np.clip(rng.normal(0, 0.2, 2), -0.4, 0.4)
    s = float(np.sign(feet[1])) or 1.0
    return {server: feet, other: np.array([_sx(other) * rng.uniform(2.6, 3.4), -s * rng.uniform(0.5, 1.0)])}


def make_player_session(rng: np.random.Generator, phys: PhysicsParams, n_rallies: int = 6,
                        rest_s: tuple[float, float] = (7.0, 11.0), p_pickup: float = 0.5, p_neighbor: float = 0.3,
                        ends: dict[str, float] | None = None, **kw: Any) -> tuple[list[SyntheticRally], list[ExtraShot]]:
    """``make_session`` のラリーと、ラリーの間の打球（``ExtraShot``）。

    ラリーの間ごとに、``p_pickup`` の割合で拾って返す打球、``p_neighbor`` の割合でとなりのコートの打球を 1 本入れる
    （どちらも、次のサーブの 1.5 秒より前に床に落ちるものだけ）。
    """
    session = make_session(rng, phys, n_rallies, rest_s=rest_s, ends=ends, **kw)
    extras: list[ExtraShot] = []
    for k, (a, b) in enumerate(zip(session[:-1], session[1:])):
        u = rng.uniform()
        land = a.segments[-1].landing_xy
        if u < p_pickup:
            seg = pickup_shot(land, a.t_end + rng.uniform(1.2, 2.0), rng, phys)
            if seg is not None and seg.t_land < b.t_start - 1.5:
                extras.append(ExtraShot(seg, "pickup", k, "o" if land[0] >= 0 else "h"))
        elif u < p_pickup + p_neighbor:
            seg = neighbor_shot(a.t_end + rng.uniform(2.0, 3.0), rng, phys)
            if seg.t_land < b.t_start - 1.5:
                extras.append(ExtraShot(seg, "neighbor", k))
    return session, extras


def _plan(session: list[SyntheticRally], extras: list[ExtraShot], rng: np.random.Generator,
          court: dict[str, Any] | None = None) -> dict[str, list[_Waypoint]]:
    """選手ごとの、行く所と着く時刻の並び。"""
    c = court or {}
    half_w = c.get("half_width_doubles_m", 3.05)
    plan: dict[str, list[_Waypoint]] = {"h": [], "o": []}
    by_rest = {e.after_rally: e for e in extras}
    for k, r in enumerate(session):
        pos = serve_positions(r.segments[0], rng)
        hold = rng.uniform(1.0, 1.8)
        e_prev = by_rest.get(k - 1)
        if e_prev is not None and e_prev.kind == "neighbor":
            hold = max(hold, r.t_start - e_prev.seg.t_hit + rng.uniform(0.5, 1.0))  # となりのシャトルの前から待つ
        for side in ("h", "o"):
            plan[side].append(_Waypoint(r.t_start - hold, pos[side], WALK_PACE_MPS, on_time=True))
            plan[side].append(_Waypoint(r.t_start, pos[side], WALK_PACE_MPS))
        # ラリーの中：打つ選手は打点の近くへ、打った後はホームへ戻る
        last_hit = {"h": r.t_start, "o": r.t_start}
        for seg in r.segments[1:]:
            side = seg.side
            home = np.array([_sx(side) * HOME_X_M, 0.0]) + rng.normal(0, 0.2, 2)
            if seg.t_hit - last_hit[side] > 1.2:
                plan[side].append(_Waypoint(last_hit[side] + 0.7, home, PLAY_PACE_MPS))
            feet = seg.p0[:2] + np.clip(rng.normal(0, REACH_M, 2), -1.0, 1.0)
            plan[side].append(_Waypoint(seg.t_hit, feet, PLAY_PACE_MPS))
            last_hit[side] = seg.t_hit
        # ラリーが終わったら、その場で止まる（そこから歩いて、拾いに行く・次のサーブの位置へ行く）
        # ラリーの後：拾って返す
        e = by_rest.get(k)
        if e is not None and e.kind == "pickup":
            picker, other = e.picker_side, "o" if e.picker_side == "h" else "h"
            plan[picker].append(_Waypoint(e.seg.t_hit - rng.uniform(0.3, 0.6), e.seg.p0[:2].copy(), WALK_PACE_MPS))
            plan[picker].append(_Waypoint(e.seg.t_hit, e.seg.p0[:2].copy(), WALK_PACE_MPS))
            if rng.uniform() < 0.3:   # 拾わない方がコートの外へ（タオル・水）
                off = np.array([_sx(other) * rng.uniform(2.0, 5.0),
                                float(rng.choice([-1.0, 1.0])) * (half_w + rng.uniform(0.8, 1.5))])
                plan[other].append(_Waypoint(e.seg.t_land, off, WALK_PACE_MPS))
            else:                     # 受け取りに行く
                catch = np.clip(e.seg.landing_xy, [-6.0, -half_w], [6.0, half_w])
                plan[other].append(_Waypoint(e.seg.t_land, catch, WALK_PACE_MPS))
    return plan


def _follow(t: np.ndarray, start: np.ndarray, plan: list[_Waypoint]) -> np.ndarray:
    """``plan`` の所へ順に動いた足元の位置 (n, 2)。行く時刻に間に合わなければ、すぐ出て遅れて着く。"""
    xy = np.empty((len(t), 2))
    xy[:] = start
    pos, now = start.copy(), -np.inf
    for w in sorted(plan, key=lambda w: w.t):
        dist = float(np.linalg.norm(w.xy - pos))
        if dist < 1e-6:
            continue
        dur = MOVE_BASE_S + dist / w.pace
        if w.on_time and np.isfinite(now) and now + dur > w.t:
            dur = max(w.t - now, 0.5)
        a = max(now, w.t - dur) if np.isfinite(now) else w.t - dur
        b = a + dur
        inside = (t >= a) & (t < b)
        xy[inside] = pos + (w.xy - pos) * _min_jerk((t[inside] - a) / dur)[:, None]
        xy[t >= b] = w.xy
        pos, now = w.xy.copy(), b
    return xy


def session_feet(session: list[SyntheticRally], extras: list[ExtraShot], times: np.ndarray,
                 rng: np.random.Generator, court: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
    """各時刻の 2 人（h・o。シングルス）の足元の正解 [m]。"""
    plan = _plan(session, extras, rng, court)
    return {side: _follow(np.asarray(times, float), plan[side][0].xy.copy(), plan[side]) for side in ("h", "o")}


def observe_player_session(session: list[SyntheticRally], extras: list[ExtraShot], camera: Camera, fps: float,
                           phys: PhysicsParams, rng: np.random.Generator, court: dict[str, Any] | None = None
                           ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """シャトルの検出（ラリーの間の打球も入れた ``observe_session``）と、同じコマの ``players.csv`` の形、足元の正解。"""
    flights = sorted([*session, *(e.as_rally() for e in extras)], key=lambda r: r.t_start)
    det = observe_session(flights, camera, fps, phys, rng)
    t = det["timestamp"].to_numpy(float)
    feet = session_feet(session, extras, t, rng, court)
    players = observe_players(t, feet, camera, rng)
    return det, players, feet
