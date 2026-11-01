"""打つ人の足元の自動推定（ROADMAP D3）。

スマッシュの解析では、打つ人の足元（コートの座標）と着地点から飛行の面を決め、打点の位置の手がかりにも使う
（``pipeline.AnalyzeOptions.hitter_xy``・``events.impact_from_hitter_position``）。今まではこれを手で入れるか、
候補探しで「最初に写ったシャトルを打点の高さにあるとみなす」大まかな推定（``batch``）で入れていた。

ここでは選手の追跡（``players.csv``。跳んでいるコマは ``players.ground_jumps`` で床の上の位置に直したもの）から、
打った時刻の前後 ``window_s`` の足元の中央値を、打った側の選手について出す。

- 側：分かっていればその側（h：x < 0、o：x > 0）の選手。分からなければ打点に近い選手。
- ダブルス（1 側に 2 人）：打点が分かれば打点に近い方。分からなければ、決められない（``ambiguous``）として近い方を返さない。
- 中央値から ``spread_m`` より離れたコマ（足が隠れて枠の下の辺が上がった・隣の人と取り違えたなど）は外して出し直す。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from smash_speed.shots import Shot, side_of


@dataclass
class HitterOptions:
    """打つ人の足元の推定の条件。"""

    window_s: float = 0.2       # 打った時刻の前後のこの幅のコマの足元を使う [s]
    min_frames: int = 3         # 幅の中にこれより少ないコマしかなければ出さない
    spread_m: float = 0.8       # 中央値からこれより離れたコマは外す [m]
    # ダブルスで、打点に近い方の選手が遠い方よりこれだけ近くないと決めない [m]
    doubles_margin_m: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HitterEstimate:
    """打つ人の足元の推定。"""

    x: float                    # 床の上の足元 [m]
    y: float
    side: str                   # h | o
    player: int
    n_frames: int               # 使ったコマの数
    airborne: bool              # 幅の中に跳んでいるコマがあった（足元は前後から補った位置）
    spread_m: float             # 使ったコマの足元の、中央値からの距離の中央値 [m]
    ambiguous: bool = False     # ダブルスでどちらが打ったか決められなかった

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("x", "y", "spread_m"):
            d[k] = round(float(d[k]), 3)
        return d


def _median_foot(g: pd.DataFrame, o: HitterOptions) -> tuple[np.ndarray, np.ndarray] | None:
    """足元の中央値と、使ったコマの印。外れを外して 1 回出し直す。"""
    xy = g[["x", "y"]].to_numpy(float)
    ok = np.isfinite(xy).all(axis=1)
    if ok.sum() < o.min_frames:
        return None
    med = np.median(xy[ok], axis=0)
    use = ok & (np.hypot(*(xy - med).T) <= o.spread_m)
    if use.sum() < o.min_frames:
        return None
    return np.median(xy[use], axis=0), use


def hitter_at(players: pd.DataFrame, t_hit: float, side: str = "", hit_xy: Any = None,
              o: HitterOptions | None = None) -> HitterEstimate | None:
    """打った時刻 ``t_hit`` の、打った人の床の上の足元。選手の位置が足りなければ None。

    ``side`` は打った側（分からなければ ""）、``hit_xy`` は打点のコートの位置（分かれば。選手を選ぶのに使う）。
    """
    o = o or HitterOptions()
    if players is None or players.empty:
        return None
    hit = None if hit_xy is None else np.asarray(hit_xy, float)[:2]
    if hit is not None and not np.isfinite(hit).all():
        hit = None
    if not side and hit is not None:
        side = side_of(float(hit[0]))
    near = players[(players["t"] - t_hit).abs() <= o.window_s]
    if side:
        near = near[near["side"] == side]
    found = []
    for (s, p), g in near.groupby(["side", "player"]):
        r = _median_foot(g, o)
        if r is not None:
            found.append((str(s), int(p), r[0], g[r[1]]))
    if not found:
        return None
    ambiguous = False
    if hit is not None:
        found.sort(key=lambda f: float(np.hypot(*(f[2] - hit))))
        if len(found) > 1 and found[0][0] == found[1][0]:
            d0, d1 = (float(np.hypot(*(f[2] - hit))) for f in found[:2])
            ambiguous = d1 - d0 < o.doubles_margin_m
    elif len(found) > 1:
        # 打点が分からないと、同じ側の 2 人・両側の選手のどちらが打ったかは決められない
        return None
    s, p, xy, used = found[0]
    spread = float(np.median(np.hypot(*(used[["x", "y"]].to_numpy(float) - xy).T)))
    airborne = bool(used["airborne"].any()) if "airborne" in used else False
    return HitterEstimate(float(xy[0]), float(xy[1]), s, p, int(len(used)), airborne, spread, ambiguous)


def attach_hitters(shots: list[Shot], players: pd.DataFrame, o: HitterOptions | None = None) -> int:
    """ショットごとに、打つ人の足元（``hitter_x_m``・``hitter_y_m``）を入れる。入れた本数を返す。

    決められないショット（選手の位置がない・ダブルスでどちらか分からない）は空欄にする。
    """
    n = 0
    for s in shots:
        hit = (s.hit_x_m, s.hit_y_m) if s.hit_x_m is not None and s.hit_y_m is not None else None
        e = hitter_at(players, s.t_hit_s, s.side, hit, o)
        if e is None or e.ambiguous:
            s.hitter_x_m = s.hitter_y_m = None
            continue
        s.hitter_x_m, s.hitter_y_m = round(e.x, 3), round(e.y, 3)
        n += 1
    return n


# =============================================================================
# 合成の足の動きでの精度
# =============================================================================
def synthetic_benchmark(n_seeds: int = 4, fps: float = 30.0, seed: int = 0, duration_s: float = 20.0,
                        o: HitterOptions | None = None) -> dict[str, Any]:
    """合成の足の動き（``synth_players``。止まるたびに 1 回打ち、そのとき枠が横へ広がる）を 3 つのカメラで写し、
    打った時刻の足元の推定と、本当の足元との距離 [m] を手前・奥ごとに測る（``movement.synthetic_benchmark`` と同じカメラ）。"""
    from smash_speed.camera import look_at
    from smash_speed.synth_players import footwork_path, observe_players

    target = np.array([0.0, 0.0, 0.5])
    cams = {"back_high": look_at(np.array([-9.0, -5.5, 3.0]), target, 1500.0, 1920, 1080),
            "side_low": look_at(np.array([0.0, -8.0, 1.3]), target, 1100.0, 1920, 1080),
            "behind": look_at(np.array([-10.0, 0.0, 2.0]), target, 1600.0, 1920, 1080)}
    err: dict[str, list[float]] = {"h": [], "o": []}
    missed = {"h": 0, "o": 0}
    for cam in cams.values():
        for k in range(n_seeds):
            rng = np.random.default_rng(seed + k)
            t, feet_h = footwork_path(duration_s, fps, "h", rng)
            _, feet_o = footwork_path(duration_s, fps, "o", rng)
            feet = {"h": feet_h, "o": feet_o}
            players = observe_players(t, feet, cam, rng)
            for side, xy in feet.items():
                # 止まったコマの始まり（``observe_players`` が打ったとみなす所）
                speed = np.r_[0.0, np.hypot(*np.diff(xy, axis=0).T)] * fps
                still = speed < 0.05
                for i in np.flatnonzero(still & ~np.r_[False, still[:-1]]):
                    if not 0.5 < t[i] < t[-1] - 0.5:
                        continue
                    e = hitter_at(players, float(t[i]), side, o=o)
                    if e is None:
                        missed[side] += 1
                        continue
                    err[side].append(float(np.hypot(e.x - xy[i, 0], e.y - xy[i, 1])))
    res: dict[str, Any] = {}
    for side, v in err.items():
        a = np.array(v)
        res[side] = {"median_m": round(float(np.median(a)), 3), "p90_m": round(float(np.percentile(a, 90)), 3),
                     "n": len(a), "missed": missed[side]}
    return res
