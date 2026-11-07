"""選手の足元と打球の間隔からの初速の目安（ROADMAP D1c）。

シャトルの検出を使わずに、一覧の初速が「ありえる大きさ」かを確かめるためのもの。
ショットを打った人の足元から、次のショットを打った人の足元まで（床の上の距離 D）を、打球の間隔 dt で飛ぶのに要る初速を、
当てはめと同じ運動方程式（空気抵抗つき。``physics.simulate``）で解く。

- 目安：打点と受ける所の高さが同じ・距離は D そのもの。
- 幅：打点・受ける所の高さ（``heights_m``）と、足元から打点までの手の届く分（D ± ``reach_m``）を動かした最小〜最大。

シャトルは打った直後に大きく減速するので、同じ D・dt でも高さ・距離の少しの違いで初速は大きく変わる。幅は広く（2 倍ほど）、
「幅に入らない」初速は、打点の位置か打った時刻が大きくずれている目印として使う。

  python -m smash_speed.feet_speed <ショットの一覧のフォルダ> [--truth 正解の打球.csv]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from smash_speed.hitter import hitter_at
from smash_speed.physics import MPS_TO_KMH, PhysicsParams, simulate


@dataclass
class FeetSpeedOptions:
    """足元からの目安の条件。"""

    reach_m: float = 1.0                       # 足元から打点・受ける所までの手の届く分 [m]
    heights_m: tuple[float, float] = (0.3, 2.8)  # 打点・受ける所の高さの範囲 [m]
    match_s: float = 0.15                      # 正解の打球と一覧のショットを合わせる時刻の差 [s]
    tol_m: float = 0.02                        # 解けたとみなす位置のずれ [m]


def speed_to_reach(distance_m: float, dz_m: float, dt_s: float, phys: PhysicsParams | None = None,
                   tol_m: float = 0.02) -> float:
    """床の上で ``distance_m`` 先・高さ ``dz_m`` 上の所に ``dt_s`` 秒で着く初速 [km/h]。解けなければ NaN。"""
    phys = phys or PhysicsParams()
    if not (distance_m > 0 and dt_s > 0):
        return float("nan")

    def res(v: np.ndarray) -> list[float]:
        p, _ = simulate(np.zeros(3), np.array([v[0], 0.0, v[1]]), 0.0, np.array([dt_s]), phys)
        return [p[0, 0] - distance_m, p[0, 2] - dz_m]

    # 抵抗のないときの初速から始める
    v_init = [distance_m / dt_s, (dz_m + 0.5 * phys.g * dt_s * dt_s) / dt_s]
    r = least_squares(res, v_init, bounds=([0.0, -200.0], [400.0, 200.0]))
    if not np.all(np.isfinite(r.fun)) or np.max(np.abs(r.fun)) > tol_m:
        return float("nan")
    return float(np.hypot(*r.x) * MPS_TO_KMH)


def speed_range(distance_m: float, dt_s: float, phys: PhysicsParams | None = None,
                o: FeetSpeedOptions | None = None) -> tuple[float, float, float]:
    """(目安, 幅の最小, 幅の最大) [km/h]。"""
    o = o or FeetSpeedOptions()
    nominal = speed_to_reach(distance_m, 0.0, dt_s, phys, o.tol_m)
    found = []
    for d in (max(distance_m - o.reach_m, 0.3), distance_m + o.reach_m):
        for z0 in o.heights_m:
            for z1 in o.heights_m:
                v = speed_to_reach(d, z1 - z0, dt_s, phys, o.tol_m)
                if np.isfinite(v):
                    found.append(v)
    if not found:
        return nominal, float("nan"), float("nan")
    return nominal, float(min(found)), float(max(found))


def compare(shots: pd.DataFrame, players: pd.DataFrame, hits: pd.DataFrame | None = None,
            phys: PhysicsParams | None = None, o: FeetSpeedOptions | None = None) -> pd.DataFrame:
    """続けて打った 2 本ごとに、足元からの目安と一覧の初速を並べる。

    ``hits``（列 ``t_s``・``side``・``rally``。コマで決めた正解の打球）があれば、その時刻・側で組を作り、
    ``match_s`` 以内の一覧のショットの初速と比べる（一覧の打球の誤りに左右されない）。なければ一覧の打球で組を作る。
    """
    o = o or FeetSpeedOptions()
    shots = shots.sort_values("t_hit_s").reset_index(drop=True)
    if hits is None:
        hits = shots.rename(columns={"t_hit_s": "t_s", "rally_id": "rally"})[["t_s", "side", "rally"]]
    hits = hits.sort_values("t_s").reset_index(drop=True)
    rows = []
    for i in range(len(hits) - 1):
        a, b = hits.iloc[i], hits.iloc[i + 1]
        if a["rally"] != b["rally"] or a["side"] == b["side"]:
            continue
        fa, fb = hitter_at(players, float(a["t_s"]), str(a["side"])), hitter_at(players, float(b["t_s"]), str(b["side"]))
        if fa is None or fb is None:
            continue
        near = shots[(shots["t_hit_s"] - a["t_s"]).abs() <= o.match_s]
        if near.empty:
            continue
        shot = near.iloc[int((near["t_hit_s"] - a["t_s"]).abs().argmin())]
        dt = float(b["t_s"] - a["t_s"])
        dist = float(np.hypot(fb.x - fa.x, fb.y - fa.y))
        nominal, lo, hi = speed_range(dist, dt, phys, o)
        v = float(shot["speed_kmh"])
        rows.append({
            "t_s": round(float(a["t_s"]), 3), "side": a["side"], "side_ok": shot["side"] == a["side"],
            "shot_type": shot["shot_type"], "speed_kmh": round(v, 1) if np.isfinite(v) else np.nan,
            "dt_s": round(dt, 3), "distance_m": round(dist, 2), "feet_kmh": round(nominal, 1),
            "feet_low_kmh": round(lo, 1), "feet_high_kmh": round(hi, 1),
            "inside": bool(lo <= v <= hi) if np.isfinite(v) else None,
            "ratio": round(v / nominal, 3) if np.isfinite(v) and nominal > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="選手の足元と打球の間隔からの初速の目安と、一覧の初速を比べる")
    ap.add_argument("shots_dir", type=Path, help="ショットの一覧のフォルダ（shots.csv・players.csv）")
    ap.add_argument("--truth", type=Path, help="正解の打球（t_s,side,rally）")
    ap.add_argument("--out", type=Path, help="結果の CSV")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    shots = pd.read_csv(a.shots_dir / "shots.csv")
    players = pd.read_csv(a.shots_dir / "players.csv")
    hits = pd.read_csv(a.truth) if a.truth else None
    df = compare(shots, players, hits)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(df.to_string(index=False))
    judged = df.dropna(subset=["inside"]) if len(df) else df
    if len(judged):
        print(f"幅に入る {int(judged['inside'].sum())} / {len(judged)} 本（初速の出ないもの {len(df) - len(judged)} 本）、"
              f"目安との比の中央値 {np.nanmedian(df['ratio']):.2f}")
    if a.out:
        df.to_csv(a.out, index=False, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
