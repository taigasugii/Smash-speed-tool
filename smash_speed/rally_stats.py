"""ラリーの統計（ROADMAP C2）：ラリーの長さ、ショットの種類の割合、種類ごとの平均・最高速度、着地点の分布。

入力は ``shots.csv``・``rallies.csv`` の形（``shots.Shot``・``shots.Rally``）。``rally_stats`` が数字を
まとめた辞書（``stats.json`` に書く）を、``landing_svg`` が着地点の分布図（コートを上から見た図を種類ごとに並べたもの、
SVG）を返す。

- ラリーの長さ：本数（1 ラリーのショットの数）と秒（最初に打ってからシャトルが落ちるまで）の平均・中央値・最長と、
  本数の分布（``LENGTH_BINS``）。終わり方（in・out・net）と、得点した側ごとの数。
- ショットの種類：種類ごとの本数・割合・平均と最高の初速（全体と、打った側 h・o ごと）。種類が分からない
  （unknown）ショットは本数には数えるが、割合は分かったものだけで出す。初速が出なかったショットは速さに数えない。
- 着地点：「誰も打たなかったら床に着いた所」（``Shot.land_x_m``・``land_y_m``。打ち返されたショットも、軌道を床まで
  延ばして決める。種類の判定と同じ）。落ちた側の半面を、ネットからの深さ（前・中・後）とコートの幅（3 つ）に
  分けた 9 つの区画ごとに数える。

``python -m smash_speed.rally_stats <ショットの一覧の出力先>`` で、``shots.csv``・``rallies.csv`` から
``stats.json``・``landing.svg`` を書き出す（``shot_list.analyze_video`` も最後に書き出す）。
"""

from __future__ import annotations

import json
import math
from html import escape
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from smash_speed.shots import (RALLIES_FILE, SHOT_TYPE_LABELS, SHOT_TYPES, SHOTS_FILE, Rally, Shot, read_rallies,
                               read_shots, shots_of)

STATS_FILE = "stats.json"
LANDING_FILE = "landing.svg"

# ラリーの本数の分布の区切り（下限, 上限, 名前）
LENGTH_BINS = ((1, 2, "1〜2 本"), (3, 4, "3〜4 本"), (5, 8, "5〜8 本"), (9, 12, "9〜12 本"), (13, 10**9, "13 本以上"))
DEPTH_NAMES = ("前", "中", "後")          # ネットに近い方から
WIDTH_NAMES = ("y−", "中央", "y+")        # コートの y の負の側から（上から見た図の上から下）
END_REASON_LABELS = {"in": "相手のコートに落ちた", "out": "外に出た", "net": "ネットを越えなかった", "": "分からない"}
SIDE_LABELS = {"h": "h の側（x < 0）", "o": "o の側（x > 0）"}


def _summary(values: Iterable[float | None]) -> dict[str, Any]:
    v = [float(x) for x in values if x is not None and math.isfinite(x)]
    if not v:
        return {"n": 0, "mean": None, "median": None, "max": None}
    return {"n": len(v), "mean": round(float(np.mean(v)), 2), "median": round(float(np.median(v)), 2),
            "max": round(float(np.max(v)), 2)}


def _speeds(shots: list[Shot]) -> dict[str, Any]:
    v = [s.speed_kmh for s in shots if s.speed_kmh is not None]
    return {"n_speed": len(v), "mean_kmh": round(float(np.mean(v)), 1) if v else None,
            "max_kmh": round(float(np.max(v)), 1) if v else None}


def type_table(shots: list[Shot]) -> dict[str, Any]:
    """種類ごとの本数・割合（種類が分かったものの中で）・平均と最高の初速。本数のない種類は入れない。"""
    known = sum(s.shot_type != "unknown" for s in shots)
    out: dict[str, Any] = {}
    for k in SHOT_TYPES:
        mine = [s for s in shots if s.shot_type == k]
        if not mine:
            continue
        share = None if k == "unknown" or not known else round(len(mine) / known, 4)
        out[k] = {"label": SHOT_TYPE_LABELS[k], "n": len(mine), "share": share, **_speeds(mine)}
    return out


def zone_of(x_m: float, y_m: float, court: dict[str, Any] | None = None) -> tuple[str, int, int] | None:
    """着地点の区画：（落ちた側 h | o、深さの番号 0〜2、幅の番号 0〜2）。コートの外なら None。"""
    c = court or {}
    half_len, half_w = c.get("half_length_m", 6.70), c.get("half_width_doubles_m", 3.05)
    if x_m == 0 or abs(x_m) > half_len or abs(y_m) > half_w:
        return None
    d = min(int(abs(x_m) / (half_len / 3)), 2)
    w = min(int((y_m + half_w) / (2 * half_w / 3)), 2)
    return ("h" if x_m < 0 else "o"), d, w


def landing_points(shots: list[Shot]) -> list[dict[str, Any]]:
    return [{"t": round(s.t_hit_s, 3), "type": s.shot_type, "side": s.side, "x": round(s.land_x_m, 3),
             "y": round(s.land_y_m, 3), "rally_id": s.rally_id}
            for s in shots if s.land_x_m is not None and s.land_y_m is not None]


def landing_zones(shots: list[Shot], court: dict[str, Any] | None = None) -> dict[str, Any]:
    """着地点の区画ごとの数。``by_type[種類][落ちた側]`` が 3×3（深さ × 幅）の数の表。コートの外は ``out`` に数える。"""
    by_type: dict[str, dict[str, list[list[int]]]] = {}
    out_n: dict[str, int] = {}
    for p in landing_points(shots):
        z = zone_of(p["x"], p["y"], court)
        if z is None:
            out_n[p["type"]] = out_n.get(p["type"], 0) + 1
            continue
        side, d, w = z
        grid = by_type.setdefault(p["type"], {"h": [[0] * 3 for _ in range(3)], "o": [[0] * 3 for _ in range(3)]})
        grid[side][d][w] += 1
    return {"depth_names": list(DEPTH_NAMES), "width_names": list(WIDTH_NAMES), "by_type": by_type, "out": out_n}


def rally_stats(shots: list[Shot], rallies: list[Rally], court: dict[str, Any] | None = None) -> dict[str, Any]:
    """1 本の動画（またはその一部）のショットとラリーの統計。"""
    n_by = [len(shots_of(r, shots)) or r.n_shots for r in rallies]
    dur = [None if r.t_end_s is None else r.t_end_s - r.t_start_s for r in rallies]
    hist = [{"label": name, "lo": lo, "hi": hi, "n": sum(lo <= n <= hi for n in n_by)} for lo, hi, name in LENGTH_BINS]
    longest = max(rallies, key=lambda r: (len(shots_of(r, shots)) or r.n_shots, r.t_start_s), default=None)
    ends: dict[str, int] = {}
    wins: dict[str, int] = {}
    for r in rallies:
        ends[r.end_reason] = ends.get(r.end_reason, 0) + 1
        if r.winner_side:
            wins[r.winner_side] = wins.get(r.winner_side, 0) + 1
    by_side = {side: type_table([s for s in shots if s.side == side]) for side in ("h", "o")}
    return {
        "n_shots": len(shots), "n_rallies": len(rallies),
        "n_shots_with_speed": sum(s.speed_kmh is not None for s in shots),
        "rally_shots": _summary(n_by), "rally_seconds": _summary(dur), "rally_length_hist": hist,
        "longest_rally": None if longest is None else {
            "rally_id": longest.rally_id, "n_shots": len(shots_of(longest, shots)) or longest.n_shots,
            "t_start_s": round(longest.t_start_s, 2),
            "seconds": None if longest.t_end_s is None else round(longest.t_end_s - longest.t_start_s, 2)},
        "end_reasons": {k: {"label": END_REASON_LABELS.get(k, k), "n": v} for k, v in ends.items()},
        "points_won": wins,
        "types": type_table(shots), "types_by_side": by_side,
        "landing": {"points": landing_points(shots), **landing_zones(shots, court)},
    }


# =============================================================================
# 着地点の分布図（SVG）
# =============================================================================
# 色は役割ごとの CSS の変数で持ち、明るい画面・暗い画面で切り替える（レポートの HTML でも同じ名前を使う）
SVG_STYLE = """
.lm { --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781; --line: #c3c2b7;
      --dot: #104281; --z1: #e6f0fd; --z2: #cde2fb; --z3: #9ec5f4; --z4: #6da7ec;
      font-family: system-ui, -apple-system, "Segoe UI", "Yu Gothic UI", "Hiragino Sans", sans-serif; }
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .lm { --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7;
      --line: #383835; --dot: #cde2fb; --z1: #0d2a50; --z2: #0d366b; --z3: #104281; --z4: #1c5cab; }
}
:root[data-theme="dark"] .lm { --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7;
      --line: #383835; --dot: #cde2fb; --z1: #0d2a50; --z2: #0d366b; --z3: #104281; --z4: #1c5cab; }
.lm .bg { fill: var(--surface); }
.lm .court { fill: none; stroke: var(--line); stroke-width: 1; }
.lm .net { stroke: var(--muted); stroke-width: 2; }
.lm .z0 { fill: none; }
.lm .z1 { fill: var(--z1); } .lm .z2 { fill: var(--z2); } .lm .z3 { fill: var(--z3); } .lm .z4 { fill: var(--z4); }
.lm .zn { fill: var(--ink2); font-size: 9px; font-variant-numeric: tabular-nums; }
.lm .dot { fill: var(--dot); stroke: var(--surface); stroke-width: 1.5; }
.lm .title { fill: var(--ink); font-size: 13px; font-weight: 600; }
"""


def _court_lines(court: dict[str, Any]) -> list[tuple[float, float, float, float]]:
    L, W, Ws = court["half_length_m"], court["half_width_doubles_m"], court["half_width_singles_m"]
    ss, ls = court["short_service_m"], court["long_service_doubles_m"]
    lines = [(-L, -W, L, -W), (-L, W, L, W), (-L, -Ws, L, -Ws), (-L, Ws, L, Ws)]
    for sx in (-1, 1):
        lines += [(sx * L, -W, sx * L, W), (sx * ls, -W, sx * ls, W), (sx * ss, -W, sx * ss, W),
                  (sx * ss, 0, sx * L, 0)]
    return lines


def _panel(shots: list[Shot], title: str, court: dict[str, Any], scale: float, ox: float, oy: float) -> list[str]:
    """1 つの種類のコート（上から見た図。左が h の側、右が o の側）。"""
    L, W = court["half_length_m"], court["half_width_doubles_m"]
    X = lambda x: ox + (x + L) * scale  # noqa: E731
    Y = lambda y: oy + 22 + (y + W) * scale  # noqa: E731
    out = [f'<text class="title" x="{ox:.1f}" y="{oy + 13:.1f}">{escape(title)}</text>']
    zones = landing_zones(shots, court)["by_type"]
    grid = {"h": np.zeros((3, 3), int), "o": np.zeros((3, 3), int)}
    for g in zones.values():
        for side in grid:
            grid[side] += np.array(g[side])
    top = max(1, int(max(grid["h"].max(), grid["o"].max())))
    dl, wl = L / 3, 2 * W / 3
    for side, sx in (("h", -1), ("o", 1)):
        for d in range(3):
            for w in range(3):
                n = int(grid[side][d][w])
                x0, x1 = sorted((sx * d * dl, sx * (d + 1) * dl))
                y0 = -W + w * wl
                level = 0 if n == 0 else max(1, math.ceil(4 * n / top))  # 区画の色の濃さ 1〜4
                out.append(f'<rect class="z{level}" x="{X(x0):.1f}" y="{Y(y0):.1f}" width="{(x1 - x0) * scale:.1f}" '
                           f'height="{wl * scale:.1f}"><title>{"h" if sx < 0 else "o"} の側・'
                           f'{DEPTH_NAMES[d]}・{WIDTH_NAMES[w]}：{n} 本</title></rect>')
                if n:  # 数は区画の左上の隅に小さく（点と重ならないように）
                    out.append(f'<text class="zn" x="{X(x0) + 3:.1f}" y="{Y(y0) + 10:.1f}">{n}</text>')
    for x0, y0, x1, y1 in _court_lines(court):
        out.append(f'<line class="court" x1="{X(x0):.1f}" y1="{Y(y0):.1f}" x2="{X(x1):.1f}" y2="{Y(y1):.1f}"/>')
    out.append(f'<line class="net" x1="{X(0):.1f}" y1="{Y(-W) - 3:.1f}" x2="{X(0):.1f}" y2="{Y(W) + 3:.1f}"/>')
    for p in landing_points(shots):
        if abs(p["x"]) > L + 1.0 or abs(p["y"]) > W + 1.0:
            continue  # 図の外（当てはめが外れたもの）は描かない（区画の数の「外」には数える）
        kmh = next((s.speed_kmh for s in shots if s.t_hit_s == p["t"]), None)
        tip = f'{p["t"]:.2f} 秒' + (f"・{kmh:.0f} km/h" if kmh else "")
        out.append(f'<circle class="dot" cx="{X(p["x"]):.1f}" cy="{Y(p["y"]):.1f}" r="3.5"><title>{tip}</title>'
                   f'</circle>')
    return out


def landing_types(shots: list[Shot]) -> list[str]:
    """着地点のあるショットの種類を、多い順（同じなら ``SHOT_TYPES`` の順）に。"""
    pts = [s for s in shots if s.land_x_m is not None and s.land_y_m is not None]
    return sorted({s.shot_type for s in pts}, key=lambda k: (-sum(s.shot_type == k for s in pts), SHOT_TYPES.index(k)))


def landing_svg(shots: list[Shot], court: dict[str, Any] | None = None, columns: int = 2,
                panel_width: float = 300.0, types: Iterable[str] | None = None) -> str:
    """着地点の分布図：種類ごとに 1 面のコート（上から見た図）を並べた SVG。

    区画の色の濃さは、その種類の中で着地点が多い区画ほど濃い（数も書く）。点は 1 本ずつの着地点。
    着地点のあるショットの多い種類から並べる。着地点が 1 つもなければ空文字。
    """
    from smash_speed.config import load_config

    court = court or load_config()["court"]
    pts = [s for s in shots if s.land_x_m is not None and s.land_y_m is not None]
    names = [k for k in (landing_types(shots) if types is None else types) if any(s.shot_type == k for s in pts)]
    if not names:
        return ""
    scale = panel_width / (2 * court["half_length_m"])
    panel_h = 2 * court["half_width_doubles_m"] * scale + 22
    gap = 18.0
    cols = max(1, min(columns, len(names)))
    rows = math.ceil(len(names) / cols)
    w = cols * panel_width + (cols - 1) * gap + 8
    h = rows * (panel_h + gap) + 4
    body = [f'<rect class="bg" x="0" y="0" width="{w:.0f}" height="{h:.0f}"/>']
    for i, k in enumerate(names):
        mine = [s for s in pts if s.shot_type == k]
        ox, oy = 4 + (i % cols) * (panel_width + gap), 2 + (i // cols) * (panel_h + gap)
        body += _panel(mine, f"{SHOT_TYPE_LABELS[k]}（{len(mine)} 本）", court, scale, ox, oy)
    return (f'<svg class="lm" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}" '
            f'width="100%" role="img" aria-label="ショットの種類ごとの着地点の分布">'
            f'<style>{SVG_STYLE}</style>' + "".join(body) + "</svg>")


# =============================================================================
# 書き出し
# =============================================================================
def write_stats(out_dir: str | Path, shots: list[Shot] | None = None, rallies: list[Rally] | None = None,
                court: dict[str, Any] | None = None) -> dict[str, Any]:
    """``<出力先>/stats.json``・``landing.svg`` を書く（ショット・ラリーを省略すると CSV から読む）。統計を返す。"""
    out_dir = Path(out_dir)
    shots = read_shots(out_dir / SHOTS_FILE) if shots is None else shots
    rallies = read_rallies(out_dir / RALLIES_FILE) if rallies is None and (out_dir / RALLIES_FILE).exists() \
        else (rallies or [])
    st = rally_stats(shots, rallies, court)
    (out_dir / STATS_FILE).write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    svg = landing_svg(shots, court)
    if svg:
        (out_dir / LANDING_FILE).write_text(svg, encoding="utf-8")
    return st


def format_stats(st: dict[str, Any]) -> str:
    """統計の要約（数行の日本語）。"""
    rs, sec = st["rally_shots"], st["rally_seconds"]
    lines = [f"ラリー {st['n_rallies']} 個・ショット {st['n_shots']} 本（初速が出たもの {st['n_shots_with_speed']} 本）"]
    if rs["n"]:
        lines.append(f"ラリーの長さ：平均 {rs['mean']:.1f} 本（最長 {rs['max']:.0f} 本）"
                     + (f"、平均 {sec['mean']:.1f} 秒（最長 {sec['max']:.1f} 秒）" if sec["n"] else ""))
    parts = []
    for v in st["types"].values():
        p = f"{v['label']} {v['n']} 本" + (f"（{100 * v['share']:.0f}%）" if v["share"] is not None else "")
        if v["max_kmh"] is not None:
            p += f" 平均 {v['mean_kmh']:.0f}・最高 {v['max_kmh']:.0f} km/h"
        parts.append(p)
    if parts:
        lines.append("種類：" + "、".join(parts))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    import argparse

    from smash_speed.config import load_config

    ap = argparse.ArgumentParser(description="ショットの一覧の結果から、ラリーの統計と着地点の分布図を書き出す")
    ap.add_argument("out_dir", type=Path, help="ショットの一覧の出力先（shots.csv がある所）")
    ap.add_argument("--config", type=Path, default=None)
    a = ap.parse_args(argv)
    st = write_stats(a.out_dir, court=load_config(a.config)["court"])
    print(format_stats(st))
    print(f"{a.out_dir / STATS_FILE} に書き出しました")


if __name__ == "__main__":
    import sys

    if (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    main()
