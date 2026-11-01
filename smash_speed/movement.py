"""選手の動き（ROADMAP D2）：コートの上の足元の位置から、移動距離・移動の速さ・いた場所の分布（ヒートマップ）を出す。

入力は ``players.track_players``（と ``ground_jumps``）の ``players.csv``。足元の位置はコマごとに揺れる
（枠の下の辺は、足を開く・腕を伸ばすだけで数 px 動き、奥の選手では数 px で 1 m ほど動く）ので、
そのまま点をつなぐと移動距離が 2〜4 倍に出る。そこで選手ごとに、

1. 外れを落とす：足が隠れて枠の下が上がったコマなどは、1 秒の中央値から 1.5 m より離れるので、前後から補う。
   位置をならす：短い中央値（一瞬の外れを落とす）の後、ガウスの重みでならす。移動距離といた場所には
   ``smooth_s``（0.2 秒）、最高の速さには ``peak_smooth_s``（0.12 秒。強くならすと速い踏み込みの山が削れる）。
   合成の足の動き（``synth_players``。実動画で測った揺れと、打つときに枠が横へ広がる分を入れたもの）で、
   移動距離 ±4%・最高の速さ ±11% ほど（30 fps）。
2. ``max_gap_s`` より長く位置が出ていない所で区間を分け、区間をまたいでは数えない。
3. 移動距離はならした点の間の距離の合計。いた時間はコマの間隔で数え、自分の半面の 9 つの区画（ネットからの深さ
   前・中・後 × 幅。``rally_stats.zone_of`` と同じ分け方）ごとの割合と、``cell_m`` 四方のマスごとの秒数（ヒートマップ）にする。
4. ラリー（``rallies.csv``）があれば、ラリーの間（始まりの 0.5 秒前〜終わり）の移動距離をラリーごとに出す。

``write_movement`` が ``movement.json``（数字）・``movement.csv``（ならした位置と速さ、1 コマ 1 人）・
``movement.svg``（コートを上から見たいた場所の分布）を書く。``players.analyze_players`` も最後に書き出す。
``python -m smash_speed.movement <players.csv のある出力先>`` で書き直せる。
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, median_filter

from smash_speed.rally_stats import DEPTH_NAMES, SVG_STYLE, WIDTH_NAMES, _court_lines, zone_of
from smash_speed.shots import RALLIES_FILE, Rally, read_rallies

MOVEMENT_FILE = "movement.json"
MOVEMENT_CSV = "movement.csv"
MOVEMENT_SVG = "movement.svg"
SIDE_NAMES = {"h": "h の側", "o": "o の側"}  # 手前・奥はカメラの位置で変わるので書かない（rally_report と同じ）


@dataclass
class MovementOptions:
    """移動距離・速さ・ヒートマップの条件。"""

    median_s: float = 0.17       # 一瞬の外れを落とす中央値の幅 [s]
    # 足が隠れて枠の下の辺が上がると、奥の選手は 0.5 秒ほどの間 2〜4 m 遠くへ飛ぶ（IMG_4397 97.2 s）。
    # ``outlier_window_s`` の中央値から ``outlier_m`` より離れたコマは、前後の位置から直線で補う
    outlier_window_s: float = 1.0
    outlier_m: float = 1.5
    smooth_s: float = 0.2        # 移動距離・いた場所のならし（ガウスの標準偏差）[s]
    peak_smooth_s: float = 0.12  # 最高の速さのならし [s]
    max_gap_s: float = 0.6       # これより長く位置が出ていない所で区間を分ける [s]
    moving_mps: float = 1.0      # これより速いコマを「動いている」とみなす [m/s]
    fast_mps: float = 3.0        # これより速いコマを「速く動いている」とみなす [m/s]
    cell_m: float = 0.5          # ヒートマップのマスの大きさ [m]
    margin_m: float = 1.5        # ヒートマップに入れる、コートの外側の線からの広げ幅 [m]
    rally_lead_s: float = 0.5    # ラリーの移動距離に含める、最初に打つ前の時間 [s]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# =============================================================================
# ならした位置と速さ
# =============================================================================
def _segments(t: np.ndarray, max_gap_s: float) -> np.ndarray:
    """時刻の並びを、``max_gap_s`` より長い抜けで区切った区間の番号。"""
    return np.r_[0, np.cumsum(np.diff(t) > max_gap_s)] if len(t) else np.zeros(0, int)


def _smooth(v: np.ndarray, fps: float, median_s: float, sigma_s: float) -> np.ndarray:
    m = max(1, int(round(median_s * fps)) | 1)   # 奇数
    out = median_filter(v, m, mode="nearest") if m > 1 and len(v) >= m else v.astype(float)
    return gaussian_filter1d(out, sigma_s * fps, mode="nearest") if sigma_s > 0 and len(v) > 1 else out


def _drop_outliers(t: np.ndarray, v: np.ndarray, fps: float, o: MovementOptions) -> np.ndarray:
    """``outlier_window_s`` の中央値から ``outlier_m`` より離れたコマ（位置 (n, 2)）の印。"""
    m = max(1, int(round(o.outlier_window_s * fps)) | 1)
    if len(t) < m:
        return np.zeros(len(t), bool)
    med = np.column_stack([median_filter(v[:, 0], m, mode="nearest"), median_filter(v[:, 1], m, mode="nearest")])
    return np.hypot(*(v - med).T) > o.outlier_m


def _speed(t: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(t) < 2:
        return np.zeros(len(t))
    return np.hypot(np.gradient(x, t), np.gradient(y, t))


def smooth_track(track: pd.DataFrame, o: MovementOptions | None = None) -> pd.DataFrame:
    """1 人の位置（``t, x, y``）をならす。``segment, x, y``（ならした位置）、``speed``（その速さ [m/s]）、
    ``peak_speed``（弱くならした位置の速さ）、``dt``（そのコマが代表する時間 [s]）、``outlier``（外れとして
    補ったコマ）を付けて返す。"""
    o = o or MovementOptions()
    g = track.dropna(subset=["x", "y"]).sort_values("t").reset_index(drop=True)
    t = g["t"].to_numpy(float)
    out = g[[c for c in ("frame", "t", "side", "player") if c in g]].copy()
    out["raw_x"], out["raw_y"] = g["x"].to_numpy(float), g["y"].to_numpy(float)
    out["segment"] = _segments(t, o.max_gap_s)
    for c in ("x", "y", "speed", "peak_speed", "dt"):
        out[c] = np.nan
    out["outlier"] = False
    if g.empty:
        return out
    step = np.diff(t)
    frame_dt = float(np.median(step)) if len(step) else 0.0
    fps = 1.0 / frame_dt if frame_dt > 0 else 30.0
    for _, idx in out.groupby("segment").groups.items():
        k = np.asarray(idx)
        ts, xs, ys = t[k], out.loc[k, "raw_x"].to_numpy(), out.loc[k, "raw_y"].to_numpy()
        bad = _drop_outliers(ts, np.column_stack([xs, ys]), fps, o)
        if bad.any() and (~bad).sum() >= 2:
            xs, ys = np.interp(ts, ts[~bad], xs[~bad]), np.interp(ts, ts[~bad], ys[~bad])
            out.loc[k, "outlier"] = bad
        sx, sy = _smooth(xs, fps, o.median_s, o.smooth_s), _smooth(ys, fps, o.median_s, o.smooth_s)
        px, py = _smooth(xs, fps, o.median_s, o.peak_smooth_s), _smooth(ys, fps, o.median_s, o.peak_smooth_s)
        out.loc[k, "x"], out.loc[k, "y"] = sx, sy
        out.loc[k, "speed"], out.loc[k, "peak_speed"] = _speed(ts, sx, sy), _speed(ts, px, py)
        # いた時間：前後のコマとの間隔の半分ずつ（区間の端は 1 コマ分）
        d = np.diff(ts)
        out.loc[k, "dt"] = (np.r_[frame_dt, d] + np.r_[d, frame_dt]) / 2 if len(ts) > 1 else frame_dt
    return out


def movement_tracks(players: pd.DataFrame, o: MovementOptions | None = None) -> pd.DataFrame:
    """``players.csv`` の全員をならす（1 コマ 1 人。``smooth_track`` の列）。"""
    o = o or MovementOptions()
    parts = [smooth_track(g, o) for _, g in players.groupby(["side", "player"])] if not players.empty else []
    if not parts:
        return pd.DataFrame(columns=["frame", "t", "side", "player", "raw_x", "raw_y", "segment", "x", "y",
                                     "speed", "peak_speed", "dt", "outlier"])
    return pd.concat(parts, ignore_index=True).sort_values(["t", "side", "player"]).reset_index(drop=True)


def path_distance(track: pd.DataFrame, t0: float | None = None, t1: float | None = None) -> float:
    """ならした位置の移動距離 [m]（区間をまたがない）。``t0, t1`` で時間を切れる。"""
    g = track
    if t0 is not None:
        g = g[g["t"] >= t0]
    if t1 is not None:
        g = g[g["t"] <= t1]
    total = 0.0
    for _, s in g.groupby("segment"):
        if len(s) > 1:
            total += float(np.hypot(np.diff(s["x"].to_numpy(float)), np.diff(s["y"].to_numpy(float))).sum())
    return total


# =============================================================================
# まとめ
# =============================================================================
def zone_shares(track: pd.DataFrame, side: str, court: dict[str, Any] | None = None) -> dict[str, Any]:
    """自分の半面の 9 つの区画（深さ × 幅）にいた時間の割合と、相手の半面・コートの外にいた割合。"""
    grid = np.zeros((3, 3))
    other = outside = 0.0
    total = float(track["dt"].sum())
    for x, y, dt in track[["x", "y", "dt"]].itertuples(index=False):
        z = zone_of(x, y, court)
        if z is None:
            outside += dt
        elif z[0] != side:
            other += dt
        else:
            grid[z[1], z[2]] += dt
    n = max(total, 1e-9)
    return {"grid": np.round(grid / n, 4).tolist(), "other_half": round(other / n, 4), "outside": round(outside / n, 4)}


def heat_grid(track: pd.DataFrame, court: dict[str, Any] | None = None, o: MovementOptions | None = None
              ) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """マスごとにいた秒数（行 = y、列 = x）と、範囲（x の始め・終わり、y の始め・終わり [m]）。"""
    o = o or MovementOptions()
    c = court or {}
    xl = c.get("half_length_m", 6.70) + o.margin_m
    yl = c.get("half_width_doubles_m", 3.05) + o.margin_m
    nx, ny = math.ceil(2 * xl / o.cell_m), math.ceil(2 * yl / o.cell_m)
    xl, yl = nx * o.cell_m / 2, ny * o.cell_m / 2   # マスの大きさにそろえる
    grid, _, _ = np.histogram2d(track["y"].to_numpy(float), track["x"].to_numpy(float), bins=(ny, nx),
                                range=((-yl, yl), (-xl, xl)), weights=track["dt"].to_numpy(float))
    return grid, (-xl, xl, -yl, yl)


def heat_brief(track: pd.DataFrame, court: dict[str, Any] | None = None, o: MovementOptions | None = None
               ) -> dict[str, Any]:
    """``heat_grid`` を JSON に入れる形にしたもの（マスの大きさ・範囲 [m]・マスごとの秒数。行 = y、列 = x）。"""
    o = o or MovementOptions()
    grid, extent = heat_grid(track, court, o)
    return {"cell_m": o.cell_m, "extent_m": [round(v, 3) for v in extent], "seconds": np.round(grid, 2).tolist()}


def _r(v: float | None, nd: int = 2) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v), nd)


def player_movement(track: pd.DataFrame, side: str, court: dict[str, Any] | None = None,
                    rallies: list[Rally] | None = None, o: MovementOptions | None = None) -> dict[str, Any]:
    """1 人の移動距離・速さ・いた場所のまとめ。"""
    o = o or MovementOptions()
    seconds = float(track["dt"].sum())
    dist = path_distance(track)
    w = track["dt"].to_numpy(float)
    sp = track["speed"].to_numpy(float)
    peak = track["peak_speed"].to_numpy(float)
    k = int(np.nanargmax(peak)) if len(peak) and np.isfinite(peak).any() else None
    res: dict[str, Any] = {
        "seconds": round(seconds, 2), "distance_m": round(dist, 1),
        "distance_per_min_m": _r(60 * dist / seconds if seconds > 0 else None, 1),
        "mean_speed_mps": _r(dist / seconds if seconds > 0 else None),
        "max_speed_mps": None if k is None else _r(peak[k]),
        "max_speed_t": None if k is None else _r(float(track["t"].iloc[k]), 2),
        "moving_share": _r(float(w[sp > o.moving_mps].sum()) / seconds if seconds > 0 else None, 3),
        "fast_share": _r(float(w[sp > o.fast_mps].sum()) / seconds if seconds > 0 else None, 3),
        "outlier_share": _r(float(track["outlier"].mean()) if len(track) else None, 4),
        "mean_x": _r(float(np.average(track["x"], weights=w)) if seconds > 0 else None),
        "mean_y": _r(float(np.average(track["y"], weights=w)) if seconds > 0 else None),
        "zones": zone_shares(track, side, court),
        "heatmap": heat_brief(track, court, o),
    }
    if rallies:
        per = []
        for r in sorted(rallies, key=lambda r: r.t_start_s):
            if r.t_end_s is None:
                continue
            t0, t1 = r.t_start_s - o.rally_lead_s, r.t_end_s
            mine = track[(track["t"] >= t0) & (track["t"] <= t1)]
            cover = float(mine["dt"].sum()) / max(t1 - t0, 1e-9)
            per.append({"rally_id": r.rally_id, "seconds": round(t1 - t0, 2), "distance_m": round(path_distance(mine), 1),
                        "coverage": round(min(cover, 1.0), 3)})
        used = [p for p in per if p["coverage"] >= 0.8]   # 位置が 8 割以上出たラリーだけで平均する
        res["rallies"] = per
        res["rally_mean_distance_m"] = _r(float(np.mean([p["distance_m"] for p in used])) if used else None, 1)
    return res


def movement_summary(players: pd.DataFrame, court: dict[str, Any] | None = None, rallies: list[Rally] | None = None,
                     o: MovementOptions | None = None) -> tuple[dict[str, Any], pd.DataFrame]:
    """全員のまとめ（``movement.json`` の中身）と、ならした位置（``movement.csv`` の中身）。"""
    o = o or MovementOptions()
    tracks = movement_tracks(players, o)
    by: dict[str, Any] = {}
    for (side, player), g in tracks.groupby(["side", "player"]):
        by[f"{side}{int(player)}"] = {"side": side, "player": int(player),
                                     **player_movement(g, side, court, rallies, o)}
    return {"players": by, "depth_names": list(DEPTH_NAMES), "width_names": list(WIDTH_NAMES),
            "options": o.to_dict()}, tracks


# =============================================================================
# いた場所の分布図（SVG）
# =============================================================================
HEAT_LEVELS = 4


def movement_svg(tracks: pd.DataFrame, court: dict[str, Any] | None = None, o: MovementOptions | None = None,
                 width: float = 560.0) -> str:
    """コートを上から見た図（左が h の側、右が o の側）に、選手がいたマスを色の濃さで描いた SVG。

    色の濃さは、その選手がいちばん長くいたマスに対する割合（側ごと）。十字は平均の位置。位置がなければ空文字。
    """
    from smash_speed.config import load_config

    o = o or MovementOptions()
    court = court or load_config()["court"]
    if tracks.empty or not tracks["dt"].notna().any():
        return ""
    _, (xa, xb, ya, yb) = heat_grid(tracks.iloc[:0], court, o)
    scale = width / (xb - xa)
    h = (yb - ya) * scale + 26
    X = lambda x: 4 + (x - xa) * scale  # noqa: E731
    Y = lambda y: 24 + (y - ya) * scale  # noqa: E731
    body = [f'<rect class="bg" x="0" y="0" width="{width + 8:.0f}" height="{h + 4:.0f}"/>',
            '<text class="title" x="4" y="15">選手がいた場所（左が h の側、右が o の側）</text>']
    for (side, player), g in tracks.groupby(["side", "player"]):
        grid, _ = heat_grid(g, court, o)
        top = grid.max()
        if top <= 0:
            continue
        for iy, ix in zip(*np.nonzero(grid)):
            level = max(1, math.ceil(HEAT_LEVELS * grid[iy, ix] / top))
            x0, y0 = xa + ix * o.cell_m, ya + iy * o.cell_m
            body.append(f'<rect class="z{level}" x="{X(x0):.1f}" y="{Y(y0):.1f}" width="{o.cell_m * scale:.1f}" '
                        f'height="{o.cell_m * scale:.1f}"><title>{escape(SIDE_NAMES.get(side, side))}：'
                        f'{grid[iy, ix]:.1f} 秒</title></rect>')
    for x0, y0, x1, y1 in _court_lines(court):
        body.append(f'<line class="court" x1="{X(x0):.1f}" y1="{Y(y0):.1f}" x2="{X(x1):.1f}" y2="{Y(y1):.1f}"/>')
    W = court["half_width_doubles_m"]
    body.append(f'<line class="net" x1="{X(0):.1f}" y1="{Y(-W) - 3:.1f}" x2="{X(0):.1f}" y2="{Y(W) + 3:.1f}"/>')
    for (side, player), g in tracks.groupby(["side", "player"]):
        w = g["dt"].to_numpy(float)
        if w.sum() <= 0:
            continue
        mx, my = X(float(np.average(g["x"], weights=w))), Y(float(np.average(g["y"], weights=w)))
        body.append(f'<path class="mean" d="M{mx - 6:.1f},{my:.1f}H{mx + 6:.1f}M{mx:.1f},{my - 6:.1f}V{my + 6:.1f}">'
                    f'<title>{escape(SIDE_NAMES.get(side, side))}の平均の位置</title></path>')
    style = SVG_STYLE + ".lm .mean { stroke: var(--ink); stroke-width: 2; fill: none; }\n"
    return (f'<svg class="lm" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width + 8:.0f} {h + 4:.0f}" '
            f'width="100%" role="img" aria-label="選手がいた場所の分布"><style>{style}</style>'
            + "".join(body) + "</svg>")


# =============================================================================
# 合成の足の動きでの精度
# =============================================================================
def synthetic_benchmark(n_seeds: int = 4, fps: float = 30.0, seed: int = 0, duration_s: float = 20.0,
                        o: MovementOptions | None = None) -> dict[str, Any]:
    """合成の足の動き（``synth_players``）を 3 つのカメラ（後ろの角の高い所・横の低い所・真後ろ）で写し、
    移動距離・最高の速さの正解との比（−1）を手前・奥ごとに測る。"""
    from smash_speed.camera import look_at
    from smash_speed.synth_players import footwork_path, observe_players, path_length

    target = np.array([0.0, 0.0, 0.5])
    cams = {"back_high": look_at(np.array([-9.0, -5.5, 3.0]), target, 1500.0, 1920, 1080),
            "side_low": look_at(np.array([0.0, -8.0, 1.3]), target, 1100.0, 1920, 1080),
            "behind": look_at(np.array([-10.0, 0.0, 2.0]), target, 1600.0, 1920, 1080)}
    err: dict[str, dict[str, list[float]]] = {s: {"distance": [], "max_speed": []} for s in ("h", "o")}
    for name, cam in cams.items():
        for k in range(n_seeds):
            rng = np.random.default_rng(seed + k)
            t, feet_h = footwork_path(duration_s, fps, "h", rng)
            _, feet_o = footwork_path(duration_s, fps, "o", rng)
            feet = {"h": feet_h, "o": feet_o}
            summary, _ = movement_summary(observe_players(t, feet, cam, rng), o=o)
            for side, xy in feet.items():
                p = summary["players"][f"{side}0"]
                v_true = float(np.hypot(*np.gradient(xy, t, axis=0).T).max())
                err[side]["distance"].append(p["distance_m"] / path_length(xy) - 1)
                err[side]["max_speed"].append(p["max_speed_mps"] / v_true - 1)
    res: dict[str, Any] = {}
    for side, e in err.items():
        for k, v in e.items():
            a = np.array(v)
            res[f"{side}_{k}"] = {"bias": round(float(a.mean()), 4), "max_abs": round(float(np.abs(a).max()), 4),
                                  "n": len(a)}
    return res


# =============================================================================
# 書き出し
# =============================================================================
def write_movement(out_dir: str | Path, players: pd.DataFrame | None = None, court: dict[str, Any] | None = None,
                   rallies: list[Rally] | None = None, o: MovementOptions | None = None) -> dict[str, Any]:
    """``<出力先>/movement.json``・``movement.csv``・``movement.svg`` を書く。まとめを返す。

    ``players`` を省略すると ``players.csv`` を、``rallies`` を省略すると ``rallies.csv``（あれば）を読む。
    """
    from smash_speed.players import PLAYERS_FILE

    out_dir = Path(out_dir)
    players = pd.read_csv(out_dir / PLAYERS_FILE) if players is None else players
    if rallies is None and (out_dir / RALLIES_FILE).exists():
        rallies = read_rallies(out_dir / RALLIES_FILE)
    summary, tracks = movement_summary(players, court, rallies, o)
    (out_dir / MOVEMENT_FILE).write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    cols = ["frame", "t", "side", "player", "segment", "x", "y", "speed", "peak_speed", "outlier"]
    tracks[[c for c in cols if c in tracks]].to_csv(out_dir / MOVEMENT_CSV, index=False, float_format="%.4f")
    svg = movement_svg(tracks, court, o)
    if svg:
        (out_dir / MOVEMENT_SVG).write_text(svg, encoding="utf-8")
    return summary


def load_movement(out_dir: str | Path) -> dict[str, Any] | None:
    """出力先の ``movement.json``（この解析で選手の位置も出したとき＝``shot_list.json`` の ``players.movement`` が
    あるときだけ。前の解析の ``movement.json`` が残っていても使わない）。なければ None。"""
    out_dir = Path(out_dir)
    info_path = out_dir / "shot_list.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    if not (info.get("players") or {}).get("movement") or not (out_dir / MOVEMENT_FILE).exists():
        return None
    return json.loads((out_dir / MOVEMENT_FILE).read_text(encoding="utf-8"))


def movement_stats(summary: dict[str, Any]) -> dict[str, Any]:
    """``stats.json`` に入れる選手の動き：選手ごと（``h0``・``o0`` など。側は ``side``）の移動距離・速さ・
    いた区画の割合・ヒートマップ（``heat_brief``）。外れのコマの割合などの細かい数字は ``movement.json`` に残す。"""
    keys = ("side", "player", "seconds", "distance_m", "distance_per_min_m", "mean_speed_mps", "max_speed_mps",
            "max_speed_t", "moving_share", "fast_share", "rally_mean_distance_m", "mean_x", "mean_y", "zones", "heatmap")
    return {"players": {k: {c: p.get(c) for c in keys} for k, p in summary.get("players", {}).items()},
            "depth_names": summary.get("depth_names", list(DEPTH_NAMES)),
            "width_names": summary.get("width_names", list(WIDTH_NAMES))}


def movement_brief(summary: dict[str, Any]) -> dict[str, Any]:
    """``players.json``・``shot_list.json`` に入れる短いまとめ（選手ごとの移動距離・1 分あたり・最高の速さ）。"""
    keys = ("seconds", "distance_m", "distance_per_min_m", "max_speed_mps", "rally_mean_distance_m")
    return {k: {c: p.get(c) for c in keys} for k, p in summary["players"].items()}


def format_movement(summary: dict[str, Any]) -> str:
    """まとめの要約（1 人 1 行の日本語）。"""
    lines = []
    for p in summary["players"].values():
        s = (f"{SIDE_NAMES.get(p['side'], p['side'])}：{p['seconds']:.0f} 秒で {p['distance_m']:.0f} m"
             f"（1 分あたり {p['distance_per_min_m'] or 0:.0f} m、最高 {p['max_speed_mps'] or 0:.1f} m/s）")
        if p.get("rally_mean_distance_m") is not None:
            s += f"、1 ラリー平均 {p['rally_mean_distance_m']:.1f} m"
        lines.append(s)
    return "\n".join(lines) or "選手の位置がありません"


def main(argv: list[str] | None = None) -> None:
    import argparse

    from smash_speed.config import load_config

    ap = argparse.ArgumentParser(description="players.csv から、選手の移動距離・速さ・いた場所の分布を書き出す")
    ap.add_argument("out_dir", type=Path, help="players.csv のある出力先")
    ap.add_argument("--config", type=Path, default=None)
    a = ap.parse_args(argv)
    summary = write_movement(a.out_dir, court=load_config(a.config)["court"])
    print(format_movement(summary))
    print(f"{a.out_dir / MOVEMENT_FILE} に書き出しました")


if __name__ == "__main__":
    import sys

    if (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    main()
