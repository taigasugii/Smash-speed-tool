"""ダブルスの解析（ROADMAP D4）：誰が打ったか・陣形・ローテーション・カバー・2 人の間。

選手の追跡（``players.csv``。1 側に 2 人。``player_ids``）とショットの一覧（``shots.csv``）から出す。
選手は P1〜P4 と呼ぶ（h の側の 0・1 番が P1・P2、o の側が P3・P4。``player_ids.player_label``）。

1. 誰が打ったか（``pick_hitter``・``assign_doubles``）：打った側の 2 人それぞれに、打った時刻の
   - 画像の上で、シャトルの点から枠までの距離（枠の高さが単位。``hits.player_reach`` と同じ測り方）
   - シャトルの視線が、その人の足元の真上を通るときの高さと横のずれ（手の届く高さ・幅を超えた分）
   - 枠の形の変わり方（腕を上げる・踏み込むと枠が高く・広くなる）
   - （当てはめの後）打点と、1 本前のショットの軌道がその時刻にいた所から、足元までのコートの上の距離
   を足した費用を出し、小さい方を打った人にする。2 人の差が確かさ（``hitter_conf``）。
   当てはめの前は上の 3 つだけで選び、確かなものだけを打点の手がかり（D3c の足元）にする。
2. 陣形（``formation_table``）：2 人を結ぶ向きがコートの長さの向きに近ければ「前後（攻めの形）」、
   幅の向きに近ければ「横並び（守りの形）」、その間は「移っている途中」。短いぶれはならす。
3. ローテーション（``rotation_stats``）：上げた後（ロブ・クリア）に相手が打つまでに横並びになれたか、
   攻めた後（スマッシュ・ドロップ・プッシュ・ネット）に前後になれたかと、かかった時間。
4. カバー（``player_stats``）：選手ごとの打った本数・速さ・打った場所（前・中・後、左右）・移動距離。
5. 2 人の間（``gap_events``）：自分の側の床に落とされたシャトルが、相手が打った時の 2 人の立ち位置から見て
   どこだったか（2 人の間・前・後ろ・外側・体の近く）。

``write_doubles`` が ``doubles.json``（まとめ）と ``formation.csv``（コマごとの陣形）を書く。
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

from smash_speed.camera import Camera
from smash_speed.hitter import HitterOptions, hitter_at
from smash_speed.movement import MovementOptions, movement_tracks
from smash_speed.physics import PhysicsParams
from smash_speed.player_ids import _runs, player_label
from smash_speed.rally_stats import DEPTH_NAMES, SVG_STYLE, WIDTH_NAMES, _court_lines, zone_of
from smash_speed.shots import SHOT_TYPE_LABELS, Rally, Shot

DOUBLES_FILE = "doubles.json"
FORMATION_FILE = "formation.csv"
FORMATIONS = ("attack", "defence", "transition")
FORMATION_LABELS = {"attack": "前後（攻めの形）", "defence": "横並び（守りの形）", "transition": "移っている途中"}
ATTACK_TYPES = ("smash", "drop", "push", "net")   # 打った後、前後になるショット
LIFT_TYPES = ("lob", "clear")                     # 打った後、横並びになるショット（上げた球）
GAP_KINDS = {"between": "2 人の間", "front": "前（ネット側）", "back": "後ろ", "side": "外側（サイド）",
             "near": "体の近く"}


@dataclass
class DoublesOptions:
    """ダブルスの解析の条件。"""

    # 誰が打ったか
    reach_window_s: float = 0.1     # 打った時刻の前後この幅のシャトルの点を使う [s]
    box_window_s: float = 0.2       # 打った時刻からこの幅の中の、いちばん近いコマの枠を使う [s]
    reach_scale: float = 0.3        # 画像の上の距離（枠の高さが単位）の費用の単位
    reach_xy_m: float = 1.0         # 足元から横に手の届く幅 [m]
    reach_z_m: tuple[float, float] = (0.0, 3.1)   # 打点の高さの範囲 [m]
    ray_scale_m: float = 0.5        # 届く範囲を超えた分の費用の単位 [m]
    swing_window_s: float = 0.15    # 枠の形の変わり方を見る幅 [s]
    swing_gain: float = 0.15        # 枠が高く・広くなった割合の単位
    swing_weight: float = 0.5
    court_scale_m: float = 1.2      # 打点から足元までのコートの上の距離の費用の単位 [m]
    prior_min_conf: float = 0.75    # 当てはめの前に選んだ人の足元を打点の手がかりにする、確かさの下限
    # 陣形
    attack_deg: float = 55.0        # 2 人を結ぶ向きがコートの幅の向きからこれより立っていれば前後
    defence_deg: float = 35.0       # これより寝ていれば横並び
    min_dist_m: float = 0.8         # 2 人がこれより近いときは向きが決まらないので、前の陣形のまま
    min_dwell_s: float = 0.25       # これより短い陣形は前の陣形にまとめる
    # ローテーション
    settle_s: float = 0.3           # 目当ての陣形がこれだけ続いたら「なれた」
    max_rotation_s: float = 3.0     # 打ってからこれより後は見ない
    # 2 人の間
    land_margin_m: float = 0.3      # コートの線からこれだけ外までを「落とされた」に数える
    between_frac: float = 0.6       # 2 人を結ぶ線の中ほど（端から端を −1〜1 として ±この値）
    between_off_m: float = 1.2      # その線からの離れがこれ以下なら 2 人の間
    near_m: float = 1.0             # 近い方の選手からこれ以下なら体の近く
    edge_m: float = 0.5             # 2 人のどちらよりもこれだけ前・後ろ・外なら前・後ろ・外側

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def pair_sides(players: pd.DataFrame | None) -> list[str]:
    """2 人の選手がいる側（``players.csv`` の ``player`` が 2 種類ある側）。"""
    if players is None or players.empty:
        return []
    n = players.groupby("side")["player"].nunique()
    return [s for s in ("h", "o") if n.get(s, 0) >= 2]


# =============================================================================
# 1. 誰が打ったか
# =============================================================================
@dataclass
class HitterPick:
    """打った人の推定。"""

    side: str
    player: int
    conf: float                      # 0.5（決め手なし）〜 1
    n_candidates: int                # 比べた人数
    costs: dict[int, float]          # 選手 → 費用（小さいほど打った人らしい）


class PairTracks:
    """打った人を選ぶのに使う、選手ごとの枠の表（時刻の順）と、ふだんの枠の形。"""

    def __init__(self, players: pd.DataFrame, camera: Camera | None = None):
        from smash_speed.players import body_heights

        self.camera = camera
        self.by: dict[tuple[str, int], dict[str, np.ndarray]] = {}
        pl = players.dropna(subset=["x0", "y0", "x1", "y1"])
        for (side, player), g in pl.groupby(["side", "player"]):
            g = g.sort_values("t")
            d = {c: g[c].to_numpy(float) for c in ("t", "x", "y", "x0", "y0", "x1", "y1")}
            h = np.maximum(d["y1"] - d["y0"], 1.0)
            # 枠の高さ（足元の位置で測った [m]。カメラがなければ px）と、縦横の比。ふだんの値との比を見る
            tall = body_heights(g, camera) if camera is not None else h
            d["tall"] = tall / (np.nanmedian(tall) or 1.0)
            wide = (d["x1"] - d["x0"]) / h
            d["wide"] = wide - np.nanmedian(wide)
            self.by[(str(side), int(player))] = d

    def players_of(self, side: str) -> list[int]:
        return sorted(p for s, p in self.by if s == side)


def _reach(p: np.ndarray, box: np.ndarray) -> float:
    """シャトルの点 (n, 2) から枠（x0, y0, x1, y1）までのいちばん近い距離（枠の高さが単位。枠の中なら 0）。"""
    hh = max(float(box[3] - box[1]), 1.0)
    du = np.maximum(0.0, np.maximum(box[0] - p[:, 0], p[:, 0] - box[2]))
    dv = np.maximum(0.0, np.maximum(box[1] - p[:, 1], p[:, 1] - box[3]))
    return float(np.min(np.hypot(du, dv))) / hh


def _ray_cost(camera: Camera, p: np.ndarray, foot: np.ndarray, o: DoublesOptions) -> float:
    """シャトルの視線が足元の真上にいちばん近づく所の、横のずれと高さが手の届く範囲を超えた分（``ray_scale_m`` が単位）。"""
    c, d = camera.rays(p)
    best = np.inf
    for dk in d:
        n2 = float(dk[0] ** 2 + dk[1] ** 2)
        if n2 < 1e-9:
            continue
        s = float((foot - c[:2]) @ dk[:2]) / n2
        if s <= 0:
            continue
        miss = float(np.hypot(*(c[:2] + s * dk[:2] - foot)))
        z = float(c[2] + s * dk[2])
        over = max(0.0, miss - o.reach_xy_m) + max(0.0, z - o.reach_z_m[1]) + max(0.0, o.reach_z_m[0] - z)
        best = min(best, over / o.ray_scale_m)
    return best if np.isfinite(best) else 0.0


def pick_hitter(tracks: PairTracks, side: str, t_hit: float, shuttle: tuple[np.ndarray, np.ndarray] | None = None,
                points_xy: list[np.ndarray] | None = None, o: DoublesOptions | None = None) -> HitterPick | None:
    """``side`` の側の選手のうち、時刻 ``t_hit`` に打った人。選手がいなければ None。

    Args:
        shuttle: シャトルの見えた点の（時刻 (n,)、画像の上の位置 (n, 2)）
        points_xy: 打点のコートの上の位置の見積もり（当てはめた打点・1 本前の軌道がその時刻にいた所など）
    """
    o = o or DoublesOptions()
    costs: dict[int, float] = {}
    pts = None
    if shuttle is not None and len(shuttle[0]):
        m = np.abs(shuttle[0] - t_hit) <= o.reach_window_s
        pts = shuttle[1][m] if m.any() else None
    for p in tracks.players_of(side):
        d = tracks.by[(side, p)]
        k = int(np.argmin(np.abs(d["t"] - t_hit)))
        if abs(d["t"][k] - t_hit) > o.box_window_s:
            continue
        foot = np.array([d["x"][k], d["y"][k]])
        cost = 0.0
        if pts is not None:
            cost += _reach(pts, np.array([d[c][k] for c in ("x0", "y0", "x1", "y1")])) / o.reach_scale
            if tracks.camera is not None and np.isfinite(foot).all():
                cost += _ray_cost(tracks.camera, pts, foot, o)
        w = np.abs(d["t"] - t_hit) <= o.swing_window_s
        if w.any() and np.isfinite(d["tall"][w]).any():
            swing = max(0.0, float(np.nanmax(d["tall"][w])) - 1.0) + max(0.0, float(np.nanmax(d["wide"][w])))
            cost -= o.swing_weight * min(swing / o.swing_gain, 2.0)
        for xy in points_xy or []:
            if xy is not None and np.isfinite(xy).all() and np.isfinite(foot).all():
                cost += float(np.hypot(*(np.asarray(xy, float)[:2] - foot))) / o.court_scale_m
        costs[p] = cost
    if not costs:
        return None
    order = sorted(costs, key=costs.get)
    conf = 0.5 if len(order) < 2 else 1.0 / (1.0 + math.exp(-2.0 * (costs[order[1]] - costs[order[0]])))
    return HitterPick(side, order[0], round(conf, 3), len(order), {p: round(c, 3) for p, c in costs.items()})


def _visible(det: pd.DataFrame | None) -> tuple[np.ndarray, np.ndarray] | None:
    if det is None or not len(det):
        return None
    from smash_speed.hits import _visible_points

    return _visible_points(det)


def prefit_feet(times: list[float], sides: list[str], players: pd.DataFrame, det: pd.DataFrame | None,
                camera: Camera | None, o: DoublesOptions | None = None, ho: HitterOptions | None = None
                ) -> list[np.ndarray | None]:
    """当てはめの前の、打球ごとの打つ人の足元（打点の手がかり）。1 人の側はその人、2 人の側は確かに選べたときだけ。"""
    o = o or DoublesOptions()
    pairs = pair_sides(players)
    tracks = PairTracks(players, camera) if pairs else None
    vis = _visible(det) if pairs else None
    out: list[np.ndarray | None] = []
    for t, side in zip(times, sides):
        if side in pairs:
            pick = pick_hitter(tracks, side, t, vis, None, o)
            e = None if pick is None or pick.conf < o.prior_min_conf else hitter_at(players, t, side, o=ho,
                                                                                      player=pick.player)
        else:
            e = hitter_at(players, t, side, o=ho)
        out.append(None if e is None else np.array([e.x, e.y]))
    return out


def _incoming_xy(prev: Shot | None, s: Shot, phys: PhysicsParams | None) -> np.ndarray | None:
    """1 本前のショット（同じラリー）の軌道が、``s`` を打った時刻にいたコートの上の位置。出せなければ None。"""
    if prev is None or prev.rally_id is None or prev.rally_id != s.rally_id or prev.side == s.side:
        return None
    traj = prev.trajectory(phys)
    if traj is None or not len(traj["t"]) or s.t_hit_s > traj["t"][-1] + 0.1 or s.t_hit_s < traj["t"][0]:
        return None
    k = int(np.argmin(np.abs(traj["t"] - s.t_hit_s)))
    return np.asarray(traj["pos"][k][:2], float)


def assign_doubles(shots: list[Shot], players: pd.DataFrame, det: pd.DataFrame | None = None,
                   camera: Camera | None = None, phys: PhysicsParams | None = None, o: DoublesOptions | None = None,
                   ho: HitterOptions | None = None) -> dict[str, Any]:
    """ショットごとに、打った人（``hitter_id``・``hitter_conf``）とその足元（``hitter_x_m``・``hitter_y_m``）を入れる。

    1 人の側はその人（確かさは空欄）。入れた本数などを返す。2 人の側が 1 つもなければ何もしない（``n_pairs`` = 0）。
    """
    o = o or DoublesOptions()
    pairs = pair_sides(players)
    info: dict[str, Any] = {"n_pairs": len(pairs), "n": 0, "n_low_conf": 0, "mean_conf": None}
    if not pairs:
        return info
    tracks = PairTracks(players, camera)
    vis = _visible(det)
    order = sorted(shots, key=lambda s: s.t_hit_s)
    confs = []
    for prev, s in zip([None, *order[:-1]], order):
        s.hitter_id, s.hitter_conf = "", None
        if s.side not in ("h", "o"):
            s.hitter_x_m = s.hitter_y_m = None
            continue
        if s.side in pairs:
            hit = None if s.hit_x_m is None or s.hit_y_m is None else np.array([s.hit_x_m, s.hit_y_m])
            pick = pick_hitter(tracks, s.side, s.t_hit_s, vis, [hit, _incoming_xy(prev, s, phys)], o)
            if pick is None:
                s.hitter_x_m = s.hitter_y_m = None
                continue
            e = hitter_at(players, s.t_hit_s, s.side, o=ho, player=pick.player)
            s.hitter_id, s.hitter_conf = player_label(s.side, pick.player), pick.conf
            confs.append(pick.conf)
        else:
            e = hitter_at(players, s.t_hit_s, s.side, o=ho)
            if e is not None:
                s.hitter_id = player_label(s.side, e.player)
        s.hitter_x_m, s.hitter_y_m = (None, None) if e is None else (round(e.x, 3), round(e.y, 3))
        info["n"] += bool(s.hitter_id)
    if confs:
        info["n_low_conf"] = int(np.sum(np.array(confs) < o.prior_min_conf))
        info["mean_conf"] = round(float(np.mean(confs)), 3)
    return info


# =============================================================================
# 2. 陣形
# =============================================================================
FORMATION_COLUMNS = ["frame", "t", "side", "formation", "angle_deg", "dist_m", "front", "left"]


def classify_formation(dx: np.ndarray, dy: np.ndarray, t: np.ndarray, o: DoublesOptions | None = None) -> np.ndarray:
    """2 人の位置の差（長さの向き ``dx``・幅の向き ``dy`` [m]）の列から、コマごとの陣形（``FORMATIONS`` の文字列）。"""
    o = o or DoublesOptions()
    ang = np.degrees(np.arctan2(np.abs(dx), np.abs(dy)))
    lab = np.where(ang >= o.attack_deg, 0, np.where(ang <= o.defence_deg, 1, 2))
    close = np.hypot(dx, dy) < o.min_dist_m
    for k in np.flatnonzero(close):          # 近すぎて向きが決まらないコマは、前のコマのまま
        if k > 0:
            lab[k] = lab[k - 1]
    # 短い陣形は前の陣形にまとめる
    if len(lab) > 1:
        start = np.r_[0, np.flatnonzero(np.diff(lab)) + 1]
        end = np.r_[start[1:], len(lab)]
        dt = float(np.median(np.diff(t)))
        for a, b in zip(start[1:], end[1:]):
            if t[b - 1] - t[a] + dt < o.min_dwell_s:
                lab[a:b] = lab[a - 1]
    return np.array(FORMATIONS, dtype=object)[lab]


def formation_table(players: pd.DataFrame, o: DoublesOptions | None = None, mo: MovementOptions | None = None,
                    tracks: pd.DataFrame | None = None) -> pd.DataFrame:
    """2 人の選手がいる側の、コマごとの陣形の表（``FORMATION_COLUMNS``）。

    位置は ``movement.movement_tracks`` でならしたもの（``tracks`` に渡せば使い回す）。``front`` は前後のとき
    ネットに近い方の選手の番号、``left`` は自分の側からネットを向いて左にいる方（h の側は y の大きい方）。
    """
    o = o or DoublesOptions()
    pairs = pair_sides(players)
    if not pairs:
        return pd.DataFrame(columns=FORMATION_COLUMNS)
    tracks = movement_tracks(players, mo) if tracks is None else tracks
    out = []
    for side in pairs:
        g = tracks[tracks["side"] == side]
        a, b = (g[g["player"] == p][["frame", "t", "x", "y"]] for p in sorted(g["player"].unique())[:2])
        j = a.merge(b, on="frame", suffixes=("_a", "_b")).sort_values("t_a").set_index("frame")
        if j.empty:
            continue
        dx, dy = (j["x_a"] - j["x_b"]).to_numpy(float), (j["y_a"] - j["y_b"]).to_numpy(float)
        t = j["t_a"].to_numpy(float)
        sign = -1.0 if side == "h" else 1.0   # 自分の側を正にした長さの向き（小さい方がネットに近い）
        out.append(pd.DataFrame({
            "frame": j.index.to_numpy(), "t": t, "side": side, "formation": classify_formation(dx, dy, t, o),
            "angle_deg": np.round(np.degrees(np.arctan2(np.abs(dx), np.abs(dy))), 1),
            "dist_m": np.round(np.hypot(dx, dy), 2),
            "front": np.where(sign * dx <= 0, 0, 1),
            "left": np.where(sign * dy <= 0, 0, 1),
        }))
    if not out:
        return pd.DataFrame(columns=FORMATION_COLUMNS)
    return pd.concat(out, ignore_index=True)[FORMATION_COLUMNS]


def formation_at(table: pd.DataFrame, side: str, t: float, max_dt: float = 0.2) -> str:
    """時刻 ``t`` の ``side`` の側の陣形（近いコマがなければ空文字）。"""
    g = table[table["side"] == side]
    if g.empty:
        return ""
    tt = g["t"].to_numpy(float)
    k = int(np.argmin(np.abs(tt - t)))
    return str(g["formation"].iloc[k]) if abs(tt[k] - t) <= max_dt else ""


def _in_rallies(t: np.ndarray, rallies: list[Rally] | None) -> np.ndarray:
    if not rallies:
        return np.ones(len(t), bool)
    m = np.zeros(len(t), bool)
    for r in rallies:
        if r.t_end_s is not None:
            m |= (t >= r.t_start_s) & (t <= r.t_end_s)
    return m


def formation_summary(table: pd.DataFrame, rallies: list[Rally] | None = None) -> dict[str, Any]:
    """側ごとの、陣形ごとの時間の割合（ラリーの中だけ）・入れ替わった回数・前にいた割合。"""
    out: dict[str, Any] = {}
    for side, g in table.groupby("side"):
        g = g.sort_values("t")
        t = g["t"].to_numpy(float)
        dt = np.r_[np.diff(t), np.median(np.diff(t)) if len(t) > 1 else 0.0]
        dt = np.where(dt > 0.5, 0.0, dt)        # 位置の出ていない間は数えない
        m = _in_rallies(t, rallies)
        total = float(dt[m].sum())
        f = g["formation"].to_numpy()
        shares = {k: round(float(dt[m & (f == k)].sum()) / total, 3) if total > 0 else None for k in FORMATIONS}
        stable = f[m & (f != "transition")]
        att = m & (f == "attack")
        front = g["front"].to_numpy()
        att_s = float(dt[att].sum())
        out[str(side)] = {
            "seconds": round(total, 1), **shares,
            "n_changes": int(np.sum(stable[1:] != stable[:-1])) if len(stable) > 1 else 0,
            "mean_dist_m": round(float(np.average(g["dist_m"].to_numpy(float)[m], weights=dt[m])), 2)
            if total > 0 else None,
            "front_share": {player_label(str(side), p): round(float(dt[att & (front == p)].sum()) / att_s, 3)
                            for p in (0, 1)} if att_s > 0 else {},
        }
    return out


# =============================================================================
# 3. ローテーション
# =============================================================================
def rotation_events(shots: list[Shot], table: pd.DataFrame, o: DoublesOptions | None = None) -> list[dict[str, Any]]:
    """上げた・攻めたショットごとに、打ってから目当ての陣形（横並び・前後）になるまでの時間。

    1 件は ``{t, side, kind（lift | attack）, target, already（打った時にもうその形）, seconds（なれた時刻 − 打った時刻。
    なれなければ None）, in_time（相手が打つまでになれたか。相手が打たなければ None）}``。
    """
    o = o or DoublesOptions()
    order = sorted(shots, key=lambda s: s.t_hit_s)
    out = []
    for i, s in enumerate(order):
        kind = "lift" if s.shot_type in LIFT_TYPES else "attack" if s.shot_type in ATTACK_TYPES else ""
        g = table[table["side"] == s.side]
        if not kind or g.empty:
            continue
        target = "defence" if kind == "lift" else "attack"
        nxt = next((n for n in order[i + 1:] if n.rally_id == s.rally_id and s.rally_id is not None), None)
        t_reply = nxt.t_hit_s if nxt is not None and nxt.side != s.side else None
        w = g[(g["t"] >= s.t_hit_s - 0.05) & (g["t"] <= s.t_hit_s + o.max_rotation_s)]
        if len(w) < 3:
            continue
        t = w["t"].to_numpy(float)
        ok = (w["formation"] == target).to_numpy()
        seconds = None
        for a, b in _runs(ok):
            if t[b - 1] - t[a] >= o.settle_s or b == len(t):
                seconds = max(0.0, float(t[a] - s.t_hit_s))
                break
        already = bool(ok[0])
        out.append({"t": round(s.t_hit_s, 3), "side": s.side, "kind": kind, "target": target, "already": already,
                    "seconds": None if seconds is None else round(0.0 if already else seconds, 2),
                    "in_time": None if t_reply is None else bool(seconds is not None
                                                                 and s.t_hit_s + seconds <= t_reply)})
    return out


def rotation_stats(events: list[dict[str, Any]]) -> dict[str, Any]:
    """側ごと・上げた後／攻めた後ごとの、本数・相手が打つまでになれた本数・かかった時間の中央値。"""
    out: dict[str, Any] = {}
    for side in sorted({e["side"] for e in events}):
        out[side] = {}
        for kind in ("lift", "attack"):
            ev = [e for e in events if e["side"] == side and e["kind"] == kind]
            judged = [e for e in ev if e["in_time"] is not None]
            moved = [e["seconds"] for e in ev if e["seconds"] is not None and not e["already"]]
            out[side][kind] = {
                "n": len(ev), "n_already": sum(e["already"] for e in ev),
                "n_judged": len(judged), "n_in_time": sum(bool(e["in_time"]) for e in judged),
                "median_s": round(float(np.median(moved)), 2) if moved else None,
            }
    return out


# =============================================================================
# 4. 選手ごと
# =============================================================================
def player_stats(shots: list[Shot], movement: dict[str, Any] | None = None, court: dict[str, Any] | None = None
                 ) -> dict[str, Any]:
    """選手（P1〜P4）ごとの、打った本数・ペアの中の割合・種類ごとの本数・速さ・打った場所・移動距離。

    打った場所は打つ人の足元の区画（``rally_stats.zone_of``。行 = 前・中・後、列 = 幅の向き）。
    """
    by: dict[str, Any] = {}
    side_n = {side: sum(1 for s in shots if s.side == side and s.hitter_id) for side in ("h", "o")}
    for s in shots:
        if not s.hitter_id:
            continue
        p = by.setdefault(s.hitter_id, {"side": s.side, "n_shots": 0, "by_type": {}, "speeds": [],
                                        "zones": [[0] * 3 for _ in range(3)], "fastest": None})
        p["n_shots"] += 1
        p["by_type"][s.shot_type] = p["by_type"].get(s.shot_type, 0) + 1
        if s.speed_kmh is not None:
            p["speeds"].append(s.speed_kmh)
            if p["fastest"] is None or s.speed_kmh > p["fastest"]["kmh"]:
                p["fastest"] = {"kmh": round(s.speed_kmh, 1), "t": round(s.t_hit_s, 2), "type": s.shot_type}
        if s.hitter_x_m is not None and s.hitter_y_m is not None:
            z = zone_of(s.hitter_x_m, s.hitter_y_m, court)
            if z is not None and z[0] == s.side:
                p["zones"][z[1]][z[2]] += 1
    moves = {player_label(m["side"], m["player"]): m for m in (movement or {}).get("players", {}).values()}
    for name, p in by.items():
        v = p.pop("speeds")
        p["share_of_pair"] = round(p["n_shots"] / side_n[p["side"]], 3) if side_n[p["side"]] else None
        p["mean_kmh"] = round(float(np.mean(v)), 1) if v else None
        p["max_kmh"] = round(float(np.max(v)), 1) if v else None
        p["by_type"] = {k: {"label": SHOT_TYPE_LABELS.get(k, k), "n": n} for k, n in
                        sorted(p["by_type"].items(), key=lambda kv: -kv[1])}
        p["n_front"], p["n_mid"], p["n_back"] = (int(sum(row)) for row in p["zones"])
        m = moves.get(name)
        if m is not None:
            p["distance_m"], p["distance_per_min_m"] = m.get("distance_m"), m.get("distance_per_min_m")
            p["max_speed_mps"] = m.get("max_speed_mps")
            grid = (m.get("zones") or {}).get("grid")
            if grid:
                p["time_front"], p["time_mid"], p["time_back"] = (round(float(sum(row)), 3) for row in grid)
    return dict(sorted(by.items()))


# =============================================================================
# 5. 2 人の間
# =============================================================================
def classify_gap(land: np.ndarray, a: np.ndarray, b: np.ndarray, side: str, o: DoublesOptions | None = None) -> str:
    """落ちた所 ``land`` が、2 人の立ち位置 ``a``・``b``（コートの座標 [m]）から見てどこか（``GAP_KINDS`` のキー）。"""
    o = o or DoublesOptions()
    ab = b - a
    n = float(np.hypot(*ab))
    mid = (a + b) / 2
    if n > 1e-6:
        e = ab / n
        along = float((land - mid) @ e) / (n / 2)
        off = abs(float((land - mid) @ np.array([-e[1], e[0]])))
        if abs(along) <= o.between_frac and off <= o.between_off_m and n >= 2 * o.near_m:
            return "between"
    if min(float(np.hypot(*(land - a))), float(np.hypot(*(land - b)))) <= o.near_m:
        return "near"
    sign = -1.0 if side == "h" else 1.0
    depth = sign * np.array([land[0], a[0], b[0]])      # ネットからの距離
    if depth[0] < min(depth[1:]) - o.edge_m:
        return "front"
    if depth[0] > max(depth[1:]) + o.edge_m:
        return "back"
    if land[1] < min(a[1], b[1]) - o.edge_m or land[1] > max(a[1], b[1]) + o.edge_m:
        return "side"
    return "between" if n > 1e-6 and abs(along) <= 1.0 else "near"


def gap_events(shots: list[Shot], tracks: pd.DataFrame, court: dict[str, Any] | None = None,
               o: DoublesOptions | None = None) -> list[dict[str, Any]]:
    """打ち返せずに自分の側の床に落ちたシャトルごとに、相手が打った時の 2 人の立ち位置と、落ちた所の分類。

    ``tracks`` は ``movement.movement_tracks``（ならした位置）。ラリーの最後のショット（次の打球が同じラリーにない）で、
    着地点が受ける側のコートの中（線から ``land_margin_m`` 外まで）のものを数える。
    """
    o = o or DoublesOptions()
    c = court or {}
    L, W = c.get("half_length_m", 6.70), c.get("half_width_doubles_m", 3.05)
    order = sorted(shots, key=lambda s: s.t_hit_s)
    pairs = {side: [g[g["player"] == p].sort_values("t") for p in sorted(g["player"].unique())[:2]]
             for side, g in tracks.groupby("side") if g["player"].nunique() >= 2}
    out = []
    for i, s in enumerate(order):
        recv = "o" if s.side == "h" else "h" if s.side == "o" else ""
        if recv not in pairs or s.land_x_m is None or s.land_y_m is None or s.rally_id is None:
            continue
        if any(n.rally_id == s.rally_id for n in order[i + 1:i + 2]):
            continue                                   # 打ち返された
        land = np.array([s.land_x_m, s.land_y_m])
        sx = land[0] * (1.0 if recv == "o" else -1.0)
        if not (0.0 <= sx <= L + o.land_margin_m and abs(land[1]) <= W + o.land_margin_m):
            continue                                   # ネットにかかった・外に出た
        pos = []
        for g in pairs[recv]:
            t = g["t"].to_numpy(float)
            k = int(np.argmin(np.abs(t - s.t_hit_s))) if len(t) else -1
            if k < 0 or abs(t[k] - s.t_hit_s) > 0.3:
                break
            pos.append(np.array([float(g["x"].iloc[k]), float(g["y"].iloc[k])]))
        if len(pos) < 2:
            continue
        d = [float(np.hypot(*(land - p))) for p in pos]
        out.append({"t": round(s.t_hit_s, 3), "side": recv, "type": s.shot_type,
                    "x": round(float(land[0]), 2), "y": round(float(land[1]), 2),
                    "kind": classify_gap(land, pos[0], pos[1], recv, o),
                    "players": [[round(float(v), 2) for v in p] for p in pos],
                    "d_near_m": round(min(d), 2), "nearest": player_label(recv, int(np.argmin(d)))})
    return out


def gap_stats(events: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for side in sorted({e["side"] for e in events}):
        ev = [e for e in events if e["side"] == side]
        out[side] = {"n": len(ev),
                     "by_kind": {k: {"label": GAP_KINDS[k], "n": sum(e["kind"] == k for e in ev)} for k in GAP_KINDS},
                     "mean_d_near_m": round(float(np.mean([e["d_near_m"] for e in ev])), 2) if ev else None,
                     "events": ev}
    return out


# =============================================================================
# まとめと書き出し
# =============================================================================
def analyze_doubles(shots: list[Shot], rallies: list[Rally] | None, players: pd.DataFrame,
                    court: dict[str, Any] | None = None, movement: dict[str, Any] | None = None,
                    o: DoublesOptions | None = None) -> tuple[dict[str, Any], pd.DataFrame]:
    """ダブルスのまとめ（``doubles.json`` の中身）と、コマごとの陣形の表。2 人の側がなければ（{}, 空の表）。

    ショットの ``formation``・``opp_formation``（打った時の陣形）もここで入れる。
    打った人（``hitter_id``）は ``assign_doubles`` で先に入れておく。
    """
    o = o or DoublesOptions()
    pairs = pair_sides(players)
    if not pairs:
        return {}, pd.DataFrame(columns=FORMATION_COLUMNS)
    tracks = movement_tracks(players)
    table = formation_table(players, o, tracks=tracks)
    for s in shots:
        other = "o" if s.side == "h" else "h"
        s.formation = formation_at(table, s.side, s.t_hit_s) if s.side in pairs else ""
        s.opp_formation = formation_at(table, other, s.t_hit_s) if s.side and other in pairs else ""
    confs = [s.hitter_conf for s in shots if s.hitter_conf is not None]
    summary = {
        "per_side": {side: int(players.loc[players["side"] == side, "player"].nunique()) for side in ("h", "o")},
        "player_names": {player_label(side, p): {"side": side, "player": p} for side in pairs for p in (0, 1)},
        "players": player_stats(shots, movement, court),
        "formation": formation_summary(table, rallies),
        "rotation": rotation_stats(rotation_events(shots, table, o)),
        "gaps": gap_stats(gap_events(shots, tracks, court, o)),
        "hitters": {"n": sum(bool(s.hitter_id) for s in shots), "n_shots": len(shots),
                    "n_low_conf": int(np.sum(np.array(confs) < o.prior_min_conf)) if confs else 0,
                    "mean_conf": round(float(np.mean(confs)), 3) if confs else None},
        "formation_labels": dict(FORMATION_LABELS), "depth_names": list(DEPTH_NAMES),
        "width_names": list(WIDTH_NAMES), "options": o.to_dict(),
    }
    return summary, table


def write_doubles(out_dir: str | Path, shots: list[Shot], rallies: list[Rally] | None, players: pd.DataFrame,
                  court: dict[str, Any] | None = None, movement: dict[str, Any] | None = None,
                  o: DoublesOptions | None = None) -> dict[str, Any]:
    """``<出力先>/doubles.json``・``formation.csv`` を書く。2 人の側がなければ、前の解析のファイルを消して {} を返す。"""
    out_dir = Path(out_dir)
    summary, table = analyze_doubles(shots, rallies, players, court, movement, o)
    if not summary:
        for name in (DOUBLES_FILE, FORMATION_FILE):
            (out_dir / name).unlink(missing_ok=True)
        return {}
    (out_dir / DOUBLES_FILE).write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    table.to_csv(out_dir / FORMATION_FILE, index=False, float_format="%.3f")
    return summary


def load_doubles(out_dir: str | Path) -> dict[str, Any] | None:
    path = Path(out_dir) / DOUBLES_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def doubles_brief(summary: dict[str, Any]) -> dict[str, Any]:
    """``shot_list.json``・``stats.json`` に入れる短いまとめ（落とされた 1 件ずつなどの細かい表は ``doubles.json`` に残す）。"""
    if not summary:
        return {}
    gaps = {side: {k: v for k, v in g.items() if k != "events"} for side, g in summary.get("gaps", {}).items()}
    return {k: summary.get(k) for k in ("per_side", "players", "formation", "rotation", "hitters")} | {"gaps": gaps}


# =============================================================================
# 図（落とされた所と、そのときの 2 人の立ち位置）
# =============================================================================
def gaps_svg(summary: dict[str, Any], court: dict[str, Any] | None = None, width: float = 560.0) -> str:
    """コートを上から見た図に、落とされた所（×）と、相手が打った時の 2 人の立ち位置（●。線で結ぶ）を描いた SVG。なければ空文字。"""
    from smash_speed.config import load_config

    events = [e for g in summary.get("gaps", {}).values() for e in g.get("events", [])]
    if not events:
        return ""
    court = court or load_config()["court"]
    L, W = court["half_length_m"], court["half_width_doubles_m"]
    pad = 0.6
    scale = width / (2 * (L + pad))
    h = 2 * (W + pad) * scale + 26
    X = lambda x: 4 + (x + L + pad) * scale  # noqa: E731
    Y = lambda y: 24 + (y + W + pad) * scale  # noqa: E731
    body = [f'<rect class="bg" x="0" y="0" width="{width + 8:.0f}" height="{h + 4:.0f}"/>',
            '<text class="title" x="4" y="15">落とされた所（×）と、そのときの 2 人（●）</text>']
    for x0, y0, x1, y1 in _court_lines(court):
        body.append(f'<line class="court" x1="{X(x0):.1f}" y1="{Y(y0):.1f}" x2="{X(x1):.1f}" y2="{Y(y1):.1f}"/>')
    body.append(f'<line class="net" x1="{X(0):.1f}" y1="{Y(-W) - 3:.1f}" x2="{X(0):.1f}" y2="{Y(W) + 3:.1f}"/>')
    for e in events:
        (ax, ay), (bx, by) = e["players"]
        tip = escape(f"{e['t']:.1f} 秒・{GAP_KINDS[e['kind']]}（近い方の選手から {e['d_near_m']:.1f} m）")
        body.append(f'<g><title>{tip}</title>'
                    f'<line class="pair" x1="{X(ax):.1f}" y1="{Y(ay):.1f}" x2="{X(bx):.1f}" y2="{Y(by):.1f}"/>'
                    f'<circle class="who" cx="{X(ax):.1f}" cy="{Y(ay):.1f}" r="3.5"/>'
                    f'<circle class="who" cx="{X(bx):.1f}" cy="{Y(by):.1f}" r="3.5"/>'
                    f'<path class="miss" d="M{X(e["x"]) - 4:.1f},{Y(e["y"]) - 4:.1f}l8,8m0,-8l-8,8"/></g>')
    style = (SVG_STYLE + ".lm .pair { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 3 3; }\n"
             ".lm .who { fill: var(--dot); }\n.lm .miss { stroke: #c2410c; stroke-width: 2; fill: none; }\n")
    return (f'<svg class="lm" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width + 8:.0f} {h + 4:.0f}" '
            f'width="{width + 8:.0f}" role="img" aria-label="落とされた所と 2 人の立ち位置"><style>{style}</style>'
            + "".join(body) + "</svg>")
