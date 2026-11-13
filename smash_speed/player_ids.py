"""ダブルスの選手の追跡（ROADMAP D4a）：1 側の 2 人を、同じ人に同じ番号を付けたまま追う。

シングルスの追跡（``players._track_singles``）は「コートの中にいた点の多い軌跡の組」を 1 人分選ぶだけで、
2 人目は残りの検出から選び直していた。これだと 2 人がすれ違う・重なるたびに番号が入れ替わる。ここでは 3 段階で決める。

1. どの検出が選手か（``choose_chains``）：近いコマどうしをつないだ短い軌跡（``players.link_tracklets``）を、
   別の人に乗り移った所（1 コマで大きく飛んだ・枠の中の色が変わった）で切ってから（``split_impure``）、
   時間が重ならない並びを **2 本同時に** 選ぶ。重みは「コートの線の中にいた時間」（線の外にいた時間は少しだけ数える）、
   並びの中のつなぎには「画像の上で動ける範囲か」と離れ具合の費用を付け、合計が最も良い 2 本を最小費用流（線形計画）で出す。
   待っている人・隣のコートの人は、線の外にいるので重みがほとんどなく、選手の軌跡と時間が重なるので選ばれない。
2. どちらがどちらか（``resolve_identities``）：2 本の並びは、2 人が重なった所・片方が相手の近くで消えた所・
   長く見えなかった所（「入れ替わりうる所」）で入れ替わっているかもしれない。入れ替わりうる所で区切った区間ごとに
   「そのまま・入れ替え」を、(a) 区切りの前後の動き（位置と速さの続き具合）と (b) 枠の中の色（``people.csv`` の ``app``。
   2 人がはっきり違う色のときだけ使う）から、いちばんありそうな並びに決める（ビタビ）。
   同じ色の服の 2 人は色では分けられないので、動きだけで決める。
3. 自動（``per_side`` = 0）：2 本目が「選手らしい」（コートの線の内側に長くいて、1 本目と同時に写っていて、動いている）
   ときだけ、その側をダブルスとして扱う。そうでなければシングルスの追跡の結果をそのまま返す。

番号（``player``）は 0・1。最初に 2 人とも写った区間で y が小さい方が 0。表では h の側の 0・1 を P1・P2、
o の側を P3・P4 と呼ぶ（``PLAYER_LABELS``）。コートの入れ替え（チェンジエンド）の前後で同じ人を結ぶことはしていない。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from smash_speed.players import (SIDES, PlayerOptions, _fill_gaps, _player_table, _track_singles, decode_appearance,
                                 link_tracklets, side_region)

PLAYER_LABELS = {("h", 0): "P1", ("h", 1): "P2", ("o", 0): "P3", ("o", 1): "P4"}


def player_label(side: str, player: int) -> str:
    """選手の呼び名（P1〜P4）。h の側の 0・1 が P1・P2、o の側が P3・P4。"""
    return PLAYER_LABELS.get((side, int(player)), f"{side}{int(player)}")


@dataclass
class IdOptions:
    """2 人を同じ番号のまま追う条件。距離は画像の上の足元の距離を枠の高さで割ったもの（``players.link_tracklets`` と同じ）。"""

    # 近いコマどうしをつなぐとき（``link_with_motion``）
    predict_s: float = 0.2        # 軌跡の速さで先を見越す長さの上限 [s]（0 なら ``players.link_tracklets`` と同じ）
    vel_blend: float = 0.5        # 速さの見積もりに、新しい 1 コマの動きを混ぜる割合
    # 軌跡を切る所（別の人に乗り移った所）
    split_jump_h: float = 0.25    # 1 コマの動きが、前のコマの動きからこれより大きく変わったら切る
    split_app: float = 0.8        # 前後 ``split_app_n`` コマの色の中央値の差（割合の差の合計。0〜4）がこれより大きければ切る
    split_app_n: int = 7
    # どの検出が選手か（最小費用流）。重み・費用の単位は「コートの中にいた秒数 × 検出のスコア」
    outside_cap_s: float = 0.1    # 1 本の軌跡で、線の外にいた分として数える重みの上限
    track_cost: float = 0.12      # 軌跡 1 本を選ぶ費用（線の外にしかいない軌跡は選ばない）
    link_gap_s: float = 2.0       # これより短い途切れは、動ける範囲の中の軌跡にだけつなぐ。長ければどこへでもつなげる
    link_cost: float = 0.15       # つなぎの離れ具合（動ける範囲いっぱいで 1）にかける費用
    gap_cost: float = 0.07        # 途切れの長さ [s] にかける費用
    idle_cost: float = 0.03       # 長い途切れの後につなぐ費用
    # 入れ替わりうる所
    contact_h: float = 0.5        # 2 人の足元がこれより近いコマは重なっている
    contact_iou: float = 0.4      # 2 人の枠の重なり（IoU）がこれより大きいコマも重なっている
    merge_h: float = 1.0          # 片方が、相手からこれより近い所で消えた・現れたら、その間は 1 つの枠にまとまっていたとみなす
    pad_s: float = 0.1            # 重なりの前後もこの幅だけ含める
    join_s: float = 0.3           # これより短い間の空いた 2 つの重なりは 1 つにまとめる
    # 動きの続き具合
    vel_window_s: float = 0.3     # 区切りの前の速さを測る幅
    vel_horizon_s: float = 0.4    # 速さで先を見越す長さの上限
    motion_sigma_h: float = 0.35  # 位置の続き具合の許す幅（区切りの長さ 0 のとき）
    motion_speed_h: float = 1.5   # 区切りが長いほど広げる分 [枠の高さ/s]
    stay_prior: float = 0.5       # 手がかりがないときに「そのまま」を選ぶ分（対数の比）
    same_track: float = 5.0       # 区切りの前後が同じ軌跡（コマごとに続けて追えていた）のときに「そのまま」に足す分
    max_logodds: float = 8.0
    # 色
    min_clean: int = 4            # 区間の色を出すのに要る、重なっていないコマの数
    app_min_sep: float = 3.0      # 2 人の色の離れ具合（区間ごとのばらつきが単位）がこれより小さければ色を使わない
                                  # （区間が少ないと偶然離れて見えるので、下限を 1 + 4 / 区間の数 倍にする）
    app_min_segments: int = 4     # 2 人とも色の出た区間がこれより少なければ色を使わない
    app_cap: float = 12.0
    # 自動でダブルスとみなす条件（2 本目の選手について）
    auto_min_cover: float = 0.35  # 位置が出たコマの割合の下限
    auto_min_both: float = 0.3    # どちらかが写っているコマのうち、2 人とも写っているコマの割合の下限
    auto_inset_m: float = 0.3     # コートの後ろの線・横の線からこれだけ内側を「線の内側」とみなす
    auto_min_inside: float = 0.6  # 線の内側にいたコマの割合の下限（待っている人は後ろの線の上・外にいる）
    auto_min_range_m: float = 0.8 # 動いた幅（x と y の 10〜90% の幅の和）の下限（4 秒より短い区間では見ない）

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# =============================================================================
# 1. どの検出が選手か
# =============================================================================
def link_with_motion(cand: pd.DataFrame, o: PlayerOptions, ido: IdOptions | None = None) -> np.ndarray:
    """``players.link_tracklets`` と同じつなぎ方で、軌跡の速さから見越した位置との距離で割り当てる。

    最後の位置との距離だけで割り当てると、止まっている人が隠れた所を別の人が通ったとき、通った人の検出が
    止まっていた人の軌跡につながる。速さを見越すと、動いている人の検出は動いている軌跡につながりやすい。
    """
    from scipy.optimize import linear_sum_assignment

    ido = ido or IdOptions()
    ids = np.full(len(cand), -1, int)
    active: list[list[float]] = []   # [軌跡の番号, 最後の時刻, u, v, 枠の高さ, 速さ u, 速さ v]
    next_id = 0
    frames = cand["frame"].to_numpy()
    ts = cand["t"].to_numpy(float)
    xy = np.column_stack([cand[["u", "v"]].to_numpy(float), (cand["y1"] - cand["y0"]).to_numpy(float)])
    order = np.argsort(frames, kind="stable")
    bounds = np.flatnonzero(np.diff(frames[order])) + 1
    for rows in np.split(order, bounds):
        if not len(rows):
            continue
        t = ts[rows[0]]
        active = [a for a in active if t - a[1] <= o.max_gap_s]
        used = set()
        if active:
            last = np.array([a[2:] for a in active])
            dt = t - np.array([a[1] for a in active])
            pred = last[:, :2] + last[:, 3:5] * np.minimum(dt, ido.predict_s)[:, None]
            scale = (last[:, None, 2] + xy[rows][None, :, 2]) / 2
            cost = np.linalg.norm(pred[:, None, :] - xy[rows][None, :, :2], axis=2) / scale
            gate = o.gate_h + o.max_speed_h * dt
            big = 1e6
            c = np.where(cost <= gate[:, None], cost, big)
            for ai, ri in zip(*linear_sum_assignment(c)):
                if c[ai, ri] >= big:
                    continue
                r = rows[ri]
                a = active[ai]
                ids[r] = int(a[0])
                step = (xy[r, :2] - np.array(a[2:4])) / max(dt[ai], 1e-6)
                vel = (1 - ido.vel_blend) * np.array(a[5:7]) + ido.vel_blend * step if dt[ai] <= ido.predict_s else step * 0
                active[ai][1:] = [t, *xy[r], *vel]
                used.add(ri)
        for ri, r in enumerate(rows):
            if ri not in used:
                ids[r] = next_id
                active.append([next_id, t, *xy[r], 0.0, 0.0])
                next_id += 1
    return ids


def split_impure(cand: pd.DataFrame, tid: np.ndarray, ido: IdOptions | None = None) -> np.ndarray:
    """軌跡が別の人に乗り移った所で切り、付け直した軌跡の番号を返す。

    近いコマどうしをつなぐだけだと、止まっている人の前を選手が通って隠したときなどに、軌跡が別の人へ移る。
    (a) 1 コマの動き（枠の高さが単位）が前のコマの動きから ``split_jump_h`` より大きく変わった所と、
    (b) 色（``app``）の前後 ``split_app_n`` コマの中央値が ``split_app`` より大きく違う所（その山の頂）で切る。
    """
    ido = ido or IdOptions()
    out = np.asarray(tid).copy()
    nxt = int(out.max()) + 1 if len(out) else 0
    app = decode_appearance(cand["app"]) if "app" in cand else None
    order = np.lexsort((cand["frame"].to_numpy(), out))
    bounds = np.flatnonzero(np.diff(out[order])) + 1
    uv = cand[["u", "v"]].to_numpy(float)
    hh = np.maximum((cand["y1"] - cand["y0"]).to_numpy(float), 1.0)
    n_app = ido.split_app_n
    for rows in np.split(order, bounds):
        if len(rows) < 4:
            continue
        step = np.diff(uv[rows], axis=0) / hh[rows][1:, None]
        cut = np.zeros(len(rows), bool)             # cut[i]：i 番目の行から新しい軌跡
        cut[2:] = np.hypot(*np.diff(step, axis=0).T) > ido.split_jump_h
        # 1 コマだけの外れ（次のコマで戻る）は切らない：続けて 2 つ切れる所は外れとみなす
        both = cut[:-1] & cut[1:]
        cut[:-1][both] = False
        cut[1:][both] = False
        if app is not None and len(rows) >= 2 * n_app:
            a = pd.DataFrame(app[rows])
            before = a.rolling(n_app, min_periods=n_app).median().shift(1).to_numpy()
            after = a[::-1].rolling(n_app, min_periods=n_app).median()[::-1].to_numpy()
            d = np.nan_to_num(np.abs(before - after).sum(axis=1))
            for i0, i1 in _runs(d > ido.split_app):
                cut[i0 + int(np.argmax(d[i0:i1]))] = True
        for i in np.flatnonzero(cut):
            out[rows[i:]] = nxt
            nxt += 1
    return out


def chain_spans(cand: pd.DataFrame, o: PlayerOptions, ido: IdOptions) -> pd.DataFrame:
    """軌跡ごとの始まり・終わりと、並びを選ぶときの重み [s]：線の中にいた時間 × スコア ＋ 線の外の分（上限つき）− 選ぶ費用。"""
    ft = cand.drop_duplicates("frame").sort_values("frame")["t"].to_numpy(float)
    dt = float(np.median(np.diff(ft))) if len(ft) > 1 else 1.0
    inside = cand["weight"].to_numpy(float) > o.outside_weight * cand["score"].to_numpy(float) + 1e-9
    w = cand["weight"].to_numpy(float) * dt
    g = pd.DataFrame({"tid": cand["tid"].to_numpy(), "t": cand["t"].to_numpy(float), "win": np.where(inside, w, 0.0),
                      "wout": np.where(inside, 0.0, w)}).groupby("tid")
    spans = pd.DataFrame({"t0": g["t"].min(), "t1": g["t"].max(),
                          "weight": g["win"].sum() + np.minimum(g["wout"].sum(), ido.outside_cap_s) - ido.track_cost})
    return spans[spans["t1"] - spans["t0"] >= o.min_track_s].rename_axis("id").reset_index()


def choose_chains(cand: pd.DataFrame, o: PlayerOptions, ido: IdOptions | None = None, k: int = 2) -> list[list[int]]:
    """軌跡（``cand`` の ``tid``）から、時間が重ならない並びを ``k`` 本、重みの合計 − つなぎの費用が最も良くなるよう選ぶ。

    軌跡 i の終わりから軌跡 j の始まりへは、途切れが ``link_gap_s`` 以下なら動ける範囲（``gate_h + max_speed_h × 時間``）
    の中のときだけ、それより長ければいつでもつなげる。選んだ軌跡の番号の並び（時刻の順）を、重みの大きい順に返す。
    """
    ido = ido or IdOptions()
    spans = chain_spans(cand, o, ido).sort_values(["t0", "t1"]).reset_index(drop=True)
    n = len(spans)
    if n == 0:
        return []
    c = cand.sort_values(["tid", "frame"])
    g = c.groupby("tid")
    first, last = g.first().loc[spans["id"]], g.last().loc[spans["id"]]
    t0, t1, w = (spans[k_].to_numpy(float) for k_ in ("t0", "t1", "weight"))
    p0, p1 = first[["u", "v"]].to_numpy(float), last[["u", "v"]].to_numpy(float)
    h0, h1 = (first["y1"] - first["y0"]).to_numpy(float), (last["y1"] - last["y0"]).to_numpy(float)

    src: list[np.ndarray] = []
    dst: list[np.ndarray] = []
    cost: list[np.ndarray] = []
    cap: list[np.ndarray] = []

    def add(a: Any, b: Any, cst: Any, cp: float = 1.0) -> None:
        a, b, cst = np.atleast_1d(a), np.atleast_1d(b), np.atleast_1d(np.asarray(cst, float))
        src.append(np.broadcast_to(a, cst.shape) if a.shape != cst.shape else a)
        dst.append(np.broadcast_to(b, cst.shape) if b.shape != cst.shape else b)
        cost.append(cst)
        cap.append(np.full(cst.shape, cp))

    starts = np.unique(t0)                       # 長い途切れの後に入る所（待ちの節）
    m = len(starts)
    SRC, SINK = -1, -2
    idle = lambda q: n + q                       # noqa: E731
    add(SRC, idle(0), 0.0, float(k))
    if m > 1:
        add(idle(np.arange(m - 1)), idle(np.arange(1, m)), np.zeros(m - 1), float(k))
    add(idle(np.searchsorted(starts, t0)), np.arange(n), ido.idle_cost - w)
    add(np.arange(n), SINK, np.zeros(n))
    q = np.searchsorted(starts, t1 + ido.link_gap_s, side="right")
    ok = q < m
    add(np.flatnonzero(ok), idle(q[ok]), np.zeros(int(ok.sum())))
    lo = np.searchsorted(t0, t1 - o.max_overlap_s, side="left")
    hi = np.searchsorted(t0, t1 + ido.link_gap_s, side="right")
    for i in range(n):
        j = np.arange(max(lo[i], i + 1), hi[i])
        j = j[t1[j] > t1[i]]
        if not len(j):
            continue
        dt = t0[j] - t1[i]
        dist = np.hypot(*(p0[j] - p1[i]).T) / np.maximum((h0[j] + h1[i]) / 2, 1.0)
        gate = o.gate_h + o.max_speed_h * np.abs(dt)
        keep = dist <= gate
        if keep.any():
            j, dt, dist, gate = j[keep], dt[keep], dist[keep], gate[keep]
            add(i, j, ido.link_cost * dist / gate + ido.gap_cost * np.maximum(dt, 0.0) - w[j])
    s, d, cst, cp = (np.concatenate(v) for v in (src, dst, cost, cap))
    e = np.arange(len(s))
    n_nodes = n + m
    into, out = d >= 0, s >= 0
    cons = coo_matrix((np.r_[np.ones(into.sum()), -np.ones(out.sum())],
                       (np.r_[d[into], s[out]], np.r_[e[into], e[out]])), shape=(n_nodes, len(e))).tocsr()
    into_t = (d >= 0) & (d < n)
    one = coo_matrix((np.ones(into_t.sum()), (d[into_t], e[into_t])), shape=(n, len(e))).tocsr()
    res = linprog(cst, A_ub=one, b_ub=np.ones(n), A_eq=cons, b_eq=np.zeros(n_nodes),
                  bounds=np.column_stack([np.zeros(len(e)), cp]), method="highs")
    if res.x is None:
        return []
    flow = np.round(res.x).astype(int)
    nxt: dict[int, list[int]] = {}
    for a, b, f in zip(s[flow > 0], d[flow > 0], flow[flow > 0]):
        nxt.setdefault(int(a), []).extend([int(b)] * int(f))
    chains: list[list[int]] = []
    ids = spans["id"].to_numpy()
    for _ in range(len(nxt.get(SRC, []))):
        node, chain = nxt[SRC].pop(), []
        while node != SINK and nxt.get(node):
            if node < n:
                chain.append(int(ids[node]))
            node = nxt[node].pop()
        if node != SINK and 0 <= node < n:
            chain.append(int(ids[node]))
        if chain:
            chains.append(chain)
    wsum = dict(zip(spans["id"], w))
    return sorted(chains, key=lambda ch: -sum(wsum[i] for i in ch))


# =============================================================================
# 2. どちらがどちらか
# =============================================================================
def _dense(rows: pd.DataFrame, frames: np.ndarray) -> dict[str, np.ndarray]:
    """1 本の並びの検出を、コマの並び ``frames`` にそろえた配列（ない所は NaN）。"""
    pos = np.searchsorted(frames, rows["frame"].to_numpy())
    out = {}
    for c in ("u", "v", "x0", "y0", "x1", "y1", "y"):
        a = np.full(len(frames), np.nan)
        a[pos] = rows[c].to_numpy(float)
        out[c] = a
    out["h"] = out["y1"] - out["y0"]
    out["present"] = np.isfinite(out["u"])
    out["tid"] = np.full(len(frames), -1)
    if "tid" in rows:
        out["tid"][pos] = rows["tid"].to_numpy()
    app = np.full((len(frames), 0), np.nan)
    if "app" in rows:
        f = decode_appearance(rows["app"])
        app = np.full((len(frames), f.shape[1]), np.nan)
        app[pos] = np.sqrt(f)                     # 割合の平方根（少ない色の差も効くように）
    out["app"] = app
    return out


def _pair_distance(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """2 人の足元の距離（枠の高さの平均が単位）と枠の IoU（どちらかがいないコマは NaN・0）。"""
    with np.errstate(invalid="ignore"):
        dist = np.hypot(a["u"] - b["u"], a["v"] - b["v"]) / np.maximum((a["h"] + b["h"]) / 2, 1.0)
        iw = np.minimum(a["x1"], b["x1"]) - np.maximum(a["x0"], b["x0"])
        ih = np.minimum(a["y1"], b["y1"]) - np.maximum(a["y0"], b["y0"])
        inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
        union = (a["x1"] - a["x0"]) * a["h"] + (b["x1"] - b["x0"]) * b["h"] - inter
        iou = np.where(union > 0, inter / union, 0.0)
    return dist, np.nan_to_num(iou)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """真が続く区間の [始まり, 終わり) の並び。"""
    d = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def swap_events(a: dict[str, np.ndarray], b: dict[str, np.ndarray], t: np.ndarray, ido: IdOptions) -> np.ndarray:
    """2 本の並びが入れ替わりうるコマ（真偽）。重なっている・片方が相手の近くで消えた／現れた・長く見えなかった所。"""
    n = len(t)
    dist, iou = _pair_distance(a, b)
    with np.errstate(invalid="ignore"):
        ev = (dist < ido.contact_h) | (iou > ido.contact_iou)
    dt = float(np.median(np.diff(t))) if n > 1 else 1.0
    for me, other in ((a, b), (b, a)):
        seen = np.flatnonzero(me["present"])
        for i0, i1 in zip(seen[:-1], seen[1:]):
            if i1 - i0 < 2:
                continue
            near = False
            for i in (i0, i1):
                # 消えた・現れたコマの前後で相手がいちばん近かった所
                lo, hi = max(0, i - 2), min(n, i + 3)
                with np.errstate(invalid="ignore"):
                    dd = np.hypot(me["u"][i] - other["u"][lo:hi], me["v"][i] - other["v"][lo:hi]) \
                        / np.maximum((me["h"][i] + other["h"][lo:hi]) / 2, 1.0)
                near |= bool(np.isfinite(dd).any() and np.nanmin(dd) < ido.merge_h)
            if near or t[i1] - t[i0] > ido.link_gap_s:
                ev[i0:i1 + 1] = True
    pad = max(1, int(round(ido.pad_s / dt)))
    for i0, i1 in _runs(ev):
        ev[max(0, i0 - pad):min(n, i1 + pad)] = True
    join = int(round(ido.join_s / dt))
    for i0, i1 in _runs(~ev):
        if 0 < i0 and i1 < n and i1 - i0 <= join:
            ev[i0:i1] = True
    return ev


def _edge_state(s: dict[str, np.ndarray], t: np.ndarray, i0: int, i1: int, tail: bool, ido: IdOptions
                ) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """区間 [i0, i1) の終わり（``tail``）・始まりの、足元の位置・速さ [px/s]・枠の高さ・その時刻。いなければ None。"""
    idx = np.flatnonzero(s["present"][i0:i1]) + i0
    if not len(idx):
        return None
    ref = t[idx[-1]] if tail else t[idx[0]]
    near = idx[np.abs(t[idx] - ref) <= ido.vel_window_s]
    edge = near[-3:] if tail else near[:3]
    pos = np.array([np.median(s["u"][edge]), np.median(s["v"][edge])])
    vel = np.zeros(2)
    if len(near) >= 3 and t[near[-1]] > t[near[0]]:
        vel = np.array([np.polyfit(t[near], s[c][near], 1)[0] for c in ("u", "v")])
    return pos, vel, float(np.median(s["h"][edge])), float(np.median(t[edge]))


def _motion_logodds(a: dict[str, np.ndarray], b: dict[str, np.ndarray], t: np.ndarray, seg0: tuple[int, int],
                    seg1: tuple[int, int], ido: IdOptions) -> float:
    """区切りをはさんだ 2 区間で「そのまま」が「入れ替え」よりどれだけありそうか（対数の比。＋ならそのまま）。

    区切りの前の位置と速さから後の位置を見越したずれと、後の位置と速さから前の位置を見越したずれの両方で比べる
    （区切りの間に走り出した人は、前からだけ見越すと止まっていた人と取り違える）。
    """
    before = [_edge_state(s, t, *seg0, True, ido) for s in (a, b)]
    after = [_edge_state(s, t, *seg1, False, ido) for s in (a, b)]
    num = 0.0
    used = False
    for src, dst, sign in ((before, after, 1.0), (after, before, -1.0)):
        for k in (0, 1):
            if src[k] is None or dst[0] is None or dst[1] is None:
                continue
            pos, vel, h, ts = src[k]
            dur = max(0.0, sign * (dst[k][3] - ts))
            pred = pos + sign * vel * min(dur, ido.vel_horizon_s)
            sig = (ido.motion_sigma_h + ido.motion_speed_h * dur) * max(h, 1.0)
            stay = float(np.sum((pred - dst[k][0]) ** 2))
            cross = float(np.sum((pred - dst[1 - k][0]) ** 2))
            num += (cross - stay) / (2 * sig * sig) / 2     # 前向き・後ろ向きは同じ動きを 2 度見ているので半分ずつ
            used = True
    # 区切りの間も同じ軌跡で追えていた人がいれば、その人は入れ替わっていない（軌跡は ``split_impure`` で
    # 飛んだ所を切ってある。区切りの前後の位置だけだと、すれ違いは入れ替わったように見える）
    same = 0.0
    for s in (a, b):
        i0 = np.flatnonzero(s["present"][seg0[0]:seg0[1]])
        i1 = np.flatnonzero(s["present"][seg1[0]:seg1[1]])
        if len(i0) and len(i1) and s["tid"][seg0[0] + i0[-1]] == s["tid"][seg1[0] + i1[0]] >= 0:
            same = ido.same_track
    if not used:
        return ido.stay_prior + same
    return float(np.clip(num, -ido.max_logodds, ido.max_logodds)) + ido.stay_prior + same


def _appearance_scores(feats: list[list[np.ndarray | None]], ido: IdOptions) -> tuple[np.ndarray, dict[str, Any]]:
    """区間ごとの色（``feats[区間][並び]``）から、区間ごとの「そのまま − 入れ替え」の対数の比と、色の分かれ具合。

    2 人とも色の出た区間の差（並び 0 − 並び 1）は、番号が合っていれば同じ向き、入れ替わっていれば逆の向きになる。
    差のいちばん大きい向き（主成分）に色を写し、その値が 2 つの山にはっきり分かれるときだけ使う。
    """
    n = len(feats)
    zero = np.zeros(n)
    both = [k for k in range(n) if feats[k][0] is not None and feats[k][1] is not None]
    if len(both) < ido.app_min_segments:
        return zero, {"used": False, "n_segments": len(both)}
    min_sep = ido.app_min_sep * (1 + 4 / len(both))
    d = np.array([feats[k][0] - feats[k][1] for k in both])
    v = np.linalg.svd(d, full_matrices=False)[2][0]
    c = np.array([[np.nan if f is None else float(f @ v) for f in fk] for fk in feats])   # (区間, 並び)
    state = np.where(np.nan_to_num(c[:, 0] - c[:, 1]) >= 0, 0, 1)                      # 0：並び 0 が値の大きい人
    score = zero
    info: dict[str, Any] = {"used": False, "n_segments": len(both)}
    for _ in range(3):
        hi = np.r_[c[state == 0, 0], c[state == 1, 1]]
        lo = np.r_[c[state == 0, 1], c[state == 1, 0]]
        hi, lo = hi[np.isfinite(hi)], lo[np.isfinite(lo)]
        if len(hi) < 2 or len(lo) < 2:
            return zero, info
        mu_hi, mu_lo = float(np.median(hi)), float(np.median(lo))
        sig = 1.4826 * float(np.median(np.abs(np.r_[hi - mu_hi, lo - mu_lo]))) + 1e-4
        sep = (mu_hi - mu_lo) / sig
        info = {"used": bool(sep >= min_sep), "n_segments": len(both), "separation": round(sep, 2)}
        if sep < min_sep:
            return zero, info

        def ll(x: np.ndarray, mu: float) -> np.ndarray:
            return np.where(np.isfinite(x), -((x - mu) ** 2) / (2 * sig * sig), 0.0)

        score = (ll(c[:, 0], mu_hi) + ll(c[:, 1], mu_lo)) - (ll(c[:, 0], mu_lo) + ll(c[:, 1], mu_hi))
        score = np.clip(score, -ido.app_cap, ido.app_cap)
        new = np.where(score >= 0, 0, 1)
        if (new == state).all():
            break
        state = new
    return score, info


def _viterbi(unary: np.ndarray, stay: np.ndarray) -> np.ndarray:
    """区間ごとの状態（0：そのまま、1：入れ替え）。``unary`` は状態 0 − 状態 1 の対数の比、``stay`` は隣どうしが同じ状態である分。"""
    n = len(unary)
    em = np.column_stack([unary / 2, -unary / 2])
    score = em[0].copy()
    back = np.zeros((n, 2), int)
    for k in range(1, n):
        tr = np.array([[stay[k - 1] / 2, -stay[k - 1] / 2], [-stay[k - 1] / 2, stay[k - 1] / 2]])
        cand = score[:, None] + tr
        back[k] = np.argmax(cand, axis=0)
        score = cand[back[k], [0, 1]] + em[k]
    st = int(np.argmax(score))
    path = [st]
    for k in range(n - 1, 0, -1):
        st = int(back[k, st])
        path.append(st)
    return np.array(path[::-1])


def resolve_identities(slots: list[pd.DataFrame], frame_t: pd.Series, ido: IdOptions | None = None
                       ) -> tuple[list[np.ndarray], dict[str, Any]]:
    """2 本の並び（``choose_chains`` の検出。1 コマ 1 行）の行ごとに、どちらの人か（0・1）を決める。

    返すのは並びごとの番号の配列（``slots[k]`` の行の順）と、確かめ用の数（入れ替わりうる所の数・入れ替えた数・色を使ったか）。
    """
    ido = ido or IdOptions()
    frames = frame_t.index.to_numpy()
    t = frame_t.to_numpy(float)
    a, b = (_dense(s, frames) for s in slots)
    ev = swap_events(a, b, t, ido)
    any_present = a["present"] | b["present"]
    segs = [(i0, i1) for i0, i1 in _runs(~ev) if any_present[i0:i1].any()]
    info: dict[str, Any] = {"n_events": max(0, len(segs) - 1), "n_swaps": 0, "appearance": {"used": False}}
    state = np.zeros(max(1, len(segs)), int)
    if len(segs) > 1:
        dist, iou = _pair_distance(a, b)
        with np.errstate(invalid="ignore"):
            clean = ~ev & (iou < 0.05)                 # 2 人の枠が重なっていないコマの色だけを使う
        feats: list[list[np.ndarray | None]] = []
        for i0, i1 in segs:
            fk: list[np.ndarray | None] = []
            for s in (a, b):
                ok = np.flatnonzero(clean[i0:i1] & s["present"][i0:i1]) + i0
                ok = ok[np.isfinite(s["app"][ok]).all(axis=1)] if s["app"].shape[1] else ok[:0]
                fk.append(np.median(s["app"][ok], axis=0) if len(ok) >= ido.min_clean else None)
            feats.append(fk)
        unary, info["appearance"] = _appearance_scores(feats, ido)
        stay = np.array([_motion_logodds(a, b, t, segs[k], segs[k + 1], ido) for k in range(len(segs) - 1)])
        state = _viterbi(unary, stay)
        info["n_swaps"] = int(np.sum(state[1:] != state[:-1]))
    # コマごとの状態：区間の中はその区間の状態、区切りの中は途中のコマで切り替える
    per_frame = np.zeros(len(frames), int)
    if len(segs):
        per_frame[:segs[0][0]] = state[0]
        for k, (i0, i1) in enumerate(segs):
            per_frame[i0:i1] = state[k]
            if k + 1 == len(segs):
                per_frame[i1:] = state[k]
                break
            j0, j1 = i1, segs[k + 1][0]
            cut = (j0 + j1) // 2
            if state[k] != state[k + 1] and j1 > j0:
                # 切り替えるコマ：どちらかが見えていなかった間の真ん中。2 人とも見えていたなら、いちばん近づいたコマ
                gone = np.flatnonzero(~(a["present"][j0:j1] & b["present"][j0:j1]))
                d, _ = _pair_distance(a, b)
                if len(gone):
                    cut = j0 + int(gone[len(gone) // 2])
                elif np.isfinite(d[j0:j1]).any():
                    cut = j0 + int(np.nanargmin(d[j0:j1]))
            per_frame[j0:cut] = state[k]
            per_frame[cut:j1] = state[k + 1]
    # 0 番は、最初に 2 人とも写った区間で y が小さい方
    flip = 0
    for i0, i1 in segs:
        m = a["present"][i0:i1] & b["present"][i0:i1]
        if m.sum() >= 3:
            ya = np.where(per_frame[i0:i1] == 0, a["y"][i0:i1], b["y"][i0:i1])[m]
            yb = np.where(per_frame[i0:i1] == 0, b["y"][i0:i1], a["y"][i0:i1])[m]
            flip = int(np.median(ya) > np.median(yb))
            break
    ident = []
    for k, s in enumerate(slots):
        st = per_frame[np.searchsorted(frames, s["frame"].to_numpy())]
        ident.append((st ^ k) ^ flip)
    return ident, info


# =============================================================================
# 1 側の 2 人・両側
# =============================================================================
def track_pair(cand: pd.DataFrame, frame_t: pd.Series, o: PlayerOptions, ido: IdOptions | None = None
               ) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    """1 側の選手とみなす範囲の検出（``weight`` つき）から、2 人の検出の表（補う前。0 番・1 番の順）と確かめ用の数。"""
    ido = ido or IdOptions()
    info: dict[str, Any] = {"n_chains": 0}
    if cand.empty:
        return [], info
    cand = cand.reset_index(drop=True)
    linked = link_with_motion(cand, o, ido) if ido.predict_s > 0 else link_tracklets(cand, o)
    cand["tid"] = split_impure(cand, linked, ido)
    chains = choose_chains(cand, o, ido, 2)
    info["n_chains"] = len(chains)
    # 同じコマに 2 つ残ったとき（軌跡の境目）は重みの大きい方
    slots = [cand[cand["tid"].isin(ch)].sort_values("weight", ascending=False).drop_duplicates("frame")
             .sort_values("frame") for ch in chains]
    if len(slots) < 2:
        return slots, info
    ident, id_info = resolve_identities(slots, frame_t, ido)
    info.update(id_info)
    rows = pd.concat([s.assign(ident=i) for s, i in zip(slots, ident)], ignore_index=True)
    return [rows[rows["ident"] == k].drop(columns="ident").sort_values("frame") for k in (0, 1)], info


def looks_like_pair(tracks: list[pd.DataFrame], frame_t: pd.Series, side: str, cfg: dict[str, Any],
                    o: PlayerOptions, ido: IdOptions | None = None) -> tuple[bool, dict[str, Any]]:
    """2 人目も選手らしいか（自動でダブルスとみなすか）と、その根拠の数。"""
    ido = ido or IdOptions()
    if len(tracks) < 2 or any(tr.empty for tr in tracks):
        return False, {"reason": "2 人目がいない"}
    n = max(1, len(frame_t))
    L = float(cfg["court"]["half_length_m"])
    W = float(cfg["court"]["half_width_doubles_m"])
    cover, inside, rng = [], [], []
    for tr in tracks:
        sx = tr["x"].to_numpy(float) * (1.0 if side == "o" else -1.0)
        y = tr["y"].to_numpy(float)
        cover.append(len(tr) / n)
        inside.append(float(np.mean((sx >= -o.net_margin_m) & (sx <= L - ido.auto_inset_m)
                                    & (np.abs(y) <= W - ido.auto_inset_m))))
        rng.append(float(np.subtract(*np.percentile(sx, [90, 10])) + np.subtract(*np.percentile(y, [90, 10]))))
    f0, f1 = (set(tr["frame"]) for tr in tracks)
    both = len(f0 & f1) / max(1, len(f0 | f1))
    stats = {"cover": [round(v, 3) for v in cover], "inside": [round(v, 3) for v in inside],
             "range_m": [round(v, 2) for v in rng], "both": round(both, 3)}
    long_enough = float(frame_t.iloc[-1] - frame_t.iloc[0]) >= 4.0
    ok = (min(cover) >= ido.auto_min_cover and both >= ido.auto_min_both and min(inside) >= ido.auto_min_inside
          and (not long_enough or min(rng) >= ido.auto_min_range_m))
    return bool(ok), stats


def track_sides(people: pd.DataFrame, frame_t: pd.Series, cfg: dict[str, Any], o: PlayerOptions | None = None,
                ido: IdOptions | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """側ごとに、ダブルス（2 人）かシングルス（1 人）で選手を追う。``players.csv`` の形の表と、側ごとの人数・確かめ用の数。

    ``o.per_side`` が 2 なら両側とも 2 人、0 なら側ごとに ``looks_like_pair`` で決める（1 人の側はシングルスの追跡と同じ結果）。
    """
    o = o or PlayerOptions()
    ido = ido or IdOptions()
    out = []
    info: dict[str, Any] = {"per_side": {}, "sides": {}}
    for side in SIDES:
        region, weight = side_region(people, side, cfg, o)
        cand = people[region].assign(weight=weight[region]).reset_index(drop=True)
        tracks, side_info = track_pair(cand, frame_t, o, ido) if o.per_side != 1 else ([], {})
        if o.per_side == 0:
            pair, side_info["auto"] = looks_like_pair(tracks, frame_t, side, cfg, o, ido)
            if not pair:
                tracks = []
        info["sides"][side] = side_info
        if not tracks:
            single = _track_singles(people, frame_t, cfg, o, [side])
            info["per_side"][side] = int(not single.empty)
            if not single.empty:
                out.append(single)
            continue
        info["per_side"][side] = len(tracks)
        for k, tr in enumerate(tracks):
            tr = tr.drop(columns=[c for c in ("tid", "weight", "app") if c in tr])
            out.append(_fill_gaps(tr, frame_t, o).assign(side=side, player=k))
    return _player_table(out), info
