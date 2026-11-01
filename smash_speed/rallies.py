"""ラリーの区切り（ROADMAP C1）：打球の間隔・シャトルの着地・シャトルが飛んでいるかから、ラリーの始まりと終わりを出す。

ショット（``shots.Shot``。打った時刻・当てはめた軌道つき）の並びの、となり合う 2 本の間ごとに、次の順で決める。

1. 間が ``hard_gap_s`` より空いた → 区切る（「間隔」）。
2. 前のショットの軌道が、次の打球より ``land_margin_s`` 以上前に床に着く → シャトルは誰にも打たれずに落ちたので
   区切る（「着地」）。打ち返されたショットは、床に着く前（高さ 0.2 m 以上）に打たれるので、着地はその後になる。
   初速が出なかった（当てはめに失敗した）ショットの軌道は使わない。
3. 間が ``max_gap_s`` より空き、その間シャトルがほとんど見えていない（見えたコマが ``flying_frac`` 未満）→ 区切る
   （「間隔」）。見えていれば、打球を見落としただけでラリーは続いているとみなす。
4. それ以外は同じラリー。

ラリーの終わりの時刻は、最後のショットの着地の時刻（なければ最後にシャトルが見えた時刻、それもなければ打った時刻）。
最後のショットの着地点が分かれば、終わり方（``Rally.end_reason``：in・out・net）と得点した側（``winner_side``）を
決める。相手のコートの中に落ちれば打った側の得点、外・自分の側（ネットを越えない）なら相手の得点。
線の近く（``line_margin_m`` 以内）に落ちたものは、メモに「線の近く」と書く（当てはめの誤差で入・出が変わりうる）。
得点（``score_h``・``score_o``）は、動画の始まりの得点が分からないので空欄のまま。

打球音は、打球を見つけるとき（``hits.find_hits``）にもう使っているので、ここでは使わない。
選手の動き（打った後に選手が止まる・サーブの位置に戻る）は、選手の検出（ROADMAP D1）ができてから足す。

``python -m smash_speed.rallies --bench`` で、続けて打った合成ラリー（``synth_rally.make_session``）での精度を、
打球の間隔だけで分ける今までのやり方（``shots.make_rallies``）と比べる。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from smash_speed.shots import Rally, Shot, side_of

# 区切った理由（``Rally.note`` に書く）
REASON_GAP = "間隔"
REASON_LANDED = "着地"
REASON_END = "動画の終わり"


@dataclass
class RallyOptions:
    max_gap_s: float = 3.0       # 打球の間がこれより空き、その間シャトルが飛んでいなければ区切る [s]
    hard_gap_s: float = 8.0      # これより空いたら、シャトルが見えていても区切る [s]
    land_margin_s: float = 0.3   # 着地がこれより前（次の打球から見て）なら、床に落ちたとみなす [s]
    flying_frac: float = 0.4     # 打球の間でシャトルが見えたコマの割合がこれ以上なら、飛んでいたとみなす
    singles: bool = False        # 入・出をシングルスのサイドラインで決める（偽ならダブルス）
    line_margin_m: float = 0.1   # 着地点が線からこれより近ければ「線の近く」とメモする [m]


@dataclass
class Boundary:
    """となり合う 2 本のショットの間の判定。"""

    t_prev: float
    t_next: float
    split: bool
    reason: str = ""                 # REASON_GAP | REASON_LANDED | ""
    t_land: float | None = None      # 前のショットの軌道が床に着く時刻
    flying_frac: float | None = None  # 間でシャトルが見えたコマの割合（検出がなければ None）


def _land_time(s: Shot) -> float | None:
    """当てはめに成功したショットの、軌道が床に着く時刻。"""
    if s.speed_kmh is None or s.t_land_s is None or not np.isfinite(s.t_land_s):
        return None
    return float(s.t_land_s)


def _visible_times(det: pd.DataFrame | None) -> np.ndarray:
    if det is None or not len(det):
        return np.zeros(0)
    d = det[det.visible.astype(bool)]
    return np.sort(d.timestamp.to_numpy(float))


def _frame_dt(det: pd.DataFrame | None) -> float | None:
    if det is None or len(det) < 2:
        return None
    dt = np.diff(np.sort(det.timestamp.to_numpy(float)))
    dt = dt[dt > 0]
    return float(np.median(dt)) if len(dt) else None


def visible_fraction(vis_t: np.ndarray, dt: float | None, t0: float, t1: float, pad_s: float = 0.05) -> float | None:
    """``t0``〜``t1`` の間（前後 ``pad_s`` を除く）で、シャトルが見えたコマの割合。分からなければ None。"""
    a, b = t0 + pad_s, t1 - pad_s
    if dt is None or b <= a:
        return None
    n = np.count_nonzero((vis_t > a) & (vis_t < b))
    return float(min(1.0, n / max((b - a) / dt, 1.0)))


def judge_gap(prev: Shot, nxt: Shot, vis_t: np.ndarray, dt: float | None, o: RallyOptions) -> Boundary:
    """ショット ``prev`` と次のショット ``nxt`` の間で、ラリーが区切れるか。"""
    gap = nxt.t_hit_s - prev.t_hit_s
    t_land = _land_time(prev)
    frac = visible_fraction(vis_t, dt, prev.t_hit_s, nxt.t_hit_s)
    b = Boundary(prev.t_hit_s, nxt.t_hit_s, False, "", t_land, frac)
    if gap > o.hard_gap_s:
        b.split, b.reason = True, REASON_GAP
    elif t_land is not None and t_land < nxt.t_hit_s - o.land_margin_s:
        b.split, b.reason = True, REASON_LANDED
    elif gap > o.max_gap_s and (frac is None or frac < o.flying_frac):
        b.split, b.reason = True, REASON_GAP
    return b


def rally_outcome(last: Shot, o: RallyOptions, court: dict[str, Any] | None = None) -> tuple[str, str, str]:
    """最後のショットの着地点から、終わり方（in | out | net | ""）・得点した側・メモ。"""
    if last.land_x_m is None or last.land_y_m is None or _land_time(last) is None:
        return "", "", ""
    c = court or {}
    half_len = c.get("half_length_m", 6.70)
    half_w = c.get("half_width_singles_m", 2.59) if o.singles else c.get("half_width_doubles_m", 3.05)
    hitter = last.side or side_of(last.hit_x_m)
    if not hitter:
        return "", "", ""
    other = "o" if hitter == "h" else "h"
    x, y = float(last.land_x_m), float(last.land_y_m)
    if side_of(x) == hitter:
        return "net", other, ""
    near = min(abs(abs(x) - half_len), abs(abs(y) - half_w)) <= o.line_margin_m
    note = "線の近く" if near else ""
    if abs(x) <= half_len and abs(y) <= half_w:
        return "in", hitter, note
    return "out", other, note


def split_rallies(shots: list[Shot], det: pd.DataFrame | None = None, opts: RallyOptions | None = None,
                  court: dict[str, Any] | None = None) -> list[Rally]:
    """ショットをラリーに分け、ラリーの番号と順番をショットに付ける（``shots`` の ``rally_id``・``index_in_rally`` を書き換える）。

    Args:
        shots: ショット（順番は問わない）。着地の時刻（``t_land_s``）と着地点があれば、着地で区切り、終わり方を決める
        det: 動画全体のシャトルの検出結果（列 ``timestamp, visible``）。打球の間でシャトルが飛んでいたかに使う
        court: 設定の ``court``（線の位置。省略時は BWF の寸法）

    Returns:
        時刻の順のラリー。``note`` に終わった理由（間隔・着地・動画の終わり）を書く
    """
    o = opts or RallyOptions()
    order = sorted(shots, key=lambda s: s.t_hit_s)
    vis_t, dt = _visible_times(det), _frame_dt(det)
    bounds = [judge_gap(a, b, vis_t, dt, o) for a, b in zip(order[:-1], order[1:])]
    rallies: list[Rally] = []
    for i, s in enumerate(order):
        if i == 0 or bounds[i - 1].split:
            rallies.append(Rally(rally_id=len(rallies) + 1, t_start_s=s.t_hit_s))
        r = rallies[-1]
        r.n_shots += 1
        s.rally_id, s.index_in_rally = r.rally_id, r.n_shots
        if i < len(bounds) and not bounds[i].split:
            continue
        reason, t_limit = (bounds[i].reason, order[i + 1].t_hit_s) if i < len(bounds) else (REASON_END, np.inf)
        # ラリーの終わり：着地 → 最後に見えた時刻 → 打った時刻
        t_land = _land_time(s)
        landed = t_land is not None and t_land < t_limit
        seen = vis_t[(vis_t > s.t_hit_s) & (vis_t < t_limit)]
        r.t_end_s = t_land if landed else float(seen[-1]) if len(seen) else s.t_hit_s
        r.end_reason, r.winner_side, note = rally_outcome(s, o, court) if landed else ("", "", "")
        r.note = "・".join(x for x in (f"終わり：{reason}", note) if x)
    return rallies


# =============================================================================
# 正解との比べ方と、合成ラリーでの測り方
# =============================================================================
def match_to_truth(pred: list[Shot], truth: list[Shot], tol_s: float = 0.05) -> list[int | None]:
    """推定したショットごとに、時刻の差 ``tol_s`` 以内の正解のショットの番号（1 対 1。なければ None）。"""
    pairs = sorted((abs(p.t_hit_s - t.t_hit_s), i, j) for i, p in enumerate(pred) for j, t in enumerate(truth)
                   if abs(p.t_hit_s - t.t_hit_s) <= tol_s)
    out: list[int | None] = [None] * len(pred)
    used: set[int] = set()
    for _, i, j in pairs:
        if out[i] is None and j not in used:
            out[i] = j
            used.add(j)
    return out


def rally_scores(pred: list[Shot], pred_rallies: list[Rally], truth: list[Shot],
                 true_rallies: list[Rally], tol_s: float = 0.05) -> dict[str, Any]:
    """ラリーの区切りの精度。正解と組になった打球だけで数える（打球の見落とし・間違いは ``hits`` の精度）。

    - ``boundary_recall``・``boundary_precision``：となり合う 2 本（どちらも正解と組になったもの）の間の区切りの再現率・適合率
    - ``exact``：正解のラリーのうち、組になった打球がちょうど 1 つの推定したラリーにまとまったものの割合
    - ``end_err_s``：ちょうどまとまったラリーの終わりの時刻のずれの中央値 [s]
    - ``outcome_acc``：ちょうどまとまったラリーのうち、得点した側が正解と同じものの割合（推定が空欄なら数えない）
    - ``by_reason``：正解の終わり方（in・out・net）ごとの［終わり方と得点した側が両方とも合ったラリーの数, ちょうどまとまったラリーの数］
    """
    m = match_to_truth(pred, truth, tol_s)
    seq = sorted((p.t_hit_s, p.rally_id, truth[j].rally_id) for p, j in zip(pred, m) if j is not None)
    tp = fp = fn = 0
    for (_, pa, ta), (_, pb, tb) in zip(seq[:-1], seq[1:]):
        true_split, pred_split = ta != tb, pa != pb
        tp += true_split and pred_split
        fp += pred_split and not true_split
        fn += true_split and not pred_split
    by_true: dict[int, set[int]] = {}
    by_pred: dict[int, set[int]] = {}
    for _, pa, ta in seq:
        by_true.setdefault(ta, set()).add(pa)
        by_pred.setdefault(pa, set()).add(ta)
    pr = {r.rally_id: r for r in pred_rallies}
    tr = {r.rally_id: r for r in true_rallies}
    exact, end_err, outcome = [], [], []
    by_reason: dict[str, list[int]] = {}
    for ta in tr:
        ps = by_true.get(ta, set())
        ok = len(ps) == 1 and by_pred[next(iter(ps))] == {ta}
        exact.append(ok)
        if ok:
            p = pr[next(iter(ps))]
            if p.t_end_s is not None and tr[ta].t_end_s is not None:
                end_err.append(abs(p.t_end_s - tr[ta].t_end_s))
            if p.winner_side:
                outcome.append(p.winner_side == tr[ta].winner_side)
            c = by_reason.setdefault(tr[ta].end_reason, [0, 0])
            c[0] += p.end_reason == tr[ta].end_reason and p.winner_side == tr[ta].winner_side
            c[1] += 1
    return {"n_true": len(tr), "n_pred": len(pred_rallies), "boundary_tp": tp, "boundary_fp": fp, "boundary_fn": fn,
            "boundary_recall": tp / max(tp + fn, 1), "boundary_precision": tp / max(tp + fp, 1),
            "exact": float(np.mean(exact)) if exact else 1.0,
            "end_err_s": float(np.median(end_err)) if end_err else None,
            "outcome_acc": float(np.mean(outcome)) if outcome else None, "n_outcome": len(outcome),
            "by_reason": by_reason}


def synthetic_benchmark(n_sessions: int = 4, n_rallies: int = 6, fps: float = 30.0, seed: int = 0,
                        mode: str = "fit", rest_s: tuple[float, float] = (1.0, 8.0),
                        opts: RallyOptions | None = None, cfg: dict[str, Any] | None = None,
                        workers: int = 1, ends: dict[str, float] | None = None) -> dict[str, Any]:
    """続けて打った合成ラリーで、ラリーの区切りの精度を測る（``rally_scores``）。

    Args:
        mode: "fit" は動画全体の検出から ``find_hits``（打球音あり）→ ``shot_list.analyze_hits``（軌道の当てはめ）を
            通したショットを区切る（実際の流れと同じ）。"truth" は正解のショット（打った時刻に 5 ms、着地の時刻に
            0.1 s のずれを足す）を区切る（速いので多くのラリーで測れる）
        rest_s: ラリーの間の休み（着地から次のサーブまで）の範囲 [s]。短い休みは間隔だけでは区切れない
        ends: ラリーの終わり方の割合（``synth_rally.make_session``。省略時はすべて相手のコートの中）

    Returns:
        ``new``（``split_rallies``）と ``gap``（``make_rallies``。打球の間隔だけ）の ``rally_scores`` を足し合わせたもの
    """
    from smash_speed.config import load_config
    from smash_speed.physics import PhysicsParams
    from smash_speed.shots import make_rallies
    from smash_speed.synth_rally import make_session, observe_session, rally_camera

    cfg = cfg or load_config()
    phys = PhysicsParams.from_config(cfg)
    camera = rally_camera(cfg)
    o = opts or RallyOptions()
    tot: dict[str, dict[str, list[Any]]] = {"new": {}, "gap": {}}
    n_shots = 0
    for k in range(n_sessions):
        rng = np.random.default_rng([seed, 7, k])
        session = make_session(rng, phys, n_rallies, rest_s=rest_s, ends=ends)
        truth: list[Shot] = []
        true_rallies: list[Rally] = []
        for i, r in enumerate(session, start=1):
            truth += r.to_shots(rally_id=i, phys=phys)
            true_rallies.append(r.to_rally(rally_id=i))
        det = observe_session(session, camera, fps, phys, np.random.default_rng([seed, 8, k]))
        if mode == "truth":
            noise = np.random.default_rng([seed, 9, k])
            pred = [Shot(t_hit_s=s.t_hit_s + noise.normal(0, 0.005), side=s.side, speed_kmh=s.speed_kmh,
                         hit_x_m=s.hit_x_m, land_x_m=s.land_x_m, land_y_m=s.land_y_m,
                         t_land_s=s.t_land_s + noise.normal(0, 0.1)) for s in truth]
        else:
            from smash_speed.hits import find_hits
            from smash_speed.shot_list import analyze_hits

            onsets = [(s.t_hit_s + float(np.linalg.norm(camera.center - s.hit_point)) / 343.0, 30.0) for s in truth]
            hits = find_hits(det, fps, onsets, camera=camera, rally_gap_s=o.max_gap_s)
            pred = analyze_hits(det, hits, camera, phys, cfg, o.max_gap_s, ci=False, workers=workers,
                                progress=lambda _: None)
        n_shots += len(truth)
        for name in tot:
            shots = [Shot(**vars(s)) for s in pred]
            rallies = split_rallies(shots, det, o, cfg["court"]) if name == "new" else \
                make_rallies(shots, max_gap_s=o.max_gap_s)
            for key, v in rally_scores(shots, rallies, truth, true_rallies).items():
                tot[name].setdefault(key, []).append(v)
    out: dict[str, Any] = {"n_sessions": n_sessions, "n_rallies": n_sessions * n_rallies, "n_shots": n_shots,
                           "fps": fps, "mode": mode}
    for name, d in tot.items():
        tp, fp, fn = (sum(d[f"boundary_{x}"]) for x in ("tp", "fp", "fn"))
        n_true = sum(d["n_true"])
        end_errs = [e for e in d["end_err_s"] if e is not None]
        n_out = sum(d["n_outcome"])
        by_reason: dict[str, list[int]] = {}
        for br in d["by_reason"]:
            for k, (ok, n) in br.items():
                c = by_reason.setdefault(k, [0, 0])
                c[0], c[1] = c[0] + ok, c[1] + n
        out[name] = {"boundary_recall": tp / max(tp + fn, 1), "boundary_precision": tp / max(tp + fp, 1),
                     "exact": float(np.dot(d["exact"], d["n_true"]) / max(n_true, 1)),
                     "n_pred": sum(d["n_pred"]),
                     "end_err_s": float(np.mean(end_errs)) if end_errs else None,
                     "outcome_acc": (sum(a * n for a, n in zip(d["outcome_acc"], d["n_outcome"]) if a is not None)
                                     / n_out) if n_out else None, "n_outcome": n_out, "by_reason": by_reason}
    return out


def main(argv: list[str] | None = None) -> None:
    import argparse
    import time

    ap = argparse.ArgumentParser(description="続けて打った合成ラリーで、ラリーの区切りの精度を測る")
    ap.add_argument("--bench", action="store_true", help="合成ラリーで測る")
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--rallies", type=int, default=6, help="1 つの動画のラリーの数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=float, nargs="+", default=[30.0])
    ap.add_argument("--mode", nargs="+", default=["truth", "fit"], choices=["truth", "fit"])
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    for fps in a.fps:
        for mode in a.mode:
            t0 = time.time()
            r = synthetic_benchmark(a.sessions, a.rallies, fps, a.seed, mode, workers=a.workers)
            for name, label in (("new", "着地・飛行も使う"), ("gap", "間隔だけ")):
                v = r[name]
                print(f"{fps:4.0f} fps {mode:5s} {label}：区切りの再現率 {v['boundary_recall']:.3f}、"
                      f"適合率 {v['boundary_precision']:.3f}、ちょうど分かれたラリー {v['exact']:.3f}"
                      f"（{r['n_rallies']} 個中、推定 {v['n_pred']} 個）、終わりの時刻のずれ "
                      + ("-" if v["end_err_s"] is None else f"{v['end_err_s']:.2f} s")
                      + "、得点した側 " + ("-" if v["outcome_acc"] is None
                                         else f"{v['outcome_acc']:.3f}（{v['n_outcome']} 個）"))
            print(f"  （{r['n_shots']} 本、{time.time() - t0:.0f} 秒）")


if __name__ == "__main__":
    import sys

    if (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    main()
