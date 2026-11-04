"""ラリーの区切りに選手の動きを足したチェック：ラリーの間の打球を入れた合成ラリーで、選手の動きを使う前と後を比べる（C）。

ラリーの間に、床に落ちたシャトルを拾って相手へ返す打球と、となりのコートで打たれたシャトルを入れ
（``synth_rally_players``）、選手の足元の動き（ラリーの中は素早く、ラリーの間は歩いてサーブの位置へ行き、
サーブの前は止まる）を ``players.csv`` の形で足す。

1. 正解のショット（ラリーの間の打球も入れ、打った時刻・着地の時刻に少しずれを足したもの）を区切る
   （``rallies.player_benchmark`` の truth）。
2. 横から撮った合成を 30 fps で観測し、動画全体の検出から打球を見つけ（打球音あり）、軌道を当てはめてから区切る（同 fit）。

選手の動きを使った後のラリーの適合率が ``MIN_PRECISION`` を下回る、ラリーではないラリーが減らない、
ラリーの再現率（ちょうど分かれた正解のラリーの割合）が使う前より ``MAX_RECALL_DROP`` より下がる、
または区切りの再現率・適合率が ``MIN_BOUNDARY`` を下回ったら失敗にする。

使い方::

    .venv\\Scripts\\python.exe agent\\checks\\check_rally_players.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SEED = 0
N_TRUTH_SESSIONS, N_FIT_SESSIONS, N_RALLIES = 4, 1, 5
# 作った時点の値（truth 4×5 ラリー：ラリーの適合率 0.571 → 1.000、fit 1×5 ラリー：0.556 → 1.000。
# どちらも再現率 1.000 のまま）から、たまたまの違いでは落ちないように余裕をとった基準
MIN_PRECISION = 0.9
MAX_RECALL_DROP = 0.05
MIN_BOUNDARY = 0.9


def run(ctx: dict[str, Any] | None = None) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    from smash_speed.rallies import player_benchmark

    workers = max(1, min(8, (os.cpu_count() or 2) - 1))
    bad, parts, metrics = [], [], {}
    for mode, n in (("truth", N_TRUTH_SESSIONS), ("fit", N_FIT_SESSIONS)):
        r = player_benchmark(n, N_RALLIES, 30.0, SEED, mode, workers=workers)
        before, after = r["before"], r["after"]
        metrics[mode] = {name: {k: round(v, 3) if isinstance(v, float) else v for k, v in r[name].items()}
                         for name in ("before", "after")}
        metrics[mode]["n_extra"] = r["n_extra"]
        parts.append(f"{mode} ラリーの適合率 {before['rally_precision']:.3f} → {after['rally_precision']:.3f}"
                     f"・再現率 {before['exact']:.3f} → {after['exact']:.3f}"
                     f"（ラリーではないもの {before['n_false']} → {after['n_false']} 個）")
        if after["rally_precision"] < MIN_PRECISION:
            bad.append(f"{mode} のラリーの適合率 {after['rally_precision']:.3f}（基準 {MIN_PRECISION}）")
        if before["n_false"] and after["n_false"] >= before["n_false"]:
            bad.append(f"{mode} でラリーではないラリーが減らない（{before['n_false']} → {after['n_false']}）")
        if after["exact"] < before["exact"] - MAX_RECALL_DROP:
            bad.append(f"{mode} のラリーの再現率が下がった（{before['exact']:.3f} → {after['exact']:.3f}）")
        for k in ("boundary_recall", "boundary_precision"):
            if after[k] < MIN_BOUNDARY:
                bad.append(f"{mode} の {k} {after[k]:.3f}（基準 {MIN_BOUNDARY}）")
    ok = not bad
    summary = "ラリーの区切りと選手の動き：" + ("、".join(parts) if ok else "基準を下回った：" + "、".join(bad))
    return {"ok": ok, "summary": summary, "metrics": metrics}


if __name__ == "__main__":
    if (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    res = run()
    print(("OK " if res["ok"] else "NG ") + res["summary"])
    sys.exit(0 if res["ok"] else 1)
