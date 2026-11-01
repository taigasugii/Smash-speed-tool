"""実動画に近い条件での打球の見つけ方のチェック：雑音・人・隣のコートを足した合成ラリーで、軌道の絞り込みと打球を測る。

``synth_clutter.hits_benchmark``：コートの角の後ろの低いカメラ（``corner_low``）を 30 fps で、検出の抜け（まとめて抜ける区間・
打つ瞬間に隠れる）、振ったラケット・走る選手の頭、隣のコートのシャトルと打球音、足音などを足した観測から、
``shot_list.shuttle_tracks`` → ``hits.find_hits`` を通す。当てはめはしない（時間がかかるので、初速・種類・ラリーの区切りは
``python -m smash_speed.synth_clutter --bench`` で測って JOURNAL に書く）。

雑音のない横の合成ラリー（``check_hits``）より低い所に基準を置く。今の値より悪くなったことを見つけるためのもので、
良くなったら基準を上げてよい。

使い方::

    .venv\\Scripts\\python.exe agent\\checks\\check_clutter_hits.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CAMERA = "corner_low"
N_SESSIONS, N_RALLIES = 3, 5
SEED = 0
# 作った時点の値（再現率 0.586・適合率 0.439・打った側 0.867。3 本の動画・15 ラリー）から、たまたまの数本の違いでは
# 落ちないように余裕をとった基準
MIN = {"recall": 0.50, "precision": 0.37, "side_accuracy": 0.78}


def run(ctx: dict[str, Any] | None = None) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    from smash_speed.synth_clutter import hits_benchmark

    r = hits_benchmark(CAMERA, N_SESSIONS, N_RALLIES, 30.0, SEED)
    metrics = {k: round(r[k], 4) for k in ("recall", "precision", "side_accuracy", "shuttle_kept")}
    bad = [f"{k} {r[k]:.3f}（基準 {v}）" for k, v in MIN.items() if r[k] < v]
    ok = not bad
    summary = ("雑音を足した合成ラリーの打球（角の低いカメラ・30fps）：" + (
        f"再現率 {r['recall']:.3f}・適合率 {r['precision']:.3f}・打った側 {r['side_accuracy']:.3f}"
        f"（シャトルの検出が残った割合 {r['shuttle_kept']:.2f}）" if ok else "基準を下回った：" + "、".join(bad)))
    return {"ok": ok, "summary": summary, "metrics": metrics}


if __name__ == "__main__":
    if (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    res = run()
    print(("OK " if res["ok"] else "NG ") + res["summary"])
    sys.exit(0 if res["ok"] else 1)
