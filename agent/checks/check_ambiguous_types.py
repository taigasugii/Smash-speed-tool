"""種類の範囲を重ねた「あいまいな」合成ラリーで、ショットの種類の判定の正解率を測る（B4 の境目での間違い方）。

``synth_ambiguous.AMBIGUOUS_PROFILES``（遅いスマッシュ・速いドロップ・高めのドライブ・低いロブ・打点の高いプッシュ・
少し深いネットなど、範囲を少し広げて重ねたもの）で合成ラリーを作り、正解の軌道から判定する（当てはめの誤差なし。
ルールの境目そのものの限界を見る）。既定の範囲（``check_shot_types``）より低い所に基準を置く。今の値より悪くなったことを
見つけるためのもので、B7 などで判定が良くなったら基準を上げてよい。

使い方::

    .venv\\Scripts\\python.exe agent\\checks\\check_ambiguous_types.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
N_RALLIES = 40
SEED = 0
# 作った時点の値（正解率 0.858、346 本。100 ラリーでは 0.868）から、たまたまの数本の違いでは落ちないように
# 余裕をとった基準。サーブはラリーの最初かどうかだけで決まるので、いつも 1.0
MIN_ACCURACY = 0.82
MIN_SERVE_RECALL = 1.0


def run(ctx: dict[str, Any] | None = None) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    from smash_speed.shot_types import mistakes
    from smash_speed.shots import SHOT_TYPE_LABELS
    from smash_speed.synth_ambiguous import ambiguous_benchmark

    r = ambiguous_benchmark("truth", N_RALLIES, seed=SEED)
    sc = r["scores"]
    acc, serve = sc["accuracy"], sc["by_type"]["serve"]["recall"]
    metrics = {"accuracy": round(acc, 4), "n": sc["n"],
               "recall": {k: round(v["recall"], 3) for k, v in sc["by_type"].items() if v["recall"] is not None}}
    bad = []
    if acc < MIN_ACCURACY:
        bad.append(f"正解率 {acc:.3f}（基準 {MIN_ACCURACY}）")
    if serve is not None and serve < MIN_SERVE_RECALL:
        bad.append(f"サーブの再現率 {serve:.3f}（基準 {MIN_SERVE_RECALL}）")
    worst = mistakes(r["matrix"], 3)
    ok = not bad
    summary = ("あいまいな範囲の合成ラリーの種類（正解の軌道）：" + (
        f"正解率 {acc:.3f}（{sc['n']} 本、多い間違い "
        + "・".join(f"{SHOT_TYPE_LABELS[a]}→{SHOT_TYPE_LABELS[b]} {n}" for a, b, n in worst) + "）"
        if ok else "基準を下回った：" + "、".join(bad)))
    return {"ok": ok, "summary": summary, "metrics": metrics}


if __name__ == "__main__":
    if (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    res = run()
    print(("OK " if res["ok"] else "NG ") + res["summary"])
    sys.exit(0 if res["ok"] else 1)
