"""ショットの種類の判定の精度のチェック：合成ラリーで、ルールによる判定の混同行列を出す（B4）。

1. 正解の軌道から判定する（``shot_types.truth_benchmark``）。ルールそのものの精度。
2. 横から撮った合成ラリー（``synth_rally.rally_camera``）を 30 fps で観測し、打球を ``find_hits``（打球音あり）で見つけ、
   ``shot_fit.fit_shots`` で当てはめた軌道から判定する（``shot_types.synthetic_benchmark``）。

正解率が ``MIN_TRUTH_ACCURACY``・``MIN_FIT_ACCURACY`` を下回る、または当てはめた軌道で
どれかの種類（``MIN_TYPE_N`` 本以上あるもの）の再現率が ``MIN_TYPE_RECALL`` を下回ったら失敗にする。

使い方::

    .venv\\Scripts\\python.exe agent\\checks\\check_shot_types.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
N_TRUTH_RALLIES = 40
N_FIT_RALLIES = 5
SEED = 0
# 作った時点の値（正解の軌道 0.998・30fps の当てはめ 1.000、別の 12 ラリーでは 0.981）から、
# たまたまの違いでは落ちないように余裕をとった基準。種類ごとの再現率は MIN_TYPE_N 本以上ある種類だけ見る
MIN_TRUTH_ACCURACY = 0.97
MIN_FIT_ACCURACY = 0.85
MIN_TYPE_RECALL = 0.6
MIN_TYPE_N = 3


def run(ctx: dict[str, Any] | None = None) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    from smash_speed.shot_types import mistakes, synthetic_benchmark, truth_benchmark
    from smash_speed.shots import SHOT_TYPE_LABELS

    bad, metrics = [], {}
    t = truth_benchmark(N_TRUTH_RALLIES, SEED)
    f = synthetic_benchmark(N_FIT_RALLIES, 30.0, SEED, hits="audio")
    acc_t, acc_f = t["scores"]["accuracy"], f["scores"]["accuracy"]
    recalls = {k: round(v["recall"], 3) for k, v in f["scores"]["by_type"].items() if v["n"] >= MIN_TYPE_N}
    metrics = {"truth": {"accuracy": round(acc_t, 4), "n": t["scores"]["n"]},
               "fit_30fps": {"accuracy": round(acc_f, 4), "n": f["scores"]["n"], "recall": recalls}}
    if acc_t < MIN_TRUTH_ACCURACY:
        bad.append(f"正解の軌道からの正解率 {acc_t:.3f}（基準 {MIN_TRUTH_ACCURACY}）")
    if acc_f < MIN_FIT_ACCURACY:
        bad.append(f"30fps の当てはめからの正解率 {acc_f:.3f}（基準 {MIN_FIT_ACCURACY}）")
    for k, v in recalls.items():
        if v < MIN_TYPE_RECALL:
            bad.append(f"30fps の {SHOT_TYPE_LABELS[k]} の再現率 {v:.2f}（基準 {MIN_TYPE_RECALL}）")
    worst = mistakes(f["matrix"], 2)
    ok = not bad
    summary = ("ショットの種類の判定：" + (
        f"正解の軌道 {acc_t:.3f}（{t['scores']['n']} 本）、30fps の当てはめ {acc_f:.3f}（{f['scores']['n']} 本"
        + ("、多い間違い " + "・".join(f"{SHOT_TYPE_LABELS[a]}→{SHOT_TYPE_LABELS[b]} {n}" for a, b, n in worst)
           if worst else "") + "）"
        if ok else "基準を下回った：" + "、".join(bad)))
    return {"ok": ok, "summary": summary, "metrics": metrics}


if __name__ == "__main__":
    if (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    res = run()
    print(("OK " if res["ok"] else "NG ") + res["summary"])
    sys.exit(0 if res["ok"] else 1)
