"""ラリーの区切りの精度のチェック：続けて打った合成ラリーで、ラリーの区切りを測る（C1）。

ラリーの間の休み（着地から次のサーブまで）を短く（1〜2 秒）して、打球の間隔だけでは区切れない場合を作る。

1. 正解のショット（打った時刻・着地の時刻に少しずれを足したもの）を区切る（``rallies.synthetic_benchmark`` の truth）。
2. 横から撮った合成ラリーを 30 fps で観測し、動画全体の検出から打球を見つけ（打球音あり）、軌道を当てはめてから区切る
   （同 fit。実際の流れと同じ）。

区切りの再現率・適合率が基準を下回る、または打球の間隔だけで分けるやり方（``shots.make_rallies``）より
再現率が低くなったら失敗にする。

使い方::

    .venv\\Scripts\\python.exe agent\\checks\\check_rallies.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SEED = 0
REST_S = (1.0, 2.0)
N_TRUTH_SESSIONS, N_FIT_SESSIONS, N_RALLIES = 4, 1, 5
# 作った時点の値（truth 20×6 ラリー・休み 1〜8 秒で 1.000、fit 2×5 ラリー・休み 1〜2 秒で 1.000。
# 間隔だけでは 0.92・0.50）から、たまたまの違いでは落ちないように余裕をとった基準
MIN_TRUTH = 0.95
MIN_FIT = 0.8


def run(ctx: dict[str, Any] | None = None) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    from smash_speed.rallies import synthetic_benchmark

    workers = max(1, min(8, (os.cpu_count() or 2) - 1))
    bad, parts, metrics = [], [], {}
    for mode, n, lim in (("truth", N_TRUTH_SESSIONS, MIN_TRUTH), ("fit", N_FIT_SESSIONS, MIN_FIT)):
        r = synthetic_benchmark(n, N_RALLIES, 30.0, SEED, mode, rest_s=REST_S, workers=workers)
        new, gap = r["new"], r["gap"]
        metrics[mode] = {k: round(v, 3) if isinstance(v, float) else v for k, v in new.items()}
        metrics[mode]["gap_recall"] = round(gap["boundary_recall"], 3)
        parts.append(f"{mode} 再現率 {new['boundary_recall']:.3f}・適合率 {new['boundary_precision']:.3f}"
                     f"（間隔だけ {gap['boundary_recall']:.3f}）")
        for k in ("boundary_recall", "boundary_precision"):
            if new[k] < lim:
                bad.append(f"{mode} の {k} {new[k]:.3f}（基準 {lim}）")
        if new["boundary_recall"] < gap["boundary_recall"]:
            bad.append(f"{mode} で間隔だけのやり方より再現率が低い")
    ok = not bad
    summary = f"ラリーの区切り（休み {REST_S[0]:.0f}〜{REST_S[1]:.0f} 秒）：" + ("、".join(parts) if ok else
                                                                         "基準を下回った：" + "、".join(bad))
    return {"ok": ok, "summary": summary, "metrics": metrics}


if __name__ == "__main__":
    if (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    res = run()
    print(("OK " if res["ok"] else "NG ") + res["summary"])
    sys.exit(0 if res["ok"] else 1)
