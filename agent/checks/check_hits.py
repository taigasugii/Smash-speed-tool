"""打球を見つける精度のチェック：合成ラリーで、すべての打球の再現率・適合率・打った側の正解率を測る（B2）。

横から撮った合成ラリー（``synth_rally.rally_camera``）を 30 fps・120 fps、打球音のあり・なしで解析し、
どれかが基準（``MIN``）を下回ったら失敗にする。打球音には、打球ではない音（隣のコートなど）も 3 個まぜる。

使い方::

    .venv\\Scripts\\python.exe agent\\checks\\check_hits.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
N_RALLIES = 12
SEED = 0
# 作った時点の値（すべて 1.000）から、たまたまの 1〜2 本の違いでは落ちないように余裕をとった基準
MIN = {"recall": 0.95, "precision": 0.95, "side_accuracy": 0.95}


def run(ctx: dict[str, Any] | None = None) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    from smash_speed.hits import synthetic_benchmark

    parts, bad, metrics = [], [], {}
    for fps in (30.0, 120.0):
        for use_audio in (True, False):
            r = synthetic_benchmark(N_RALLIES, fps, SEED, use_audio)
            key = f"{fps:.0f}fps_{'audio' if use_audio else 'track'}"
            metrics[key] = {k: round(r[k], 4) for k in ("recall", "precision", "side_accuracy")}
            parts.append(f"{fps:.0f}fps 音{'あり' if use_audio else 'なし'} 再現率 {r['recall']:.3f}・"
                         f"適合率 {r['precision']:.3f}・打った側 {r['side_accuracy']:.3f}")
            bad += [f"{key} の {k} {r[k]:.3f}（基準 {v}）" for k, v in MIN.items() if r[k] < v]
    ok = not bad
    summary = "合成ラリーの打球：" + ("、".join(parts) if ok else "基準を下回った：" + "、".join(bad))
    return {"ok": ok, "summary": summary, "metrics": metrics}


if __name__ == "__main__":
    if (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    res = run()
    print(("OK " if res["ok"] else "NG ") + res["summary"])
    sys.exit(0 if res["ok"] else 1)
