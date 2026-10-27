"""ショットごとの初速の精度のチェック：合成ラリーで、すべての種類のショットの初速の誤差を測る（B3）。

横から撮った合成ラリー（``synth_rally.rally_camera``）を 30 fps・120 fps で観測し、打球の時刻は
``find_hits``（打球音あり）で見つけたものを使って ``shot_fit.fit_shots`` で当てはめる。
全体の平均絶対誤差（MAPE）が ``MAX_MAPE`` を超える、当てはめられなかったショットがある、
どれかの種類の MAPE が ``MAX_TYPE_MAPE``（サーブは ``MAX_TYPE_MAPE_OVERRIDE``）を超えたら失敗にする。

使い方::

    .venv\\Scripts\\python.exe agent\\checks\\check_shot_speeds.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
N_RALLIES = 5
SEED = 0
# 作った時点の値（全体 30fps 2.47%・120fps 1.76%、種類ごとはサーブ 8.3% のほかは 6% 以下）から、
# たまたまの違いでは落ちないように余裕をとった基準。サーブは最初のショットで、前の軌道がなく打点の時刻を
# 直せないので緩める（5 ラリーではサーブが 5 本しかなく、ばらつきも大きい）
MAX_MAPE = 4.0
MAX_TYPE_MAPE = 10.0
MAX_TYPE_MAPE_OVERRIDE = {"serve": 15.0}


def run(ctx: dict[str, Any] | None = None) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    from smash_speed.shot_fit import synthetic_benchmark

    parts, bad, metrics = [], [], {}
    for fps in (30.0, 120.0):
        r = synthetic_benchmark(N_RALLIES, fps, SEED, hits="audio")
        key = f"{fps:.0f}fps"
        types = {k: round(v["mape"], 2) for k, v in r["by_type"].items() if v["mape"] is not None}
        metrics[key] = {"mape": round(r["mape"], 3), "n_failed": r["n_failed"], "n_shots": len(r["rows"]),
                        "by_type": types}
        worst = max(types, key=types.get)
        parts.append(f"{fps:.0f}fps 初速のずれ {r['mape']:.2f}%（{len(r['rows'])} 本、いちばん大きい種類 {worst} "
                     f"{types[worst]:.1f}%）")
        if r["mape"] > MAX_MAPE:
            bad.append(f"{key} の全体 {r['mape']:.2f}%（基準 {MAX_MAPE}%）")
        if r["n_failed"]:
            bad.append(f"{key} で当てはめられなかったショット {r['n_failed']} 本")
        for k, v in types.items():
            lim = MAX_TYPE_MAPE_OVERRIDE.get(k, MAX_TYPE_MAPE)
            if v > lim:
                bad.append(f"{key} の {k} {v:.1f}%（基準 {lim}%）")
    ok = not bad
    summary = "合成ラリーのショットの初速：" + ("、".join(parts) if ok else "基準を超えた：" + "、".join(bad))
    return {"ok": ok, "summary": summary, "metrics": metrics}


if __name__ == "__main__":
    if (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    res = run()
    print(("OK " if res["ok"] else "NG ") + res["summary"])
    sys.exit(0 if res["ok"] else 1)
