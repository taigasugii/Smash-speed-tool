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
# まとめて解く当てはめ（``fit_backend="batch"``。ショットの一覧の既定）も同じ条件で測り、1 つずつの当てはめとの初速の差が
# これを超えたら失敗にする（作った時点では最大 0.0004 km/h。合成ラリーは収束しない当てはめがないので、差は積分の誤差だけ）[km/h]
MAX_BATCH_DIFF_KMH = 0.05


def run(ctx: dict[str, Any] | None = None) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    from smash_speed.shot_fit import synthetic_benchmark

    parts, bad, metrics = [], [], {}
    for fps in (30.0, 120.0):
        r = synthetic_benchmark(N_RALLIES, fps, SEED, hits="audio")
        rb = synthetic_benchmark(N_RALLIES, fps, SEED, hits="audio", backend="batch")
        diff = float((r["rows"].est_kmh - rb["rows"].est_kmh).abs().max()) if len(r["rows"]) else 0.0
        key = f"{fps:.0f}fps"
        metrics[f"{key}_batch"] = {"mape": round(rb["mape"], 3), "n_failed": rb["n_failed"],
                                   "max_diff_kmh": round(diff, 5)}
        if not (diff <= MAX_BATCH_DIFF_KMH) or rb["n_failed"] != r["n_failed"]:
            bad.append(f"{key} のまとめた当てはめが 1 つずつと違う（初速の差 最大 {diff:.3f} km/h）")
        types = {k: round(v["mape"], 2) for k, v in r["by_type"].items() if v["mape"] is not None}
        metrics[key] = {"mape": round(r["mape"], 3), "n_failed": r["n_failed"], "n_shots": len(r["rows"]),
                        "by_type": types}
        worst = max(types, key=types.get)
        parts.append(f"{fps:.0f}fps 初速のずれ {r['mape']:.2f}%（{len(r['rows'])} 本、いちばん大きい種類 {worst} "
                     f"{types[worst]:.1f}%。まとめた当てはめとの差 最大 {diff:.4f} km/h）")
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
