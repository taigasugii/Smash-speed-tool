"""打つ人の足元の自動推定の精度のチェック：合成の足の動きで、打った時刻の足元の推定と本当の足元の距離を測る（D3）。

ホームと打つ所を行き来する足元の軌跡（``synth_players.footwork_path``）を 3 つのカメラで写し、実動画で測った揺れ
（打つたびに枠が横へ広がる分を含む）を足した位置から、止まって打った時刻ごとに ``hitter.hitter_at`` で足元を出す。
手前・奥のどちらかで、距離の中央値か 90% 点が基準を超えたら、または出せなかった打球があったら失敗にする。

使い方::

    .venv\\Scripts\\python.exe agent\\checks\\check_hitter.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SEED, N_SEEDS, FPS = 0, 4, 30.0
# 作った時点の値（30 fps、3 カメラ × 4 本 × 20 秒）：中央値 手前 0.18 m・奥 0.19 m、90% 点 0.23 m・0.23 m。
# たまたまの違いでは落ちないように余裕をとった基準
MAX_MEDIAN_M = 0.3
MAX_P90_M = 0.4


def run(ctx: dict[str, Any] | None = None) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    from smash_speed.hitter import synthetic_benchmark

    r = synthetic_benchmark(N_SEEDS, FPS, SEED)
    bad, parts = [], []
    for side, name in (("h", "手前"), ("o", "奥")):
        m = r[side]
        parts.append(f"{name} 中央値 {m['median_m']:.2f} m・90% 点 {m['p90_m']:.2f} m（{m['n']} 本）")
        if m["median_m"] > MAX_MEDIAN_M:
            bad.append(f"{name}の中央値 {m['median_m']:.2f} m（基準 {MAX_MEDIAN_M} m）")
        if m["p90_m"] > MAX_P90_M:
            bad.append(f"{name}の 90% 点 {m['p90_m']:.2f} m（基準 {MAX_P90_M} m）")
        if m["missed"]:
            bad.append(f"{name}で足元を出せなかった打球 {m['missed']} 本")
    ok = not bad
    summary = "打つ人の足元（合成、本当の足元との距離）：" + ("、".join(parts) if ok else "基準を超えた：" + "、".join(bad))
    return {"ok": ok, "summary": summary, "metrics": r}


if __name__ == "__main__":
    if (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    res = run()
    print(("OK " if res["ok"] else "NG ") + res["summary"])
    sys.exit(0 if res["ok"] else 1)
