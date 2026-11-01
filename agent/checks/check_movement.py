"""選手の移動距離・速さの精度のチェック：合成の足の動きで、正解との比を測る（D2）。

ホームと打つ所を行き来する足元の軌跡（``synth_players.footwork_path``）を、3 つのカメラ（後ろの角の高い所・
横の低い所・真後ろ）で写し、実動画で測った揺れ（コマごとの揺れ・ゆっくりした揺れ・打つときに枠が横へ広がる分・
ときどき大きく外れるコマ）を足した位置から、``movement`` で移動距離と最高の速さを出す。
手前・奥のどちらかで、正解とのずれの最大が基準を超えたら失敗にする。

使い方::

    .venv\\Scripts\\python.exe agent\\checks\\check_movement.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SEED, N_SEEDS, FPS = 0, 4, 30.0
# 作った時点の値（30 fps、3 カメラ × 4 本 × 20 秒）：移動距離 手前 3.2%・奥 2.3%、最高の速さ 手前 11.4%・奥 8.7%。
# たまたまの違いでは落ちないように余裕をとった基準
MAX_DISTANCE = 0.08
MAX_PEAK = 0.2


def run(ctx: dict[str, Any] | None = None) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    from smash_speed.movement import synthetic_benchmark

    r = synthetic_benchmark(N_SEEDS, FPS, SEED)
    bad, parts = [], []
    for side, name in (("h", "手前"), ("o", "奥")):
        d, v = r[f"{side}_distance"], r[f"{side}_max_speed"]
        parts.append(f"{name} 距離 {100 * d['max_abs']:.1f}%・最高の速さ {100 * v['max_abs']:.1f}%")
        if d["max_abs"] > MAX_DISTANCE:
            bad.append(f"{name}の移動距離のずれ {100 * d['max_abs']:.1f}%（基準 {100 * MAX_DISTANCE:.0f}%）")
        if v["max_abs"] > MAX_PEAK:
            bad.append(f"{name}の最高の速さのずれ {100 * v['max_abs']:.1f}%（基準 {100 * MAX_PEAK:.0f}%）")
    ok = not bad
    summary = "選手の動き（合成、ずれの最大）：" + ("、".join(parts) if ok else "基準を超えた：" + "、".join(bad))
    return {"ok": ok, "summary": summary, "metrics": r}


if __name__ == "__main__":
    if (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    res = run()
    print(("OK " if res["ok"] else "NG ") + res["summary"])
    sys.exit(0 if res["ok"] else 1)
