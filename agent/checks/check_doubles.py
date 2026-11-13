"""ダブルスの追跡・誰が打ったか・陣形のチェック：合成のダブルスで正解と比べる（D4）。

4 人の足の動き・誰が打ったか・陣形の正解がある合成ラリー（``synth_doubles``。待つ人・隣のコートの人・横を歩く人つき、
2 人が重なると検出が抜ける・1 つの枠にまとまる）を 3 つのカメラで写し、

- 自動でダブルスと分かるか、シングルスの合成ではシングルスのままで、結果が今までの追跡と同じか
- 番号の正しさ（追ったコマのうち、番号と人が合っていた割合）・選手ではない人を追った割合
- 誰が打ったか（打球の時刻と側は正解。当てはめの前 = 画像の手がかりだけ／打点が 0.5 m の誤差で分かったとき）
- 陣形（コマごとの前後・横並び・移っている途中が、正解の足元から出したものと合う割合）

を測る。カメラごとに基準を下回ったら失敗にする。服の色は 4 人とも違う場合（色も使える）で測る。
同じ色の服の数字は ``docs/doubles.md``（``python -m smash_speed.synth_doubles`` で出せる）。

使い方::

    .venv\\Scripts\\python.exe agent\\checks\\check_doubles.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SEED, N_SESSIONS, N_RALLIES, FPS = 0, 2, 4, 30.0
# 作った時点の値（30 fps、2 本 × 4 ラリー、服の色は 4 人とも違う）は下の MADE。たまたまの違いでは落ちないよう余裕をとった基準
# カメラ → (番号の正しさ, 選手ではない人の割合の上限, 打った人・前, 打った人・後, 陣形)
LIMITS = {"back_high": (0.95, 0.05, 0.85, 0.88, 0.92),
          "corner_low": (0.90, 0.08, 0.75, 0.78, 0.85),
          "side": (0.97, 0.03, 0.93, 0.93, 0.94)}


def run(ctx: dict[str, Any] | None = None) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    from smash_speed.synth_doubles import doubles_benchmark, singles_auto_benchmark

    r = doubles_benchmark(N_SESSIONS, N_RALLIES, FPS, SEED, cameras=tuple(LIMITS))
    single = singles_auto_benchmark(N_SESSIONS, N_RALLIES, FPS, SEED, cameras=tuple(LIMITS))
    bad, parts = [], []
    for cam, (lim_id, lim_extra, lim_pre, lim_post, lim_form) in LIMITS.items():
        m = r[cam]
        parts.append(f"{cam} 番号 {m['id_accuracy']:.0%}・打った人 {m['hitter_pre']:.0%}/{m['hitter_post']:.0%}・"
                     f"陣形 {m['formation']:.0%}")
        if m["auto_doubles"] < m["n_sessions"]:
            bad.append(f"{cam}：自動でダブルスにならなかった（{m['auto_doubles']}/{m['n_sessions']}）")
        for name, value, limit, higher in (("番号の正しさ", m["id_accuracy"], lim_id, True),
                                           ("選手ではない人の割合", m["extra_share"], lim_extra, False),
                                           ("打った人（当てはめの前）", m["hitter_pre"], lim_pre, True),
                                           ("打った人（打点つき）", m["hitter_post"], lim_post, True),
                                           ("陣形", m["formation"], lim_form, True)):
            if (value < limit) if higher else (value > limit):
                bad.append(f"{cam}：{name} {value:.1%}（基準 {limit:.0%}）")
        s = single[cam]
        if s["singles"] < s["n"] or s["same_as_singles"] < s["n"]:
            bad.append(f"{cam}：シングルスの合成でシングルスのままにならなかった（{s}）")
    ok = not bad
    summary = "ダブルス（合成）：" + ("、".join(parts) if ok else "基準を外れた：" + "、".join(bad))
    return {"ok": ok, "summary": summary, "metrics": {"doubles": r, "singles": single}}


if __name__ == "__main__":
    if (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    res = run()
    print(("OK " if res["ok"] else "NG ") + res["summary"])
    sys.exit(0 if res["ok"] else 1)
