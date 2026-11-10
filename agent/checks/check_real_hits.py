"""実動画の打球のチェック：コマを見て決めた正解（``agent/truth/*_hits.csv``）と、打球を見つけた結果を比べる。

区間ごと（``real_hits_clips.json``）に、ショットの一覧と同じ手順（重なった軌道も残した検出 → シャトルらしい軌道の
絞り込み → 選手の位置 → ``hits.find_hits``）で打球を見つけ、時刻の差 0.3 秒以内で 1 対 1 に組にして（``agent/truth/README.md``
の数え方）再現率・適合率・打った側の正解の数を出す。基準（``expected``）より、全体で再現率・適合率が 0.03 より
下がったか、打った側の正解の割合が下がったら失敗にする。当てはめ（種類・初速）はしない。

キャリブレーションは中身（カメラ）を ``real_hits_clips.json`` に入れてある。動画は ``.agent/config.json`` の
``videos_dir`` から探し、ない区間は飛ばす（1 本もなければ「飛ばした」として通す）。4 区間で 2 分ほど。

使い方::

    .venv\\Scripts\\python.exe agent\\checks\\check_real_hits.py              # チェックだけ（区間ごとの打球も出す）
    .venv\\Scripts\\python.exe agent\\checks\\check_real_hits.py --make-baseline   # 今の結果を基準にする（人が決めたときだけ）
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CLIPS = Path(__file__).with_name("real_hits_clips.json")
TRUTH = ROOT / "agent" / "truth"
TOL_S = 0.3             # 正解と組にする時刻の差 [s]
MARGIN = 0.03           # 基準からこれより下がったら失敗（たまたまの 1〜2 本の違いでは落ちないように）


def analyze_clip(clip: dict[str, Any], video: Path, cfg: dict[str, Any], work: Path) -> dict[str, Any]:
    """1 区間の打球を見つけて正解と比べる。"""
    import pandas as pd

    from smash_speed.calibration.io import load_calibration
    from smash_speed.hits import find_hits, match_hits
    from smash_speed.pipeline import _camera_for_video
    from smash_speed.players import PLAYERS_FILE, analyze_players
    from smash_speed.shot_list import detect_all, hit_onsets, shuttle_tracks
    from smash_speed.video_io import probe

    work.mkdir(parents=True, exist_ok=True)
    calib = work / "calibration.json"
    calib.write_text(json.dumps({"camera": clip["camera"]}), encoding="utf-8")
    s0, s1 = clip.get("start_s"), clip.get("end_s")
    meta = probe(video, None, cfg)
    camera, _ = load_calibration(calib)
    camera = _camera_for_video(camera, meta)
    onsets = hit_onsets(meta, cfg, work, None, s0, s1)
    det = detect_all(meta, cfg, s0, s1, exclusive=False, progress=lambda s: None)
    det, _ = shuttle_tracks(det, camera, cfg)
    analyze_players(video, calib, work, cfg, start_s=s0, end_s=s1, progress=lambda s: None)
    players = pd.read_csv(work / PLAYERS_FILE)
    hits = find_hits(det, meta.capture_fps, onsets, camera=camera, players=players)
    truth = pd.read_csv(TRUTH / clip["truth"])
    m = match_hits(hits, truth.t_s.to_numpy(float), list(truth.side), tol_s=TOL_S)
    return {"n_true": len(truth), "n_found": len(hits), "tp": m["tp"], "side_correct": m["side_correct"],
            "hits": [[round(h.t, 2), h.side, h.method] for h in hits]}


def totals(rows: dict[str, dict[str, Any]]) -> dict[str, float]:
    n_true = sum(r["n_true"] for r in rows.values())
    n_found = sum(r["n_found"] for r in rows.values())
    tp = sum(r["tp"] for r in rows.values())
    side = sum(r["side_correct"] for r in rows.values())
    return {"recall": tp / max(n_true, 1), "precision": tp / max(n_found, 1), "side": side / max(tp, 1)}


def run(ctx: dict[str, Any], keep_hits: bool = False) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    from smash_speed.config import load_config

    if not CLIPS.exists():
        return {"ok": True, "skipped": True, "summary": "実動画の打球：区間の一覧がないので飛ばしました"}
    clips = json.loads(CLIPS.read_text(encoding="utf-8"))["clips"]
    videos_dir = (ctx.get("config") or {}).get("videos_dir")
    found = [(c, Path(videos_dir) / c["video"] if videos_dir else None) for c in clips]
    todo = [(c, v) for c, v in found if v is not None and v.exists()]
    missing = [c["name"] for c, v in found if v is None or not v.exists()]
    if not todo:
        return {"ok": True, "skipped": True, "summary": "実動画の打球：動画が見つからないので飛ばしました"}
    cfg = load_config()
    rows: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory() as td:
        for c, v in todo:
            rows[c["name"]] = analyze_clip(c, v, cfg, Path(td) / c["name"])
    cur = totals(rows)
    exp = totals({c["name"]: c["expected"] for c, _ in todo if c.get("expected")}) if all(
        c.get("expected") for c, _ in todo) else None
    bad = []
    if exp:
        bad = [f"{k} {exp[k]:.3f}→{cur[k]:.3f}" for k in ("recall", "precision") if cur[k] < exp[k] - MARGIN]
        if cur["side"] < exp["side"] - 1e-9:
            bad.append(f"打った側 {exp['side']:.3f}→{cur['side']:.3f}")
    parts = [f"{n} {r['tp']}/{r['n_true']}・見つけた {r['n_found']}" for n, r in rows.items()]
    summary = (f"実動画の打球 {len(rows)} 区間：再現率 {cur['recall']:.3f}・適合率 {cur['precision']:.3f}・"
               f"打った側 {cur['side']:.3f}（" + "、".join(parts) + "）"
               + (f"、基準より下がった：{'、'.join(bad)}" if bad else "")
               + ("" if exp else "、基準がないので比べていない")
               + (f"（動画がなく飛ばした：{'、'.join(missing)}）" if missing else ""))
    if not keep_hits:
        rows = {n: {k: v for k, v in r.items() if k != "hits"} for n, r in rows.items()}
    return {"ok": not bad, "summary": summary, "metrics": {"total": cur, "clips": rows, "missing": missing}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--make-baseline", action="store_true", help="今の結果を基準（expected）にする")
    a = ap.parse_args(argv)
    if (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    cfg_path = ROOT / ".agent" / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    t = time.time()
    r = run({"root": ROOT, "config": cfg}, keep_hits=True)
    for name, row in r.get("metrics", {}).get("clips", {}).items():
        print(f"  {name}：" + " ".join(f"{t_:.2f}{s or '-'}" for t_, s, _ in row["hits"]))
    if a.make_baseline and not r.get("skipped"):
        data = json.loads(CLIPS.read_text(encoding="utf-8"))
        for c in data["clips"]:
            row = r["metrics"]["clips"].get(c["name"])
            if row:
                c["expected"] = {k: row[k] for k in ("n_true", "n_found", "tp", "side_correct")}
        CLIPS.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"基準を書き直しました → {CLIPS}")
    print(f"[{'飛ばした' if r.get('skipped') else 'OK' if r['ok'] else 'NG'}] {r['summary']}（{time.time() - t:.0f} 秒）")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
