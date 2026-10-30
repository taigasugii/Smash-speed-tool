"""実動画の回帰チェック：基準にした実動画のスマッシュを解析し直し、初速が基準から 3% 以上ずれていないかを調べる。

基準（``real_videos_baseline.json``）には、動画ごとに次を固めて入れてある（本体の ``outputs/`` が変わっても基準は変わらない）。

- キャリブレーション（``outputs/*_calibration.json`` などの中身）
- 順位表（``outputs/*_smashes/ranking.csv``）に載った候補の区間・打つ人・着地点・候補探しで検出したシャトルの位置
- 基準の初速：基準を作った時点のコードで解析し直した値（``kmh``）。``ranking.csv`` の値は ``ranking_kmh`` に参考として残す
  （``ranking.csv`` は古いコードで作られたものがあり、今のコードとは手順が違うため）

解析し直しは GUI の「詳しく解析する」（``batch.analyze_selected``）と同じ手順で、動画は ``.agent/config.json`` の
``videos_dir`` か、基準を作ったときの場所から探す。動画がない PC では「飛ばした」として通す。

使い方::

    .venv\\Scripts\\python.exe agent\\checks\\check_real_videos.py            # チェックだけ
    .venv\\Scripts\\python.exe agent\\checks\\check_real_videos.py --make-baseline   # 本体の outputs から基準を作り直す（人が決めたときだけ）
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASELINE = Path(__file__).with_name("real_videos_baseline.json")
TOLERANCE = 0.03        # 基準の初速からこの割合以上ずれたら失敗


# =============================================================================
# 基準を作る
# =============================================================================
def make_baseline(outputs_dir: Path) -> dict[str, Any]:
    """本体の ``outputs/`` にある ``*_smashes/ranking.csv`` から基準を作る。"""
    videos = []
    for rank_path in sorted(Path(outputs_dir).glob("*_smashes/ranking.csv")):
        out_dir = rank_path.parent
        data = json.loads((out_dir / "candidates.json").read_text(encoding="utf-8"))
        info = data["info"]
        stem = out_dir.name[: -len("_smashes")]
        calib = Path(info.get("calib") or "")
        if not calib.exists():
            calib = Path(outputs_dir) / f"{stem}_calibration.json"
        ranking = pd.read_csv(rank_path, encoding="utf-8-sig")
        by_id = {c["id"]: c for c in data["candidates"]}
        cands = []
        for _, r in ranking.iterrows():
            c = by_id[int(r["id"])]
            det = out_dir / c["detections"] if c.get("detections") else None
            cands.append({
                "id": c["id"], "t_onset": c["t_onset"], "strength": c["strength"],
                "start_s": c["start_s"], "end_s": c["end_s"], "hitter": c["hitter"], "landing": c["landing"],
                "confirmed": c["confirmed"], "rough_kmh": c.get("rough_kmh"),
                "detections_csv": det.read_text(encoding="utf-8") if det is not None and det.exists() else None,
                "kmh": float(r["speed_initial_kmh"]),
                "ranking_kmh": float(r["speed_initial_kmh"]),     # ranking.csv の値（作った時点のコードの結果。参考）
            })
        videos.append({"name": stem, "video": Path(info["video"]).name, "video_path": info["video"],
                       "capture_fps": info.get("capture_fps"), "audio": info.get("audio"),
                       "calibration": json.loads(calib.read_text(encoding="utf-8")), "candidates": cands})
    return {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "source": str(outputs_dir), "tolerance": TOLERANCE,
            "videos": videos}


# =============================================================================
# 解析し直す
# =============================================================================
def find_video(entry: dict[str, Any], videos_dir: str | None) -> Path | None:
    for p in ([Path(videos_dir) / entry["video"]] if videos_dir else []) + [Path(entry["video_path"])]:
        if p.exists():
            return p
    return None


def rerun_video(entry: dict[str, Any], video: Path, cfg: dict[str, Any], work: Path) -> dict[int, dict[str, Any]]:
    """1 本の動画の基準の候補を、GUI の「詳しく解析する」と同じ手順で解析し直す。"""
    sys.path.insert(0, str(ROOT))      # 別のプロセスで動かすとき用
    from smash_speed.batch import Candidate, analyze_selected, save_candidates

    work.mkdir(parents=True, exist_ok=True)
    (work / "det").mkdir(exist_ok=True)
    calib = work / "calibration.json"
    calib.write_text(json.dumps(entry["calibration"], ensure_ascii=False), encoding="utf-8")
    cands = []
    for b in entry["candidates"]:
        c = Candidate(id=b["id"], t_onset=b["t_onset"], strength=b["strength"], start_s=b["start_s"],
                      end_s=b["end_s"], status="ok", hitter=b["hitter"], landing=b["landing"],
                      confirmed=b["confirmed"], rough_kmh=b.get("rough_kmh"), selected=True)
        if b.get("detections_csv"):
            c.detections = f"det/{c.label}.csv"
            (work / c.detections).write_text(b["detections_csv"], encoding="utf-8")
        cands.append(c)
    save_candidates(work, cands, {"video": str(video), "calib": str(calib), "capture_fps": entry.get("capture_fps"),
                                  "audio": entry.get("audio")})
    with contextlib.redirect_stdout(io.StringIO()):      # 解析の「[注意]」は品質チェックの出力に混ぜない
        done = analyze_selected(work, cfg, overlay=False, progress=lambda s: None)
    return {c.id: c.result or {} for c in done}


def rerun_all(todo: list[tuple[dict[str, Any], Path]], baseline: Path) -> dict[str, dict[int, dict[str, Any]]]:
    """動画ごとに別のプロセス（このファイルの ``--rerun``）で並べて解析し直す（1 本ずつだと 4 本で 12 分ほどかかる）。

    品質チェックはこのファイルを名前のないモジュールとして読み込むので、multiprocessing ではなく
    コマンドとして起動する。
    """
    res = {}
    with tempfile.TemporaryDirectory() as td:
        procs = []
        for e, v in todo:
            out = Path(td) / f"{e['name']}.json"
            cmd = [sys.executable, str(Path(__file__).resolve()), "--rerun", e["name"], "--video", str(v),
                   "--baseline", str(baseline), "--out", str(out)]
            procs.append((e["name"], out, subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL,
                                                           stderr=subprocess.PIPE)))
        for name, out, p in procs:
            _, err = p.communicate()
            if p.returncode != 0 or not out.exists():
                msg = err.decode("utf-8", "replace").strip().splitlines()[-1:] or ["不明"]
                raise RuntimeError(f"{name} の解析し直しが止まりました：{msg[0]}")
            res[name] = {int(k): v for k, v in json.loads(out.read_text(encoding="utf-8")).items()}
    return res


def compare(entry: dict[str, Any], results: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for b in entry["candidates"]:
        r = results.get(b["id"], {})
        kmh = r.get("kmh")
        dev = None if kmh is None else (kmh - b["kmh"]) / b["kmh"]
        rows.append({"video": entry["name"], "id": b["id"], "t_onset": b["t_onset"], "base_kmh": round(b["kmh"], 1),
                     "kmh": None if kmh is None else round(kmh, 1), "dev_pct": None if dev is None else round(dev * 100, 2),
                     "ok": dev is not None and abs(dev) <= TOLERANCE, "error": r.get("error")})
    return rows


def run(ctx: dict[str, Any]) -> dict[str, Any]:
    if not BASELINE.exists():
        return {"ok": True, "skipped": True, "summary": "実動画：基準がないので飛ばしました"}
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    videos_dir = (ctx.get("config") or {}).get("videos_dir")
    found = [(e, find_video(e, videos_dir)) for e in base["videos"]]
    todo = [(e, v) for e, v in found if v is not None]
    missing = [e["name"] for e, v in found if v is None]
    if not todo:
        return {"ok": True, "skipped": True, "summary": "実動画：動画が見つからないので飛ばしました"}

    results = rerun_all(todo, BASELINE)
    rows = []
    for e, _ in todo:
        rows += compare(e, results[e["name"]])
    bad = [r for r in rows if not r["ok"]]
    devs = [abs(r["dev_pct"]) for r in rows if r["dev_pct"] is not None]
    summary = (f"実動画 {len(todo)} 本・{len(rows)} 本のスマッシュ：最大のずれ {max(devs, default=0):.2f}%"
               + (f"、3% を超えたもの {len(bad)} 本（" + "、".join(
                   f"{r['video']} #{r['id']} {r['base_kmh']}→{r['kmh'] if r['kmh'] is not None else '解析できず'}"
                   for r in bad) + "）" if bad else "")
               + (f"（動画がなく飛ばした：{'、'.join(missing)}）" if missing else ""))
    return {"ok": not bad, "summary": summary, "metrics": {"rows": rows, "max_dev_pct": max(devs, default=0.0),
                                                           "missing": missing}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--make-baseline", action="store_true", help="本体の outputs から基準を作り直す")
    ap.add_argument("--outputs", help="基準を作る outputs フォルダ（既定は .agent/config.json の main_repo の outputs）")
    ap.add_argument("--rerun", help=argparse.SUPPRESS)       # 内部用：1 本の動画を解析し直して --out に書く
    ap.add_argument("--video", help=argparse.SUPPRESS)
    ap.add_argument("--baseline", help=argparse.SUPPRESS)
    ap.add_argument("--out", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    if a.rerun:
        sys.path.insert(0, str(ROOT))
        from smash_speed.config import load_config

        base = json.loads(Path(a.baseline).read_text(encoding="utf-8"))
        entry = next(e for e in base["videos"] if e["name"] == a.rerun)
        with tempfile.TemporaryDirectory() as td:
            res = rerun_video(entry, Path(a.video), load_config(), Path(td) / entry["name"])
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        return 0
    cfg_path = ROOT / ".agent" / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    if a.make_baseline:
        outputs = Path(a.outputs) if a.outputs else Path(cfg.get("main_repo", ROOT)) / "outputs"
        b = make_baseline(outputs)
        # 基準の初速は、今のコードで解析し直した値にする（動画がなければ ranking.csv の値のまま）
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "baseline.json"
            tmp.write_text(json.dumps(b, ensure_ascii=False), encoding="utf-8")
            todo = [(e, v) for e in b["videos"] if (v := find_video(e, cfg.get("videos_dir"))) is not None]
            results = rerun_all(todo, tmp)
        for e in b["videos"]:
            for c in e["candidates"]:
                kmh = results.get(e["name"], {}).get(c["id"], {}).get("kmh")
                if kmh is not None:
                    c["kmh"] = kmh
                print(f"  {e['name']} #{c['id']}：ranking.csv {c['ranking_kmh']:.1f} → 基準 {c['kmh']:.1f} km/h")
        BASELINE.write_text(json.dumps(b, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"基準を作りました：{len(b['videos'])} 本の動画・"
              f"{sum(len(v['candidates']) for v in b['videos'])} 本のスマッシュ → {BASELINE}")
        return 0
    t = time.time()
    r = run({"root": ROOT, "config": cfg})
    for row in r.get("metrics", {}).get("rows", []):
        print(f"  {row['video']} #{row['id']} {row['t_onset']:.1f}s：基準 {row['base_kmh']} → {row['kmh']} km/h"
              f"（{row['dev_pct']}%）{'' if row['ok'] else '  ← NG ' + (row['error'] or '')}")
    print(f"[{'飛ばした' if r.get('skipped') else 'OK' if r['ok'] else 'NG'}] {r['summary']}（{time.time() - t:.0f} 秒）")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
