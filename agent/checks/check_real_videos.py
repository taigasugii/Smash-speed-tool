"""実動画の回帰チェック：基準にした実動画のスマッシュを解析し直し、初速が基準から 3% 以上ずれていないかを調べる。

基準（``real_videos_baseline.json``）には、動画ごとに次を固めて入れてある（本体の ``outputs/`` が変わっても基準は変わらない）。

- キャリブレーション（``outputs/*_calibration.json`` などの中身）
- 順位表（``outputs/*_smashes/ranking.csv``）に載った候補の区間・打つ人・着地点・候補探しで検出したシャトルの位置
- 基準の初速：基準を作った時点のコードで解析し直した値（``kmh``）。``ranking.csv`` の値は ``ranking_kmh`` に参考として残す
  （``ranking.csv`` は古いコードで作られたものがあり、今のコードとは手順が違うため）

正しい初速（``real_videos_truth.json``、コマを見て決めたもの。作り方は ``agent/truth/README.md``）があれば、
その誤差も出す（打った時刻の速さどうしと、解析の打った時刻での正しい軌道の速さとの差）。正しい初速の不確かさが大きい
（30 fps では打った時刻の誤差だけで ±20〜110 km/h ある）ので、参考として出すだけで失敗にはしない。

解析し直しは GUI の「詳しく解析する」（``batch.analyze_selected``）と同じ手順で、動画は ``.agent/config.json`` の
``videos_dir`` から名前で探す（基準には動画の場所を残さない）。動画がない PC では「飛ばした」として通す。

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
TRUTH = Path(__file__).with_name("real_videos_truth.json")
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
        # 動画の場所（人のフォルダの名前が入る）は基準に残さない。動画は ``videos_dir`` から名前で探す
        calibration = json.loads(calib.read_text(encoding="utf-8"))
        if isinstance(calibration.get("meta"), dict) and calibration["meta"].get("video"):
            calibration["meta"]["video"] = Path(calibration["meta"]["video"]).name
        videos.append({"name": stem, "video": Path(info["video"]).name,
                       "capture_fps": info.get("capture_fps"), "audio": info.get("audio"),
                       "calibration": calibration, "candidates": cands})
    return {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "source": str(outputs_dir), "tolerance": TOLERANCE,
            "videos": videos}


# =============================================================================
# 解析し直す
# =============================================================================
def find_video(entry: dict[str, Any], videos_dir: str | None) -> Path | None:
    # ``video_path`` は古い基準にだけある（今の基準には動画の場所を入れない）
    old = [Path(entry["video_path"])] if entry.get("video_path") else []
    for p in ([Path(videos_dir) / entry["video"]] if videos_dir else []) + old:
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
    res = {}
    for c in done:
        r = dict(c.result or {})
        fit = Path(r.get("dir", "")) / "fit_result.json"
        if fit.is_file():       # 正しい初速と同じ時刻で比べるため、解析の打った時刻も返す
            r["t_impact_s"] = json.loads(fit.read_text(encoding="utf-8")).get("t_ref_s")
        res[c.id] = r
    return res


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


def retrack_hitters(entry: dict[str, Any], video: Path, cfg: dict[str, Any], detector: Any, work: Path) -> int:
    """1 本の動画の基準の候補の打つ人の足元を、選手の追跡から出した足元にする（``find-smashes --players`` と同じ
    ``batch.candidate_hitters``）。置き換えた数を返す。

    元の足元は ``hitter_rough`` に残し、出どころを ``hitter_source`` に入れる。入れられなかった候補は元の足元のままにし、
    理由を ``hitter_note`` に書く。
    """
    sys.path.insert(0, str(ROOT))
    from smash_speed.batch import Candidate, candidate_hitters
    from smash_speed.calibration.io import load_calibration
    from smash_speed.pipeline import _camera_for_video
    from smash_speed.video_io import probe

    work.mkdir(parents=True, exist_ok=True)
    calib = work / "calibration.json"
    calib.write_text(json.dumps(entry["calibration"], ensure_ascii=False), encoding="utf-8")
    meta = probe(video, entry.get("capture_fps"), cfg)
    camera, _ = load_calibration(calib)
    camera = _camera_for_video(camera, meta)
    by_id = {b["id"]: b for b in entry["candidates"]}
    # 作り直すときは、前に入れた追跡の足元ではなく元の大まかな推定から選手を選ぶ
    cands = [Candidate(id=b["id"], t_onset=b["t_onset"], strength=b["strength"], start_s=b["start_s"],
                       end_s=b["end_s"], status="ok", hitter=dict(b.get("hitter_rough") or b["hitter"]),
                       landing=b["landing"], confirmed=b["confirmed"]) for b in entry["candidates"]]
    notes: dict[int, str] = {}
    candidate_hitters(cands, meta, camera, cfg, detector, progress=lambda s: None, notes=notes)
    n = 0
    for c in cands:
        b = by_id[c.id]
        b["hitter_rough"] = dict(b.get("hitter_rough") or b["hitter"])
        b["hitter"] = c.hitter
        b["hitter_source"] = c.hitter_source
        b.pop("hitter_note", None)
        if c.hitter_source:
            n += 1
        else:
            b["hitter_note"] = "確認済みの足元" if c.confirmed else notes.get(c.id, "足元を入れられなかった")
    return n


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


def compare_truth(rows: list[dict[str, Any]], results: dict[str, dict[int, dict[str, Any]]],
                  truth: dict[str, Any]) -> list[dict[str, Any]]:
    """解析し直した初速と、コマを見て決めた正しい初速の差（参考）。"""
    sys.path.insert(0, str(ROOT / "agent" / "truth"))
    from true_speeds import speed_at

    out = []
    for ref in truth["smashes"]:
        row = next((r for r in rows if r["video"] == ref["video"] and r["id"] == ref["id"]), None)
        if row is None:
            continue
        t_imp = results.get(ref["video"], {}).get(ref["id"], {}).get("t_impact_s")
        kmh = row["kmh"]
        same = None if t_imp is None else round(speed_at(ref, t_imp), 1)
        out.append({"video": ref["video"], "id": ref["id"], "true_kmh": ref["kmh"], "true_sigma_kmh": ref["sigma_kmh"],
                    "kmh": kmh, "err_kmh": None if kmh is None else round(kmh - ref["kmh"], 1),
                    "z": None if kmh is None else round((kmh - ref["kmh"]) / ref["sigma_kmh"], 2),
                    "dt_hit_ms": None if t_imp is None else round((t_imp - ref["t_hit_s"]) * 1000, 1),
                    "true_kmh_at_t": same, "err_same_t_kmh": None if same is None or kmh is None else round(kmh - same, 1),
                    "sigma_same_t_kmh": ref["sigma_same_time_kmh"]})
    return out


def truth_summary(tr: list[dict[str, Any]]) -> str:
    z = [abs(r["z"]) for r in tr if r["z"] is not None]
    e = [abs(r["err_same_t_kmh"]) / r["true_kmh_at_t"] for r in tr if r["err_same_t_kmh"] is not None]
    if not z:
        return ""
    return (f"。正しい初速との差（参考、{len(z)} 本）：|差|/σ の中央値 {float(pd.Series(z).median()):.2f}、2σ を超えたもの "
            f"{sum(v > 2 for v in z)} 本" + (f"、同じ時刻の速さの差の中央値 {float(pd.Series(e).median()) * 100:.0f}%" if e else ""))


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
    truth = compare_truth(rows, results, json.loads(TRUTH.read_text(encoding="utf-8"))) if TRUTH.exists() else []
    devs = [abs(r["dev_pct"]) for r in rows if r["dev_pct"] is not None]
    summary = (f"実動画 {len(todo)} 本・{len(rows)} 本のスマッシュ：最大のずれ {max(devs, default=0):.2f}%"
               + (f"、3% を超えたもの {len(bad)} 本（" + "、".join(
                   f"{r['video']} #{r['id']} {r['base_kmh']}→{r['kmh'] if r['kmh'] is not None else '解析できず'}"
                   for r in bad) + "）" if bad else "")
               + (f"（動画がなく飛ばした：{'、'.join(missing)}）" if missing else "") + truth_summary(truth))
    return {"ok": not bad, "summary": summary, "metrics": {"rows": rows, "max_dev_pct": max(devs, default=0.0),
                                                           "missing": missing, "truth": truth}}


def retrack_baseline(cfg: dict[str, Any]) -> int:
    """基準の打つ人の足元を選手の追跡から出し直して（``retrack_hitters``）、初速を解析し直し、基準を書き直す。

    前の初速は ``kmh_before`` に残す。正しい初速があれば、前後の |差|/σ も出す。
    """
    sys.path.insert(0, str(ROOT))
    from smash_speed.config import load_config
    from smash_speed.players import PersonDetector, PlayerOptions, ensure_model

    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    todo = [(e, v) for e in base["videos"] if (v := find_video(e, cfg.get("videos_dir"))) is not None]
    if not todo:
        print("動画が見つからないので、基準は変えませんでした")
        return 1
    po = PlayerOptions()
    detector = PersonDetector(ensure_model(po.model, progress=lambda s: None), po.min_score, po.nms_iou)
    acfg = load_config()
    before = {(e["name"], c["id"]): c["kmh"] for e in base["videos"] for c in e["candidates"]}
    with tempfile.TemporaryDirectory() as td:
        for e, v in todo:
            n = retrack_hitters(e, v, acfg, detector, Path(td) / e["name"])
            print(f"  {e['name']}：追跡の足元を入れた候補 {n}/{len(e['candidates'])} 本")
        tmp = Path(td) / "baseline.json"
        tmp.write_text(json.dumps(base, ensure_ascii=False), encoding="utf-8")
        results = rerun_all(todo, tmp)
    for e, _ in todo:
        for c in e["candidates"]:
            kmh = results.get(e["name"], {}).get(c["id"], {}).get("kmh")
            c["kmh_before"] = before[(e["name"], c["id"])]
            if kmh is None:
                raise RuntimeError(f"{e['name']} #{c['id']} が解析できなかったので、基準は変えませんでした")
            c["kmh"] = kmh
            h, r = c["hitter"], c["hitter_rough"]
            print(f"  {e['name']} #{c['id']} {c['t_onset']:.1f}s：足元 ({r['x']:.2f}, {r['y']:.2f}) → ({h['x']:.2f}, {h['y']:.2f})"
                  f"、初速 {c['kmh_before']:.1f} → {kmh:.1f} km/h" + (f"（{c['hitter_note']}）" if c.get("hitter_note") else ""))
    if TRUTH.exists():
        truth = json.loads(TRUTH.read_text(encoding="utf-8"))
        for label, key in (("前", "kmh_before"), ("後", "kmh")):
            rows = [{"video": e["name"], "id": c["id"], "kmh": c[key]} for e, _ in todo for c in e["candidates"]]
            tr = compare_truth(rows, {}, truth)
            print(f"  正しい初速との差（{label}）：|差|/σ の合計 {sum(abs(x['z']) for x in tr):.1f}"
                  + "".join(f"、{x['video']} #{x['id']} {x['z']:+.1f}σ" for x in tr))
    base["hitters_retracked"] = time.strftime("%Y-%m-%d %H:%M:%S")
    BASELINE.write_text(json.dumps(base, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"基準を書き直しました → {BASELINE}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--make-baseline", action="store_true", help="本体の outputs から基準を作り直す")
    ap.add_argument("--outputs", help="基準を作る outputs フォルダ（既定は .agent/config.json の main_repo の outputs）")
    ap.add_argument("--retrack-hitters", action="store_true",
                    help="基準の打つ人の足元を選手の追跡から出し直し、初速を解析し直して基準を書き直す（人が決めたときだけ）")
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
    if a.retrack_hitters:
        return retrack_baseline(cfg)
    t = time.time()
    r = run({"root": ROOT, "config": cfg})
    for row in r.get("metrics", {}).get("rows", []):
        print(f"  {row['video']} #{row['id']} {row['t_onset']:.1f}s：基準 {row['base_kmh']} → {row['kmh']} km/h"
              f"（{row['dev_pct']}%）{'' if row['ok'] else '  ← NG ' + (row['error'] or '')}")
    for tr in r.get("metrics", {}).get("truth", []):
        print(f"  正しい初速 {tr['video']} #{tr['id']}：{tr['true_kmh']} ± {tr['true_sigma_kmh']} → 解析 {tr['kmh']} km/h"
              f"（差 {tr['err_kmh']}、{tr['z']}σ）／解析の打った時刻 {tr['dt_hit_ms']} ms ずれ、その時刻の正しい速さ "
              f"{tr['true_kmh_at_t']} km/h との差 {tr['err_same_t_kmh']}（±{tr['sigma_same_t_kmh']}）")
    print(f"[{'飛ばした' if r.get('skipped') else 'OK' if r['ok'] else 'NG'}] {r['summary']}（{time.time() - t:.0f} 秒）")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
