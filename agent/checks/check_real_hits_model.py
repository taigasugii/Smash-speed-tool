"""実動画の打球のチェック（学習したモデル）：ショットの一覧の既定の検出（ft1。``detection.shot_list_model``）で、
``check_real_hits`` と同じ 4 区間の打球を見つけ、正解と比べる（I1）。

検出だけを学習したモデル（``tracknet_train.detect_video``、閾値は ``shot_list.TRACKNET_THRESHOLD``）に替え、後の流れ
（シャトルらしい軌道の絞り込み → 選手の位置 → ``hits.find_hits``）と数え方は ``check_real_hits`` と同じ。基準は
``real_hits_clips.json`` の各区間の ``expected_model``。全体の再現率・適合率・打った側の正解の割合のどれかが、基準から
0.03 より下がったら失敗にする。

モデルのファイルがないか、熱の地図を作れない（この Python に torch がなく、隣の ``.onnx`` と onnxruntime も
``.venv-train`` もない）PC では飛ばす。デスクトップ（ONNX・DirectML）で 4 区間 2 分半ほど。

使い方::

    .venv\\Scripts\\python.exe agent\\checks\\check_real_hits_model.py              # チェックだけ
    .venv\\Scripts\\python.exe agent\\checks\\check_real_hits_model.py --make-baseline   # 今の結果を基準にする（人が決めたときだけ）
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_KEY = "expected_model"
LABEL = "実動画の打球（学習したモデル）"
MARGIN = 0.03           # 基準からこれより下がったら失敗（打った側も同じ幅）


def _base():
    """同じフォルダーの ``check_real_hits``（区間の読み方・数え方を共有する）。"""
    p = Path(__file__).with_name("check_real_hits.py")
    spec = importlib.util.spec_from_file_location("check_real_hits", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def model_path() -> tuple[Path | None, str | None]:
    """使う学習したモデルと、熱の地図の作り方（``torch``・``onnx``・``train-env``）。使えなければ (None, 飛ばす理由)。"""
    sys.path.insert(0, str(ROOT))
    from smash_speed.config import PROJECT_ROOT, load_config
    from smash_speed.train.tracknet_train import predict_backend

    p = (load_config().get("detection") or {}).get("shot_list_model")
    if not p:
        return None, "設定に学習したモデル（detection.shot_list_model）がない"
    p = Path(p) if Path(p).is_absolute() else PROJECT_ROOT / p
    if not p.exists():
        return None, f"モデル {p.name} がない"
    how = predict_backend(p)
    if how is None:
        return None, "熱の地図を作れない（torch・.venv-train・隣の .onnx と onnxruntime のどれもない）"
    return p, how


def run(ctx: dict[str, Any], keep_hits: bool = False) -> dict[str, Any]:
    model, how = model_path()
    if model is None:
        return {"ok": True, "skipped": True, "summary": f"{LABEL}：{how}ので飛ばしました"}
    r = _base().run(ctx, keep_hits, tracknet=model, expected_key=EXPECTED_KEY, label=LABEL, side_margin=MARGIN)
    if not r.get("skipped"):
        r["summary"] += f"（{model.name}・{how}）"
        r["metrics"]["model"] = model.name
        r["metrics"]["backend"] = how
    return r


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--make-baseline", action="store_true", help="今の結果を基準（expected_model）にする")
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
        clips = _base().CLIPS
        data = json.loads(clips.read_text(encoding="utf-8"))
        for c in data["clips"]:
            row = r["metrics"]["clips"].get(c["name"])
            if row:
                c[EXPECTED_KEY] = {k: row[k] for k in ("n_true", "n_found", "tp", "side_correct")}
        clips.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"基準を書き直しました → {clips}")
    print(f"[{'飛ばした' if r.get('skipped') else 'OK' if r['ok'] else 'NG'}] {r['summary']}（{time.time() - t:.0f} 秒）")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
