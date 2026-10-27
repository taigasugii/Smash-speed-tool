"""エージェントの品質チェック：これが通らない変更は main に入れない。

1. テスト（pytest）がすべて通る
2. 合成データでのスマッシュの初速の誤差（MAPE）が、基準（``agent/baseline.json``）より悪くなっていない
3. ``agent/checks/check_*.py`` の追加のチェック（実動画・ショットの種類など。エージェントが足していく）

使い方::

    .venv\\Scripts\\python.exe agent\\eval_gate.py                # すべて
    .venv\\Scripts\\python.exe agent\\eval_gate.py --skip-tests   # テスト以外（調べるとき用）
    .venv\\Scripts\\python.exe agent\\eval_gate.py --update-baseline   # 基準を今の値にする（人が決めたときだけ）

追加のチェックは ``run(ctx) -> dict`` を持つファイル。``dict`` は ``ok``（bool）、``summary``（1 行の日本語）、
任意で ``skipped``（動画がないなどで調べなかった）と ``metrics``。``ctx`` は ``root``（リポジトリ）と
``config``（``.agent/config.json`` の中身。``main_repo``・``videos_dir`` など）。

結果は ``.agent/gate.json`` に書く。``tree`` は調べた時点の作業ツリーの中身のハッシュ。
``run.ps1`` はこのファイルを信じず、回の終わりに自分でもう一度このチェックを実行する。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASELINE = Path(__file__).with_name("baseline.json")
CHECKS = Path(__file__).with_name("checks")
STATE = ROOT / ".agent"

# 合成データの条件（基準と同じでないと比べられない）
SYN_SHOTS = 30
SYN_FPS = [30, 240]
SYN_SEED = 0
SYN_KEYS = [("physics", "audio"), ("physics", "hitter_position")]
TOLERANCE_PT = 0.5      # 基準からこれだけ（パーセントポイント）悪くなったら失敗


def worktree_tree(root: Path = ROOT) -> str | None:
    """作業ツリーの今の中身（無視するファイルを除く）の tree ハッシュ。Git がなければ None。"""
    try:
        git_dir = subprocess.run(["git", "rev-parse", "--absolute-git-dir"], cwd=root, capture_output=True,
                                 text=True, check=True).stdout.strip()
        with tempfile.TemporaryDirectory() as td:
            idx = Path(td) / "index"
            if (Path(git_dir) / "index").exists():
                shutil.copy(Path(git_dir) / "index", idx)
            env = {**os.environ, "GIT_INDEX_FILE": str(idx)}
            subprocess.run(["git", "add", "-A"], cwd=root, env=env, capture_output=True, check=True)
            return subprocess.run(["git", "write-tree"], cwd=root, env=env, capture_output=True, text=True,
                                  check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_tests() -> dict[str, Any]:
    t = time.time()
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    tail = (r.stdout + r.stderr).strip().splitlines()[-15:]
    counts = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|error|errors|skipped)", "\n".join(tail))}
    ok = r.returncode == 0
    summary = f"テスト：{counts.get('passed', 0)} 件通過" + ("" if ok else f"、{counts.get('failed', 0) + counts.get('error', 0) + counts.get('errors', 0)} 件失敗")
    return {"name": "tests", "ok": ok, "summary": summary, "metrics": counts, "seconds": round(time.time() - t),
            "detail": "\n".join(tail) if not ok else ""}


def synthetic_mape() -> dict[str, float]:
    sys.path.insert(0, str(ROOT))
    from smash_speed.config import load_config
    from smash_speed.validate import synthetic_study

    with tempfile.TemporaryDirectory() as td:
        df = synthetic_study(load_config(), Path(td), n_shots=SYN_SHOTS, fps_list=SYN_FPS, seed=SYN_SEED)
    df = df[df.est_kmh.notna()]
    out = {}
    for m, imp in SYN_KEYS:
        for fps in SYN_FPS:
            g = df[(df.method == m) & (df.impact == imp) & (df.fps == fps)]
            out[f"{m}/{imp}/{fps}fps"] = round(float(((g.est_kmh - g.true_kmh).abs() / g.true_kmh).mean() * 100), 3)
            out[f"{m}/{imp}/{fps}fps/n"] = int(len(g))
    return out


def check_synthetic(update: bool) -> dict[str, Any]:
    t = time.time()
    cur = synthetic_mape()
    if update or not BASELINE.exists():
        BASELINE.write_text(json.dumps({"synthetic_mape": cur, "conditions": {
            "n_shots": SYN_SHOTS, "fps": SYN_FPS, "seed": SYN_SEED}}, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8")
        return {"name": "synthetic", "ok": True, "summary": "合成データ：基準を今の値にしました", "metrics": cur,
                "seconds": round(time.time() - t)}
    base = json.loads(BASELINE.read_text(encoding="utf-8"))["synthetic_mape"]
    worse, parts = [], []
    for k, b in base.items():
        if k.endswith("/n"):
            if cur.get(k, 0) < b:      # 推定できた本数が減った
                worse.append(f"{k[:-2]} の推定できた本数 {b}→{cur.get(k, 0)}")
            continue
        c = cur.get(k)
        parts.append(f"{k} {c}%（基準 {b}%）")
        if c is None or c > b + TOLERANCE_PT:
            worse.append(f"{k} {b}%→{c}%")
    ok = not worse
    return {"name": "synthetic", "ok": ok, "metrics": cur, "seconds": round(time.time() - t),
            "summary": "合成データの初速 MAPE：" + ("、".join(parts) if ok else "悪くなった：" + "、".join(worse))}


def run_extra_checks(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    res = []
    for p in sorted(CHECKS.glob("check_*.py")):
        t = time.time()
        try:
            spec = importlib.util.spec_from_file_location(p.stem, p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            r = dict(mod.run({"root": ROOT, "config": cfg}))
            r.setdefault("summary", "")
            r["ok"] = bool(r.get("ok"))
        except Exception:  # noqa: BLE001  チェック自体が壊れていたら失敗として扱う
            r = {"ok": False, "summary": "チェックが途中で止まりました", "detail": traceback.format_exc()[-2000:]}
        r["name"] = p.stem
        r["seconds"] = round(time.time() - t)
        res.append(r)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--out", default=str(STATE / "gate.json"))
    a = ap.parse_args(argv)
    if (sys.stdout.encoding or "").lower() != "utf-8":     # パイプに書くとき（エージェント・run.ps1）も文字化けしないように
        sys.stdout.reconfigure(encoding="utf-8")

    cfg_path = STATE / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    tree = worktree_tree()
    steps = [] if a.skip_tests else [run_tests()]
    steps.append(check_synthetic(a.update_baseline))
    steps += run_extra_checks(cfg)

    ok = all(s["ok"] for s in steps) and not a.skip_tests
    for s in steps:
        mark = "飛ばした" if s.get("skipped") else ("OK" if s["ok"] else "NG")
        print(f"[{mark}] {s['name']}: {s['summary']}（{s['seconds']} 秒）")
        if not s["ok"] and s.get("detail"):
            print(s["detail"])
    print("品質チェック：" + ("テストを飛ばしたので判定なし" if a.skip_tests else "通った" if ok else "通らなかった"))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"ok": ok, "tree": tree, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": steps},
                              indent=1, ensure_ascii=False), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
