"""フォームを比べる画面（ROADMAP E4）の API。

出力の場所の下にある ``*/form/form_features.csv``（``smash_speed.form`` の結果）をすべて集め、
ショットを並べて 2 本を選び、正規化した姿勢（棒人間）と特徴量を重ねて比べる。
解析は ``python -m smash_speed.form`` を裏で実行する（サーバーの処理の仕組みを使う）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from flask import Flask, jsonify, request

from smash_speed.form import FEATURES_FILE, FORM_DIR, INFO_FILE, read_rows
from smash_speed.pose import KEYPOINTS, SKELETON
from smash_speed.pose_features import FEATURE_INFO, angle_deg, body_height, turn_angles
from smash_speed.pose_prep import NormalizedPose

# 一覧に出す特徴量（多すぎると表が読めないので、主なものだけ）
LIST_FEATURES = ("wrist_height_ratio", "elbow_angle_deg", "shoulder_angle_deg", "wrist_speed_max",
                 "twist_max_deg", "chain_in_order", "step_forward_ratio")


def _date_of(d: Path, info: dict[str, Any]) -> str:
    """撮った日（動画のファイルの更新日時 → 解析した日時）。"""
    video = Path(info.get("video") or "")
    try:
        if video.exists():
            return time.strftime("%Y-%m-%d", time.localtime(video.stat().st_mtime))
    except OSError:
        pass
    return str(info.get("analyzed_at") or "")[:10]


def find_form_dirs(root: Path) -> list[Path]:
    """出力の場所の下の、フォームの解析の結果（``form/form_features.csv`` を持つ場所）。"""
    root = Path(root)
    if not root.exists():
        return []
    found = {p.parent for p in root.glob(f"*/{FORM_DIR}/{FEATURES_FILE}")}
    found |= {p.parent for p in root.glob(f"{FORM_DIR}/{FEATURES_FILE}")}
    return sorted(found)


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def list_shots(root: Path) -> list[dict[str, Any]]:
    """すべてのショット（撮った日・動画・時刻の順）。``analysis.csv``（E7）があれば、そちらの列も足す。"""
    out = []
    for d in find_form_dirs(root):
        try:
            info = json.loads((d / INFO_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            info = {}
        date = _date_of(d, info)
        extra: dict[str, dict[str, Any]] = {}
        if (d / "analysis.csv").exists():
            extra = {r["shot_id"]: r for r in read_rows(d / "analysis.csv") if r.get("shot_id")}
        for r in read_rows(d / FEATURES_FILE):
            if not (d / "poses" / f"{r['shot_id']}.npz").exists():
                continue
            row = {**extra.get(r["shot_id"], {}), **r}
            item = {"dir": str(d), "id": r["shot_id"], "video": r.get("video"), "date": date,
                    "t": _num(r.get("t_impact_s")), "method": r.get("impact_method"),
                    "features": {k: _num(row.get(k)) for k in FEATURE_INFO},
                    "score": _num(row.get("score_dtw")), "score_cos": _num(row.get("score_cos")),
                    "ball_speed_kmh": _num(row.get("ball_speed_kmh"))}
            out.append(item)
    out.sort(key=lambda s: (s["date"], s["video"] or "", s["t"] or 0.0))
    return out


def _clean(a: np.ndarray, d: int = 3) -> list:
    """JSON にする（NaN は null）。"""
    a = np.round(np.asarray(a, float), d)
    return [None if not np.isfinite(v) else float(v) for v in a.ravel()] if a.ndim == 1 else \
        [_clean(x, d) for x in a]


def pose_payload(form_dir: str | Path, shot: str) -> dict[str, Any]:
    """1 本のショットの正規化した姿勢と、時間の列（肘・肩の角度、胸の回り、手首の速さ）。"""
    d = Path(form_dir)
    path = (d / "poses" / f"{shot}.npz").resolve()
    if d.resolve() not in path.parents:
        raise ValueError("ショットの ID が正しくありません")
    n = NormalizedPose.load(path)
    H = body_height(n)
    sh, el, wr = n.joint("right_shoulder"), n.joint("right_elbow"), n.joint("right_wrist")
    v = np.hypot(*np.gradient(wr, n.t, axis=0).T) / H if np.isfinite(H) else np.full(len(n.t), np.nan)
    turn = turn_angles(n, "shoulder")
    series = {
        "elbow": angle_deg(sh, el, wr),
        "shoulder": angle_deg(el, sh, n.joint("right_hip")),
        "trunk": turn if turn is not None else np.full(len(n.t), np.nan),
        "wrist_speed": v,
    }
    return {"t": _clean(n.t), "xy": _clean(n.xy), "hip": _clean(n.hip), "height": _num(H),
            "series": {k: _clean(s, 2) for k, s in series.items()},
            "keypoints": list(KEYPOINTS), "skeleton": SKELETON, "meta": n.meta}


def install(app: Flask, out_root: Callable[[], Path], state: Callable[[], dict[str, Any]],
            run_task: Callable[[str, list[list[str]]], Any], python_exe: Callable[[], str],
            busy: Callable[[], bool]) -> None:
    @app.get("/api/form")
    def form_list():
        # 特徴量の名前は並び順を保つため [名前, 表示名, 単位] の並びで返す（辞書は JSON で並べ替わる）
        return jsonify(ok=True, shots=list_shots(out_root()), info=[[k, *v] for k, v in FEATURE_INFO.items()],
                       list_features=LIST_FEATURES)

    @app.get("/api/form/pose")
    def form_pose():
        try:
            return jsonify(ok=True, **pose_payload(request.args.get("dir", ""), request.args.get("id", "")))
        except (OSError, ValueError, KeyError) as e:
            return jsonify(ok=False, error=f"姿勢を読めません: {e}"), 400

    @app.post("/api/run/form")  # サーバーの /api/run/<kind> より先に当たる（決まった道の方が優先）
    def form_run():
        if busy():
            return jsonify(ok=False, error="ほかの処理が実行中です。終わるまで待つか「中止」を押してください。"), 400
        st = state()
        if not st.get("video"):
            return jsonify(ok=False, error="先に「ホーム」で動画を選んでください。"), 400
        b = request.get_json(silent=True) or {}
        out = out_root() / f"{Path(st['video']).stem}_form"
        cmd = [python_exe(), "-m", "smash_speed.form", st["video"], "--out", str(out),
               "--model", b.get("model") or "rtmpose", "--distance", str(float(b.get("distance") or 7.0))]
        if b.get("left_handed"):
            cmd.append("--left-handed")
        # 続けて、1 ショット 1 行の表（E7）も作り直す
        t = run_task("フォームの解析", [cmd, [python_exe(), "-m", "smash_speed.form_table", str(out)]])
        return jsonify(ok=True, task=t.id, dir=str(out))
