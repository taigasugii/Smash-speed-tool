"""smash-speed の Web アプリ（ローカル専用）。

``smash-speed-gui.bat`` から起動すると、この PC の中だけで動くサーバー（127.0.0.1）を立ち上げ、ブラウザで画面を開く。
動画はこの PC から外に出ない。重い処理（解析・検証など）は ``smash-speed`` コマンドを裏で実行し、その出力を画面に流す。
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from collections import OrderedDict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, request, send_file, send_from_directory

from smash_speed.calibration import calibrate, landmark_table, load_calibration, save_calibration
from smash_speed.camera import Camera
from smash_speed.config import PROJECT_ROOT, load_config
from smash_speed.court import court_segments
from smash_speed.geometry import floor_point
from smash_speed.video_io import VideoMeta, probe

APP_ID = "smash-speed-web"
STATIC = Path(__file__).parent / "static"
STATE_FILE = PROJECT_ROOT / "outputs" / "web_state.json"
DEFAULT_PORT = 8765

app = Flask(__name__, static_folder=None)


# =============================================================================
# 状態
# =============================================================================
class Session:
    """開いている動画・カメラ・設定。1人で使う前提なので、プロセスに1つだけ持つ。"""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.meta: VideoMeta | None = None
        self.cap: cv2.VideoCapture | None = None
        self.pos = -1
        self.cache: OrderedDict[tuple[int, int], bytes] = OrderedDict()
        self.camera: Camera | None = None
        self.state: dict[str, Any] = {"out_root": str(PROJECT_ROOT / "outputs")}
        self.last_ping = 0.0
        if STATE_FILE.exists():
            try:
                self.state.update(json.loads(STATE_FILE.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                pass

    def save(self) -> None:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(self.state, ensure_ascii=False, indent=1), encoding="utf-8")

    def cfg(self) -> dict[str, Any]:
        p = self.state.get("config")
        return load_config(p if p and Path(p).exists() else None)


S = Session()


# =============================================================================
# 裏で動かすコマンド
# =============================================================================
class Task:
    def __init__(self, title: str, commands: list[list[str]]):
        self.id = uuid.uuid4().hex[:10]
        self.title = title
        self.commands = commands
        self.lines: list[str] = []
        self.status = "running"   # running | done | failed | cancelled
        self.proc: subprocess.Popen | None = None
        self.cancelled = False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        for cmd in self.commands:
            if self.cancelled:
                break
            self.lines.append("$ smash-speed " + " ".join(cmd[3:]))
            try:
                self.proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                             env=env, creationflags=flags, text=True, encoding="utf-8", errors="replace")
            except OSError as e:
                self.lines.append(f"起動できませんでした: {e}")
                self.status = "failed"
                return
            for line in self.proc.stdout:
                self.lines.append(line.rstrip("\n"))
            if self.proc.wait() != 0:
                self.status = "cancelled" if self.cancelled else "failed"
                return
        self.status = "cancelled" if self.cancelled else "done"

    def cancel(self) -> None:
        self.cancelled = True
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()


TASKS: dict[str, Task] = {}


def python_exe() -> str:
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and (exe.parent / "python.exe").exists():
        return str(exe.parent / "python.exe")
    return str(exe)


def cli(*args: Any) -> list[str]:
    """``smash-speed`` のサブコマンドを実行するコマンド列（上書き設定があれば付ける）。"""
    cmd = [python_exe(), "-m", "smash_speed.cli"]
    if S.state.get("config"):
        cmd += ["--config", S.state["config"]]
    return cmd + [str(a) for a in args]


def start_task(title: str, commands: list[list[str]]) -> Task:
    t = Task(title, commands)
    TASKS[t.id] = t
    return t


def busy() -> bool:
    return any(t.status == "running" for t in TASKS.values())


# =============================================================================
# 共通
# =============================================================================
def ok(**kw: Any):
    return jsonify({"ok": True, **kw})


def err(msg: str, code: int = 400):
    return jsonify({"ok": False, "error": msg}), code


def body() -> dict[str, Any]:
    return request.get_json(silent=True) or {}


def video_args() -> list[str]:
    a = ["--video", S.state["video"]]
    if S.state.get("capture_fps"):
        a += ["--capture-fps", str(S.state["capture_fps"])]
    return a


def out_root() -> Path:
    return Path(S.state.get("out_root") or PROJECT_ROOT / "outputs")


def meta_dict(m: VideoMeta) -> dict[str, Any]:
    return {"path": m.path, "name": Path(m.path).name, "width": m.width, "height": m.height,
            "fps": m.fps_meta, "capture_fps": m.capture_fps, "slow": m.time_scale != 1.0,
            "n_frames": m.n_frames, "duration": float(m.timestamps[-1]), "vfr": m.is_vfr,
            "has_audio": m.has_audio, "timestamps": [round(float(t), 5) for t in m.timestamps]}


def load_camera() -> dict[str, Any] | None:
    """保存済みのキャリブレーションを読み、動画のサイズに合わせる。"""
    S.camera = None
    p = S.state.get("calib")
    if not p or not Path(p).exists() or S.meta is None:
        return None
    from smash_speed.pipeline import _camera_for_video
    cam, data = load_calibration(p)
    try:
        S.camera = _camera_for_video(cam, S.meta)
    except ValueError:
        return {"error": "この動画とキャリブレーションの画像サイズが合いません"}
    return {"rmse_px": data.get("rmse_px"), "n_points": len(data.get("points", [])),
            "points": {q["name"]: q["image_px"] for q in data.get("points", [])},
            "camera_center": S.camera.center.round(2).tolist(), "warnings": data.get("warnings", []),
            "view": data.get("meta", {}).get("view")}


def project_court(cam: Camera, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """コートのラインとネットを画像に投影した折れ線（画面への重ね描き用）。"""
    out = []
    for a, b, kind in court_segments(cfg):
        s = np.linspace(0, 1, 30)[:, None]
        uv = cam.project(a + s * (b - a))
        pts = [[round(float(u), 1), round(float(v), 1)] for u, v in uv if np.isfinite(u) and np.isfinite(v)]
        if len(pts) >= 2:
            out.append({"kind": kind, "pts": pts})
    return out


# =============================================================================
# ページ・ファイル
# =============================================================================
@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.get("/static/<path:name>")
def static_file(name: str):
    resp = send_from_directory(STATIC, name)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/file")
def file():
    """出力フォルダ・プロジェクトの中のファイルだけを返す（画像・動画・CSV）。"""
    p = Path(request.args.get("path", "")).resolve()
    roots = [out_root().resolve(), PROJECT_ROOT.resolve()]
    if not any(str(p).startswith(str(r)) for r in roots) or not p.is_file():
        abort(404)
    return send_file(p, conditional=True)


@app.get("/api/ping")
def ping():
    S.last_ping = time.time()
    return ok(app=APP_ID, busy=busy())


@app.post("/api/shutdown")
def shutdown():
    for t in TASKS.values():
        t.cancel()
    threading.Timer(0.5, lambda: os._exit(0)).start()
    return ok()


# =============================================================================
# 状態の保存・ファイル選択
# =============================================================================
@app.get("/api/state")
def get_state():
    return ok(state=S.state, project_root=str(PROJECT_ROOT))


@app.post("/api/state")
def set_state():
    S.state.update(body())
    S.save()
    return ok(state=S.state)


@app.post("/api/pick")
def pick():
    """この PC のファイル選択ダイアログを開く（ブラウザからは実際のパスが分からないため）。"""
    b = body()
    kind = b.get("kind", "file")
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    init = b.get("initial") or str(out_root())
    try:
        if kind == "video":
            p = filedialog.askopenfilename(parent=root, title="動画を選ぶ", initialdir=init,
                                           filetypes=[("動画", "*.mp4 *.mov *.MOV *.MP4 *.m4v *.avi"), ("すべて", "*.*")])
        elif kind == "dir":
            p = filedialog.askdirectory(parent=root, title=b.get("title", "フォルダを選ぶ"), initialdir=init)
        elif kind == "save":
            p = filedialog.asksaveasfilename(parent=root, title=b.get("title", "保存先"), initialdir=init,
                                             initialfile=b.get("name", ""), defaultextension=b.get("ext", ""))
        else:
            p = filedialog.askopenfilename(parent=root, title=b.get("title", "ファイルを選ぶ"), initialdir=init,
                                           filetypes=[(b.get("label", "ファイル"), b.get("pattern", "*.*")), ("すべて", "*.*")])
    finally:
        root.destroy()
    return ok(path=p or "")


@app.post("/api/open")
def open_path():
    """ファイル・フォルダを Windows の既定のアプリで開く。"""
    p = Path(body().get("path", ""))
    if not p.exists():
        return err("見つかりません")
    os.startfile(str(p))  # noqa: S606
    return ok()


# =============================================================================
# 動画
# =============================================================================
@app.post("/api/video")
def open_video():
    b = body()
    path = b.get("path") or S.state.get("video")
    if not path or not Path(path).exists():
        return err("動画が見つかりません")
    fps = b.get("capture_fps", S.state.get("capture_fps"))
    fps = float(fps) if fps not in (None, "", 0) else None
    try:
        meta = probe(path, fps)
    except Exception as e:  # noqa: BLE001
        return err(f"動画を読み込めませんでした: {e}")
    with S.lock:
        if S.cap is not None:
            S.cap.release()
        S.meta, S.cap, S.pos = meta, cv2.VideoCapture(meta.path), -1
        S.cache.clear()
    if S.state.get("video") != path:
        stem = Path(path).stem
        S.state.update(video=path, start=None, end=None, shot="", hitter=None, landing=None, roi=None, draft_points={},
                       draft_skipped=[], calib_view=None, last_result_ok=False, last_result=None, audio=None,
                       calib=S.state.get("calib") if S.state.get("calib_video") == path else
                       str(out_root() / f"{stem}_calibration.json"))
    S.state["capture_fps"] = fps
    S.save()
    calib = load_camera()
    return ok(meta=meta_dict(meta), calib=calib, state=S.state)


@app.get("/api/frame")
def frame():
    if S.meta is None:
        abort(404)
    i = int(np.clip(int(request.args.get("i", 0)), 0, S.meta.n_frames - 1))
    w = int(request.args.get("w", 1280))
    key = (i, w)
    with S.lock:
        if key in S.cache:
            data = S.cache[key]
        else:
            if i != S.pos + 1:
                S.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            good, img = S.cap.read()
            if not good:
                abort(404)
            S.pos = i
            if w and w < img.shape[1]:
                img = cv2.resize(img, (w, int(round(img.shape[0] * w / img.shape[1]))), interpolation=cv2.INTER_AREA)
            data = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()
            S.cache[key] = data
            while len(S.cache) > 60:
                S.cache.popitem(last=False)
    resp = app.response_class(data, mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "max-age=600"
    return resp


# =============================================================================
# キャリブレーション
# =============================================================================
@app.get("/api/landmarks")
def landmarks():
    cfg = S.cfg()
    segs = [{"a": a.tolist(), "b": b.tolist(), "kind": k} for a, b, k in court_segments(cfg)]
    return ok(landmarks=[{"name": lm["name"], "xyz": lm["xyz"]} for lm in cfg["landmarks"]], segments=segs,
              court=cfg["court"])


@app.post("/api/calib/solve")
def calib_solve():
    """クリックした点からカメラを推定し、重ね描き用の線と、全ランドマークの予測位置を返す（保存はしない）。"""
    if S.meta is None:
        return err("動画を開いてください")
    b = body()
    pts = {k: v for k, v in (b.get("points") or {}).items() if v}
    cfg = S.cfg()
    table = landmark_table(cfg)
    names = [n for n in table if n in pts]
    if len(names) < 4:
        return err(f"4点以上必要です（今 {len(names)} 点）")
    try:
        res = calibrate(names, np.array([table[n] for n in names]), np.array([pts[n] for n in names]),
                        S.meta.width, S.meta.height, estimate_k1=bool(b.get("k1")),
                        warn_px=cfg["calibration"]["reproj_warn_px"])
    except (RuntimeError, ValueError) as e:
        return err(str(e))
    pred = {}
    for n, X in table.items():
        uv = res.camera.project(X[None])[0]
        if np.isfinite(uv).all():
            pred[n] = [round(float(uv[0]), 1), round(float(uv[1]), 1)]
    return ok(rmse=res.rmse_px, errors={n: float(e) for n, e in zip(res.names, res.errors_px)},
              warnings=res.warnings, overlay=project_court(res.camera, cfg), predicted=pred,
              camera_center=res.camera.center.round(2).tolist(), f=res.camera.f,
              warn_px=cfg["calibration"]["reproj_warn_px"])


@app.post("/api/calib/save")
def calib_save():
    if S.meta is None:
        return err("動画を開いてください")
    b = body()
    pts = {k: v for k, v in (b.get("points") or {}).items() if v}
    cfg = S.cfg()
    table = landmark_table(cfg)
    names = [n for n in table if n in pts]
    path = b.get("path") or S.state.get("calib")
    if not path:
        return err("保存先がありません")
    try:
        res = calibrate(names, np.array([table[n] for n in names]), np.array([pts[n] for n in names]),
                        S.meta.width, S.meta.height, estimate_k1=bool(b.get("k1")),
                        warn_px=cfg["calibration"]["reproj_warn_px"])
    except (RuntimeError, ValueError) as e:
        return err(str(e))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_calibration(res, path, video=S.meta.path, frame_time_s=b.get("time"), extra={"view": b.get("view")})
    S.state.update(calib=path, calib_video=S.meta.path, draft_points={})
    S.save()
    return ok(calib=load_camera(), path=path)


@app.post("/api/calib/load")
def calib_load():
    b = body()
    if b.get("path"):
        S.state["calib"] = b["path"]
        S.save()
    info = load_camera()
    if info is None:
        return ok(calib=None)
    return ok(calib=info, overlay=project_court(S.camera, S.cfg()) if S.camera is not None else [])


@app.get("/api/calib/overlay")
def calib_overlay():
    if S.camera is None:
        return ok(overlay=[])
    return ok(overlay=project_court(S.camera, S.cfg()))


@app.post("/api/floor")
def floor():
    """画像上の点 → 床面（z = 0）上のコート座標 [m]。"""
    if S.camera is None:
        return err("先にキャリブレーションをしてください")
    b = body()
    X = floor_point(S.camera, np.array([[float(b["u"]), float(b["v"])]]))[0]
    if not np.isfinite(X).all():
        return err("床の上の点ではありません（地平線より上をクリックしました）")
    return ok(x=round(float(X[0]), 3), y=round(float(X[1]), 3))


@app.post("/api/project")
def project_points():
    """コート座標 [m] の点 → 画像座標（飛行平面の表示用）。"""
    if S.camera is None:
        return ok(points=[])
    pts = np.array(body().get("points", []), float).reshape(-1, 3)
    uv = S.camera.project(pts) if len(pts) else np.empty((0, 2))
    return ok(points=[[float(u), float(v)] if np.isfinite([u, v]).all() else None for u, v in uv])


# =============================================================================
# 処理の実行
# =============================================================================
@app.post("/api/run/<kind>")
def run(kind: str):
    if busy():
        return err("ほかの処理が実行中です。終わるまで待つか「中止」を押してください。")
    b = body()
    st = S.state
    root = out_root()
    if kind == "onsets":
        args = ["onsets", *video_args()]
        if st.get("audio"):
            args += ["--audio", st["audio"]]
        t = start_task("打球音を探す", [cli(*args)])
    elif kind == "analyze":
        name = st.get("shot") or f"{Path(st['video']).stem}_{float(st['start']):.1f}s"
        out = root / name
        common = [*video_args(), "--start", f"{float(st['start']):.3f}", "--end", f"{float(st['end']):.3f}"]
        args = ["analyze", *common, "--calib", st["calib"], "--out", str(out), "--name", name]
        if st.get("hitter") and st.get("landing"):
            h, la = st["hitter"], st["landing"]
            args += [f"--hitter={h['x']},{h['y']}", f"--landing={la['x']},{la['y']}"]
        args += ["--impact-method", st.get("impact_method") or "auto"]
        if (st.get("impact_method") == "manual") and st.get("impact_time") is not None:
            args += ["--impact-time", str(st["impact_time"])]
        if st.get("audio"):
            args += ["--audio", st["audio"]]
        args += ["--ci", st.get("ci") or "jacobian"]
        if not st.get("overlay", True):
            args.append("--no-overlay")
        args.append("--force")
        cmds = []
        if st.get("detector") == "tracknet":
            out.mkdir(parents=True, exist_ok=True)
            det = out / "detections_tracknet.csv"
            cmds.append(cli("detect", *common, "--detector", "tracknet", "--out", str(det)))
            args += ["--detections", str(det)]
        elif st.get("roi"):
            args += ["--roi", ",".join(f"{v:.0f}" for v in st["roi"])]
        cmds.append(cli(*args))
        st["last_result"] = str(out)
        S.save()
        t = start_task(f"解析：{name}", cmds)
    elif kind == "synth":
        speed, fps = float(b.get("speed", 80)), float(b.get("fps", 240))
        out = root / f"synthetic_{speed:g}mps_{fps:g}fps"
        st["synth_dir"] = str(out)
        t = start_task("合成データの作成", [cli("synth", "--out", out, "--speed", speed, "--fps", fps,
                                               "--exposure", float(b.get("exposure", 0)))])
    elif kind == "lowfps":
        out = b.get("out") or str(root / "lowfps")
        args = ["make-lowfps", *video_args(), "--out", out, "--target-fps", *[str(f) for f in b.get("fps", [30, 60, 120])]]
        if b.get("labels"):
            args += ["--labels", b["labels"]]
        if b.get("blur", True):
            args.append("--blur")
        t = start_task("疑似低 fps 動画の作成", [cli(*args)])
    elif kind == "validate_synthetic":
        t = start_task("精度検証（合成データ）", [cli("validate", "--synthetic", "--n", int(b.get("n", 100)),
                                                    "--out", root / "validation_synthetic")])
    elif kind == "validate_dataset":
        args = ["validate", "--dataset", b["index"], "--out", str(root / "validation")]
        if b.get("reference"):
            args += ["--reference", b["reference"]]
        t = start_task("精度検証（実データ）", [cli(*args)])
    elif kind == "export_tracknet":
        t = start_task("TrackNetV3 の学習データ", [cli("export-tracknet", "--index", b["index"],
                                                       "--out", root / "tracknet_dataset")])
    elif kind == "summarize":
        t = start_task("集計表", [cli("summarize", "--results", root, "--out", root / "summary.csv")])
    else:
        return err("不明な処理です")
    return ok(task=t.id)


@app.get("/api/task/<tid>")
def task(tid: str):
    t = TASKS.get(tid)
    if t is None:
        return err("ありません", 404)
    since = int(request.args.get("since", 0))
    return ok(status=t.status, title=t.title, lines=t.lines[since:], n=len(t.lines))


@app.post("/api/task/<tid>/cancel")
def task_cancel(tid: str):
    t = TASKS.get(tid)
    if t:
        t.cancel()
    return ok()


@app.get("/api/onsets")
def onsets_result():
    t = TASKS.get(request.args.get("task", ""))
    rows = []
    for line in (t.lines if t else []):
        parts = line.split(",")
        if len(parts) == 2:
            try:
                rows.append([float(parts[0]), float(parts[1])])
            except ValueError:
                pass
    if S.meta is not None:
        S.state["onsets"] = {"video": S.meta.path, "rows": rows}
        S.save()
    return ok(rows=rows)


@app.get("/api/synth_info")
def synth_info():
    d = Path(S.state.get("synth_dir", ""))
    f = d / "truth.json"
    if not f.exists():
        return err("合成データがありません")
    info = json.loads(f.read_text(encoding="utf-8"))
    S.state.update(video=str(d / "synthetic.mp4"), calib=str(d / "calibration.json"), calib_video=str(d / "synthetic.mp4"),
                   audio=str(d / "synthetic.wav"), capture_fps=None, start=0.0, end=round(info["t_land_video_s"] + 0.15, 3),
                   shot=d.name + "_analysis", hitter={"x": round(info["hitter_xy"][0], 3), "y": round(info["hitter_xy"][1], 3)},
                   landing={"x": round(info["landing_xy"][0], 3), "y": round(info["landing_xy"][1], 3)},
                   roi=None, impact_method="auto")
    S.save()
    return ok(info=info)


# =============================================================================
# 結果
# =============================================================================
def _row(csv: Path) -> dict[str, Any] | None:
    try:
        r = pd.read_csv(csv).iloc[0].to_dict()
    except (OSError, pd.errors.ParserError, IndexError):
        return None
    return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items()}


@app.get("/api/results")
def results():
    rows = []
    for p in sorted(out_root().glob("**/result.csv"), key=lambda p: p.stat().st_mtime, reverse=True):
        r = _row(p)
        if r:
            r["dir"] = str(p.parent)
            r["mtime"] = p.stat().st_mtime
            rows.append(r)
    return ok(rows=rows)


@app.get("/api/result")
def result():
    d = Path(request.args.get("dir") or S.state.get("last_result") or "")
    r = _row(d / "result.csv")
    if r is None:
        return err("結果がありません", 404)
    log = []
    fr = d / "fit_result.json"
    if fr.exists():
        log = json.loads(fr.read_text(encoding="utf-8")).get("log", [])
    files = {k: str(d / f) for k, f in (("image_fit", "image_fit.png"), ("speed_time", "speed_time.png"),
                                        ("trajectory", "trajectory.png"), ("overlay", "overlay_web.mp4"),
                                        ("overlay_raw", "overlay.mp4"), ("csv", "result.csv")) if (d / f).exists()}
    return ok(row=r, log=log, files=files, dir=str(d))


# =============================================================================
# アノテーション
# =============================================================================
@app.get("/api/labels")
def labels_get():
    d = Path(request.args.get("dir", ""))
    f = d / "labels.csv"
    init = request.args.get("init")
    src = f if f.exists() else (Path(init) if init and Path(init).exists() else None)
    if src is None:
        return ok(rows=[], source=None)
    from smash_speed.annotate import normalize_labels
    df = normalize_labels(pd.read_csv(src))
    rows = []
    for r in df.itertuples():
        vis = int(r.visible) if r.visible == r.visible else -1
        rows.append({"frame": int(r.frame), "x": None if r.x_px != r.x_px else float(r.x_px),
                     "y": None if r.y_px != r.y_px else float(r.y_px), "visible": vis,
                     "event": r.event if isinstance(r.event, str) else "",
                     "source": getattr(r, "source", "") if isinstance(getattr(r, "source", ""), str) else ("init" if src != f else "")})
    if src != f:
        for r in rows:
            if r["visible"] == 0:
                r["visible"] = -1  # 検出できなかったフレームは「未ラベル」として扱う
            r["source"] = "init" if r["visible"] == 1 else ""
    return ok(rows=rows, source=str(src))


@app.post("/api/labels")
def labels_save():
    if S.meta is None:
        return err("動画を開いてください")
    from smash_speed.annotate import AnnotationState, save_labels
    b = body()
    rows = sorted(b.get("rows", []), key=lambda r: r["frame"])
    if not rows:
        return err("ラベルがありません")
    frames = np.array([r["frame"] for r in rows])
    st = AnnotationState(frames, S.meta.timestamps[frames])
    for k, r in enumerate(rows):
        st.visible[k] = int(r.get("visible", -1))
        st.x[k] = r["x"] if r.get("x") is not None else np.nan
        st.y[k] = r["y"] if r.get("y") is not None else np.nan
        st.source[k] = r.get("source", "")
        st.event[k] = r.get("event", "")
    d = Path(b["dir"])
    save_labels(st, d, Path(S.meta.path).stem, S.meta, b.get("start"), b.get("end"))
    return ok(path=str(d / "labels.csv"))


# =============================================================================
# 起動
# =============================================================================
def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _watchdog() -> None:
    """ブラウザを閉じて 2 分たち、実行中の処理もなければ終了する（裏にサーバーが残り続けないように）。"""
    while True:
        time.sleep(15)
        if S.last_ping and time.time() - S.last_ping > 120 and not busy():
            os._exit(0)


def main(open_browser: bool = True, port: int = DEFAULT_PORT) -> None:
    url = f"http://127.0.0.1:{port}/"
    if _port_in_use(port):
        # すでに起動している場合は、ブラウザで開くだけ
        if open_browser:
            webbrowser.open(url)
        return
    threading.Thread(target=_watchdog, daemon=True).start()
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=port, threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    main(open_browser="--no-browser" not in sys.argv,
         port=int(next((a.split("=")[1] for a in sys.argv if a.startswith("--port=")), DEFAULT_PORT)))
