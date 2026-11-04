"""関節・打点を付ける画面（ROADMAP E1・E1b の正解づくり）の API。

動画は「動画のフォルダ」（画面で選んだフォルダ → 開発エージェントの設定 ``videos_dir`` → 開いている動画のフォルダ の順）の
中だけから読む。ファイルは名前（フォルダからの相対の道）で受け取り、フォルダの外を指すものは断る。
付けた関節・打点は出力の場所の ``pose_labels/`` に書く（形は ``smash_speed.pose_label``）。

下書き（関節の位置の提案）は、フォームの解析の結果（``*/form/raw/*.npz``）にそのコマがあればそれを、
なければ選んだ姿勢推定のモデルをこのプロセスで動かして出す。
"""

from __future__ import annotations

import importlib.util
import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from flask import Flask, abort, jsonify, request

from smash_speed import pose_label as PL
from smash_speed.pose import NUM_KEYPOINTS, PoseSequence

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi"}
MAX_VIDEOS = 500
SOUND_DELAY_S = 7.0 / 343.0      # 打った音は、カメラ（約 7 m）まで届く分だけ遅れて聞こえる


def videos_root(state: dict[str, Any]) -> tuple[Path | None, str]:
    """動画のフォルダと、どこで決まったか（setting | agent | video）。"""
    if state.get("label_videos_dir"):
        return Path(state["label_videos_dir"]), "setting"
    try:
        from smash_speed.webapp.agent import agent_dir
        cfg = json.loads((agent_dir() / ".agent" / "config.json").read_text(encoding="utf-8-sig"))
        if cfg.get("videos_dir") and Path(cfg["videos_dir"]).is_dir():
            return Path(cfg["videos_dir"]), "agent"
    except (OSError, ValueError):
        pass
    if state.get("video"):
        return Path(state["video"]).parent, "video"
    return None, ""


def list_videos(root: Path) -> list[dict[str, str]]:
    """フォルダとその 1 つ下のフォルダの動画（名前は同じなら最初の 1 本だけ。ラベルは名前で動画を区別するため）。"""
    if not root.is_dir():
        return []
    found: dict[str, dict[str, str]] = {}
    for p in sorted(list(root.glob("*")) + list(root.glob("*/*"))):
        if p.suffix.lower() in VIDEO_EXT and p.is_file() and p.name not in found:
            found[p.name] = {"name": p.name, "rel": p.relative_to(root).as_posix()}
        if len(found) >= MAX_VIDEOS:
            break
    return sorted(found.values(), key=lambda v: v["rel"])


def video_path(root: Path | None, rel: str) -> Path:
    """フォルダの中の動画の道。フォルダの外・動画でないものは ValueError。"""
    if root is None or not rel:
        raise ValueError("動画のフォルダが決まっていません")
    r = root.resolve()
    p = (r / rel).resolve()
    if r not in p.parents or not p.is_file() or p.suffix.lower() not in VIDEO_EXT:
        raise ValueError("動画のフォルダの中の動画を選んでください")
    return p


class _Videos:
    """選んだ動画のコマを読む（1 本ずつ開いておく）。"""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.metas: dict[tuple[str, float], Any] = {}
        self.path: str | None = None
        self.cap: cv2.VideoCapture | None = None
        self.pos = -1
        self.cache: OrderedDict[tuple[str, int, int], bytes] = OrderedDict()

    def meta(self, path: Path):
        from smash_speed.video_io import probe

        key = (str(path), path.stat().st_mtime)
        if key not in self.metas:
            self.metas[key] = probe(path)
        return self.metas[key]

    def read(self, path: Path, i: int) -> np.ndarray | None:
        from smash_speed.video_io import seek_read

        meta = self.meta(path)
        i = int(np.clip(i, 0, meta.n_frames - 1))
        with self.lock:
            if self.path != str(path):
                if self.cap is not None:
                    self.cap.release()
                self.cap, self.path, self.pos = cv2.VideoCapture(str(path)), str(path), -1
            if i == self.pos + 1:
                good, img = self.cap.read()
                img = img if good else None
            else:
                img = seek_read(self.cap, meta, i)
            self.pos = i if img is not None else -1
            return img

    def jpeg(self, path: Path, i: int, w: int) -> bytes | None:
        key = (str(path), i, w)
        if key in self.cache:
            return self.cache[key]
        img = self.read(path, i)
        if img is None:
            return None
        if w and w < img.shape[1]:
            img = cv2.resize(img, (w, int(round(img.shape[0] * w / img.shape[1]))), interpolation=cv2.INTER_AREA)
        data = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
        self.cache[key] = data
        while len(self.cache) > 80:
            self.cache.popitem(last=False)
        return data


VIDEOS = _Videos()
_MODELS: dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()
_ONSETS: dict[str, list[float]] = {}
_RAW: dict[tuple[str, float], PoseSequence] = {}


def available_models() -> dict[str, bool]:
    return {"mediapipe": importlib.util.find_spec("mediapipe") is not None,
            "rtmpose": importlib.util.find_spec("rtmlib") is not None}


def _people(people: list[tuple[np.ndarray, np.ndarray]]) -> list[dict[str, Any]]:
    """人ごとの関節（大きく写っている順）。"""
    def size(p):
        ok = np.isfinite(p[0]).all(axis=1) & (p[1] >= 0.3)
        if ok.sum() < 3:
            return 0.0
        span = p[0][ok].max(axis=0) - p[0][ok].min(axis=0)
        return float(span[0] * span[1])

    out = []
    for xy, sc in sorted(people, key=size, reverse=True):
        xy = np.asarray(xy, float).reshape(NUM_KEYPOINTS, 2)
        out.append({"xy": [[None, None] if not np.isfinite(p).all() else [round(float(p[0]), 1), round(float(p[1]), 1)]
                           for p in xy],
                    "score": [round(float(s), 3) for s in np.asarray(sc, float).reshape(NUM_KEYPOINTS)]})
    return out


def prefill_from_form(out: Path, name: str, t: float, fps: float) -> list[dict[str, Any]]:
    """フォームの解析で推定したままの姿勢（``raw/*.npz``）に、そのコマがあればその人。"""
    tol = 0.5 / fps if fps else 1e-3
    for res in PL.form_results(out):
        if res["video"] != name:
            continue
        for p in sorted((res["dir"] / "raw").glob("*.npz")):
            key = (str(p), p.stat().st_mtime)
            if key not in _RAW:
                _RAW[key] = PoseSequence.load(p)
            seq = _RAW[key]
            if len(seq) == 0 or not (seq.t[0] - tol <= t <= seq.t[-1] + tol):
                continue
            i = int(np.argmin(np.abs(seq.t - t)))
            if abs(seq.t[i] - t) <= tol and np.isfinite(seq.xy[i]).any():
                return _people([(seq.xy[i], seq.score[i])])
    return []


def prefill_from_model(model: str, img: np.ndarray) -> list[dict[str, Any]]:
    from smash_speed.pose import make_model

    with _MODEL_LOCK:
        if model not in _MODELS:
            _MODELS[model] = make_model(model)
        return _people(_MODELS[model](img))


def swings(out: Path, name: str, timestamps: np.ndarray) -> list[dict[str, Any]]:
    """付けるスイングの候補（手で付けた打点 → フォームの解析の打点 → 打った音 の順。0.3 秒以内は同じスイング）。"""
    cands = [(r["t_s"], "手で付けた打点") for r in PL.read_impacts(out / PL.LABELS_DIR / PL.IMPACTS_FILE, name)]
    for res in PL.form_results(out):
        if res["video"] == name:
            cands += [(float(r["t_impact_s"]), "フォームの解析") for r in res["rows"] if r.get("t_impact_s") is not None]
    cands += [(t, "打った音") for t in _ONSETS.get(name, [])]
    chosen: list[tuple[float, str]] = []
    for t, src in cands:
        if all(abs(t - c) > PL.SAME_SWING_S for c, _ in chosen):
            chosen.append((t, src))
    return [{"t": round(t, 4), "source": src, "plan": PL.plan_frames(t, timestamps)} for t, src in sorted(chosen)]


def install(app: Flask, out_root: Callable[[], Path], state: Callable[[], dict[str, Any]]) -> None:
    def labels_dir() -> Path:
        return out_root() / PL.LABELS_DIR

    def chosen() -> tuple[Path, Any]:
        """引数の動画（``video``：フォルダからの相対の道）とその情報。"""
        src = request.args if request.method == "GET" else (request.get_json(silent=True) or {})
        root, _ = videos_root(state())
        p = video_path(root, str(src.get("video") or ""))
        return p, VIDEOS.meta(p)

    def fail(e: Exception, code: int = 400):
        return jsonify(ok=False, error=str(e)), code

    @app.get("/api/poselab/videos")
    def poselab_videos():
        root, how = videos_root(state())
        vids = list_videos(root) if root else []
        joints = PL.read_joints(labels_dir() / PL.JOINTS_FILE)
        prog = PL.progress(joints)
        imps = PL.read_impacts(labels_dir() / PL.IMPACTS_FILE)
        for v in vids:
            v["frames"] = prog["per_video"].get(v["name"], 0)
            v["impacts"] = sum(r["video"] == v["name"] for r in imps)
        return jsonify(ok=True, root=str(root) if root else "", root_from=how, videos=vids, progress=prog,
                       impacts=len(imps), labels_dir=str(labels_dir()), models=available_models())

    @app.get("/api/poselab/video")
    def poselab_video():
        try:
            p, m = chosen()
        except (OSError, ValueError) as e:
            return fail(e)
        ts = np.asarray(m.timestamps, float)
        joints = PL.read_joints(labels_dir() / PL.JOINTS_FILE)
        pre = PL.read_prefill(labels_dir() / PL.PREFILL_FILE)
        done = {str(f): {"n": int((d["visible"] > 0).sum()), "prefill": pre.get((p.name, f), "")}
                for (v, f), d in joints.items() if v == p.name}
        return jsonify(ok=True, meta={"name": p.name, "width": m.width, "height": m.height, "fps": m.capture_fps,
                                      "n_frames": m.n_frames, "duration": float(ts[-1]), "has_audio": m.has_audio,
                                      "timestamps": [round(float(t), 5) for t in ts]},
                       swings=swings(out_root(), p.name, ts), done=done,
                       impacts=PL.read_impacts(labels_dir() / PL.IMPACTS_FILE, p.name),
                       has_form=any(r["video"] == p.name for r in PL.form_results(out_root())))

    @app.post("/api/poselab/onsets")
    def poselab_onsets():
        try:
            p, m = chosen()
        except (OSError, ValueError) as e:
            return fail(e)
        from smash_speed.form import FormOptions, _merge_close, video_onsets

        opts = FormOptions()
        on = video_onsets(p, m.time_scale) if m.has_audio else []
        _ONSETS[p.name] = [t - SOUND_DELAY_S for t in _merge_close([t for t, s in on if s >= opts.audio_min_strength],
                                                                     opts.min_gap_s)]
        return jsonify(ok=True, n=len(_ONSETS[p.name]), has_audio=m.has_audio,
                       swings=swings(out_root(), p.name, np.asarray(m.timestamps, float)))

    @app.get("/api/poselab/frame")
    def poselab_frame():
        try:
            p, _ = chosen()
            data = VIDEOS.jpeg(p, int(request.args.get("i", 0)), int(request.args.get("w", 1920)))
        except (OSError, ValueError):
            abort(404)
        if data is None:
            abort(404)
        resp = app.response_class(data, mimetype="image/jpeg")
        resp.headers["Cache-Control"] = "max-age=600"
        return resp

    @app.get("/api/poselab/joints")
    def poselab_joints_get():
        try:
            p, _ = chosen()
            frame = int(request.args.get("frame", -1))
        except (OSError, ValueError) as e:
            return fail(e)
        d = PL.read_joints(labels_dir() / PL.JOINTS_FILE).get((p.name, frame))
        if d is None:
            return jsonify(ok=True, joints=None)
        pre = PL.read_prefill(labels_dir() / PL.PREFILL_FILE).get((p.name, frame), "")
        return jsonify(ok=True, prefill=pre, joints=[[float(x), float(y), int(v)] if v else None
                                                     for (x, y), v in zip(d["xy"], d["visible"])])

    @app.post("/api/poselab/joints")
    def poselab_joints_save():
        b = request.get_json(silent=True) or {}
        try:
            p, m = chosen()
            frame = int(b.get("frame"))
            if not 0 <= frame < m.n_frames:
                raise ValueError("コマの番号が動画の外です")
            xy, vis = PL.check_joints(b.get("joints"))
        except (OSError, ValueError, TypeError) as e:
            return fail(e)
        prog = PL.save_frame(labels_dir(), p.name, frame, float(m.timestamps[frame]), xy, vis,
                             str(b.get("prefill") or "")[:40])
        return jsonify(ok=True, progress=prog, path=str(labels_dir() / PL.JOINTS_FILE))

    @app.post("/api/poselab/joints/delete")
    def poselab_joints_delete():
        b = request.get_json(silent=True) or {}
        try:
            p, _ = chosen()
            prog = PL.delete_frame(labels_dir(), p.name, int(b.get("frame")))
        except (OSError, ValueError, TypeError) as e:
            return fail(e)
        return jsonify(ok=True, progress=prog)

    @app.get("/api/poselab/prefill")
    def poselab_prefill():
        model = request.args.get("model", "form")
        try:
            p, m = chosen()
            frame = int(request.args.get("frame", -1))
            if not 0 <= frame < m.n_frames:
                raise ValueError("コマの番号が動画の外です")
        except (OSError, ValueError) as e:
            return fail(e)
        t = float(m.timestamps[frame])
        if model == "form":
            return jsonify(ok=True, model="form", people=prefill_from_form(out_root(), p.name, t, m.capture_fps))
        if model not in available_models() or not available_models()[model]:
            return fail(ValueError("このモデルは入っていません（pip install -e .[pose]）"))
        img = VIDEOS.read(p, frame)
        if img is None:
            return fail(ValueError("コマを読めませんでした"))
        try:
            people = prefill_from_model(model, img)
        except Exception as e:  # noqa: BLE001  モデルの取得・実行の失敗は画面に出す
            return fail(RuntimeError(f"下書きを作れませんでした: {e}"), 500)
        return jsonify(ok=True, model=model, people=people)

    @app.post("/api/poselab/impacts")
    def poselab_impact_add():
        b = request.get_json(silent=True) or {}
        try:
            p, m = chosen()
            frame = int(b.get("frame"))
            if not 0 <= frame < m.n_frames:
                raise ValueError("コマの番号が動画の外です")
        except (OSError, ValueError, TypeError) as e:
            return fail(e)
        rows = PL.add_impact(labels_dir(), p.name, frame, float(m.timestamps[frame]))
        return jsonify(ok=True, impacts=rows, swings=swings(out_root(), p.name, np.asarray(m.timestamps, float)))

    @app.post("/api/poselab/impacts/delete")
    def poselab_impact_delete():
        b = request.get_json(silent=True) or {}
        try:
            p, m = chosen()
            rows = PL.delete_impact(labels_dir(), p.name, int(b.get("frame")))
        except (OSError, ValueError, TypeError) as e:
            return fail(e)
        return jsonify(ok=True, impacts=rows, swings=swings(out_root(), p.name, np.asarray(m.timestamps, float)))
