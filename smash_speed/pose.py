"""姿勢推定（ROADMAP E1）：動画のコマから、打つ人の関節の位置と信頼度を出す。

関節は COCO の 17 点（``KEYPOINTS``）にそろえる。モデルは差し替えられる形にしていて、
「BGR の画像 → 写っている人ごとの（関節の座標 (17, 2) [px]、信頼度 (17,)）の並び」を返す呼び出せるもの
（``PoseModel``）なら何でもよい。テストでは、合成の骨格（``synth_pose``）を返す偽物のモデルを使う。

用意しているモデル（どちらも Apache License 2.0。``docs/pose_features.md`` に比べ方と注意を書いた）：

- ``rtmpose``：RTMPose（rtmlib。人の検出 YOLOX → 関節 RTMPose の 2 段）。ONNX のモデルなので、
  ``smash_speed.accel.make_session`` に載せ替えて、デスクトップでは DirectML の GPU で動かす。
  重みは初めて使うときに rtmlib が OpenMMLab の配布元から取ってくる（``balanced`` で合わせて約 100 MB）。
- ``mediapipe``：MediaPipe Pose Landmarker（33 点を 17 点に読み替える）。Python 版は CPU で動く。
  重み（lite 約 6 MB・full 約 9 MB・heavy 約 30 MB）は初めて使うときに ``ckpts/pose/`` に取ってくる。

1 コマに複数の人が写るときは、前のコマの打つ人にいちばん近い人（最初のコマは関節の範囲がいちばん大きい人）を選ぶ
（``pick_person``）。信頼度の低い関節（隠れた非利き腕など）は ``clean`` で欠け（NaN）にし、短い欠けだけ前後から埋める。

精度の測り方（PCK）は ``load_labels``・``pck``。手で付けた関節の位置の CSV の形は ``LABEL_COLUMNS``。

``python -m smash_speed.pose estimate <動画> --model rtmpose --start 10 --end 12 --out pose.npz`` で推定し、
``python -m smash_speed.pose pck <labels.csv> <pose.npz> [<pose2.npz> ...]`` でモデルを比べる。
"""

from __future__ import annotations

import argparse
import csv
import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

import numpy as np

# COCO の 17 点（順番も COCO と同じ）
KEYPOINTS = ("nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
             "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
             "left_knee", "right_knee", "left_ankle", "right_ankle")
KP = {name: i for i, name in enumerate(KEYPOINTS)}
NUM_KEYPOINTS = len(KEYPOINTS)
KEYPOINT_LABELS = {"nose": "鼻", "left_eye": "左目", "right_eye": "右目", "left_ear": "左耳", "right_ear": "右耳",
                   "left_shoulder": "左肩", "right_shoulder": "右肩", "left_elbow": "左肘", "right_elbow": "右肘",
                   "left_wrist": "左手首", "right_wrist": "右手首", "left_hip": "左腰", "right_hip": "右腰",
                   "left_knee": "左膝", "right_knee": "右膝", "left_ankle": "左足首", "right_ankle": "右足首"}
# 左右を入れ替えるときの相手（左利きの反転などに使う）
FLIP_INDEX = [KP[n.replace("left_", "@").replace("right_", "left_").replace("@", "right_")] for n in KEYPOINTS]
# 骨格を線で描くときの組
SKELETON = [("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"), ("left_shoulder", "left_hip"),
            ("right_shoulder", "right_hip"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"), ("left_hip", "left_knee"),
            ("left_knee", "left_ankle"), ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
            ("nose", "left_eye"), ("nose", "right_eye"), ("left_eye", "left_ear"), ("right_eye", "right_ear")]

MODELS = ("rtmpose", "mediapipe")
MODEL_DIR = Path(__file__).resolve().parents[1] / "ckpts" / "pose"
MEDIAPIPE_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_{v}/float16/latest/"
                 "pose_landmarker_{v}.task")
# MediaPipe の 33 点のうち、COCO の 17 点に当たるもの（同じ順番）
MEDIAPIPE_TO_COCO = (0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)

MIN_SCORE = 0.3          # これより信頼度の低い関節は欠けにする
MAX_GAP_FRAMES = 3       # これ以下の長さの欠けは、前後から直線で埋める


@dataclass
class PoseSequence:
    """1 人の関節の位置の時間の列。

    Attributes:
        t: コマの時刻 (T,) [s]（動画の実時間）
        xy: 関節の画像の座標 (T, 17, 2) [px]。見つからなければ NaN
        score: 関節の信頼度 (T, 17)（0〜1。見つからなければ 0）
        width, height: 画像の大きさ [px]
        fps: コマの速さ [Hz]
        model: 推定に使ったモデルの名前
        meta: そのほかの情報（動画の名前・左利きか・反転したかなど）
    """

    t: np.ndarray
    xy: np.ndarray
    score: np.ndarray
    width: int = 0
    height: int = 0
    fps: float = 0.0
    model: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, float).reshape(-1)
        self.xy = np.asarray(self.xy, float).reshape(len(self.t), NUM_KEYPOINTS, 2)
        self.score = np.asarray(self.score, float).reshape(len(self.t), NUM_KEYPOINTS)

    def __len__(self) -> int:
        return len(self.t)

    def joint(self, name: str) -> np.ndarray:
        """関節 ``name`` の座標 (T, 2)。"""
        return self.xy[:, KP[name]]

    def copy(self, **changes: Any) -> "PoseSequence":
        d = {"t": self.t.copy(), "xy": self.xy.copy(), "score": self.score.copy(), "width": self.width,
             "height": self.height, "fps": self.fps, "model": self.model, "meta": dict(self.meta)}
        d.update(changes)
        return PoseSequence(**d)

    def slice(self, t0: float, t1: float) -> "PoseSequence":
        """時刻 [t0, t1] のコマだけ。"""
        keep = (self.t >= t0 - 1e-9) & (self.t <= t1 + 1e-9)
        return self.copy(t=self.t[keep], xy=self.xy[keep], score=self.score[keep])

    def save(self, path: str | Path) -> Path:
        """``.npz`` に書き出す。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, t=self.t, xy=self.xy, score=self.score,
                            info=json.dumps({"width": self.width, "height": self.height, "fps": self.fps,
                                             "model": self.model, "meta": self.meta}, ensure_ascii=False))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PoseSequence":
        with np.load(path, allow_pickle=False) as z:
            info = json.loads(str(z["info"]))
            return cls(t=z["t"], xy=z["xy"], score=z["score"], width=info["width"], height=info["height"],
                       fps=info["fps"], model=info["model"], meta=info.get("meta", {}))


# --- モデル ------------------------------------------------------------------------

Person = tuple[np.ndarray, np.ndarray]  # （関節の座標 (17, 2) [px]、信頼度 (17,)）


class PoseModel(Protocol):
    """姿勢推定のモデル：BGR の画像から、写っている人ごとの関節を返す。"""

    name: str

    def __call__(self, image: np.ndarray) -> list[Person]: ...


class RTMPoseModel:
    """RTMPose（rtmlib）。ONNX のセッションを ``accel.make_session`` で作り直し、GPU があれば GPU で動かす。

    Args:
        mode: rtmlib のモデルの大きさ（``lightweight`` / ``balanced`` / ``performance``）
        prefer: 装置の選び方（``accel`` の auto / gpu / cpu）
    """

    name = "rtmpose"

    def __init__(self, mode: str = "balanced", prefer: str | None = None):
        try:
            from rtmlib import Body
        except ImportError as e:
            raise ImportError("rtmlib が入っていません（pip install -e .[pose]）") from e
        from smash_speed.accel import make_session

        self.body = Body(mode=mode, backend="onnxruntime", device="cpu")
        for part in (getattr(self.body, "det_model", None), self.body.pose_model):
            if part is not None:
                part.session = make_session(part.onnx_model, prefer)
        self.provider = self.body.pose_model.session.get_providers()[0]

    def __call__(self, image: np.ndarray) -> list[Person]:
        kpts, scores = self.body(image)
        kpts = np.asarray(kpts, float).reshape(-1, NUM_KEYPOINTS, 2)
        scores = np.asarray(scores, float).reshape(-1, NUM_KEYPOINTS)
        # RTMPose（SimCC）の信頼度は 0〜1 を超えることがあるので、0〜1 に収める
        return [(k, np.clip(s, 0.0, 1.0)) for k, s in zip(kpts, scores)]


def mediapipe_model_path(variant: str = "full", download: bool = True) -> Path:
    """MediaPipe Pose Landmarker の重みのファイル（なければ取ってくる）。"""
    path = MODEL_DIR / f"pose_landmarker_{variant}.task"
    if not path.exists():
        if not download:
            raise FileNotFoundError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MEDIAPIPE_URL.format(v=variant), tmp)
        tmp.replace(path)
    return path


class MediaPipeModel:
    """MediaPipe Pose Landmarker。33 点のうち COCO の 17 点に当たるものを返す（信頼度は visibility）。

    Args:
        variant: 重みの大きさ（``lite`` / ``full`` / ``heavy``）
        num_poses: 1 コマで探す人の数の上限
    """

    name = "mediapipe"

    def __init__(self, variant: str = "full", num_poses: int = 2, model_path: str | Path | None = None):
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except ImportError as e:
            raise ImportError("mediapipe が入っていません（pip install -e .[pose]）") from e
        self._mp = mp
        path = Path(model_path) if model_path else mediapipe_model_path(variant)
        opts = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(path)),
                                            running_mode=vision.RunningMode.IMAGE, num_poses=num_poses)
        self.landmarker = vision.PoseLandmarker.create_from_options(opts)

    def __call__(self, image: np.ndarray) -> list[Person]:
        import cv2

        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        res = self.landmarker.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
        out: list[Person] = []
        for lms in res.pose_landmarks:
            pts = [lms[i] for i in MEDIAPIPE_TO_COCO]
            xy = np.array([[p.x * w, p.y * h] for p in pts], float)
            score = np.array([min(p.visibility or 0.0, p.presence or 1.0) for p in pts], float)
            out.append((xy, np.clip(score, 0.0, 1.0)))
        return out

    def close(self) -> None:
        self.landmarker.close()


def make_model(name: str, prefer: str | None = None, **kwargs: Any) -> PoseModel:
    """名前（``MODELS``）からモデルを作る。"""
    if name == "rtmpose":
        return RTMPoseModel(prefer=prefer, **kwargs)
    if name == "mediapipe":
        return MediaPipeModel(**kwargs)
    raise ValueError(f"姿勢推定のモデルは {', '.join(MODELS)} のどれかにしてください: {name!r}")


# --- 推定 ------------------------------------------------------------------------

def _center_and_size(xy: np.ndarray, score: np.ndarray, min_score: float = MIN_SCORE
                     ) -> tuple[np.ndarray, float] | None:
    ok = (score >= min_score) & np.isfinite(xy).all(axis=1)
    if ok.sum() < 3:
        return None
    p = xy[ok]
    span = p.max(axis=0) - p.min(axis=0)
    return p.mean(axis=0), float(span[0] * span[1])


def pick_person(people: list[Person], prev_center: np.ndarray | None = None, max_jump_px: float | None = None
                ) -> int | None:
    """打つ人を選ぶ。前のコマの中心が分かれば、いちばん近い人（``max_jump_px`` より離れていれば選ばない）。
    分からなければ、関節の範囲（外接の四角の面積）がいちばん大きい人。誰もいなければ None。"""
    stats = [_center_and_size(xy, s) for xy, s in people]
    cand = [i for i, st in enumerate(stats) if st is not None]
    if not cand:
        return None
    if prev_center is None:
        return max(cand, key=lambda i: stats[i][1])
    dist = {i: float(np.linalg.norm(stats[i][0] - prev_center)) for i in cand}
    best = min(cand, key=lambda i: dist[i])
    if max_jump_px is not None and dist[best] > max_jump_px:
        return None
    return best


def estimate_frames(frames: Iterable[tuple[float, np.ndarray]], model: PoseModel | Callable[[np.ndarray], list[Person]],
                    width: int = 0, height: int = 0, fps: float = 0.0, max_jump_frac: float = 0.15,
                    max_lost: int = 10, progress: Callable[[int], None] | None = None) -> PoseSequence:
    """コマ（時刻 [s]、BGR の画像）の並びの姿勢を推定し、打つ人の列を返す。

    ``max_jump_frac``：前のコマの打つ人から、画像の幅のこの割合より離れた人は別の人として選ばない。
    ``max_lost`` コマ続けて見失ったら、前の位置を忘れて、いちばん大きい人から選び直す。
    """
    ts, xys, scores = [], [], []
    prev = None
    lost = 0
    for k, (t, img) in enumerate(frames):
        if not width:
            height, width = img.shape[:2]
        people = model(img)
        i = pick_person(people, prev, max_jump_frac * width if prev is not None else None)
        if i is None:
            xy, sc = np.full((NUM_KEYPOINTS, 2), np.nan), np.zeros(NUM_KEYPOINTS)
            lost += 1
            if lost > max_lost:
                prev = None
        else:
            lost = 0
            xy, sc = people[i]
            st = _center_and_size(xy, sc)
            prev = st[0] if st is not None else prev
        ts.append(t)
        xys.append(xy)
        scores.append(sc)
        if progress is not None:
            progress(k)
    n = len(ts)
    return PoseSequence(t=np.array(ts), xy=np.array(xys).reshape(n, NUM_KEYPOINTS, 2),
                        score=np.array(scores).reshape(n, NUM_KEYPOINTS), width=width, height=height, fps=fps,
                        model=getattr(model, "name", type(model).__name__))


def estimate_video(video: str | Path, model: PoseModel, start_s: float | None = None, end_s: float | None = None,
                   capture_fps: float | None = None, scale: float = 1.0) -> PoseSequence:
    """動画の区間 [start_s, end_s] [s] のすべてのコマの姿勢を推定する（座標は元の画像の大きさに戻す）。"""
    from smash_speed.video_io import VideoReader, probe

    meta = probe(video, capture_fps=capture_fps)
    with VideoReader(meta, scale=scale) as reader:
        seq = estimate_frames(((t, img) for _, t, img in reader.iter_range(start_s, end_s)), model,
                              width=meta.width, height=meta.height, fps=meta.capture_fps)
    if scale != 1.0:
        seq.xy = seq.xy / scale
    seq.width, seq.height = meta.width, meta.height
    seq.meta["video"] = Path(video).name
    return seq


def clean(seq: PoseSequence, min_score: float = MIN_SCORE, max_gap: int = MAX_GAP_FRAMES) -> PoseSequence:
    """信頼度が ``min_score`` より低い関節を欠け（NaN）にし、``max_gap`` コマ以下の欠けを前後から直線で埋める。

    長い欠け（隠れた非利き腕など）はそのまま残す。埋めた関節の信頼度は前後の小さい方にする。
    """
    out = seq.copy()
    bad = (out.score < min_score) | ~np.isfinite(out.xy).all(axis=2)
    out.xy[bad] = np.nan
    out.score[bad] = 0.0
    idx = np.arange(len(out))
    for j in range(NUM_KEYPOINTS):
        miss = bad[:, j]
        if not miss.any() or miss.all():
            continue
        good = np.flatnonzero(~miss)
        # 欠けの続く区間ごとに、長さと前後があるかを見る
        starts = np.flatnonzero(miss & ~np.r_[False, miss[:-1]])
        for s in starts:
            e = s
            while e + 1 < len(miss) and miss[e + 1]:
                e += 1
            if e - s + 1 > max_gap or s == 0 or e == len(miss) - 1:
                continue
            for c in range(2):
                out.xy[s:e + 1, j, c] = np.interp(idx[s:e + 1], good, out.xy[good, j, c])
            out.score[s:e + 1, j] = min(out.score[s - 1, j], out.score[e + 1, j])
    return out


# --- 精度（手で付けた関節との比べ方） --------------------------------------------------

# 手で付けた関節の位置の CSV：1 行 1 関節。visible は COCO と同じ（0 付けていない／1 隠れているが位置は推測／2 見える）
LABEL_COLUMNS = ["video", "frame", "t_s", "joint", "x", "y", "visible"]


@dataclass
class Labels:
    """手で付けた関節の位置。``xy`` (N, 17, 2) [px]、``visible`` (N, 17)、コマの時刻 ``t`` (N,) [s]。"""

    t: np.ndarray
    xy: np.ndarray
    visible: np.ndarray
    frames: np.ndarray
    video: str = ""


def write_labels(path: str | Path, labels: Labels) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(LABEL_COLUMNS)
        for k in range(len(labels.t)):
            for j, name in enumerate(KEYPOINTS):
                v = int(labels.visible[k, j])
                x, y = labels.xy[k, j]
                w.writerow([labels.video, int(labels.frames[k]), f"{labels.t[k]:.6f}", name,
                            "" if v == 0 else f"{x:.2f}", "" if v == 0 else f"{y:.2f}", v])
    return path


def load_labels(path: str | Path) -> Labels:
    """``LABEL_COLUMNS`` の CSV を読む。書いていない関節は visible = 0。"""
    rows: dict[int, dict[str, Any]] = {}
    video = ""
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            k = int(r["frame"])
            d = rows.setdefault(k, {"t": float(r["t_s"]) if r.get("t_s") else np.nan,
                                    "xy": np.full((NUM_KEYPOINTS, 2), np.nan), "v": np.zeros(NUM_KEYPOINTS, int)})
            video = r.get("video") or video
            if r["joint"] not in KP:
                raise ValueError(f"知らない関節の名前です: {r['joint']!r}（{', '.join(KEYPOINTS)}）")
            j = KP[r["joint"]]
            v = int(r.get("visible") or 2)
            if v > 0 and r["x"] != "" and r["y"] != "":
                d["xy"][j] = (float(r["x"]), float(r["y"]))
                d["v"][j] = v
    frames = np.array(sorted(rows), int)
    return Labels(t=np.array([rows[k]["t"] for k in frames]), xy=np.array([rows[k]["xy"] for k in frames]),
                  visible=np.array([rows[k]["v"] for k in frames]), frames=frames, video=video)


def torso_length(xy: np.ndarray) -> np.ndarray:
    """肩の中点 → 腰の中点の長さ (N,) [px]（PCK の基準の長さ）。"""
    sh = (xy[..., KP["left_shoulder"], :] + xy[..., KP["right_shoulder"], :]) / 2
    hp = (xy[..., KP["left_hip"], :] + xy[..., KP["right_hip"], :]) / 2
    return np.linalg.norm(sh - hp, axis=-1)


def match_frames(seq: PoseSequence, labels: Labels, tol_s: float | None = None) -> np.ndarray:
    """ラベルの各コマに当たる、推定の列のコマの番号（なければ −1）。時刻のいちばん近いもの。"""
    if len(seq) == 0:
        return np.full(len(labels.t), -1)
    tol = tol_s if tol_s is not None else (0.5 / seq.fps if seq.fps else 1e-3)
    out = []
    for t in labels.t:
        i = int(np.argmin(np.abs(seq.t - t)))
        out.append(i if abs(seq.t[i] - t) <= tol else -1)
    return np.array(out, int)


def pck(seq: PoseSequence, labels: Labels, alpha: float = 0.2, min_score: float = MIN_SCORE,
        include_hidden: bool = False) -> dict[str, Any]:
    """PCK（関節の位置の誤差が、胴体の長さ × ``alpha`` 以内の割合）。

    胴体の長さはラベルの肩・腰から（4 点そろわないコマは数えない）。推定で欠け・信頼度が ``min_score`` より
    低い関節は「外れ」に数える。``include_hidden`` が False なら、隠れている（visible = 1）関節は数えない。

    Returns:
        ``pck``（全体）、``per_joint``（関節ごと）、``mean_error_px``、``median_error_norm``（胴体の長さで割った誤差）、
        ``n``（数えた関節の数）、``n_frames``、``missing``（ラベルのコマが推定にない数）
    """
    idx = match_frames(seq, labels)
    norm = torso_length(labels.xy)
    use_vis = labels.visible >= (1 if include_hidden else 2)
    hits, errs, nerrs = [], [], []
    per = {name: [] for name in KEYPOINTS}
    n_frames = 0
    for k, i in enumerate(idx):
        if i < 0 or not np.isfinite(norm[k]) or norm[k] <= 0:
            continue
        n_frames += 1
        for j, name in enumerate(KEYPOINTS):
            if not use_vis[k, j]:
                continue
            p = seq.xy[i, j]
            ok_pred = np.isfinite(p).all() and seq.score[i, j] >= min_score
            e = float(np.linalg.norm(p - labels.xy[k, j])) if ok_pred else np.inf
            hit = e <= alpha * norm[k]
            hits.append(hit)
            per[name].append(hit)
            if ok_pred:
                errs.append(e)
                nerrs.append(e / norm[k])
    return {
        "pck": float(np.mean(hits)) if hits else None,
        "alpha": alpha,
        "per_joint": {k: (float(np.mean(v)) if v else None) for k, v in per.items()},
        "mean_error_px": float(np.mean(errs)) if errs else None,
        "median_error_norm": float(np.median(nerrs)) if nerrs else None,
        "n": len(hits),
        "n_frames": n_frames,
        "missing": int((idx < 0).sum()),
    }


def labels_from_sequence(seq: PoseSequence, frames: np.ndarray | None = None, video: str = "") -> Labels:
    """推定の列（合成の正解など）からラベルを作る（テスト・付け方の見本用）。"""
    frames = np.arange(len(seq)) if frames is None else np.asarray(frames, int)
    vis = np.where(np.isfinite(seq.xy[frames]).all(axis=2), 2, 0)
    return Labels(t=seq.t[frames], xy=seq.xy[frames], visible=vis, frames=frames, video=video)


# --- CLI ------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="姿勢推定（関節の位置）と、手で付けた関節との比べ方")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("estimate", help="動画の区間の姿勢を推定して .npz に書き出す")
    e.add_argument("video")
    e.add_argument("--model", choices=MODELS, default="rtmpose")
    e.add_argument("--start", type=float, default=None, help="始め [s]")
    e.add_argument("--end", type=float, default=None, help="終わり [s]")
    e.add_argument("--capture-fps", type=float, default=None)
    e.add_argument("--device", choices=("auto", "gpu", "cpu"), default=None)
    e.add_argument("--out", required=True)
    c = sub.add_parser("pck", help="手で付けた関節（CSV）と推定（.npz）を比べる")
    c.add_argument("labels")
    c.add_argument("poses", nargs="+")
    c.add_argument("--alpha", type=float, default=0.2)
    args = p.parse_args(argv)

    if args.cmd == "estimate":
        model = make_model(args.model, prefer=args.device)
        try:
            seq = estimate_video(args.video, model, args.start, args.end, capture_fps=args.capture_fps)
        finally:
            if hasattr(model, "close"):
                model.close()
        seq.save(args.out)
        found = np.isfinite(seq.xy).all(axis=2).any(axis=1).mean() if len(seq) else 0.0
        print(f"{len(seq)} コマを推定しました（人が見つかったコマ {found:.0%}）: {args.out}")
        return 0
    labels = load_labels(args.labels)
    print(f"ラベル: {len(labels.t)} コマ")
    for path in args.poses:
        seq = PoseSequence.load(path)
        r = pck(seq, labels, alpha=args.alpha)
        val = "—" if r["pck"] is None else f"{r['pck']:.3f}"
        err = "—" if r["mean_error_px"] is None else f"{r['mean_error_px']:.1f} px"
        print(f"{Path(path).name}（{seq.model}）: PCK@{args.alpha} {val}、平均の誤差 {err}、"
              f"数えた関節 {r['n']}、推定にないコマ {r['missing']}")
        worst = sorted(((v, k) for k, v in r["per_joint"].items() if v is not None))[:3]
        if worst:
            print("  低い関節: " + "、".join(f"{KEYPOINT_LABELS[k]} {v:.2f}" for v, k in worst))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
