"""動画読み込み：ffprobe によるメタデータ取得、VFR 対応のタイムスタンプ、撮影 fps の上書き、区間の切り出し。

時間はすべて **秒** で扱い、フレーム番号ではなくタイムスタンプを基準にする（可変フレームレート対策）。
スマホのスロー動画は、240fps で撮影しても再生用に 30fps として記録されることがある。
その場合は ``capture_fps`` を指定すると、タイムスタンプを実時間に換算する。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def find_tool(name: str, cfg: dict[str, Any] | None = None) -> str:
    """ffmpeg / ffprobe の実行ファイルのパスを返す（設定ファイル → PATH の順に探す）。"""
    if cfg is not None and cfg.get("video", {}).get(name):
        return str(cfg["video"][name])
    path = shutil.which(name)
    if path is None:
        # winget でインストールした直後は PATH が反映されていないことがあるので、既定の場所も探す
        import os
        base = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
        found = sorted(base.glob(f"Gyan.FFmpeg*/*/bin/{name}.exe"))
        if found:
            path = str(found[-1])
    if path is None:
        raise FileNotFoundError(
            f"{name} が見つかりません。ffmpeg をインストールして PATH に通すか、"
            f"設定ファイルの video.{name} にパスを書いてください。"
        )
    return path


def _run_json(cmd: list[str]) -> dict[str, Any]:
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=True)
    return json.loads(out.stdout)


@dataclass
class VideoMeta:
    """動画のメタデータ。

    Attributes:
        path: 動画ファイル
        width, height: 表示時の画像サイズ [px]（回転を反映）
        fps_meta: メタデータ上の平均フレームレート [Hz]
        capture_fps: 実際の撮影フレームレート [Hz]（上書き指定がなければ fps_meta）
        time_scale: メタデータの時刻 → 実時間の倍率（= fps_meta / capture_fps）
        n_frames: フレーム数
        duration_s: 動画の長さ（メタデータ上）[s]
        rotation_deg: 表示時の回転 [deg]
        is_vfr: 可変フレームレートと判定したか
        has_audio: 音声トラックの有無
        timestamps: 各フレームの実時間のタイムスタンプ [s]（先頭フレームが 0）
        tags: 撮影機種などのタグ（位置情報は除く）
    """

    path: str
    width: int
    height: int
    fps_meta: float
    capture_fps: float
    time_scale: float
    n_frames: int
    duration_s: float
    rotation_deg: int
    is_vfr: bool
    has_audio: bool
    timestamps: np.ndarray = field(repr=False)
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["timestamps"] = self.timestamps.tolist()
        return d

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")

    def frame_at(self, t: float) -> int:
        """実時間 t [s] に最も近いフレーム番号。"""
        return int(np.argmin(np.abs(self.timestamps - t)))


_PRIVATE_TAG_WORDS = ("location", "iso6709", "gps")


def probe(path: str | Path, capture_fps: float | None = None, cfg: dict[str, Any] | None = None
          ) -> VideoMeta:
    """ffprobe で動画のメタデータと各フレームのタイムスタンプを取得する。

    Args:
        path: 動画ファイル
        capture_fps: 実際の撮影 fps [Hz]（スロー動画で記録 fps と異なる場合に指定）
        cfg: 設定（ffprobe のパス）
    """
    ffprobe = find_tool("ffprobe", cfg)
    path = Path(path)
    info = _run_json([ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)])
    vstreams = [s for s in info["streams"] if s.get("codec_type") == "video"]
    if not vstreams:
        raise ValueError(f"映像トラックがありません: {path}")
    vs = vstreams[0]
    has_audio = any(s.get("codec_type") == "audio" for s in info["streams"])

    rotation = 0
    for sd in vs.get("side_data_list", []):
        if "rotation" in sd:
            rotation = int(round(float(sd["rotation"])))
    if "rotate" in vs.get("tags", {}):
        rotation = int(vs["tags"]["rotate"])
    w, h = int(vs["width"]), int(vs["height"])
    if abs(rotation) % 180 == 90:
        w, h = h, w

    # 各フレームの表示時刻（パケットの pts をデコードせずに取得し、表示順に並べる）。
    # iPhone などの動画は、表示しない（デコードの準備用の）パケットが先頭にあり、flags に "D"（discard）が付いて
    # pts が負になっている。これを含めて先頭の pts を 0 に合わせると、全フレームの時刻がずれて音声と合わなくなる
    # （例：IMG_4521 では 0.1 s 遅れていた）。表示されるパケットだけを使い、ストリームの開始時刻を 0 とする
    pk = _run_json([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                    "packet=pts_time,flags", "-of", "json", str(path)])
    packets = [p for p in pk.get("packets", []) if "pts_time" in p]
    shown = [p for p in packets if "D" not in p.get("flags", "")]
    pts = np.array(sorted(float(p["pts_time"]) for p in shown))
    if len(pts) and pts.min() < 0:  # flags が無い形式：負の時刻は表示されない
        pts = pts[pts >= 0]
    if len(pts) == 0:
        raise ValueError(f"タイムスタンプを取得できません: {path}")
    start = float(vs.get("start_time", 0.0) or 0.0)
    pts -= max(start, 0.0) if pts[0] >= max(start, 0.0) else pts[0]

    avg = vs.get("avg_frame_rate", "0/1")
    fps_meta = float(Fraction(avg)) if avg not in ("0/0", "0/1") else (len(pts) - 1) / pts[-1]
    dts = np.diff(pts)
    is_vfr = bool(len(dts) > 1 and np.std(dts) > 0.05 * np.median(dts))
    cap = float(capture_fps) if capture_fps else fps_meta
    scale = fps_meta / cap
    tags = {k: v for k, v in info.get("format", {}).get("tags", {}).items()
            if not any(word in k.lower() for word in _PRIVATE_TAG_WORDS)}
    return VideoMeta(
        path=str(path), width=w, height=h, fps_meta=fps_meta, capture_fps=cap, time_scale=scale,
        n_frames=len(pts), duration_s=float(info["format"].get("duration", pts[-1])),
        rotation_deg=rotation, is_vfr=is_vfr, has_audio=has_audio, timestamps=pts * scale, tags=tags,
    )


def seek_read(cap: cv2.VideoCapture, meta: "VideoMeta", index: int, tries: int = 3) -> np.ndarray | None:
    """フレーム番号 index（本当の表示順）の画像を OpenCV で読む。

    OpenCV はフレーム番号を「番号 / 公称 fps」の時刻に直して読む位置を決めるので、実際の fps が公称と少し違う
    （iPhone の 30 fps は 29.97 fps 前後）動画では、動画の後ろの方で数コマずれたフレームが返ることがある
    （IMG_4521 の 125 秒付近では 2 コマ）。読んだフレームの本当の時刻（CAP_PROP_POS_MSEC）から番号を確かめ、
    違えばその分だけ読む位置を直して読み直す。続けて ``cap.read()`` すれば次の番号のフレームになる。
    """
    target = index
    frame = None
    for _ in range(tries):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target))
        ok, frame = cap.read()
        if not ok:
            return None
        got = meta.frame_at(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 * meta.time_scale)
        if got == index:
            return frame
        target += index - got
    return frame


class VideoReader:
    """OpenCV でフレームを読む。時刻は ``VideoMeta.timestamps``（実時間）を使う。"""

    def __init__(self, meta: VideoMeta, scale: float = 1.0):
        """
        Args:
            meta: ``probe`` の結果
            scale: 読み込んだフレームの拡大縮小率（4K 動画を縮小して処理する場合など）
        """
        self.meta = meta
        self.scale = scale
        self.cap = cv2.VideoCapture(meta.path)
        if not self.cap.isOpened():
            raise OSError(f"動画を開けません: {meta.path}")

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        if self.scale == 1.0:
            return frame
        return cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)

    def read_frame(self, index: int) -> np.ndarray:
        """フレーム番号 ``index`` の画像（BGR）を返す。"""
        frame = seek_read(self.cap, self.meta, index)
        if frame is None:
            raise IndexError(f"フレーム {index} を読めません")
        return self._resize(frame)

    def read_at(self, t: float) -> tuple[int, np.ndarray]:
        """実時間 t [s] に最も近いフレームを返す。"""
        i = self.meta.frame_at(t)
        return i, self.read_frame(i)

    def iter_range(self, start_s: float | None = None, end_s: float | None = None
                   ) -> Iterator[tuple[int, float, np.ndarray]]:
        """区間 [start_s, end_s] [s] のフレームを (番号, 実時間 [s], 画像) で順に返す。"""
        ts = self.meta.timestamps
        i0 = 0 if start_s is None else int(np.searchsorted(ts, start_s))
        i1 = len(ts) - 1 if end_s is None else int(np.searchsorted(ts, end_s, side="right")) - 1
        frame = seek_read(self.cap, self.meta, i0)  # 最初だけ番号を確かめ、あとは順に読む
        for i in range(i0, i1 + 1):
            if i > i0:
                ok, frame = self.cap.read()
                if not ok:
                    break
            if frame is None:
                break
            yield i, float(ts[i]), self._resize(frame)

    def close(self) -> None:
        self.cap.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def extract_audio(video: str | Path, out_wav: str | Path, rate: int = 48000,
                  start_s: float | None = None, end_s: float | None = None,
                  cfg: dict[str, Any] | None = None) -> Path:
    """動画の最初の音声トラックをモノラル WAV（16 bit）として書き出す。

    ``start_s`` / ``end_s`` はメタデータ上の時刻 [s]。
    """
    ffmpeg = find_tool("ffmpeg", cfg)
    cmd = [ffmpeg, "-v", "error", "-y"]
    if start_s is not None:
        cmd += ["-ss", f"{start_s:.6f}"]
    cmd += ["-i", str(video)]
    if end_s is not None:
        cmd += ["-t", f"{end_s - (start_s or 0.0):.6f}"]
    cmd += ["-map", "0:a:0", "-ac", "1", "-ar", str(rate), "-c:a", "pcm_s16le", str(out_wav)]
    subprocess.run(cmd, check=True)
    return Path(out_wav)


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """WAV を読み込み、(波形 [-1, 1], サンプリング周波数 [Hz]) を返す。"""
    from scipy.io import wavfile
    rate, raw = wavfile.read(str(path))
    data = raw.astype(np.float64)
    if np.issubdtype(raw.dtype, np.integer):
        data /= float(np.iinfo(raw.dtype).max) + 1.0
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, int(rate)


def cut_clip(video: str | Path, out: str | Path, start_s: float, end_s: float,
             scale_width: int | None = None, cfg: dict[str, Any] | None = None) -> Path:
    """区間 [start_s, end_s]（メタデータ上の時刻）[s] を再エンコードして切り出す。

    フレーム単位で正確に切るため再エンコードする。``scale_width`` で横幅を縮小できる。
    """
    ffmpeg = find_tool("ffmpeg", cfg)
    cmd = [ffmpeg, "-v", "error", "-y", "-ss", f"{start_s:.6f}", "-i", str(video),
           "-t", f"{end_s - start_s:.6f}"]
    if scale_width:
        cmd += ["-vf", f"scale={scale_width}:-2"]
    cmd += ["-c:v", "libx264", "-crf", "16", "-preset", "fast", "-c:a", "aac", str(out)]
    subprocess.run(cmd, check=True)
    return Path(out)
