"""ショットの一覧の結果を動画に重ねて書き出す（シャトルの軌跡・当てはめた軌道・種類と初速）と、見どころの切り出し。

ショットの一覧（``shot_list``）の出力先にある ``shot_list.json``・``shots.csv``・``rallies.csv``・``detections.csv``
と元の動画から、次のものを書き出す。

- 重ねた動画（``render_overlay``）：シャトルの最近の軌跡（検出の点）、いまのショットの当てはめた軌道、
  打った所の近くに種類と初速（例「スマッシュ 312 km/h」）を約 1.5 秒。
- 見どころ（``write_highlights``）：速いスマッシュの上位 N 本（打つ 1 秒前〜2 秒後）や、長いラリーの上位 N 本を
  1 本ずつの短い動画にし、つないだ 1 本の動画も作る。``highlights.json`` に一覧を書く。

フレームは 1 枚ずつ読んで書くので、長い動画でも全部をメモリに載せない。ffmpeg があれば H.264 で書き、
元の動画の音も入れる（スロー動画は音の長さが合わないので入れない）。なければ OpenCV（mp4v・音なし）で書く。
4K の動画は既定で 1080p に縮小する。

``python -m smash_speed.overlay <ショットの一覧の出力先> [--out 動画.mp4] [--start 秒 --end 秒]`` で重ねた動画、
``--highlights`` で見どころを書き出す（``smash-speed overlay`` も同じ）。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd

from smash_speed.calibration.io import load_calibration
from smash_speed.errors import AnalysisError
from smash_speed.physics import PhysicsParams
from smash_speed.shots import RALLIES_FILE, SHOT_TYPE_LABELS, SHOTS_FILE, Rally, Shot, read_rallies, read_shots
from smash_speed.video_io import VideoMeta, VideoReader, find_tool, probe

OVERLAY_FILE = "overlay.mp4"
HIGHLIGHTS_DIR = "highlights"
HIGHLIGHTS_FILE = "highlights.json"
DEFAULT_HEIGHT = 1080            # 4K などの大きい動画は、この高さに縮小して書く（小さい動画はそのまま）

# ショットの種類ごとの色（BGR）
TYPE_COLORS = {"smash": (40, 40, 230), "clear": (230, 150, 40), "drop": (60, 200, 60), "drive": (0, 170, 255),
               "lob": (200, 90, 200), "net": (200, 200, 40), "push": (40, 200, 200), "serve": (180, 180, 180),
               "unknown": (160, 160, 160)}
TRAIL_COLOR = (0, 230, 255)      # シャトルの軌跡（黄色）
FONT_FILES = ("meiryob.ttc", "meiryo.ttc", "YuGothB.ttc", "YuGothM.ttc", "msgothic.ttc",
              "NotoSansCJK-Bold.ttc", "NotoSansCJK-Regular.ttc")
FONT_DIRS = (Path("C:/Windows/Fonts"), Path("/usr/share/fonts/opentype/noto"), Path("/System/Library/Fonts"))


@dataclass
class DrawOptions:
    """何を描くか。"""

    trail: bool = True           # シャトルの最近の軌跡（検出の点）
    trail_s: float = 0.4         # 軌跡を残す長さ [s]
    trajectory: bool = True      # いまのショットの当てはめた軌道
    labels: bool = True          # 種類と初速
    label_s: float = 1.5         # 種類と初速を出しておく長さ [s]


@dataclass
class OverlayOptions:
    shots_dir: Path                          # ショットの一覧の出力先
    out: Path | None = None                  # 書き出す動画（省略時は <出力先>/overlay.mp4）
    video: Path | None = None                # 元の動画（省略時は shot_list.json の video）
    start_s: float | None = None             # 書き出す区間（動画の実時間 [s]）
    end_s: float | None = None
    height: int | None = None                # 書き出す高さ [px]（省略時は DEFAULT_HEIGHT と元の小さい方）
    draw: DrawOptions = field(default_factory=DrawOptions)
    audio: bool = True                       # 元の音を入れる（ffmpeg があるとき）
    crf: int = 20                            # H.264 の画質（小さいほどきれい・大きい）


# =============================================================================
# 結果を読む
# =============================================================================
@dataclass
class _ShotDraw:
    """1 本のショットを描くのに要るもの（書き出す画像の座標）。"""

    shot: Shot
    t_end: float                 # 軌道を描く終わり（着地か次のショット）
    t: np.ndarray                # 当てはめた軌道の時刻
    uv: np.ndarray               # 同・画像の座標 (N, 2)。軌道がなければ空
    anchor: tuple[int, int] | None   # 種類と初速を出す場所（打点）


class OverlayData:
    """ショットの一覧の結果と動画を、書き出す大きさに合わせて読む。"""

    def __init__(self, shots_dir: Path, video: Path | None = None, height: int | None = None,
                 cfg: dict[str, Any] | None = None):
        from smash_speed.pipeline import _camera_for_video
        from smash_speed.shot_list import DETECTIONS_FILE, SHOT_LIST_FILE

        d = Path(shots_dir)
        if not (d / SHOT_LIST_FILE).exists() or not (d / SHOTS_FILE).exists():
            raise AnalysisError(f"{d} にショットの一覧がありません。", ["先に「ショットの一覧」を作ってください。"])
        self.dir = d
        self.info = json.loads((d / SHOT_LIST_FILE).read_text(encoding="utf-8"))
        video = Path(video or self.info.get("video") or "")
        if not video.is_file():
            raise AnalysisError(f"元の動画 {video} が見つかりません。", ["動画を動かしていないか確かめてください。"])
        self.meta: VideoMeta = probe(video, self.info.get("capture_fps"), cfg)
        h = int(height or min(DEFAULT_HEIGHT, self.meta.height))
        self.scale = min(1.0, h / self.meta.height)
        self.size = (_even(self.meta.width * self.scale), _even(self.meta.height * self.scale))
        camera = None
        calib = self.info.get("calib")
        if calib and Path(calib).is_file():
            camera = _camera_for_video(load_calibration(calib)[0], self.meta).scaled(self.size[0] / self.meta.width)
        self.camera = camera
        self.shots = sorted(read_shots(d / SHOTS_FILE), key=lambda s: s.t_hit_s)
        self.rallies = read_rallies(d / RALLIES_FILE) if (d / RALLIES_FILE).exists() else []
        det = pd.read_csv(d / DETECTIONS_FILE) if (d / DETECTIONS_FILE).exists() else pd.DataFrame()
        if len(det):
            det = det[det.visible > 0].sort_values("timestamp")
            self.det_t = det.timestamp.to_numpy(float)
            self.det_uv = det[["x_px", "y_px"]].to_numpy(float) * (self.size[0] / self.meta.width)
            self.det_frame = det.frame.to_numpy(int)
        else:
            self.det_t, self.det_uv, self.det_frame = np.zeros(0), np.zeros((0, 2)), np.zeros(0, int)
        phys = PhysicsParams.from_config(cfg) if cfg else PhysicsParams()
        self.draws = [self._shot_draw(i, phys) for i in range(len(self.shots))]
        self.dt = 1.0 / self.meta.capture_fps

    def _shot_draw(self, i: int, phys: PhysicsParams) -> _ShotDraw:
        s = self.shots[i]
        nxt = self.shots[i + 1].t_hit_s if i + 1 < len(self.shots) else np.inf
        t_end = min(nxt, s.t_land_s if s.t_land_s is not None else s.t_hit_s + 2.0)
        t, uv = np.zeros(0), np.zeros((0, 2))
        anchor = None
        if self.camera is not None:
            try:
                tr = s.trajectory(phys)
            except Exception:  # noqa: BLE001  描けない軌道は描かないだけ
                tr = None
            if tr is not None:
                keep = tr["t"] <= t_end + 1e-9
                t, uv = tr["t"][keep], self.camera.project(tr["pos"][keep])
                ok = np.isfinite(uv).all(axis=1)
                t, uv = t[ok], uv[ok]
            p = s.hit_point
            if p is not None:
                a = self.camera.project(p)[0]
                if np.isfinite(a).all():
                    anchor = (int(a[0]), int(a[1]))
        if anchor is None and len(self.det_t):
            k = int(np.argmin(np.abs(self.det_t - s.t_hit_s)))
            if abs(self.det_t[k] - s.t_hit_s) < 0.2:
                anchor = (int(self.det_uv[k, 0]), int(self.det_uv[k, 1]))
        return _ShotDraw(s, float(t_end), t, uv, anchor)


def _even(x: float) -> int:
    return max(2, int(round(x / 2.0)) * 2)


# =============================================================================
# 描く
# =============================================================================
def shot_label(s: Shot) -> str:
    """「スマッシュ 312 km/h」のような文字（初速がなければ種類だけ）。"""
    name = SHOT_TYPE_LABELS.get(s.shot_type, s.shot_type)
    return name if s.speed_kmh is None else f"{name} {s.speed_kmh:.0f} km/h"


def _find_font() -> Path | None:
    for d in FONT_DIRS:
        for name in FONT_FILES:
            if (d / name).is_file():
                return d / name
    return None


class TextStamp:
    """日本語の文字を画像にして（Pillow）、フレームに貼る。同じ文字は作り直さない。"""

    def __init__(self, size_px: int):
        self.size = max(12, int(size_px))
        self._cache: dict[tuple[str, tuple[int, int, int]], np.ndarray] = {}
        self._font = None
        path = _find_font()
        try:
            from PIL import ImageFont
            self._font = ImageFont.truetype(str(path), self.size) if path else None
        except (ImportError, OSError):
            self._font = None

    def image(self, text: str, color: tuple[int, int, int]) -> np.ndarray:
        """文字の画像（BGRA。背景は半透明の黒）。"""
        key = (text, color)
        if key in self._cache:
            return self._cache[key]
        pad = self.size // 4
        if self._font is not None:
            from PIL import Image, ImageDraw
            x0, y0, x1, y1 = self._font.getbbox(text)
            w, h = x1 - x0 + 2 * pad, y1 - y0 + 2 * pad
            im = Image.new("RGBA", (w, h), (0, 0, 0, 150))
            ImageDraw.Draw(im).text((pad - x0, pad - y0), text, font=self._font, fill=(*color[::-1], 255),
                                    stroke_width=max(1, self.size // 16), stroke_fill=(0, 0, 0, 255))
            img = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGBA2BGRA)
        else:   # 日本語の字体がなければ、種類は英語で OpenCV の字体で書く
            text = text.encode("ascii", "replace").decode()
            scale = self.size / 30.0
            (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
            img = np.zeros((h + base + 2 * pad, w + 2 * pad, 4), np.uint8)
            img[..., 3] = 150
            cv2.putText(img, text, (pad, pad + h), cv2.FONT_HERSHEY_SIMPLEX, scale, (*color, 255), 2, cv2.LINE_AA)
        self._cache[key] = img
        return img

    def paste(self, frame: np.ndarray, text: str, xy: tuple[int, int], color: tuple[int, int, int],
              anchor: str = "bottom") -> None:
        """``xy`` の上（``anchor="bottom"``）か、左上（``"topleft"``）に文字を貼る。はみ出す分は内側に寄せる。"""
        img = self.image(text, color)
        h, w = img.shape[:2]
        H, W = frame.shape[:2]
        if anchor == "bottom":
            x, y = xy[0] - w // 2, xy[1] - h - self.size // 2
        else:
            x, y = xy
        x, y = int(np.clip(x, 0, max(0, W - w))), int(np.clip(y, 0, max(0, H - h)))
        w, h = min(w, W - x), min(h, H - y)
        if w <= 0 or h <= 0:
            return
        a = img[:h, :w, 3:4].astype(np.float32) / 255.0
        roi = frame[y:y + h, x:x + w]
        roi[:] = (img[:h, :w, :3] * a + roi * (1.0 - a)).astype(np.uint8)


class Painter:
    """1 枚のフレームに、その時刻の軌跡・軌道・文字を描く。"""

    def __init__(self, data: OverlayData, draw: DrawOptions):
        self.d = data
        self.o = draw
        h = data.size[1]
        self.thick = max(1, int(round(h / 360)))
        self.text = TextStamp(int(h / 22))
        self.caption = TextStamp(int(h / 28))

    def paint(self, frame: np.ndarray, t: float, caption: str | None = None) -> np.ndarray:
        d, o = self.d, self.o
        if o.trajectory:
            for sd in d.draws:
                if sd.shot.t_hit_s - 1e-6 <= t <= sd.t_end + 0.3 and len(sd.uv) >= 2:
                    pts = sd.uv.astype(np.int32)
                    color = TYPE_COLORS.get(sd.shot.shot_type, TYPE_COLORS["unknown"])
                    cv2.polylines(frame, [pts], False, color, self.thick, cv2.LINE_AA)
                    # 打ってから今までの部分は太く
                    k = int(np.searchsorted(sd.t, t, side="right"))
                    if k >= 2:
                        cv2.polylines(frame, [pts[:k]], False, color, 2 * self.thick + 1, cv2.LINE_AA)
        if o.trail and len(d.det_t):
            i0 = int(np.searchsorted(d.det_t, t - o.trail_s))
            i1 = int(np.searchsorted(d.det_t, t + 1e-6, side="right"))
            n = i1 - i0
            for k in range(i0, i1):
                p = (int(d.det_uv[k, 0]), int(d.det_uv[k, 1]))
                age = (k - i0 + 1) / max(n, 1)          # 新しい点ほど大きく
                if k > i0 and d.det_frame[k] - d.det_frame[k - 1] <= 3:
                    q = (int(d.det_uv[k - 1, 0]), int(d.det_uv[k - 1, 1]))
                    cv2.line(frame, q, p, TRAIL_COLOR, max(1, int(self.thick * (0.5 + age))), cv2.LINE_AA)
                cv2.circle(frame, p, max(2, int(self.thick * (1 + 2 * age))), TRAIL_COLOR, -1, cv2.LINE_AA)
        if o.labels:
            for sd in d.draws:
                if sd.anchor is not None and sd.shot.t_hit_s - 1e-6 <= t < sd.shot.t_hit_s + o.label_s:
                    color = TYPE_COLORS.get(sd.shot.shot_type, TYPE_COLORS["unknown"])
                    self.text.paste(frame, shot_label(sd.shot), sd.anchor, _bright(color))
        if caption:
            m = self.caption.size // 2
            self.caption.paste(frame, caption, (m, m), (255, 255, 255), anchor="topleft")
        return frame


def _bright(c: tuple[int, int, int]) -> tuple[int, int, int]:
    """文字は線より明るい色にする。"""
    return tuple(int(min(255, v + (255 - v) * 0.4)) for v in c)  # type: ignore[return-value]


# =============================================================================
# 書き出す
# =============================================================================
class FrameWriter:
    """フレームを 1 枚ずつ動画に書く。ffmpeg があれば H.264（と元の音）、なければ OpenCV。"""

    def __init__(self, out: Path, size: tuple[int, int], fps: float, cfg: dict[str, Any] | None = None,
                 audio_from: tuple[Path, float, float] | None = None, crf: int = 20, use_ffmpeg: bool = True):
        self.out = Path(out)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.size = size
        self.proc = None
        self.cv = None
        self.n = 0
        ffmpeg = None
        if use_ffmpeg:
            try:
                ffmpeg = find_tool("ffmpeg", cfg)
            except FileNotFoundError:
                ffmpeg = None
        self.encoder = "ffmpeg" if ffmpeg else "opencv"
        if ffmpeg:
            cmd = [ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                   "-s", f"{size[0]}x{size[1]}", "-r", f"{fps:.6f}", "-i", "-"]
            if audio_from is not None:
                src, a0, dur = audio_from
                cmd += ["-ss", f"{max(0.0, a0):.6f}", "-t", f"{dur:.6f}", "-i", str(src),
                        "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac", "-b:a", "160k"]
            cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(self.out)]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            self.cv = cv2.VideoWriter(str(self.out), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            if not self.cv.isOpened():
                raise OSError(f"動画を書けません: {self.out}")

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self.size[0] or frame.shape[0] != self.size[1]:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        if self.proc is not None:
            try:
                self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            except (BrokenPipeError, OSError) as e:
                raise OSError(f"ffmpeg が止まりました: {self._stderr()}") from e
        else:
            self.cv.write(frame)
        self.n += 1

    def _stderr(self) -> str:
        try:
            return (self.proc.stderr.read() or b"").decode(errors="replace").strip()[-500:]
        except (OSError, ValueError):
            return ""

    def close(self) -> Path:
        if self.proc is not None:
            self.proc.stdin.close()
            code = self.proc.wait()
            if code != 0:
                raise OSError(f"ffmpeg が失敗しました（{code}）: {self._stderr()}")
            self.proc.stderr.close()
        elif self.cv is not None:
            self.cv.release()
        return self.out


def render_range(data: OverlayData, painter: Painter, writer: FrameWriter, start_s: float | None,
                 end_s: float | None, caption: str | None = None,
                 progress: Callable[[int, int], None] | None = None) -> int:
    """区間 [start_s, end_s] のフレームを読み、描いて書く。書いたフレームの数を返す。"""
    ts = data.meta.timestamps
    i0 = 0 if start_s is None else int(np.searchsorted(ts, start_s))
    i1 = len(ts) - 1 if end_s is None else int(np.searchsorted(ts, end_s, side="right")) - 1
    total = max(0, i1 - i0 + 1)
    n = 0
    with VideoReader(data.meta, scale=data.scale) as rd:
        for _, t, frame in rd.iter_range(start_s, end_s):
            writer.write(painter.paint(frame, t, caption))
            n += 1
            if progress is not None and (n % 30 == 0 or n == total):
                progress(n, total)
    return n


def _audio_window(data: OverlayData, start_s: float | None, end_s: float | None, want: bool
                  ) -> tuple[Path, float, float] | None:
    """元の動画から音を取る区間（メタデータの時刻）。スロー動画・音なしなら None。"""
    m = data.meta
    if not want or not m.has_audio or abs(m.time_scale - 1.0) > 0.01:
        return None
    ts = m.timestamps
    i0 = 0 if start_s is None else int(np.searchsorted(ts, start_s))
    i1 = len(ts) - 1 if end_s is None else int(np.searchsorted(ts, end_s, side="right")) - 1
    a0 = float(ts[i0]) / m.time_scale
    return Path(m.path), a0, max(0.0, (i1 - i0 + 1) / m.fps_meta)


def _progress_printer(progress: Callable[[str], None], what: str) -> Callable[[int, int], None]:
    last = [-1]

    def f(n: int, total: int) -> None:
        pct = int(100 * n / max(total, 1))
        if pct // 5 != last[0] // 5 or n == total:
            last[0] = pct
            progress(f"[進行] {what} {pct}%（{n} / {total} コマ）")
    return f


def render_overlay(o: OverlayOptions, cfg: dict[str, Any] | None = None, progress: Callable[[str], None] = print,
                   use_ffmpeg: bool = True, data: OverlayData | None = None) -> Path:
    """ショットの一覧の結果を重ねた動画を書き出し、そのパスを返す。"""
    data = data or OverlayData(o.shots_dir, o.video, o.height, cfg)
    out = Path(o.out) if o.out else Path(o.shots_dir) / OVERLAY_FILE
    w = FrameWriter(out, data.size, data.meta.fps_meta, cfg, _audio_window(data, o.start_s, o.end_s, o.audio),
                    o.crf, use_ffmpeg)
    try:
        n = render_range(data, Painter(data, o.draw), w, o.start_s, o.end_s,
                         progress=_progress_printer(progress, "重ねた動画を書き出し中"))
    finally:
        w.close()
    progress(f"[進行] 完了：{out}（{n} コマ、{data.size[0]}x{data.size[1]}、{w.encoder}）")
    return out


# =============================================================================
# 見どころ
# =============================================================================
@dataclass
class Highlight:
    rank: int
    kind: str                    # fastest | rally
    start_s: float
    end_s: float
    title: str                   # 動画の左上に出す文字
    t_hit_s: float | None = None
    speed_kmh: float | None = None
    shot_type: str | None = None
    rally_id: int | None = None
    n_shots: int | None = None
    file: str = ""


def pick_fastest(shots: list[Shot], n: int = 5, shot_type: str | None = "smash", before_s: float = 1.0,
                 after_s: float = 2.0) -> list[Highlight]:
    """初速の速い順に N 本（``shot_type`` を決めればその種類だけ）。"""
    cand = [s for s in shots if s.speed_kmh is not None and (shot_type is None or s.shot_type == shot_type)]
    cand.sort(key=lambda s: -float(s.speed_kmh))
    out = []
    for r, s in enumerate(cand[:n], 1):
        name = SHOT_TYPE_LABELS.get(s.shot_type, s.shot_type)
        out.append(Highlight(r, "fastest", max(0.0, s.t_hit_s - before_s), s.t_hit_s + after_s,
                             f"速い{name} {r} 位：{s.speed_kmh:.0f} km/h", t_hit_s=s.t_hit_s,
                             speed_kmh=s.speed_kmh, shot_type=s.shot_type, rally_id=s.rally_id))
    return out


def pick_longest_rallies(rallies: list[Rally], shots: list[Shot], n: int = 3, before_s: float = 1.0,
                         after_s: float = 1.5, max_s: float = 60.0) -> list[Highlight]:
    """ショットの数の多い順に N ラリー（同じ数なら長い方）。長すぎるラリーは最初の ``max_s`` 秒だけ。"""
    rows = []
    for ra in rallies:
        mine = [s for s in shots if s.rally_id == ra.rally_id]
        n_shots = ra.n_shots or len(mine)
        if n_shots <= 0:
            continue
        t_last = max([s.t_hit_s for s in mine], default=ra.t_start_s)
        t_end = max(ra.t_end_s or t_last, t_last)
        rows.append((n_shots, t_end - ra.t_start_s, ra, t_end))
    rows.sort(key=lambda r: (-r[0], -r[1]))
    out = []
    for r, (n_shots, _, ra, t_end) in enumerate(rows[:n], 1):
        s0 = max(0.0, ra.t_start_s - before_s)
        out.append(Highlight(r, "rally", s0, min(t_end + after_s, s0 + max_s),
                             f"長いラリー {r} 位：{n_shots} 本", rally_id=ra.rally_id, n_shots=n_shots))
    return out


@dataclass
class HighlightOptions:
    shots_dir: Path
    out_dir: Path | None = None              # 省略時は <出力先>/highlights
    video: Path | None = None
    n_fastest: int = 5
    shot_type: str | None = "smash"          # 速い順に選ぶ種類（None ならすべて）
    n_rallies: int = 0
    before_s: float = 1.0
    after_s: float = 2.0
    height: int | None = None
    draw: DrawOptions = field(default_factory=DrawOptions)
    audio: bool = True
    crf: int = 20
    join: bool = True                        # つないだ 1 本（highlights.mp4）も作る


def write_highlights(o: HighlightOptions, cfg: dict[str, Any] | None = None, progress: Callable[[str], None] = print,
                     use_ffmpeg: bool = True) -> dict[str, Any]:
    """見どころを 1 本ずつの動画とつないだ動画に書き、``highlights.json`` の中身を返す。"""
    data = OverlayData(o.shots_dir, o.video, o.height, cfg)
    out = Path(o.out_dir) if o.out_dir else Path(o.shots_dir) / HIGHLIGHTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    hl = pick_fastest(data.shots, o.n_fastest, o.shot_type, o.before_s, o.after_s) if o.n_fastest > 0 else []
    if o.n_fastest > 0 and not hl and o.shot_type is not None:
        progress(f"[進行] {SHOT_TYPE_LABELS.get(o.shot_type, o.shot_type)}がないので、すべてのショットから速い順に選びます")
        hl = pick_fastest(data.shots, o.n_fastest, None, o.before_s, o.after_s)
    if o.n_rallies > 0:
        hl += pick_longest_rallies(data.rallies, data.shots, o.n_rallies, o.before_s)
    if not hl:
        raise AnalysisError("見どころにできるショットがありません。", ["初速の出たショットがある一覧を選んでください。"])
    end = float(data.meta.timestamps[-1])
    painter = Painter(data, o.draw)
    files: list[Path] = []
    for k, h in enumerate(hl, 1):
        h.end_s = min(h.end_s, end)
        name = f"{k:02d}_{h.kind}_{h.rank}.mp4"
        w = FrameWriter(out / name, data.size, data.meta.fps_meta, cfg,
                        _audio_window(data, h.start_s, h.end_s, o.audio), o.crf, use_ffmpeg)
        try:
            render_range(data, painter, w, h.start_s, h.end_s, h.title,
                         progress=_progress_printer(progress, f"見どころ {k} / {len(hl)} を書き出し中"))
        finally:
            w.close()
        h.file = name
        files.append(out / name)
    joined = None
    if o.join:
        joined = join_videos(files, out / "highlights.mp4", cfg, use_ffmpeg)
    info = {"shots_dir": str(Path(o.shots_dir).resolve()), "video": data.meta.path,
            "joined": joined.name if joined else None, "size": list(data.size),
            "clips": [h.__dict__ for h in hl]}
    (out / HIGHLIGHTS_FILE).write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    progress(f"[進行] 完了：見どころ {len(hl)} 本（{out}）")
    return info


def join_videos(files: list[Path], out: Path, cfg: dict[str, Any] | None = None, use_ffmpeg: bool = True) -> Path:
    """同じ大きさ・同じ書き方の動画をつなぐ。ffmpeg があれば書き直さずにつなぎ、なければ OpenCV で書き直す。"""
    ffmpeg = None
    if use_ffmpeg:
        try:
            ffmpeg = find_tool("ffmpeg", cfg)
        except FileNotFoundError:
            ffmpeg = None
    if len(files) == 1:
        shutil.copyfile(files[0], out)
        return out
    if ffmpeg:
        lst = out.with_suffix(".txt")
        lst.write_text("".join(f"file '{Path(f).resolve().as_posix()}'\n" for f in files), encoding="utf-8")
        try:
            subprocess.run([ffmpeg, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                            "-c", "copy", "-movflags", "+faststart", str(out)], check=True)
        finally:
            lst.unlink(missing_ok=True)
        return out
    cap = cv2.VideoCapture(str(files[0]))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    w = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for f in files:
        cap = cv2.VideoCapture(str(f))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            w.write(frame)
        cap.release()
    w.release()
    return out


# =============================================================================
# コマンド
# =============================================================================
def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("shots_dir", help="ショットの一覧の出力先（smash-speed shots の --out）")
    p.add_argument("--video", help="元の動画（省略時は shot_list.json に書いた動画）")
    p.add_argument("--out", help="書き出す動画（省略時は <出力先>/overlay.mp4。--highlights ではフォルダ）")
    p.add_argument("--start", type=float, help="書き出す区間の始め [s]")
    p.add_argument("--end", type=float, help="書き出す区間の終わり [s]")
    p.add_argument("--height", type=int, help=f"書き出す高さ [px]（省略時は {DEFAULT_HEIGHT} と元の小さい方）")
    p.add_argument("--no-trail", action="store_true", help="シャトルの軌跡を描かない")
    p.add_argument("--no-trajectory", action="store_true", help="当てはめた軌道を描かない")
    p.add_argument("--no-labels", action="store_true", help="種類と初速を出さない")
    p.add_argument("--no-audio", action="store_true", help="元の音を入れない")
    p.add_argument("--opencv", action="store_true", help="ffmpeg を使わず OpenCV で書く（音なし）")
    p.add_argument("--highlights", action="store_true", help="見どころ（速いスマッシュ・長いラリー）を書き出す")
    p.add_argument("--top", type=int, default=5, help="--highlights：速い順に何本か")
    p.add_argument("--type", default="smash", help="--highlights：速い順に選ぶ種類（all ならすべて）")
    p.add_argument("--rallies", type=int, default=0, help="--highlights：長いラリーを何本足すか")
    p.add_argument("--before", type=float, default=1.0, help="--highlights：打つ何秒前から")
    p.add_argument("--after", type=float, default=2.0, help="--highlights：打った何秒後まで")


def run_from_args(a: argparse.Namespace, cfg: dict[str, Any] | None = None) -> int:
    draw = DrawOptions(trail=not a.no_trail, trajectory=not a.no_trajectory, labels=not a.no_labels)
    if a.highlights:
        write_highlights(HighlightOptions(Path(a.shots_dir), Path(a.out) if a.out else None,
                                          Path(a.video) if a.video else None, a.top,
                                          None if a.type == "all" else a.type, a.rallies, a.before, a.after,
                                          a.height, draw, not a.no_audio), cfg, use_ffmpeg=not a.opencv)
    else:
        render_overlay(OverlayOptions(Path(a.shots_dir), Path(a.out) if a.out else None,
                                      Path(a.video) if a.video else None, a.start, a.end, a.height, draw,
                                      not a.no_audio), cfg, use_ffmpeg=not a.opencv)
    return 0


def main(argv: list[str] | None = None) -> None:
    from smash_speed.config import load_config

    ap = argparse.ArgumentParser(prog="python -m smash_speed.overlay", description=__doc__.splitlines()[0])
    add_arguments(ap)
    ap.add_argument("--config", help="既定設定を上書きする YAML ファイル")
    a = ap.parse_args(argv)
    try:
        raise SystemExit(run_from_args(a, load_config(a.config)))
    except AnalysisError as e:
        for line in e.report_lines():
            print(line, flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
