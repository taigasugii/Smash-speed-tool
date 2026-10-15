"""動画のプレビュー：スライダーでのコマ送り、ROI の描画、画面クリックでの点の指定。"""

from __future__ import annotations

import threading
import tkinter as tk
from collections.abc import Callable
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from smash_speed.video_io import VideoMeta, probe


class VideoViewer(ttk.Frame):
    """動画を1フレームずつ表示する。座標は常に元の動画の px で扱う。

    ``mode``:
        "none"   クリックしても何もしない
        "roi"    ドラッグで検出範囲（ROI）を描く
        "point"  クリックした位置を ``on_point(u, v)`` に渡す（打者の足元・着地点など）
    """

    def __init__(self, master, width: int = 960, height: int = 540, on_status: Callable[[str], None] | None = None):
        super().__init__(master)
        self.cw, self.ch = width, height
        self.meta: VideoMeta | None = None
        self.cap: cv2.VideoCapture | None = None
        self.index = 0
        self._pos = -1
        self._photo = None
        self._frame: np.ndarray | None = None
        self._scale = 1.0
        self._offset = (0, 0)
        self._pending = None
        self._playing = False
        self.roi: list[float] | None = None
        self.markers: dict[str, tuple[float, float, str]] = {}  # 名前 → (u, v, 色)
        self.mode = "none"
        self.on_point: Callable[[float, float], None] | None = None
        self.on_roi: Callable[[list[float] | None], None] | None = None
        self.on_frame: Callable[[int, float], None] | None = None
        self.on_status = on_status or (lambda s: None)
        self._drag_start = None

        self.canvas = tk.Canvas(self, width=width, height=height, bg="#202020", highlightthickness=0)
        self.canvas.grid(row=0, column=0, columnspan=8, sticky="nsew")
        self.canvas.create_text(width // 2, height // 2, text="「動画を開く」から動画を選んでください",
                                fill="#cccccc", font=("Yu Gothic UI", 14), tags="msg")
        self.var_t = tk.DoubleVar(value=0.0)
        self.scale = ttk.Scale(self, from_=0, to=1, orient="horizontal", variable=self.var_t,
                               command=lambda _v: self._on_slider())
        self.scale.grid(row=1, column=0, columnspan=8, sticky="ew", pady=(4, 2))
        self.lbl = ttk.Label(self, text="--", width=34)
        self.lbl.grid(row=2, column=0, sticky="w")
        for col, (txt, cmd) in enumerate([
            ("◀◀ 1秒", lambda: self.step_time(-1.0)), ("◀ 1コマ", lambda: self.step(-1)),
            ("▶ 1コマ", lambda: self.step(1)), ("1秒 ▶▶", lambda: self.step_time(1.0)),
        ], start=1):
            ttk.Button(self, text=txt, width=8, command=cmd).grid(row=2, column=col, padx=1)
        self.btn_play = ttk.Button(self, text="▶ 再生", width=8, command=self.toggle_play)
        self.btn_play.grid(row=2, column=5, padx=1)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._motion)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.canvas.bind("<Configure>", self._resize)

    # ------------------------------------------------------------ 読み込み
    def load(self, path: str, capture_fps: float | None, done: Callable[[VideoMeta | None, str], None]) -> None:
        """別スレッドで ffprobe を実行して動画を開く（4K の長い動画は数秒かかる）。"""
        self.on_status("動画を読み込んでいます…")

        box: dict = {}

        def work():  # tkinter は別スレッドから触れないので、結果だけ置いて GUI 側で拾う
            try:
                box["meta"], box["err"] = probe(path, capture_fps), ""
            except Exception as e:  # noqa: BLE001  GUI に表示する
                box["meta"], box["err"] = None, str(e)

        def check():
            if "err" in box:
                self._loaded(box["meta"], box["err"], done)
            else:
                self.after(100, check)

        threading.Thread(target=work, daemon=True).start()
        self.after(100, check)

    def _loaded(self, meta, err, done) -> None:
        if meta is None:
            self.on_status(f"読み込みに失敗しました: {err}")
            done(None, err)
            return
        if self.cap is not None:
            self.cap.release()
        self.meta = meta
        self.cap = cv2.VideoCapture(meta.path)
        self._pos = -1
        self.roi = None
        self.markers.clear()
        self.scale.configure(from_=0, to=float(meta.timestamps[-1]))
        self.canvas.delete("msg")
        self.show(0)
        done(meta, "")

    # ------------------------------------------------------------ 表示
    @property
    def time(self) -> float:
        return float(self.meta.timestamps[self.index]) if self.meta else 0.0

    def show(self, index: int) -> None:
        if self.meta is None or self.cap is None:
            return
        index = int(np.clip(index, 0, self.meta.n_frames - 1))
        if index != self._pos + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if not ok:
            return
        self._pos = index
        self.index = index
        self._frame = frame
        self.var_t.set(self.time)
        self.lbl.configure(text=f"{self.time:8.3f} s   フレーム {index} / {self.meta.n_frames - 1}")
        self._redraw()
        if self.on_frame:
            self.on_frame(index, self.time)

    def _redraw(self) -> None:
        if self._frame is None:
            return
        h, w = self._frame.shape[:2]
        s = min(self.cw / w, self.ch / h)
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        self._scale = s
        self._offset = ((self.cw - nw) // 2, (self.ch - nh) // 2)
        img = cv2.cvtColor(cv2.resize(self._frame, (nw, nh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(img))
        self.canvas.delete("frame", "ov")
        self.canvas.create_image(*self._offset, image=self._photo, anchor="nw", tags="frame")
        if self.roi:
            x0, y0 = self.to_canvas(self.roi[0], self.roi[1])
            x1, y1 = self.to_canvas(self.roi[2], self.roi[3])
            self.canvas.create_rectangle(x0, y0, x1, y1, outline="#00e0ff", width=2, dash=(6, 3), tags="ov")
            self.canvas.create_text(x0 + 4, y0 + 4, text="ROI", anchor="nw", fill="#00e0ff", tags="ov")
        for name, (u, v, color) in self.markers.items():
            x, y = self.to_canvas(u, v)
            self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, outline=color, width=2, tags="ov")
            self.canvas.create_text(x + 9, y - 9, text=name, anchor="sw", fill=color,
                                    font=("Yu Gothic UI", 10, "bold"), tags="ov")

    def to_canvas(self, u: float, v: float) -> tuple[float, float]:
        return self._offset[0] + u * self._scale, self._offset[1] + v * self._scale

    def to_image(self, x: float, y: float) -> tuple[float, float]:
        return (x - self._offset[0]) / self._scale, (y - self._offset[1]) / self._scale

    def _resize(self, e) -> None:
        self.cw, self.ch = max(e.width, 10), max(e.height, 10)
        self._redraw()

    # ------------------------------------------------------------ 操作
    def _on_slider(self) -> None:
        if self.meta is None or self._playing:
            return
        if self._pending:
            self.after_cancel(self._pending)
        t = self.var_t.get()
        self._pending = self.after(60, lambda: self.show(self.meta.frame_at(t)))  # 動かしている間は間引く

    def step(self, n: int) -> None:
        self.show(self.index + n)

    def step_time(self, dt: float) -> None:
        if self.meta:
            self.show(self.meta.frame_at(self.time + dt))

    def seek(self, t: float) -> None:
        if self.meta:
            self.show(self.meta.frame_at(t))

    def toggle_play(self, until: float | None = None) -> None:
        if self.meta is None:
            return
        self._playing = not self._playing
        self.btn_play.configure(text="■ 停止" if self._playing else "▶ 再生")
        if self._playing:
            self._play_until = until
            self._tick()

    def play_range(self, t0: float, t1: float) -> None:
        if self._playing:
            self.toggle_play()
        self.seek(t0)
        self.toggle_play(until=t1)

    def _tick(self) -> None:
        if not self._playing:
            return
        if self.index >= self.meta.n_frames - 1 or (self._play_until is not None and self.time >= self._play_until):
            self.toggle_play()
            return
        self.step(1)
        self.after(max(1, int(1000 / min(self.meta.fps_meta, 30))), self._tick)

    def _press(self, e) -> None:
        if self._frame is None:
            return
        if self.mode == "roi":
            self._drag_start = (e.x, e.y)
        elif self.mode == "point" and self.on_point:
            u, v = self.to_image(e.x, e.y)
            self.on_point(u, v)

    def _motion(self, e) -> None:
        if self.mode == "roi" and self._drag_start:
            self.canvas.delete("drag")
            self.canvas.create_rectangle(*self._drag_start, e.x, e.y, outline="#00e0ff", width=2, tags="drag")

    def _release(self, e) -> None:
        if self.mode == "roi" and self._drag_start:
            self.canvas.delete("drag")
            (x0, y0), (x1, y1) = self.to_image(*self._drag_start), self.to_image(e.x, e.y)
            self._drag_start = None
            w, h = self.meta.width, self.meta.height
            box = [float(np.clip(min(x0, x1), 0, w)), float(np.clip(min(y0, y1), 0, h)),
                   float(np.clip(max(x0, x1), 0, w)), float(np.clip(max(y0, y1), 0, h))]
            self.roi = box if (box[2] - box[0] > 10 and box[3] - box[1] > 10) else None
            self._redraw()
            if self.on_roi:
                self.on_roi(self.roi)

    def set_marker(self, name: str, uv: tuple[float, float] | None, color: str = "#ffcc00") -> None:
        if uv is None:
            self.markers.pop(name, None)
        else:
            self.markers[name] = (uv[0], uv[1], color)
        self._redraw()
