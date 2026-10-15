"""OpenCV ウィンドウの共通部品：拡大表示つきの画像ビューと、順番に点をクリックさせるピッカー。

状態の更新（キー・マウス）と描画を分けてあるので、ウィンドウを開かずにテストできる。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

# cv2.waitKeyEx の矢印キー（Windows / Linux）
KEY_LEFT = {2424832, 65361}
KEY_UP = {2490368, 65362}
KEY_RIGHT = {2555904, 65363}
KEY_DOWN = {2621440, 65364}
KEY_ENTER = {13, 10}
KEY_ESC = {27}
KEY_BACKSPACE = {8}


class ImageView:
    """画像をウィンドウに収めて表示し、ズーム・パンを扱う。"""

    def __init__(self, img_w: int, img_h: int, disp_w: int = 1600, disp_h: int = 900):
        self.img_w, self.img_h = img_w, img_h
        self.base = min(disp_w / img_w, disp_h / img_h)
        self.disp_w, self.disp_h = int(round(img_w * self.base)), int(round(img_h * self.base))
        self.zoom = 1.0
        self.center = np.array([img_w / 2, img_h / 2])

    @property
    def scale(self) -> float:
        return self.base * self.zoom

    @property
    def origin(self) -> np.ndarray:
        """表示の左上に対応する画像座標 [px]。"""
        o = self.center - np.array([self.disp_w, self.disp_h]) / (2 * self.scale)
        return o

    def to_image(self, x: float, y: float) -> np.ndarray:
        """表示座標 → 画像座標 [px]。"""
        return self.origin + np.array([x, y]) / self.scale

    def to_display(self, uv: np.ndarray) -> np.ndarray:
        return (np.asarray(uv) - self.origin) * self.scale

    def zoom_at(self, x: float, y: float, factor: float) -> None:
        """表示座標 (x, y) の点を固定したまま拡大縮小する。"""
        p = self.to_image(x, y)
        self.zoom = float(np.clip(self.zoom * factor, 1.0, 32.0))
        self.center = p - (np.array([x, y]) - np.array([self.disp_w, self.disp_h]) / 2) / self.scale
        self._clamp()

    def pan(self, dx_disp: float, dy_disp: float) -> None:
        self.center = self.center - np.array([dx_disp, dy_disp]) / self.scale
        self._clamp()

    def reset(self) -> None:
        self.zoom = 1.0
        self.center = np.array([self.img_w / 2, self.img_h / 2])

    def _clamp(self) -> None:
        half = np.array([self.disp_w, self.disp_h]) / (2 * self.scale)
        lo, hi = half, np.array([self.img_w, self.img_h]) - half
        self.center = np.where(lo <= hi, np.clip(self.center, lo, hi), [self.img_w / 2, self.img_h / 2])

    def render(self, img: np.ndarray) -> np.ndarray:
        s, o = self.scale, self.origin
        M = np.array([[s, 0, -s * o[0]], [0, s, -s * o[1]]], np.float32)
        interp = cv2.INTER_NEAREST if self.zoom >= 4 else cv2.INTER_AREA if self.zoom < 1.5 else cv2.INTER_LINEAR
        return cv2.warpAffine(img, M, (self.disp_w, self.disp_h), flags=interp)


@dataclass
class PointPicker:
    """ターゲットを順番に提示し、画像上の点をクリックさせる。

    操作:
        左クリック: 現在のターゲットに点を置く（置いた点が選択状態になる）
        s: スキップ（画面に映っていない点）   u / BackSpace: 元に戻す
        b / n: 前 / 次のターゲットへ移動       矢印キー: 選択中の点を微調整
        [ / ]: 微調整の刻み（0.25 / 1 / 5 px）  ホイール / z / x: 拡大・縮小   右ドラッグ: 移動   r: 表示リセット
        c: 計算して再投影を表示（キャリブレーション時）  Enter: 確定   Esc: 中止   h: ヘルプ表示の切り替え
    """

    image: np.ndarray
    targets: list[str]
    view: ImageView = field(init=False)
    points: dict[str, np.ndarray | None] = field(default_factory=dict)
    idx: int = 0
    selected: str | None = None
    step_px: float = 1.0
    show_help: bool = True
    overlay: Callable[[np.ndarray, ImageView], None] | None = None
    diagram: Callable[[str | None, dict[str, np.ndarray | None]], np.ndarray] | None = None
    on_compute: Callable[["PointPicker"], None] | None = None
    status: str = ""
    _history: list[tuple[str, np.ndarray | None, bool, int]] = field(default_factory=list)
    _drag: tuple[int, int] | None = None
    _mouse: tuple[int, int] = (0, 0)

    STEPS = (0.25, 1.0, 5.0)

    def __post_init__(self) -> None:
        h, w = self.image.shape[:2]
        self.view = ImageView(w, h)

    # ------------------------------------------------------------ 状態
    @property
    def current(self) -> str | None:
        return self.targets[self.idx] if self.idx < len(self.targets) else None

    def placed(self) -> dict[str, np.ndarray]:
        """クリック済みの点（スキップを除く）。"""
        return {k: v for k, v in self.points.items() if v is not None}

    def _record(self, name: str) -> None:
        self._history.append((name, None if self.points.get(name) is None else self.points[name].copy(),
                              name in self.points, self.idx))

    def place(self, uv: np.ndarray) -> None:
        name = self.current
        if name is None:
            return
        self._record(name)
        self.points[name] = np.asarray(uv, float)
        self.selected = name
        self.idx += 1

    def skip(self) -> None:
        name = self.current
        if name is None:
            return
        self._record(name)
        self.points[name] = None
        self.idx += 1

    def undo(self) -> None:
        if not self._history:
            return
        name, prev, existed, idx = self._history.pop()
        if existed:
            self.points[name] = prev
        else:
            self.points.pop(name, None)
        self.idx = idx
        self.selected = name if self.points.get(name) is not None else None

    def nudge(self, dx: float, dy: float) -> None:
        if self.selected is None or self.points.get(self.selected) is None:
            return
        self._record(self.selected)
        self.points[self.selected] = self.points[self.selected] + np.array([dx, dy]) * self.step_px

    # ------------------------------------------------------------ 入力
    def handle_key(self, key: int) -> str | None:
        """キー入力を処理する。確定なら "done"、中止なら "abort" を返す。"""
        if key < 0:
            return None
        if key in KEY_ENTER:
            return "done"
        if key in KEY_ESC:
            return "abort"
        if key in KEY_LEFT:
            self.nudge(-1, 0)
        elif key in KEY_RIGHT:
            self.nudge(1, 0)
        elif key in KEY_UP:
            self.nudge(0, -1)
        elif key in KEY_DOWN:
            self.nudge(0, 1)
        elif key in KEY_BACKSPACE:
            self.undo()
        else:
            ch = chr(key & 0xFF).lower()
            if ch == "s":
                self.skip()
            elif ch == "u":
                self.undo()
            elif ch == "b":
                self.idx = max(self.idx - 1, 0)
            elif ch == "n":
                self.idx = min(self.idx + 1, len(self.targets))
            elif ch == "[":
                self.step_px = self.STEPS[max(self.STEPS.index(self.step_px) - 1, 0)]
            elif ch == "]":
                self.step_px = self.STEPS[min(self.STEPS.index(self.step_px) + 1, len(self.STEPS) - 1)]
            elif ch == "z":
                self.view.zoom_at(*self._mouse, 1.5)
            elif ch == "x":
                self.view.zoom_at(*self._mouse, 1 / 1.5)
            elif ch == "r":
                self.view.reset()
            elif ch == "h":
                self.show_help = not self.show_help
            elif ch == "c" and self.on_compute is not None:
                self.on_compute(self)
        return None

    def handle_mouse(self, event: int, x: int, y: int, flags: int) -> None:
        self._mouse = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.place(self.view.to_image(x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._drag = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self._drag is not None:
            self.view.pan(x - self._drag[0], y - self._drag[1])
            self._drag = (x, y)
        elif event == cv2.EVENT_RBUTTONUP:
            self._drag = None
        elif event == cv2.EVENT_MOUSEWHEEL:
            self.view.zoom_at(x, y, 1.25 if flags > 0 else 1 / 1.25)

    # ------------------------------------------------------------ 描画
    def render(self) -> np.ndarray:
        img = self.image.copy()
        if self.overlay is not None:
            self.overlay(img, self.view)
        disp = self.view.render(img)
        for name, uv in self.points.items():
            if uv is None:
                continue
            p = self.view.to_display(uv)
            color = (0, 0, 255) if name == self.selected else (0, 255, 0)
            q = tuple(np.round(p).astype(int))
            cv2.drawMarker(disp, q, color, cv2.MARKER_CROSS, 18, 1, cv2.LINE_AA)
            cv2.putText(disp, name, (q[0] + 8, q[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        bar = f"[{self.idx + 1}/{len(self.targets)}] {self.current or 'ALL DONE - press c / Enter'}"
        bar += f"   zoom x{self.view.zoom:.1f}  step {self.step_px}px"
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(disp, bar, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        if self.status:
            cv2.rectangle(disp, (0, disp.shape[0] - 24), (disp.shape[1], disp.shape[0]), (0, 0, 0), -1)
            cv2.putText(disp, self.status, (8, disp.shape[0] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1, cv2.LINE_AA)
        if self.show_help:
            lines = ["click: place   s: skip   u/BS: undo   b/n: prev/next",
                     "arrows: nudge   [ ]: step   wheel/z/x: zoom   R-drag: pan   r: reset",
                     "c: compute & reproject   Enter: save   Esc: abort   h: hide help"]
            for i, t in enumerate(lines):
                cv2.putText(disp, t, (8, 46 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(disp, t, (8, 46 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        if self.diagram is not None:
            d = self.diagram(self.current, self.points)
            dh, dw = d.shape[:2]
            disp[30:30 + dh, disp.shape[1] - dw - 6:disp.shape[1] - 6] = d
        return disp

    def run(self, window: str = "smash-speed") -> str:
        """ウィンドウを開いて操作を受け付ける。"done" か "abort" を返す。"""
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window, lambda e, x, y, f, _p: self.handle_mouse(e, x, y, f))
        try:
            while True:
                cv2.imshow(window, self.render())
                r = self.handle_key(cv2.waitKeyEx(20))
                if r is not None:
                    return r
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    return "abort"
        finally:
            cv2.destroyWindow(window)
