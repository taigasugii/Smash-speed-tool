"""アノテーションツール：シャトル位置・不可視・打点・着地のラベル付け。

出力（``out_dir``）:
    labels.csv            frame, timestamp, x_px, y_px, visible, event, source（本ツールの形式）
    <動画名>_ball.csv      Frame, Visibility, X, Y（TrackNetV3 の学習データ形式。フレーム番号は区間の先頭が 0）
    annotation_meta.json  動画・区間・作業日時

操作（OpenCV ウィンドウ）:
    左クリック: シャトルの中心を置く（自動で次のフレームへ。m で切り替え）
    x: このフレームは見えない      Space: 提案位置（補間 / 初期値）を採用
    a / d: 前 / 次のフレーム        A / D: 10 フレーム戻る / 進む   f / F: 30 フレーム進む / 戻る
    h: 打点フレームの切り替え       g: 着地フレームの切り替え
    i: 前後のラベルから中間フレームを線形補間（提案として入れる）
    u: このフレームのラベルを消す   m: 自動で次へ進むかの切り替え
    ホイール / z / c: 拡大・縮小   右ドラッグ: 移動   r: 表示リセット
    w: 保存   Esc / q: 保存して終了
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from smash_speed.repro import run_metadata
from smash_speed.ui import KEY_ESC, KEY_LEFT, KEY_RIGHT, ImageView
from smash_speed.video_io import VideoMeta


@dataclass
class AnnotationState:
    """ラベルの状態（ウィンドウなしで操作・テストできる）。

    ``visible``: 1 = 見える、0 = 見えない、-1 = 未ラベル。``source``: "manual" | "interp" | "init" | ""。
    """

    frames: np.ndarray
    timestamps: np.ndarray
    x: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)
    visible: np.ndarray = field(init=False)
    source: np.ndarray = field(init=False)
    event: np.ndarray = field(init=False)
    cur: int = 0
    auto_advance: bool = True

    def __post_init__(self) -> None:
        n = len(self.frames)
        self.x = np.full(n, np.nan)
        self.y = np.full(n, np.nan)
        self.visible = np.full(n, -1, int)
        self.source = np.array([""] * n, dtype=object)
        self.event = np.array([""] * n, dtype=object)

    @property
    def n(self) -> int:
        return len(self.frames)

    def go(self, delta: int) -> None:
        self.cur = int(np.clip(self.cur + delta, 0, self.n - 1))

    def set_point(self, x: float, y: float, source: str = "manual") -> None:
        i = self.cur
        self.x[i], self.y[i], self.visible[i], self.source[i] = x, y, 1, source
        if self.auto_advance:
            self.go(1)

    def set_invisible(self) -> None:
        i = self.cur
        self.x[i] = self.y[i] = np.nan
        self.visible[i], self.source[i] = 0, "manual"
        if self.auto_advance:
            self.go(1)

    def clear(self) -> None:
        i = self.cur
        self.x[i] = self.y[i] = np.nan
        self.visible[i], self.source[i] = -1, ""

    def toggle_event(self, name: str) -> None:
        """打点（hit）・着地（land）は動画に1つずつ。別のフレームに付け直すと移動する。"""
        i = self.cur
        if self.event[i] == name:
            self.event[i] = ""
            return
        self.event[self.event == name] = ""
        self.event[i] = name

    def suggestion(self, i: int | None = None) -> tuple[float, float] | None:
        """未ラベルのフレームに対する提案位置（前後の手動ラベルの線形補間）。"""
        i = self.cur if i is None else i
        if self.visible[i] == 1:
            return float(self.x[i]), float(self.y[i])
        man = np.flatnonzero((self.visible == 1) & (self.source == "manual"))
        before, after = man[man < i], man[man > i]
        if len(before) == 0 or len(after) == 0:
            return None
        a, b = before[-1], after[0]
        if np.any(self.visible[a + 1:b] == 0):  # 間に「見えない」があれば補間しない
            return None
        w = (self.timestamps[i] - self.timestamps[a]) / (self.timestamps[b] - self.timestamps[a])
        return float(self.x[a] + w * (self.x[b] - self.x[a])), float(self.y[a] + w * (self.y[b] - self.y[a]))

    def interpolate_gaps(self) -> int:
        """手動ラベルの間の未ラベルのフレームを線形補間で埋める（source = "interp"）。埋めた数を返す。"""
        count = 0
        for i in np.flatnonzero(self.visible == -1):
            s = self.suggestion(int(i))
            if s is not None:
                self.x[i], self.y[i], self.visible[i], self.source[i] = s[0], s[1], 1, "interp"
                count += 1
        return count

    def accept_suggestion(self) -> bool:
        s = self.suggestion()
        if s is None:
            return False
        src = self.source[self.cur] if self.source[self.cur] in ("interp", "init") else "manual"
        self.set_point(*s, source=src if src else "manual")
        return True

    def load_initial(self, df: pd.DataFrame) -> int:
        """検出結果や既存ラベルを初期値として読み込む（未ラベルのフレームだけ）。読み込んだ数を返す。"""
        df = normalize_labels(df)
        idx = {int(f): k for k, f in enumerate(self.frames)}
        count = 0
        for r in df.itertuples():
            k = idx.get(int(r.frame))
            if k is None or self.visible[k] != -1:
                continue
            if r.visible == 1 and np.isfinite(r.x_px):
                self.x[k], self.y[k], self.visible[k], self.source[k] = r.x_px, r.y_px, 1, "init"
                count += 1
            ev = getattr(r, "event", "")
            if isinstance(ev, str) and ev:
                self.event[k] = ev
        return count

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"frame": self.frames, "timestamp": self.timestamps, "x_px": self.x,
                             "y_px": self.y, "visible": self.visible, "event": self.event,
                             "source": self.source})


def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """本ツール・検出 CSV・TrackNetV3 の形式（Frame, Visibility, X, Y）を共通の列名にそろえる。"""
    if "Frame" in df.columns:
        df = df.rename(columns={"Frame": "frame", "Visibility": "visible", "X": "x_px", "Y": "y_px"})
        df.loc[df.visible == 0, ["x_px", "y_px"]] = np.nan
    if "event" not in df.columns:
        df = df.assign(event="")
    df["event"] = df["event"].fillna("")
    return df


def save_labels(state: AnnotationState, out_dir: Path, video_name: str, meta: VideoMeta,
                start_s: float | None, end_s: float | None) -> None:
    """``labels.csv``・TrackNetV3 形式の CSV・メタデータを保存する。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = state.to_frame()
    df.to_csv(out_dir / "labels.csv", index=False)
    tn = pd.DataFrame({"Frame": df.frame - df.frame.iloc[0],
                       "Visibility": (df.visible == 1).astype(int),
                       "X": np.where(df.visible == 1, df.x_px, 0.0).round(1),
                       "Y": np.where(df.visible == 1, df.y_px, 0.0).round(1)})
    tn.to_csv(out_dir / f"{video_name}_ball.csv", index=False)
    ev = {e: float(df.timestamp[df.event == e].iloc[0]) for e in ("hit", "land") if (df.event == e).any()}
    (out_dir / "annotation_meta.json").write_text(json.dumps({
        "video": meta.path, "start_s": start_s, "end_s": end_s, "capture_fps": meta.capture_fps,
        "n_frames": int(state.n), "n_labeled": int((df.visible >= 0).sum()),
        "events_s": ev, "meta": run_metadata()}, ensure_ascii=False, indent=1), encoding="utf-8")


class _FrameCache:
    """直近のフレームを保持する（4K では1枚 約 25 MB なので上限つき）。"""

    def __init__(self, path: str, max_bytes: float = 1.5e9, meta=None):
        self.cap = cv2.VideoCapture(path)
        self.meta = meta  # あれば、番号で読んだフレームの時刻を確かめる（``seek_read``）
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.max_bytes = max_bytes
        self.pos = -1

    def get(self, i: int) -> np.ndarray:
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        if i == self.pos + 1:
            ok, img = self.cap.read()
        elif self.meta is not None:
            from smash_speed.video_io import seek_read

            img = seek_read(self.cap, self.meta, i)
            ok = img is not None
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, img = self.cap.read()
        if not ok:
            raise IndexError(i)
        self.pos = i
        self.cache[i] = img
        while len(self.cache) * img.nbytes > self.max_bytes and len(self.cache) > 4:
            self.cache.popitem(last=False)
        return img


def run_annotation(meta: VideoMeta, out_dir: Path, start_s: float | None = None, end_s: float | None = None,
                   init_csv: Path | None = None) -> AnnotationState:
    """アノテーションのウィンドウを開く。既存の ``labels.csv`` があれば続きから始める。"""
    ts = meta.timestamps
    i0 = 0 if start_s is None else int(np.searchsorted(ts, start_s))
    i1 = len(ts) - 1 if end_s is None else int(np.searchsorted(ts, end_s, side="right")) - 1
    frames = np.arange(i0, i1 + 1)
    state = AnnotationState(frames, ts[frames])
    existing = out_dir / "labels.csv"
    if existing.exists():
        prev = pd.read_csv(existing).fillna({"event": "", "source": ""})
        idx = {int(f): k for k, f in enumerate(frames)}
        for r in prev.itertuples():
            k = idx.get(int(r.frame))
            if k is not None:
                state.x[k], state.y[k], state.visible[k] = r.x_px, r.y_px, int(r.visible)
                state.source[k], state.event[k] = r.source, r.event
        print(f"既存のラベルを読み込みました: {existing}")
    if init_csv is not None:
        n = state.load_initial(pd.read_csv(init_csv))
        print(f"初期値を {n} フレーム読み込みました: {init_csv}")
    name = Path(meta.path).stem
    cache = _FrameCache(meta.path, meta=meta)
    view = ImageView(meta.width, meta.height)
    win = "annotate"
    mouse = {"xy": (0, 0), "drag": None}

    def on_mouse(e: int, x: int, y: int, flags: int, _p) -> None:
        mouse["xy"] = (x, y)
        if e == cv2.EVENT_LBUTTONDOWN:
            state.set_point(*view.to_image(x, y))
        elif e == cv2.EVENT_RBUTTONDOWN:
            mouse["drag"] = (x, y)
        elif e == cv2.EVENT_MOUSEMOVE and mouse["drag"] is not None:
            view.pan(x - mouse["drag"][0], y - mouse["drag"][1])
            mouse["drag"] = (x, y)
        elif e == cv2.EVENT_RBUTTONUP:
            mouse["drag"] = None
        elif e == cv2.EVENT_MOUSEWHEEL:
            view.zoom_at(x, y, 1.25 if flags > 0 else 1 / 1.25)

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    try:
        while True:
            img = cache.get(int(state.frames[state.cur])).copy()
            lo = max(0, state.cur - 15)
            for k in range(lo, state.cur):  # 直前のラベルの軌跡
                if state.visible[k] == 1:
                    cv2.circle(img, (int(state.x[k]), int(state.y[k])), 3, (0, 200, 0), -1)
            disp = view.render(img)
            i = state.cur
            if state.visible[i] == 1:
                q = view.to_display((state.x[i], state.y[i])).round().astype(int)
                color = {"manual": (0, 0, 255), "interp": (255, 200, 0), "init": (0, 200, 255)}.get(state.source[i], (0, 0, 255))
                cv2.drawMarker(disp, tuple(q), color, cv2.MARKER_CROSS, 24, 1)
            else:
                s = state.suggestion()
                if s is not None:
                    q = view.to_display(s).round().astype(int)
                    cv2.circle(disp, tuple(q), 10, (255, 200, 0), 1, cv2.LINE_AA)
            lab = {1: "visible", 0: "INVISIBLE", -1: "-"}[int(state.visible[i])]
            txt = (f"frame {state.frames[i]} ({i + 1}/{state.n})  t={state.timestamps[i]:.4f}s  {lab} "
                   f"{state.source[i]}  {('[' + state.event[i].upper() + ']') if state.event[i] else ''}"
                   f"  labeled {(state.visible >= 0).sum()}/{state.n}  auto-advance={'on' if state.auto_advance else 'off'}")
            cv2.rectangle(disp, (0, 0), (disp.shape[1], 26), (0, 0, 0), -1)
            cv2.putText(disp, txt, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(win, disp)
            k = cv2.waitKeyEx(15)
            if k < 0:
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
                continue
            ch = chr(k & 0xFF) if k < 0x10000 else ""
            if k in KEY_ESC or ch == "q":
                break
            if k in KEY_RIGHT or ch == "d":
                state.go(1)
            elif k in KEY_LEFT or ch == "a":
                state.go(-1)
            elif ch == "D":
                state.go(10)
            elif ch == "A":
                state.go(-10)
            elif ch == "f":
                state.go(30)
            elif ch == "F":
                state.go(-30)
            elif ch == "x":
                state.set_invisible()
            elif ch == " ":
                state.accept_suggestion()
            elif ch == "h":
                state.toggle_event("hit")
            elif ch == "g":
                state.toggle_event("land")
            elif ch == "i":
                print(f"{state.interpolate_gaps()} フレームを補間しました")
            elif ch == "u":
                state.clear()
            elif ch == "m":
                state.auto_advance = not state.auto_advance
            elif ch == "z":
                view.zoom_at(*mouse["xy"], 1.5)
            elif ch == "c":
                view.zoom_at(*mouse["xy"], 1 / 1.5)
            elif ch == "r":
                view.reset()
            elif ch == "w":
                save_labels(state, out_dir, name, meta, start_s, end_s)
                print("保存しました")
    finally:
        cv2.destroyWindow(win)
        save_labels(state, out_dir, name, meta, start_s, end_s)
        print(f"保存しました: {out_dir}")
    return state
