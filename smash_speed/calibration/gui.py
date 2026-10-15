"""キャリブレーションと飛行平面指定の GUI（OpenCV ウィンドウ）。"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from smash_speed.calibration.io import landmark_table
from smash_speed.calibration.solve import CalibrationResult, Intrinsics, calibrate
from smash_speed.camera import Camera
from smash_speed.court import court_segments, draw_court
from smash_speed.geometry import FlightPlane, floor_point
from smash_speed.ui import ImageView, PointPicker


def court_diagram(cfg: dict[str, Any], current: str | None, points: dict[str, Any],
                  width: int = 260) -> np.ndarray:
    """コートの上面図に、ランドマーク（現在のもの＝赤、クリック済み＝緑、スキップ＝灰）を描く。

    上面図は x を右、y を上にとる。高さのある点は二重丸で示す。
    """
    L, W = cfg["court"]["half_length_m"], cfg["court"]["half_width_doubles_m"]
    margin = 10
    s = (width - 2 * margin) / (2 * L)
    height = int(2 * W * s + 2 * margin + 18)
    img = np.full((height, width, 3), 40, np.uint8)

    def to_px(x: float, y: float) -> tuple[int, int]:
        return int(round(margin + (x + L) * s)), int(round(margin + (W - y) * s))

    for a, b, kind in court_segments(cfg):
        if kind == "line" or (a[2] == 0 and b[2] == 0):
            cv2.line(img, to_px(a[0], a[1]), to_px(b[0], b[1]),
                     (200, 200, 200) if kind == "line" else (200, 0, 200), 1, cv2.LINE_AA)
    cv2.putText(img, "hitter (x<0)", (margin, height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
    for lm in cfg["landmarks"]:
        x, y, z = lm["xyz"]
        name = lm["name"]
        if name == current:
            color, r = (0, 0, 255), 5
        elif name in points:
            color, r = ((0, 220, 0), 4) if points[name] is not None else ((120, 120, 120), 3)
        else:
            color, r = (255, 255, 255), 2
        p = to_px(x, y)
        cv2.circle(img, p, r, color, -1, cv2.LINE_AA)
        if z > 0:
            cv2.circle(img, p, r + 4, color, 1, cv2.LINE_AA)
    return img


def run_calibration_gui(frame: np.ndarray, cfg: dict[str, Any],
                        initial: dict[str, tuple[float, float]] | None = None,
                        intrinsics: Intrinsics | None = None,
                        estimate_k1: bool = False) -> CalibrationResult | None:
    """ランドマークを順にクリックさせ、カメラを推定する。確定しなければ None。

    ``c`` キーで推定し、コートのラインとネットを再投影して重ね描きする（誤差が大きい点は赤丸）。
    """
    table = landmark_table(cfg)
    names = list(table)
    h, w = frame.shape[:2]
    warn = cfg["calibration"]["reproj_warn_px"]
    state: dict[str, Any] = {"result": None}

    picker = PointPicker(frame, names)
    if initial:
        for n in names:
            if n in initial:
                picker.points[n] = np.asarray(initial[n], float)
        picker.idx = len(names)
    picker.diagram = lambda cur, pts: court_diagram(cfg, cur, pts)

    def compute(p: PointPicker) -> None:
        placed = p.placed()
        if len(placed) < 4:
            p.status = f"need >= 4 points (now {len(placed)})"
            return
        ks = [n for n in names if n in placed]
        try:
            res = calibrate(ks, np.array([table[k] for k in ks]), np.array([placed[k] for k in ks]),
                            w, h, intrinsics=intrinsics, estimate_k1=estimate_k1, warn_px=warn)
        except (RuntimeError, ValueError) as e:
            p.status = f"failed: {e}"
            return
        state["result"] = res
        n_bad = int((res.errors_px > warn).sum())
        p.status = (f"RMSE {res.rmse_px:.2f}px  f={res.camera.f:.0f}px  points={len(ks)}  "
                    f"bad(>{warn}px)={n_bad}   Enter: save")
        print("\n=== 再投影誤差 ===")
        for n, e in zip(res.names, res.errors_px):
            print(f"  {n:24s} {e:6.2f} px{'  <-- 大きい' if e > warn else ''}")
        for wmsg in res.warnings:
            print("  [警告]", wmsg)

        def overlay(img: np.ndarray, view: ImageView) -> None:
            t = max(2, int(round(2 / view.scale)))
            draw_court(img, res.camera, cfg, thickness=t)
            for name, e in zip(res.names, res.errors_px):
                uv = res.camera.project(table[name][None])[0]
                color = (0, 0, 255) if e > warn else (255, 255, 0)
                cv2.circle(img, tuple(np.round(uv).astype(int)), max(6, int(e * 2)), color, t, cv2.LINE_AA)
        p.overlay = overlay

    picker.on_compute = compute
    if initial:
        compute(picker)
    while True:
        r = picker.run("calibration")
        if r == "abort":
            return None
        if state["result"] is None or picker.status.startswith("need") or "failed" in picker.status:
            compute(picker)
        if state["result"] is not None:
            return state["result"]


def run_plane_gui(frame: np.ndarray, camera: Camera, cfg: dict[str, Any]) -> FlightPlane | None:
    """打者の足元と着地点（床面）をクリックさせ、飛行平面を作る。"""
    picker = PointPicker(frame, ["hitter_foot", "landing_point"])
    picker.overlay = lambda img, view: draw_court(img, camera, cfg, thickness=max(2, int(2 / view.scale)))
    picker.status = "click hitter's foot position, then the shuttle landing point (on the floor)"
    if picker.run("flight plane") != "done":
        return None
    pts = picker.placed()
    if len(pts) < 2:
        print("2点ともクリックしてください")
        return None
    a = floor_point(camera, pts["hitter_foot"][None])[0]
    b = floor_point(camera, pts["landing_point"][None])[0]
    print(f"打者の足元 {a.round(2)} m、着地点 {b.round(2)} m")
    return FlightPlane.from_points(a[:2], b[:2])
