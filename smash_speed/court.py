"""コートのライン・ネットの3次元形状（再投影の重ね描き・合成動画の描画に使う）。"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from smash_speed.camera import Camera


def court_segments(cfg: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """コートのライン（ラインの外側の縁）とネットの線分を返す。

    Returns:
        (始点 [m], 終点 [m], 種類 "line" | "net") のリスト
    """
    c = cfg["court"]
    L, W, Ws = c["half_length_m"], c["half_width_doubles_m"], c["half_width_singles_m"]
    ss, ls = c["short_service_m"], c["long_service_doubles_m"]
    hc, hp = c["net_height_center_m"], c["net_height_post_m"]
    P = lambda x, y, z=0.0: np.array([x, y, z], float)  # noqa: E731
    segs: list[tuple[np.ndarray, np.ndarray, str]] = []
    for sx in (-1, 1):
        segs.append((P(sx * L, -W), P(sx * L, W), "line"))          # バックバウンダリー
        segs.append((P(sx * ls, -W), P(sx * ls, W), "line"))        # ダブルスのロングサービス
        segs.append((P(sx * ss, -W), P(sx * ss, W), "line"))        # ショートサービス
        segs.append((P(sx * ss, 0), P(sx * L, 0), "line"))          # センター
    for sy in (-1, 1):
        segs.append((P(-L, sy * W), P(L, sy * W), "line"))          # ダブルスのサイド
        segs.append((P(-L, sy * Ws), P(L, sy * Ws), "line"))        # シングルスのサイド
        segs.append((P(0, sy * W), P(0, sy * W, hp), "net"))        # ネットポスト
    segs.append((P(0, -W, hp), P(0, 0, hc), "net"))                 # ネット上端（中央でたわむ）
    segs.append((P(0, 0, hc), P(0, W, hp), "net"))
    segs.append((P(0, -W), P(0, W), "net"))                         # ネット直下の床
    return segs


def draw_court(img: np.ndarray, camera: Camera, cfg: dict[str, Any],
               color_line: tuple[int, int, int] = (0, 255, 255),
               color_net: tuple[int, int, int] = (255, 0, 255),
               thickness: int = 2, n_samples: int = 40) -> np.ndarray:
    """カメラでコートを投影して画像に重ね描きする（歪みを反映するため線分を細かく分割）。"""
    for a, b, kind in court_segments(cfg):
        s = np.linspace(0.0, 1.0, n_samples)[:, None]
        uv = camera.project(a + s * (b - a))
        ok = np.isfinite(uv).all(axis=1)
        pts = uv[ok].round().astype(np.int32)
        if len(pts) >= 2:
            cv2.polylines(img, [pts], False, color_net if kind == "net" else color_line,
                          thickness, cv2.LINE_AA)
    return img
