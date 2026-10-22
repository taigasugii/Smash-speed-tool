"""半自動キャリブレーション（おおよその 4 点 → コートのラインに吸着）のテスト。"""

import cv2
import numpy as np

from smash_speed.calibration import landmark_table
from smash_speed.calibration.auto_court_calibrator import LINE_WIDTH_M, _apply_h
from smash_speed.calibration.snap import LINES, CourtSnapper
from smash_speed.config import load_config


def _court_image(Hm: np.ndarray, size=(1080, 1920), line_bgr=(60, 110, 70), seams=True, seed=0,
                 cross_seams=False) -> np.ndarray:
    """木の床（明るい茶色・暗い継ぎ目）に、暗い緑のコートのラインを描いた画像。

    ``cross_seams`` なら、床板の端の継ぎ目（もう 1 方向、0.9 m ごと）も描く（実際の床では両方向に線分がある）。
    """
    h, w = size
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), (150, 190, 215), np.float32) + rng.normal(0, 3, (h, w, 3))
    ss = 4                                                    # 細いラインも滑らかに描くため 4 倍で描いて縮める
    big = np.zeros((h * ss, w * ss), np.uint8)
    if seams:
        # ラインと平行な床板の継ぎ目（細く暗い線）を 0.13 m ごとに
        seam = np.zeros_like(big)
        for x in np.arange(-9, 9, 0.13):
            p = _apply_h(Hm, np.array([[x, -6.0], [x, 6.0]])) * ss
            cv2.line(seam, tuple(np.int32(p[0])), tuple(np.int32(p[1])), 255, ss, cv2.LINE_AA)
        for y in (np.arange(-6, 6, 0.9) if cross_seams else []):
            p = _apply_h(Hm, np.array([[-9.0, y], [9.0, y]])) * ss
            cv2.line(seam, tuple(np.int32(p[0])), tuple(np.int32(p[1])), 255, ss, cv2.LINE_AA)
        m = cv2.resize(seam, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)[..., None] / 255
        img = img * (1 - 0.35 * m)
    for name, (ax, c, spans) in LINES.items():
        for a, b in spans:
            d = LINE_WIDTH_M / 2
            quad = (np.array([[c - d, a], [c + d, a], [c + d, b], [c - d, b]]) if ax == "x" else
                    np.array([[a, c - d], [b, c - d], [b, c + d], [a, c + d]]))
            cv2.fillPoly(big, [np.int32(_apply_h(Hm, quad) * ss)], 255, cv2.LINE_AA)
    m = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)[..., None] / 255
    img = img * (1 - m) + np.array(line_bgr, np.float32) * m
    return np.clip(img, 0, 255).astype(np.uint8)


def _side_view_h() -> np.ndarray:
    """コートの横・少し高い所から撮ったときの床のホモグラフィ（手前のサイドラインが画面の下）。"""
    world = np.float32([[-6.7, -3.05], [6.7, -3.05], [6.7, 3.05], [-6.7, 3.05]])
    img = np.float32([[180, 900], [1760, 880], [1380, 520], [560, 530]])
    return cv2.getPerspectiveTransform(world, img).astype(float)


def test_snap_recovers_court_from_rough_clicks_despite_floor_seams():
    table = landmark_table(load_config())
    Hm = _side_view_h()
    frame = _court_image(Hm)
    rng = np.random.default_rng(3)
    names = ["back_corner_h_n", "back_corner_h_p", "short_service_side_h_n", "short_service_side_h_p"]
    clicks = {n: tuple(_apply_h(Hm, table[n][None, :2])[0] + rng.normal(0, 7, 2)) for n in names}
    res = CourtSnapper().snap(frame, clicks, table)
    snapped = [n for n, s in res.status.items() if s == "snapped"]
    assert len(snapped) >= 15
    err = [np.linalg.norm(np.array(res.points[n]) - _apply_h(Hm, table[n][None, :2])[0]) for n in res.points]
    assert np.median(err) < 1.0
    assert max(err) < 3.0
    # クリックはずれていたが、クリックした点も吸い付けた位置では正しい
    for n in names:
        assert np.linalg.norm(np.array(res.points[n]) - _apply_h(Hm, table[n][None, :2])[0]) < 1.5


def test_snap_finds_bright_lines_too():
    """体育館によってラインは白いこともある（床より明るい）。"""
    table = landmark_table(load_config())
    Hm = _side_view_h()
    frame = _court_image(Hm, line_bgr=(245, 245, 245), seed=1)
    rng = np.random.default_rng(4)
    names = ["back_corner_h_n", "back_corner_h_p", "post_base_n", "post_base_p"]
    clicks = {n: tuple(_apply_h(Hm, table[n][None, :2])[0] + rng.normal(0, 7, 2)) for n in names}
    res = CourtSnapper().snap(frame, clicks, table)
    err = [np.linalg.norm(np.array(res.points[n]) - _apply_h(Hm, table[n][None, :2])[0]) for n in res.points]
    assert np.median(err) < 1.0


def test_snap_needs_four_floor_points():
    table = landmark_table(load_config())
    frame = np.zeros((100, 100, 3), np.uint8)
    clicks = {"back_corner_h_n": (1, 1), "back_corner_h_p": (5, 1), "post_top_n": (3, 3), "post_top_p": (4, 4)}
    try:
        CourtSnapper().snap(frame, clicks, table)
    except ValueError as e:
        assert "4 点" in str(e)
    else:
        raise AssertionError("床の点が 2 点しかないのに通った")


def _camera_h(C=(-8.5, -8.0, 1.3), target=(0.0, 0.0, 0.0), f=1500.0, size=(1080, 1920)) -> np.ndarray:
    """ピンホールカメラ（主点は画像の中心・正方画素）で撮ったときの床 (x, y, 1) → 画像のホモグラフィ。"""
    C, T = np.asarray(C, float), np.asarray(target, float)
    zc = (T - C) / np.linalg.norm(T - C)                     # カメラの前
    xc = np.cross(zc, [0, 0, 1.0])
    xc /= np.linalg.norm(xc)                                 # 画像の右
    yc = np.cross(zc, xc)                                    # 画像の下
    R = np.vstack([xc, yc, zc])                              # 世界 → カメラ
    K = np.array([[f, 0, size[1] / 2], [0, f, size[0] / 2], [0, 0, 1]])
    t = -R @ C
    Hm = K @ np.c_[R[:, 0], R[:, 1], t]
    return Hm / Hm[2, 2]


def test_one_click_finds_court_from_single_corner():
    from smash_speed.calibration.oneclick import OneClickCalibrator

    table = landmark_table(load_config())
    Hm = _camera_h()
    frame = _court_image(Hm, seed=2, cross_seams=True)
    p = _apply_h(Hm, table["back_corner_h_n"][None, :2])[0] + np.array([1.5, -1.0])
    res = OneClickCalibrator().calibrate(frame, "back_corner_h_n", tuple(p), table, cam_xy=(-7.5, -9.0))
    err = [np.linalg.norm(np.array(res.points[n]) - _apply_h(Hm, table[n][None, :2])[0]) for n in res.points]
    assert sum(s == "snapped" for s in res.status.values()) >= 10
    assert np.median(err) < 1.5


def test_one_click_rejects_non_floor_point():
    from smash_speed.calibration.oneclick import OneClickCalibrator

    table = landmark_table(load_config())
    try:
        OneClickCalibrator().calibrate(np.zeros((100, 100, 3), np.uint8), "post_top_n", (5, 5), table)
    except ValueError as e:
        assert "床の点" in str(e)
    else:
        raise AssertionError("支柱の先端で通った")
