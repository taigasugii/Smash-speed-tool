"""半自動キャリブレーション（おおよその 4 点 → コートのラインに吸着）のテスト。"""

import cv2
import numpy as np
import pytest

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


def test_propose_finds_court_without_clicks():
    from smash_speed.calibration.propose import propose_calibrations

    table = landmark_table(load_config())
    Hm = _side_view_h()
    frame = _court_image(Hm)
    props = propose_calibrations(frame, table, n=3)
    assert props and props[0].n_snapped >= 15
    err = [np.linalg.norm(np.array(props[0].result.points[n]) - _apply_h(Hm, table[n][None, :2])[0])
           for n in props[0].result.points]
    assert np.median(err) < 1.5
    assert [p.rank_score for p in props] == sorted((p.rank_score for p in props), reverse=True)


def test_propose_puts_hitter_on_the_camera_end():
    from smash_speed.calibration.propose import _orient

    Hm = _camera_h()                                         # カメラは x < 0 の端の後ろ
    flipped = Hm @ np.diag([-1.0, -1.0, 1.0])
    assert np.allclose(_orient(flipped, (1080, 1920), None), Hm)
    assert np.allclose(_orient(Hm, (1080, 1920), None), Hm)
    # 手順 ① でカメラを x > 0 の側に置いたなら入れ替える
    assert np.allclose(_orient(Hm, (1080, 1920), (8.0, -8.0)), flipped)


def test_propose_returns_nothing_for_blank_frame():
    from smash_speed.calibration.propose import propose_calibrations

    assert propose_calibrations(np.zeros((360, 640, 3), np.uint8), landmark_table(load_config())) == []


def _median_err(Hm: np.ndarray, Ht: np.ndarray, table) -> float:
    """床のランドマークの、2 つのホモグラフィでの画像の位置のずれの中央値 [px]（180° 回したコートも同じとみなす）。"""
    W = np.array([v[:2] for v in table.values() if abs(v[2]) < 1e-6])
    ref = _apply_h(Ht, W)
    return min(float(np.median(np.linalg.norm(_apply_h(H, W) - ref, axis=1)))
               for H in (Hm, Hm @ np.diag([-1.0, -1.0, 1.0])))


def test_vanishing_candidates_find_low_corner_camera():
    from smash_speed.calibration.propose import vanishing_candidates

    table = landmark_table(load_config())
    # 低い位置（0.95 m）・手前の角から撮った画像。手順 ① のカメラの位置は 1.5 m ずれている
    Hm = _camera_h(C=(-9.5, -5.2, 0.95), target=(-1.5, 0.5, 0.0), f=1000.0)
    frame = _court_image(Hm, seed=4, cross_seams=True)
    cands = vanishing_candidates(frame, (-8.4, -4.2))
    assert cands
    assert min(_median_err(H, Hm, table) for H, _ in cands[:3]) < 5.0


def test_propose_uses_camera_position_for_vanishing_candidates():
    from smash_speed.calibration.propose import propose_calibrations

    table = landmark_table(load_config())
    Hm = _camera_h(C=(-9.5, -5.2, 0.95), target=(-1.5, 0.5, 0.0), f=1000.0)
    frame = _court_image(Hm, seed=4, cross_seams=True)
    steps = []
    props = propose_calibrations(frame, table, n=3, cam_xy=(-8.4, -4.2), progress=lambda f, m: steps.append((f, m)))
    assert props and _median_err(props[0].result.homography, Hm, table) < 2.0
    # 進み具合は 0 → 1 に増えていき、流れすぎない（GUI の処理の欄に出す）
    fr = [f for f, _ in steps]
    assert fr[0] == 0.0 and fr[-1] == 1.0 and fr == sorted(fr) and 5 <= len(steps) <= 60
    assert all(isinstance(m, str) and m for _, m in steps)


def _distorted(img: np.ndarray, kappa: float) -> np.ndarray:
    """ゆがみのない画像を、``snap.undistort_points`` の κ でゆがめた画像（広角のレンズの樽型のゆがみ）。"""
    from smash_speed.calibration.snap import undistort_points

    h, w = img.shape[:2]
    g = np.stack(np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float)), -1).reshape(-1, 2)
    src = undistort_points(g, kappa, np.array([w / 2, h / 2]), max(h, w)).reshape(h, w, 2).astype(np.float32)
    return cv2.remap(img, src[..., 0], src[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def test_snap_estimates_lens_distortion():
    """ゆがんだ画像では、ラインに吸い付けるときにゆがみ κ も求め、ランドマークがゆがみなしより合う（ROADMAP v2 の J3）。"""
    from smash_speed.calibration.snap import distort_points

    table = landmark_table(load_config())
    Hm = _side_view_h()
    kappa = 0.25
    frame = _distorted(_court_image(Hm), kappa)
    h, w = frame.shape[:2]
    c, s = np.array([w / 2, h / 2]), max(h, w)
    names = ["back_corner_h_n", "back_corner_h_p", "short_service_side_h_n", "short_service_side_h_p"]
    clicks = {n: tuple(distort_points(_apply_h(Hm, table[n][None, :2]), kappa, c, s)[0]) for n in names}
    errs = {}
    for est in (False, True):
        res = CourtSnapper(estimate_kappa=est).snap(frame, clicks, table)
        snapped = [n for n, st in res.status.items() if st == "snapped"]
        truth = {n: distort_points(_apply_h(Hm, table[n][None, :2]), kappa, c, s)[0] for n in snapped}
        errs[est] = float(np.median([np.linalg.norm(np.array(res.points[n]) - truth[n]) for n in snapped]))
        if est:
            assert abs(res.kappa - kappa) < 0.05
            np.testing.assert_allclose(res.project(table[snapped[0]][None, :2])[0], res.points[snapped[0]], atol=0.1)
    assert errs[True] < 1.0
    assert errs[True] < errs[False]


def test_distortion_round_trip():
    """``distort_points`` は ``undistort_points`` の逆。"""
    from smash_speed.calibration.snap import distort_points, undistort_points

    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1920, (50, 2))
    c = np.array([960.0, 540.0])
    for kappa in (-0.2, 0.0, 0.3):
        np.testing.assert_allclose(undistort_points(distort_points(p, kappa, c, 1920.0), kappa, c, 1920.0), p, atol=1e-6)


def _floor_render(C, target, f, size=(1080, 1920), seed=0, ss=2) -> tuple[np.ndarray, np.ndarray]:
    """カメラから床へ光線を飛ばして描いた画像（地平線より上は壁、カメラの後ろの床は描かない）と、床のホモグラフィ。

    ``_court_image`` はホモグラフィで描くので、カメラの後ろの床も地平線の上に写ってしまう（低い・横のカメラで使えない）。
    """
    C, T = np.asarray(C, float), np.asarray(target, float)
    zc = (T - C) / np.linalg.norm(T - C)
    xc = np.cross(zc, [0, 0, 1.0]); xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    R = np.vstack([xc, yc, zc])
    h, w = size
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    Hm = K @ np.c_[R[:, 0], R[:, 1], -R @ C]
    U, V = np.meshgrid((np.arange(w * ss) + 0.5) / ss, (np.arange(h * ss) + 0.5) / ss)
    d = np.stack([(U - w / 2) / f, (V - h / 2) / f, np.ones_like(U)], -1) @ R
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = -C[2] / d[..., 2]
    floor = (d[..., 2] < 0) & (lam > 0) & (lam < 60)
    X, Y = C[0] + lam * d[..., 0], C[1] + lam * d[..., 1]
    line = np.zeros(U.shape, bool)
    for ax, c, spans in LINES.values():
        a, b = (X, Y) if ax == "x" else (Y, X)
        line |= (np.abs(a - c) <= LINE_WIDTH_M / 2) & np.any([(b >= lo) & (b <= hi) for lo, hi in spans], axis=0)
    seam = (np.abs(((X + 0.065) % 0.13) - 0.065) < 0.004) | (np.abs(((Y + 0.45) % 0.9) - 0.45) < 0.004)
    img = np.empty(U.shape + (3,), np.float32)
    img[:] = (150, 190, 215)
    img[seam & floor] *= 0.65
    img[line & floor] = (60, 110, 70)
    img[~floor] = (170, 170, 165)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA) + np.random.default_rng(seed).normal(0, 3, (h, w, 3))
    return np.clip(img, 0, 255).astype(np.uint8), Hm / Hm[2, 2]


def test_floor_depths_and_tilt():
    from smash_speed.calibration.propose import _CORNERS, _tilt_deg, floor_depths

    # ネットの近くの横（コートの外、1.3 m）から斜めに撮るカメラでは、手前の角がカメラの後ろにある
    _, Hm = _floor_render((-1.0, -4.5, 1.3), (1.5, 0.5, 0.0), 700.0, size=(54, 96), ss=1)
    K = np.array([[700.0, 0, 48], [0, 700.0, 27], [0, 0, 1]])
    d = floor_depths(Hm, K, _CORNERS)
    assert d[0] < 0 < min(d[1:])
    # 符号はホモグラフィの倍率によらない
    assert np.array_equal(np.sign(floor_depths(-3.0 * Hm, K, _CORNERS)), np.sign(d))
    # 世界の上向きが画像の上：傾き 0°、横倒し：90°
    assert _tilt_deg(np.c_[[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]]) == pytest.approx(0.0)
    assert _tilt_deg(np.eye(3)) == pytest.approx(90.0)


def test_propose_finds_court_from_net_side_oblique_camera():
    from smash_speed.calibration.propose import propose_calibrations

    table = landmark_table(load_config())
    # G1 の練習の動画の撮り方（ネットの近くの横・1.3 m から斜め上へ）。手前の角はカメラの後ろ。手順 ① は 1.5 m ずれている
    frame, Hm = _floor_render((-1.0, -4.5, 1.3), (1.5, 0.5, 2.3), 700.0)
    props = propose_calibrations(frame, table, n=3, cam_xy=(0.5, -6.0))
    assert props
    W = np.array([v[:2] for v in table.values() if abs(v[2]) < 1e-6])
    ref = _apply_h(Hm, W)
    q = np.c_[W, np.ones(len(W))] @ Hm.T
    m = (q[:, 2] > 0) & (ref[:, 0] >= 0) & (ref[:, 0] < 1920) & (ref[:, 1] >= 0) & (ref[:, 1] < 1080)
    err = min(float(np.median(np.linalg.norm(_apply_h(H, W[m]) - ref[m], axis=1)))
              for H in (props[0].result.homography, props[0].result.homography @ np.diag([-1.0, -1, 1])))
    assert err < 3.0


def test_vanishing_candidates_without_camera_position():
    from smash_speed.calibration.propose import vanishing_candidates

    table = landmark_table(load_config())
    # 練習の動画の撮り方（手前の半面の横、外側の線から 0.9 m 外・高さ 1.5 m からネットの向こうへ）。手順 ① の位置はない
    frame, Hm = _floor_render((-6.1, 3.9, 1.5), (4.0, -1.0, 1.2), 900.0, size=(540, 960))
    cands = vanishing_candidates(frame, None)
    assert cands
    W = np.array([v[:2] for v in table.values() if abs(v[2]) < 1e-6])
    ref = _apply_h(Hm, W)
    q = np.c_[W, np.ones(len(W))] @ Hm.T
    m = (q[:, 2] > 0) & (ref[:, 0] >= 0) & (ref[:, 0] < 960) & (ref[:, 1] >= 0) & (ref[:, 1] < 540)
    errs = [min(float(np.median(np.linalg.norm(_apply_h(H @ s, W[m]) - ref[m], axis=1)))
                for s in (np.eye(3), np.diag([-1.0, -1, 1]))) for H, _ in cands]
    assert min(errs) < 5.0
