"""自動キャリブレーション（コートのラインの自動検出）の単体テスト。

木目調の床にバドミントンのライン（青・黄）を引き、バスケットのセンターサークル・3 ポイントライン・
斜めの線・床板の継ぎ目を混ぜた合成画像を、斜め上から見た画角で作る。他競技のラインに惑わされずに
四隅を見つけ、再投影誤差が 5 px 以内に収まることを確かめる。

``python tests/test_auto_court_calibrator.py`` でも、pytest でも実行できる。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smash_speed.calibration.auto_court_calibrator import (  # noqa: E402
    COURT_LENGTH_M, COURT_WIDTH_M, CORNERS, LINE_WIDTH_M, X_LINES, Y_LINES, AutoCourtCalibrator,
)
from smash_speed.camera import Camera, look_at  # noqa: E402

W, H = 1920, 1080
PX_PER_M = 100            # 床の模様（木目・他競技のライン）を描く解像度
TEX_X0, TEX_Y0 = -10.0, -10.0
TEX_W_M, TEX_H_M = 34.0, 27.0


def _floor_texture(rng: np.random.Generator) -> np.ndarray:
    """木目調の床（薄茶色の床板）と、バスケット・その他のラインを床の座標で描いた画像。"""
    tw, th = int(TEX_W_M * PX_PER_M), int(TEX_H_M * PX_PER_M)
    base = np.array([105, 160, 205], float)                       # 薄茶色（BGR）
    tex = np.empty((th, tw, 3), np.float32)
    plank = 0.12                                                   # 床板の幅 [m]（長手方向に並ぶ）
    ys = TEX_Y0 + (np.arange(th) + 0.5) / PX_PER_M
    idx = np.floor(ys / plank).astype(int)
    shade = rng.normal(0, 9, idx.max() - idx.min() + 1)[idx - idx.min()]
    tex[:] = base[None, None, :] + shade[:, None, None]
    seam = np.abs((ys / plank) - np.round(ys / plank)) * plank < 0.004
    tex[seam] *= 0.85                                              # 床板の継ぎ目
    xs = TEX_X0 + (np.arange(tw) + 0.5) / PX_PER_M
    tex += (6 * np.sin(xs[None, :] * 7.0 + ys[:, None] * 0.5))[..., None]   # 木目のむら
    tex += rng.normal(0, 3, tex.shape).astype(np.float32)
    tex = np.clip(tex, 0, 255).astype(np.uint8)

    def P(x: float, y: float) -> tuple[int, int]:
        return int(round((x - TEX_X0) * PX_PER_M)), int(round((y - TEX_Y0) * PX_PER_M))

    red, white, green = (40, 40, 200), (235, 235, 235), (60, 140, 40)
    t = int(0.05 * PX_PER_M)
    # バスケットボール（28 m × 15 m、コートに対して中心をずらして重ねる）
    bx0, by0 = -5.0, -4.5
    cv2.rectangle(tex, P(bx0, by0), P(bx0 + 28, by0 + 15), red, t, cv2.LINE_AA)
    cv2.line(tex, P(bx0 + 14, by0), P(bx0 + 14, by0 + 15), red, t, cv2.LINE_AA)
    cv2.circle(tex, P(bx0 + 14, by0 + 7.5), int(1.8 * PX_PER_M), red, t, cv2.LINE_AA)       # センターサークル
    for hx, sgn in ((bx0 + 1.575, 1), (bx0 + 28 - 1.575, -1)):                             # 3 ポイントライン
        a0, a1 = (-78, 78) if sgn > 0 else (102, 258)
        cv2.ellipse(tex, P(hx, by0 + 7.5), (int(6.75 * PX_PER_M),) * 2, 0, a0, a1, red, t, cv2.LINE_AA)
        cv2.circle(tex, P(hx + sgn * 4.0, by0 + 7.5), int(1.8 * PX_PER_M), white, t, cv2.LINE_AA)
    # 斜めの線（ほかの消失点に向かう線）
    cv2.line(tex, P(-3, 9), P(14, -2), white, t, cv2.LINE_AA)
    cv2.line(tex, P(2, 12), P(20, 1), green, t, cv2.LINE_AA)
    return tex


def _texture_to_world() -> np.ndarray:
    return np.array([[1 / PX_PER_M, 0, TEX_X0 + 0.5 / PX_PER_M], [0, 1 / PX_PER_M, TEX_Y0 + 0.5 / PX_PER_M],
                     [0, 0, 1]])


def _floor_h(cam: Camera) -> np.ndarray:
    """床 (X, Y, 1) → 画像のホモグラフィ（歪みなし）。"""
    R, t = cam.R, cam.tvec
    return cam.K @ np.c_[R[:, 0], R[:, 1], t]


def render_scene(eye, target, f: float, line_color, seed: int = 0) -> tuple[np.ndarray, Camera]:
    """合成画像と、正解のカメラ（座標はコートの角 P1 が原点）。"""
    rng = np.random.default_rng(seed)
    cam = look_at(np.array(eye, float), np.array(target, float), f, W, H)
    Hw = _floor_h(cam)
    tex = _floor_texture(rng)
    img = np.full((H, W, 3), (150, 150, 145), np.uint8)               # 壁
    floor = cv2.warpPerspective(tex, Hw @ _texture_to_world(), (W, H), flags=cv2.INTER_AREA,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask = cv2.warpPerspective(np.full(tex.shape[:2], 255, np.uint8), Hw @ _texture_to_world(), (W, H))
    img[mask > 0] = floor[mask > 0]
    # バドミントンのライン（幅 40 mm）は画像の上で多角形として正確に描く
    lw = LINE_WIDTH_M
    rects = []
    for _, e, s in X_LINES:
        x0, x1 = sorted((e, e + s * lw))
        rects.append((x0, 0.0, x1, COURT_WIDTH_M))
    for name, e, s in Y_LINES:
        y0, y1 = sorted((e, e + s * lw)) if s else (e - lw / 2, e + lw / 2)
        spans = ((0.0, 4.72), (8.68, COURT_LENGTH_M)) if name == "center" else ((0.0, COURT_LENGTH_M),)
        for a, b in spans:
            rects.append((a, y0, b, y1))
    over = img.copy()
    for x0, y0, x1, y1 in rects:
        # 長い線は分割して描く（透視投影で台形がゆがまないように）
        n = max(1, int(max(x1 - x0, y1 - y0) / 0.5))
        for k in range(n):
            if x1 - x0 > y1 - y0:
                a, b = x0 + (x1 - x0) * k / n, x0 + (x1 - x0) * (k + 1) / n
                quad = [(a, y0), (b, y0), (b, y1), (a, y1)]
            else:
                a, b = y0 + (y1 - y0) * k / n, y0 + (y1 - y0) * (k + 1) / n
                quad = [(x0, a), (x1, a), (x1, b), (x0, b)]
            uv = cam.project(np.c_[np.array(quad), np.zeros(4)])
            if not np.isfinite(uv).all():
                continue
            cv2.fillConvexPoly(over, np.round(uv * 16).astype(np.int32), line_color, cv2.LINE_AA, shift=4)
    img = cv2.addWeighted(over, 0.9, img, 0.1, 0)
    # 選手のかわりに、ラインの一部を隠す人影
    for X, Y in ((2.0, 1.0), (10.5, 4.6)):
        uv = cam.project(np.array([[X, Y, 0.0], [X, Y, 1.7]]))
        if np.isfinite(uv).all():
            wpx = max(8, int(abs(uv[0, 1] - uv[1, 1]) * 0.28))
            cv2.rectangle(img, (int(uv[1, 0]) - wpx // 2, int(uv[1, 1])), (int(uv[0, 0]) + wpx // 2, int(uv[0, 1])),
                          (40, 30, 30), -1)
    img = cv2.GaussianBlur(img, (0, 0), 0.6)
    img = np.clip(img + rng.normal(0, 2.0, img.shape), 0, 255).astype(np.uint8)
    return img, cam


def _check(img: np.ndarray, cam: Camera, hint=None, tol_px: float = 5.0):
    cal = AutoCourtCalibrator()
    res = cal.calibrate(img, line_color_hint=hint)
    assert res.success, res.message
    assert res.confidence_score >= 0.6, (res.confidence_score, res.message)
    assert res.reprojection_error_rms < 5.0, res.reprojection_error_rms
    truth = cam.project(np.array([[X, Y, 0.0] for _, X, Y, _ in CORNERS]))
    err = np.linalg.norm(res.corners_2d - truth, axis=1)
    visible = (truth[:, 0] > 0) & (truth[:, 0] < W) & (truth[:, 1] > 0) & (truth[:, 1] < H)
    assert visible.sum() >= 2
    assert err[visible].max() < tol_px, (err, res.message)
    # 測った交点も正しい位置にある（解析用のランドマーク名で、ライン外側の縁の角）
    from smash_speed.calibration.auto_court_calibrator import to_analysis_xyz  # noqa: F401
    for name, uv in res.keypoints_2d.items():
        X = _landmark_world(name)
        p = cam.project(np.array([[*X, 0.0]]))[0]
        assert np.linalg.norm(np.array(uv) - p) < tol_px, (name, uv, p)
    # 推定したカメラの位置が正解に近い
    C = res.camera_center
    assert np.linalg.norm(C - cam.center) < 0.05 * np.linalg.norm(cam.center - [6.7, 3.05, 0]), (C, cam.center)
    return res


def _landmark_world(name: str) -> tuple[float, float]:
    from smash_speed.config import load_config
    lm = {d["name"]: d["xyz"] for d in load_config()["landmarks"]}
    x, y, _ = lm[name]
    return x + 6.70, y + 3.05


def test_blue_lines_oblique_view_with_basketball_lines():
    img, cam = render_scene(eye=(-3.5, -3.0, 3.2), target=(7.0, 3.6, 0.0), f=1050, line_color=(170, 90, 20))
    res = _check(img, cam)
    assert set(res.keypoints_2d) >= {"back_corner_h_n", "back_corner_o_n", "back_corner_o_p", "back_corner_h_p"}


def test_yellow_lines_side_view_with_hint():
    img, cam = render_scene(eye=(8.0, -5.0, 2.4), target=(5.0, 3.4, 0.0), f=950, line_color=(40, 210, 230), seed=1)
    _check(img, cam, hint="yellow")
    _check(img, cam)                                     # 色の指定がなくても見つかる


def test_rotate_180_swaps_hitter_side():
    img, cam = render_scene(eye=(-3.5, -3.0, 3.2), target=(7.0, 3.6, 0.0), f=1050, line_color=(170, 90, 20))
    cal = AutoCourtCalibrator(debug=False)
    res = cal.calibrate(img)
    rot = cal.rotate_180(res)
    assert rot.success
    np.testing.assert_allclose(rot.corners_2d[0], res.corners_2d[2], atol=1.0)
    assert rot.camera_center[0] > 6.7 > res.camera_center[0]
    assert abs(rot.reprojection_error_rms - res.reprojection_error_rms) < 0.5


def test_no_court_is_not_confident():
    rng = np.random.default_rng(3)
    img = np.clip(rng.normal(120, 30, (H, W, 3)), 0, 255).astype(np.uint8)
    for _ in range(25):                                  # でたらめな線
        p = rng.integers(0, [W, H], 2)
        q = rng.integers(0, [W, H], 2)
        cv2.line(img, tuple(int(v) for v in p), tuple(int(v) for v in q), (230, 230, 230), 3)
    res = AutoCourtCalibrator().calibrate(img)
    assert not res.success or res.confidence_score < 0.6


if __name__ == "__main__":
    import time
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        t0 = time.perf_counter()
        try:
            fn()
            print(f"PASS  {fn.__name__}  ({time.perf_counter() - t0:.1f} s)")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
