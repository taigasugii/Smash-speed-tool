"""Phase 2：キャリブレーション（推定・保存・GUI の状態遷移）のテスト。"""

import numpy as np
import pytest

from smash_speed import synth
from smash_speed.calibration import calibrate, dlt, landmark_table, load_calibration, save_calibration
from smash_speed.calibration.gui import court_diagram
from smash_speed.config import load_config
from smash_speed.ui import KEY_RIGHT, ImageView, PointPicker


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def truth(cfg):
    return synth.default_camera(cfg)


def _visible_landmarks(cfg, cam, rng, noise=0.7, floor_only=False):
    table = landmark_table(cfg)
    names, world, image = [], [], []
    for n, X in table.items():
        if floor_only and X[2] > 0:
            continue
        uv = cam.project(X[None])[0]
        if np.isfinite(uv).all() and 0 <= uv[0] < cam.width and 0 <= uv[1] < cam.height:
            names.append(n)
            world.append(X)
            image.append(uv + rng.normal(0, noise, 2))
    return names, np.array(world), np.array(image)


def test_calibration_recovers_camera(cfg, truth):
    rng = np.random.default_rng(0)
    names, world, image = _visible_landmarks(cfg, truth, rng)
    res = calibrate(names, world, image, truth.width, truth.height)
    assert res.rmse_px < 1.5
    assert abs(res.camera.f - truth.f) / truth.f < 0.03
    assert np.linalg.norm(res.camera.center - truth.center) < 0.15  # [m]


def test_calibration_with_floor_points_only(cfg, truth):
    rng = np.random.default_rng(1)
    names, world, image = _visible_landmarks(cfg, truth, rng, floor_only=True)
    res = calibrate(names, world, image, truth.width, truth.height)
    assert any("同一平面" in w for w in res.warnings)
    assert abs(res.camera.f - truth.f) / truth.f < 0.05


def test_calibration_flags_bad_click(cfg, truth):
    rng = np.random.default_rng(2)
    names, world, image = _visible_landmarks(cfg, truth, rng)
    image[3] += [40.0, 0.0]
    res = calibrate(names, world, image, truth.width, truth.height, warn_px=5.0)
    assert res.errors_px[3] > 5.0
    assert any(names[3] in w for w in res.warnings)


def test_dlt_reproduces_projection(cfg, truth):
    table = landmark_table(cfg)
    world = np.array(list(table.values()))
    image = truth.project(world)
    P = dlt(world, image)
    h = np.c_[world, np.ones(len(world))] @ P.T
    np.testing.assert_allclose(h[:, :2] / h[:, 2:], image, atol=1e-6)


def test_save_and_load_roundtrip(tmp_path, cfg, truth):
    rng = np.random.default_rng(3)
    names, world, image = _visible_landmarks(cfg, truth, rng)
    res = calibrate(names, world, image, truth.width, truth.height)
    save_calibration(res, tmp_path / "calib.json", video="x.mp4", frame_time_s=1.0)
    cam, data = load_calibration(tmp_path / "calib.json")
    np.testing.assert_allclose(cam.project(world), res.camera.project(world))
    assert data["points"][0]["name"] == names[0]


def test_distortion_k1_estimation(cfg, truth):
    from smash_speed.camera import Camera
    cam = Camera(truth.f, truth.cx, truth.cy, truth.width, truth.height, truth.rvec, truth.tvec, k1=-0.08)
    rng = np.random.default_rng(4)
    names, world, image = _visible_landmarks(cfg, cam, rng, noise=0.3)
    res = calibrate(names, world, image, cam.width, cam.height, estimate_k1=True)
    assert res.rmse_px < 1.0
    assert abs(res.camera.k1 - (-0.08)) < 0.03


def test_ray_roundtrip_with_distortion(truth):
    from smash_speed.camera import Camera
    cam = Camera(truth.f, truth.cx, truth.cy, truth.width, truth.height, truth.rvec, truth.tvec, k1=-0.1, k2=0.02)
    X = np.array([[-3.0, 1.0, 2.0], [2.0, -1.0, 0.5]])
    uv = cam.project(X)
    c, d = cam.rays(uv)
    # 視線上に元の点があること
    for x, di in zip(X, d):
        v = x - c
        assert np.linalg.norm(np.cross(v / np.linalg.norm(v), di)) < 1e-6


def test_image_view_roundtrip():
    v = ImageView(3840, 2160)
    v.zoom_at(400, 300, 4.0)
    p = v.to_image(123, 456)
    np.testing.assert_allclose(v.to_display(p), [123, 456])


def test_point_picker_state_machine():
    img = np.zeros((100, 200, 3), np.uint8)
    p = PointPicker(img, ["a", "b", "c"])
    p.place(np.array([10.0, 20.0]))
    p.skip()
    assert p.current == "c"
    p.place(np.array([30.0, 40.0]))
    p.step_px = 0.25
    p.handle_key(next(iter(KEY_RIGHT)))
    np.testing.assert_allclose(p.points["c"], [30.25, 40.0])
    p.undo()  # 微調整を戻す
    np.testing.assert_allclose(p.points["c"], [30.0, 40.0])
    p.undo()  # c の配置を戻す
    assert "c" not in p.points and p.current == "c"
    p.undo()  # b のスキップを戻す
    assert p.current == "b"
    assert set(p.placed()) == {"a"}
    assert p.handle_key(13) == "done"
    assert p.render().shape[2] == 3


def test_court_diagram_renders(cfg):
    img = court_diagram(cfg, "post_top_n", {"back_corner_h_n": np.zeros(2), "back_corner_h_p": None})
    assert img.ndim == 3 and img.shape[1] == 260
