"""Phase 3 / 8：平面拘束と三角測量のテスト。"""

import numpy as np
import pytest

from smash_speed import synth
from smash_speed.camera import look_at
from smash_speed.config import load_config
from smash_speed.geometry import FlightPlane, floor_point, intersect_plane, triangulate
from smash_speed.physics import FitOptions, Observations, PhysicsParams, fit_trajectory


@pytest.fixture(scope="module")
def setup():
    cfg = load_config()
    phys = PhysicsParams.from_config(cfg)
    cam = synth.default_camera(cfg)
    shot = synth.make_shot(75.0, cfg, phys)
    return cfg, phys, cam, shot


def test_plane_intersection_recovers_trajectory(setup):
    cfg, phys, cam, shot = setup
    df = synth.observe(shot, cam, 120, phys, np.random.default_rng(0), noise_px=0.0, dropout=0.0)
    X = intersect_plane(cam, df[["x_px", "y_px"]].values, shot.plane)
    np.testing.assert_allclose(X, df[["x_m", "y_m", "z_m"]].values, atol=1e-6)


def test_floor_point_and_plane_from_points(setup):
    _, _, cam, shot = setup
    a, b = np.array([-4.5, -0.8, 0.0]), np.array([3.0, 0.6, 0.0])
    uv = cam.project(np.vstack([a, b]))
    P = floor_point(cam, uv)
    np.testing.assert_allclose(P, np.vstack([a, b]), atol=1e-6)
    plane = FlightPlane.from_points(P[0], P[1])
    np.testing.assert_allclose(plane.direction, shot.plane.direction, atol=1e-6)


def test_plane_misspecification_effect(setup):
    """飛行平面の向きを 3° ずらしたときの初速の誤差（README の仮定の節の根拠）。"""
    cfg, phys, cam, shot = setup
    df = synth.observe(shot, cam, 60, phys, np.random.default_rng(1), noise_px=1.0, dropout=0.0)
    obs = Observations(cam, df.timestamp.values, df[["x_px", "y_px"]].values)
    ang = np.deg2rad(3.0)
    d = shot.plane.direction
    rot = np.array([d[0] * np.cos(ang) - d[1] * np.sin(ang), d[0] * np.sin(ang) + d[1] * np.cos(ang), 0])
    wrong = FlightPlane(shot.plane.origin, rot)
    fixed = fit_trajectory(obs, 0.0, "plane", wrong, phys, FitOptions(ci_method="none"))
    assert abs(fixed.speed0 - 75.0) / 75.0 > 0.10  # 平面を固定すると 3° のずれで 10% 超の誤差
    refined = fit_trajectory(obs, 0.0, "plane", wrong, phys,
                             FitOptions(ci_method="none", refine_plane=True))
    assert abs(refined.speed0 - 75.0) / 75.0 < 0.03
    assert refined.rms_px < fixed.rms_px


def test_triangulation_and_free_fit(setup):
    cfg, phys, cam1, shot = setup
    cam2 = look_at(np.array([-9.0, 4.0, 3.5]), np.array([0.0, 0.0, 1.0]), 1400.0, 1920, 1080)
    rng = np.random.default_rng(2)
    d1 = synth.observe(shot, cam1, 120, phys, rng, noise_px=0.0, dropout=0.0, phase=0.3)
    d2 = synth.observe(shot, cam2, 120, phys, rng, noise_px=0.0, dropout=0.0, phase=0.3)
    X = triangulate([cam1, cam2], [d1[["x_px", "y_px"]].values, d2[["x_px", "y_px"]].values])
    np.testing.assert_allclose(X, d1[["x_m", "y_m", "z_m"]].values, atol=1e-5)

    d1 = synth.observe(shot, cam1, 60, phys, rng, noise_px=1.0, dropout=0.1)
    d2 = synth.observe(shot, cam2, 60, phys, rng, noise_px=1.0, dropout=0.1)
    obs = [Observations(cam1, d.timestamp.values, d[["x_px", "y_px"]].values)
           for d in (d1[d1.visible == 1],)]
    obs.append(Observations(cam2, d2[d2.visible == 1].timestamp.values, d2[d2.visible == 1][["x_px", "y_px"]].values))
    fit = fit_trajectory(obs, 0.0, "free", phys=phys)
    assert abs(fit.speed0 - 75.0) / 75.0 < 0.03
