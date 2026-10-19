"""打つ人の位置・運動方程式を手がかりにしたシャトルの探し方（detection/guided.py）のテスト。"""

import json

import cv2
import numpy as np
import pytest

from smash_speed import synth
from smash_speed.calibration.io import landmark_table, save_calibration
from smash_speed.calibration.solve import calibrate
from smash_speed.camera import look_at
from smash_speed.config import load_config
from smash_speed.detection import guided
from smash_speed.geometry import FlightPlane
from smash_speed.physics import PhysicsParams
from smash_speed.pipeline import AnalyzeOptions, analyze


def test_corridor_polygon_contains_flight():
    cfg = load_config()
    cam = synth.default_camera(cfg)
    plane = FlightPlane.from_points([-4.8, -0.8], [3.0, 0.6])
    length = float(np.linalg.norm(np.array([3.0, 0.6]) - [-4.8, -0.8]))
    poly = guided.corridor_polygon(cam, plane, length)
    assert poly is not None
    inside = [plane.to_world(np.array([s]), np.array([z]))[0] for s, z in ((0.3, 2.5), (length / 2, 1.5), (length, 0.0))]
    for X in inside:
        uv = cam.project(X[None])[0].astype(np.float32)
        assert cv2.pointPolygonTest(poly.astype(np.float32), tuple(map(float, uv)), False) >= 0
    # カメラ側のコートの外の床（観客・隣のコートの人がいる所）は範囲の外
    uv = cam.project(np.array([[-1.0, -6.5, 0.0]]))[0]
    assert cv2.pointPolygonTest(poly.astype(np.float32), tuple(map(float, uv)), False) < 0
    assert 0.0 < guided.polygon_area_ratio(poly, cam.width, cam.height) < 0.9


def test_corridor_polygon_none_when_camera_inside():
    cam = look_at(np.array([-3.0, 0.0, 1.5]), np.array([3.0, 0.0, 1.0]), 1000.0, 1280, 720)
    plane = FlightPlane.from_points([-4.8, 0.0], [3.0, 0.0])
    assert guided.corridor_polygon(cam, plane, 7.8) is None


def test_shuttle_like_keeps_dots_and_streaks_not_outlines():
    d = np.zeros((200, 300), np.float32)
    cv2.circle(d, (40, 40), 3, 80, -1)                  # 小さな点（シャトル）
    cv2.line(d, (100, 30), (160, 60), 80, 3)            # まっすぐな残像
    cv2.ellipse(d, (220, 120), (25, 60), 0, 0, 360, 80, 2)  # 人の輪郭（輪）
    w = guided._shuttle_like(d, 10.0, 3.0)
    assert w[40, 40] == 1 and w[45, 130] == 1
    assert w[120, 195] == 0


def _streak_frames(bg_split: bool = True):
    """背景の左半分が暗く右半分が明るい画像の上を、長い残像（x 100〜400、y 200）が通る 3 コマ。"""
    bg = np.full((400, 600), 150, np.uint8)
    if bg_split:
        bg[:, :250] = 60                                  # 暗い所では残像との差が大きい
    prev, cur, nxt = bg.copy(), bg.copy(), bg.copy()
    cv2.line(cur, (100, 200), (400, 200), 200, 3)
    return prev, cur, nxt


def test_redetect_joins_streak_fragments_along_motion():
    prev, cur, nxt = _streak_frames(bg_split=False)
    cur[:, 230:262] = 150                                 # 残像の途中が雑音で切れて 2 つの塊になる
    predict = lambda t: np.tile([[255.0, 200.0]], (len(np.atleast_1d(t)), 1)) + \
        np.c_[(np.atleast_1d(t) - 1.0) * 9000.0, np.zeros(len(np.atleast_1d(t)))]  # noqa: E731
    frames = [(0, 1.0 - 1 / 30, prev), (1, 1.0, cur), (2, 1.0 + 1 / 30, nxt)]
    rd = guided.redetect_along(frames, predict, unit_px=3.0)
    assert len(rd) == 1 and abs(rd.x_px.iloc[0] - 250) < 6


def test_evidence_map_keeps_faint_long_streak_next_to_strong_outline():
    bg = np.full((300, 400), 120, np.uint8)
    frames = [(i, i / 30, bg.copy()) for i in range(3)]
    cv2.line(frames[1][2], (40, 100), (180, 80), 135, 2)   # 薄く長い残像（差分 15）
    cv2.ellipse(frames[1][2], (300, 180), (30, 70), 0, 0, 360, 250, 3)  # 人の輪郭（差分 130）
    (_, _, e), = guided.evidence_maps(frames)
    assert e[90, 110] >= 0.5 * 255


@pytest.fixture(scope="module")
def behind_session(tmp_path_factory):
    """打つ人の斜め後ろから撮った、選手・歩く人の写った合成動画（従来の選び方では人の動きを拾ってしまう）。"""
    cfg = load_config()
    phys = PhysicsParams.from_config(cfg)
    cam = look_at(np.array([-10.5, -2.5, 2.4]), np.array([1.5, 0.3, 1.2]), 870.0, 1280, 720)
    shot = synth.make_shot(80.0, cfg, phys, elevation_deg=-10.0)
    d = tmp_path_factory.mktemp("behind")
    synth.render_cluttered_video(shot, cam, cfg, phys, d / "video.mp4", fps=60.0, audio_path=d / "audio.wav",
                                 walkers=[((-5.0, 4.6), (1.4, 0.0)), ((8.5, -2.0), (0.0, 1.6)),
                                          ((9.0, 3.5), (-0.3, -1.2))])
    rng = np.random.default_rng(0)
    names, world, image = [], [], []
    for n, X in landmark_table(cfg).items():
        uv = cam.project(X[None])[0]
        if np.isfinite(uv).all() and 0 <= uv[0] < cam.width and 0 <= uv[1] < cam.height:
            names.append(n)
            world.append(X)
            image.append(uv + rng.normal(0, 0.5, 2))
    save_calibration(calibrate(names, np.array(world), np.array(image), cam.width, cam.height), d / "calib.json")
    hitter = shot.plane.origin[:2] - cfg["events"]["contact_offset_m"] * shot.plane.direction[:2]
    return cfg, d, shot, hitter


def test_analyze_behind_view_with_clutter(behind_session):
    cfg, d, shot, hitter = behind_session
    out = d / "out"
    o = AnalyzeOptions(video=d / "video.mp4", calib=d / "calib.json", out_dir=out, audio_path=d / "audio.wav",
                       hitter_xy=tuple(hitter + [0.1, -0.1]), landing_xy=tuple(shot.landing_xy + [0.15, 0.1]),
                       overlay=False, start_s=0.0)
    row = analyze(o, cfg)
    assert row["speed_initial_mps"] == pytest.approx(80.0, rel=0.1)
    fr = json.loads((out / "fit_result.json").read_text(encoding="utf-8"))
    assert fr["n_obs"] >= 15
    assert any("打球音の時刻に打ち出した軌道" in line for line in fr["log"])
    assert any("自動で絞りました" in line for line in fr["log"])
    assert (out / "observations.csv").exists() and (out / "candidates.csv").exists()
