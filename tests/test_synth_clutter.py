"""雑音・人・隣のコート・画面の外を足した合成ラリー（synth_clutter）のテスト。"""

import numpy as np
import pytest

from smash_speed.config import load_config
from smash_speed.physics import PhysicsParams
from smash_speed.synth_clutter import (PARTS, ClutterOptions, clutter_benchmark, clutter_camera, hits_benchmark,
                                       make_scene, observe_cluttered)

CFG = load_config()
PHYS = PhysicsParams.from_config(CFG)


@pytest.fixture(scope="module")
def scene():
    return make_scene(np.random.default_rng(0), PHYS, n_rallies=3, opts=ClutterOptions.clean())


def test_clean_observation_is_only_the_shuttle(scene):
    cam = clutter_camera("side", CFG)
    det, onsets, truth = observe_cluttered(scene, cam, 30.0, np.random.default_rng(1), ClutterOptions.clean())
    vis = det[det.visible > 0]
    assert set(vis.source) == {"shuttle"}
    # 1 コマ 1 行で、すべてのコマがある
    assert det.frame.is_unique and len(det) == len(truth)
    # 軌道は打球ごとに分かれる（1 つの軌道に 2 本のショットが混ざらない）
    shot_of = truth.set_index("frame").shot_index
    assert (vis.groupby("track_id").frame.agg(lambda f: shot_of[f].nunique()) == 1).all()
    # 打球音は打球ごとに 1 つ（遅れつき・強さ 30）
    hits = scene.hits()
    assert len(onsets) == len(hits)
    for (t, p0, *_), (ta, s) in zip(hits, onsets):
        assert ta == pytest.approx(t + np.linalg.norm(cam.center - p0) / 343.0) and s == 30.0


def test_cluttered_observation_has_people_neighbor_and_gaps(scene):
    cam = clutter_camera("side", CFG)
    o = ClutterOptions()
    scene_n = make_scene(np.random.default_rng(0), PHYS, n_rallies=3, opts=o)
    assert len(scene_n.neighbor) >= 1
    # 隣のコートのラリーを足しても、元のラリーは同じ
    assert [h[0] for h in scene_n.hits()] == [h[0] for h in scene.hits()]
    det, onsets, truth = observe_cluttered(scene_n, cam, 30.0, np.random.default_rng(1), o)
    vis = det[det.visible > 0]
    assert {"shuttle", "neighbor", "racket", "body"} <= set(vis.source)
    # 同じコマに何点もある（重なった軌道を含む検出）
    assert vis.frame.duplicated().any()
    # 打った直後は体・ラケットに隠れてシャトルが見えない
    sh = vis[vis.source == "shuttle"]
    for t, *_ in scene_n.hits():
        assert not ((sh.timestamp > t) & (sh.timestamp < t + 0.005)).any()
    # 打球音のほかに、隣のコートの打球音・足音などが入る
    assert len(onsets) > len(scene_n.hits())


def test_racket_passes_through_the_hit_point(scene):
    t, p0, *_ = scene.hits()[2]
    pts = scene.racket(t, p0, np.array([t - 0.3, t - 0.05, t, t + 0.05]))
    assert np.isnan(pts[0]).all()
    np.testing.assert_allclose(pts[2], p0, atol=1e-9)
    assert np.linalg.norm(pts[1] - p0) > 0.3 and np.linalg.norm(pts[3] - p0) > 0.3


def test_low_camera_loses_high_shots(scene):
    """横の低いカメラでは、高いショットの一部が画面の上に出る（横の高いカメラではすべて見える）。"""
    def in_frame(name):
        cam = clutter_camera(name, CFG)
        _, _, truth = observe_cluttered(scene, cam, 30.0, np.random.default_rng(1), ClutterOptions.clean())
        fl = truth[truth.shot_index >= 0]
        uv = cam.project(fl[["x_m", "y_m", "z_m"]].to_numpy())
        return float(((uv[:, 0] >= 0) & (uv[:, 0] < cam.width) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.height)).mean())
    assert in_frame("side") > 0.99
    assert in_frame("side_low") < 0.95


def test_only_adds_one_part():
    for part, keys in PARTS.items():
        o = ClutterOptions.only(part)
        clean, full = ClutterOptions.clean(), ClutterOptions()
        for k in vars(o):
            assert getattr(o, k) == (getattr(full, k) if k in keys else getattr(clean, k))


def test_hits_benchmark_clean_vs_cluttered():
    clean = hits_benchmark("side", n_sessions=1, n_rallies=2, opts=ClutterOptions.clean())
    assert clean["recall"] >= 0.9 and clean["precision"] >= 0.9
    assert clean["shuttle_kept"] > 0.99 and set(clean["selected"]) == {"shuttle"}
    full = hits_benchmark("side", n_sessions=1, n_rallies=2)
    assert full["recall"] < clean["recall"]


def test_clutter_benchmark_runs_the_whole_flow():
    r = clutter_benchmark("side", n_sessions=1, n_rallies=2, opts=ClutterOptions.clean())
    assert r["hits"]["recall"] >= 0.8
    assert r["speed"]["n"] >= 4 and r["speed"]["mape"] < 15
    assert r["types"]["scores"]["accuracy"] >= 0.7
    assert r["rallies"]["boundary_recall"] == 1.0
    assert set(r["rows"].columns) >= {"true", "pred", "err_pct"}
