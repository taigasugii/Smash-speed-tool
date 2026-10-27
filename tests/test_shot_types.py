"""ショットの種類の判定（shot_types.py）のテスト。"""

import numpy as np
import pytest

from smash_speed.physics import PhysicsParams
from smash_speed.shot_types import (
    ShotFeatures, classify_features, classify_shots, confusion_matrix, features_from_trajectory, mistakes, scores,
    shot_features, synthetic_benchmark, truth_benchmark,
)
from smash_speed.shots import Shot, read_shots, write_shots
from smash_speed.synth_rally import SHOT_PROFILES, make_rally, make_typed_shot

PHYS = PhysicsParams()


def test_features_of_a_clear():
    """奥から打った山なりのショット：ネットの上を高く越え、相手側の奥に落ちる。"""
    p0 = np.array([-5.5, 0.0, 2.5])
    v0 = 30.0 * np.array([np.cos(np.radians(40)), 0.0, np.sin(np.radians(40))])
    f = features_from_trajectory(p0, v0, PHYS)
    assert f.speed_mps == pytest.approx(30.0)
    assert f.elevation_deg == pytest.approx(40.0)
    assert f.hit_z_m == 2.5 and f.hit_depth_m == 5.5
    assert f.apex_z_m > f.net_z_m > 3.0
    assert 3.0 < f.land_depth_m < 7.0
    assert classify_features(f) == "clear"


def test_features_do_not_depend_on_side():
    """反対側（x > 0）から打っても、深さ・高さは同じ。"""
    p0 = np.array([-3.0, 0.5, 1.0])
    v0 = np.array([12.0, -1.0, 6.0])
    a = features_from_trajectory(p0, v0, PHYS)
    b = features_from_trajectory(p0 * [-1, 1, 1], v0 * [-1, 1, 1], PHYS)
    for k in ("land_depth_m", "net_z_m", "apex_z_m", "hit_depth_m"):
        assert getattr(a, k) == pytest.approx(getattr(b, k), abs=1e-6)


def test_short_shot_lands_on_own_side():
    """ネットを越えないショットは、ネットの上の高さが nan、着地点の深さが負。"""
    f = features_from_trajectory(np.array([-2.0, 0.0, 0.5]), np.array([2.0, 0.0, 1.0]), PHYS)
    assert np.isnan(f.net_z_m)
    assert f.land_depth_m < 0
    assert classify_features(f) == "net"


@pytest.mark.parametrize("profile", [p for p in SHOT_PROFILES if not p.startswith("serve")])
def test_each_synthetic_type(profile):
    """合成の 1 本のショットは、どの種類も（ほとんど）正しく判定される。"""
    rng = np.random.default_rng(3)
    want = SHOT_PROFILES[profile].shot_type
    got = []
    for _ in range(8):
        s = make_typed_shot(profile, rng, PHYS, side="h" if rng.uniform() < 0.5 else "o")
        got.append(classify_features(features_from_trajectory(s.p0, s.v0, PHYS)))
    assert got.count(want) >= 7, got


def test_first_shot_is_serve():
    f = ShotFeatures(30.0, 45.0, 1.0, 2.5, 6.0, 3.0, 6.0, first_in_rally=True)
    assert classify_features(f) == "serve"
    f.first_in_rally = False
    assert classify_features(f) == "lob"


def test_classify_shots_keeps_given_types(tmp_path):
    """種類の決まったショットはそのまま、打点・速度のないショットは unknown のまま。CSV に書いて読める。"""
    rally = make_rally(np.random.default_rng(5), PHYS, n_shots=6)
    truth = rally.to_shots(phys=PHYS)
    blank = [Shot(**{**s.__dict__, "shot_type": "unknown"}) for s in truth]
    blank.append(Shot(t_hit_s=99.0))
    got = classify_shots(blank, PHYS)
    assert [s.shot_type for s in got[:-1]] == [s.shot_type for s in truth]
    assert got[-1].shot_type == "unknown"
    assert blank[0].shot_type == "unknown"  # 元の列は変えない
    kept = classify_shots([Shot(**{**truth[1].__dict__, "shot_type": "drop"})], PHYS)
    assert kept[0].shot_type == "drop"
    assert classify_shots(kept, PHYS, overwrite=True)[0].shot_type == truth[1].shot_type
    path = write_shots(tmp_path / "shots.csv", got)
    assert [s.shot_type for s in read_shots(path)] == [s.shot_type for s in got]


def test_shot_features_uses_index_in_rally():
    s = make_rally(np.random.default_rng(1), PHYS, n_shots=3).to_shots(phys=PHYS)
    assert shot_features(s[0], PHYS).first_in_rally
    assert not shot_features(s[1], PHYS).first_in_rally
    assert shot_features(Shot(t_hit_s=0.0)) is None


def test_confusion_matrix_and_scores():
    m = confusion_matrix(["smash", "smash", "drop", "net"], ["smash", "drop", "drop", "unknown"])
    assert m.loc["smash", "smash"] == 1 and m.loc["smash", "drop"] == 1 and m.loc["net", "unknown"] == 1
    sc = scores(m)
    assert sc["n"] == 4 and sc["accuracy"] == pytest.approx(0.5)
    assert sc["by_type"]["drop"]["precision"] == pytest.approx(0.5)
    assert sc["by_type"]["smash"]["recall"] == pytest.approx(0.5)
    assert sc["by_type"]["clear"]["recall"] is None
    assert mistakes(m)[0][2] == 1


def test_truth_benchmark():
    """正解の軌道からは、ほぼすべて正しく判定される。"""
    r = truth_benchmark(15, seed=4, phys=PHYS)
    assert r["scores"]["n"] > 50
    assert r["scores"]["accuracy"] > 0.97


def test_fitted_benchmark():
    """観測して当てはめた軌道からも、大半は正しく判定される（30 fps・打球音あり・2 ラリー）。"""
    r = synthetic_benchmark(2, 30.0, seed=1, hits="audio", phys=PHYS)
    assert r["scores"]["n"] >= 6
    assert r["scores"]["accuracy"] > 0.75
