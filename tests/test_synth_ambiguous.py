"""種類の範囲を重ねた「あいまいな」合成ラリー（synth_ambiguous）のテスト。"""

import numpy as np
import pytest

from smash_speed.physics import PhysicsParams
from smash_speed.synth_ambiguous import AMBIGUOUS_PROFILES, ambiguous_benchmark, use_profiles
from smash_speed.synth_rally import SHOT_PROFILES, make_typed_shot

PHYS = PhysicsParams()


def test_use_profiles_swaps_and_restores():
    before = dict(SHOT_PROFILES)
    with use_profiles(AMBIGUOUS_PROFILES):
        assert SHOT_PROFILES["smash"] is AMBIGUOUS_PROFILES["smash"]
        # 遅いスマッシュ（既定の範囲の下限 40 m/s より遅いもの）ができる
        rng = np.random.default_rng(0)
        speeds = [make_typed_shot("smash", rng, PHYS).speed0 for _ in range(15)]
        assert min(speeds) < 40.0
    assert SHOT_PROFILES == before
    with pytest.raises(RuntimeError):
        with use_profiles(AMBIGUOUS_PROFILES):
            raise RuntimeError
    assert SHOT_PROFILES == before
    with pytest.raises(ValueError):
        with use_profiles({k: v for k, v in AMBIGUOUS_PROFILES.items() if k != "net"}):
            pass


def test_ambiguous_profiles_overlap_but_keep_types():
    for k, p in AMBIGUOUS_PROFILES.items():
        d = SHOT_PROFILES[k]
        assert p.shot_type == d.shot_type and p.solve == d.solve
        # 既定の範囲を含む（広げるだけ。狭めない）か、少しだけずらしたもの
        for f in ("speed_mps", "elevation_deg", "hit_z_m", "land_depth_m"):
            lo, hi = getattr(p, f)
            assert lo <= getattr(d, f)[0] + 2.0 and hi >= getattr(d, f)[1] - 2.0
    # 打点の高さで、頭より上のショットと下のショットが重なる
    assert AMBIGUOUS_PROFILES["drop"].hit_z_m[0] < AMBIGUOUS_PROFILES["push"].hit_z_m[1]


def test_ambiguous_benchmark_mixes_types():
    r = ambiguous_benchmark("truth", n_rallies=6, seed=0)
    sc = r["scores"]
    assert 0.5 < sc["accuracy"] < 1.0
    assert sc["by_type"]["serve"]["recall"] == 1.0
    assert SHOT_PROFILES["smash"].speed_mps == (40.0, 110.0)  # 終わったら既定の範囲に戻る
