"""選手の足元と打球の間隔からの初速の目安（D1c）のテスト。"""

import numpy as np
import pandas as pd
import pytest

from smash_speed.feet_speed import compare, speed_range, speed_to_reach
from smash_speed.physics import MPS_TO_KMH, PhysicsParams, simulate

PHYS = PhysicsParams()


@pytest.mark.parametrize("speed_kmh, elev_deg, dt", [(300.0, -8.0, 0.5), (100.0, 25.0, 1.2), (60.0, 5.0, 0.9)])
def test_speed_to_reach_recovers_a_simulated_shot(speed_kmh, elev_deg, dt):
    v = speed_kmh / MPS_TO_KMH
    e = np.radians(elev_deg)
    p, _ = simulate(np.zeros(3), np.array([v * np.cos(e), 0.0, v * np.sin(e)]), 0.0, np.array([dt]), PHYS)
    got = speed_to_reach(float(p[0, 0]), float(p[0, 2]), dt, PHYS)
    assert got == pytest.approx(speed_kmh, rel=0.005)


def test_speed_range_contains_the_nominal_and_grows_with_distance():
    nominal, lo, hi = speed_range(8.0, 1.0, PHYS)
    assert lo < nominal < hi
    assert speed_range(10.0, 1.0, PHYS)[0] > nominal      # 遠いほど速く
    assert speed_range(8.0, 0.6, PHYS)[0] > nominal       # 短い時間ほど速く
    assert np.isnan(speed_to_reach(0.0, 0.0, 1.0, PHYS))


def _players(t, feet):
    """側ごとに決まった足元に立つ選手。"""
    rows = [pd.DataFrame({"t": t, "side": s, "player": 0, "x": xy[0], "y": xy[1], "airborne": False})
            for s, xy in feet.items()]
    return pd.concat(rows, ignore_index=True)


def test_compare_uses_the_true_hits_and_flags_a_slow_fit():
    t = np.arange(0, 4, 1 / 30)
    players = _players(t, {"h": (-4.0, 1.0), "o": (4.0, 1.0)})
    truth = pd.DataFrame({"t_s": [0.5, 1.5, 2.6], "side": ["h", "o", "h"], "rally": [1, 1, 1]})
    nominal = speed_to_reach(8.0, 0.0, 1.0, PHYS)
    shots = pd.DataFrame({
        "t_hit_s": [0.52, 1.0, 1.48, 2.61],       # 1.0 s は打球でないショット（正解に合わないので使わない）
        "side": ["h", "o", "o", "h"],
        "shot_type": ["drive", "net", "drive", "lob"],
        "speed_kmh": [nominal, 500.0, nominal * 0.3, 120.0],
        "rally_id": [1, 1, 1, 1],
    })
    df = compare(shots, players, truth)
    assert list(df["t_s"]) == [0.5, 1.5]          # 最後の打球は次がないので比べない
    assert df["distance_m"].tolist() == [8.0, 8.0]
    assert df["feet_kmh"].iloc[0] == pytest.approx(nominal, abs=0.1)
    assert df["inside"].tolist() == [True, False]
    assert df["side_ok"].all()
    # 正解がなければ一覧の打球で組を作る（1.0 s の打球も入り、同じ側が続く 1.0 → 1.48 s は比べない）
    own = compare(shots, players)
    assert list(own["t_s"]) == [0.52, 1.48]
