"""打つ人の足元の自動推定（D3）のテスト。"""

import numpy as np
import pandas as pd
import pytest

from smash_speed.hitter import HitterOptions, attach_hitters, hitter_at, synthetic_benchmark
from smash_speed.shots import SHOTS_FILE, Shot, read_shots, write_shots


def _track(side: str, player: int, xy, t, airborne=False) -> pd.DataFrame:
    xy = np.broadcast_to(np.asarray(xy, float), (len(t), 2))
    return pd.DataFrame({"frame": np.arange(len(t)), "t": t, "side": side, "player": player,
                         "x": xy[:, 0], "y": xy[:, 1], "airborne": airborne})


T = np.arange(0, 3, 1 / 30)


def test_hitter_at_uses_the_hitting_side_and_drops_wild_frames():
    near = _track("h", 0, (-5.0, 1.0), T)
    # 足が隠れて枠の下の辺が上がり、2 m 奥へ飛んだコマ（打つ直前の 2 コマ）
    near.loc[(near.t > 1.4) & (near.t < 1.5), "x"] = -3.0
    far = _track("o", 0, (4.0, -1.0), T)
    players = pd.concat([near, far], ignore_index=True)
    e = hitter_at(players, 1.5, "h")
    assert (e.x, e.y, e.side, e.player) == (-5.0, 1.0, "h", 0)
    win = near[(near.t - 1.5).abs() <= 0.2]
    assert e.n_frames == int((win.x == -5.0).sum()) < len(win)   # 外れのコマは使わない
    assert not e.airborne and not e.ambiguous
    # 側が分からなくても、打点に近い選手を選ぶ
    assert hitter_at(players, 1.5, "", hit_xy=(3.6, -0.7, 2.5)).side == "o"
    # 側も打点も分からないと、どちらが打ったか決められない
    assert hitter_at(players, 1.5) is None
    # 打った時刻の前後に位置がない
    assert hitter_at(players, 10.0, "h") is None
    assert hitter_at(pd.DataFrame(), 1.0, "h") is None


def test_hitter_at_follows_a_moving_player_and_reports_jumps():
    xy = np.column_stack([-6.0 + T, np.zeros_like(T)])          # 1 m/s で前へ
    players = _track("h", 0, xy, T, airborne=(T > 1.9) & (T < 2.3))
    e = hitter_at(players, 1.0, "h")
    assert e.x == pytest.approx(-5.0, abs=0.02) and e.spread_m < 0.15
    assert not e.airborne
    assert hitter_at(players, 2.0, "h").airborne


def test_hitter_at_doubles_picks_the_player_near_the_hit():
    players = pd.concat([_track("h", 0, (-5.5, -1.5), T), _track("h", 1, (-2.5, 1.5), T)], ignore_index=True)
    e = hitter_at(players, 1.0, "h", hit_xy=(-2.2, 1.2))
    assert (e.player, e.x, e.ambiguous) == (1, -2.5, False)
    # 2 人の真ん中の打点では決められない
    assert hitter_at(players, 1.0, "h", hit_xy=(-4.0, 0.0)).ambiguous
    assert hitter_at(players, 1.0, "h") is None


def test_attach_hitters_fills_the_shot_list(tmp_path):
    players = pd.concat([_track("h", 0, (-5.0, 1.0), T), _track("o", 0, (4.0, -1.0), T)], ignore_index=True)
    shots = [Shot(t_hit_s=0.5, side="h", hit_x_m=-4.6, hit_y_m=1.2, hit_z_m=2.6),
             Shot(t_hit_s=1.2, side="o"),
             Shot(t_hit_s=2.0, side=""),       # 側も打点も分からない
             Shot(t_hit_s=9.0, side="h")]      # 選手の位置がない時刻
    assert attach_hitters(shots, players, HitterOptions()) == 2
    assert [(s.hitter_x_m, s.hitter_y_m) for s in shots] == [(-5.0, 1.0), (4.0, -1.0), (None, None), (None, None)]
    write_shots(tmp_path / SHOTS_FILE, shots)
    back = read_shots(tmp_path / SHOTS_FILE)
    assert [(s.hitter_x_m, s.hitter_y_m) for s in back] == [(s.hitter_x_m, s.hitter_y_m) for s in shots]


def test_synthetic_benchmark_is_within_a_foot_length():
    r = synthetic_benchmark(n_seeds=1, duration_s=12.0)
    for side in ("h", "o"):
        assert r[side]["n"] >= 10 and r[side]["missed"] == 0
        assert r[side]["median_m"] < 0.3 and r[side]["p90_m"] < 0.4
