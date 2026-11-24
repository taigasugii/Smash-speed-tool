"""ショットごとのフットワークと出来の目安（smash_speed.footwork）のテスト。"""

import json

import numpy as np
import pandas as pd

from smash_speed.config import load_config
from smash_speed.footwork import (FOOTWORK_COLUMNS, FOOTWORK_CSV, FOOTWORK_FILE, FootworkOptions, footwork_table,
                                  player_tracks, quality_score, quality_terms, shot_footwork, synthetic_benchmark,
                                  truth_track, write_footwork)
from smash_speed.shots import Shot
from smash_speed.synth_players import observe_players
from smash_speed.synth_rally import corner_rally_camera, rally_camera

CFG = load_config()
COURT = CFG["court"]
FPS = 30.0
T = np.arange(0, 6, 1 / FPS)


def _quiet(feet, lift=None, camera=None):
    """揺れのない ``players.csv`` の形。"""
    return observe_players(T, feet, camera or rally_camera(CFG), np.random.default_rng(0), jitter=0, outlier_rate=0,
                           drift=0, swing=0, lift=lift)


def _bump(t0, h, dur):
    w = (T - t0) / (dur / 2)
    return np.where(np.abs(w) < 1, h * (1 - w * w), 0.0)


def _walk(a, b, t0, t1):
    k = np.clip((T - t0) / (t1 - t0), 0, 1)[:, None]
    return np.asarray(a, float) + (np.asarray(b, float) - np.asarray(a, float)) * k


def test_split_step_found_with_timing():
    # o の側の選手が、h の打った 2.0 秒の 0.05 秒前にいちばん高く 8 cm 跳ぶ。h は止まっている
    feet = {"h": np.tile([-3.0, 0.0], (len(T), 1)), "o": np.tile([3.0, 0.5], (len(T), 1))}
    lift = {"o": _bump(1.95, 0.08, 0.24)}
    for cam in (rally_camera(CFG), corner_rally_camera()):
        tracks = player_tracks(_quiet(feet, lift, cam), cam)
        r = shot_footwork(tracks[("o", 0)], 2.8, 2.0, None, np.array([3.0, 0.5]))
        assert r["split_step"] is True
        assert abs(r["split_dt_s"] + 0.05) <= 0.04
        assert abs(r["split_height_m"] - 0.08) < 0.02
        # 跳ばなかった h はスプリットステップなし
        r = shot_footwork(tracks[("h", 0)], 2.0, 1.2, None, np.array([-3.0, 0.0]))
        assert r["split_step"] is False


def test_reaction_move_prep_recovery_on_truth():
    # 相手が 1.0 秒に打ち、0.25 秒後に動き出して 1.0 秒かけて 3 m 先へ、2.6 秒に打って、0.4 秒でもとへ戻る
    xy = _walk([3.0, 0.0], [5.0, 2.236], 1.25, 2.25)
    back = _walk([0.0, 0.0], [-2.0, -2.236], 2.7, 3.1)
    xy = np.where((T >= 2.7)[:, None], xy + back, xy)
    tr = truth_track(T, xy)
    r = shot_footwork(tr, 2.6, 1.0, 4.0, np.array([3.0, 0.0]))
    assert abs(r["reaction_s"] - 0.25) < 0.05
    assert abs(r["move_distance_m"] - 3.0) < 0.05
    assert abs(r["prep_s"] - 0.48) < 0.05      # 打つ所の 0.4 m 手前を 2.12 秒に過ぎ、2.6 秒に打つ
    assert abs(r["recovery_m"] - 3.0) < 0.05
    assert 0.3 < r["recovery_s"] < 0.5


def test_table_rows_and_uncertain_at_edge():
    feet = {"h": np.tile([-3.0, 0.0], (len(T), 1)), "o": np.tile([3.0, 0.5], (len(T), 1))}
    players = _quiet(feet)
    shots = [Shot(t_hit_s=1.0, side="h", shot_type="serve", rally_id=1, index_in_rally=1),
             Shot(t_hit_s=2.0, side="o", shot_type="clear", rally_id=1, index_in_rally=2, hit_z_m=2.5,
                  land_x_m=-5.5, land_y_m=0.0),
             Shot(t_hit_s=3.2, side="h", shot_type="smash", rally_id=1, index_in_rally=3, hit_z_m=2.6, speed_kmh=250)]
    tab = footwork_table(shots, player_tracks(players, rally_camera(CFG)), COURT)
    assert list(tab.columns) == FOOTWORK_COLUMNS
    assert tab["prev_t_hit_s"].tolist()[0] is None or np.isnan(tab["prev_t_hit_s"].iloc[0])
    assert tab["prev_t_hit_s"].iloc[1] == 1.0 and tab["prev_t_hit_s"].iloc[2] == 2.0
    assert not tab["uncertain"].any()
    # o の選手の枠が画面の上の端に付いていたら、確かでない
    cut = players.copy()
    cut.loc[cut["side"] == "o", "y0"] = 0.0
    tab = footwork_table(shots, player_tracks(cut, rally_camera(CFG)), COURT)
    assert tab["uncertain"].tolist() == [False, True, False]
    assert "画面の端" in tab["uncertain_reason"].iloc[1]


def test_quality_terms():
    s = Shot(t_hit_s=1.0, side="h", shot_type="clear", hit_z_m=2.3, land_x_m=5.1, land_y_m=0.0)
    q = quality_terms(s, {"prep_s": 0.2, "uncertain": False})
    assert abs(q["height"] - 0.5) < 1e-9 and q["speed"] is None
    assert abs(q["depth"] - 0.5) < 1e-9 and q["prep"] == 0.5
    assert quality_score(q) == 50.0
    # 自分の側に落ちた（相手に届いていない）なら深さは出さない。項目が 1 つなら点も出さない
    s = Shot(t_hit_s=1.0, side="h", shot_type="drop", hit_z_m=None, land_x_m=-1.0)
    q = quality_terms(s, {"prep_s": None})
    assert q["depth"] is None and quality_score(q) is None
    # ネット前に浅く落としたドロップは満点
    s = Shot(t_hit_s=1.0, side="o", shot_type="drop", hit_z_m=2.7, land_x_m=-1.2)
    q = quality_terms(s, {"prep_s": 0.5, "uncertain": False})
    assert q["depth"] == 1.0 and quality_score(q) == 100.0


def test_write_footwork(tmp_path):
    feet = {"h": np.tile([-3.0, 0.0], (len(T), 1)), "o": np.tile([3.0, 0.5], (len(T), 1))}
    shots = [Shot(t_hit_s=1.0, side="h", shot_type="serve", rally_id=1, index_in_rally=1),
             Shot(t_hit_s=2.0, side="o", shot_type="clear", rally_id=1, index_in_rally=2)]
    summary = write_footwork(tmp_path, shots, _quiet(feet, {"o": _bump(0.95, 0.08, 0.24)}), rally_camera(CFG), COURT)
    tab = pd.read_csv(tmp_path / FOOTWORK_CSV)
    assert len(tab) == 2 and bool(tab["split_step"].iloc[1]) is True
    data = json.loads((tmp_path / FOOTWORK_FILE).read_text(encoding="utf-8"))
    assert data["players"]["o0"]["split_rate"] == 1.0
    assert data["rallies"][0]["rally_id"] == 1
    assert summary["players"]["h0"]["n_shots"] == 1


def test_synthetic_benchmark_small():
    r = synthetic_benchmark(n_sessions=1, n_rallies=4)
    assert r["split"]["recall"] >= 0.8 and r["split"]["precision"] >= 0.8
    assert r["prep_s"]["median_abs"] <= 0.07
    assert r["reaction_s"]["median_abs"] <= 0.1
    assert r["uncertain_share"] < 0.2


def test_foot_noise_and_uncertain_feet_are_used():
    """K3：``foot_noise_m`` の大きい選手は最高の速さを強くならし、「足元が不確か」の多い窓は確かでないとする。"""
    import numpy as np
    import pandas as pd

    from smash_speed.footwork import FootworkOptions, _uncertain, player_tracks

    rng = np.random.default_rng(1)
    t = np.arange(0, 4, 1 / 30)
    x = np.where(t < 2, 4.0, 5.0) + rng.normal(0, 0.15, len(t))
    pl = pd.DataFrame({"t": t, "frame": np.arange(len(t)), "side": "o", "player": 0, "x": x, "y": 0.0,
                       "foot_noise_m": 0.25, "foot_uncertain": t > 1.0})
    on = player_tracks(pl, o=FootworkOptions())[("o", 0)]
    off = player_tracks(pl, o=FootworkOptions(use_foot_noise=False))[("o", 0)]
    assert on["peak_speed"].max() < off["peak_speed"].max()
    why = _uncertain(on, 1.5, 3.0, True, FootworkOptions())
    assert any("足元が不確か" in w for w in why)
    assert not any("足元が不確か" in w for w in _uncertain(off, 1.5, 3.0, True, FootworkOptions()))
