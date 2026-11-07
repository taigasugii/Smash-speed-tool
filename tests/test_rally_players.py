"""ラリーの区切りに選手の動きを足したもの（rallies.drop_false_rallies）と、その合成データ（synth_rally_players）のテスト。"""

import json

import numpy as np
import pandas as pd
import pytest

from smash_speed.config import load_config
from smash_speed.movement import movement_tracks, write_movement
from smash_speed.physics import PhysicsParams
from smash_speed.rallies import (REASON_NO_HITTER, REASON_NOT_READY, REASON_OFF_COURT, RallyOptions,
                                 check_short_rally, player_benchmark, rally_precision, split_rallies)
from smash_speed.rally_report import write_report
from smash_speed.rally_stats import STATS_FILE, write_stats
from smash_speed.shot_list import ShotListOptions, rallies_with_players
from smash_speed.shots import Rally, Shot, read_rallies, read_shots, write_rallies, write_shots
from smash_speed.synth_rally_players import make_player_session, session_feet

CFG = load_config()
COURT = CFG["court"]
PHYS = PhysicsParams.from_config(CFG)
FPS = 30.0


def _players(t, feet):
    """足元の位置（側 → (n, 2)）をそのまま ``players.csv`` の形にする（揺れなし）。"""
    return pd.concat([pd.DataFrame({"frame": np.arange(len(t)), "t": t, "side": side, "player": 0,
                                    "x": xy[:, 0], "y": xy[:, 1]}) for side, xy in feet.items()], ignore_index=True)


def _walk(t, a, b, t0, t1):
    """``t0``〜``t1`` に a から b へ一定の速さで歩く（その前は a、後は b）。"""
    k = np.clip((t - t0) / (t1 - t0), 0, 1)[:, None]
    return np.asarray(a, float) + (np.asarray(b, float) - np.asarray(a, float)) * k


T = np.arange(0, 6, 1 / FPS)
SERVE_H, SERVE_O = np.array([-2.5, -0.7]), np.array([3.0, 0.7])


def _check(shots, feet, t_end=None):
    tracks = movement_tracks(_players(T, feet))
    return check_short_rally(shots, t_end, tracks, RallyOptions(), COURT)


def test_serve_only_rally_with_players_ready_is_kept():
    # 2 人とも止まって構え、サービスコートからサーブ（サービスエース・サーブのミス）→ 落とさない
    feet = {"h": np.tile(SERVE_H, (len(T), 1)), "o": np.tile(SERVE_O, (len(T), 1))}
    serve = Shot(t_hit_s=3.0, side="h", hit_x_m=-2.4, hit_y_m=-0.6)
    assert _check([serve], feet, t_end=4.0) == ""


def test_neighbor_court_shuttle_is_dropped():
    # 選手はサーブの位置に止まって待っている。打点はとなりのコート（y = 8 m）
    feet = {"h": np.tile(SERVE_H, (len(T), 1)), "o": np.tile(SERVE_O, (len(T), 1))}
    shot = Shot(t_hit_s=3.0, side="o", hit_x_m=2.0, hit_y_m=8.2)
    assert _check([shot], feet, t_end=4.0) == REASON_NO_HITTER


def test_pickup_while_walking_is_dropped():
    # h の選手が歩いて拾いに行き、拾ってすぐ返す。o の選手は歩いて受け取りに行く
    h = _walk(T, [-1.0, 1.5], [-4.0, -1.0], 1.0, 3.0)
    o = _walk(T, [1.0, -1.0], [3.5, 0.5], 2.0, 4.5)
    shot = Shot(t_hit_s=3.1, side="h", hit_x_m=-4.0, hit_y_m=-1.0)
    assert _check([shot], {"h": h, "o": o}, t_end=4.5) == REASON_NOT_READY


def test_player_off_court_is_dropped():
    # o の選手がコートの外（サイドラインの 1.2 m 外）へ歩いていった
    feet = {"h": np.tile(SERVE_H, (len(T), 1)), "o": np.tile([3.0, 4.25], (len(T), 1))}
    shot = Shot(t_hit_s=3.0, side="h", hit_x_m=-2.4, hit_y_m=-0.6)
    assert _check([shot], feet, t_end=4.0) == REASON_OFF_COURT


def test_fast_movement_means_play_even_if_the_hitter_is_late():
    # 受ける選手が打点へ素早く動いている（間に合わず遠い）→ プレーしているので落とさない
    h = np.tile(SERVE_H, (len(T), 1))
    o = _walk(T, SERVE_O, [0.8, -1.5], 3.0, 3.6)   # 0.6 秒で 3 m（最高 5 m/s ほど）
    shots = [Shot(t_hit_s=3.0, side="h", hit_x_m=-2.4, hit_y_m=-0.6), Shot(t_hit_s=3.3, side="o", hit_x_m=0.3, hit_y_m=-1.8)]
    assert _check(shots, {"h": h, "o": o}, t_end=4.0) == ""


def test_no_player_positions_means_no_check():
    feet = {"h": np.tile(SERVE_H, (len(T), 1))}   # o の側の位置がない
    shot = Shot(t_hit_s=3.0, side="o", hit_x_m=2.0, hit_y_m=8.2)
    assert _check([shot], feet, t_end=4.0) == ""


def _two_rallies_and_a_neighbor():
    # ラリー 1（3 本、0〜2.6 秒）・となりのコートのシャトル（7 秒）・ラリー 2（2 本、12 秒〜）
    shots = [Shot(t_hit_s=0.5, side="h", speed_kmh=50, hit_x_m=-2.4, hit_y_m=-0.6, t_land_s=3.0),
             Shot(t_hit_s=1.4, side="o", speed_kmh=60, hit_x_m=3.0, hit_y_m=0.5, t_land_s=2.5),
             Shot(t_hit_s=2.2, side="h", speed_kmh=150, hit_x_m=-3.0, hit_y_m=0.0, t_land_s=2.6,
                  land_x_m=4.0, land_y_m=1.0),
             Shot(t_hit_s=7.0, side="o", speed_kmh=40, hit_x_m=2.0, hit_y_m=8.2, t_land_s=8.0,
                  land_x_m=-3.0, land_y_m=8.0),
             Shot(t_hit_s=12.0, side="h", speed_kmh=50, hit_x_m=-2.4, hit_y_m=-0.6, t_land_s=14.0),
             Shot(t_hit_s=13.0, side="o", speed_kmh=60, hit_x_m=3.0, hit_y_m=0.5, t_land_s=13.6,
                  land_x_m=-2.0, land_y_m=0.0)]
    t = np.arange(0, 15, 1 / FPS)
    feet = {"h": np.tile(SERVE_H, (len(t), 1)), "o": np.tile(SERVE_O, (len(t), 1))}
    feet["o"][(t > 13.0) & (t < 13.5)] += [0.0, 0.0]
    # ラリーの中は素早く動く（ラリー 1 の 1〜2 秒、ラリー 2 の 12.3〜12.9 秒）
    feet["h"] = feet["h"] + _walk(t, [0, 0], [0, 2.5], 1.0, 1.5) - _walk(t, [0, 0], [0, 2.5], 1.6, 2.1)
    feet["o"] = feet["o"] + _walk(t, [0, 0], [-2.5, 0], 12.3, 12.8) - _walk(t, [0, 0], [-2.5, 0], 13.2, 13.7)
    return shots, _players(t, feet)


def test_split_rallies_drops_false_rally_and_renumbers():
    shots, players = _two_rallies_and_a_neighbor()
    before = split_rallies([Shot(**vars(s)) for s in shots])
    assert [r.n_shots for r in before] == [3, 1, 2]
    rallies = split_rallies(shots, players=players)
    assert [(r.rally_id, r.n_shots) for r in rallies] == [(1, 3), (2, 2)]
    assert [s.rally_id for s in shots] == [1, 1, 1, None, 2, 2]
    assert shots[3].index_in_rally is None and shots[3].note.startswith(f"ラリーの外：{REASON_NO_HITTER}")
    # 選手の位置がなければ今までと同じ
    again = split_rallies([Shot(**vars(s)) for s in shots], players=None)
    assert [r.n_shots for r in again] == [3, 1, 2]


def test_rally_precision_counts_false_and_lost_rallies():
    truth = [Shot(t_hit_s=t, rally_id=r) for t, r in ((0.0, 1), (1.0, 1), (5.0, 2))]
    true_rallies = [Rally(1, 0.0), Rally(2, 5.0)]
    pred = [Shot(t_hit_s=0.0, rally_id=1), Shot(t_hit_s=1.0, rally_id=1), Shot(t_hit_s=3.0, rally_id=2),
            Shot(t_hit_s=5.0, rally_id=3)]
    r = rally_precision(pred, [Rally(1, 0.0), Rally(2, 3.0), Rally(3, 5.0)], truth, true_rallies)
    assert r == {"rally_precision": pytest.approx(2 / 3), "n_false": 1, "n_lost": 0}
    pred[3].rally_id = None
    r = rally_precision(pred, [Rally(1, 0.0), Rally(2, 3.0)], truth, true_rallies)
    assert r["n_lost"] == 1 and r["rally_precision"] == pytest.approx(1 / 2)


def test_player_session_has_extras_between_rallies_and_players_wait_before_serves():
    rng = np.random.default_rng(3)
    session, extras = make_player_session(rng, PHYS, 4, p_pickup=0.6, p_neighbor=0.4)
    assert extras and all(e.kind in ("pickup", "neighbor") for e in extras)
    for e in extras:
        a, b = session[e.after_rally], session[e.after_rally + 1]
        assert a.t_end < e.seg.t_hit and e.seg.t_land < b.t_start - 1.5
        if e.kind == "neighbor":
            assert e.seg.p0[1] > 4.0
    t = np.arange(session[0].t_start - 2.0, session[-1].t_end, 1 / FPS)
    feet = session_feet(session, extras, t, np.random.default_rng(4), COURT)
    for r in session[1:]:   # 最初のサーブの前は、2 人とも止まっている
        m = (t > r.t_start - 0.5) & (t < r.t_start)
        for xy in feet.values():
            assert np.ptp(xy[m], axis=0).max() < 0.05
    m = (t > session[0].t_start) & (t < session[0].t_end)
    assert max(np.hypot(*np.diff(xy[m], axis=0).T).max() * FPS for xy in feet.values()) > 3.0


def test_truth_benchmark_drops_false_rallies_without_losing_real_ones():
    r = player_benchmark(2, 4, FPS, seed=1, mode="truth")
    assert sum(r["n_extra"].values()) > 0
    before, after = r["before"], r["after"]
    assert before["n_false"] > 0 and after["n_false"] < before["n_false"]
    assert after["rally_precision"] > before["rally_precision"] and after["rally_precision"] >= 0.9
    assert after["n_lost"] == 0 and after["exact"] == before["exact"]


def test_rallies_with_players_rewrites_outputs_stats_and_report(tmp_path):
    shots, players = _two_rallies_and_a_neighbor()
    rallies = split_rallies(shots)
    write_shots(tmp_path / "shots.csv", shots)
    write_rallies(tmp_path / "rallies.csv", rallies)
    players.to_csv(tmp_path / "players.csv", index=False)
    write_movement(tmp_path, players, COURT)
    info = {"n_rallies": 3, "players": {"movement": {"h0": {}}}}
    o = ShotListOptions(video="v.mp4", calib="c.json", out_dir=str(tmp_path))
    new = rallies_with_players(shots, rallies, None, o, CFG, tmp_path, info, progress=lambda _: None)
    assert len(new) == 2 and info["n_rallies"] == 2 and info["players"]["n_shots_outside_rallies"] == 1
    assert [s.rally_id for s in read_shots(tmp_path / "shots.csv")] == [1, 1, 1, None, 2, 2]
    assert len(read_rallies(tmp_path / "rallies.csv")) == 2
    assert {p["rallies"][-1]["rally_id"] for p in json.loads((tmp_path / "movement.json").read_text("utf-8"))
            ["players"].values()} == {2}
    (tmp_path / "shot_list.json").write_text(json.dumps(info), encoding="utf-8")

    st = write_stats(tmp_path, court=COURT)
    assert st["n_shots_outside_rallies"] == 1
    saved = json.loads((tmp_path / STATS_FILE).read_text(encoding="utf-8"))
    p = saved["players"]["players"]["h0"]
    assert p["side"] == "h" and p["distance_m"] > 0 and p["max_speed_mps"] > 0
    heat = p["heatmap"]
    assert heat["cell_m"] == 0.5 and len(heat["seconds"]) * 0.5 == pytest.approx(heat["extent_m"][3] - heat["extent_m"][2])
    assert sum(map(sum, heat["seconds"])) == pytest.approx(p["seconds"], abs=0.1)
    html = write_report(tmp_path, COURT).read_text(encoding="utf-8")
    assert "ラリーの外の打球 1 本" in html and "いた場所の割合" in html and "コートの外 %" in html

    # 選手の位置を出さなかった解析：stats.json に players は入らない
    (tmp_path / "shot_list.json").write_text(json.dumps({}), encoding="utf-8")
    assert "players" not in write_stats(tmp_path, court=COURT)


def test_rallies_with_players_rewrites_movement_with_final_rallies(tmp_path):
    # 選手の位置は当てはめの前に出すので、movement.json のラリーは古い（ここではない）。外す打球がなくても書き直す
    shots, players = _two_rallies_and_a_neighbor()
    shots = [s for s in shots if s.hit_y_m < 5.0]   # となりのコートの打球なし
    rallies = split_rallies(shots)
    players.to_csv(tmp_path / "players.csv", index=False)
    write_movement(tmp_path, players, COURT)
    assert "rallies" not in json.loads((tmp_path / "movement.json").read_text("utf-8"))["players"]["h0"]
    info = {"players": {"movement": {"h0": {}}}}
    o = ShotListOptions(video="v.mp4", calib="c.json", out_dir=str(tmp_path))
    new = rallies_with_players(shots, rallies, None, o, CFG, tmp_path, info, progress=lambda _: None)
    assert len(new) == 2 and info["players"]["n_shots_outside_rallies"] == 0
    saved = json.loads((tmp_path / "movement.json").read_text("utf-8"))["players"]["h0"]
    assert [r["rally_id"] for r in saved["rallies"]] == [1, 2]
