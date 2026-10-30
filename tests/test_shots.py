"""ショットとラリーのデータ形式（shots.csv・rallies.csv）のテスト。"""

import csv
import json

import numpy as np
import pytest

from smash_speed.batch import Candidate, save_candidates
from smash_speed.physics import PhysicsParams, simulate_until_ground
from smash_speed.shots import (SHOT_COLUMNS, Rally, Shot, confidence_from_ci, export_batch, make_rallies,
                               read_rallies, read_shots, shot_from_fit, shots_of, side_of, write_rallies,
                               write_shots)


def _fit(p0, v0, t_ref=1.0, vt=6.7, ci=(70.0, 80.0)):
    return {"t_ref_s": t_ref, "p0_m": list(p0), "v0_mps": list(v0), "terminal_velocity_mps": vt,
            "speed0_ci_mps": list(ci)}


def test_shots_round_trip_keeps_values_and_blanks(tmp_path):
    shots = [Shot(t_hit_s=2.5, side="o", shot_type="clear", speed_kmh=150.0, confidence=0.5, rally_id=1,
                  index_in_rally=2, note="奥から"),
             Shot(t_hit_s=1.0, side="h", shot_type="serve", hit_x_m=-2.0, hit_y_m=0.3, hit_z_m=1.0)]
    p = write_shots(tmp_path / "shots.csv", shots)
    back = read_shots(p)
    # 打った時刻の順に並ぶ。空欄は None、文字列の空欄は空文字に戻る
    assert [s.t_hit_s for s in back] == [1.0, 2.5]
    assert back[1] == shots[0]
    assert back[0] == shots[1]
    assert back[0].speed_kmh is None and back[0].note == "" and back[0].rally_id is None
    with p.open(encoding="utf-8-sig", newline="") as f:
        assert next(csv.reader(f)) == SHOT_COLUMNS


def test_read_shots_ignores_unknown_columns_and_fills_missing(tmp_path):
    p = tmp_path / "shots.csv"
    p.write_text("t_hit_s,shot_type,speed_kmh,memo\n3.0,smash,300.5,手で足した列\n", encoding="utf-8-sig")
    s, = read_shots(p)
    assert s.t_hit_s == 3.0 and s.shot_type == "smash" and s.speed_kmh == 300.5
    assert s.side == "" and s.confidence is None


def test_write_shots_rejects_bad_values(tmp_path):
    with pytest.raises(ValueError):
        write_shots(tmp_path / "a.csv", [Shot(t_hit_s=0.0, shot_type="hairpin")])
    with pytest.raises(ValueError):
        write_shots(tmp_path / "a.csv", [Shot(t_hit_s=0.0, side="near")])
    with pytest.raises(ValueError):
        write_shots(tmp_path / "a.csv", [Shot(t_hit_s=0.0, confidence=1.5)])


def test_rallies_round_trip_and_grouping(tmp_path):
    shots = [Shot(t_hit_s=t) for t in (10.0, 0.0, 1.2, 2.0, 11.0)]
    rallies = make_rallies(shots, max_gap_s=3.0)
    assert [(r.rally_id, r.n_shots, r.t_start_s, r.t_end_s) for r in rallies] == [(1, 3, 0.0, 2.0), (2, 2, 10.0, 11.0)]
    assert [s.t_hit_s for s in shots_of(rallies[0], shots)] == [0.0, 1.2, 2.0]
    assert [s.index_in_rally for s in shots_of(rallies[1], shots)] == [1, 2]
    rallies[0].winner_side, rallies[0].score_h, rallies[0].score_o = "h", 1, 0
    p = write_rallies(tmp_path / "rallies.csv", rallies)
    back = read_rallies(p)
    assert back == rallies
    assert back[1].score_h is None and isinstance(back[0].score_h, int)


def test_side_of_uses_court_x_sign():
    assert side_of(-3.0) == "h" and side_of(2.0) == "o"
    assert side_of(None) == "" and side_of(float("nan")) == ""


def test_confidence_from_ci():
    assert confidence_from_ci(300.0, 285.0, 315.0) == pytest.approx(0.9)
    assert confidence_from_ci(100.0, 0.0, 300.0) == 0.0
    assert confidence_from_ci(300.0, None, None) is None


def test_shot_from_fit_fills_hit_landing_and_trajectory():
    phys = PhysicsParams()
    p0, v0 = np.array([-4.0, 0.5, 2.8]), np.array([70.0, -2.0, -20.0])
    s = shot_from_fit(_fit(p0, v0, ci=(70.0, 75.0)), phys=phys, shot_type="smash")
    assert s.side == "h" and s.shot_type == "smash" and s.t_hit_s == 1.0
    assert s.speed_kmh == pytest.approx(np.linalg.norm(v0) * 3.6)
    assert s.speed_ci_low_kmh == pytest.approx(252.0) and s.speed_ci_high_kmh == pytest.approx(270.0)
    ref = simulate_until_ground(p0, v0, 1.0, phys)
    assert s.land_x_m == pytest.approx(ref["pos"][-1, 0]) and s.land_y_m == pytest.approx(ref["pos"][-1, 1])
    assert s.t_land_s == pytest.approx(float(ref["t_land"]))
    assert 0.0 < s.land_x_m < 6.7  # 相手のコートに落ちる
    tr = s.trajectory(phys)
    assert np.allclose(tr["pos"][0], p0)


def test_export_batch_from_candidates(tmp_path):
    """一括解析の結果（candidates.json と fit_result.json）から shots.csv を書き出す。"""
    c1 = Candidate(id=1, t_onset=5.0, strength=1.0, start_s=4.8, end_s=5.9, status="ok", t_impact=5.01,
                   hitter={"x": -4.0, "y": 0.5}, landing={"x": 3.0, "y": 0.0}, confirmed=True,
                   result={"kmh": 280.0, "ci_low_kmh": 266.0, "ci_high_kmh": 294.0, "rms_px": 1.0, "n_frames": 9})
    c2 = Candidate(id=2, t_onset=2.0, strength=0.5, start_s=1.8, end_s=2.9, status="ok", t_impact=2.0,
                   hitter={"x": 4.0, "y": 0.0}, landing={"x": -3.0, "y": 1.0},
                   result={"kmh": 200.0, "ci_low_kmh": None, "ci_high_kmh": None})
    c3 = Candidate(id=3, t_onset=8.0, strength=0.5, start_s=7.8, end_s=8.9, status="ok",
                   result={"error": "解析できませんでした"})
    save_candidates(tmp_path, [c1, c2, c3])
    (tmp_path / c1.label).mkdir()
    (tmp_path / c1.label / "fit_result.json").write_text(
        json.dumps(_fit([-4.0, 0.5, 2.8], [75.0, -1.0, -22.0], t_ref=5.01, ci=(73.0, 82.0))), encoding="utf-8")
    shots = read_shots(export_batch(tmp_path, PhysicsParams()))
    # 解析できなかった候補は入らない。時刻の順
    assert [s.t_hit_s for s in shots] == [2.0, 5.01]
    a, b = shots
    # fit_result.json がない候補：打った時刻・側・着地点だけ
    assert a.shot_type == "smash" and a.side == "o" and a.speed_kmh == 200.0 and a.hit_x_m is None
    assert (a.land_x_m, a.land_y_m) == (-3.0, 1.0) and a.confidence is None and a.note
    # ある候補：初速は順位表と同じ値、打点・速度・着地点は当てはめから
    assert b.speed_kmh == 280.0 and b.confidence == pytest.approx(0.9)
    assert (b.hit_x_m, b.hit_z_m) == (-4.0, 2.8) and b.v0_x_mps == 75.0 and b.land_x_m > 0
    assert b.ref == c1.label and b.source == "smash_batch" and b.note == ""
    assert b.trajectory() is not None
