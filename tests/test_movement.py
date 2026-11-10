"""選手の動き（movement）のテスト。合成の足の動き（synth_players）で、移動距離・速さ・いた場所を確かめる。"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from smash_speed.camera import look_at
from smash_speed.config import load_config
from smash_speed.movement import (MOVEMENT_CSV, MOVEMENT_FILE, MOVEMENT_SVG, movement_summary, movement_svg,
                                  heat_grid, synthetic_benchmark, write_movement)
from smash_speed.rally_report import write_report
from smash_speed.shots import Rally, write_rallies, write_shots
from smash_speed.synth_players import footwork_path, observe_players, path_length

FPS = 30.0
CAMERA = look_at(np.array([-9.0, -5.5, 3.0]), np.array([0.0, 0.0, 0.5]), 1500.0, 1920, 1080)


def _standing(points: list[tuple[float, float, float]], fps: float = FPS) -> pd.DataFrame:
    """（始まり [s]、x、y）の所に、次の始まりまで立っている 1 人（最後は 1 秒）。"""
    rows = []
    for k, (t0, x, y) in enumerate(points):
        t1 = points[k + 1][0] if k + 1 < len(points) else t0 + 1.0
        for t in np.arange(t0, t1 - 1e-9, 1 / fps):
            rows.append({"frame": int(round(t * fps)), "t": t, "side": "h", "player": 0, "x": x, "y": y})
    return pd.DataFrame(rows)


def test_synthetic_footwork_distance_and_peak_speed():
    res = synthetic_benchmark(n_seeds=1, fps=FPS)
    for side in ("h", "o"):
        assert res[f"{side}_distance"]["max_abs"] < 0.08, res
        assert res[f"{side}_max_speed"]["max_abs"] < 0.2, res


def test_smoothing_removes_jitter_that_inflates_the_raw_path():
    rng = np.random.default_rng(3)
    t, feet = footwork_path(15.0, FPS, "o", rng)
    players = observe_players(t, {"o": feet}, CAMERA, rng)
    raw = path_length(players[["x", "y"]].to_numpy())
    summary, tracks = movement_summary(players)
    true = path_length(feet)
    assert raw > 1.5 * true   # 点をそのままつなぐと揺れの分だけ長く出る
    assert abs(summary["players"]["o0"]["distance_m"] / true - 1) < 0.08
    assert len(tracks) == len(players)


def test_occluded_feet_jump_is_dropped():
    """足が隠れて 0.4 秒だけ 3 m 遠くへ飛んだコマは、外れとして補い、距離・最高の速さに入れない。"""
    rng = np.random.default_rng(5)
    t, feet = footwork_path(12.0, FPS, "o", rng)
    players = observe_players(t, {"o": feet}, CAMERA, rng)
    clean, _ = movement_summary(players)
    bad = (t > 6.0) & (t < 6.4)
    players.loc[bad, "x"] += 3.0
    summary, tracks = movement_summary(players)
    p, c = summary["players"]["o0"], clean["players"]["o0"]
    assert tracks["outlier"].sum() >= bad.sum() - 2
    assert abs(p["distance_m"] / c["distance_m"] - 1) < 0.05
    assert p["max_speed_mps"] < 7.0


def test_long_gaps_split_the_path():
    # 0〜2 秒は (−3, 0)、位置が出ない 2 秒の後、4〜6 秒は (−5, 2)：抜けをまたぐ 2.8 m は数えない
    a = _standing([(0.0, -3.0, 0.0), (1.0, -3.0, 0.0)])
    b = _standing([(4.0, -5.0, 2.0), (5.0, -5.0, 2.0)])
    summary, tracks = movement_summary(pd.concat([a, b], ignore_index=True))
    p = summary["players"]["h0"]
    assert tracks["segment"].nunique() == 2
    assert p["distance_m"] < 0.05
    assert abs(p["seconds"] - 4.0) < 0.1


def test_zones_heat_grid_and_rallies():
    court = load_config()["court"]
    # 後ろ・中央（−5.5, 0）に 3 秒、前・y− の区画（−1.0, −2.0）に 1 秒、その間を 1 回だけ動く
    players = _standing([(0.0, -5.5, 0.0), (3.0, -1.0, -2.0)])
    summary, tracks = movement_summary(players, court, [Rally(1, t_start_s=2.0, t_end_s=3.9), Rally(2, 10.0)])
    p = summary["players"]["h0"]
    grid = np.array(p["zones"]["grid"])
    assert grid[2, 1] > 0.7 and grid[0, 0] > 0.15            # 深さ（前・中・後）× 幅（y−・中央・y+）
    assert abs(grid.sum() + p["zones"]["outside"] + p["zones"]["other_half"] - 1) < 1e-3
    true = float(np.hypot(4.5, 2.0))
    assert abs(p["distance_m"] / true - 1) < 0.1
    heat, (xa, xb, ya, yb) = heat_grid(tracks, court)
    assert abs(heat.sum() - p["seconds"]) < 1e-6
    assert xa < -court["half_length_m"] and yb > court["half_width_doubles_m"]
    # ラリー 1 の間（1.5〜3.9 秒）に、動いた所が入る。終わりの分からないラリー 2 は数えない
    assert [r["rally_id"] for r in p["rallies"]] == [1]
    assert abs(p["rallies"][0]["distance_m"] / true - 1) < 0.15


def test_write_movement_and_report_section(tmp_path):
    rng = np.random.default_rng(1)
    t, h = footwork_path(8.0, FPS, "h", rng)
    _, o = footwork_path(8.0, FPS, "o", rng)
    observe_players(t, {"h": h, "o": o}, CAMERA, rng).to_csv(tmp_path / "players.csv", index=False)
    write_rallies(tmp_path / "rallies.csv", [Rally(1, t_start_s=1.0, t_end_s=7.0, n_shots=5)])
    write_shots(tmp_path / "shots.csv", [])
    summary = write_movement(tmp_path, court=load_config()["court"])
    assert set(summary["players"]) == {"h0", "o0"}
    assert summary["players"]["h0"]["rally_mean_distance_m"] is not None
    saved = json.loads((tmp_path / MOVEMENT_FILE).read_text(encoding="utf-8"))
    assert saved["players"]["o0"]["distance_m"] == summary["players"]["o0"]["distance_m"]
    assert len(pd.read_csv(tmp_path / MOVEMENT_CSV)) == 2 * len(t)
    ET.fromstring((tmp_path / MOVEMENT_SVG).read_text(encoding="utf-8"))   # SVG として読める

    # レポート：この解析で選手の位置を出したとき（shot_list.json の players.movement）だけ節が入る
    (tmp_path / "shot_list.json").write_text(json.dumps({"players": {}}), encoding="utf-8")
    assert "選手の動き" not in write_report(tmp_path).read_text(encoding="utf-8")
    (tmp_path / "shot_list.json").write_text(json.dumps({"players": {"movement": {"h0": {}}}}), encoding="utf-8")
    html = write_report(tmp_path).read_text(encoding="utf-8")
    assert "選手の動き" in html and "<svg" in html


def test_empty_players():
    summary, tracks = movement_summary(pd.DataFrame(columns=["frame", "t", "side", "player", "x", "y"]))
    assert summary["players"] == {} and tracks.empty
    assert movement_svg(tracks) == ""
