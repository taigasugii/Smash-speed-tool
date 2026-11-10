"""ラリーの統計と着地点の分布図（rally_stats.py）のテスト。"""

import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from smash_speed.config import load_config
from smash_speed.physics import PhysicsParams
from smash_speed.rally_stats import (LANDING_FILE, STATS_FILE, format_stats, landing_svg, landing_zones, main,
                                     rally_stats, type_table, write_stats, zone_of)
from smash_speed.shots import Rally, Shot, write_rallies, write_shots
from smash_speed.synth_rally import make_session

CFG = load_config()
COURT = CFG["court"]


def _shots():
    # ラリー 1：サーブ・クリア・スマッシュ（3 本、0〜4 秒）、ラリー 2：サーブ・ネット（2 本、10〜12 秒）
    return [
        Shot(t_hit_s=0.0, side="h", shot_type="serve", speed_kmh=60.0, land_x_m=5.0, land_y_m=0.5, rally_id=1,
             index_in_rally=1),
        Shot(t_hit_s=1.5, side="o", shot_type="clear", speed_kmh=150.0, land_x_m=-6.0, land_y_m=-2.5, rally_id=1,
             index_in_rally=2),
        Shot(t_hit_s=3.0, side="h", shot_type="smash", speed_kmh=300.0, land_x_m=3.0, land_y_m=2.5, rally_id=1,
             index_in_rally=3),
        Shot(t_hit_s=10.0, side="h", shot_type="serve", speed_kmh=40.0, land_x_m=2.2, land_y_m=-0.5, rally_id=2,
             index_in_rally=1),
        Shot(t_hit_s=11.0, side="o", shot_type="unknown", rally_id=2, index_in_rally=2),
    ]


def _rallies():
    return [Rally(1, 0.0, 4.0, 3, winner_side="h", end_reason="in"), Rally(2, 10.0, 12.0, 2, end_reason="")]


def test_rally_lengths_and_outcomes():
    st = rally_stats(_shots(), _rallies(), COURT)
    assert st["n_rallies"] == 2 and st["n_shots"] == 5 and st["n_shots_with_speed"] == 4
    assert st["rally_shots"] == {"n": 2, "mean": 2.5, "median": 2.5, "max": 3.0}
    assert st["rally_seconds"]["mean"] == 3.0 and st["rally_seconds"]["max"] == 4.0
    hist = {h["label"]: h["n"] for h in st["rally_length_hist"]}
    assert hist["1〜2 本"] == 1 and hist["3〜4 本"] == 1 and sum(hist.values()) == 2
    assert st["longest_rally"]["rally_id"] == 1 and st["longest_rally"]["seconds"] == 4.0
    assert st["end_reasons"]["in"]["n"] == 1 and st["end_reasons"][""]["n"] == 1
    assert st["points_won"] == {"h": 1}
    json.dumps(st)  # JSON に書ける


def test_type_shares_and_speeds():
    t = type_table(_shots())
    # 割合は種類が分かった 4 本の中で。分からない（unknown）は本数だけ
    assert t["serve"]["n"] == 2 and t["serve"]["share"] == 0.5
    assert t["serve"]["mean_kmh"] == 50.0 and t["serve"]["max_kmh"] == 60.0
    assert t["unknown"]["share"] is None and t["unknown"]["mean_kmh"] is None
    assert "drop" not in t
    by_side = rally_stats(_shots(), _rallies(), COURT)["types_by_side"]
    assert by_side["h"]["smash"]["n"] == 1 and "smash" not in by_side["o"]


@pytest.mark.parametrize("x, y, expected", [
    (1.0, 0.0, ("o", 0, 1)), (-6.5, -3.0, ("h", 2, 0)), (3.0, 2.9, ("o", 1, 2)), (7.0, 0.0, None), (2.0, 3.2, None),
])
def test_zone_of(x, y, expected):
    assert zone_of(x, y, COURT) == expected


def test_landing_zones_count_by_type_and_side():
    z = landing_zones(_shots() + [Shot(t_hit_s=20.0, shot_type="smash", land_x_m=8.0, land_y_m=0.0)], COURT)
    assert np.array(z["by_type"]["serve"]["o"]).sum() == 2
    assert z["by_type"]["clear"]["h"][2][0] == 1
    assert z["out"] == {"smash": 1}


def test_landing_svg_has_one_court_per_type():
    svg = landing_svg(_shots(), COURT)
    root = ET.fromstring(svg)  # 正しい XML
    ns = {"s": "http://www.w3.org/2000/svg"}
    titles = [t.text for t in root.findall(".//s:text", ns) if t.get("class") == "title"]
    assert titles == ["サーブ（2 本）", "スマッシュ（1 本）", "クリア（1 本）"]  # 多い順、同じなら SHOT_TYPES の順
    assert len(root.findall(".//s:circle", ns)) == 4
    # 暗い画面の色もある
    assert "prefers-color-scheme: dark" in svg and 'data-theme="dark"' in svg
    assert landing_svg([Shot(t_hit_s=0.0)], COURT) == ""


def test_write_stats_and_cli_on_a_synthetic_session(tmp_path, capsys):
    phys = PhysicsParams.from_config(CFG)
    session = make_session(np.random.default_rng(3), phys, n_rallies=3)
    shots, rallies = [], []
    for i, r in enumerate(session, start=1):
        shots += r.to_shots(rally_id=i, phys=phys)
        rallies.append(r.to_rally(rally_id=i))
    write_shots(tmp_path / "shots.csv", shots)
    write_rallies(tmp_path / "rallies.csv", rallies)
    main([str(tmp_path)])
    out = capsys.readouterr().out
    assert "ラリー 3 個" in out
    st = json.loads((tmp_path / STATS_FILE).read_text(encoding="utf-8"))
    assert st["n_rallies"] == 3 and st["n_shots"] == len(shots)
    assert st["rally_shots"]["max"] == max(len(r.segments) for r in session)
    assert sum(v["n"] for v in st["types"].values()) == len(shots)
    assert abs(sum(v["share"] for v in st["types"].values()) - 1.0) < 1e-3
    # 合成ラリーの着地点はどれもコートの中
    assert st["landing"]["out"] == {} and len(st["landing"]["points"]) == len(shots)
    assert (tmp_path / LANDING_FILE).exists()
    assert write_stats(tmp_path)["n_shots"] == len(shots)
    assert "種類：" in format_stats(st)
