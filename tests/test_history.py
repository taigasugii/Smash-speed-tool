"""セッションをまたいだ記録（history.py）のテスト。"""

import csv
import json
import os

import pytest

from smash_speed.history import (HISTORY_CSV, HISTORY_HTML, SHARE_TYPES, build_history_page, find_sessions,
                                 history_rows, main, share_bars, speed_chart, write_history)
from smash_speed.shots import Rally, Shot, write_rallies, write_shots


def _session(root, name, shots, recorded_at=None, video=None, rallies=None):
    d = root / name
    d.mkdir(parents=True)
    write_shots(d / "shots.csv", shots)
    write_rallies(d / "rallies.csv", rallies or [Rally(1, 0.0, 1.0, len(shots))])
    info = {"video": video or f"/nowhere/{name}.MOV"}
    if recorded_at:
        info["recorded_at"] = recorded_at
    (d / "shot_list.json").write_text(json.dumps(info), encoding="utf-8")
    return d


def _smashes(*kmh):
    return [Shot(t_hit_s=float(i), shot_type="smash", speed_kmh=v) for i, v in enumerate(kmh)]


def test_sessions_are_grouped_by_recorded_day(tmp_path):
    _session(tmp_path, "a_shots", _smashes(300.0, 280.0) + [Shot(t_hit_s=5.0, shot_type="clear", speed_kmh=150.0)],
             "2026-10-01T09:00:00")
    _session(tmp_path, "b_shots", _smashes(320.0) + [Shot(t_hit_s=5.0, shot_type="unknown")], "2026-10-01T18:00:00")
    _session(tmp_path / "sub", "c_shots", _smashes(250.0) + [Shot(t_hit_s=3.0, shot_type="drop", speed_kmh=330.0)],
             "2026-10-05T10:00:00", rallies=[Rally(1, 0.0, 1.0, 4), Rally(2, 5.0, 6.0, 2)])
    # スマッシュの一括解析の出力先（shot_list.json がない）は入れない
    (tmp_path / "x_smashes").mkdir()
    write_shots(tmp_path / "x_smashes" / "shots.csv", _smashes(400.0))

    sessions = find_sessions(tmp_path)
    assert [s.date for s in sessions] == ["2026-10-01", "2026-10-01", "2026-10-05"]
    assert all(s.date_source == "recorded_at" for s in sessions)
    rows = history_rows(sessions)
    assert [r["date"] for r in rows] == ["2026-10-01", "2026-10-05"]
    a, c = rows
    assert a["n_videos"] == 2 and a["n_shots"] == 5 and a["n_rallies"] == 2
    assert a["smash_n"] == 3 and a["smash_max_kmh"] == 320.0 and a["smash_mean_kmh"] == 300.0
    # 割合は種類の分かった 4 本の中で
    assert a["share_smash"] == 0.75 and a["share_clear"] == 0.25 and a["share_drop"] == 0.0
    assert c["max_kmh"] == 330.0 and c["max_type"] == "drop" and c["rally_mean_shots"] == 3.0


def test_date_falls_back_to_the_video_file_time(tmp_path):
    video = tmp_path / "IMG_9.MOV"
    video.write_bytes(b"")
    t = 1767225600  # 2026-01-01 00:00 UTC
    os.utime(video, (t, t))
    _session(tmp_path, "v_shots", _smashes(200.0), video=str(video))
    s, = find_sessions(tmp_path)
    assert s.date_source == "video_mtime" and s.date.startswith(("2025-12-31", "2026-01-01"))
    assert s.video == "IMG_9.MOV"


def test_write_history_and_page(tmp_path, capsys):
    for d in range(4):
        _session(tmp_path, f"d{d}_shots", _smashes(250.0 + 10 * d, 200.0) + [Shot(t_hit_s=9.0, shot_type="net")],
                 f"2026-10-{d + 10}T10:00:00")
    rows, csv_path, html_path = write_history(tmp_path)
    assert len(rows) == 4 and csv_path.name == HISTORY_CSV and html_path.name == HISTORY_HTML
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        back = list(csv.DictReader(f))
    assert [r["date"] for r in back] == [r["date"] for r in rows] and back[3]["smash_max_kmh"] == "280.0"
    html = html_path.read_text(encoding="utf-8")
    assert html.count('class="chart"') == 2  # 広い画面用とスマホ用
    assert "prefers-color-scheme: dark" in html and '<meta name="viewport"' in html
    assert html.count('class="row"') == 4
    main([str(tmp_path)])
    assert "4 日分" in capsys.readouterr().out


def test_charts_handle_missing_values():
    rows = [{"date": "2026-10-01", "smash_n": 0, "smash_max_kmh": None, "smash_mean_kmh": None,
             **{f"share_{k}": None for k in SHARE_TYPES}}]
    assert speed_chart(rows) == "" and share_bars(rows) == ""
    page = build_history_page([])
    assert "<h2>日ごとの数字</h2>" in page


@pytest.mark.parametrize("n", [1, 12])
def test_speed_chart_points(n):
    rows = [{"date": f"2026-10-{i + 1:02d}", "smash_n": 3, "smash_max_kmh": 300.0 + i, "smash_mean_kmh": 250.0}
            for i in range(n)]
    svg = speed_chart(rows)
    assert svg.count('<g class="day">') == n and svg.count("<circle") == 2 * n
