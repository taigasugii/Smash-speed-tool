"""解析用の表（form_table.py、E7）と、動画のショットごとの打球の結果（side_clear.analyze_balls、E6）のテスト。"""

import numpy as np
import pandas as pd
import pytest

from smash_speed.form import FORM_DIR, FormOptions, analyze, read_rows
from smash_speed.form_score import main as score_main
from smash_speed.form_table import ANALYSIS_FILE, COLUMN_INFO, COLUMNS, COLUMNS_FILE, build_table, main, write_table
from smash_speed.physics import PhysicsParams, simulate
from smash_speed.pose import KP
from smash_speed.side_clear import BALL_FILE, analyze_balls
from smash_speed.synth_pose import joints_3d
from tests.test_form import CAM, FPS, INFO, SWINGS, SessionModel, _frames

SPEED = {5.0: 25.0, 10.0: 30.0, 15.0: 35.0}   # スイングごとのシャトルの初速 [m/s]


def _detect(t0, t1):
    """スイングの打点で手首の少し上から 40° に打ち出したクリアを、合成の側方のカメラで見た検出。"""
    phys = PhysicsParams()
    times = np.arange(np.ceil(t0 * FPS), np.floor(t1 * FPS) + 1) / FPS
    rows = []
    for c, p in SWINGS.items():
        if not (t0 <= c <= t1):
            continue
        p0 = joints_3d(p, np.array([0.0]))[0, KP["right_wrist"]] + [0.0, 0.0, 0.5]
        v0 = SPEED[c] * np.array([np.cos(np.radians(40)), 0.0, np.sin(np.radians(40))])
        tt = times[times >= c]
        pos, _ = simulate(p0, v0, c, tt, phys)
        uv = CAM.project(pos)
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < CAM.width) & (uv[:, 1] >= 0) & (uv[:, 1] < CAM.height)
        rows += [{"timestamp": t, "x_px": u, "y_px": v, "visible": 1} for t, (u, v) in zip(tt[ok], uv[ok])]
    return pd.DataFrame(rows, columns=["timestamp", "x_px", "y_px", "visible"])


@pytest.fixture()
def form_dir(tmp_path):
    analyze("clear_01.mp4", tmp_path, FormOptions(), frames=_frames, info=INFO,
            onsets=[(c + 7.0 / 343.0, 100.0) for c in SWINGS], model=SessionModel(SWINGS), progress=lambda s: None)
    return tmp_path


def test_analyze_balls_from_the_body_scale(form_dir):
    rows = analyze_balls(form_dir, stature_m=1.70, hitter_from_net_m=5.5, detect_fn=_detect)
    assert len(rows) == 3 and (form_dir / FORM_DIR / BALL_FILE).exists()
    for r, c in zip(rows, sorted(SWINGS)):
        assert r["ball_note"] == ""
        assert r["ball_speed_kmh"] == pytest.approx(SPEED[c] * 3.6, rel=0.1)
        assert r["ball_elevation_deg"] == pytest.approx(40.0, abs=4.0)
        assert r["land_depth_m"] == pytest.approx(r["land_from_hitter_m"] - 5.5)
        assert 0 < r["ball_in_frame_s"] < 0.3   # 横向きのカメラでは、すぐに画面の上へ出る
    # 速く打ったショットほど初速が大きく、遠くへ飛ぶ
    assert rows[0]["ball_speed_kmh"] < rows[1]["ball_speed_kmh"] < rows[2]["ball_speed_kmh"]
    assert rows[0]["land_from_hitter_m"] < rows[2]["land_from_hitter_m"]
    # シャトルが見つからなければ、理由を書いて空にする
    empty = analyze_balls(form_dir, 1.70, detect_fn=lambda a, b: _detect(a, b).iloc[:0])
    assert empty[0].get("ball_speed_kmh") is None
    assert "当てはめられません" in empty[0]["ball_note"]


def test_table_joins_features_scores_and_ball(form_dir):
    poses = form_dir / FORM_DIR / "poses"
    ref = form_dir / "ref.npz"
    score_main(["make-ref", str(ref), str(poses / "clear_01_000500.npz"), "--name", "mine"])
    score_main(["score", str(form_dir), "--ref", str(ref)])
    analyze_balls(form_dir, stature_m=1.70, detect_fn=_detect)
    path = write_table(form_dir)
    assert path.name == ANALYSIS_FILE
    rows = read_rows(path)
    assert [r["shot_id"] for r in rows] == ["clear_01_000500", "clear_01_001000", "clear_01_001500"]
    assert rows[0]["score_dtw"] == pytest.approx(100.0, abs=0.01) and rows[0]["reference"] == "mine"
    assert rows[1]["score_dtw"] < 95                         # 肘を曲げたスイング
    assert rows[2]["score_dtw"] > 99                         # お手本と同じ形
    assert rows[1]["elbow_angle_deg"] == pytest.approx(140.0, abs=3.0)
    assert rows[1]["ball_speed_kmh"] == pytest.approx(108.0, rel=0.1)
    assert rows[0]["land_depth_m"] is None                   # ネットからの距離を渡していない
    header = path.read_text(encoding="utf-8-sig").splitlines()[0].split(",")
    assert header == COLUMNS
    desc = read_rows(form_dir / FORM_DIR / COLUMNS_FILE)
    assert [d["column"] for d in desc] == COLUMNS and all(d["description"] for d in desc)
    assert set(COLUMNS) <= set(COLUMN_INFO)


def test_table_without_scores_or_ball(form_dir):
    rows = build_table(form_dir)
    assert len(rows) == 3 and "score_dtw" not in rows[0] and "ball_speed_kmh" not in rows[0]
    assert main([str(form_dir)]) == 0
    back = read_rows(form_dir / FORM_DIR / ANALYSIS_FILE)
    assert back[0]["score_dtw"] is None and back[0]["ball_speed_kmh"] is None
