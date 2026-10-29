"""動画 1 本のショットの一覧（検出 → 打球 → 軌道と初速 → 種類）のテスト。"""

import json

import numpy as np
import pandas as pd
import pytest

from smash_speed.config import load_config
from smash_speed.hits import Hit, find_hits
from smash_speed.physics import PhysicsParams
from smash_speed.shot_list import (DETECTIONS_FILE, SHOT_LIST_FILE, ShotListOptions, TrackFilter, analyze_hits,
                                   analyze_video, court_depth_range, main, one_per_frame, rally_groups,
                                   select_tracks, shuttle_tracks, summarize, track_table)
from smash_speed.shots import RALLIES_FILE, SHOTS_FILE, Shot, read_rallies, read_shots
from smash_speed.synth_rally import make_rally, observe_rally, rally_camera, render_rally_video

CFG = load_config()
PHYS = PhysicsParams.from_config(CFG)
CAM = rally_camera(CFG)


def _match(est, truth, tol_s=0.05):
    """正解のショットごとに、時刻がいちばん近い推定のショット（なければ None）。"""
    out = []
    for s in truth:
        d = [abs(e.t_hit_s - s.t_hit_s) for e in est]
        out.append(est[int(np.argmin(d))] if d and min(d) <= tol_s else None)
    return out


def test_rally_groups_split_at_long_gaps():
    hits = [Hit(t) for t in (1.0, 1.8, 2.5, 7.0, 7.9, 20.0)]
    assert rally_groups(hits, 3.0) == [[0, 1, 2], [3, 4], [5]]
    assert rally_groups([], 3.0) == []


def test_summarize_counts_and_speeds_by_type():
    shots = [Shot(1.0, shot_type="smash", speed_kmh=200.0), Shot(2.0, shot_type="smash", speed_kmh=250.0),
             Shot(3.0, shot_type="net"), Shot(4.0, shot_type="clear", speed_kmh=150.0)]
    s = summarize(shots)
    assert list(s) == ["smash", "clear", "net"]
    assert s["smash"] == {"label": "スマッシュ", "n": 2, "mean_kmh": 225.0, "max_kmh": 250.0}
    assert s["net"]["n"] == 1 and s["net"]["max_kmh"] is None


def test_two_rallies_are_fitted_separately():
    """2 つのラリーの検出が 1 本につながっていても、ラリーごとに分けて当てはめ、種類と初速が出る。"""
    rng = np.random.default_rng(11)
    r1 = make_rally(rng, PHYS, n_shots=5, t_start=0.5)
    r2 = make_rally(rng, PHYS, n_shots=5, server_side="o", t_start=r1.t_end + 5.0)
    fps = 60.0
    det = pd.concat([observe_rally(r, CAM, fps, PHYS, np.random.default_rng(k)) for k, r in enumerate((r1, r2))],
                    ignore_index=True)
    det["frame"] = np.arange(len(det))
    segs = r1.segments + r2.segments
    onsets = [(s.t_hit + float(np.linalg.norm(CAM.center - s.p0)) / 343.0, 30.0) for s in segs]
    hits = find_hits(det, fps, onsets, camera=CAM)
    shots = analyze_hits(det, hits, CAM, PHYS, CFG, ci=False, progress=lambda m: None)
    assert [s.t_hit_s for s in shots] == sorted(s.t_hit_s for s in shots)
    assert {s.rally_id for s in shots} == {1, 2}
    truth = r1.to_shots(0.0, phys=PHYS) + r2.to_shots(0.0, phys=PHYS)
    found = _match(shots, truth)
    assert sum(f is not None for f in found) >= len(truth) - 1
    # ラリーの最初はサーブ、ラリーの番号は正解のラリーと同じ
    for f, s, r in zip(found, truth, [1] * len(r1.segments) + [2] * len(r2.segments)):
        if f is not None:
            assert f.rally_id == r
            assert (f.shot_type == "serve") == (s.shot_type == "serve")
    ok = [f.shot_type == s.shot_type for f, s in zip(found, truth) if f is not None]
    assert np.mean(ok) >= 0.85
    err = [abs(f.speed_kmh / s.speed_kmh - 1) for f, s in zip(found, truth) if f is not None and f.speed_kmh]
    assert len(err) >= len(truth) - 2 and np.median(err) < 0.05


@pytest.mark.parametrize("kw, why", [({"max_kmh": 1.0}, "ありえない速さ"), ({"max_rms_frac": 0.0}, "合わない")])
def test_failed_fits_are_dropped(kw, why):
    """ありえない初速・画像の上で検出と合わない軌道は、速さが出なかったもの（種類は不明）として扱う。"""
    rally = make_rally(np.random.default_rng(2), PHYS, n_shots=3, t_start=0.5)
    det = observe_rally(rally, CAM, 30.0, PHYS, np.random.default_rng(0))
    hits = find_hits(det, 30.0, None, camera=CAM)
    shots = analyze_hits(det, hits, CAM, PHYS, CFG, ci=False, progress=lambda m: None, **kw)
    assert shots and all(s.speed_kmh is None and s.v0 is None for s in shots)
    assert all(s.shot_type == "unknown" and why in s.note for s in shots)


def test_simulate_skips_absurd_speeds_quickly():
    """最適化の途中の極端な初速（数十万 m/s）は積分せずに NaN を返す（以前は 1 回の当てはめが何十分もかかった）。"""
    import time

    from smash_speed.physics import simulate

    t0 = time.time()
    pos, vel = simulate(np.array([0.0, 0.0, 2.0]), np.array([4e5, 0.0, 0.0]), 0.0, np.linspace(0, 0.3, 10), PHYS)
    assert time.time() - t0 < 0.5
    assert np.isnan(pos).all() and np.isnan(vel).all()
    pos, _ = simulate(np.array([0.0, 0.0, 2.0]), np.array([90.0, 0.0, 0.0]), 0.0, np.linspace(0, 0.3, 10), PHYS)
    assert np.isfinite(pos).all()


def _track(tid, t, uv, conf=0.9):
    """検出の行（重なりのある形）。"""
    return pd.DataFrame({"frame": np.round(t * 30.0).astype(int), "timestamp": t, "x_px": uv[:, 0], "y_px": uv[:, 1],
                         "visible": 1, "confidence": conf, "track_id": tid})


def test_court_depth_range_finds_points_in_the_court_space():
    flt = TrackFilter()
    inside = CAM.project(np.array([[0.0, 0.0, 2.0], [-6.0, 2.5, 0.5], [5.0, -2.0, 8.0]]))
    lo, hi = court_depth_range(CAM, inside, CFG, flt)
    assert (hi > lo).all() and (lo >= 0).all()
    # 高さ 40 m の点の視線は、コートの空間（高さ 12 m まで）の上を通る
    lo, hi = court_depth_range(CAM, CAM.project(np.array([[0.0, 0.0, 40.0]])), CFG, flt)
    assert not (hi > lo).any()


def test_shuttle_tracks_keep_the_rally_and_drop_clutter():
    """合成ラリーのシャトルは全部残り、止まった物・飛び移る軌道・ゆっくり歩く人・コートの空間の外は落ちる。"""
    rally = make_rally(np.random.default_rng(3), PHYS, n_shots=6, t_start=0.5)
    obs = observe_rally(rally, CAM, 30.0, PHYS, np.random.default_rng(1))
    obs = obs[obs.visible == 1]
    shuttle = _track(0, obs.timestamp.to_numpy(), obs[["x_px", "y_px"]].to_numpy(), conf=0.8)
    shuttle["frame"] = obs.frame.to_numpy()
    t = np.arange(30, 90) / 30.0
    rng = np.random.default_rng(0)
    floor = CAM.project(np.array([[2.0, 1.0, 1.0]]))[0]
    still = _track(1, t, floor + rng.normal(0, 1.0, (len(t), 2)), conf=0.99)
    jump = _track(2, t[:8], np.where(np.arange(8)[:, None] % 2 == 0, floor, floor + [CAM.width * 0.4, 0]), conf=0.99)
    walk = _track(3, t, CAM.project(np.c_[np.linspace(-3, -2, len(t)), np.zeros(len(t)), np.full(len(t), 1.2)]),
                  conf=0.99)
    sky = _track(4, t, CAM.project(np.c_[np.linspace(-10, 10, len(t)), np.zeros(len(t)), np.full(len(t), 40.0)]),
                 conf=0.99)
    for d in (still, jump, walk, sky):
        d["frame"] = obs.frame.iloc[0] + np.arange(len(d))
        d["timestamp"] = obs.timestamp.iloc[0] + np.arange(len(d)) / 30.0
    det = pd.concat([shuttle, still, jump, walk, sky], ignore_index=True)
    tab = track_table(det, CAM, CFG).set_index("track_id")
    assert tab.keep.to_dict() == {0: True, 1: False, 2: False, 3: False, 4: False}
    assert tab.loc[2, "image_speed"] > 1.2 and tab.loc[3, "speed_max_mps"] < 3.0 and tab.loc[4, "inside"] == 0.0
    out, _ = shuttle_tracks(det, CAM, CFG)
    # 1 コマ 1 点で、シャトルの点はすべて残る（ほかの物の確からしさの方が高くても）
    assert out.frame.is_unique and set(out.track_id[out.visible == 1]) == {0}
    assert int(out.visible.sum()) == len(shuttle)


def _tab(rows):
    """``track_table`` の形の表（track_id, n, t_start, t_end, edge_start。すべて keep）。"""
    tab = pd.DataFrame(rows, columns=["track_id", "n", "t_start", "t_end", "edge_start"])
    tab["keep"] = True
    return tab


def test_select_tracks_drops_a_track_overlapping_the_rally():
    """同じ時間に重なった軌道からは、重ならない組で点の数の合計が多いもの（ラリーの続き）を選ぶ。"""
    # 0・1 はラリーのシャトル（打球の前と後）。2 は両方に重なる隣のコートのシャトル。3 は少しだけ重なる続き。
    # 4 はほかと重ならない（短くても残す）
    tab = _tab([(0, 13, 0.0, 0.40, False), (1, 12, 0.47, 0.83, False), (2, 11, 0.17, 0.53, False),
                (3, 10, 0.80, 1.10, False), (4, 5, 2.0, 2.15, False)])
    assert list(select_tracks(tab)) == [True, True, False, True, True]
    # 画面の左右の端から始まった軌道は軽い（点が多くても、重なったラリーの軌道に負ける）
    tab = _tab([(0, 9, 0.0, 0.27, False), (1, 12, 0.03, 0.40, True)])
    assert list(select_tracks(tab)) == [True, False]
    tab = _tab([(0, 9, 0.0, 0.27, False), (1, 12, 0.03, 0.40, False)])
    assert list(select_tracks(tab)) == [False, True]
    # keep でない軌道は選ばない
    tab.loc[1, "keep"] = False
    assert list(select_tracks(tab)) == [True, False]


def test_shuttle_tracks_skip_a_neighbor_track_with_higher_confidence():
    """確からしさの高い隣のコートの軌道が打球の所に重なっても、ラリーの軌道の点が 1 コマ 1 点に残る。"""
    t = np.arange(0, 24) / 30.0
    before = _track(0, t[:12], CAM.project(np.c_[np.linspace(4, -4, 12), np.zeros(12), np.linspace(5, 2.5, 12)]))
    after = _track(1, t[13:], CAM.project(np.c_[np.linspace(-4, 3, 11), np.zeros(11), np.linspace(2.5, 0.5, 11)]))
    W = CAM.width
    left = CAM.project(np.array([[-6.0, -1.0, 2.0]]))[0]
    side = _track(2, t[6:18], np.c_[np.linspace(0.005 * W, 0.12 * W, 12), np.full(12, left[1])], conf=0.99)
    det = pd.concat([before, after, side], ignore_index=True)
    # 合成のカメラでは画面の端がコートの空間の外なので、その条件は外して選び方だけを確かめる
    out, tab = shuttle_tracks(det, CAM, CFG, TrackFilter(min_inside=0.0, min_speed_mps=0.0))
    tab = tab.set_index("track_id")
    assert tab.keep.all() and bool(tab.loc[2, "edge_start"]) and not tab.loc[2, "selected"]
    assert set(out.track_id[out.visible == 1]) == {0, 1} and int(out.visible.sum()) == 23


def test_one_per_frame_prefers_the_more_confident_track():
    t = np.arange(10) / 30.0
    a = _track(0, t, np.c_[np.arange(10) * 10.0, np.zeros(10)], conf=0.5)
    b = _track(1, t[5:], np.c_[np.arange(5) * 10.0, np.ones(5)], conf=0.9)
    out = one_per_frame(pd.concat([a, b], ignore_index=True))
    assert list(out.track_id) == [0] * 5 + [1] * 5
    out = one_per_frame(pd.concat([a, b], ignore_index=True), keep=pd.Series([True] * 10 + [False] * 5))
    assert list(out.track_id) == [0] * 10
    # 点を使わないコマも、見えないコマとして残る
    out = one_per_frame(a, keep=pd.Series([True] * 5 + [False] * 5))
    assert len(out) == 10 and list(out.visible) == [1] * 5 + [0] * 5 and (out.track_id[5:] == -1).all()


@pytest.fixture(scope="module")
def rally_video(tmp_path_factory):
    d = tmp_path_factory.mktemp("rally")
    rally = make_rally(np.random.default_rng(6), PHYS, n_shots=5)
    render_rally_video(rally, CAM, CFG, PHYS, d / "rally.mp4", fps=30.0, pre_s=0.3, audio_path=d / "rally.wav")
    (d / "calibration.json").write_text(json.dumps({"camera": CAM.to_dict()}), encoding="utf-8")
    return d, rally.to_shots(0.3 - rally.t_start, phys=PHYS)


def test_analyze_video_writes_the_shot_list(rally_video, tmp_path):
    d, truth = rally_video
    out = tmp_path / "list"
    msgs = []
    info = analyze_video(ShotListOptions(d / "rally.mp4", d / "calibration.json", out, audio=d / "rally.wav"),
                         CFG, progress=msgs.append)
    for name in (SHOTS_FILE, RALLIES_FILE, DETECTIONS_FILE, SHOT_LIST_FILE):
        assert (out / name).exists()
    shots = read_shots(out / SHOTS_FILE)
    assert len(read_rallies(out / RALLIES_FILE)) == 1
    found = _match(shots, truth)
    assert all(f is not None for f in found)
    assert [f.shot_type for f in found] == [s.shot_type for s in truth]
    for f, s in zip(found, truth):
        # サーブは前の軌道がなく打点の時刻を直せないので、30 fps では大きくずれる（ROADMAP の提案）
        assert f.speed_kmh == pytest.approx(s.speed_kmh, rel=0.35 if s.shot_type == "serve" else 0.12)
        assert f.side == s.side
        assert f.source == "shot_list" and f.confidence is not None
    saved = json.loads((out / SHOT_LIST_FILE).read_text(encoding="utf-8"))
    assert saved["used_audio"] and saved["n_hits"] == len(shots) == info["n_hits"]
    assert sum(v["n"] for v in saved["by_type"].values()) == len(shots)
    assert saved["fps"] == pytest.approx(30.0)
    # 合成の動画にはシャトルしか動く物がないので、軌道は落とされない
    assert saved["n_tracks_kept"] == saved["n_tracks"] >= 1
    assert any("完了" in m for m in msgs) and any("検出中" in m for m in msgs)
    # ラリーの統計と着地点の分布図（C2）
    stats = json.loads((out / "stats.json").read_text(encoding="utf-8"))
    assert stats["n_shots"] == len(shots) and stats["n_rallies"] == 1
    assert stats["rally_shots"]["max"] == len(shots)
    assert (out / "landing.svg").read_text(encoding="utf-8").startswith("<svg")
    # 1 枚のレポート（C3）
    assert "<h2>ラリーの一覧</h2>" in (out / "report.html").read_text(encoding="utf-8")
    assert "recorded_at" in saved


def test_cli_without_audio(rally_video, tmp_path, capsys):
    """音を使わなくても（軌道だけで打球を見つけて）一覧が出る。"""
    d, truth = rally_video
    main([str(d / "rally.mp4"), "--calib", str(d / "calibration.json"), "--out", str(tmp_path), "--no-audio"])
    saved = json.loads((tmp_path / SHOT_LIST_FILE).read_text(encoding="utf-8"))
    assert not saved["used_audio"] and saved["n_onsets"] is None
    shots = read_shots(tmp_path / SHOTS_FILE)
    found = _match(shots, truth)
    assert sum(f is not None for f in found) >= len(truth) - 1
    assert "本" in capsys.readouterr().out
