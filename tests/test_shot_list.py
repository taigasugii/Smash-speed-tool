"""動画 1 本のショットの一覧（検出 → 打球 → 軌道と初速 → 種類）のテスト。"""

import json

import numpy as np
import pandas as pd
import pytest

from smash_speed.config import load_config
from smash_speed.hits import Hit, find_hits
from smash_speed.physics import PhysicsParams
from smash_speed.shot_list import (DETECTIONS_FILE, SHOT_LIST_FILE, ShotListOptions, analyze_hits, analyze_video,
                                   main, rally_groups, summarize)
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
    assert any("完了" in m for m in msgs) and any("検出中" in m for m in msgs)


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
