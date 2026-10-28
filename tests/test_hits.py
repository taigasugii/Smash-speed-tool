"""すべての打球を見つける（hits.py）のテスト。"""

import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile

from smash_speed.config import load_config
from smash_speed.events import audio_onsets
from smash_speed.hits import (Hit, _most_likely_sides, find_hits, hits_to_shots, match_hits, synthetic_benchmark,
                              track_breaks)
from smash_speed.physics import PhysicsParams
from smash_speed.synth import write_click_audio
from smash_speed.synth_rally import make_rally, observe_rally, rally_camera

PHYS = PhysicsParams()


@pytest.fixture(scope="module")
def cam():
    return rally_camera(load_config())


def _det(t, uv):
    t, uv = np.asarray(t, float), np.asarray(uv, float)
    return pd.DataFrame({"frame": np.arange(len(t)), "timestamp": t, "x_px": uv[:, 0], "y_px": uv[:, 1],
                         "visible": 1, "confidence": 1.0})


def _rally_audio(rally, cam, tmp_path, t0, extra=()):
    """打つたびのクリック音（音の遅れつき）と ``extra`` の音を入れて、``audio_onsets`` で拾った結果（映像の時刻）。"""
    clicks = [(s.t_hit - t0 + np.linalg.norm(cam.center - s.p0) / 343.0, 0.5) for s in rally.segments]
    clicks += [(t - t0, a) for t, a in extra]
    write_click_audio(tmp_path / "a.wav", rally.t_end - t0 + 0.5, clicks, 48000, 0)
    rate, x = wavfile.read(tmp_path / "a.wav")
    return [(t + t0, s) for t, s in audio_onsets(x.astype(float) / 32768.0, rate)]


@pytest.mark.parametrize("fps", [30.0, 120.0])
def test_track_only_finds_every_hit_and_side(cam, fps):
    """音がなくても、軌道の向きが変わる所から、合成ラリーのすべての打球と打った側が出る。"""
    rng = np.random.default_rng(7)
    total = {"tp": 0, "fp": 0, "fn": 0, "side": 0}
    for _ in range(3):
        rally = make_rally(rng, PHYS, n_shots=8)
        det = observe_rally(rally, cam, fps, PHYS, rng)
        hits = find_hits(det, fps, camera=cam)
        m = match_hits(hits, [s.t_hit for s in rally.segments], [s.side for s in rally.segments])
        total["tp"] += m["tp"]
        total["fp"] += m["fp"]
        total["fn"] += m["fn"]
        total["side"] += m["side_correct"]
        assert m["mean_abs_dt_s"] < (0.012 if fps < 60 else 0.005)
    assert total["fn"] <= 1 and total["fp"] <= 1
    assert total["side"] >= total["tp"] - 1


def test_audio_sets_the_time_and_drops_clicks_without_a_hit(cam, tmp_path):
    """打球音は軌道の変化と組み合わせて時刻に使い（音の遅れは見積もって引く）、
    ラリーの外・シャトルの向きが変わっていない所の音は捨てる。"""
    rally = make_rally(np.random.default_rng(3), PHYS, n_shots=8)
    fps = 60.0
    det = observe_rally(rally, cam, fps, PHYS, np.random.default_rng(1))
    segs = rally.segments
    # ラリーの前・後と、飛んでいる途中（打球と打球のまん中）の音
    mid = 0.5 * (segs[1].t_hit + segs[2].t_hit)
    extra = [(rally.t_start - 0.25, 0.5), (rally.t_end + 0.3, 0.5), (mid, 0.5)]
    onsets = _rally_audio(rally, cam, tmp_path, rally.t_start - 0.4, extra)
    assert len(onsets) == len(segs) + 3
    hits = find_hits(det, fps, onsets, camera=cam)
    m = match_hits(hits, [s.t_hit for s in segs], [s.side for s in segs], tol_s=0.02)
    assert m["fp"] == 0 and m["fn"] == 0
    assert m["side_correct"] == len(segs)
    # 音が届くまでの遅れ（約 45 ms）を引いて、打った時刻に 10 ms 以内で合う
    assert m["mean_abs_dt_s"] < 0.01
    assert sum(h.method == "audio+track" for h in hits) >= len(segs) - 1
    # 打球音の時刻（遅れを引く前）は、音を使った打球だけに残る（shot_fit が打点からの距離で遅れを引き直す）
    for h in hits:
        if h.method.startswith("audio"):
            assert h.t_audio is not None and 0.0 < h.t_audio - h.t < 0.1
        else:
            assert h.t_audio is None


def test_audio_only_without_detections(tmp_path, cam):
    rally = make_rally(np.random.default_rng(4), PHYS, n_shots=5)
    onsets = _rally_audio(rally, cam, tmp_path, rally.t_start - 0.4)
    hits = find_hits(None, 30.0, onsets)
    assert [h.method for h in hits] == ["audio"] * len(rally.segments)
    assert all(h.side == "" for h in hits)


def test_landing_and_frame_edge_are_not_hits():
    """床に落ちて止まった所・画面の上から出て戻ってきた所は、打球にしない（音がないとき）。"""
    fps = 60.0
    # 右へ飛んで、床で止まる
    t = np.arange(0, 1.5, 1 / fps)
    x = np.where(t < 1.0, 200 + 600 * t, 800.0)
    y = np.where(t < 1.0, 300 + 400 * t ** 2, 700.0)
    br = track_breaks(_det(t, np.c_[x, y]), fps, noise_px=1.0, frame_size=(1920, 1080))
    assert [b.kind for b in br] == ["track_start"]
    # 上へ飛んで画面の上から出て、0.5 秒後に下りてくる（向きは逆だが打球ではない）
    t1 = np.arange(0, 0.4, 1 / fps)
    t2 = np.arange(0.9, 1.4, 1 / fps)
    up = np.c_[500 + 200 * t1, 200 - 500 * t1]
    down = np.c_[500 + 200 * t2, 500 * (t2 - 0.9)]
    br = track_breaks(_det(np.r_[t1, t2], np.r_[up, down]), fps, noise_px=1.0, frame_size=(1920, 1080))
    assert [b.kind for b in br] == ["track_start", "track_edge"]
    hits = find_hits(_det(np.r_[t1, t2], np.r_[up, down]), fps, noise_px=1.0, frame_size=(1920, 1080))
    assert len(hits) == 1 and hits[0].method == "track_start"


def test_reversal_inside_a_gap_is_a_hit():
    """検出が途切れた間に向きが変わっていれば、その間に打球を置く。"""
    fps = 30.0
    t1 = np.arange(0, 0.5, 1 / fps)
    t2 = np.arange(0.8, 1.3, 1 / fps)
    uv = np.r_[np.c_[300 + 800 * t1, 500 + 0 * t1], np.c_[820 - 800 * (t2 - 0.65), 500 + 0 * t2]]
    br = track_breaks(_det(np.r_[t1, t2], uv), fps, noise_px=1.0)
    assert [b.kind for b in br] == ["track_start", "track_gap"]
    assert br[1].t == pytest.approx(0.65, abs=0.05)


def test_sides_alternate_but_follow_strong_votes():
    # 票が 1 つ弱く間違っていても、交互の並びを選ぶ
    assert _most_likely_sides([0.8, -0.7, -0.1, -0.8, 0.9], 0.9) == ["h", "o", "h", "o", "h"]
    # 打球を 1 つ見落として同じ側が続く所は、強い票に従う
    assert _most_likely_sides([0.9, -0.9, -0.9, 0.9], 0.9) == ["h", "o", "o", "h"]
    assert _most_likely_sides([0.0, 0.0], 0.9) == ["", ""]


def test_match_hits_counts_one_to_one():
    pred = [Hit(1.00, "h"), Hit(1.02, "h"), Hit(2.0, "o"), Hit(5.0, "h")]
    m = match_hits(pred, [1.01, 2.03, 3.0], ["h", "h", "o"], tol_s=0.05)
    assert (m["tp"], m["fp"], m["fn"]) == (2, 2, 1)
    assert m["side_correct"] == 1
    assert m["recall"] == pytest.approx(2 / 3) and m["precision"] == pytest.approx(0.5)


def test_hits_to_shots_groups_rallies(tmp_path):
    from smash_speed.shots import read_shots, write_shots
    hits = [Hit(1.0, "h", "audio+track", confidence=0.95), Hit(1.8, "o", "track", confidence=0.7),
            Hit(9.0, "o", "audio", confidence=0.4)]
    shots = read_shots(write_shots(tmp_path / "shots.csv", hits_to_shots(hits)))
    assert [(s.rally_id, s.index_in_rally) for s in shots] == [(1, 1), (1, 2), (2, 1)]
    assert [s.side for s in shots] == ["h", "o", "o"]
    assert shots[0].shot_type == "unknown" and shots[0].note == "audio+track" and shots[0].source == "hits"


def test_synthetic_benchmark_reports_recall_and_precision():
    r = synthetic_benchmark(n_rallies=3, fps=60.0, seed=2, use_audio=True)
    assert r["n_hits"] == r["tp"] + r["fn"]
    assert r["recall"] >= 0.9 and r["precision"] >= 0.9 and r["side_accuracy"] >= 0.9
