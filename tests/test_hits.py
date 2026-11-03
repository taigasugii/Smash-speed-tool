"""すべての打球を見つける（hits.py）のテスト。"""

import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile

from smash_speed.config import load_config
from smash_speed.events import audio_onsets
from smash_speed.hits import (Hit, _drop_same_side, _most_likely_sides, find_hits, hits_to_shots, match_hits,
                              player_reach, synthetic_benchmark, track_breaks)
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


def _rally_players(rally, cam, fps):
    """合成ラリーの ``players.csv`` の形の表。各側の選手は、その側の時刻のいちばん近い打点の真下に立つ
    （枠は足元から頭か上げた手（打点の 0.6 m 下）の高い方まで、幅は高さの 0.4 倍。人の検出の枠も上げた腕を含む）。"""
    rows = []
    t = np.arange(rally.t_start, rally.t_end, 1 / fps)
    for side in "ho":
        segs = [sg for sg in rally.segments if sg.side == side]
        for i, ti in enumerate(t):
            sg = min(segs, key=lambda sg: abs(sg.t_hit - ti))
            top = max(1.7, sg.p0[2] - 0.6)
            foot, head = cam.project(np.array([[sg.p0[0], sg.p0[1], 0.0], [sg.p0[0], sg.p0[1], top]]))
            hh = abs(foot[1] - head[1])
            rows.append({"frame": i, "t": ti, "side": side, "player": 0, "x0": foot[0] - 0.2 * hh,
                         "y0": min(foot[1], head[1]), "x1": foot[0] + 0.2 * hh, "y1": max(foot[1], head[1])})
    return pd.DataFrame(rows)


def test_player_reach_in_box_heights():
    det = _det([1.0, 1.05], [[150.0, 50.0], [160.0, 50.0]])
    players = pd.DataFrame({"t": [1.0, 1.0], "side": ["h", "o"], "player": [0, 0],
                            "x0": [100.0, 600.0], "y0": [100.0, 100.0], "x1": [140.0, 640.0], "y1": [300.0, 200.0]})
    (r,) = player_reach([Hit(1.0)], det, players)
    # h：枠の右上の角から (10, 50) px、枠の高さ 200 px。o：枠の左上の角から (440, 50) px、高さ 100 px
    assert r["h"] == pytest.approx(np.hypot(10, 50) / 200)
    assert r["o"] == pytest.approx(np.hypot(440, 50) / 100)
    assert player_reach([Hit(3.0)], det, players) == [{}]       # シャトルが見えていない


@pytest.mark.parametrize("fps", [30.0, 120.0])
def test_players_keep_every_synthetic_hit(cam, fps):
    """選手が打点の近くにいれば、選手の位置を使っても合成ラリーの打球と打った側は変わらない。"""
    rng = np.random.default_rng(7)
    for _ in range(3):
        rally = make_rally(rng, PHYS, n_shots=8)
        det = observe_rally(rally, cam, fps, PHYS, rng)
        true_t, true_side = [s.t_hit for s in rally.segments], [s.side for s in rally.segments]
        base = match_hits(find_hits(det, fps, camera=cam), true_t, true_side)
        m = match_hits(find_hits(det, fps, camera=cam, players=_rally_players(rally, cam, fps)), true_t, true_side)
        assert m["tp"] >= base["tp"] and m["fp"] <= base["fp"] and m["side_correct"] >= base["side_correct"]


def test_players_drop_a_turn_out_of_reach(cam):
    """飛んでいる途中で向きが変わって見えた所（どちらの選手にも届かない所）は、選手の位置があれば打球にしない。"""
    fps = 60.0
    rally = make_rally(np.random.default_rng(5), PHYS, n_shots=4)
    det = observe_rally(rally, cam, fps, PHYS, np.random.default_rng(1), noise_px=0.5, dropout=0.0)
    # いちばん長いショットの真ん中から後ろの点を、横へずらし続ける（向きの変化に見える）
    k = int(np.argmax([b.t_hit - a.t_hit for a, b in zip(rally.segments[:-1], rally.segments[1:])]))
    t_mid = 0.5 * (rally.segments[k].t_hit + rally.segments[k + 1].t_hit)
    after = (det.timestamp > t_mid) & (det.timestamp < rally.segments[k + 1].t_hit - 0.15)
    det.loc[after, "y_px"] += 1500.0 * (det.timestamp[after] - t_mid)
    true_t, true_side = [s.t_hit for s in rally.segments], [s.side for s in rally.segments]
    plain = find_hits(det, fps, camera=cam)
    assert any(abs(h.t - t_mid) < 0.1 for h in plain)
    hits = find_hits(det, fps, camera=cam, players=_rally_players(rally, cam, fps))
    assert not any(abs(h.t - t_mid) < 0.1 for h in hits)
    assert match_hits(hits, true_t, true_side)["tp"] >= match_hits(plain, true_t, true_side)["tp"]


def test_same_side_twice_in_a_short_time_keeps_one():
    """同じ側が続けて打ったと出た 2 本の間が短ければ、打球音のある方（なければ選手に近い方）だけを残す。"""
    hits = [Hit(1.0, "h", "track", score=3.0), Hit(1.3, "h", "audio", audio_strength=40.0), Hit(1.9, "o", "track")]
    kept, _ = _drop_same_side(hits, None, 0.6)
    assert [h.t for h in kept] == [1.3, 1.9]
    reach = [{"h": 0.5, "o": 2.0}, {"h": 0.1, "o": 2.0}]
    kept, r = _drop_same_side([Hit(1.0, "h"), Hit(1.2, "h")], reach, 0.6)
    assert [h.t for h in kept] == [1.2] and r == [reach[1]]
    # 間が長い（相手が打ち返す間がある）・側が違うなら両方残す
    assert len(_drop_same_side([Hit(1.0, "h"), Hit(1.7, "h")], None, 0.6)[0]) == 2
    assert len(_drop_same_side([Hit(1.0, "h"), Hit(1.2, "o")], None, 0.6)[0]) == 2
