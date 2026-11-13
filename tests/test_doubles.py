"""ダブルス（ROADMAP D4）：2 人を同じ番号のまま追う・誰が打ったか・陣形・ローテーション・2 人の間のテスト。"""

import json

import numpy as np
import pandas as pd
import pytest

from smash_speed.config import load_config
from smash_speed.doubles import (DOUBLES_FILE, FORMATION_FILE, PairTracks, analyze_doubles, assign_doubles,
                                 classify_formation, classify_gap, doubles_brief, formation_table, gap_events, gaps_svg,
                                 pair_sides, pick_hitter, prefit_feet, rotation_events, rotation_stats, write_doubles)
from smash_speed.hitter import hitter_at
from smash_speed.hits import _visible_points
from smash_speed.movement import movement_tracks
from smash_speed.physics import PhysicsParams
from smash_speed.player_ids import IdOptions, choose_chains, player_label, resolve_identities, track_sides
from smash_speed.players import (APP_DIM, PlayerOptions, add_court_positions, box_appearance, decode_appearance,
                                 detect_people, encode_appearance, ground_jumps, link_tracklets, track_players)
from smash_speed.rally_report import build_report
from smash_speed.shots import Rally, Shot
from smash_speed.synth_clutter import clutter_camera
from smash_speed.synth_doubles import (PEOPLE, doubles_benchmark, extra_feet, local_who, make_doubles_scene, make_kits,
                                       observe_doubles, observe_people, singles_auto_benchmark, tracking_scores,
                                       truth_players, who_is_tracked)

CFG = load_config()
PHYS = PhysicsParams.from_config(CFG)


# =============================================================================
# 枠の中の色
# =============================================================================
def test_box_appearance_separates_colours_and_round_trips():
    img = np.zeros((200, 100, 3), np.uint8)
    img[:100] = (0, 0, 255)       # 上は赤
    img[100:] = (255, 0, 0)       # 下は青
    a = box_appearance(img, (10, 0, 90, 200))
    assert a.shape == (APP_DIM,) and a.dtype == np.uint8
    half = APP_DIM // 2
    assert a[:half].argmax() != a[half:].argmax() and a[:half].max() >= 250 and a[half:].max() >= 250
    back = decode_appearance([encode_appearance(a), "", None])
    assert back.shape == (3, APP_DIM) and np.allclose(back[0], a / 255) and np.isnan(back[1:]).all()
    assert not box_appearance(img, (0, 0, 2, 3)).any()   # 小さすぎる枠


def test_detect_people_adds_appearance_only_when_asked():
    class Fake:
        def detect(self, img):
            return np.array([[10.0, 20.0, 50.0, 180.0, 0.9]])

    frames = [(0, 0.0, np.full((200, 100, 3), 128, np.uint8))]
    assert "app" not in detect_people(frames, Fake())
    with_app = detect_people(frames, Fake(), scale=0.5, appearance=True)
    assert list(with_app[["x0", "y1"]].iloc[0]) == [20.0, 360.0] and len(with_app["app"].iloc[0]) == 2 * APP_DIM


# =============================================================================
# 2 人を同じ番号のまま追う
# =============================================================================
@pytest.fixture(scope="module")
def scene():
    """合成のダブルス（後ろの高いカメラ・待つ人と隣のコートの人つき）と、追跡の結果。"""
    camera = clutter_camera("back_high", CFG)
    rng = np.random.default_rng(3)
    sc = make_doubles_scene(rng, PHYS, n_rallies=3)
    det, people, feet = observe_doubles(sc, camera, 30.0, PHYS, rng)
    frame_t = pd.Series(det["timestamp"].to_numpy(float), index=det["frame"].to_numpy())
    players, info = track_sides(people.drop(columns="who"), frame_t, CFG, PlayerOptions(per_side=0))
    return {"scene": sc, "camera": camera, "det": det, "people": people, "feet": feet, "frame_t": frame_t,
            "players": ground_jumps(players, camera), "info": info}


def test_auto_finds_two_players_per_side_and_keeps_identities(scene):
    assert scene["info"]["per_side"] == {"h": 2, "o": 2}
    assert pair_sides(scene["players"]) == ["h", "o"]
    sc = tracking_scores(scene["players"], scene["people"], len(scene["frame_t"]))
    for side in ("h", "o"):
        assert sc[side]["id_accuracy"] >= 0.97, sc[side]
        assert sc[side]["switches"] == 0 and sc[side]["extra_share"] <= 0.03 and sc[side]["coverage"] >= 0.8
    # 番号 0 は、最初に 2 人とも写ったとき y が小さい方
    first = scene["players"].sort_values("t").groupby(["side", "player"]).head(5).groupby(["side", "player"])["y"].mean()
    assert first["h", 0] < first["h", 1] and first["o", 0] < first["o", 1]


def test_auto_keeps_singles_and_matches_singles_tracking():
    r = singles_auto_benchmark(1, n_rallies=2, cameras=("corner_low",))
    assert r["corner_low"] == {"n": 1, "singles": 1, "same_as_singles": 1}


def _crossing(app: bool):
    """止まっている人の後ろ…ではなく前を、もう 1 人が横切る（横切る間、止まっている人は隠れて検出が抜ける）。"""
    camera = clutter_camera("back_high", CFG)
    n = 90
    t = np.arange(n) / 30.0
    still = np.tile([-4.0, 0.0], (n, 1))
    mover = np.column_stack([np.full(n, -5.0), np.linspace(-2.0, 2.0, n)])
    rng = np.random.default_rng(0)
    kits = make_kits(rng, ["h0", "h1"], "distinct" if app else "same")
    people = observe_people(t, {"h0": still, "h1": mover}, camera, rng, kits, miss_rate=0.0, merge_rate=0.0)
    return people, pd.Series(t, index=np.arange(n))


@pytest.mark.parametrize("app", [True, False])
def test_crossing_does_not_swap_identities(app):
    people, frame_t = _crossing(app)
    assert people.groupby("who").size().min() < len(frame_t)       # 隠れて抜けたコマがある
    players, info = track_sides(people.drop(columns="who"), frame_t, CFG, PlayerOptions(per_side=2))
    sc = tracking_scores(players, people, len(frame_t))["h"]
    assert sc["switches"] == 0 and sc["id_accuracy"] >= 0.97, sc
    assert info["sides"]["h"]["n_swaps"] == 0


def test_appearance_repairs_a_swapped_chain():
    """2 本の並びが途中で入れ替わっていても、色がはっきり違えば元に戻す（色が同じなら、そのまま）。"""
    n = 300
    t = np.arange(n) / 30.0
    frame_t = pd.Series(t, index=np.arange(n))
    gaps = [(55, 65), (115, 125), (175, 185), (235, 245)]     # 2 人とも見えない所（入れ替わりうる所）

    def slots_for(kits_mode):
        rng = np.random.default_rng(1)
        kits = make_kits(rng, ["h0", "h1"], kits_mode)
        rows = []
        for name, u0 in (("h0", 400.0), ("h1", 520.0)):      # 近くにいる 2 人（見えない間にどちらにもなりうる）
            for i in range(n):
                if any(a <= i < b for a, b in gaps):
                    continue
                noisy = 0.85 * kits[name] + 0.15 * rng.dirichlet(np.ones(APP_DIM)) * 2
                seg = sum(i >= b for _, b in gaps)
                rows.append({"frame": i, "t": t[i], "x0": u0 - 30, "y0": 300.0, "x1": u0 + 30, "y1": 500.0, "score": 0.9,
                             "u": u0, "v": 500.0, "x": -4.0, "y": -1.0 if name == "h0" else 1.0, "who": name,
                             "app": encode_appearance(np.round(255 * noisy / 2)), "tid": 2 * seg + (name == "h1"),
                             "seg": seg})
        cand = pd.DataFrame(rows)
        # 3 つ目の区間から入れ替わった並び
        swapped = (cand["seg"] >= 2) != (cand["who"] == "h1")
        return [cand[swapped == k].sort_values("frame") for k in (False, True)]

    slots = slots_for("distinct")
    assert slots[0]["who"].nunique() == 2
    ident, info = resolve_identities(slots, frame_t, IdOptions())
    assert info["appearance"]["used"] and info["n_events"] == 4 and info["n_swaps"] == 1
    for s, i in zip(slots, ident):
        assert ((i == 0) == (s["who"] == "h0").to_numpy()).all()
    # 同じ色の 2 人：色は使わず、位置が続いている「そのまま」を選ぶ
    _, info = resolve_identities(slots_for("same"), frame_t, IdOptions())
    assert not info["appearance"]["used"] and info["n_swaps"] == 0


def test_choose_chains_prefers_court_time_over_long_bystander_track():
    """線の外に長くいる人の軌跡より、線の中にいる選手の短い軌跡の並びを選ぶ。"""
    o = PlayerOptions(per_side=2)
    rows = []
    for i in range(300):     # 線の外にずっといる人（重みは外の分 0.3）
        rows.append((i, i / 30, 0.27, 100.0, 500.0, 200.0, 0.9, 9))
    for tid, (a, b), u in ((0, (0, 140), 600.0), (1, (150, 300), 620.0), (2, (0, 300), 1000.0)):
        for i in range(a, b):
            rows.append((i, i / 30, 0.9, u, 500.0, 200.0, 0.9, tid))
    cand = pd.DataFrame(rows, columns=["frame", "t", "weight", "u", "v", "h", "score", "tid"])
    cand["y0"], cand["y1"] = cand["v"] - cand["h"], cand["v"]
    chains = choose_chains(cand, o, IdOptions(), 2)
    assert sorted(map(sorted, chains)) == [[0, 1], [2]]


def test_forced_singles_is_unchanged_by_the_doubles_code(scene):
    """``per_side`` = 1 は今までのシングルスの追跡のまま（色の列があっても同じ）。"""
    people = scene["people"].drop(columns="who")
    a = track_players(people, scene["frame_t"], CFG, PlayerOptions(per_side=1))
    b = track_players(people.drop(columns="app"), scene["frame_t"], CFG, PlayerOptions(per_side=1))
    assert a.equals(b) and set(a["player"]) == {0}
    assert len(np.unique(link_tracklets(people.assign(u=people.u, v=people.v), PlayerOptions()))) > 0


# =============================================================================
# 誰が打ったか
# =============================================================================
def test_pick_hitter_uses_reach_and_court_distance():
    camera = clutter_camera("side", CFG)
    t = np.arange(60) / 30.0
    feet = {"h0": np.tile([-2.0, -1.5], (60, 1)), "h1": np.tile([-5.0, 1.5], (60, 1)),
            "o0": np.tile([2.0, 0.0], (60, 1)), "o1": np.tile([5.0, 0.0], (60, 1))}
    players = truth_players(t, feet, camera)
    tracks = PairTracks(players, camera)
    assert tracks.players_of("h") == [0, 1]
    # シャトルが h1 の頭の上（高さ 2.4 m）にある
    uv = camera.project(np.array([[-5.0, 1.5, 2.4]]))
    pick = pick_hitter(tracks, "h", 1.0, (np.array([1.0]), uv))
    assert pick.player == 1 and pick.conf > 0.9 and pick.n_candidates == 2
    # シャトルが見えていなくても、打点の位置が分かれば近い方
    pick = pick_hitter(tracks, "h", 1.0, None, [np.array([-2.2, -1.0])])
    assert pick.player == 0 and pick.conf > 0.75
    # 手がかりがなければ確かさは 0.5（どちらとも言えない）
    assert pick_hitter(tracks, "h", 1.0).conf == pytest.approx(0.5)
    assert pick_hitter(tracks, "h", 99.0) is None
    # 足元：選んだ人の足元
    e = hitter_at(players, 1.0, "h", player=1)
    assert (e.x, e.y, e.player) == (pytest.approx(-5.0), pytest.approx(1.5), 1)
    # 当てはめの前：確かに選べた打球だけ足元を返す
    det = pd.DataFrame({"timestamp": [1.0], "x_px": [uv[0, 0]], "y_px": [uv[0, 1]], "visible": [1]})
    got = prefit_feet([1.0, 1.0], ["h", "o"], players, det, camera)
    assert got[0] == pytest.approx([-5.0, 1.5]) and got[1] is None or got[1] is not None


def test_hitters_on_synthetic_doubles(scene):
    """合成のダブルスで、正解の打球の時刻・側から打った人を選ぶ（画像の手がかりだけ・打点つき）。"""
    sc, players, people = scene["scene"], scene["players"], scene["people"]
    tr = who_is_tracked(players, people)
    tracks = PairTracks(players, scene["camera"])
    vis = _visible_points(scene["det"])
    rng = np.random.default_rng(0)
    ok = {"pre": 0, "post": 0}
    hits = sc.hits()
    for t, side, who, p0, _ in hits:
        for mode in ok:
            pick = pick_hitter(tracks, side, t, vis, None if mode == "pre" else [p0[:2] + rng.normal(0, 0.5, 2)])
            ok[mode] += int(pick is not None and local_who(tr, side, pick.player, t) == who)
    assert ok["pre"] / len(hits) >= 0.8 and ok["post"] / len(hits) >= 0.85, (ok, len(hits))


# =============================================================================
# 陣形・ローテーション・2 人の間
# =============================================================================
def test_classify_formation_thresholds_and_smoothing():
    t = np.arange(8) / 10.0
    dx = np.array([3.0, 3.0, 3.0, 0.2, 3.0, 0.2, 0.2, 0.2])
    dy = np.array([0.3, 0.3, 0.3, 3.0, 0.3, 3.0, 3.0, 3.0])
    lab = classify_formation(dx, dy, t)
    # 1 コマだけの横並び（0.1 秒）は前の陣形にまとめる。最後の横並び（0.3 秒）は残る
    assert list(lab) == ["attack"] * 5 + ["defence"] * 3
    assert classify_formation(np.array([1.5]), np.array([1.5]), np.array([0.0]))[0] == "transition"
    # 近すぎると向きが決まらないので前のコマのまま
    assert list(classify_formation(np.array([3.0, 0.1]), np.array([0.2, 0.5]), np.array([0.0, 1.0]))) == ["attack"] * 2


def _static_players(seconds=4.0, fps=30.0, moves=None):
    """4 人が立っている ``players.csv`` の形。``moves`` は {人: [(時刻, (x, y)), ...]}（その時刻からそこにいる）。"""
    camera = clutter_camera("side", CFG)
    t = np.arange(int(seconds * fps)) / fps
    base = {"h0": (-2.0, 0.0), "h1": (-5.0, 0.0), "o0": (3.5, -1.5), "o1": (3.5, 1.5)}
    feet = {}
    for name, xy in base.items():
        a = np.tile(np.array(xy, float), (len(t), 1))
        for t0, new in (moves or {}).get(name, []):
            a[t >= t0] = new
        feet[name] = a
    return truth_players(t, feet, camera), camera


def test_formation_table_and_summary():
    players, _ = _static_players()
    table = formation_table(players)
    assert set(table["side"]) == {"h", "o"}
    h, o = table[table["side"] == "h"], table[table["side"] == "o"]
    assert (h["formation"] == "attack").all() and (o["formation"] == "defence").all()
    assert (h["front"] == 0).all()                    # h0 がネットに近い
    assert (o["left"] == 0).all()                     # o の側はネットを向いて左が y の小さい方
    summary, _ = analyze_doubles([], [Rally(1, 0.5, 3.5)], players)
    f = summary["formation"]
    assert f["h"]["attack"] == pytest.approx(1.0) and f["o"]["defence"] == pytest.approx(1.0)
    assert f["h"]["front_share"] == {"P1": 1.0, "P2": 0.0} and f["h"]["n_changes"] == 0
    assert summary["per_side"] == {"h": 2, "o": 2} and set(summary["player_names"]) == {"P1", "P2", "P3", "P4"}
    assert analyze_doubles([], [], players[players["player"] == 0])[0] == {}


def test_rotation_after_lift_and_attack():
    # h の側：1.0 秒にロブを上げ、1.5 秒から横並び。o の側：2.5 秒にスマッシュ、3.0 秒から前後
    players, _ = _static_players(5.0, moves={"h0": [(1.5, (-3.5, -1.5))], "h1": [(1.5, (-3.5, 1.5))],
                                              "o0": [(3.0, (2.0, 0.0))], "o1": [(3.0, (5.0, 0.0))]})
    shots = [Shot(1.0, "h", "lob", rally_id=1, index_in_rally=1), Shot(2.5, "o", "smash", rally_id=1, index_in_rally=2),
             Shot(3.2, "h", "drive", rally_id=1, index_in_rally=3)]
    table = formation_table(players)
    ev = rotation_events(shots, table)
    assert [(e["side"], e["kind"], e["already"]) for e in ev] == [("h", "lift", False), ("o", "attack", False)]
    assert ev[0]["seconds"] == pytest.approx(0.5, abs=0.2) and ev[0]["in_time"] is True
    assert ev[1]["seconds"] == pytest.approx(0.5, abs=0.2) and ev[1]["in_time"] is True
    st = rotation_stats(ev)
    assert st["h"]["lift"] == {"n": 1, "n_already": 0, "n_judged": 1, "n_in_time": 1,
                               "median_s": pytest.approx(0.5, abs=0.2)}
    # 相手がすぐ打ち返すと間に合わない
    late = rotation_events([Shot(1.0, "h", "lob", rally_id=1), Shot(1.2, "o", "smash", rally_id=1)], table)
    assert late[0]["in_time"] is False


def test_classify_gap_kinds():
    a, b = np.array([-3.5, -1.5]), np.array([-3.5, 1.5])      # h の側の横並び
    assert classify_gap(np.array([-3.6, 0.1]), a, b, "h") == "between"
    assert classify_gap(np.array([-3.3, -1.9]), a, b, "h") == "near"
    assert classify_gap(np.array([-1.0, 0.0]), a, b, "h") == "front"
    assert classify_gap(np.array([-6.3, -0.3]), a, b, "h") == "back"
    f, k = np.array([2.0, 0.0]), np.array([5.0, 0.0])         # o の側の前後
    assert classify_gap(np.array([3.5, 0.2]), f, k, "o") == "between"
    assert classify_gap(np.array([3.5, 2.6]), f, k, "o") == "side"


def test_gap_events_count_only_unreturned_shots_inside_the_court():
    players, _ = _static_players()
    tracks = movement_tracks(players)
    shots = [Shot(1.0, "h", "smash", land_x_m=3.6, land_y_m=0.0, rally_id=1, index_in_rally=1),     # 打ち返された
             Shot(1.6, "o", "lob", land_x_m=-3.5, land_y_m=2.0, rally_id=1, index_in_rally=2),      # h の側に落ちた
             Shot(3.0, "h", "smash", land_x_m=3.4, land_y_m=0.1, rally_id=2, index_in_rally=1),     # o の 2 人の間
             Shot(3.5, "h", "clear", land_x_m=8.0, land_y_m=0.0, rally_id=3, index_in_rally=1)]     # 外
    ev = gap_events(shots, tracks, CFG["court"])
    assert [(e["side"], e["kind"]) for e in ev] == [("h", "side"), ("o", "between")]
    assert ev[1]["d_near_m"] == pytest.approx(1.5, abs=0.15) and ev[1]["nearest"] in ("P3", "P4")
    assert len(ev[1]["players"]) == 2


# =============================================================================
# まとめ・書き出し・レポート
# =============================================================================
def test_doubles_summary_files_and_report(scene, tmp_path):
    sc, players = scene["scene"], scene["players"]
    shots, rallies = [], []
    for k, r in enumerate(sc.session, start=1):
        shots += r.to_shots(rally_id=k, phys=PHYS)
        rallies.append(r.to_rally(rally_id=k))
    info = assign_doubles(shots, players, scene["det"], scene["camera"], PHYS)
    assert info["n_pairs"] == 2 and info["n"] >= 0.9 * len(shots) and 0.5 < info["mean_conf"] <= 1.0
    # 打った人の正しさ（番号と人の対応は、追跡の答え合わせの対応）
    mapping = tracking_scores(players, scene["people"], len(scene["frame_t"]))
    truth = [who for _, _, who, _, _ in sc.hits()]
    names = {player_label(side, p): nm for side in ("h", "o") for p, nm in mapping[side]["mapping"].items()}
    right = sum(names.get(s.hitter_id) == who for s, who in zip(shots, truth))
    assert right / len(shots) >= 0.85, (right, len(shots))
    for s in shots:
        if s.hitter_id:
            assert s.hitter_id[0] == "P" and s.hitter_x_m is not None and 0.5 <= s.hitter_conf <= 1.0
            assert (s.hitter_id in ("P1", "P2")) == (s.side == "h")
    summary = write_doubles(tmp_path, shots, rallies, players, CFG["court"])
    saved = json.loads((tmp_path / DOUBLES_FILE).read_text(encoding="utf-8"))
    assert saved["hitters"]["n"] == info["n"] and set(saved["players"]) <= {"P1", "P2", "P3", "P4"}
    assert sum(p["n_shots"] for p in saved["players"].values()) == info["n"]
    for side in ("h", "o"):
        f = saved["formation"][side]
        assert f["attack"] + f["defence"] + f["transition"] == pytest.approx(1.0, abs=0.01) and f["attack"] > 0.1
        assert saved["rotation"][side]["lift"]["n"] + saved["rotation"][side]["attack"]["n"] > 0
    # ラリーは必ずどちらかの側に落ちて終わる（中に落ちたもの）
    assert sum(g["n"] for g in saved["gaps"].values()) >= 1
    table = pd.read_csv(tmp_path / FORMATION_FILE)
    assert set(table["formation"]) <= {"attack", "defence", "transition"}
    assert all(s.formation in ("attack", "defence", "transition", "") for s in shots)
    assert any(s.formation for s in shots) and any(s.opp_formation for s in shots)
    brief = doubles_brief(summary)
    assert "events" not in json.dumps(brief) and brief["per_side"] == {"h": 2, "o": 2}
    assert gaps_svg(summary, CFG["court"]).startswith("<svg")
    html = build_report(shots, rallies, court=CFG["court"], doubles=summary)
    for text in ("<h2>ダブルス</h2>", "陣形", "ローテーション", "落とされた所", "P1"):
        assert text in html
    assert "<h2>ダブルス</h2>" not in build_report(shots, rallies, court=CFG["court"])
    # 2 人の側がなければファイルを消す
    assert write_doubles(tmp_path, shots, rallies, players[players["player"] == 0], CFG["court"]) == {}
    assert not (tmp_path / DOUBLES_FILE).exists()


def test_doubles_benchmark_runs():
    r = doubles_benchmark(1, n_rallies=2, cameras=("side",))["side"]
    assert r["auto_doubles"] == 1 and r["id_accuracy"] >= 0.97 and r["hitter_post"] >= 0.9 and r["formation"] >= 0.9
    assert set(PEOPLE) == {"h0", "h1", "o0", "o1"} and callable(extra_feet) and callable(add_court_positions)


def test_auto_needs_two_players_on_both_sides(scene):
    """片側だけ 2 人（1 対 1 の練習に球出しの人がいるなど）は、自動ではシングルスのまま。側ごとに決めれば 2 人の側だけ 2 人。"""
    people = scene["people"][scene["people"]["who"] != "o1"].drop(columns="who")
    auto, info = track_sides(people, scene["frame_t"], CFG, PlayerOptions(per_side=0))
    assert info["per_side"] == {"h": 1, "o": 1}
    assert auto.equals(track_players(people, scene["frame_t"], CFG, PlayerOptions(per_side=1)))
    each, info = track_sides(people, scene["frame_t"], CFG, PlayerOptions(per_side=0, auto_each_side=True))
    assert info["per_side"] == {"h": 2, "o": 1}
    assert each[each["side"] == "o"].reset_index(drop=True).equals(auto[auto["side"] == "o"].reset_index(drop=True))
