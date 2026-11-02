"""選手の検出と追跡（players）のテスト。人の検出のモデルは使わず、小さな ONNX と作った枠で確かめる。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smash_speed.camera import look_at
from smash_speed.config import load_config
from smash_speed.players import (PlayerOptions, PersonDetector, add_court_positions, body_heights, choose_tracklets,
                                 coverage, ensure_model, ground_jumps, track_players)

FPS = 30.0


def _constant_model(raw: np.ndarray, size: int) -> bytes:
    """入力 (1, 3, size, size) によらず、決まった YOLOX の出力 (1, N, 6) を返す ONNX。"""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    out = numpy_helper.from_array(raw[None].astype(np.float32), "raw")
    nodes = [helper.make_node("Constant", [], ["c"], value=out), helper.make_node("Identity", ["c"], ["output"])]
    graph = helper.make_graph(nodes, "const", [helper.make_tensor_value_info("images", TensorProto.FLOAT,
                                                                             [1, 3, size, size])],
                              [helper.make_tensor_value_info("output", TensorProto.FLOAT, list(raw[None].shape))])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    return m.SerializeToString()


def test_person_detector_decodes_boxes_and_suppresses_duplicates():
    size = 64
    n = sum((size // s) ** 2 for s in (8, 16, 32))
    raw = np.zeros((n, 6), np.float32)

    def put(k: int, gx: int, gy: int, s: int, box: tuple[float, ...], obj: float, cls: float) -> None:
        cx, cy, w, h = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]
        raw[k] = [cx / s - gx, cy / s - gy, np.log(w / s), np.log(h / s), obj, cls]

    # 入力の上の (20, 10)-(40, 40)。元の画像（横 128・縦 96）は 0.5 倍されているので (40, 20)-(80, 80)
    put(3 * 8 + 3, 3, 3, 8, (20, 10, 40, 40), 0.9, 0.9)
    put(3 * 8 + 4, 4, 3, 8, (21, 11, 41, 41), 0.8, 0.8)    # ほぼ同じ枠（重なりでまとめられる）
    put(64 + 0, 0, 0, 16, (0, 0, 10, 20), 0.5, 0.4)       # スコア 0.2（下限 0.3 より低い）
    det = PersonDetector(_constant_model(raw, size), min_score=0.3, prefer="cpu")
    boxes = det.detect(np.zeros((96, 128, 3), np.uint8))
    assert boxes.shape == (1, 5)
    np.testing.assert_allclose(boxes[0, :4], [40, 20, 80, 80], atol=1e-3)
    assert boxes[0, 4] == pytest.approx(0.81, abs=1e-4)


def test_choose_tracklets_picks_heaviest_non_overlapping_set():
    spans = pd.DataFrame({"id": [0, 1, 2, 3], "t0": [0.0, 0.0, 2.1, 1.0],
                          "t1": [2.0, 5.0, 5.0, 1.5], "weight": [30.0, 40.0, 25.0, 100.0]})
    # 0 と 2（55）より 1（40）は軽いが、3 は 0・1 と重なる：3 + 2 = 125 が最大
    assert choose_tracklets(spans) == [3, 2]


def test_choose_tracklets_allows_short_overlap():
    # 1 人に枠が 2 つ出て軌跡が分かれ、前後と 0.2〜0.3 秒重なった場合（IMG_5109_1 の奥の選手）
    spans = pd.DataFrame({"id": [0, 1, 2], "t0": [0.0, 14.21, 16.35],
                          "t1": [14.48, 16.58, 18.85], "weight": [348.0, 44.0, 58.0]})
    assert choose_tracklets(spans) == [0, 2]                # 重なりを許さないと、間の 2.4 秒が落ちる
    assert choose_tracklets(spans, 0.4) == [0, 1, 2]
    # 長く重なる軌跡（別の人）は、許す幅を超えるので一緒には選ばない
    spans.loc[1, "t0"] = 13.0
    assert choose_tracklets(spans, 0.4) == [0, 2]


def test_ensure_model_uses_existing_file_and_rejects_unknown(tmp_path):
    (tmp_path / "yolox_s.onnx").write_bytes(b"x")
    assert ensure_model("yolox_s", tmp_path) == tmp_path / "yolox_s.onnx"
    with pytest.raises(ValueError):
        ensure_model("yolov8", tmp_path)


def _scene(per_side: int = 1, seed: int = 0):
    """後ろの角の高い所から撮ったカメラと、選手（・待っている人・隣のコートの人・一瞬の誤検出）の枠。"""
    cfg = load_config()
    cam = look_at(np.array([-9.0, -5.5, 3.0]), np.array([0.0, 0.0, 0.5]), 1500.0, 1920, 1080)
    rng = np.random.default_rng(seed)
    n = 150
    t = np.arange(n) / FPS
    truth = []   # (side, 足元 (n, 2))
    for k in range(per_side):
        y0 = -1.2 + 2.4 * k if per_side == 2 else 0.0
        truth.append(("h", np.column_stack([-4.0 + 1.5 * np.sin(2 * t + k), y0 + 0.8 * np.cos(1.3 * t)])))
        truth.append(("o", np.column_stack([4.0 + 1.5 * np.sin(1.7 * t + k), y0 - 0.8 * np.sin(1.1 * t)])))
    others = [np.tile([3.0, 3.6], (n, 1)),                                  # 奥のコートの横（線の外）で待っている人
              np.column_stack([-3.0 + 1.0 * t, np.full(n, 6.5)])]           # 隣のコートを歩く人（範囲の外）
    rows, true_rows = [], []

    def box(foot: np.ndarray) -> tuple[float, ...]:
        uv = cam.project(np.array([[*foot, 0.0], [*foot, 1.75]]))
        h = uv[0, 1] - uv[1, 1]
        u = uv[0, 0] + rng.normal(0, 1.5)
        v = uv[0, 1] + rng.normal(0, 1.5)
        return u - 0.2 * h, v - h, u + 0.2 * h, v

    for i in range(n):
        for side, foot in truth:
            if rng.random() < 0.15:   # 抜け
                continue
            rows.append((i, t[i], *box(foot[i]), 0.85))
        for foot in others:
            rows.append((i, t[i], *box(foot[i]), 0.9))
        if i in (40, 41):              # 一瞬の誤検出（コートの中）
            rows.append((i, t[i], *box(np.array([-1.0, 2.5])), 0.4))
    people = pd.DataFrame(rows, columns=["frame", "t", "x0", "y0", "x1", "y1", "score"])
    frame_t = pd.Series(t, index=np.arange(n))
    return cfg, add_court_positions(people, cam), frame_t, truth


def test_track_players_follows_both_sides_and_ignores_bystanders():
    cfg, people, frame_t, truth = _scene()
    players = track_players(people, frame_t, cfg)
    # 最初と最後のコマの抜けは補えない
    assert all(c > 0.97 for c in coverage(players, len(frame_t)).values())
    for side, foot in truth:
        p = players[players["side"] == side].sort_values("frame")
        err = np.hypot(p["x"].to_numpy() - foot[p["frame"], 0], p["y"].to_numpy() - foot[p["frame"], 1])
        assert np.median(err) < 0.1 and err.max() < 0.4, side
        assert p["interpolated"].sum() > 0       # 抜けたコマは補っている
        assert p.loc[~p["interpolated"], "score"].notna().all()


def test_track_players_keeps_the_player_when_boxes_split():
    # 奥の選手に、少しずれた 2 つ目の枠が数コマ出る（体とラケットの腕など）。軌跡が分かれても選手を見失わない
    cfg, people, frame_t, truth = _scene()
    far = people[(people["score"] == 0.85) & (people["x"] > 0)]
    extra = []
    for f in (50, 51, 52, 95, 96, 97):
        r = far[far["frame"] == f]
        if len(r):
            r = r.iloc[0].copy()
            h = r["y1"] - r["y0"]
            for c in ("x0", "x1", "u"):
                r[c] -= 0.1 * h
            r["score"] = 0.4
            extra.append(r)
    people = pd.concat([people, pd.DataFrame(extra)], ignore_index=True)
    players = track_players(people, frame_t, cfg)
    assert coverage(players, len(frame_t))["o"] > 0.97
    assert coverage(track_players(people, frame_t, cfg, PlayerOptions(max_overlap_s=0.0)), len(frame_t))["o"] < 0.8


def test_track_players_doubles_finds_two_per_side():
    cfg, people, frame_t, truth = _scene(per_side=2, seed=1)
    players = track_players(people, frame_t, cfg, PlayerOptions(per_side=2))
    cov = coverage(players, len(frame_t), per_side=2)
    assert cov["h"] > 0.95 and cov["o"] > 0.95
    for side in ("h", "o"):
        feet = [f for s, f in truth if s == side]
        for pl in (0, 1):
            p = players[(players["side"] == side) & (players["player"] == pl)]
            # どちらかの選手に（ほぼ）ずっと付いている
            errs = [np.median(np.hypot(p["x"].to_numpy() - f[p["frame"], 0], p["y"].to_numpy() - f[p["frame"], 1]))
                    for f in feet]
            assert min(errs) < 0.15, (side, pl, errs)


def test_track_players_empty():
    cfg = load_config()
    people = add_court_positions(pd.DataFrame(columns=["frame", "t", "x0", "y0", "x1", "y1", "score"], dtype=float),
                                 look_at(np.array([-9.0, 0.0, 3.0]), np.zeros(3), 1500.0, 1920, 1080))
    players = track_players(people, pd.Series([0.0], index=[0]), cfg)
    assert players.empty and coverage(players, 1) == {"h": 0.0, "o": 0.0}


def test_ground_jumps_fills_the_floor_position_while_airborne():
    """跳んでいる間は足元が浮いて床の位置が遠くへずれる。前後から補い、腕を上げただけのコマは跳んだとみなさない。"""
    cfg = load_config()
    cam = look_at(np.array([-12.0, -1.0, 1.3]), np.array([0.0, 0.0, 0.5]), 1500.0, 1920, 1080)   # 後ろの低い所
    n = 90
    t = np.arange(n) / FPS
    foot = np.column_stack([-5.0 + 0.6 * t, -1.0 + 0.3 * t])
    lift = np.where((t >= 1.0) & (t <= 1.5), 4 * 0.5 * (t - 1.0) * (1.5 - t) / 0.25, 0.0)   # 0.5 m 跳ぶ
    arm = np.where((t >= 0.3) & (t <= 0.6), 0.5, 0.0)                                       # 腕を上げる（床の上）
    rows = []
    for i in range(n):
        uv = cam.project(np.array([[*foot[i], lift[i]], [*foot[i], lift[i] + 1.75 + arm[i]]]))
        h = (uv[0, 1] - uv[1, 1]) / (1.75 + arm[i]) * 1.75
        rows.append((i, t[i], uv[0, 0] - 0.2 * h, uv[1, 1], uv[0, 0] + 0.2 * h, uv[0, 1], 0.9))
    people = add_court_positions(pd.DataFrame(rows, columns=["frame", "t", "x0", "y0", "x1", "y1", "score"]), cam)
    raw = track_players(people, pd.Series(t, index=np.arange(n)), cfg)
    raw = raw[raw["side"] == "h"].reset_index(drop=True)
    jump = (lift > 0.02)[raw["frame"]]
    raw_err = np.hypot(raw["x"] - foot[raw["frame"], 0], raw["y"] - foot[raw["frame"], 1])
    assert raw_err[jump].max() > 1.5                       # 補う前は、跳んだ所で 1.5 m 以上ずれる
    assert np.median(body_heights(raw, cam)) == pytest.approx(1.75, abs=0.05)

    p = ground_jumps(raw, cam)
    err = np.hypot(p["x"] - foot[p["frame"], 0], p["y"] - foot[p["frame"], 1])
    assert err.max() < 0.15
    air = p["airborne"].to_numpy()
    assert air[(lift > 0.15)[p["frame"]]].all()            # 高く跳んでいる間は必ず
    assert not air[(arm > 0)[p["frame"]]].any()            # 腕を上げただけでは跳んだとみなさない
    assert not air[(t < 0.9) | (t > 1.6)].any()
    np.testing.assert_allclose(p["u"], raw["u"])            # 画像の上の足元は検出のまま
    assert ground_jumps(raw.iloc[:0], cam).empty
