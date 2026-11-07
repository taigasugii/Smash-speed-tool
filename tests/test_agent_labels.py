"""F1a：実動画のシャトルの正解を画像を見て付ける道具（怪しいコマの選び方、見本帳、答え → 正解ラベル）。"""

import numpy as np
import pandas as pd
import pytest

from smash_speed.train.agent_labels import (answers_to_labels, compare_answers, compare_label_dirs,
                                            detections_as_pred, evaluate_dir, implausible_jumps, item_tiles,
                                            merge_labels, parse_answer, pick_frames, pick_recheck, pick_sequence,
                                            save_labels, seq_windows, snap, summary, tolerance_for_width,
                                            write_check_sheets, write_sheets, write_zoom)

W, H = 640, 360


def _det() -> tuple[pd.DataFrame, pd.DataFrame]:
    """選んだ軌道 1（0〜29 コマ。12〜14 が途切れ、20 で跳ぶ）、負けた軌道 2、落ちた軌道 3 と、点のないコマ。"""
    rows = []
    for f in range(40):
        rows.append({"frame": f, "timestamp": f / 30, "x_px": np.nan, "y_px": np.nan, "visible": 0,
                     "confidence": 0.0, "track_id": -1})
    for f in range(30):
        if f in (12, 13, 14):
            continue
        x = 100 + 10 * f + (150 if f >= 20 else 0)
        rows.append({"frame": f, "timestamp": f / 30, "x_px": float(x), "y_px": 200.0 - 2 * f, "visible": 1,
                     "confidence": 0.9, "track_id": 1})
    for f in range(5, 15):
        rows.append({"frame": f, "timestamp": f / 30, "x_px": 500.0, "y_px": 100.0 + f, "visible": 1,
                     "confidence": 0.5, "track_id": 2})
    for f in range(30, 34):
        rows.append({"frame": f, "timestamp": f / 30, "x_px": 50.0 + f, "y_px": 300.0, "visible": 1,
                     "confidence": 0.3, "track_id": 3})
    det = pd.DataFrame(rows)
    tab = pd.DataFrame({"track_id": [1, 2, 3], "keep": [True, True, False], "selected": [True, False, False]})
    return det, tab


def test_pick_frames_reasons():
    det, tab = _det()
    c = pick_frames(det, tab, W, H, n=60, normal_frac=0.2, seed=1)
    by = {int(r.frame): r for r in c.itertuples()}
    assert "gap" in by[13].reason and by[13].predicted          # 途切れの真ん中は予測の位置
    assert by[13].cand_x == pytest.approx(100 + 10 * 13, abs=1)
    assert "gap" in by[11].reason and not by[11].predicted
    assert "jump" in by[19].reason and "jump" in by[20].reason
    assert "track_end" in by[0].reason and "track_end" in by[29].reason
    rej = c[c.reason.str.contains("rejected")]
    assert len(rej) >= 2 and set(rej.frame) & {6, 10, 13, 31, 32}
    assert (c.reason == "normal").sum() > 0
    assert c.frame.is_unique and c.frame.is_monotonic_increasing
    # もう付けたコマは選ばない・数を絞る
    c2 = pick_frames(det, tab, W, H, n=10, seed=1, exclude=set(c.frame[:5]))
    assert len(c2) <= 10 and not set(c2.frame) & set(c.frame[:5])


def _frames():
    """黒い背景の上を白い点（3×3）が右へ動く動画。"""
    imgs = {}
    for f in range(10):
        im = np.zeros((H, W, 3), np.uint8)
        x, y = 100 + 20 * f, 150
        im[y - 1:y + 2, x - 1:x + 2] = 255
        imgs[f] = im

    def get(i):
        if i not in imgs:
            raise IndexError(i)
        return imgs[i]
    return get


def test_snap_moves_to_blob():
    get = _frames()
    x, y, ok = snap(get, 5, 205.0, 154.0)          # 本当は (200, 150)
    assert ok and x == pytest.approx(200, abs=0.6) and y == pytest.approx(150, abs=0.6)
    x, y, ok = snap(get, 5, 400.0, 300.0)          # 近くに動くものがなければそのまま
    assert not ok and (x, y) == (400.0, 300.0)
    assert snap(get, 9, 280.0, 150.0)[2] is False   # 次のコマがない


def test_parse_answer():
    assert parse_answer("ok")[0] == "ok"
    assert parse_answer(" p 12.5 30 ") == ("p", 12.5, 30.0)
    for a in ("n", "x", "o", "u"):
        assert parse_answer(a)[0] == a
    with pytest.raises(ValueError):
        parse_answer("p 1")
    with pytest.raises(ValueError):
        parse_answer("yes")


def test_answers_to_labels_and_save(tmp_path):
    get = _frames()
    cands = pd.DataFrame({"id": range(6), "frame": [2, 3, 4, 5, 6, 7], "timestamp": np.arange(2, 8) / 30,
                          "reason": ["normal", "gap", "rejected", "jump", "edge", "rejected"],
                          "cand_x": [141.0, 300.0, 50.0, 200.0, 10.0, 240.0], "cand_y": [150.0] * 6,
                          "predicted": False})
    answers = pd.DataFrame({"id": [0, 1, 2, 3, 4, 5], "answer": ["ok", "p 161 151", "x", "u", "n", "o"]})
    lab = answers_to_labels(cands, answers, get)
    r = lab.set_index("frame")
    assert r.loc[2, "visible"] == 1 and r.loc[2, "x_px"] == pytest.approx(140, abs=0.6)
    assert r.loc[2, "source"] == "agent_ok_snap"
    assert r.loc[3, "visible"] == 1 and r.loc[3, "x_px"] == pytest.approx(160, abs=0.6)
    assert r.loc[4, "visible"] == -1 and r.loc[4, "source"] == "agent_x"
    assert r.loc[5, "visible"] == -1 and r.loc[6, "visible"] == 0
    assert r.loc[7, "visible"] == -1 and np.isfinite(r.loc[7, "x_px"])   # 隣のコートは位置を残す
    s = summary(lab)
    assert (s["visible"], s["not_visible"], s["unknown"], s["not_shuttle"], s["other_court"]) == (2, 1, 1, 1, 1)
    assert s["by_reason"]["rejected"] == {"n": 2, "candidate_ok": 0, "moved": 0, "not_visible": 0,
                                          "not_shuttle": 1, "other_court": 1, "unknown": 0}

    save_labels(lab, tmp_path, "clip")
    tn = pd.read_csv(tmp_path / "clip_ball.csv")
    assert list(tn.Frame) == [2, 3, 6]               # 使えるコマだけ、元のコマ番号のまま
    assert list(tn.Visibility) == [1, 1, 0]
    old = pd.read_csv(tmp_path / "labels.csv")
    new = answers_to_labels(cands.iloc[[0]], pd.DataFrame({"id": [0], "answer": ["n"]}))
    m = merge_labels(old, new)
    assert len(m) == 6 and m.set_index("frame").loc[2, "visible"] == 0


def test_write_sheets(tmp_path):
    get = _frames()
    det = pd.DataFrame({"frame": [3, 3], "x_px": [160.0, 300.0], "y_px": [150.0, 100.0], "visible": [1, 1]})
    items = pd.DataFrame({"id": [0, 1, 2], "frame": [3, 0, 9], "reason": ["normal", "gap", "edge"],
                          "cand_x": [160.0, 100.0, 630.0], "cand_y": [150.0, 150.0, 5.0],
                          "predicted": [False, True, False]})
    paths = write_sheets(get, items, det, tmp_path / "s", half=40, per_row=2, rows=1)
    assert len(paths) == 2 and all(p.exists() for p in paths)
    lab = pd.DataFrame({"frame": [3, 4], "x_px": [160.0, np.nan], "y_px": [150.0, np.nan], "visible": [1, 0]})
    assert len(write_check_sheets(get, lab, tmp_path / "c")) == 1


def test_recheck_pick_and_compare():
    cands = pd.DataFrame({"id": range(20), "frame": range(0, 40, 2), "timestamp": np.arange(20) / 15,
                          "reason": "normal", "cand_x": 100.0, "cand_y": 50.0, "predicted": False})
    answers = pd.DataFrame({"id": range(15), "answer": "ok"})
    r = pick_recheck(cands, answers, frac=0.2, seed=3)
    assert len(r) == 3 and set(r.id) <= set(range(15)) and r.frame.is_monotonic_increasing

    a1 = pd.DataFrame({"id": [0, 1, 2, 3, 4], "answer": ["ok", "p 110 52", "x", "ok", "u"]})
    a2 = pd.DataFrame({"id": [0, 1, 2, 3, 4], "answer": ["p 102 50", "ok", "x", "x", "u"]})
    c = cands.iloc[:5]
    out = compare_answers(answers_to_labels(c, a1), answers_to_labels(c, a2))
    # ok と p はどちらも「見える」。3 番だけ見える → シャトルでない、に変わった
    assert out["n"] == 5 and out["agree"] == 4 and out["visible_agree_rate"] == pytest.approx(0.8)
    assert out["n_both_visible"] == 2 and out["dist_max_px"] == pytest.approx(np.hypot(10, 2))
    assert out["n_over_5px"] == 1 and out["pairs"]["vis->x"] == 1   # 0 番は 2 px、1 番は 10 px ずれる
    assert sorted(out["disagree_frames"]) == [2, 6]


def test_implausible_jumps():
    lab = pd.DataFrame({"frame": [0, 1, 2, 3, 10], "x_px": [100.0, 110.0, 400.0, 410.0, 600.0],
                        "y_px": [50.0] * 5, "visible": [1, 1, 1, 1, 1]})
    j = implausible_jumps(lab, W)                    # 1 コマで 290 px は 640 の 6% を超える。10 コマ離れた組は見ない
    assert list(zip(j.frame_a, j.frame_b)) == [(1, 2)]
    assert implausible_jumps(lab[lab.visible == 0], W).empty


def test_pick_sequence():
    det, tab = _det()
    c = pick_sequence(det, tab, [8, 28], 6)
    assert list(c.frame) == [8, 9, 10, 11, 12, 13, 28, 29, 30, 31, 32, 33]
    assert (c.reason == "seq").all()
    r = c.set_index("frame")
    assert not r.loc[9, "predicted"] and r.loc[9, "cand_x"] == pytest.approx(190)
    assert r.loc[13, "predicted"] and r.loc[13, "cand_x"] == pytest.approx(230)    # 途切れは前後の点を結んだ位置
    assert not r.loc[31, "predicted"] and r.loc[31, "cand_x"] == pytest.approx(81)  # 軌道の後は、ほかの軌道の点
    starts = seq_windows(det, tab, 3, 8, seed=0)
    assert len(starts) >= 2 and all(b - a >= 8 for a, b in zip(starts, starts[1:]))
    assert all(0 <= s <= 32 for s in starts)
    assert seq_windows(det, tab, 3, 8, exclude=set(range(40))) == []              # 付けたコマは選ばない


def test_overview_and_zoom(tmp_path):
    get = _frames()
    det = pd.DataFrame({"frame": [3], "x_px": [160.0], "y_px": [150.0], "visible": [1]})
    item = pd.Series({"id": 0, "frame": 3, "reason": "seq", "cand_x": 160.0, "cand_y": 150.0, "predicted": False})
    plain = item_tiles(get, item, det, 40, 1.0)
    wide = item_tiles(get, item, det, 40, 1.0, overview=True)
    assert wide.shape[0] == plain.shape[0] and wide.shape[1] > plain.shape[1]
    assert write_zoom(get, [(5, 200.0, 150.0), (0, 100.0, 150.0)], tmp_path / "z.jpg", half=20).exists()


def test_detections_as_pred_and_tolerance():
    det, tab = _det()
    p = detections_as_pred(det, {1}).set_index("frame")
    assert len(p) == 40 and p.loc[13, "visible"] == 0 and p.loc[5, "x_px"] == 150.0
    a = detections_as_pred(det).set_index("frame")
    assert a.loc[13, "x_px"] == 500.0 and a.loc[31, "visible"] == 1   # 負けた・落ちた軌道の点も使う
    assert tolerance_for_width(512) == 4.0 and tolerance_for_width(2560) == 20.0


def test_evaluate_dir_compare_and_span_reasons(tmp_path):
    det, tab = _det()
    d = tmp_path / "e"
    d.mkdir()
    det.to_csv(d / "detections_None_None.csv", index=False)
    tab.to_csv(d / "tracks.csv", index=False)
    pd.DataFrame({"id": [0], "frame": [0], "video": ["v.mp4"]}).to_csv(d / "candidates.csv", index=False)
    lab = pd.DataFrame({"frame": [5, 13, 31, 35], "timestamp": 0.0, "x_px": [151.0, 230.0, np.nan, np.nan],
                        "y_px": [190.0, 174.0, np.nan, np.nan], "visible": [1, 1, 0, -1], "event": "",
                        "source": ["agent_ok_snap", "agent_p", "agent_n", "agent_u"],
                        "reason": ["span", "span_gap", "span_gap", "span_gap"]})
    lab.to_csv(d / "labels.csv", index=False)
    e = evaluate_dir(d, width=640)                                  # 距離 5 px
    s = e["selected"]
    assert e["tolerance_px"] == 5.0 and (s["tp"], s["fn"], s["tn"]) == (1, 1, 1)   # 13 は途切れ、31 は選んでいない
    assert e["all"]["fp_far"] == 1 and e["all"]["fp_invisible"] == 1  # 負けた軌道（13）と落ちた軌道（31）の点
    assert (d / "eval.json").exists()
    d2 = tmp_path / "f"
    d2.mkdir()
    lab.assign(source=["agent_ok", "agent_n", "agent_n", "agent_u"], visible=[1, 0, 0, -1],
               x_px=[153.0, np.nan, np.nan, np.nan]).to_csv(d2 / "labels.csv", index=False)
    c = compare_label_dirs(d, d2)
    assert c["n"] == 4 and c["agree"] == 3 and c["disagree_frames"] == [13] and c["dist_median_px"] == 2.0
    by = summary(lab)["by_reason"]                                   # 「span」は「span_gap」を数えない
    assert by["span"]["n"] == 1 and by["span_gap"]["n"] == 3 and by["span_gap"]["moved"] == 1
