"""エージェントの正解で TrackNetV3 と同じ形のモデルを学習する道具（F1b）のテスト。

torch を使うテストは ``.venv-train`` の Python で動かす（ない PC では飛ばす。``test_torch_device.py`` と同じ）。
"""

import json

import numpy as np
import pandas as pd
import pytest

from smash_speed.train import tracknet_train as tt
from tests.test_torch_device import _run_in_train_env


def test_target_and_mask_by_kind():
    """見える・写っていないはコマ全体、シャトルでない印は周りだけ、分からないは数えない。"""
    y, m = tt.target_and_mask(tt.VISIBLE, 100.0, 50.0)
    assert y[50, 100] == 1 and y[50, 103] == 0 and y.sum() > 10 and m.all()
    y, m = tt.target_and_mask(tt.ABSENT, np.nan, np.nan)
    assert y.sum() == 0 and m.all()
    y, m = tt.target_and_mask(tt.NOT_SHUTTLE, 200.0, 100.0)
    assert y.sum() == 0 and m[100, 200] == 1 and m[100, 215] == 0 and 150 < m.sum() < 250
    y, m = tt.target_and_mask(tt.IGNORE, np.nan, np.nan)
    assert y.sum() == 0 and m.sum() == 0


def test_clip_starts_cover_every_label():
    frames = np.array([3, 4, 5, 30, 31, 90, 95, 199])
    starts = tt.clip_starts(frames, 200, 8, np.random.default_rng(1))
    assert all(0 <= s <= 192 for s in starts)
    for f in frames:
        assert any(s <= f < s + 8 for s in starts)
    assert len(starts) < len(frames)  # 近いコマは同じ区間に入る
    cont = tt.clip_starts(np.arange(10, 30), 200, 8, contiguous=True)
    assert cont == [10, 18, 26]
    assert tt.clip_starts(np.arange(190, 200), 200, 8, contiguous=True) == [190, 192]  # 最後は動画の終わりに合わせる


def _label_dir(d, rows, cands):
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["frame", "timestamp", "x_px", "y_px", "visible", "event", "source", "reason"]).to_csv(
        d / "labels.csv", index=False)
    pd.DataFrame(cands, columns=["id", "frame", "timestamp", "reason", "cand_x", "cand_y", "predicted", "video"]).to_csv(
        d / "candidates.csv", index=False)


def test_load_label_dir_uses_candidate_for_not_shuttle(tmp_path):
    rows = [(1, 0, 100.0, 50.0, 1, "", "agent_ok_snap", "gap"), (2, 0, np.nan, np.nan, -1, "", "agent_x", "jump"),
            (3, 0, np.nan, np.nan, 0, "", "agent_n", "edge"), (4, 0, np.nan, np.nan, -1, "", "agent_u", "gap"),
            (5, 0, 300.0, 20.0, -1, "", "agent_o", "gap")]
    cands = [(i, f, 0.0, "gap", 10.0 * f, 5.0 * f, False, "v.mp4") for i, f in enumerate(range(1, 6))]
    _label_dir(tmp_path / "a", rows, cands)
    lab = tt.load_label_dir(tmp_path / "a")
    assert lab.frame.tolist() == [1, 2, 3]
    assert lab.kind.tolist() == [tt.VISIBLE, tt.NOT_SHUTTLE, tt.ABSENT]
    assert lab.loc[1, ["x_px", "y_px"]].tolist() == [20.0, 10.0]


def test_heatmap_point_takes_largest_blob():
    hm = np.zeros((tt.HEIGHT, tt.WIDTH), np.uint8)
    assert tt.heatmap_point(hm) is None
    hm[10:12, 10:12] = 200
    hm[100:105, 300:309] = 255
    x, y = tt.heatmap_point(hm)
    assert (x, y) == (304.5, 102.5)


def test_evaluate_counts_against_labels(tmp_path):
    """熱の地図を元の解像度の点に戻し、正解と比べる（今の検出の eval.json も並べる）。"""
    d = tmp_path / "eval_dir"
    W, H = 1024, 576
    rows = [(f, 0, 200.0 + 10 * f, 100.0, 1, "", "agent_ok", "seq") for f in range(8)]
    rows += [(f, 0, np.nan, np.nan, 0, "", "agent_n", "seq") for f in range(8, 12)]
    _label_dir(d, rows, [(0, 0, 0.0, "seq", 0.0, 0.0, False, "v.mp4")])
    (d / "eval.json").write_text(json.dumps({"selected": {"recall": 0.5}}), encoding="utf-8")
    data = tmp_path / "data"
    data.mkdir()
    (data / "meta.json").write_text(json.dumps({"seq_len": 8, "dirs": [{"dir": str(d), "width": W, "height": H}]}),
                                    encoding="utf-8")
    np.save(data / "clips.npy", np.array([[0, 0], [0, 8]]))
    heat = np.zeros((2, 8, tt.HEIGHT, tt.WIDTH), np.uint8)
    for f in range(7):  # コマ 7 は見落とし、コマ 9 は余計な点
        cx, cy = round((200 + 10 * f) * tt.WIDTH / W), round(100 * tt.HEIGHT / H)
        heat[0, f, cy - 1:cy + 2, cx - 1:cx + 2] = 255
    heat[1, 1, 10:13, 10:13] = 255
    np.save(data / "heat.npy", heat)
    r = tt.evaluate(data, data / "heat.npy")
    m = r["dirs"]["eval_dir"]["model"]
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (7, 1, 1, 3)
    assert m["median_error_px"] < 2.5
    assert r["dirs"]["eval_dir"]["current_selected"] == {"recall": 0.5}
    assert (data / "heat_eval.json").exists()

    # 閾値を並べる：コマ 0 だけ弱い（0.4）ので、閾値 0.5 では見落としが 1 つ増える
    heat[0, 0][heat[0, 0] > 0] = 102
    np.save(data / "heat.npy", heat)
    s = tt.sweep_thresholds(data, data / "heat.npy", [0.3, 0.5])
    ms = s["dirs"]["eval_dir"]["model"]
    assert (ms["0.3"]["tp"], ms["0.3"]["fn"]) == (7, 1)
    assert (ms["0.5"]["tp"], ms["0.5"]["fn"]) == (6, 2)
    assert s["dirs"]["eval_dir"]["current_selected"] == {"recall": 0.5}
    assert (data / "heat_sweep.json").exists()


def test_split_tracks_at_gaps_and_jumps():
    """長く途切れた所・1 コマで大きく跳んだ所で軌道を分ける（跳びの上限は空いたコマの数だけ広げる）。"""
    frames = np.array([0, 1, 2, 3, 10, 11, 12, 14, 15])
    x = np.array([100, 110, 120, 130, 300, 310, 900, 920, 940], float)
    y = np.full(len(frames), 50.0)
    tid = tt.split_tracks(frames, x, y, width=1000, max_jump=0.15, max_gap=3)
    # 3 → 10 は 6 コマ空いた、11 → 12 は 590 px 跳んだ、12 → 14 は 1 コマ空き・20 px で同じ軌道
    assert tid.tolist() == [0, 0, 0, 0, 1, 1, 2, 2, 2]
    assert tt.split_tracks(np.array([5]), np.array([1.0]), np.array([1.0]), 100).tolist() == [0]
    # 打球：速く右下へ動いていた点が左上へ折り返したら分ける。頂点の近くのゆっくりした折り返しでは分けない
    f = np.arange(8)
    hit = tt.split_tracks(f, np.array([0, 40, 80, 120, 90, 60, 30, 0.0]), np.array([0, 20, 40, 60, 30, 0, -30, -60.0]), 1000)
    assert hit.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    # 打つ前の球が遅くなっていても（1 コマ 12 px・幅 3840 px）、速く返っていく折り返しは分ける
    slow = tt.split_tracks(np.arange(4), np.array([0, 4, 8, 4.0]), np.array([0, 11, 22, -38.0]), 3840)
    assert slow.tolist() == [0, 0, 0, 1]
    apex = tt.split_tracks(f, np.arange(8) * 2.0, np.array([0, 2, 3, 3.5, 3.5, 3, 2, 0.0]), 1000)
    assert apex.tolist() == [0] * 8


@pytest.mark.skipif(__import__("shutil").which("ffprobe") is None, reason="ffmpeg がありません")
def test_build_video_and_detections_from_heat(tmp_path):
    """動画の区間の全コマを書き出し、熱の地図を元の解像度・コマ番号・時刻の検出の表にする（埋めたコマは使わない）。"""
    from smash_speed import synth
    from smash_speed.config import load_config
    from smash_speed.physics import PhysicsParams

    cfg = load_config()
    cfg["synthetic"]["camera"].update(width=640, height=360, f_px=500.0)
    phys = PhysicsParams.from_config(cfg)
    truth = synth.render_video(synth.make_shot(70.0, cfg, phys), synth.default_camera(cfg), cfg, phys,
                               tmp_path / "v.mp4", fps=60.0)
    i0, i1 = 5, 24   # 20 コマ → 8 コマずつの 3 区間（最後の 4 コマは埋める）
    ts = truth.timestamp.to_numpy()
    d = tmp_path / "data"
    info = tt.build_video(tmp_path / "v.mp4", d, ts[i0] - 1e-3, ts[i1] + 1e-3, progress=lambda s: None)
    assert (info["n_frames"], info["n_clips"]) == (20, 3)
    assert np.load(d / "frames.npy").shape == (24, tt.HEIGHT, tt.WIDTH, 3)
    assert np.load(d / "clips.npy")[:, 1].tolist() == [5, 13, 21]
    ft = np.load(d / "frame_times.npy")
    assert ft[:, 0].tolist() == list(range(5, 25))
    np.testing.assert_allclose(ft[:, 1], ts[i0:i1 + 1], atol=1e-3)

    # 正解の位置に塊のある熱の地図（埋めたコマには離れた所に塊を置く。使われないこと）
    heat = np.zeros((3, 8, tt.HEIGHT, tt.WIDTH), np.uint8)
    sx, sy = tt.WIDTH / 640, tt.HEIGHT / 360
    for k in range(24):
        f = 5 + k
        if f > i1:
            heat[k // 8, k % 8, 5:8, 5:8] = 255
            continue
        r = truth.iloc[f]
        if r.visible == 1 and f != 12:   # コマ 12 は見落とし
            cx, cy = int(round(r.x_px * sx)), int(round(r.y_px * sy))
            heat[k // 8, k % 8, max(cy - 1, 0):cy + 2, max(cx - 1, 0):cx + 2] = 200
    np.save(d / "heat.npy", heat)
    det = tt.detections_from_heat(d, d / "heat.npy", 0.5, d / "detections.csv")
    assert det.frame.tolist() == list(range(5, 25))
    assert list(det.columns) == ["frame", "timestamp", "x_px", "y_px", "visible", "confidence", "track_id", "heat"]
    vis = det[det.visible == 1]
    m = vis.merge(truth, on="frame", suffixes=("", "_gt"))
    assert len(m) > 10 and np.hypot(m.x_px - m.x_px_gt, m.y_px - m.y_px_gt).max() < 3.0
    assert det.loc[det.frame == 12, "visible"].item() == 0 and det.loc[det.frame == 12, "track_id"].item() == -1
    assert set(vis.track_id) == {0} and np.allclose(vis.heat, 200 / 255, atol=1e-3)
    assert (det.confidence == det.visible).all()   # 0〜1 の間の値は今の検出のスコアと取り違えられる
    assert pd.read_csv(d / "detections.csv").shape == det.shape


def test_train_predict_and_load_tracknetv3_weights_on_cpu(tmp_path):
    """小さな作り物のデータで学習すると損失が下がり、熱の地図を書き出せる。TrackNetV3 の名前の重みを順番で読める。"""
    out = _run_in_train_env(f"""
import json, numpy as np, torch
from pathlib import Path
from smash_speed.train import tracknet_train as tt
from smash_speed.train.torch_device import build_tracknet
d = Path(r"{tmp_path}") / "data"; d.mkdir()
rng = np.random.default_rng(0)
n_clips, n = 4, 8
frames = np.full((n_clips * n, tt.HEIGHT, tt.WIDTH, 3), 60, np.uint8)
targets = np.full((n_clips, n, 3), np.nan, np.float32); targets[:, :, 0] = tt.IGNORE
for k in range(n_clips):
    for j in range(n):
        x, y = 50 + 40 * j + 5 * k, 60 + 10 * k
        frames[k * n + j, y - 2:y + 3, x - 2:x + 3] = 255
        if j % 2 == 0:
            targets[k, j] = (tt.VISIBLE, x, y)
np.save(d / "frames.npy", frames); np.save(d / "bgs.npy", np.full((1, tt.HEIGHT, tt.WIDTH, 3), 60, np.uint8))
np.save(d / "clips.npy", np.array([[0, 8 * k] for k in range(n_clips)])); np.save(d / "targets.npy", targets)
(d / "meta.json").write_text(json.dumps({{"seq_len": n, "dirs": []}}))
x, y, m = tt.ClipData(d).item(0, np.random.default_rng(3))
h = tt.train(d, d / "run", epochs=3, batch=2, lr=1e-3, width=4, val_frac=0.25, prefer="cpu", progress=lambda s: None)
heat = np.load(tt.predict(d, d / "run" / "model.pt", prefer="cpu"))
# TrackNetV3 の名前で保存した重みを、順番で対応させて読む
src = build_tracknet(width=4)
renamed = {{f"block_{{i}}.{{k.split('.')[-1]}}": v for i, (k, v) in enumerate(src.state_dict().items())}}
torch.save({{"model": renamed, "param_dict": {{"seq_len": 8}}}}, d / "tn.pt")
dst = build_tracknet(width=4)
n_loaded = tt.load_tracknetv3_weights(dst, d / "tn.pt")
xin = torch.rand(1, 27, 32, 64); src.eval(); dst.eval()
same = float((src(xin) - dst(xin)).abs().max())
try:
    tt.load_tracknetv3_weights(build_tracknet(width=8), d / "tn.pt"); bad = "loaded"
except ValueError as e:
    bad = "error"
print(json.dumps({{"x": list(x.shape), "y": list(y.shape), "mask_frames": int((m.reshape(n, -1).max(1) > 0).sum()),
                  "train": h["train_loss"], "heat": list(heat.shape), "n_loaded": n_loaded, "same": same, "bad": bad}}))
""", device="cpu")
    assert out["x"] == [27, tt.HEIGHT, tt.WIDTH] and out["y"] == [8, tt.HEIGHT, tt.WIDTH]
    assert out["mask_frames"] == 4
    assert out["train"][-1] < out["train"][0]
    assert out["heat"] == [4, 8, tt.HEIGHT, tt.WIDTH]
    assert out["n_loaded"] > 50 and out["same"] < 1e-6 and out["bad"] == "error"


def test_load_tracknetv3_weights_without_conv_bias(tmp_path):
    """本物の TrackNetV3 のように畳み込みに偏りがない重みも読め、偏りを 0 にしたモデルと同じ出力になる。"""
    out = _run_in_train_env(f"""
import json, torch
from pathlib import Path
from smash_speed.train import tracknet_train as tt
from smash_speed.train.torch_device import build_tracknet
src = build_tracknet(width=4)
items = list(src.state_dict().items())
# BatchNorm の前の畳み込みの偏りを 0 にして、学習済みの重みからは外す（TrackNetV3 の形）
conv_bias = {{items[i][0] for i in range(1, len(items) - 1) if items[i][0].endswith(".bias") and items[i - 1][1].dim() == 4
             and items[i + 1][1].dim() == 1}}  # 後ろに BatchNorm が続くものだけ（最後の層は偏りを持つ）
sd = src.state_dict()
for k in conv_bias:
    sd[k].zero_()
src.load_state_dict(sd)
renamed = {{f"block_{{i}}.{{'conv.' if v.dim() == 4 else 'bn.'}}{{k.split('.')[-1]}}": v
           for i, (k, v) in enumerate(src.state_dict().items()) if k not in conv_bias}}
d = Path(r"{tmp_path}")
torch.save({{"model": renamed}}, d / "tn.pt")
dst = build_tracknet(width=4)
n = tt.load_tracknetv3_weights(dst, d / "tn.pt")
xin = torch.rand(1, 27, 32, 64); src.eval(); dst.eval()
print(json.dumps({{"n": n, "n_src": len(renamed), "n_bias": len(conv_bias), "same": float((src(xin) - dst(xin)).abs().max())}}))
""", device="cpu")
    assert out["n_bias"] > 0 and out["n"] == out["n_src"] and out["same"] < 1e-6
