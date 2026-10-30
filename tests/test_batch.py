"""いちばん速いスマッシュを探す（一括解析）のテスト。"""

import json

import numpy as np
import pytest

from smash_speed.batch import (BatchOptions, _back_extrapolate, analyze_selected, find_candidates, load_candidates,
                               save_candidates, strong_onsets)
from smash_speed.config import load_config
from smash_speed.shots import read_shots, side_of
from smash_speed.validate import make_synthetic_session


def test_strong_onsets_keeps_fast_exchanges_apart():
    ons = [(1.00, 100.0), (1.03, 40.0), (1.15, 80.0), (2.0, 10.0), (3.0, 60.0)]
    got = strong_onsets(ons, min_strength=0.25, merge_s=0.08)
    # 反響（0.03 s 後）はまとめるが、0.15 s 後の返球は別の候補として残す。弱すぎる音は除く
    assert [t for t, _ in got] == [1.00, 1.15, 3.0]


def test_back_extrapolate_decelerating_track():
    # 1 コマごとに移動量が 0.6 倍になる軌道を、2 コマ前まで戻す
    steps = 100 * 0.6 ** np.arange(5)
    x = np.r_[0, np.cumsum(steps)][1:] + 0.0
    t = np.arange(len(x)) / 30.0 + 2 / 30.0
    uv = np.column_stack([x, np.zeros_like(x)])
    u0, n = _back_extrapolate(t, uv, 0.0)
    assert n == pytest.approx(2.0)
    # 最初の点の 1 コマ前の移動は 60 / 0.6 = 100、その前は 100 / 0.6
    assert u0[0] == pytest.approx(x[0] - 100 - 100 / 0.6, abs=1e-6)


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    cfg = load_config()
    cfg["synthetic"]["camera"].update(width=960, height=540, f_px=750.0)
    d = tmp_path_factory.mktemp("batch")
    info = make_synthetic_session(cfg, d, speed=75.0, fps=120.0)
    return cfg, d, info


def test_find_and_analyze_on_synthetic_video(session, tmp_path):
    cfg, d, info = session
    o = BatchOptions(video=d / "synthetic.mp4", calib=d / "calibration.json", out_dir=tmp_path / "b",
                     audio=d / "synthetic.wav")
    cands = find_candidates(o, cfg, progress=lambda m: None)
    ok = [c for c in cands if c.status == "ok"]
    assert len(ok) == 1
    c = ok[0]
    assert c.t_onset == pytest.approx(info["t_impact_video_s"], abs=0.05)
    # だいたいの初速（打つ人・着地点は推定）。この合成動画のカメラはコートの横から浅い角度で見ているので、
    # 打点の高さを決め打ちした奥行きがずれやすく、大まかにしか合わない（2 倍以内）。打つ人は 2.5 m 以内
    assert 0.5 * 270 < c.rough_kmh < 2.0 * 270
    assert np.hypot(c.hitter["x"] - info["hitter_xy"][0], c.hitter["y"] - info["hitter_xy"][1]) < 2.5
    assert c.thumb and (tmp_path / "b" / c.thumb).exists()
    c.selected = True  # だいたいの初速が速すぎると「推定が怪しい」になり、最初は選ばれない。利用者が選ぶ
    # 確認した値（正しい打つ人・着地点）で詳しく解析すると ±5%
    c.hitter = {"x": info["hitter_xy"][0], "y": info["hitter_xy"][1]}
    c.landing = {"x": info["landing_xy"][0], "y": info["landing_xy"][1]}
    c.confirmed = True
    save_candidates(tmp_path / "b", cands)
    analyze_selected(tmp_path / "b", cfg, overlay=False, progress=lambda m: None)
    back, _ = load_candidates(tmp_path / "b")
    r = [x for x in back if x.id == c.id][0].result
    assert r["kmh"] == pytest.approx(75.0 * 3.6, rel=0.05) and r["confirmed"]
    assert (tmp_path / "b" / "ranking.csv").exists()
    # 同じ結果がショットの形（shots.csv）でも出る
    shot, = read_shots(tmp_path / "b" / "shots.csv")
    assert shot.shot_type == "smash" and shot.speed_kmh == pytest.approx(r["kmh"])
    assert shot.side == side_of(info["hitter_xy"][0]) and shot.hit_z_m is not None and shot.land_x_m is not None
    json.loads((tmp_path / "b" / "candidates.json").read_text(encoding="utf-8"))  # NaN を含まない


def test_probe_ignores_discarded_packets(monkeypatch, tmp_path):
    """iPhone の動画の先頭にある、表示しないパケット（flags に D）を時刻に含めない。"""
    from smash_speed import video_io

    def fake(cmd):
        if "packet=pts_time,flags" in cmd:
            pk = [(-0.2, "KD_"), (-0.1, "_D_"), (-1 / 30, "_D_")] + [(k / 30, "___") for k in range(10)]
            return {"packets": [{"pts_time": str(t), "flags": f} for t, f in pk]}
        return {"streams": [{"codec_type": "video", "width": 1920, "height": 1080, "avg_frame_rate": "30/1",
                             "start_time": "0.000000"}, {"codec_type": "audio"}], "format": {"duration": "0.33"}}

    monkeypatch.setattr(video_io, "_run_json", fake)
    monkeypatch.setattr(video_io, "find_tool", lambda name, cfg=None: name)
    f = tmp_path / "v.mov"
    f.write_bytes(b"")
    m = video_io.probe(f)
    assert m.n_frames == 10 and m.timestamps[0] == 0.0 and m.timestamps[3] == pytest.approx(0.1)
