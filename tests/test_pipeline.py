"""Phase 4：合成動画で ``analyze`` を通しで動かすテスト。"""

import json

import pandas as pd
import pytest

from smash_speed.config import load_config
from smash_speed.pipeline import AnalyzeOptions, analyze
from smash_speed.validate import make_synthetic_session


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    cfg = load_config()
    cfg["synthetic"]["camera"].update(width=960, height=540, f_px=750.0)
    d = tmp_path_factory.mktemp("session")
    info = make_synthetic_session(cfg, d, speed=75.0, fps=120.0)
    return cfg, d, info


@pytest.mark.parametrize("use_audio", [True, False])
def test_analyze_end_to_end(session, use_audio):
    cfg, d, info = session
    out = d / f"out_{use_audio}"
    o = AnalyzeOptions(video=d / "synthetic.mp4", calib=d / "calibration.json", out_dir=out,
                       audio_path=(d / "synthetic.wav") if use_audio else None,
                       hitter_xy=tuple(info["hitter_xy"]), landing_xy=tuple(info["landing_xy"]),
                       overlay=use_audio)
    row = analyze(o, cfg)
    assert row["speed_initial_mps"] == pytest.approx(75.0, rel=0.05)
    assert row["impact_method"] == ("audio" if use_audio else "hitter_position")
    for f in ("config_used.yaml", "video_meta.json", "detections.csv", "events.json", "plane.json",
              "fit_result.json", "result.csv", "speed_time.png", "trajectory.png", "image_fit.png"):
        assert (out / f).exists(), f
    if use_audio:
        assert (out / "overlay.mp4").stat().st_size > 0
    res = pd.read_csv(out / "result.csv")
    assert res.loc[0, "speed_initial_kmh"] == pytest.approx(row["speed_initial_mps"] * 3.6)
    ev = json.loads((out / "events.json").read_text(encoding="utf-8"))
    assert ev["impact"]["t"] == pytest.approx(info["t_impact_video_s"], abs=0.006)


def test_analyze_reuses_cached_detections(session):
    cfg, d, info = session
    out = d / "out_True"
    before = (out / "detections.csv").stat().st_mtime
    o = AnalyzeOptions(video=d / "synthetic.mp4", calib=d / "calibration.json", out_dir=out,
                       audio_path=d / "synthetic.wav", hitter_xy=tuple(info["hitter_xy"]),
                       landing_xy=tuple(info["landing_xy"]), overlay=False)
    analyze(o, cfg)
    assert (out / "detections.csv").stat().st_mtime == before


def test_analyze_reports_cause_when_shuttle_not_found(session, capsys):
    # 検出する範囲にシャトルが通らないとき：トレースバックではなく、原因と対処を返す
    from smash_speed import cli
    from smash_speed.errors import AnalysisError

    cfg, d, info = session
    o = AnalyzeOptions(video=d / "synthetic.mp4", calib=d / "calibration.json", out_dir=d / "out_noshuttle",
                       hitter_xy=tuple(info["hitter_xy"]), landing_xy=tuple(info["landing_xy"]),
                       overlay=False, roi=[0, 500, 40, 540], start_s=0.0, end_s=0.6)
    with pytest.raises(AnalysisError) as e:
        analyze(o, cfg)
    assert "シャトル" in e.value.message and any("検出する範囲" in h for h in e.value.hints)
    code = cli.main(["analyze", "--video", str(d / "synthetic.mp4"), "--calib", str(d / "calibration.json"),
                     "--out", str(d / "out_noshuttle_cli"), "--start", "0", "--end", "0.6",
                     f"--hitter={info['hitter_xy'][0]},{info['hitter_xy'][1]}",
                     f"--landing={info['landing_xy'][0]},{info['landing_xy'][1]}", "--roi", "0,500,40,540",
                     "--no-overlay"])
    out = capsys.readouterr().out
    assert code == 2 and "[エラー]" in out and "[対処]" in out and "Traceback" not in out


@pytest.mark.parametrize("only", [True, False])
def test_analyze_with_external_candidates(session, tmp_path, only):
    # 外の検出（学習したモデルの点など）を候補として渡す：打点から打ち出した軌道の探索・漏れたコマの探し直しは
    # 自前で行うので、外の検出が 3 コマに 1 つしかなく、別の物の軌道が混じっていても、打った後の点を補って選ぶ。
    # only=False（自前の候補に足す）では、保存する candidates.csv は自前の候補のまま
    cfg, d, info = session
    det = pd.read_csv(d / "out_True" / "detections.csv")
    det = det[det.visible == 1].drop_duplicates("frame")[["frame", "timestamp", "x_px", "y_px", "visible"]]
    det = det.assign(confidence=1.0, track_id=0)
    sparse = det.iloc[::3]
    first, t0 = int(det.frame.min()), float(det.timestamp.min())
    dt = float(det.timestamp.diff().median())
    n = 40   # 打つ前に、画面の端をゆっくり動く物（打つ前の球の代わり）を、打った後の軌道より長く追っておく
    decoy = pd.DataFrame({"frame": range(first - n, first), "timestamp": [t0 - (n - k) * dt for k in range(n)],
                          "x_px": [30.0 + 2 * k for k in range(n)], "y_px": [500.0] * n, "visible": 1,
                          "confidence": 1.0, "track_id": 1})
    cand = tmp_path / "model.csv"
    pd.concat([decoy, sparse]).to_csv(cand, index=False)
    o = AnalyzeOptions(video=d / "synthetic.mp4", calib=d / "calibration.json", out_dir=tmp_path / "out",
                       audio_path=d / "synthetic.wav", candidates=cand, candidates_only=only,
                       hitter_xy=tuple(info["hitter_xy"]), landing_xy=tuple(info["landing_xy"]), overlay=False)
    row = analyze(o, cfg)
    assert row["speed_initial_mps"] == pytest.approx(75.0, rel=0.05)
    assert row["n_frames_used"] > 1.5 * len(sparse)
    used = pd.read_csv(tmp_path / "out" / "candidates.csv")
    if only:
        assert list(used.columns) == ["frame", "timestamp", "x", "y"] and len(used) == len(decoy) + len(sparse)
    else:
        assert "strength" in used.columns and not ((used.x == 30.0) & (used.y == 500.0)).any()
    ev = json.loads((tmp_path / "out" / "events.json").read_text(encoding="utf-8"))
    assert ev["impact"]["t"] == pytest.approx(info["t_impact_video_s"], abs=0.006)


def test_analyze_with_hand_labels(session, tmp_path):
    # 手でつけた位置（打った瞬間の印つき）からも解析できる。打つ前のシャトルは使わない
    cfg, d, info = session
    det = pd.read_csv(d / "out_True" / "detections.csv")
    det = det[det.visible == 1].drop_duplicates("frame")[["frame", "timestamp", "x_px", "y_px", "visible"]]
    det = det.assign(event="", source="manual")
    first = det.frame.min()
    det.loc[det.frame == first, "event"] = "hit"
    # 打つ前（打点のコマより前）に、ありえない位置の点を足しておく
    det = pd.concat([pd.DataFrame({"frame": [first - 2, first - 1], "timestamp": [0.0, 0.001],
                                   "x_px": [5.0, 6.0], "y_px": [5.0, 6.0], "visible": [1, 1],
                                   "event": ["", ""], "source": ["manual"] * 2}), det])
    lab = tmp_path / "labels.csv"
    det.to_csv(lab, index=False)
    o = AnalyzeOptions(video=d / "synthetic.mp4", calib=d / "calibration.json", out_dir=tmp_path / "out",
                       audio_path=d / "synthetic.wav", detections=lab, hitter_xy=tuple(info["hitter_xy"]),
                       landing_xy=tuple(info["landing_xy"]), overlay=False)
    row = analyze(o, cfg)
    assert row["speed_initial_mps"] == pytest.approx(75.0, rel=0.05)
