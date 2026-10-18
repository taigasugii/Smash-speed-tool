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
