"""実動画の回帰チェック（agent/checks/check_real_videos.py）の、動画を使わない部分のテスト。"""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def chk(tmp_path, monkeypatch):
    p = ROOT / "agent" / "checks" / "check_real_videos.py"
    spec = importlib.util.spec_from_file_location("check_real_videos", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "BASELINE", tmp_path / "baseline.json")
    return mod


def _fake_outputs(d: Path) -> Path:
    """本体の outputs に似せたフォルダ（1 本の動画・2 本の候補のうち 1 本が順位表に載っている）。"""
    out = d / "outputs"
    sm = out / "CLIP_smashes"
    (sm / "det").mkdir(parents=True)
    (out / "CLIP_calibration.json").write_text(json.dumps({"camera": {"f_px": 1000.0}}), encoding="utf-8")
    (sm / "det" / "smash_03_5.0s.csv").write_text("frame,t,u,v\n1,0.1,10,20\n", encoding="utf-8")
    cand = {"id": 3, "t_onset": 5.0, "strength": 40.0, "start_s": 4.7, "end_s": 6.0, "status": "ok",
            "hitter": {"x": -5.0, "y": 0.5}, "landing": {"x": 4.0, "y": -1.0}, "confirmed": True, "rough_kmh": 250.0,
            "detections": "det/smash_03_5.0s.csv", "result": {"kmh": 300.0}}
    other = {**cand, "id": 4, "t_onset": 9.0, "detections": None, "result": None}
    info = {"video": str(d / "videos" / "CLIP.MOV"), "calib": str(d / "missing.json"), "capture_fps": None}
    (sm / "candidates.json").write_text(json.dumps({"info": info, "candidates": [cand, other]}), encoding="utf-8")
    pd.DataFrame([{"rank": 1, "id": 3, "t_onset_s": 5.0, "speed_initial_kmh": 300.0}]).to_csv(
        sm / "ranking.csv", index=False, encoding="utf-8-sig")
    return out


def test_make_baseline_freezes_ranked_candidates(chk, tmp_path):
    b = chk.make_baseline(_fake_outputs(tmp_path))
    assert len(b["videos"]) == 1
    v = b["videos"][0]
    assert v["name"] == "CLIP" and v["video"] == "CLIP.MOV"
    # candidates.json の calib がなければ outputs/<名前>_calibration.json を使い、中身ごと固める
    assert v["calibration"] == {"camera": {"f_px": 1000.0}}
    assert [c["id"] for c in v["candidates"]] == [3]
    c = v["candidates"][0]
    assert c["kmh"] == 300.0 and c["hitter"] == {"x": -5.0, "y": 0.5}
    assert c["detections_csv"].startswith("frame,t,u,v")


def test_compare_flags_deviation_over_3_percent(chk):
    entry = {"name": "CLIP", "candidates": [{"id": 1, "t_onset": 1.0, "kmh": 300.0},
                                            {"id": 2, "t_onset": 2.0, "kmh": 200.0},
                                            {"id": 3, "t_onset": 3.0, "kmh": 250.0}]}
    rows = chk.compare(entry, {1: {"kmh": 308.0}, 2: {"kmh": 207.0}, 3: {"error": "解析できませんでした"}})
    assert [r["ok"] for r in rows] == [True, False, False]
    assert rows[0]["dev_pct"] == pytest.approx(2.67, abs=0.01)
    assert rows[2]["kmh"] is None and rows[2]["error"]


def test_run_skips_without_videos(chk, tmp_path):
    chk.BASELINE.write_text(json.dumps(chk.make_baseline(_fake_outputs(tmp_path))), encoding="utf-8")
    r = chk.run({"root": ROOT, "config": {"videos_dir": str(tmp_path / "none")}})
    assert r["ok"] and r["skipped"]


def test_run_skips_without_baseline(chk):
    r = chk.run({"root": ROOT, "config": {}})
    assert r["ok"] and r["skipped"]


def test_find_video_prefers_videos_dir(chk, tmp_path):
    (tmp_path / "v").mkdir()
    (tmp_path / "v" / "CLIP.MOV").write_bytes(b"")
    entry = {"video": "CLIP.MOV", "video_path": str(tmp_path / "old" / "CLIP.MOV")}
    assert chk.find_video(entry, str(tmp_path / "v")) == tmp_path / "v" / "CLIP.MOV"
    assert chk.find_video(entry, str(tmp_path / "none")) is None
