"""コマを見て決めた正しい初速（agent/truth/true_speeds.py）と、実動画のチェックでの比べ方のテスト。"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from smash_speed.camera import look_at
from smash_speed.physics import MPS_TO_KMH, PhysicsParams, simulate

ROOT = Path(__file__).resolve().parents[1]


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ts():
    return _load("agent/truth/true_speeds.py", "true_speeds")


def _smash(fps: float, n: float = 0.5):
    """後ろの低い位置から撮った 1 本のスマッシュ（打点 x = −5.5 m、300 km/h）。"""
    cam = look_at(np.array([-9.5, -5.0, 1.0]), np.array([0.0, 0.0, 1.0]), 800.0, 1920, 1080)
    p0, v0 = np.array([-5.5, -1.0, 2.6]), np.array([80.0, 4.0, -18.0])
    v0 = v0 / np.linalg.norm(v0) * 300 / MPS_TO_KMH
    t = np.arange(1, int(n * fps)) / fps
    pos, _ = simulate(p0, v0, 0.0, t, PhysicsParams())
    rng = np.random.default_rng(0)
    uv = cam.project(pos) + rng.normal(0, 0.7, (len(t), 2))
    end_t = t[-1] + 1 / fps
    end_pos, _ = simulate(p0, v0, 0.0, np.array([end_t]), PhysicsParams())
    return cam, t, uv, {"t": end_t, "xy": end_pos[0, :2].tolist(), "sigma_m": 0.5}


def test_fit_recovers_speed_240fps(ts):
    cam, t, uv, end = _smash(240.0)
    r = ts.fit_smash(cam, t, uv, 0.0, [-5.3, -0.8], 0.5, end=end)
    assert r["speed_mps"] * MPS_TO_KMH == pytest.approx(300.0, rel=0.03)
    assert r["rms_px"] < 1.5


def test_reference_speed_has_uncertainty_parts(ts):
    cam, t, uv, end = _smash(30.0)
    s = {"video": "V", "id": 1, "fps": 30, "method": "test", "t_hit": 0.0, "t_hit_sigma_s": 0.008,
         "hitter": {"xy": [-5.3, -0.8], "sigma_m": 0.5}, "end": end,
         "points": np.column_stack([t, uv]).tolist()}
    r = ts.reference_speed(s, cam)
    assert r["kmh"] == pytest.approx(300.0, rel=0.1)
    parts = r["sigma_parts_kmh"]
    assert set(parts) == {"fit", "terminal_velocity", "hit_time", "hitter_point"}
    # 打った時刻の誤差 8 ms は、打った直後の減速（300 km/h で約 1600 m/s²）で 40 km/h ほどになる
    assert 30 < parts["hit_time"] < 70
    assert r["sigma_same_time_kmh"] < r["sigma_kmh"]
    # 打った時刻の速さは軌道の始めの速さ、後の時刻ほど遅い
    assert ts.speed_at(r, 0.0) == pytest.approx(r["kmh"], abs=0.2)
    assert ts.speed_at(r, 0.05) < r["kmh"] - 50


def test_stored_truth_matches_baseline_smashes():
    truth = json.loads((ROOT / "agent" / "checks" / "real_videos_truth.json").read_text(encoding="utf-8"))
    base = json.loads((ROOT / "agent" / "checks" / "real_videos_baseline.json").read_text(encoding="utf-8"))
    ids = {(e["name"], c["id"]) for e in base["videos"] for c in e["candidates"]}
    assert truth["smashes"]
    for r in truth["smashes"]:
        assert (r["video"], r["id"]) in ids
        assert r["sigma_kmh"] > 0 and r["sigma_same_time_kmh"] <= r["sigma_kmh"]
        assert set(r["state"]) == {"t_s", "p0_m", "v0_mps", "terminal_velocity_mps"}


def test_check_compares_with_truth_without_failing(ts, monkeypatch):
    chk = _load("agent/checks/check_real_videos.py", "check_real_videos")
    ref = {"video": "V", "id": 1, "kmh": 300.0, "sigma_kmh": 50.0, "sigma_same_time_kmh": 20.0, "t_hit_s": 1.0,
           "state": {"t_s": 1.0, "p0_m": [-5.5, -1.0, 2.6], "v0_mps": [80.0, 4.0, -18.0], "terminal_velocity_mps": 6.7}}
    ref["kmh"] = round(ts.speed_at(ref, 1.0), 1)
    rows = [{"video": "V", "id": 1, "kmh": 250.0}, {"video": "V", "id": 2, "kmh": 200.0}]
    later = ts.speed_at(ref, 1.02)
    out = chk.compare_truth(rows, {"V": {1: {"t_impact_s": 1.02}}}, {"smashes": [ref]})
    assert len(out) == 1
    o = out[0]
    assert o["err_kmh"] == pytest.approx(250.0 - ref["kmh"], abs=0.1)
    assert o["dt_hit_ms"] == pytest.approx(20.0)
    assert o["true_kmh_at_t"] == pytest.approx(later, abs=0.1)
    assert o["err_same_t_kmh"] == pytest.approx(250.0 - later, abs=0.2)
    assert "正しい初速との差" in chk.truth_summary(out)
    # 解析の打った時刻がなくても、打った時刻の速さどうしは比べる
    out = chk.compare_truth(rows, {}, {"smashes": [ref]})
    assert out[0]["true_kmh_at_t"] is None and out[0]["err_kmh"] is not None
