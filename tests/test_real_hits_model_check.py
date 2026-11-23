"""学習したモデルでの実動画の打球のチェック（agent/checks/check_real_hits_model.py。I1）の、動画を使わない部分のテスト。"""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "agent" / "checks" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def chk(tmp_path, monkeypatch):
    """区間の一覧を作業フォルダーに置き、1 区間の解析を決まった数字にした ``check_real_hits_model``。"""
    base = _load("check_real_hits")
    clips = tmp_path / "clips.json"
    exp = {"n_true": 20, "n_found": 20, "tp": 20, "side_correct": 20}
    clips.write_text(json.dumps({"clips": [{"name": "A", "video": "a.mp4", "truth": "a.csv", "camera": {},
                                            "expected": exp, "expected_model": exp}]}), encoding="utf-8")
    (tmp_path / "a.mp4").write_bytes(b"")
    monkeypatch.setattr(base, "CLIPS", clips)
    seen = {}

    def fake(clip, video, cfg, work, tracknet=None):
        seen["tracknet"] = tracknet
        return dict(seen["row"])

    monkeypatch.setattr(base, "analyze_clip", fake)
    mod = _load("check_real_hits_model")
    monkeypatch.setattr(mod, "_base", lambda: base)
    monkeypatch.setattr(mod, "model_path", lambda: (tmp_path / "model_ft1.pt", "onnx"))
    ctx = {"root": tmp_path, "config": {"videos_dir": str(tmp_path)}}
    return mod, ctx, seen


def test_model_check_uses_the_model_and_its_own_baseline(chk, tmp_path):
    mod, ctx, seen = chk
    seen["row"] = {"n_true": 20, "n_found": 20, "tp": 20, "side_correct": 20, "hits": []}
    r = mod.run(ctx)
    assert r["ok"] and seen["tracknet"] == tmp_path / "model_ft1.pt"
    assert r["metrics"]["backend"] == "onnx" and "学習したモデル" in r["summary"]


def test_model_check_fails_only_when_more_than_003_worse(chk):
    mod, ctx, seen = chk
    seen["row"] = {"n_true": 20, "n_found": 20, "tp": 20, "side_correct": 20, "hits": []}
    seen["row"].update(tp=20, side_correct=19)          # 打った側 1.000 → 0.950：0.03 より悪い
    assert not mod.run(ctx)["ok"]
    seen["row"].update(n_found=21, side_correct=20)      # 適合率 1.000 → 0.952：0.03 より悪い
    assert not mod.run(ctx)["ok"]
    seen["row"] = {"n_true": 40, "n_found": 40, "tp": 40, "side_correct": 40, "hits": []}
    exp = mod._base().CLIPS
    data = json.loads(exp.read_text(encoding="utf-8"))
    data["clips"][0]["expected_model"] = {"n_true": 40, "n_found": 40, "tp": 40, "side_correct": 40}
    exp.write_text(json.dumps(data), encoding="utf-8")
    seen["row"].update(tp=39, side_correct=39)           # 再現率 0.975・適合率 0.975・打った側 1.000：0.03 以内
    assert mod.run(ctx)["ok"]


def test_model_check_is_skipped_without_the_model(chk, monkeypatch):
    mod, ctx, _ = chk
    monkeypatch.setattr(mod, "model_path", lambda: (None, "モデル model_ft1.pt がない"))
    r = mod.run(ctx)
    assert r["ok"] and r["skipped"] and "model_ft1.pt がない" in r["summary"]
