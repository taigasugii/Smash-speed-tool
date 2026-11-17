"""学習したシャトルの検出を ONNX（onnxruntime）で動かす口（``tracknet_train.predict_onnx``・``predict_backend``）。"""

import json
from pathlib import Path

import numpy as np
import pytest

from smash_speed.config import PROJECT_ROOT
from smash_speed.train import tracknet_train as tt

REAL_ONNX = PROJECT_ROOT / "ckpts" / "shuttle" / "model_ft1.onnx"


def _clip_dir(d: Path, n_clips: int = 2, seq_len: int = 8) -> Path:
    """``build_video`` と同じ形の、乱数のコマの書き出し。"""
    rng = np.random.default_rng(0)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "frames.npy", rng.integers(0, 256, (n_clips * seq_len, tt.HEIGHT, tt.WIDTH, 3), dtype=np.uint8))
    np.save(d / "bgs.npy", rng.integers(0, 256, (1, tt.HEIGHT, tt.WIDTH, 3), dtype=np.uint8))
    np.save(d / "clips.npy", np.array([[0, seq_len * c] for c in range(n_clips)], np.int64))
    tg = np.full((n_clips, seq_len, 3), np.nan, np.float32)
    tg[:, :, 0] = tt.IGNORE
    np.save(d / "targets.npy", tg)
    (d / "meta.json").write_text(json.dumps({"seq_len": seq_len, "dirs": [{"width": 1920, "height": 1080}]}),
                                 encoding="utf-8")
    return d


def test_predict_backend_prefers_torch_then_onnx_then_train_env(tmp_path, monkeypatch):
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"")
    monkeypatch.delenv(tt.BACKEND_ENV, raising=False)
    has = {"torch": False, "onnxruntime": True}
    monkeypatch.setattr(tt, "_has", lambda m: has[m])
    monkeypatch.setattr(tt, "train_python", lambda: Path("venv-train/python.exe"))
    assert tt.predict_backend(ckpt) == "train-env"          # .onnx がまだない
    tt.onnx_for(ckpt).write_bytes(b"")
    assert tt.predict_backend(ckpt) == "onnx"               # torch がなく .onnx がある
    assert tt.predict_backend(ckpt, "torch") == "train-env"  # torch を指定すれば .venv-train
    has["torch"] = True
    assert tt.predict_backend(ckpt) == "torch"              # torch がある Python ではそのまま
    assert tt.predict_backend(ckpt, "onnx") == "onnx"
    monkeypatch.setenv(tt.BACKEND_ENV, "onnx")              # 環境変数が引数より優先
    assert tt.predict_backend(ckpt, "torch") == "onnx"
    monkeypatch.delenv(tt.BACKEND_ENV)
    has.update(torch=False, onnxruntime=False)
    monkeypatch.setattr(tt, "train_python", lambda: None)
    assert tt.predict_backend(ckpt) is None and not tt.can_predict(ckpt)
    with pytest.raises(FileNotFoundError):
        tt.predict_in_train_env(tmp_path, ckpt)
    with pytest.raises(ValueError):
        tt.predict_backend(ckpt, "tpu")


def test_predict_onnx_matches_running_the_session(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from smash_speed.accel import build_bench_model, make_session

    model = tmp_path / "model.onnx"
    model.write_bytes(build_bench_model(in_ch=27, out_ch=8, width=4, depth=3))
    d = _clip_dir(tmp_path / "data")
    out = tt.predict_onnx(d, model, tmp_path / "heat.npy", prefer="cpu")
    heat = np.load(out)
    assert heat.shape == (2, 8, tt.HEIGHT, tt.WIDTH) and heat.dtype == np.uint8
    x = tt.ClipData(d).batch([0, 1])[0]
    sess = make_session(model, "cpu")
    ref = (sess.run(None, {sess.get_inputs()[0].name: x})[0] * 255).round().astype(np.uint8)
    np.testing.assert_array_equal(heat, ref)


@pytest.mark.skipif(not REAL_ONNX.exists(), reason="ckpts/shuttle/model_ft1.onnx がありません（onnx_export で作る）")
def test_real_onnx_model_gives_same_heat_on_gpu_and_cpu(tmp_path):
    pytest.importorskip("onnxruntime")
    d = _clip_dir(tmp_path / "data", n_clips=1)
    cpu = np.load(tt.predict_onnx(d, REAL_ONNX, tmp_path / "cpu.npy", prefer="cpu")).astype(int)
    gpu = np.load(tt.predict_onnx(d, REAL_ONNX, tmp_path / "gpu.npy", prefer="auto")).astype(int)
    assert cpu.shape == (1, 8, tt.HEIGHT, tt.WIDTH)
    assert np.abs(cpu - gpu).max() <= 1       # 0〜255 に丸めた所で 1 まで（torch とも同じ。docs/gpu.md）
