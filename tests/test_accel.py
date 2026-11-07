"""推論に使う装置（GPU / CPU）の切り替えのテスト。"""

import numpy as np
import pytest

from smash_speed import accel

DML = "DmlExecutionProvider"
CPU = "CPUExecutionProvider"


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv(accel.ENV_VAR, raising=False)


def test_choose_providers_prefers_gpu_when_available():
    assert accel.choose_providers("auto", [DML, CPU]) == [DML, CPU]
    assert accel.choose_providers("gpu", [DML, CPU]) == [DML, CPU]
    assert accel.choose_providers("cpu", [DML, CPU]) == [CPU]


def test_choose_providers_falls_back_to_cpu():
    """GPU がない PC では CPU になる。gpu を指定したときだけ警告を出す。"""
    assert accel.choose_providers("auto", [CPU]) == [CPU]
    assert accel.choose_providers("auto", []) == [CPU]
    with pytest.warns(RuntimeWarning):
        assert accel.choose_providers("gpu", [CPU]) == [CPU]


def test_choose_providers_order():
    """GPU のプロバイダが複数あるときは GPU_PROVIDERS の順で選ぶ。CPU は必ず最後に入る。"""
    assert accel.choose_providers("auto", [CPU, "CUDAExecutionProvider", DML]) == [DML, CPU]
    assert accel.choose_providers("auto", ["CUDAExecutionProvider", CPU]) == ["CUDAExecutionProvider", CPU]


def test_env_var_overrides_argument(monkeypatch):
    monkeypatch.setenv(accel.ENV_VAR, "CPU")
    assert accel.choose_providers("gpu", [DML, CPU]) == [CPU]
    monkeypatch.setenv(accel.ENV_VAR, "fast")
    with pytest.raises(ValueError):
        accel.choose_providers(None, [DML, CPU])


def test_device_info_without_onnxruntime(monkeypatch):
    """onnxruntime が入っていない PC でも、CPU として情報を返す。"""
    monkeypatch.setattr(accel, "_ort", lambda: None)
    info = accel.device_info()
    assert info.provider == CPU and not info.is_gpu
    with pytest.raises(ImportError):
        accel.make_session(b"")


@pytest.fixture(scope="module")
def small_model():
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    return accel.build_bench_model(in_ch=3, out_ch=1, width=4, depth=2)


def test_make_session_runs_on_cpu_and_auto(small_model):
    """CPU と auto（GPU があれば GPU）で同じ結果になる。"""
    x = np.random.default_rng(0).random((1, 3, 32, 48), dtype=np.float32)
    outs = {}
    for prefer in ("cpu", "auto"):
        sess = accel.make_session(small_model, prefer)
        outs[prefer] = sess.run(None, {"x": x})[0]
        assert outs[prefer].shape == (1, 1, 32, 48)
    assert accel.make_session(small_model, "cpu").get_providers() == [CPU]
    np.testing.assert_allclose(outs["cpu"], outs["auto"], atol=1e-4)


def test_make_session_falls_back_when_gpu_fails(small_model, monkeypatch):
    """GPU でセッションを作れないときは、警告を出して CPU で作り直す。"""
    ort = accel._ort()
    real = ort.InferenceSession

    def fake(src, sess_options=None, providers=None, **kw):
        if providers and providers[0] != CPU:
            raise RuntimeError("ドライバの不具合（テスト用）")
        return real(src, sess_options=sess_options, providers=providers, **kw)

    monkeypatch.setattr(ort, "InferenceSession", fake)
    monkeypatch.setattr(accel, "available_providers", lambda: [DML, CPU])
    with pytest.warns(RuntimeWarning, match="CPU"):
        sess = accel.make_session(small_model, "gpu")
    assert sess.get_providers() == [CPU]
