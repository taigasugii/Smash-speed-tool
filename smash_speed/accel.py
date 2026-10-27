"""推論に使う装置（GPU / CPU）の切り替え。

ONNX のモデルを onnxruntime で動かすときに、GPU（Windows では DirectML、AMD / NVIDIA / Intel のどれでも可）が
使えれば GPU、使えなければ CPU にする。onnxruntime が入っていない PC でも import できる（使う時に分かる）。

装置の選び方（``prefer``）：

- ``"auto"``（既定）：GPU があれば GPU、なければ CPU
- ``"gpu"``：GPU を使う。使えなければ CPU に戻して警告を出す
- ``"cpu"``：CPU だけ

環境変数 ``SMASH_SPEED_DEVICE`` に ``auto`` / ``gpu`` / ``cpu`` を入れると、引数より優先する。

速さの比べ方（``docs/gpu.md`` の数字はこれで測った）::

    python -m smash_speed.accel --bench
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ENV_VAR = "SMASH_SPEED_DEVICE"
# GPU として使う実行プロバイダ（先にあるものを優先する）
GPU_PROVIDERS = ("DmlExecutionProvider", "CUDAExecutionProvider", "ROCMExecutionProvider", "MIGraphXExecutionProvider")
CPU_PROVIDER = "CPUExecutionProvider"
_PREFERS = ("auto", "gpu", "cpu")


def _ort():
    """onnxruntime を返す。入っていなければ None。"""
    try:
        import onnxruntime
    except ImportError:
        return None
    return onnxruntime


def available_providers() -> list[str]:
    """この PC の onnxruntime で使える実行プロバイダの一覧（onnxruntime がなければ空）。"""
    ort = _ort()
    return list(ort.get_available_providers()) if ort is not None else []


def resolve_prefer(prefer: str | None = None) -> str:
    """環境変数と引数から、装置の選び方（auto / gpu / cpu）を決める。"""
    value = os.environ.get(ENV_VAR) or prefer or "auto"
    value = value.strip().lower()
    if value not in _PREFERS:
        raise ValueError(f"装置の指定は {', '.join(_PREFERS)} のどれかにしてください: {value!r}")
    return value


def choose_providers(prefer: str | None = None, available: list[str] | None = None) -> list[str]:
    """onnxruntime に渡す実行プロバイダの並び（先頭が使いたい装置、最後は必ず CPU）。

    Args:
        prefer: ``auto`` / ``gpu`` / ``cpu``。None なら環境変数か ``auto``。
        available: 使えるプロバイダの一覧（テスト用）。None ならこの PC の一覧。
    """
    prefer = resolve_prefer(prefer)
    if available is None:
        available = available_providers()
    if prefer == "cpu":
        return [CPU_PROVIDER]
    gpu = [p for p in GPU_PROVIDERS if p in available]
    if not gpu:
        if prefer == "gpu":
            warnings.warn("GPU を使える onnxruntime が見つからないため、CPU で動かします", RuntimeWarning, stacklevel=2)
        return [CPU_PROVIDER]
    return [gpu[0], CPU_PROVIDER]


@dataclass
class DeviceInfo:
    """推論に使える装置の情報（GUI や記録に出す用）。"""

    provider: str          # 使う実行プロバイダ
    is_gpu: bool
    name: str              # 装置の名前（分からなければプロバイダ名）
    memory_mb: int | None  # GPU のメモリ（分かるときだけ）


def _device_names() -> dict[str, tuple[str, int | None]]:
    """実行プロバイダごとの装置の名前とメモリ（onnxruntime 1.22 以降で分かる。分からなければ空）。"""
    ort = _ort()
    out: dict[str, tuple[str, int | None]] = {}
    if ort is None or not hasattr(ort, "get_ep_devices"):
        return out
    try:
        for d in ort.get_ep_devices():
            meta = dict(d.device.metadata)
            mem = meta.get("DxgiVideoMemory", "").split(" ")[0]
            # 同じプロバイダで装置が複数あるとき（内蔵 GPU など）は、最初（高性能の方）を採る
            out.setdefault(d.ep_name, (meta.get("Description") or d.device.vendor, int(mem) if mem.isdigit() else None))
    except Exception:  # 装置の問い合わせに失敗しても、推論そのものは動かせる
        return {}
    return out


def device_info(prefer: str | None = None) -> DeviceInfo:
    """``prefer`` のときに実際に使われる装置。"""
    provider = choose_providers(prefer)[0]
    name, mem = _device_names().get(provider, (provider, None))
    return DeviceInfo(provider=provider, is_gpu=provider != CPU_PROVIDER, name=name, memory_mb=mem)


def make_session(model: str | Path | bytes, prefer: str | None = None, **session_kwargs: Any):
    """ONNX のモデルを読み込んだ ``onnxruntime.InferenceSession`` を作る。

    GPU で作れなかったとき（ドライバの不具合・メモリ不足など）は、CPU で作り直して警告を出す。
    実際に使っているプロバイダは ``session.get_providers()[0]`` で分かる。
    """
    ort = _ort()
    if ort is None:
        raise ImportError("onnxruntime が入っていません（pip install onnxruntime-directml）")
    providers = choose_providers(prefer)
    src = model if isinstance(model, bytes) else str(model)
    opts = session_kwargs.pop("sess_options", None) or ort.SessionOptions()
    if providers[0] == "DmlExecutionProvider":
        # DirectML はメモリの並べ方の最適化と並列実行に対応していない（onnxruntime の説明どおり）
        opts.enable_mem_pattern = False
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    try:
        return ort.InferenceSession(src, sess_options=opts, providers=providers, **session_kwargs)
    except Exception as e:
        if providers == [CPU_PROVIDER]:
            raise
        warnings.warn(f"GPU（{providers[0]}）で読み込めなかったため、CPU で動かします: {e}", RuntimeWarning,
                      stacklevel=2)
        return ort.InferenceSession(src, sess_options=ort.SessionOptions(), providers=[CPU_PROVIDER],
                                    **session_kwargs)


# --- 速さの比べ方 -----------------------------------------------------------------

def build_bench_model(in_ch: int = 9, out_ch: int = 3, width: int = 64, depth: int = 3) -> bytes:
    """速さを比べるための、TrackNet に似た形（畳み込み → 縮小 → 拡大）の ONNX モデルを作る（重みは乱数）。

    ``onnx`` が必要（``pip install onnx``）。
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    nodes, inits = [], []
    cur, ch = "x", in_ch

    def conv(name: str, src: str, cin: int, cout: int) -> str:
        w = (rng.standard_normal((cout, cin, 3, 3)) * np.sqrt(2.0 / (cin * 9))).astype(np.float32)
        inits.append(numpy_helper.from_array(w, f"{name}_w"))
        inits.append(numpy_helper.from_array(np.zeros(cout, np.float32), f"{name}_b"))
        nodes.append(helper.make_node("Conv", [src, f"{name}_w", f"{name}_b"], [f"{name}_c"], pads=[1, 1, 1, 1]))
        nodes.append(helper.make_node("Relu", [f"{name}_c"], [name]))
        return name

    skips = []
    for i in range(depth):
        cout = width * 2 ** i
        cur = conv(f"d{i}a", cur, ch, cout)
        cur = conv(f"d{i}b", cur, cout, cout)
        skips.append((cur, cout))
        nodes.append(helper.make_node("MaxPool", [cur], [f"p{i}"], kernel_shape=[2, 2], strides=[2, 2]))
        cur, ch = f"p{i}", cout
    cur = conv("mid", cur, ch, ch * 2)
    ch *= 2
    scales = numpy_helper.from_array(np.array([1, 1, 2, 2], np.float32), "up_scales")
    inits.append(scales)
    for i in reversed(range(depth)):
        skip, sch = skips[i]
        nodes.append(helper.make_node("Resize", [cur, "", "up_scales"], [f"u{i}"], mode="nearest"))
        nodes.append(helper.make_node("Concat", [f"u{i}", skip], [f"cat{i}"], axis=1))
        cur = conv(f"u{i}a", f"cat{i}", ch + sch, sch)
        ch = sch
    w = (rng.standard_normal((out_ch, ch, 1, 1)) * 0.1).astype(np.float32)
    inits.append(numpy_helper.from_array(w, "head_w"))
    nodes.append(helper.make_node("Conv", [cur, "head_w"], ["head"]))
    nodes.append(helper.make_node("Sigmoid", ["head"], ["y"]))
    graph = helper.make_graph(
        nodes, "bench", [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["n", in_ch, "h", "w"])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["n", out_ch, "h", "w"])], inits)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m.SerializeToString()


def bench(model: bytes, prefer: str, shape: tuple[int, ...], repeats: int = 20, warmup: int = 3) -> dict[str, Any]:
    """1 回の推論にかかる時間（ミリ秒）を測る。"""
    sess = make_session(model, prefer)
    x = np.random.default_rng(1).random(shape, dtype=np.float32)
    name = sess.get_inputs()[0].name
    for _ in range(warmup):
        y = sess.run(None, {name: x})[0]
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        y = sess.run(None, {name: x})[0]
        times.append((time.perf_counter() - t0) * 1000)
    return {"provider": sess.get_providers()[0], "shape": shape, "median_ms": float(np.median(times)),
            "min_ms": float(np.min(times)), "output": y}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="推論に使う装置を表示し、GPU と CPU の速さを比べる")
    p.add_argument("--bench", action="store_true", help="TrackNet に似た大きさのモデルで GPU と CPU の速さを比べる")
    p.add_argument("--repeats", type=int, default=20)
    args = p.parse_args(argv)

    print("使えるプロバイダ:", ", ".join(available_providers()) or "（onnxruntime がありません）")
    info = device_info()
    mem = f"、メモリ {info.memory_mb} MB" if info.memory_mb else ""
    print(f"使う装置: {info.name}（{info.provider}{mem}）")
    if not args.bench:
        return 0
    model = build_bench_model()
    # TrackNetV3 の入力（288×512、3 コマ × RGB = 9 チャンネル）と、その 4 倍の大きさ
    for shape in [(1, 9, 288, 512), (8, 9, 288, 512), (1, 9, 576, 1024)]:
        res = {pr: bench(model, pr, shape, repeats=args.repeats) for pr in ("cpu", "gpu")}
        diff = float(np.max(np.abs(res["cpu"]["output"] - res["gpu"]["output"])))
        c, g = res["cpu"]["median_ms"], res["gpu"]["median_ms"]
        print(f"入力 {shape}: CPU {c:.1f} ms / {res['gpu']['provider']} {g:.1f} ms"
              f"（{c / g:.1f} 倍）、出力の差の最大 {diff:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
