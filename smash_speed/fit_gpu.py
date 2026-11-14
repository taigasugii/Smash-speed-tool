"""軌道の積分（``fit_batch.integrate`` と同じ RK4）を、ONNX のグラフにして onnxruntime で動かす。

``fit_batch`` の当てはめは、積分だけを差し替えられる（``fit_batch.get_integrator("dml")``）。ここでは、RK4 の
``SEGMENT`` 歩分を 1 つのグラフに展開し（繰り返しの Loop は DirectML では GPU に載らないため）、長い軌道はグラフを
何回か続けて呼ぶ。DirectML（GPU）は単精度、CPU は倍精度で動く。

測った結果（``docs/gpu.md`` の「軌道の当てはめ」）：この問題では GPU は速くならない。1 回の呼び出しごとの手間が、
まとめた当てはめの数によらず NumPy の何倍もあり、当てはめの時間のほとんどは「残った数本の当てはめの繰り返し」
だからである。``backend="gpu"`` は確かめるために残してあり、既定では使わない。
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

SEGMENT = 32                    # 1 つのグラフに展開する RK4 の歩数
_sessions: dict[tuple[str, int], Any] = {}
_reason: list[str] = []         # 使えない理由（``available`` が 1 回だけ調べる）


def build_model(n_steps: int = SEGMENT, double: bool = False) -> Any:
    """RK4 を ``n_steps`` 歩進めるグラフ（``onnx.ModelProto``）。

    入力：``w``（速度と初速についての傾き。抵抗係数 k をかけた値）(1, B, 3, 2)、``p``（位置）(1, B, 3, 2)、
    ``kg``（k × g を高さの向きに置いたもの）(1, B, 3, 2)、``steps``（歩ごとの h, h/2, h/6, h²/6）(4 × n_steps,)。
    出力：``ps``・``ws``（1 歩ごとの位置・速度）(n_steps, B, 3, 2)。
    """
    import onnx
    from onnx import TensorProto, helper

    dt = TensorProto.DOUBLE if double else TensorProto.FLOAT
    np_dt = np.float64 if double else np.float32
    nodes: list[Any] = []
    count = [0]

    def op(kind: str, *inputs: str, **attrs: Any) -> str:
        count[0] += 1
        out = f"t{count[0]}"
        nodes.append(helper.make_node(kind, list(inputs), [out], **attrs))
        return out

    def const(name: str, value: np.ndarray) -> Any:
        return helper.make_tensor(name, TensorProto.INT64 if value.dtype == np.int64 else dt, value.shape,
                                  value.flatten().tolist())

    init = [const("mask", np.array([0.0, 1.0, 1.0], np_dt).reshape(1, 1, 3, 1)),
            const("tiny", np.array([1e-30], np_dt)),
            const("i0", np.array([0], np.int64)), const("i1", np.array([1], np.int64)),
            const("ax2", np.array([2], np.int64)), const("ax3", np.array([3], np.int64))]

    def acc(w: str) -> str:
        # w (1, B, 3, 2)：行 0 が k × 速度、行 1・2 が初速についての傾き
        w0 = op("Slice", w, "i0", "i1", "ax2")
        dots = op("ReduceSum", op("Mul", w, w0), "ax3", keepdims=1)
        q = op("Max", op("Sqrt", op("Slice", dots, "i0", "i1", "ax2")), "tiny")
        coef = op("Mul", op("Div", dots, q), "mask")
        return op("Sub", "neg_kg", op("Add", op("Mul", coef, w0), op("Mul", q, w)))

    nodes.append(helper.make_node("Neg", ["kg"], ["neg_kg"]))
    nodes.append(helper.make_node("Split", ["steps"], [f"s{i}" for i in range(4 * n_steps)], axis=0,
                                  num_outputs=4 * n_steps))
    w, p = "w", "p"
    ps, ws = [], []
    for i in range(n_steps):
        h, hh, h6, h26 = (f"s{4 * i + j}" for j in range(4))
        a1 = acc(w)
        a2 = acc(op("Add", w, op("Mul", a1, hh)))
        a3 = acc(op("Add", w, op("Mul", a2, hh)))
        a4 = acc(op("Add", w, op("Mul", a3, h)))
        s23 = op("Add", a2, a3)
        s123 = op("Add", a1, s23)
        p = op("Add", op("Add", p, op("Mul", w, h)), op("Mul", s123, h26))
        w = op("Add", w, op("Mul", op("Add", op("Add", s123, s23), a4), h6))
        ps.append(p)
        ws.append(w)
    nodes.append(helper.make_node("Concat", ps, ["ps"], axis=0))
    nodes.append(helper.make_node("Concat", ws, ["ws"], axis=0))
    shape = [1, "B", 3, 2]
    graph = helper.make_graph(
        nodes, "rk4",
        [helper.make_tensor_value_info(n, dt, shape) for n in ("w", "p", "kg")]
        + [helper.make_tensor_value_info("steps", dt, [4 * n_steps])],
        [helper.make_tensor_value_info(n, dt, [n_steps, "B", 3, 2]) for n in ("ps", "ws")], init)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


def session(provider: str = "dml", n_steps: int = SEGMENT) -> Any:
    """グラフを動かすセッション（``provider`` は "dml"：GPU の単精度、"cpu"：CPU の倍精度）。作ったものは使い回す。"""
    key = (provider, n_steps)
    if key not in _sessions:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.log_severity_level = 3
        if provider == "dml":
            so.enable_mem_pattern = False       # DirectML の決まり
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        else:
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            providers = ["CPUExecutionProvider"]
        model = build_model(n_steps, double=provider != "dml")
        _sessions[key] = ort.InferenceSession(model.SerializeToString(), so, providers=providers)
    return _sessions[key]


def available() -> bool:
    """GPU（DirectML）で積分できるか。できない理由は ``why_not`` で分かる。"""
    if not _reason:
        try:
            import onnxruntime as ort

            if "DmlExecutionProvider" not in ort.get_available_providers():
                _reason.append("onnxruntime に DirectML がありません（pip install -e .[gpu]）")
            else:
                sess = session("dml")
                _reason.append("" if "DmlExecutionProvider" in sess.get_providers()
                               else "DirectML でグラフを読み込めませんでした")
        except Exception as e:      # noqa: BLE001  入っていない・ドライバの不具合など、どれでも CPU で続ける
            _reason.append(f"{type(e).__name__}: {e}")
        if _reason[0]:
            warnings.warn(f"軌道の当てはめの積分を GPU で動かせないので、CPU で計算します：{_reason[0]}", stacklevel=2)
    return not _reason[0]


def why_not() -> str:
    """GPU で積分できない理由（できるなら空）。"""
    available()
    return _reason[0]


def integrate(v0: np.ndarray, k: np.ndarray, g: np.ndarray, grid: np.ndarray, n_steps: np.ndarray | None = None,
              provider: str = "dml") -> tuple[np.ndarray, np.ndarray]:
    """``fit_batch.integrate`` と同じ（``n_steps`` は使わず、全部の軌道を最後まで進める）。"""
    sess = session(provider)
    dtype = np.float32 if provider == "dml" else np.float64
    b, n = len(v0), len(grid) - 1
    w = np.zeros((1, b, 3, 2), dtype)
    w[0, :, 0, :] = v0 * k[:, None]
    w[0, :, 1, 0] = 1.0
    w[0, :, 2, 1] = 1.0
    p = np.zeros((1, b, 3, 2), dtype)
    kg = np.zeros((1, b, 3, 2), dtype)
    kg[0, :, 0, 1] = k * g
    pos = np.empty((n + 1, 3, b), complex)
    vel = np.empty((n + 1, 3, b), complex)

    def packed(x: np.ndarray) -> np.ndarray:
        """(歩, B, 3, 2) の実数を、``fit_batch.integrate`` と同じ (歩, 3, B) の複素数に。"""
        x = x.astype(float)
        return (x[..., 0] + 1j * x[..., 1]).transpose(0, 2, 1)

    pos[0], vel[0] = packed(p)[0], packed(w)[0]
    h = np.diff(grid)
    for at in range(0, n, SEGMENT):
        hs = np.zeros(SEGMENT)
        m = min(SEGMENT, n - at)
        hs[:m] = h[at:at + m]
        steps = np.stack([hs, hs / 2, hs / 6, hs * hs / 6], axis=1).ravel().astype(dtype)
        ps, ws = sess.run(None, {"w": w, "p": p, "kg": kg, "steps": steps})
        pos[at + 1:at + m + 1], vel[at + 1:at + m + 1] = packed(ps[:m]), packed(ws[:m])
        p, w = ps[m - 1:m], ws[m - 1:m]
    # k をかけた値で進めたので、戻す（傾きの行は、初速も k 倍なので、そのまま）
    pos[:, 0] /= k
    vel[:, 0] /= k
    return pos, vel


def integrate_cpu(v0: np.ndarray, k: np.ndarray, g: np.ndarray, grid: np.ndarray, n_steps: np.ndarray | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
    """同じグラフを CPU の倍精度で（確かめる用）。"""
    return integrate(v0, k, g, grid, n_steps, provider="cpu")
