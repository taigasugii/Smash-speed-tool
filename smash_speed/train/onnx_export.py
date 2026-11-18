"""学習したシャトルの検出（``tracknet_train`` の ``model.pt``）を ONNX に書き出す（torch のある ``.venv-train`` で動かす）。

書き出した ``.onnx`` を ``model.pt`` の隣（同じ名前で拡張子だけ ``.onnx``）に置くと、torch のない ``.venv`` でも
onnxruntime（GPU なら DirectML）で同じ熱の地図を作れる（``tracknet_train.predict_onnx``、``docs/gpu.md``）::

    .venv-train\\Scripts\\python -m smash_speed.train.onnx_export ckpts/shuttle/model_ft1.pt

入力は ``(区間の数, 27, 288, 512)``（区間の数は自由）、出力は ``(区間の数, 8, 288, 512)`` の 0〜1。
書き出した後、乱数の入力で torch（CPU）の出力と比べた差を表示する（onnxruntime があればそれでも比べる）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from smash_speed.train.tracknet_train import HEIGHT, WIDTH

OPSET = 17
INPUT_NAME, OUTPUT_NAME = "frames", "heat"


def onnx_path_for(ckpt: str | Path) -> Path:
    """``model.pt`` に対応する ONNX のパス（同じ場所・同じ名前で ``.onnx``）。"""
    return Path(ckpt).with_suffix(".onnx")


def load_model(ckpt: Path):
    """``tracknet_train train`` の ``model.pt`` を CPU のモデルにして返す（推論の形）。"""
    import torch

    from smash_speed.train.torch_device import build_tracknet

    c = torch.load(ckpt, map_location="cpu", weights_only=True)  # 重みの数字だけを読む
    pd = c.get("param_dict", {})
    seq_len = int(pd.get("seq_len", 8))
    model = build_tracknet(in_dim=3 * (seq_len + 1), out_dim=seq_len, width=int(pd.get("width", 64)))
    model.load_state_dict(c["model"])
    return model.eval(), seq_len


def export(ckpt: Path, out: Path | None = None, check: bool = True) -> dict:
    """``ckpt`` を ONNX（``out``、既定は ``onnx_path_for(ckpt)``）に書き出し、確かめた差を返す。"""
    import torch

    ckpt = Path(ckpt)
    out = Path(out) if out is not None else onnx_path_for(ckpt)
    model, seq_len = load_model(ckpt)
    in_dim = 3 * (seq_len + 1)
    x = torch.from_numpy(np.random.default_rng(0).random((2, in_dim, HEIGHT, WIDTH), dtype=np.float32))
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(model, (x,), str(out), input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
                          dynamic_axes={INPUT_NAME: {0: "n"}, OUTPUT_NAME: {0: "n"}}, opset_version=OPSET,
                          do_constant_folding=True, dynamo=False)
    info = {"onnx": str(out), "seq_len": seq_len, "in_dim": in_dim, "mb": round(out.stat().st_size / 1e6, 1)}
    if check:
        with torch.no_grad():
            ref = model(x).numpy()
        try:
            import onnxruntime as ort
        except ImportError:
            info["check"] = "onnxruntime がないので比べていない（.venv の predict_onnx で比べる）"
        else:
            sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
            y = sess.run(None, {INPUT_NAME: x.numpy()})[0]
            info["max_abs_diff_cpu"] = float(np.max(np.abs(y - ref)))
    return info


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="学習したシャトルの検出（model.pt）を ONNX に書き出す（.venv-train）")
    p.add_argument("ckpt", type=Path)
    p.add_argument("--out", type=Path, help="既定は model.pt の隣の .onnx")
    a = p.parse_args(argv)
    print(json.dumps(export(a.ckpt, a.out), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
