"""学習（PyTorch）に使う装置の切り替えと、学習の速さの比べ方。

TrackNetV3 の追加学習（ROADMAP の F1）は PyTorch で行う。Windows の AMD の GPU では torch-directml を使う
（``docs/gpu.md``）。torch・torch-directml は大きいので本体の ``.venv`` には入れず、別の仮想環境 ``.venv-train`` に入れる::

    py -3.12 -m venv .venv-train
    .venv-train\\Scripts\\python -m pip install torch-directml      # torch 2.4.1・torchvision 0.19.1 も入る（約 1.4 GB）

装置の選び方は推論（``smash_speed/accel.py``）と同じで、``auto``（既定。GPU があれば GPU）/ ``gpu`` / ``cpu``。
環境変数 ``SMASH_SPEED_DEVICE`` が引数より優先する。torch がない環境でも import できる（使う時に分かる）。

学習の速さの比べ方（``docs/gpu.md`` の数字はこれで測った）::

    .venv-train\\Scripts\\python -m smash_speed.train.torch_device --bench
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any

from smash_speed.accel import resolve_prefer


def _torch_directml():
    """torch_directml を返す。入っていない・GPU が見つからなければ None。"""
    try:
        import torch_directml
    except ImportError:
        return None
    try:
        return torch_directml if torch_directml.is_available() else None
    except Exception:  # ドライバの問い合わせに失敗したときは使わない
        return None


def train_device(prefer: str | None = None):
    """学習に使う ``torch.device``（GPU なら DirectML の装置）と、その名前を返す。

    ``gpu`` で GPU が見つからないときは、CPU にして警告を出す。CUDA（NVIDIA）の PyTorch が入っていればそれも使う。
    """
    import torch

    prefer = resolve_prefer(prefer)
    if prefer != "cpu":
        dml = _torch_directml()
        if dml is not None:
            return dml.device(), dml.device_name(dml.default_device()).rstrip("\x00")
        if torch.cuda.is_available():
            return torch.device("cuda"), torch.cuda.get_device_name(0)
        if prefer == "gpu":
            warnings.warn("学習に使える GPU（torch-directml か CUDA）が見つからないため、CPU で動かします",
                          RuntimeWarning, stacklevel=2)
    return torch.device("cpu"), "CPU"


def build_tracknet(in_dim: int = 27, out_dim: int = 8, width: int = 64):
    """TrackNetV3 の TrackNet と同じ層の並びのモデル（重みは乱数）。

    入力は ``seq_len`` 8 コマ ＋ 背景（``bg_mode: concat``）の RGB で 27 チャンネル、出力は 8 コマ分の熱の地図。
    畳み込み（3×3・BatchNorm・ReLU）を 2・2・3 回ずつ重ねて 3 回縮め、3 回の底のあと、広げながら縮める前の出力をつなぐ。
    ``width`` を小さくすると軽くなる（テスト用）。
    """
    import torch
    from torch import nn

    def block(cin: int, cout: int, n: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        for i in range(n):
            layers += [nn.Conv2d(cin if i == 0 else cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU()]
        return nn.Sequential(*layers)

    w = width

    class TrackNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.down1, self.down2, self.down3 = block(in_dim, w, 2), block(w, 2 * w, 2), block(2 * w, 4 * w, 3)
            self.bottleneck = block(4 * w, 8 * w, 3)
            self.up1, self.up2, self.up3 = block(12 * w, 4 * w, 3), block(6 * w, 2 * w, 2), block(3 * w, w, 2)
            self.pool = nn.MaxPool2d(2)
            self.upsample = nn.Upsample(scale_factor=2)
            self.predictor = nn.Conv2d(w, out_dim, 1)

        def forward(self, x):
            x1 = self.down1(x)
            x2 = self.down2(self.pool(x1))
            x3 = self.down3(self.pool(x2))
            x = self.bottleneck(self.pool(x3))
            x = self.up1(torch.cat([self.upsample(x), x3], dim=1))
            x = self.up2(torch.cat([self.upsample(x), x2], dim=1))
            x = self.up3(torch.cat([self.upsample(x), x1], dim=1))
            return torch.sigmoid(self.predictor(x))

    return TrackNet()


def wbce_loss(y_pred, y):
    """TrackNetV3 の重み付き 2 値交差エントロピー（外れやすい点ほど重くする）。"""
    import torch

    p = torch.clamp(y_pred, 1e-7, 1 - 1e-7)
    loss = -(torch.square(1 - p) * y * torch.log(p) + torch.square(p) * (1 - y) * torch.log(1 - p))
    return loss.mean()


def make_adam(params, lr: float = 1e-4, device=None, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
    """Adam を作る。DirectML では ``torch.optim.Adam`` の ``lerp`` が CPU に回されて遅いので、同じ式を
    ``mul_``・``add_`` で書いたもの（:class:`_PlainAdam`）を使う。"""
    import torch

    if device is not None and device.type not in ("cpu", "cuda"):
        return _plain_adam_class()(params, lr=lr, betas=betas, eps=eps)
    return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps)


def _plain_adam_class():
    import torch

    class _PlainAdam(torch.optim.Optimizer):
        """``torch.optim.Adam``（weight_decay なし・amsgrad なし）と同じ更新を、GPU で動く演算だけで行う。"""

        def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8):
            super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

        @torch.no_grad()
        def step(self, closure=None):
            loss = None if closure is None else closure()
            for group in self.param_groups:
                b1, b2 = group["betas"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    st = self.state[p]
                    if not st:
                        st["step"] = 0
                        st["exp_avg"] = torch.zeros_like(p)
                        st["exp_avg_sq"] = torch.zeros_like(p)
                    st["step"] += 1
                    m, v, g = st["exp_avg"], st["exp_avg_sq"], p.grad
                    m.mul_(b1).add_(g, alpha=1 - b1)
                    v.mul_(b2).addcmul_(g, g, value=1 - b2)
                    bc1, bc2 = 1 - b1 ** st["step"], 1 - b2 ** st["step"]
                    denom = (v.sqrt() / bc2 ** 0.5).add_(group["eps"])
                    p.addcdiv_(m, denom, value=-group["lr"] / bc1)
            return loss

    return _PlainAdam


def _batch(batch: int, in_dim: int, out_dim: int, h: int, w: int, seed: int = 1):
    """乱数の入力と、コマごとに 1 点の熱の地図（ガウス）の正解。"""
    import torch

    g = torch.Generator().manual_seed(seed)
    x = torch.rand(batch, in_dim, h, w, generator=g)
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    cy = torch.rand(batch, out_dim, 1, 1, generator=g) * h
    cx = torch.rand(batch, out_dim, 1, 1, generator=g) * w
    y = (((yy - cy) ** 2 + (xx - cx) ** 2) < 2.5 ** 2).float()
    return x, y


def _sync(device) -> None:
    """装置の計算が終わるまで待つ（DirectML・CUDA は非同期に動くため、測る前にそろえる）。"""
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type != "cpu":
        torch.tensor(0.0, device=device).item()


def bench_train(device, batch: int = 4, size: tuple[int, int] = (288, 512), steps: int = 12, warmup: int = 4,
                in_dim: int = 27, out_dim: int = 8, width: int = 64, seed: int = 0,
                torch_adam: bool = False) -> dict[str, Any]:
    """学習の 1 歩（順伝播・逆伝播・Adam の更新）にかかる時間（ミリ秒）を測る。

    ``torch_adam`` なら装置によらず ``torch.optim.Adam`` を使う（:func:`make_adam` と比べる用）。
    """
    import torch

    torch.manual_seed(seed)
    model = build_tracknet(in_dim, out_dim, width).to(device)
    model.train()
    opt = (torch.optim.Adam(model.parameters(), lr=1e-4) if torch_adam
           else make_adam(model.parameters(), lr=1e-4, device=device))
    x, y = _batch(batch, in_dim, out_dim, *size)
    x, y = x.to(device), y.to(device)
    times, losses = [], []
    for i in range(warmup + steps):
        _sync(device)
        t0 = time.perf_counter()
        opt.zero_grad()
        loss = wbce_loss(model(x), y)
        loss.backward()
        opt.step()
        value = float(loss.item())  # 値を取り出すと装置の計算が終わるまで待つ
        if i >= warmup:
            times.append((time.perf_counter() - t0) * 1000)
            losses.append(value)
    times.sort()
    median = times[len(times) // 2]
    return {"device": str(device), "batch": batch, "size": list(size), "median_ms": median, "min_ms": times[0],
            "samples_per_s": batch / (median / 1000), "losses": losses}


def compare_step(device, batch: int = 2, size: tuple[int, int] = (288, 512), width: int = 64,
                 in_dim: int = 27, out_dim: int = 8, seed: int = 0) -> dict[str, float]:
    """同じ重み・同じ入力で 1 回の順伝播・逆伝播を CPU と ``device`` で行い、損失と勾配の差を返す。"""
    import torch

    torch.manual_seed(seed)
    ref = build_tracknet(in_dim, out_dim, width)
    other = build_tracknet(in_dim, out_dim, width)
    other.load_state_dict(ref.state_dict())
    other.to(device)
    x, y = _batch(batch, in_dim, out_dim, *size)
    out = {}
    grads = []
    for model, dev in ((ref, torch.device("cpu")), (other, device)):
        model.train()
        loss = wbce_loss(model(x.to(dev)), y.to(dev))
        loss.backward()
        out[dev.type] = float(loss.item())
        grads.append(torch.cat([p.grad.detach().to("cpu").flatten() for p in model.parameters()]))
    loss_cpu, loss_dev = out["cpu"], out[device.type]
    gdiff = (grads[0] - grads[1]).norm() / grads[0].norm().clamp_min(1e-12)
    return {"loss_cpu": loss_cpu, "loss_device": loss_dev, "loss_rel_diff": abs(loss_cpu - loss_dev) / abs(loss_cpu),
            "grad_rel_diff": float(gdiff)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="学習に使う装置を表示し、TrackNetV3 と同じ形のモデルで学習の速さを GPU と CPU で比べる")
    p.add_argument("--bench", action="store_true", help="学習の 1 歩の時間を GPU と CPU で比べる")
    p.add_argument("--batch", type=int, nargs="+", default=[1, 2],
                   help="一度に学習するコマの組の数（DirectML・20 GB では 4 で落ちることがあり、6 からメモリがあふれる）")
    p.add_argument("--size", type=int, nargs=2, default=[288, 512], metavar=("H", "W"))
    p.add_argument("--width", type=int, default=64, help="最初の層のチャンネル数（TrackNetV3 は 64）")
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--warmup", type=int, default=4, help="測る前に空で動かす歩数")
    p.add_argument("--cpu-steps", type=int, default=2, help="CPU は遅いので歩数を減らす")
    p.add_argument("--json", type=Path, help="結果を JSON で書き出す")
    args = p.parse_args(argv)

    import torch

    device, name = train_device()
    print(f"torch {torch.__version__}、学習に使う装置: {name}（{device}）")
    if not args.bench:
        return 0
    size = tuple(args.size)
    res: dict[str, Any] = {"torch": torch.__version__, "device_name": name, "size": list(size), "width": args.width}
    if device.type != "cpu":
        res["compare"] = compare_step(device, size=size, width=args.width)
        c = res["compare"]
        print(f"CPU との差：損失 {c['loss_rel_diff']:.1e}、勾配 {c['grad_rel_diff']:.1e}（相対）")
    res["runs"] = []
    for b in args.batch:
        row = {"batch": b}
        if device.type != "cpu":
            row["gpu"] = bench_train(device, batch=b, size=size, steps=args.steps, warmup=args.warmup, width=args.width)
            row["gpu_torch_adam"] = bench_train(device, batch=b, size=size, steps=args.steps, warmup=args.warmup,
                                                width=args.width, torch_adam=True)
        row["cpu"] = bench_train(torch.device("cpu"), batch=b, size=size, steps=args.cpu_steps, warmup=2,
                                 width=args.width)
        g, c = row.get("gpu"), row["cpu"]
        msg = f"1 歩（{b} 組）: CPU {c['median_ms']:.0f} ms"
        if g:
            msg += (f" / GPU {g['median_ms']:.0f} ms（{c['median_ms'] / g['median_ms']:.1f} 倍、"
                    f"torch の Adam では {row['gpu_torch_adam']['median_ms']:.0f} ms）、GPU {g['samples_per_s']:.1f} 組/秒")
        print(msg)
        res["runs"].append(row)
    if args.json:
        args.json.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
