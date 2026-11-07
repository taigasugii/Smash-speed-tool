"""学習（PyTorch）に使う装置の切り替えと、学習の 1 歩のテスト。

torch は本体の ``.venv`` に入れないので、torch を使うテストは学習用の仮想環境 ``.venv-train`` の Python で
別に動かす（``.venv-train`` がない PC では飛ばす）。
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = ROOT / ".venv-train" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def test_imports_without_torch():
    """torch がない環境でも import できる（使う時に分かる）。"""
    from smash_speed.train import torch_device

    assert callable(torch_device.train_device)


def _run_in_train_env(code: str, device: str = "cpu") -> dict:
    """``.venv-train`` の Python でコードを動かし、最後の行の JSON を返す。"""
    if not TRAIN_PY.exists():
        pytest.skip(".venv-train がない（torch-directml を入れた学習用の仮想環境）")
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONIOENCODING="utf-8", SMASH_SPEED_DEVICE=device)
    r = subprocess.run([str(TRAIN_PY), "-W", "ignore", "-c", code], capture_output=True, text=True,
                       encoding="utf-8", env=env, cwd=ROOT, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_plain_adam_matches_torch_adam():
    """GPU 用の Adam（lerp を使わない）は、CPU で torch.optim.Adam と同じ重みになる。

    BatchNorm の前の bias は勾配が丸めの誤差だけで、Adam がそれを lr の大きさに広げてしまうので、BatchNorm のないモデルで比べる。
    """
    out = _run_in_train_env("""
import json, torch
from smash_speed.train import torch_device as td
torch.manual_seed(0)
def net():
    return torch.nn.Sequential(torch.nn.Conv2d(27, 8, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(8, 8, 1),
                               torch.nn.Sigmoid())
a = net(); b = net(); b.load_state_dict(a.state_dict())
oa = torch.optim.Adam(a.parameters(), lr=1e-3); ob = td._plain_adam_class()(b.parameters(), lr=1e-3)
x, y = td._batch(2, 27, 8, 32, 64)
for _ in range(5):
    for m, o in ((a, oa), (b, ob)):
        o.zero_grad(); td.wbce_loss(m(x), y).backward(); o.step()
d = max(float((p - q).abs().max()) for p, q in zip(a.parameters(), b.parameters()))
print(json.dumps({"max_diff": d, "type": type(td.make_adam(b.parameters(), device=torch.device("cpu"))).__name__}))
""")
    assert out["max_diff"] < 1e-5
    assert out["type"] == "Adam"


def test_tracknet_shape_and_loss_goes_down_on_cpu():
    """TrackNetV3 と同じ形：27 チャンネル → 8 枚の熱の地図。同じ入力で学習すると損失が下がる。"""
    out = _run_in_train_env("""
import json, torch
from smash_speed.train import torch_device as td
dev, name = td.train_device()
m = td.build_tracknet(width=4)
y = m(torch.rand(1, 27, 32, 64))
r = td.bench_train(dev, batch=2, size=(32, 64), steps=20, warmup=0, width=4)
print(json.dumps({"device": str(dev), "name": name, "shape": list(y.shape), "losses": r["losses"]}))
""", device="cpu")
    assert out["device"] == "cpu" and out["name"] == "CPU"
    assert out["shape"] == [1, 8, 32, 64]
    assert out["losses"][-1] < 0.98 * out["losses"][0]


def test_gpu_step_matches_cpu_when_available():
    """GPU（DirectML）があれば、同じ重み・入力の損失と勾配が CPU とほぼ同じになる。"""
    out = _run_in_train_env("""
import json
from smash_speed.train import torch_device as td
dev, name = td.train_device()
res = {"device": dev.type}
if dev.type != "cpu":
    res.update(td.compare_step(dev, size=(64, 128), width=8))
print(json.dumps(res))
""", device="auto")
    if out["device"] == "cpu":
        pytest.skip("学習に使える GPU がない")
    assert out["loss_rel_diff"] < 1e-4
    assert out["grad_rel_diff"] < 2e-2
