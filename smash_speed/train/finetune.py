"""TrackNetV3 のファインチューニング（公開されている学習済み重みから）。

TrackNetV3 の ``train.py`` には任意の重みから始めるオプションがなく、``--resume_training`` で
``<save_dir>/TrackNet_cur.pt`` を読み込む仕組みだけがある。そこで学習済みの ``TrackNet_best.pt`` を
``TrackNet_cur.pt`` としてコピーし、エポック数を（学習済みのエポック + 追加分）にして再開する。

学習の設定は ``configs/train.yaml`` の ``train`` 節に書く。GPU がない場合は
``notebooks/finetune_tracknetv3_colab.ipynb`` で Google Colab を使う。
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from smash_speed.repro import run_metadata  # noqa: E402


def build_commands(cfg: dict[str, Any]) -> tuple[list[list[str]], Path, Path]:
    """実行するコマンド列・TrackNetV3 のディレクトリ・保存先を返す（実行はしない）。"""
    t = cfg["train"]
    repo = Path(cfg["detection"]["tracknet"]["repo_dir"] or t.get("repo_dir", ""))
    save_dir = repo / t.get("save_dir", "exp_finetune")
    py = cfg["detection"]["tracknet"].get("python") or sys.executable
    cmds = []
    if t.get("run_preprocess", True):
        cmds.append([py, "preprocess.py"])
    cmds.append([py, "train.py", "--model_name", "TrackNet", "--seq_len", str(t.get("seq_len", 8)),
                 "--epochs", "__EPOCHS__", "--batch_size", str(t.get("batch_size", 10)),
                 "--learning_rate", str(t.get("learning_rate", 1e-4)), "--bg_mode", t.get("bg_mode", "concat"),
                 "--alpha", str(t.get("mixup_alpha", 0.5)), "--tolerance", str(t.get("tolerance_px", 4)),
                 "--seed", str(cfg.get("seed", 13)), "--save_dir", str(save_dir.name),
                 "--resume_training", "--verbose"])
    return cmds, repo, save_dir


def prepare_resume(pretrained: Path, save_dir: Path, extra_epochs: int) -> int:
    """学習済み重みを ``TrackNet_cur.pt`` として置き、再開後の総エポック数を返す。"""
    import torch

    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(pretrained, map_location="cpu")
    start_epoch = int(ckpt.get("epoch", 0))
    # ファインチューニングでは最良の検証精度をリセットし、新しいデータでの最良を保存させる
    ckpt["max_val_acc"] = 0.0
    torch.save(ckpt, save_dir / "TrackNet_cur.pt")
    return start_epoch + 1 + extra_epochs


_NUM = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"


def parse_log(text: str) -> dict[str, list[float]]:
    """学習ログから ``loss`` / ``acc`` などの数値を拾う（ログの書式に依存しないよう正規表現で）。"""
    series: dict[str, list[float]] = {}
    for key in ("train_loss", "val_loss", "loss", "val_acc", "acc", "f1", "precision", "recall"):
        vals = [float(v) for v in re.findall(rf"\b{key}\b\s*[:=]\s*{_NUM}", text, flags=re.IGNORECASE)]
        if vals:
            series[key] = vals
    return series


def plot_log(series: dict[str, list[float]], path: Path) -> None:
    if not series:
        return
    fig, axes = plt.subplots(1, len(series), figsize=(4 * len(series), 3), squeeze=False)
    for ax, (k, v) in zip(axes[0], series.items()):
        ax.plot(range(1, len(v) + 1), v, "o-", ms=3)
        ax.set_title(k)
        ax.set_xlabel("記録の順番（エポックなど）")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def finetune_tracknet(cfg: dict[str, Any], dry_run: bool = False) -> int:
    """TrackNetV3 をファインチューニングする。``dry_run`` ならコマンドを表示するだけ。"""
    t = cfg.get("train")
    if not t:
        print("学習設定（train 節）がありません。configs/train.yaml を --config で渡してください。", file=sys.stderr)
        return 2
    cmds, repo, save_dir = build_commands(cfg)
    if dry_run:
        for c in cmds:
            print("cd", repo, "&&", " ".join(c))
        return 0
    if not (repo / "train.py").exists():
        print(f"TrackNetV3 が見つかりません: {repo}", file=sys.stderr)
        return 2
    if t.get("dataset_dir"):
        dst = repo / "data"
        if dst.exists():
            print(f"{dst} が既にあります。dataset_dir の内容を使う場合は手動で置き換えてください。")
        else:
            shutil.copytree(t["dataset_dir"], dst)
    epochs = prepare_resume(Path(cfg["detection"]["tracknet"]["tracknet_ckpt"]), save_dir, int(t.get("epochs", 10)))
    log_path = save_dir / "finetune_log.txt"
    text = ""
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"# {run_metadata()}\n")
        for c in cmds:
            c = [str(epochs) if a == "__EPOCHS__" else a for a in c]
            f.write("$ " + " ".join(c) + "\n")
            f.flush()
            p = subprocess.Popen(c, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 encoding="utf-8", errors="replace")
            for line in p.stdout:
                print(line, end="")
                f.write(line)
                text += line
            if p.wait() != 0:
                print(f"失敗しました: {' '.join(c)}", file=sys.stderr)
                return 1
    plot_log(parse_log(text), save_dir / "finetune_curves.png")
    print(f"完了しました。重み: {save_dir / 'TrackNet_best.pt'}  ログ: {log_path}")
    return 0
