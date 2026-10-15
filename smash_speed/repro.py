"""再現性のための補助：乱数シードの固定と実行メタデータの記録。"""

from __future__ import annotations

import datetime as dt
import platform
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from smash_speed import __version__
from smash_speed.config import PROJECT_ROOT


def set_seed(seed: int) -> np.random.Generator:
    """Python・NumPy（と、導入済みなら PyTorch）の乱数シードを固定する。

    Returns:
        以降の乱数生成に使う NumPy の Generator。
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
    return np.random.default_rng(seed)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def run_metadata(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """出力ファイルに埋め込む実行メタデータ（バージョン・コミット・時刻など）を返す。"""
    meta: dict[str, Any] = {
        "smash_speed_version": __version__,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    if extra:
        meta.update(extra)
    return meta


def output_dir(base: str | Path, name: str) -> Path:
    """出力先ディレクトリを作成して返す。"""
    path = Path(base) / name
    path.mkdir(parents=True, exist_ok=True)
    return path
