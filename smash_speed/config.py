"""設定ファイル（YAML）の読み込み・上書き・保存。"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """``override`` の値で ``base`` を再帰的に上書きした新しい辞書を返す。"""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """既定設定を読み込み、``path`` が与えられればその内容で上書きして返す。

    Args:
        path: ユーザー設定 YAML のパス。差分のみ書けばよい。

    Returns:
        設定の辞書。
    """
    with DEFAULT_CONFIG_PATH.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if path is not None:
        with Path(path).open(encoding="utf-8") as f:
            override = yaml.safe_load(f) or {}
        config = _deep_merge(config, override)
    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """使用した設定を YAML で保存する（再現性のためのスナップショット）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
