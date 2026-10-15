"""``calibration.json`` の保存と読み込み。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from smash_speed.calibration.solve import CalibrationResult
from smash_speed.camera import Camera
from smash_speed.repro import run_metadata


def save_calibration(result: CalibrationResult, path: str | Path, video: str | None = None,
                     frame_time_s: float | None = None, extra: dict[str, Any] | None = None) -> None:
    """推定結果を JSON に保存する。カメラを固定した撮影なら次の動画でも再利用できる。"""
    data = {
        "camera": result.camera.to_dict(),
        "method": result.method,
        "rmse_px": result.rmse_px,
        "points": [
            {"name": n, "world_m": w.tolist(), "image_px": i.tolist(), "error_px": float(e)}
            for n, w, i, e in zip(result.names, result.world, result.image, result.errors_px)
        ],
        "warnings": result.warnings,
        "video": video,
        "frame_time_s": frame_time_s,
        "meta": run_metadata(extra),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def load_calibration(path: str | Path) -> tuple[Camera, dict[str, Any]]:
    """JSON からカメラと保存内容全体を読み込む。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Camera.from_dict(data["camera"]), data


def load_points(path: str | Path) -> dict[str, tuple[float, float]]:
    """ランドマークの画像座標を JSON（``{"名前": [u, v], ...}``）から読み込む（GUI を使わない場合）。

    ``calibration.json`` を渡した場合は、その ``points`` を読む。
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "points" in data and isinstance(data["points"], list):
        return {p["name"]: tuple(p["image_px"]) for p in data["points"]}
    return {k: (float(v[0]), float(v[1])) for k, v in data.items() if v is not None}


def landmark_table(cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    """設定ファイルのランドマーク名 → 実寸座標 [m]。"""
    return {lm["name"]: np.array(lm["xyz"], float) for lm in cfg["landmarks"]}
