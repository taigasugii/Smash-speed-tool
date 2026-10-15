"""TrackNetV3（https://github.com/qaz812345/TrackNetV3, MIT License）によるシャトル検出のラッパー。

TrackNetV3 のコードは本パッケージに含めず、利用者が clone したディレクトリの ``predict.py`` を
サブプロセスで実行し、出力（``<動画名>_ball.csv``: Frame, Visibility, X, Y）を本ツールの形式に変換する。

準備（README「TrackNetV3 の導入」参照）::

    git clone https://github.com/qaz812345/TrackNetV3
    # README の Google Drive から学習済み重みを取得し、TrackNetV3/ckpts/ に置く
    # configs の detection.tracknet.repo_dir / tracknet_ckpt / inpaintnet_ckpt を設定
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from smash_speed.video_io import VideoMeta, cut_clip, probe


def tracknet_command(cfg: dict[str, Any], video: Path, save_dir: Path, large_video: bool = False) -> list[str]:
    """``predict.py`` を実行するコマンドを組み立てる。"""
    t = cfg["detection"]["tracknet"]
    if not t.get("repo_dir") or not t.get("tracknet_ckpt"):
        raise ValueError("設定 detection.tracknet.repo_dir と tracknet_ckpt を指定してください")
    repo = Path(t["repo_dir"])
    if not (repo / "predict.py").exists():
        raise FileNotFoundError(f"TrackNetV3 の predict.py が見つかりません: {repo}")
    py = t.get("python") or sys.executable
    cmd = [py, "predict.py", "--video_file", str(Path(video).resolve()),
           "--tracknet_file", str(Path(t["tracknet_ckpt"]).resolve()), "--save_dir", str(Path(save_dir).resolve())]
    if t.get("inpaintnet_ckpt"):
        cmd += ["--inpaintnet_file", str(Path(t["inpaintnet_ckpt"]).resolve())]
    if large_video:
        cmd += ["--large_video"]
    return cmd


def convert_tracknet_csv(csv: Path, clip_meta: VideoMeta, frame_offset: int = 0, time_offset: float = 0.0
                         ) -> pd.DataFrame:
    """TrackNetV3 の出力 CSV を、本ツールの検出の形式（frame, timestamp, x_px, y_px, visible, confidence）にする。"""
    tn = pd.read_csv(csv)
    vis = tn["Visibility"].astype(int).to_numpy()
    n = min(len(tn), len(clip_meta.timestamps))
    return pd.DataFrame({
        "frame": tn["Frame"].to_numpy()[:n] + frame_offset,
        "timestamp": clip_meta.timestamps[:n] + time_offset,
        "x_px": np.where(vis[:n] == 1, tn["X"].to_numpy()[:n], np.nan).astype(float),
        "y_px": np.where(vis[:n] == 1, tn["Y"].to_numpy()[:n], np.nan).astype(float),
        "visible": vis[:n], "confidence": vis[:n].astype(float), "track_id": np.where(vis[:n] == 1, 0, -1),
    })


def run_tracknet(meta: VideoMeta, start_s: float | None, end_s: float | None, cfg: dict[str, Any],
                 work_dir: Path) -> pd.DataFrame:
    """区間を切り出して TrackNetV3 で検出し、元動画のフレーム番号・実時間に戻して返す。"""
    work_dir.mkdir(parents=True, exist_ok=True)
    if start_s is not None or end_s is not None:
        t0 = start_s if start_s is not None else 0.0
        t1 = end_s if end_s is not None else float(meta.timestamps[-1])
        clip = cut_clip(meta.path, work_dir / "clip.mp4", t0 / meta.time_scale, t1 / meta.time_scale, cfg=cfg)
        frame_offset = int(np.searchsorted(meta.timestamps, t0 - 1e-9))
        time_offset = float(meta.timestamps[frame_offset])
    else:
        clip, frame_offset, time_offset = Path(meta.path), 0, 0.0
    cmd = tracknet_command(cfg, clip, work_dir)
    log = work_dir / "tracknet_log.txt"
    with log.open("w", encoding="utf-8") as f:
        r = subprocess.run(cmd, cwd=cfg["detection"]["tracknet"]["repo_dir"], stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"TrackNetV3 の実行に失敗しました（ログ: {log}）")
    out_csv = work_dir / f"{Path(clip).stem}_ball.csv"
    clip_meta = probe(clip, meta.capture_fps if meta.time_scale != 1.0 else None, cfg)
    return convert_tracknet_csv(out_csv, clip_meta, frame_offset, time_offset)
