"""高 fps 動画から疑似低 fps 動画とラベルを作る。

- 単純な間引き：``step = 元fps / 目標fps`` フレームごとに1枚（瞬間露光に近い）
- 連続フレームの平均（``blur``）：``n_avg`` 枚を平均して露光時間 ``n_avg / 元fps`` のブレを再現する。
  シャッター速度の違いを再現するため、``n_avg`` を変えたバリエーションも作れる

出力フレーム k の時刻は、平均した元フレームの時刻の平均（露光の中央）。出力動画の時刻 ``k / 目標fps`` との
差は一定値 ``time_offset_s`` で、音声もこの分だけずらして入れる（打球音による打点推定に使えるように）。
正解位置は、平均したフレームのうち見えているものの位置の平均（ブレの重心）とする。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from smash_speed.annotate import normalize_labels
from smash_speed.video_io import VideoMeta, VideoReader, find_tool


def lowfps_plan(n_frames: int, src_fps: float, target_fps: float, n_avg: int | None = None,
                blur: bool = True, offset: int = 0) -> tuple[list[np.ndarray], int, int]:
    """出力フレームごとに使う元フレームの番号を決める。

    Returns:
        (各出力フレームの元フレーム番号の配列のリスト, step, n_avg)
    """
    ratio = src_fps / target_fps
    step = int(round(ratio))
    if step < 1 or abs(ratio - step) > 0.02 * step:
        raise ValueError(f"元 fps {src_fps:.2f} は目標 fps {target_fps:.2f} の整数倍ではありません")
    if n_avg is None:
        n_avg = step if blur else 1
    n_avg = int(np.clip(n_avg, 1, step))
    groups = [np.arange(s, s + n_avg) for s in range(offset, n_frames - n_avg + 1, step)]
    return groups, step, n_avg


def make_lowfps(meta: VideoMeta, labels_csv: Path | None, target_fps: float, out_dir: Path,
                blur: bool = True, n_avg: int | None = None, offset: int = 0,
                start_s: float | None = None, end_s: float | None = None,
                with_audio: bool = True, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """疑似低 fps 動画（と、ラベルがあれば正解）を書き出す。

    Returns:
        ``video``・``labels``（なければ None）・``time_offset_s``・``n_avg``・``step``
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_all = meta.timestamps
    i0 = 0 if start_s is None else int(np.searchsorted(ts_all, start_s))
    i1 = len(ts_all) - 1 if end_s is None else int(np.searchsorted(ts_all, end_s, side="right")) - 1
    n = i1 - i0 + 1
    groups, step, n_avg = lowfps_plan(n, meta.capture_fps, target_fps, n_avg, blur, offset)
    tag = f"{Path(meta.path).stem}_{target_fps:g}fps_avg{n_avg}" + (f"_off{offset}" if offset else "")
    silent = out_dir / f"{tag}_noaudio.mp4"
    video = out_dir / f"{tag}.mp4"
    writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (meta.width, meta.height))
    acc = None
    k_group, members = 0, set(groups[0].tolist()) if groups else set()
    with VideoReader(meta) as r:
        for j, (_, _, img) in enumerate(r.iter_range(ts_all[i0], ts_all[i1])):
            if k_group >= len(groups):
                break
            if j in members:
                acc = img.astype(np.float32) if acc is None else acc + img
                if j == groups[k_group][-1]:
                    writer.write(np.clip(acc / n_avg + 0.5, 0, 255).astype(np.uint8))
                    acc = None
                    k_group += 1
                    if k_group < len(groups):
                        members = set(groups[k_group].tolist())
    writer.release()
    t_src = np.array([ts_all[i0 + g].mean() for g in groups])
    time_offset = float(t_src[0] - ts_all[i0]) if len(t_src) else 0.0  # 出力の時刻 0 ↔ 元の時刻

    # 音声を time_offset だけずらして入れる
    if with_audio and meta.has_audio:
        ffmpeg = find_tool("ffmpeg", cfg)
        a0 = (ts_all[i0] + time_offset) / meta.time_scale
        subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(silent), "-ss", f"{a0:.6f}", "-i", meta.path,
                        "-map", "0:v", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-shortest", str(video)],
                       check=True)
        silent.unlink()
    else:
        silent.replace(video)

    labels_out = None
    if labels_csv is not None:
        lab = normalize_labels(pd.read_csv(labels_csv)).set_index("frame")
        rows = []
        for k, g in enumerate(groups):
            src = lab.reindex(i0 + g)
            vis = (src.visible == 1).to_numpy()
            frac = vis.mean()
            ev = ""
            evs = [e for e in src.event.fillna("") if e]
            if evs:
                ev = evs[0]
            if frac >= 0.5:
                x, y, v = float(src.x_px[vis].mean()), float(src.y_px[vis].mean()), 1
            else:
                x, y, v = np.nan, np.nan, 0
            rows.append({"frame": k, "timestamp": k / target_fps, "x_px": x, "y_px": y, "visible": v,
                         "event": ev, "source_frames": f"{i0 + g[0]}-{i0 + g[-1]}",
                         "t_original_s": t_src[k]})
        labels_out = out_dir / f"{tag}_labels.csv"
        pd.DataFrame(rows).to_csv(labels_out, index=False)
    info = {"video": str(video), "labels": str(labels_out) if labels_out else None,
            "time_offset_s": time_offset + float(ts_all[i0]), "n_avg": n_avg, "step": step,
            "target_fps": target_fps, "source_fps": meta.capture_fps, "offset_frames": offset,
            "exposure_s": n_avg / meta.capture_fps}
    (out_dir / f"{tag}.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info
