"""学習データの作成：撮影単位での分割、TrackNetV3 形式での書き出し、データ拡張。

**フレーム単位でランダムに分割しない**（同じラリーのフレームが学習とテストに混ざり、精度を過大評価するため）。
一覧 CSV の ``group`` 列（なければ ``session`` → ``date`` → ``player`` の順に探す）の単位で、
学習 / 検証 / テストに分ける。``split`` 列があればそれを優先する。

一覧 CSV（``index``）の列::

    video, labels, [session, date, player, group, split, capture_fps]

書き出し先の構成（TrackNetV3 の README に合わせる）::

    out/train/matchN/video/<rally>.mp4
    out/train/matchN/csv/<rally>_ball.csv     Frame, Visibility, X, Y
    out/test/matchN/...
    out/test/matchN/corrected_csv/<rally>_ball.csv   （TrackNetV3 はテストのラベルをここから読む）
    out/val/matchN/...                         （TrackNetV3 の preprocess.py は val を作るので、既定では作らない）
    out/split.json                             どのグループをどこに入れたか
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from smash_speed.annotate import normalize_labels
from smash_speed.video_io import VideoReader, probe


def assign_splits(index: pd.DataFrame, ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
                  seed: int = 0) -> pd.Series:
    """グループ単位で train / val / test を割り当てる（同じグループは必ず同じ分割に入る）。"""
    if "split" in index.columns and index["split"].notna().all():
        return index["split"]
    key = next((c for c in ("group", "session", "date", "player") if c in index.columns), None)
    groups = index[key].astype(str) if key else pd.Series(np.arange(len(index)).astype(str), index=index.index)
    uniq = np.array(sorted(groups.unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_test = max(1, int(round(ratios[2] * n))) if n >= 3 else (1 if n >= 2 else 0)
    n_val = max(1, int(round(ratios[1] * n))) if n >= 3 else 0
    assign = {}
    for k, gname in enumerate(uniq):
        assign[gname] = "test" if k < n_test else "val" if k < n_test + n_val else "train"
    return groups.map(assign)


def augment_clip(frames: list[np.ndarray], xy: np.ndarray, rng: np.random.Generator,
                 p: dict[str, Any]) -> tuple[list[np.ndarray], np.ndarray]:
    """1本のクリップに同じ変換をかける：明るさ・コントラスト・ガウスノイズ・拡大縮小・平行移動・ブラー。

    Args:
        frames: BGR 画像のリスト
        xy: 各フレームのシャトル位置 (N, 2) [px]（見えないフレームは NaN）
        p: 変換の強さ（``brightness``, ``contrast``, ``noise_sd``, ``scale``, ``shift_px``, ``blur_max``）
    """
    h, w = frames[0].shape[:2]
    alpha = 1 + rng.uniform(-p.get("contrast", 0.2), p.get("contrast", 0.2))
    beta = rng.uniform(-p.get("brightness", 25), p.get("brightness", 25))
    s = 1 + rng.uniform(-p.get("scale", 0.1), p.get("scale", 0.1))
    tx, ty = rng.uniform(-p.get("shift_px", 40), p.get("shift_px", 40), 2)
    M = np.array([[s, 0, (1 - s) * w / 2 + tx], [0, s, (1 - s) * h / 2 + ty]], np.float32)
    blur_k = int(rng.integers(0, p.get("blur_max", 2) + 1)) * 2 + 1
    noise = p.get("noise_sd", 4.0)
    out = []
    for f in frames:
        g = cv2.warpAffine(f, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        g = cv2.convertScaleAbs(g, alpha=alpha, beta=beta)
        if blur_k > 1:
            g = cv2.GaussianBlur(g, (blur_k, blur_k), 0)
        if noise > 0:
            g = np.clip(g + rng.normal(0, noise, g.shape), 0, 255).astype(np.uint8)
        out.append(g)
    xy2 = xy * s + np.array([(1 - s) * w / 2 + tx, (1 - s) * h / 2 + ty])
    outside = (xy2[:, 0] < 0) | (xy2[:, 0] >= w) | (xy2[:, 1] < 0) | (xy2[:, 1] >= h)
    xy2[outside] = np.nan
    return out, xy2


def _write_rally(video: Path, labels: pd.DataFrame, out_video: Path, out_csv: Path, width: int,
                 rng: np.random.Generator | None = None, aug: dict[str, Any] | None = None) -> int:
    meta = probe(video)
    lab = normalize_labels(labels).sort_values("frame")
    lab = lab[lab.visible >= 0]
    f0, f1 = int(lab.frame.min()), int(lab.frame.max())
    s = width / meta.width
    size = (width, int(round(meta.height * s / 2)) * 2)
    frames = []
    with VideoReader(meta) as r:
        for i, _, img in r.iter_range(meta.timestamps[f0], meta.timestamps[f1]):
            frames.append(cv2.resize(img, size, interpolation=cv2.INTER_AREA))
    lab = lab.set_index("frame").reindex(range(f0, f0 + len(frames)))
    xy = np.column_stack([lab.x_px, lab.y_px]) * s
    xy[(lab.visible != 1).to_numpy()] = np.nan
    if aug is not None and rng is not None:
        frames, xy = augment_clip(frames, xy, rng, aug)
    out_video.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    wr = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), meta.capture_fps, size)
    for f in frames:
        wr.write(f)
    wr.release()
    vis = np.isfinite(xy).all(axis=1)
    pd.DataFrame({"Frame": np.arange(len(frames)), "Visibility": vis.astype(int),
                  "X": np.where(vis, xy[:, 0], 0.0).round(1), "Y": np.where(vis, xy[:, 1], 0.0).round(1)}
                 ).to_csv(out_csv, index=False)
    return len(frames)


def export_tracknet_dataset(index_csv: Path, out: Path, cfg: dict[str, Any], width: int = 1280,
                            make_val: bool = False) -> int:
    """一覧 CSV のラベル付き動画を、TrackNetV3 の学習データの形式で書き出す。

    設定 ``train.augment_copies`` の数だけ、学習用にデータ拡張したコピーも加える。
    Returns:
        書き出したラリー（動画）の数
    """
    index = pd.read_csv(index_csv)
    base = index_csv.parent
    tcfg = cfg.get("train", {})
    split = assign_splits(index, tuple(tcfg.get("split_ratios", (0.7, 0.15, 0.15))), cfg.get("seed", 0))
    if not make_val:
        split = split.replace({"val": "train"})
    rng = np.random.default_rng(cfg.get("seed", 0))
    n_copies = int(tcfg.get("augment_copies", 0))
    aug = tcfg.get("augment", {})
    match_of: dict[str, int] = {}
    count = 0
    record = []
    for (row, sp) in zip(index.itertuples(), split):
        key = next((str(getattr(row, c)) for c in ("group", "session", "date", "player") if hasattr(row, c)),
                   Path(row.video).stem)
        m = match_of.setdefault(f"{sp}/{key}", len([k for k in match_of if k.startswith(sp + "/")]) + 1)
        rally = Path(row.video).stem
        labels = pd.read_csv(base / row.labels)
        d = out / sp / f"match{m}"
        _write_rally(base / row.video, labels, d / "video" / f"{rally}.mp4", d / "csv" / f"{rally}_ball.csv", width)
        if sp == "test":  # TrackNetV3 の dataset.py はテストのラベルを corrected_csv/ から読む
            (d / "corrected_csv").mkdir(parents=True, exist_ok=True)
            shutil.copy(d / "csv" / f"{rally}_ball.csv", d / "corrected_csv" / f"{rally}_ball.csv")
        count += 1
        record.append({"video": row.video, "split": sp, "match": m, "group": key})
        if sp == "train":
            for c in range(n_copies):
                _write_rally(base / row.video, labels, d / "video" / f"{rally}_aug{c}.mp4",
                             d / "csv" / f"{rally}_aug{c}_ball.csv", width, rng, aug)
                count += 1
    (out / "split.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return count
