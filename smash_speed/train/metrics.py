"""シャトル検出の評価指標（適合率・再現率・F1）。

フレームごとに次のように数える（TrackNet の論文と同じ考え方）。

- TP: 正解も予測も「見える」で、距離が ``tolerance_px`` 以内
- FP: 予測が「見える」だが、正解が「見えない」、または距離が ``tolerance_px`` を超える
- FN: 正解が「見える」のに、予測が「見えない」
- TN: 正解も予測も「見えない」
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from smash_speed.annotate import normalize_labels


def detection_metrics(pred: pd.DataFrame, gt: pd.DataFrame, tolerance_px: float = 4.0) -> dict[str, Any]:
    """予測と正解（フレーム番号で対応）から指標を計算する。座標は同じ解像度の px であること。"""
    p = normalize_labels(pred)[["frame", "x_px", "y_px", "visible"]]
    g = normalize_labels(gt)
    g = g[g.visible >= 0][["frame", "x_px", "y_px", "visible"]]  # 未ラベルのフレームは除く
    m = g.merge(p, on="frame", how="left", suffixes=("_gt", "_pr"))
    m["visible_pr"] = m["visible_pr"].fillna(0)
    gv, pv = m.visible_gt == 1, m.visible_pr == 1
    dist = np.hypot(m.x_px_pr - m.x_px_gt, m.y_px_pr - m.y_px_gt)
    tp = int((gv & pv & (dist <= tolerance_px)).sum())
    fp_far = int((gv & pv & (dist > tolerance_px)).sum())
    fp_inv = int((~gv & pv).sum())
    fn = int((gv & ~pv).sum())
    tn = int((~gv & ~pv).sum())
    fp = fp_far + fp_inv
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * prec * rec / (prec + rec) if prec and rec and np.isfinite(prec + rec) else np.nan
    both = gv & pv
    return {"tp": tp, "fp": fp, "fp_far": fp_far, "fp_invisible": fp_inv, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1,
            "accuracy": (tp + tn) / len(m) if len(m) else np.nan,
            "median_error_px": float(np.median(dist[both])) if both.any() else np.nan,
            "tolerance_px": tolerance_px, "n_frames": int(len(m))}
