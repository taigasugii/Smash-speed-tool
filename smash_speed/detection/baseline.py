"""深層学習を使わないシャトル検出（ベースライン）。

1. 3フレーム差分（前後のフレームとの差の小さい方）で動いている画素を取り出す
2. 連結成分のうち、面積が小さいもの（シャトル・その残像）を候補にする
3. 候補をフレーム間でつなぎ、等加速度の予測からのずれが小さい軌道を作る
4. 最もシャトルらしい軌道（長く・速く・滑らか）を選び、局所的な2次式から外れた点を除く

TrackNet との比較対象、およびアノテーションの初期値として使う。
4K 動画は ``process_width`` まで縮小して処理し、座標は元の解像度に戻して出力する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import pandas as pd

from smash_speed.video_io import VideoMeta, VideoReader

DETECTION_COLUMNS = ["frame", "timestamp", "x_px", "y_px", "visible", "confidence"]


@dataclass
class BaselineParams:
    """ベースライン検出の設定（距離は縮小後の画像の px）。"""

    process_width: int = 1280
    diff_threshold: int = 15          # 差分の閾値 [輝度]
    min_area_px: int = 2
    max_area_px: int = 600
    max_streak_area_px: int = 4000    # 細長い残像なら面積がここまでの候補を許す
    max_streak_width_px: float = 8.0  # 残像とみなす平均の太さ（面積 / 長さ）の上限 [px]
    max_candidates: int = 30          # 1フレームあたりの候補の上限（面積の小さい順ではなく差分の強い順）
    max_speed_px_s: float = 8000.0    # 画像上の最大速さ [px/s]（縮小後）。対応付けの探索半径に使う
    gate_px: float = 12.0             # 予測位置からの許容ずれ [px]（縮小後）
    gate_step_ratio: float = 0.8      # 直前の移動量に対する許容ずれの割合（カメラに近づくシャトルは画面上で加速して見える）
    max_gap_frames: int = 2           # 軌道の途切れを許すフレーム数
    min_track_len: int = 4
    outlier_px: float = 4.0           # 局所2次式からのずれがこれを超える点を外れ値とする [px]（縮小後）
    roi: list[float] | None = None    # [x0, y0, x1, y1]（元解像度 px）。None なら全体
    min_brightness: int = 0           # 候補の最低輝度（白いシャトルに限る場合に使う）
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "BaselineParams":
        d = dict(cfg.get("detection", {}).get("baseline", {}))
        known = {k: d.pop(k) for k in list(d) if k in cls.__dataclass_fields__}
        return cls(**known, extra=d)


def candidates_from_frames(prev: np.ndarray, cur: np.ndarray, nxt: np.ndarray, p: BaselineParams,
                           roi_mask: np.ndarray | None = None) -> list[dict[str, float]]:
    """3枚のグレースケール画像から、現在フレームの小さな動体の候補を返す。

    Returns:
        候補のリスト（``x, y`` 重心 [px]、``area`` [px^2]、``strength`` 差分の総和、
        ``length`` 長軸の長さ [px]、``angle`` 長軸の向き [deg]、``brightness``）
    """
    d1 = cv2.absdiff(cur, prev)
    d2 = cv2.absdiff(nxt, cur)
    d = cv2.min(d1, d2)
    if roi_mask is not None:
        d = cv2.bitwise_and(d, roi_mask)
    _, mask = cv2.threshold(d, p.diff_threshold, 255, cv2.THRESH_BINARY)
    # 途切れた残像（ブレ）の断片を1つにまとめるため、膨張させた画像で連結成分を求める
    k = max(1, int(p.extra.get("group_px", 5)))
    grouped = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
    labels[mask == 0] = 0
    out = []
    for i in range(1, n):
        x, y, w, h = stats[i, :4]
        sub = labels[y:y + h, x:x + w] == i
        area = int(sub.sum())
        if area < p.min_area_px or area > p.max_streak_area_px:
            continue
        wts = d[y:y + h, x:x + w].astype(np.float64) * sub
        s = wts.sum()
        if s <= 0:
            continue
        ys, xs = np.mgrid[y:y + h, x:x + w]
        cx, cy = (wts * xs).sum() / s, (wts * ys).sum() / s
        mu20 = (wts * (xs - cx) ** 2).sum() / s
        mu02 = (wts * (ys - cy) ** 2).sum() / s
        mu11 = (wts * (xs - cx) * (ys - cy)).sum() / s
        common = np.sqrt(((mu20 - mu02) / 2) ** 2 + mu11**2)
        major = 4 * np.sqrt(max((mu20 + mu02) / 2 + common, 0))
        angle = 0.5 * np.degrees(np.arctan2(2 * mu11, mu20 - mu02))
        # 面積の大きい候補は、細長い残像（低 fps・長い露光のシャトル）だけを残す
        if area > p.max_area_px and area / max(major, 1.0) > p.max_streak_width_px:
            continue
        bright = float(cur[y:y + h, x:x + w][sub].mean())
        if bright < p.min_brightness:
            continue
        out.append({"x": cx, "y": cy, "area": float(area), "strength": float(s),
                    "length": float(major), "angle": float(angle), "brightness": bright})
    out.sort(key=lambda c: -c["strength"])
    return out[: p.max_candidates]


@dataclass
class _Track:
    ids: list[int]          # 候補の通し番号
    frames: list[int]
    t: list[float]
    xy: list[np.ndarray]
    misses: int = 0

    def predict(self, t: float) -> np.ndarray:
        """等加速度（3点以上）または等速度で位置を予測する。"""
        if len(self.t) == 1:
            return self.xy[-1]
        tt = np.array(self.t[-4:])
        xy = np.array(self.xy[-4:])
        deg = 2 if len(tt) >= 3 else 1
        c = [np.polyfit(tt - tt[-1], xy[:, j], deg) for j in range(2)]
        return np.array([np.polyval(cj, t - tt[-1]) for cj in c])


def link_tracks(cands: pd.DataFrame, p: BaselineParams) -> list[_Track]:
    """候補をフレーム間でつないで軌道を作る（貪欲法）。"""
    tracks: list[_Track] = []
    active: list[_Track] = []
    for frame, grp in cands.groupby("frame", sort=True):
        t = float(grp["timestamp"].iloc[0])
        pts = grp[["x", "y"]].to_numpy()
        ids = grp["cid"].to_numpy()
        used = np.zeros(len(pts), bool)
        # 長い軌道から優先して対応付ける
        for tr in sorted(active, key=lambda a: -len(a.t)):
            dt = t - tr.t[-1]
            pred = tr.predict(t)
            if len(tr.t) >= 2:
                # 低 fps では1フレームで大きく減速するので、直前の移動量に比例して探索半径を広げる
                step = np.linalg.norm(tr.xy[-1] - tr.xy[-2])
                radius = (p.gate_px + p.gate_step_ratio * step) * (1 + tr.misses)
            else:
                radius = p.max_speed_px_s * dt + p.gate_px
            dist = np.linalg.norm(pts - pred, axis=1)
            dist[used] = np.inf
            if len(dist) and dist.min() <= radius:
                j = int(np.argmin(dist))
                used[j] = True
                tr.ids.append(int(ids[j]))
                tr.frames.append(int(frame))
                tr.t.append(t)
                tr.xy.append(pts[j])
                tr.misses = 0
            else:
                tr.misses += 1
        active = [a for a in active if a.misses <= p.max_gap_frames]
        for j in np.flatnonzero(~used):
            tr = _Track([int(ids[j])], [int(frame)], [t], [pts[j]])
            tracks.append(tr)
            active.append(tr)
    return [tr for tr in tracks if len(tr.t) >= 2]


def merge_tracks(tracks: list[_Track], p: BaselineParams) -> list[_Track]:
    """途切れた軌道をつなぐ：A の終わりから外挿した位置の近くで B が始まっていれば1本にする。"""
    tracks = sorted(tracks, key=lambda tr: tr.t[0])
    merged = True
    while merged:
        merged = False
        for a in tracks:
            for b in tracks:
                if a is b or b.frames[0] <= a.frames[-1] or b.frames[0] - a.frames[-1] > p.max_gap_frames + 1:
                    continue
                step = np.linalg.norm(a.xy[-1] - a.xy[-2]) if len(a.xy) >= 2 else p.gate_px
                gap = b.frames[0] - a.frames[-1]
                tol = (p.gate_px + p.gate_step_ratio * step) * gap
                if np.linalg.norm(a.predict(b.t[0]) - b.xy[0]) <= tol:
                    a.ids += b.ids
                    a.frames += b.frames
                    a.t += b.t
                    a.xy += b.xy
                    tracks.remove(b)
                    merged = True
                    break
            if merged:
                break
    return tracks


def track_score(tr: _Track) -> float:
    """シャトルらしさのスコア：長さ^1.5 × 画像上の速さの中央値 ÷ (1 + 滑らかさの残差)。

    3 点以下では 2 次式が必ずぴったり合ってしまうので、直線からのずれで滑らかさを測る。
    """
    t, xy = np.array(tr.t), np.array(tr.xy)
    if len(t) < 3:
        return 0.0
    speed = np.median(np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.diff(t))
    deg = 2 if len(t) >= 5 else 1
    c = [np.polyfit(t - t[0], xy[:, j], deg) for j in range(2)]
    fit = np.column_stack([np.polyval(cj, t - t[0]) for cj in c])
    resid = np.sqrt(np.mean(np.sum((fit - xy) ** 2, axis=1)))
    return float(len(t) ** 1.5 * speed / (1.0 + resid))


def remove_outliers(t: np.ndarray, xy: np.ndarray, thr: float, n_neighbors: int = 4,
                    step_ratio: float = 0.25) -> np.ndarray:
    """近くの点に当てはめた 2 次式から大きく外れた点を、最も外れたものから 1 つずつ除く。

    許容するずれは ``thr`` [px] と「近くの点の 1 コマあたりの移動量 × step_ratio」の大きい方
    （速いシャトルの残像は重心が数 px〜十数 px ぶれるため）。端の点も、近い 4 点からの外挿で調べる。
    """
    n = len(t)
    keep = np.ones(n, bool)
    for _ in range(n):
        idx = np.flatnonzero(keep)
        if len(idx) < n_neighbors + 1:
            break
        worst, worst_r = -1, 1.0
        for pos, i in enumerate(idx):
            others = np.delete(idx, pos)
            nb = others[np.argsort(np.abs(others - i), kind="stable")[:n_neighbors]]
            nb.sort()
            tt = t[nb] - t[i]
            pred = np.array([np.polyval(np.polyfit(tt, xy[nb, j], 2), 0.0) for j in range(2)])
            step = np.median(np.linalg.norm(np.diff(xy[nb], axis=0), axis=1))
            r = np.linalg.norm(xy[i] - pred) / max(thr, step_ratio * step)
            if r > worst_r:
                worst, worst_r = i, r
        if worst < 0:
            break
        keep[worst] = False
    return keep


def detect(meta: VideoMeta, start_s: float | None = None, end_s: float | None = None,
           params: BaselineParams | None = None, n_tracks: int = 1,
           return_candidates: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """動画の区間 [start_s, end_s] [s] でシャトルを検出する。

    Args:
        meta: 動画のメタデータ
        start_s, end_s: 区間（実時間）[s]
        params: 設定
        n_tracks: 出力する軌道の数（スコアの高い順）
        return_candidates: 候補の一覧も返す

    Returns:
        フレームごとの検出（列 ``frame, timestamp, x_px, y_px, visible, confidence`` と
        ``track_id, length_px, angle_deg, area_px``。座標は元の解像度）
    """
    p = params or BaselineParams()
    scale = min(1.0, p.process_width / meta.width)
    roi_mask = None
    rows: list[dict[str, float]] = []
    frames_meta: list[tuple[int, float]] = []
    with VideoReader(meta, scale=scale) as reader:
        buf: list[tuple[int, float, np.ndarray]] = []
        for i, t, img in reader.iter_range(start_s, end_s):
            g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0)
            if roi_mask is None and p.roi is not None:
                roi_mask = np.zeros_like(g)
                x0, y0, x1, y1 = (np.array(p.roi) * scale).round().astype(int)
                roi_mask[y0:y1, x0:x1] = 255
            buf.append((i, t, g))
            frames_meta.append((i, t))
            if len(buf) < 3:
                continue
            (_, _, a), (fi, ft, b), (_, _, c) = buf[-3:]
            for cand in candidates_from_frames(a, b, c, p, roi_mask):
                rows.append({"frame": fi, "timestamp": ft, **cand})
            buf = buf[-2:]
    cands = pd.DataFrame(rows)
    out = pd.DataFrame({"frame": [f for f, _ in frames_meta], "timestamp": [t for _, t in frames_meta]})
    for col, val in (("x_px", np.nan), ("y_px", np.nan), ("visible", 0), ("confidence", 0.0),
                     ("track_id", -1), ("length_px", np.nan), ("angle_deg", np.nan), ("area_px", np.nan)):
        out[col] = val
    if cands.empty:
        return (out, cands) if return_candidates else out
    cands["cid"] = np.arange(len(cands))
    tracks = [tr for tr in merge_tracks(link_tracks(cands, p), p) if len(tr.t) >= p.min_track_len]
    tracks = sorted(tracks, key=track_score, reverse=True)[:n_tracks]
    idx_of = {f: k for k, f in enumerate(out["frame"])}
    for tid, tr in enumerate(tracks):
        t, xy = np.array(tr.t), np.array(tr.xy)
        keep = remove_outliers(t, xy, p.outlier_px)
        score = track_score(tr)
        for k, cid in enumerate(tr.ids):
            if not keep[k]:
                continue
            c = cands.iloc[cid]
            r = idx_of[tr.frames[k]]
            if out.at[r, "visible"]:
                continue
            out.loc[r, ["x_px", "y_px", "visible", "confidence", "track_id", "length_px", "angle_deg", "area_px"]] = [
                c.x / scale, c.y / scale, 1, min(1.0, score / (score + 1000.0)), tid,
                c.length / scale, c.angle, c.area / scale**2]
    out["visible"] = out["visible"].astype(int)
    out["track_id"] = out["track_id"].astype(int)
    if return_candidates:
        cands[["x", "y", "length"]] /= scale
        return out, cands
    return out
