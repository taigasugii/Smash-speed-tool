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

# 検出の方法を変えたら上げる（保存済みの検出結果を使い回さないため）
DETECTOR_VERSION = 2

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
    strength: float = 0.0   # 直前の候補の差分の強さ（似た強さの候補を優先してつなぐ）
    strengths: list[float] = field(default_factory=list)

    def predict(self, t: float) -> np.ndarray:
        """次の位置を予測する。

        3 点以上では「直前の移動 × 直前の移動量の比（減速・加速の割合）」で外挿し、向きの変化も引き継ぐ。
        遠ざかるシャトルは画面上で 1 コマごとに半分近くまで減速し、近づくシャトルは加速して見えるので、
        2 次式（等加速度）の外挿では逆向きに戻ったり行き過ぎたりする。
        """
        if len(self.t) == 1:
            return self.xy[-1]
        p1, p2 = self.xy[-2], self.xy[-1]
        dt1 = self.t[-1] - self.t[-2]
        v1 = (p2 - p1) / max(dt1, 1e-9)
        steps = (t - self.t[-1]) / max(dt1, 1e-9)  # 直前の間隔の何倍先か
        if len(self.t) == 2:
            return p2 + v1 * dt1 * steps
        p0 = self.xy[-3]
        v0 = (p1 - p0) / max(self.t[-2] - self.t[-3], 1e-9)
        n0, n1 = np.linalg.norm(v0), np.linalg.norm(v1)
        if n0 < 1e-9 or n1 < 1e-9:
            return p2 + v1 * dt1 * steps
        r = float(np.clip(n1 / n0, 0.3, 1.6))
        ang = np.arctan2(v0[0] * v1[1] - v0[1] * v1[0], v0 @ v1)
        ang = float(np.clip(ang, -np.radians(30), np.radians(30)))
        c, s_ = np.cos(ang), np.sin(ang)
        rot = np.array([[c, -s_], [s_, c]])
        # 等比級数で steps 個先まで足す（1 コマなら v1 * dt1 * r）
        pos, v = p2.copy(), v1 * dt1
        k = 0.0
        while k < steps - 1e-9:
            frac = min(1.0, steps - k)
            v = r * (rot @ v)
            pos = pos + v * frac
            k += 1.0
        return pos


def _recent_resid(tr: _Track, n: int = 5) -> float:
    """直近 n 点に 2 次式（3 点以下なら直線）を当てはめた残差の RMS [px]。"""
    if len(tr.t) < 3:
        return 0.0
    t, xy = np.array(tr.t[-n:]), np.array(tr.xy[-n:])
    deg = 2 if len(t) >= 4 else 1
    c = [np.polyfit(t - t[-1], xy[:, j], deg) for j in range(2)]
    fit = np.column_stack([np.polyval(cj, t - t[-1]) for cj in c])
    return float(np.sqrt(np.mean(np.sum((fit - xy) ** 2, axis=1))))


def link_tracks(cands: pd.DataFrame, p: BaselineParams) -> list[_Track]:
    """候補をフレーム間でつないで軌道を作る。

    1 つの候補を複数の軌道が使ってもよい（関係のない動き同士がたまたまつながった軌道に、
    シャトルの候補を取られないようにするため）。すべての候補から新しい軌道も始める。
    最後の 3 点が同じ軌道は重複とみなし、1 本だけを残す（``_better``）。2 点では、直前に無関係な点が
    付いた軌道と付いていない軌道の両方を残しておく（どちらが正しいかは次のコマで決まる）。
    """
    tracks: list[_Track] = []
    active: list[_Track] = []
    for frame, grp in cands.groupby("frame", sort=True):
        t = float(grp["timestamp"].iloc[0])
        pts = grp[["x", "y"]].to_numpy()
        ids = grp["cid"].to_numpy()
        strength = grp["strength"].to_numpy() if "strength" in grp else np.ones(len(pts))
        for tr in active:
            dt = t - tr.t[-1]
            pred = tr.predict(t)
            if len(tr.t) >= 2:
                # 低 fps では1フレームで大きく減速するので、直前の移動量に比例して探索半径を広げる
                step = np.linalg.norm(tr.xy[-1] - tr.xy[-2])
                radius = (p.gate_px + p.gate_step_ratio * step) * (1 + tr.misses)
            else:
                radius = p.max_speed_px_s * dt + p.gate_px
            dist = np.linalg.norm(pts - pred, axis=1)
            inside = dist <= radius
            if inside.any():
                # 予測位置に近く、直前と同じくらいの強さの候補を選ぶ（予測の近くの小さな雑音より、
                # 少し離れた本物の残像を選ぶため。低 fps で急に減速すると予測は行き過ぎる）
                ratio = np.log(np.maximum(strength, 1e-6) / max(tr.strength, 1e-6)) if tr.strength > 0 else 0.0
                cost = np.where(inside, dist / radius + 0.5 * np.abs(ratio), np.inf)
                j = int(np.argmin(cost))
                tr.strength = float(strength[j])
                tr.strengths.append(tr.strength)
                tr.ids.append(int(ids[j]))
                tr.frames.append(int(frame))
                tr.t.append(t)
                tr.xy.append(pts[j])
                tr.misses = 0
            else:
                tr.misses += 1
        active = [a for a in active if a.misses <= p.max_gap_frames]
        # 重複（最後の 3 点が同じ）の整理
        best: dict[tuple[int, int, int], _Track] = {}
        rest = []
        for tr in active:
            if len(tr.ids) < 3 or tr.misses:
                rest.append(tr)
                continue
            key = (tr.ids[-3], tr.ids[-2], tr.ids[-1])
            cur = best.get(key)
            if cur is None or _better(tr, cur, p):
                best[key] = tr
        active = rest + list(best.values())
        for j in range(len(pts)):
            tr = _Track([int(ids[j])], [int(frame)], [t], [pts[j]], strength=float(strength[j]),
                        strengths=[float(strength[j])])
            tracks.append(tr)
            active.append(tr)
    return _drop_subsumed([tr for tr in tracks if len(tr.t) >= 2])


def _better(a: _Track, b: _Track, p: BaselineParams) -> bool:
    """重複した 2 本の軌道のうち a を残すべきか。

    長い方を残す。ただし長い方の直近の当てはめの残差が大きい（無関係な動きから飛び移ってきた）ときは
    短い方を残す。短い軌道は点が少なく残差が小さく出るので、単純に残差を比べると長い軌道が負ける。
    """
    long_, short = (a, b) if len(a.t) >= len(b.t) else (b, a)
    step = float(np.linalg.norm(long_.xy[-1] - long_.xy[-2]))
    keep_long = _recent_resid(long_) <= p.gate_px + 0.3 * step
    winner = long_ if keep_long else short
    return winner is a


def _drop_subsumed(tracks: list[_Track]) -> list[_Track]:
    """ほかの軌道の一部（同じ候補の並び）にすぎない軌道を除く。"""
    suffixes: set[tuple[int, ...]] = set()
    out = []
    for tr in sorted(tracks, key=lambda a: -len(a.ids)):
        key = tuple(tr.ids)
        if key in suffixes:
            continue
        out.append(tr)
        suffixes.update(key[k:] for k in range(len(key)))
    return out


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
                if len(b.xy) >= 2 and len(a.xy) >= 2:
                    # つなぎ目の前後で速さが大きく変わるもの（ゆっくり動く人に、シャトルがつながるなど）はつながない
                    va = np.linalg.norm(a.xy[-1] - a.xy[-2]) / max(a.t[-1] - a.t[-2], 1e-9)
                    vb = np.linalg.norm(b.xy[1] - b.xy[0]) / max(b.t[1] - b.t[0], 1e-9)
                    if not 0.4 <= (vb + 1e-9) / (va + 1e-9) <= 2.5:
                        continue
                if np.linalg.norm(a.predict(b.t[0]) - b.xy[0]) <= tol:
                    a.ids += b.ids
                    a.frames += b.frames
                    a.t += b.t
                    a.xy += b.xy
                    a.strengths += b.strengths
                    tracks.remove(b)
                    merged = True
                    break
            if merged:
                break
    return tracks


def _local_resid(t: np.ndarray, xy: np.ndarray, w: int = 5) -> float:
    """連続する w 点ごとに 2 次式（5 点未満なら直線）を当てはめた残差 RMS の中央値 [px]。

    軌道全体に 1 本の 2 次式を当てはめると、長い軌道ほど（抵抗で形が 2 次式から外れて）残差が大きくなり、
    同じシャトルの短い一部分の方が高いスコアになってしまうため、局所的に測る。
    """
    if len(t) < w:
        c = [np.polyfit(t - t[0], xy[:, j], 1) for j in range(2)]
        fit = np.column_stack([np.polyval(cj, t - t[0]) for cj in c])
        return float(np.sqrt(np.mean(np.sum((fit - xy) ** 2, axis=1))))
    rs = []
    for k in range(len(t) - w + 1):
        tt, pp = t[k:k + w] - t[k], xy[k:k + w]
        c = [np.polyfit(tt, pp[:, j], 2) for j in range(2)]
        fit = np.column_stack([np.polyval(cj, tt) for cj in c])
        rs.append(np.sqrt(np.mean(np.sum((fit - pp) ** 2, axis=1))))
    return float(np.median(rs))


def split_at_jumps(tracks: list[_Track], min_len: int = 3, max_ratio: float = 3.0) -> list[_Track]:
    """1 コマあたりの移動量が急に max_ratio 倍以上（または 1/max_ratio 以下）になる所や、向きが逆になる所で
    軌道を分ける（別の物に飛び移った部分を、シャトルの部分から切り離すため）。"""
    out: list[_Track] = []
    for tr in tracks:
        t, xy = np.array(tr.t), np.array(tr.xy)
        if len(t) < 3:
            out.append(tr)
            continue
        v = np.diff(xy, axis=0) / np.maximum(np.diff(t), 1e-9)[:, None]
        sp = np.linalg.norm(v, axis=1)
        cut = [0]
        for k in range(1, len(v)):
            ratio = (sp[k] + 1e-6) / (sp[k - 1] + 1e-6)
            cos = float(v[k] @ v[k - 1]) / max(sp[k] * sp[k - 1], 1e-9)
            if ratio > max_ratio or ratio < 1.0 / max_ratio or cos < -0.3:
                cut.append(k)  # 点 k と k+1 の間で切る（点 k は前の区間の終わり）
        cut.append(len(t) - 1)
        if len(cut) == 2:
            out.append(tr)
            continue
        bounds = [0] + [c + 1 for c in cut[1:-1]] + [len(t)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a >= min_len:
                out.append(_Track(tr.ids[a:b], tr.frames[a:b], tr.t[a:b], tr.xy[a:b],
                                  strength=tr.strength, strengths=tr.strengths[a:b]))
    return out


def _dedupe_overlapping(tracks: list[_Track], share: float = 0.5) -> list[_Track]:
    """候補の半分以上を共有する軌道は同じものとみなし、スコアの高い方だけを残す。

    長さで選ぶと、遠くの選手などのゆっくりした動きに、途中からシャトルがつながった軌道が残ってしまう。
    """
    out: list[_Track] = []
    sets: list[set[int]] = []
    for tr in sorted(tracks, key=lambda a: (-track_score(a), a.t[0])):
        ids = set(tr.ids)
        if any(len(ids & o) >= share * min(len(ids), len(o)) for o in sets):
            continue
        out.append(tr)
        sets.append(ids)
    return out


def track_score(tr: _Track) -> float:
    """シャトルらしさのスコア：長さ^1.5 × 画像上の速さの中央値 ÷ (1 + 滑らかさの残差) × 向きのそろい方。

    3 点以下では 2 次式が必ずぴったり合ってしまうので、直線からのずれで滑らかさを測る。
    向きのそろい方は、1 コマごとの移動の向きが前のコマとどれだけ同じか（cos の平均、0〜1）の 2 乗。
    無関係な動き（選手の手足・ほかのコートの人）を飛び移る軌道は向きがばらばらになり、
    差分の強さ（写り方）もコマごとに大きく変わるので、強さの比のばらつきでも減点する。
    """
    t, xy = np.array(tr.t), np.array(tr.xy)
    if len(t) < 3:
        return 0.0
    speed = np.median(np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.diff(t))
    resid = _local_resid(t, xy)
    v = np.diff(xy, axis=0)
    nv = np.linalg.norm(v, axis=1)
    ok = (nv[:-1] > 1e-6) & (nv[1:] > 1e-6)
    cos = np.sum(v[:-1] * v[1:], axis=1)[ok] / (nv[:-1] * nv[1:])[ok]
    coherence = float(np.clip(np.mean(cos), 0.0, 1.0)) ** 2 if ok.any() else 0.0
    # 1 コマあたりの移動量が急に何倍にもなる軌道は、別の物に飛び移っている（シャトルは減速だけで、1 コマで半分程度まで）
    sp = nv / np.maximum(np.diff(t), 1e-9)
    jump = np.abs(np.diff(np.log(np.maximum(sp, 1e-6)))) if len(sp) >= 2 else np.zeros(1)
    coherence *= float(np.exp(-max(0.0, float(jump.max()) - 0.9)))
    consistency = 1.0
    if len(tr.strengths) == len(t) and len(t) >= 2:
        jumps = np.abs(np.diff(np.log(np.maximum(tr.strengths, 1e-6))))
        consistency = float(np.exp(-max(0.0, np.median(jumps) - 0.3)))  # 隣のコマとの強さの比が 1.35 倍以内なら減点なし
    return float(len(t) ** 1.5 * speed / (1.0 + resid) * coherence * consistency)


def remove_outliers(t: np.ndarray, xy: np.ndarray, thr: float, n_neighbors: int = 4,
                    step_ratio: float = 0.25) -> np.ndarray:
    """近くの点に当てはめた 2 次式から大きく外れた点を、最も外れたものから 1 つずつ除く。

    許容するずれは ``thr`` [px] と「その点の前後の 1 コマあたりの移動量 × step_ratio」の大きい方
    （速いシャトルの残像は重心が数 px〜十数 px ぶれるため）。端の点は近い 4 点からの外挿で調べるが、
    外挿は誤差が大きいので許容を 2 倍にする。打った直後は画面上で急に減速する（遠ざかるシャトル）ので、
    近くの点の移動量の中央値ではなく、その点自身の移動量を使う（序盤の大事な点を外さないため）。
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
            near = [k for k in (pos - 1, pos + 1) if 0 <= k < len(idx)]
            step = max(np.linalg.norm(xy[i] - xy[idx[k]]) / max(abs(idx[k] - i), 1) for k in near)
            extrap = 2.0 if (tt.min() > 0 or tt.max() < 0) else 1.0
            r = np.linalg.norm(xy[i] - pred) / (extrap * max(thr, step_ratio * step))
            if r > worst_r:
                worst, worst_r = i, r
        if worst < 0:
            break
        keep[worst] = False
    return keep


def detect(meta: VideoMeta, start_s: float | None = None, end_s: float | None = None,
           params: BaselineParams | None = None, n_tracks: int = 1,
           return_candidates: bool = False, exclusive: bool = True) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """動画の区間 [start_s, end_s] [s] でシャトルを検出する。

    Args:
        meta: 動画のメタデータ
        start_s, end_s: 区間（実時間）[s]
        params: 設定
        n_tracks: 出力する軌道の数（スコアの高い順）
        return_candidates: 候補の一覧も返す
        exclusive: 真なら 1 フレーム 1 点（スコアの高い軌道を優先）。偽なら上位の軌道をすべて
            （同じフレームに複数行あってよい）出力する。解析では偽にして、打球音・打つ人の位置で軌道を選ぶ

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
    tracks = split_at_jumps(merge_tracks(link_tracks(cands, p), p))
    tracks = _dedupe_overlapping([tr for tr in tracks if len(tr.t) >= p.min_track_len])
    tracks = sorted(tracks, key=track_score, reverse=True)[:n_tracks]
    idx_of = {f: k for k, f in enumerate(out["frame"])}
    extra_rows = []
    for tid, tr in enumerate(tracks):
        t, xy = np.array(tr.t), np.array(tr.xy)
        keep = remove_outliers(t, xy, p.outlier_px)
        score = track_score(tr)
        for k, cid in enumerate(tr.ids):
            if not keep[k]:
                continue
            c = cands.iloc[cid]
            vals = [c.x / scale, c.y / scale, 1, min(1.0, score / (score + 1000.0)), tid,
                    c.length / scale, c.angle, c.area / scale**2]
            r = idx_of[tr.frames[k]]
            if out.at[r, "visible"]:
                if not exclusive:
                    extra_rows.append([tr.frames[k], tr.t[k], *vals])
                continue
            out.loc[r, ["x_px", "y_px", "visible", "confidence", "track_id", "length_px", "angle_deg", "area_px"]] = vals
    if extra_rows:
        out = pd.concat([out, pd.DataFrame(extra_rows, columns=out.columns)], ignore_index=True)
        out = out.sort_values(["frame", "track_id"], kind="stable").reset_index(drop=True)
    out["visible"] = out["visible"].astype(int)
    out["track_id"] = out["track_id"].astype(int)
    if return_candidates:
        cands[["x", "y", "length"]] /= scale
        return out, cands
    return out
