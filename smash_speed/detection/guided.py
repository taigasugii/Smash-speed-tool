"""打つ人の位置・飛行平面・運動方程式を手がかりにしたシャトルの探し方。

ベースライン検出は「画面の中でシャトルらしく動く物」を探すだけなので、打った直後にシャトルが腕・ラケットと
重なったり、カメラが打つ人の後ろにあってシャトルが選手・観客の前を横切ったりすると、別の物を選んでしまう。
打つ人の足元・着地点（飛行平面）と打球音がわかっていれば、シャトルの写る位置はかなり絞り込めるので、

1. ``corridor_polygon``: 打つ人から着地点までの空間（床〜高さ 4 m・飛行平面の左右 1 m 前後）を画像に投影し、
   検出する範囲を自動で決める
2. ``search_trajectory``: 打点（打つ人の足元の少し前・高さ 2.5 m 前後）から打球音の時刻に打ち出した軌道を、
   初速・打ち出し角・向きを変えて数千本作り、検出の候補を最も多く通る軌道を選ぶ（選手・観客の動きは
   打点から出る物理的な軌道に乗らないので選ばれない）
3. ``redetect_along``: 当てはめた軌道の予測位置のまわりを元の解像度で探し直し、最初の検出で漏れたコマを拾い、
   位置を細かく求め直す（使えるコマを増やし、4K 動画では縮小による誤差を減らす）

の 3 つを行う。座標はすべて元の解像度の px。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np
import pandas as pd

from smash_speed.camera import Camera
from smash_speed.geometry import FlightPlane
from smash_speed.physics import FitResult, PhysicsParams

EZ = np.array([0.0, 0.0, 1.0])


# =============================================================================
# 1. 検出する範囲
# =============================================================================
def corridor_polygon(camera: Camera, plane: FlightPlane, length_m: float, height_m: float = 4.0,
                     back_m: float = 1.0, ahead_m: float = 1.0, half_width_m: float = 0.8,
                     spread_deg: float = 8.0) -> np.ndarray | None:
    """打つ人から着地点までの空間を画像に投影した多角形 (K, 2) [px]（凸包）。

    空間は飛行平面に沿った台形の柱で、打つ人の足元の ``back_m`` 後ろから着地点の ``ahead_m`` 先まで、
    高さは床から ``height_m``、幅は打つ人の所で ±``half_width_m``、先へ行くほど ``spread_deg`` の割合で広げる
    （飛ぶ向きのずれを見込む）。角の点がカメラの後ろに出る（カメラがこの空間の中・すぐ横にある）ときは None。
    """
    s_lo, s_hi = -back_m, length_m + ahead_m
    corners = []
    for s in (s_lo, s_hi):
        hw = half_width_m + max(s, 0.0) * np.tan(np.radians(spread_deg))
        for n in (-hw, hw):
            for z in (0.0, height_m):
                corners.append(plane.origin + s * plane.direction + n * plane.normal + z * EZ)
    uv = camera.project(np.array(corners))
    if not np.isfinite(uv).all():
        return None
    hull = cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2).astype(float)
    # 画像の外に大きくはみ出した部分は切る（マスクを作るときに困らないよう）
    w, h = camera.width, camera.height
    hull[:, 0] = np.clip(hull[:, 0], -w, 2 * w)
    hull[:, 1] = np.clip(hull[:, 1], -h, 2 * h)
    return hull


def polygon_area_ratio(poly: np.ndarray, width: int, height: int) -> float:
    """多角形が画像のうち何割を占めるか（画像の外の部分は数えない）。"""
    mask = np.zeros((height // 8 + 1, width // 8 + 1), np.uint8)
    cv2.fillPoly(mask, [np.round(poly / 8).astype(np.int32)], 1)
    return float(mask.mean())


# =============================================================================
# 2. 打点から出る軌道の探索
# =============================================================================
@dataclass
class TrajectoryHypothesis:
    """``search_trajectory`` で選んだ軌道。

    Attributes:
        t_impact: 打点時刻 [s]
        p0: 打点 [m]
        v0: 初速度 [m/s]
        score: 予測位置での「小さく動く物らしさ」（0〜1）の、コマについての和
        n_frames: 予測位置が画面内にあったコマの数
        matched: この軌道に沿って拾ったシャトルの位置（列 ``frame, timestamp, x_px, y_px`` など）
    """

    t_impact: float
    p0: np.ndarray
    v0: np.ndarray
    score: float
    n_frames: int
    matched: pd.DataFrame

    @property
    def speed0(self) -> float:
        return float(np.linalg.norm(self.v0))


def _integrate_many(p0: np.ndarray, v0: np.ndarray, phys: PhysicsParams, t_max: float, dt: float
                    ) -> np.ndarray:
    """多数の軌道 (H, 3) を RK4 でまとめて積分し、時刻 0, dt, 2dt, ... の位置 (T, H, 3) を返す。"""
    g, k = phys.g, phys.k

    def acc(v: np.ndarray) -> np.ndarray:
        a = -k * np.linalg.norm(v, axis=1, keepdims=True) * v
        a[:, 2] -= g
        return a

    n = int(np.ceil(t_max / dt)) + 1
    out = np.empty((n, len(p0), 3))
    p, v = p0.copy(), v0.copy()
    out[0] = p
    for i in range(1, n):
        k1v = acc(v)
        k1p = v
        k2v = acc(v + 0.5 * dt * k1v)
        k2p = v + 0.5 * dt * k1v
        k3v = acc(v + 0.5 * dt * k2v)
        k3p = v + 0.5 * dt * k2v
        k4v = acc(v + dt * k3v)
        k4p = v + dt * k3v
        p = p + dt / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        v = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        out[i] = p
    return out


def _shuttle_like(d: np.ndarray, thr: float, max_minor_px: float, group_px: int = 3) -> np.ndarray:
    """差分の画像 d で、シャトルらしい形（小さな点か、まっすぐな細い残像）の塊の画素を 1 にした画像。

    塊ごとに画素の広がり（共分散）を求め、短い方向の広がり（標準偏差）が ``max_minor_px`` 以下のもの、または
    細長いもの（短い方向が長い方向の 2 割以下で、``3 * max_minor_px`` 以下）を残す。
    人・ラケットの輪郭は輪の形になるので、短い方向にも大きく広がっていて除かれる。
    """
    mask = (d > thr).astype(np.uint8)
    grouped = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (group_px, group_px)))
    n, labels = cv2.connectedComponents(grouped, connectivity=8)
    if n <= 1:
        return np.zeros(d.shape, np.float32)
    lab = labels[mask > 0]
    ys, xs = np.nonzero(mask)
    cnt = np.bincount(lab, minlength=n).astype(float)
    sx, sy = np.bincount(lab, xs, n), np.bincount(lab, ys, n)
    sxx, syy, sxy = np.bincount(lab, xs * xs, n), np.bincount(lab, ys * ys, n), np.bincount(lab, xs * ys, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx, my = sx / cnt, sy / cnt
        cxx, cyy, cxy = sxx / cnt - mx**2, syy / cnt - my**2, sxy / cnt - mx * my
        common = np.sqrt(((cxx - cyy) / 2) ** 2 + cxy**2)
        minor = np.sqrt(np.maximum((cxx + cyy) / 2 - common, 0))
        major = np.sqrt(np.maximum((cxx + cyy) / 2 + common, 0))
    # カメラのすぐ近くを通るシャトルは残像が太くなる。細長ければ（太さが長さの 2 割以下）太さ 3 倍まで許す
    good = (cnt >= 1) & ((minor <= max_minor_px) | ((minor <= 3 * max_minor_px) & (minor <= 0.2 * major)))
    good[0] = False
    return good[labels].astype(np.float32) * (mask > 0)


def evidence_maps(frames: Iterable[tuple[int, float, np.ndarray]], mask: np.ndarray | None = None,
                  blob_px: float = 1.2, diff_threshold: float = 8.0, max_minor_px: float = 3.0,
                  other_weight: float = 0.2, shape_floor: float = 0.6) -> list[tuple[int, float, np.ndarray]]:
    """縮小したグレースケール画像の並びから、コマごとの「シャトルらしさ」の地図を作る。

    3 フレーム差分（前後のコマとの差の小さい方）に、小さな点を強める帯域通過フィルタ（ガウスぼかしの差）をかけ、
    さらに形で重みをつける：小さな点・まっすぐな細い残像の塊はそのまま、人・ラケットの輪郭のような
    輪の形の塊は ``other_weight`` 倍に弱める（シャトルが人と重なって 1 つの塊になったコマでも少しは残す）。
    コマの中のいちばん強い反応の 0.8 倍を 1 として 0〜1 に直し（8 bit）、予測位置のずれを許すため 3 px の範囲の最大値をとる。
    シャトルらしい形の塊の画素は、少なくとも ``shape_floor`` にする（低い位置から撮った速いシャトルは、長く薄い残像に
    なって差分が人の輪郭よりずっと弱く、明るさで正規化するだけでは地図にほとんど残らないため）。

    Returns:
        (番号, 実時間 [s], 地図 uint8) のリスト（最初と最後のコマは差分が作れないので含まない）
    """
    out = []
    buf: list[tuple[int, float, np.ndarray]] = []
    for i, t, g in frames:
        buf.append((i, t, cv2.GaussianBlur(g, (3, 3), 0).astype(np.int16)))
        if len(buf) < 3:
            continue
        (_, _, a), (fi, ft, b), (_, _, c) = buf[-3:]
        buf = buf[-2:]
        d = np.minimum(np.abs(b - a), np.abs(c - b)).astype(np.float32)
        if mask is not None:
            d *= mask > 0
        med = float(np.median(d))
        mad = float(np.median(np.abs(d - med))) + 1.0
        shape_w = _shuttle_like(d, max(diff_threshold, med + 5 * mad), max_minor_px)
        r = cv2.GaussianBlur(d, (0, 0), blob_px) - cv2.GaussianBlur(d, (0, 0), 4 * blob_px)
        r = np.maximum(r, 0)
        wgt = np.maximum(cv2.dilate(shape_w, np.ones((3, 3), np.uint8)), other_weight)
        r *= wgt
        # いちばん強い反応の 0.8 倍を 1 とする（シャトルは白く、人の輪郭より差分がずっと大きいことが多いので、
        # 上位の割合で正規化して飽和させると、その差が消えてしまう）
        top = max(0.8 * float(r.max()), 2.0)
        e = np.maximum(np.clip(r / top, 0, 1), shape_floor * shape_w)
        e = cv2.dilate(e, np.ones((3, 3), np.uint8))
        out.append((fi, ft, (e * 255).astype(np.uint8)))
    return out


def _hypotheses(hit_X: np.ndarray, plane: FlightPlane, along_m, across_m, heights_m, yaws_deg, elevs_deg, speeds
                ) -> tuple[np.ndarray, np.ndarray]:
    """打点の位置（前後・左右・高さ）× 向き × 打ち出し角 × 初速 の組み合わせ（打点 (H, 3) [m]、初速度 (H, 3) [m/s]）。"""
    d0, n0 = plane.direction, plane.normal
    P, V = [], []
    for a in np.atleast_1d(along_m):
        for c in np.atleast_1d(across_m):
            for dz in np.atleast_1d(heights_m):
                p = hit_X + a * d0 + c * n0 + dz * EZ
                for yaw in np.radians(np.atleast_1d(yaws_deg)):
                    d = np.cos(yaw) * d0 + np.sin(yaw) * n0
                    for el in np.radians(np.atleast_1d(elevs_deg)):
                        u = np.cos(el) * d + np.sin(el) * EZ
                        for sp in np.atleast_1d(speeds):
                            P.append(p)
                            V.append(sp * u)
    return np.array(P), np.array(V)


def _score_hypotheses(P: np.ndarray, V: np.ndarray, t_list: list[float], maps, camera: Camera,
                      scale: float, phys: PhysicsParams, max_flight_s: float, dt_int: float = 0.002,
                      coarse: list | None = None, steps_px: tuple[float, ...] = (30.0, 80.0),
                      frame_cost: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """各軌道 × 打点時刻の点数（予測位置での地図の値の和）と、画面内にあったコマの数。(len(t_list), H)

    ``coarse`` に、範囲の最大値をとって広げた地図の並び（``maps`` と同じコマ）を広げ方の小さい順に渡すと、
    予測位置が 1 コマに ``steps_px`` [地図の px] 以上動くコマでは、広げた地図で点数をつける（カメラのすぐ近くを
    通るシャトルは、軌道の刻みのわずかな違いで画面上の位置が大きく変わるので、ずれを広く許す）。
    画面内にあるコマごとに ``frame_cost`` を引く（ゆっくり長く画面に残る軌道が、背景の弱い反応を足し集めて
    速く画面を出ていく本物の軌道に勝たないように）。
    """
    traj = _integrate_many(P, V, phys, max_flight_s, dt_int)  # (T, H, 3)
    H = len(P)
    ts = np.array([t for _, t, _ in maps])
    scores = np.zeros((len(t_list), H))
    counts = np.zeros((len(t_list), H), int)
    mh, mw = maps[0][2].shape if maps else (0, 0)
    for ti, t_i in enumerate(t_list):
        idx = np.flatnonzero((ts > t_i) & (ts <= t_i + max_flight_s))
        if len(idx) == 0:
            continue
        x = (ts[idx] - t_i) / dt_int
        i0 = np.clip(np.floor(x).astype(int), 0, len(traj) - 2)
        w = (x - i0)[:, None, None]
        pos = traj[i0] * (1 - w) + traj[i0 + 1] * w  # (F, H, 3)
        uv = camera.project(pos.reshape(-1, 3)).reshape(len(idx), H, 2) * scale
        ok = (pos[:, :, 2] > 0.0) & np.isfinite(uv).all(axis=2)
        ix = np.where(ok, np.round(np.nan_to_num(uv[:, :, 0], nan=-1)), -1).astype(int)
        iy = np.where(ok, np.round(np.nan_to_num(uv[:, :, 1], nan=-1)), -1).astype(int)
        inside = ok & (ix >= 0) & (ix < mw) & (iy >= 0) & (iy < mh)
        if coarse:
            st = np.linalg.norm(np.diff(np.nan_to_num(uv), axis=0), axis=2)          # (F-1, H)
            st = np.vstack([st, st[-1:]]) if len(st) else np.zeros((len(idx), H))
            level = np.digitize(st, steps_px[:len(coarse)])                            # 0 = 元の地図
        for j, k in enumerate(idx):
            v = np.zeros(H)
            sel = inside[j]
            if coarse:
                for lv, src in enumerate([maps] + list(coarse)):
                    s_ = sel & (level[j] == lv)
                    if s_.any():
                        v[s_] = src[k][2][iy[j, s_], ix[j, s_]] / 255.0
            else:
                v[sel] = maps[k][2][iy[j, sel], ix[j, sel]] / 255.0
            scores[ti] += v - frame_cost * sel
        counts[ti] = inside.sum(axis=0)
    return scores, counts


def search_trajectory(maps: list[tuple[int, float, np.ndarray]], camera: Camera, plane: FlightPlane,
                      hit_X: np.ndarray, impact_times: Iterable[float], phys: PhysicsParams, scale: float,
                      max_flight_s: float = 1.2, time_offsets_s: tuple[float, ...] = (-0.006, -0.003, 0.0, 0.003)
                      ) -> TrajectoryHypothesis | None:
    """打点から打ち出した軌道を総当たりで作り、「小さく動く物らしさ」の地図の上を最も強く通るものを選ぶ。

    1 段目は初速 15〜115 m/s（5 m/s 刻み）・打ち出し角 −40〜+15°（5° 刻み）・向き ±8°（4° 刻み）・
    打点の前後・左右 ±0.4 m・高さ −0.4〜+0.3 m を調べ、2 段目でいちばんよい軌道のまわりを細かく調べる。
    個々のコマで検出の候補が取れていなくても（人と重なって 1 つの塊になっていても）、
    地図の値はシャトルの所で高くなるので、軌道全体として最もシャトルらしいものが選ばれる。

    Args:
        maps: ``evidence_maps`` の結果（縮小画像）
        camera: カメラ（元の解像度）
        plane: 飛行平面（向きの基準）
        hit_X: 打点の目安 [m]
        impact_times: 打点時刻の候補 [s]
        phys: 運動方程式の設定
        scale: 地図の縮小率（縮小後 / 元の解像度）
        max_flight_s: 打点からこの時間までを見る [s]
        time_offsets_s: 打点時刻の候補のまわりで試すずれ [s]

    Returns:
        最もよい軌道（``matched`` は空。位置は ``redetect_along`` で拾う）。見つからなければ None
    """
    impact_times = [float(t) for t in impact_times]
    if not maps or not impact_times:
        return None
    t_list = [t + off for t in impact_times for off in time_offsets_s]
    P, V = _hypotheses(hit_X, plane, (-0.4, 0.0, 0.4), (-0.4, 0.0, 0.4), (-0.4, 0.0, 0.3), np.arange(-8.0, 9.0, 4.0),
                       np.arange(-40.0, 16.0, 5.0), np.arange(15.0, 116.0, 5.0))
    # 1 段目は刻みが粗く、本当の軌道から数 px ずれるので、地図を広げて（9 px の範囲の最大値）点数をつける
    def grow(src, k):
        return [(f, t, cv2.dilate(m, np.ones((k, k), np.uint8))) for f, t, m in src]

    wide, wider, widest = grow(maps, 9), grow(maps, 17), grow(maps, 33)
    sc, cnt = _score_hypotheses(P, V, t_list, wide, camera, scale, phys, max_flight_s, coarse=[wider, widest])
    ti, h = np.unravel_index(int(np.argmax(sc)), sc.shape)
    # 2 段目：いちばんよい軌道のまわりを細かく
    p_best, v_best = P[h], V[h]
    sp = float(np.linalg.norm(v_best))
    u = v_best / sp
    el = float(np.degrees(np.arcsin(np.clip(u[2], -1, 1))))
    yaw = float(np.degrees(np.arctan2(u @ plane.normal, u @ plane.direction)))
    a0 = float((p_best - hit_X) @ plane.direction)
    c0 = float((p_best - hit_X) @ plane.normal)
    z0 = float(p_best[2] - hit_X[2])
    t0 = t_list[ti]
    t2 = [t0 + off for off in (-0.002, -0.001, 0.0, 0.001, 0.002)]
    P2, V2 = _hypotheses(hit_X, plane, a0 + np.array([-0.2, 0.0, 0.2]), c0 + np.array([-0.2, 0.0, 0.2]),
                         z0 + np.array([-0.15, 0.0, 0.15]),
                         yaw + np.arange(-3.0, 3.1, 1.5), el + np.arange(-3.0, 3.1, 1.5), sp + np.arange(-5.0, 5.1, 1.25))
    sc2, cnt2 = _score_hypotheses(P2, V2, t2, maps, camera, scale, phys, max_flight_s, coarse=[wide, wider])
    ti2, h2 = np.unravel_index(int(np.argmax(sc2)), sc2.shape)
    score, n_in, t_i, p0, v0 = float(sc2[ti2, h2]), int(cnt2[ti2, h2]), t2[ti2], P2[h2], V2[h2]
    if score < 2.0:  # 画面内のコマごとに frame_cost を引いた点数。はっきり重なるコマが 4 つほどあれば通る
        return None
    return TrajectoryHypothesis(t_i, p0, v0, score, n_in, pd.DataFrame(columns=["frame", "timestamp", "x_px", "y_px"]))


def trajectory_predictor(p0: np.ndarray, v0: np.ndarray, t_ref: float, phys: PhysicsParams, camera: Camera,
                         t_end: float):
    """時刻 t_ref に p0 [m]・v0 [m/s] で打ち出した軌道の予測位置 [px] を返す関数
    （打点より前・着地後・``t_end`` より後・画面外は NaN）。"""
    from smash_speed.physics import simulate_until_ground

    traj = simulate_until_ground(p0, v0, t_ref, phys, t_max=max(0.1, t_end - t_ref))
    t_grid = traj["t"]
    uv_grid = camera.project(traj["pos"])

    def predict(t: np.ndarray) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, float))
        out = np.full((len(t), 2), np.nan)
        ok = (t >= t_ref) & (t <= min(t_grid[-1], t_end))
        if ok.any():
            out[ok, 0] = np.interp(t[ok], t_grid, uv_grid[:, 0])
            out[ok, 1] = np.interp(t[ok], t_grid, uv_grid[:, 1])
        w, h = camera.width, camera.height
        off = ~np.isfinite(out).all(axis=1) | (out[:, 0] < 0) | (out[:, 0] >= w) | (out[:, 1] < 0) | (out[:, 1] >= h)
        out[off] = np.nan
        return out

    return predict


# =============================================================================
# 3. 予測位置のまわりでの探し直し
# =============================================================================
@dataclass
class RedetectParams:
    """``redetect_along`` の設定（距離は元の解像度の px。``unit_px`` 倍して使う値は縮小画像の px）。"""

    radius_px: float = 6.0          # 探す範囲の半径の基本値（縮小画像の px）
    radius_step_ratio: float = 0.35  # 1 コマの移動量に対する探す範囲の広げ方
    max_radius_px: float = 45.0     # 探す範囲の半径の上限（縮小画像の px）
    diff_threshold: float = 9.0     # 差分の閾値 [輝度]（最初の検出より低くして薄い残像も拾う）
    k_mad: float = 6.0              # 雑音の MAD のこの倍数より弱い差分は使わない
    local_px: float = 4.0           # 大きな塊（ラケットなどと重なった）のときに重心を取る範囲（縮小画像の px）
    group_px: float = 13.0          # 途切れた残像をまとめる膨張の大きさ（縮小画像の px。ベースライン検出と同じ）


def _minor_std(xs: np.ndarray, ys: np.ndarray) -> float:
    """画素の広がりの短い方向の標準偏差 [px]。"""
    if len(xs) < 3:
        return 0.0
    ev = np.linalg.eigvalsh(np.cov(np.vstack([np.asarray(xs, float).ravel(), np.asarray(ys, float).ravel()])))
    return float(np.sqrt(max(ev[0], 0.0)))


def redetect_along(frames: Iterable[tuple[int, float, np.ndarray]], predict, unit_px: float = 1.0,
                   p: RedetectParams | None = None) -> pd.DataFrame:
    """予測位置のまわりでシャトルを探し直す。

    Args:
        frames: 元の解像度のグレースケール画像の並び（番号, 実時間 [s], 画像）。探すコマの前後 1 コマずつを含める
        predict: 実時間 [s] の配列 → 予測位置 (N, 2) [px]（画面外・着地後は NaN）
        unit_px: 最初の検出の縮小画像の 1 px が元の解像度で何 px か
        p: 設定

    Returns:
        列 ``frame, timestamp, x_px, y_px, dist_px, area_px, merged`` の表（見つかったコマだけ）。
        ``merged`` はほかの動く物（ラケット・腕など）と 1 つの塊になっていて、予測位置の近くだけで重心を取ったもの
    """
    p = p or RedetectParams()
    buf: list[tuple[int, float, np.ndarray]] = []
    rows = []
    for i, t, g in frames:
        buf.append((i, t, cv2.GaussianBlur(g, (3, 3), 0)))
        if len(buf) < 3:
            continue
        (_, ta, a), (fi, ft, b), (_, tc, c) = buf[-3:]
        buf = buf[-2:]
        uv = predict(np.array([ft]))[0]
        if not np.isfinite(uv).all():
            continue
        nxt = predict(np.array([ft + 1e-3]))[0]
        speed_img = np.linalg.norm(nxt - uv) / 1e-3 if np.isfinite(nxt).all() else 0.0
        step = speed_img * (tc - ta) / 2  # 1 コマの間の画面上の移動量 [px]
        R = min(unit_px * p.max_radius_px, unit_px * p.radius_px + p.radius_step_ratio * step)
        h, w = b.shape[:2]
        # 探す窓は、1 コマの間に動く長さの残像がまるごと入る大きさにする
        half = max(2 * R, 0.75 * step + R)
        x0, y0 = int(max(0, np.floor(uv[0] - half))), int(max(0, np.floor(uv[1] - half)))
        x1, y1 = int(min(w, np.ceil(uv[0] + half) + 1)), int(min(h, np.ceil(uv[1] + half) + 1))
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        d = cv2.min(cv2.absdiff(b[y0:y1, x0:x1], a[y0:y1, x0:x1]), cv2.absdiff(c[y0:y1, x0:x1], b[y0:y1, x0:x1]))
        df = d.astype(np.float32)
        med = float(np.median(df))
        mad = float(np.median(np.abs(df - med))) + 1.0
        thr = max(p.diff_threshold, med + p.k_mad * mad)
        mask = (df > thr).astype(np.uint8)
        # 薄い残像は雑音で切れ切れになる。予測位置に近い切れ端だけを取ると位置が予測の側に寄るので、まとめてから重心を取る
        k = max(1, int(round(unit_px * p.group_px)) | 1)
        grouped = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
        labels[mask == 0] = 0
        ys, xs = np.mgrid[y0:y1, x0:x1]
        best = None
        for j in range(1, n):
            sub = labels == j
            wts = df * sub
            s = float(wts.sum())
            if s <= 0:
                continue
            cx, cy = float((wts * xs).sum() / s), float((wts * ys).sum() / s)
            bx, by, bw, bh = stats[j, :4]
            # 探す範囲の外まで広がる塊は、ラケット・腕・人と重なっている。予測位置の近くだけで重心を取る
            touches = bool(bx == 0 or by == 0 or bx + bw >= x1 - x0 or by + bh >= y1 - y0)
            # 細くまっすぐで、長さが 1 コマの移動量ほどまでの塊は、長くてもシャトルの残像（ほかの物と重なっていない）
            streak = (not touches and _minor_std(xs[sub], ys[sub]) <= unit_px * 2.0
                      and max(bw, bh) <= 1.3 * step + 2 * R)
            merged = bool(not streak and (bw > 2 * R or bh > 2 * R or touches))
            if merged:
                rl = unit_px * p.local_px + 0.25 * step
                c_, sw = uv.copy(), 0.0
                for _ in range(4):  # 予測位置から始めて、近くの差分の重心へ寄せる
                    near = sub & ((xs - c_[0]) ** 2 + (ys - c_[1]) ** 2 <= rl**2)
                    sw = float((df * near).sum())
                    if sw <= 0:
                        break
                    c_ = np.array([(df * near * xs).sum() / sw, (df * near * ys).sum() / sw])
                if sw <= 0:
                    continue
                cx, cy = float(c_[0]), float(c_[1])
            dist = float(np.hypot(cx - uv[0], cy - uv[1]))
            if dist > R:
                continue
            cost = dist / R + (0.5 if merged else 0.0)
            if best is None or cost < best[0]:
                best = (cost, cx, cy, dist, float(sub.sum()), merged, j)
        if best is not None:
            _, cx, cy, dist, area, merged, j = best
            if not merged and step > 0 and np.isfinite(nxt).all():
                # 長い残像は雑音で途中が切れ、別の塊になることがある。予測した動きの向きの直線の上にある
                # 切れ端（ほかの動く物と重なっていない塊）もまとめて、残像全体の重心を取る
                ax = (nxt - uv) / max(float(np.linalg.norm(nxt - uv)), 1e-9)
                band = unit_px * 3.0 + 1.0
                along = (xs - cx) * ax[0] + (ys - cy) * ax[1]
                perp = np.abs(-(xs - cx) * ax[1] + (ys - cy) * ax[0])
                ok_lab = np.zeros(n, bool)
                for jj in range(1, n):
                    bx, by, bw, bh = stats[jj, :4]
                    ok_lab[jj] = max(bw, bh) <= 1.3 * step + 2 * R
                ok_lab[j] = True
                on_line = (labels > 0) & ok_lab[labels] & (perp <= band) & (np.abs(along) <= 0.6 * step + R)
                if on_line.sum() > (labels == j).sum():
                    wl = df * on_line
                    cx2, cy2 = float((wl * xs).sum() / wl.sum()), float((wl * ys).sum() / wl.sum())
                    d2 = float(np.hypot(cx2 - uv[0], cy2 - uv[1]))
                    if d2 <= R:
                        cx, cy, dist, area = cx2, cy2, d2, float(on_line.sum())
            rows.append({"frame": fi, "timestamp": ft, "x_px": cx, "y_px": cy, "dist_px": dist,
                         "area_px": area, "merged": merged})
    return pd.DataFrame(rows, columns=["frame", "timestamp", "x_px", "y_px", "dist_px", "area_px", "merged"])


def fit_predictor(fit: FitResult, phys: PhysicsParams, camera: Camera, t_end: float):
    """当てはめた軌道の予測位置 [px] を返す関数（``trajectory_predictor`` を参照）。"""
    return trajectory_predictor(fit.p0, fit.v0, fit.t_ref, fit.phys(phys), camera, t_end)


def match_candidates(cands: pd.DataFrame, predict, unit_px: float = 1.0, p: RedetectParams | None = None
                     ) -> pd.DataFrame:
    """ベースライン検出の候補のうち、コマごとに予測位置にいちばん近いもの（探す範囲の内側）を選ぶ。

    Args:
        cands: ベースライン検出の候補（列 ``frame, timestamp, x, y``。元の解像度 px）
        predict: 実時間 [s] の配列 → 予測位置 (N, 2) [px]
        unit_px, p: ``redetect_along`` と同じ（探す範囲の決め方）

    Returns:
        列 ``frame, timestamp, x_px, y_px, dist_px`` の表
    """
    p = p or RedetectParams()
    rows = []
    for f, g in cands.groupby("frame"):
        t = float(g.timestamp.iloc[0])
        uv = predict(np.array([t]))[0]
        if not np.isfinite(uv).all():
            continue
        nxt = predict(np.array([t + 1e-3]))[0]
        dt = float(np.median(np.diff(np.sort(cands.timestamp.unique())))) if cands.timestamp.nunique() > 1 else 0.0
        step = np.linalg.norm(nxt - uv) / 1e-3 * dt if np.isfinite(nxt).all() else 0.0
        R = min(unit_px * p.max_radius_px, unit_px * p.radius_px + p.radius_step_ratio * step)
        d = np.hypot(g.x.to_numpy() - uv[0], g.y.to_numpy() - uv[1])
        k = int(np.argmin(d))
        if d[k] <= R:
            rows.append({"frame": int(f), "timestamp": t, "x_px": float(g.x.iloc[k]), "y_px": float(g.y.iloc[k]),
                         "dist_px": float(d[k])})
    return pd.DataFrame(rows, columns=["frame", "timestamp", "x_px", "y_px", "dist_px"])
