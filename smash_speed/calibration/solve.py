"""カメラパラメータの推定（クリックしたランドマークの画像座標と実寸から）。

手順:
1. 焦点距離 f の候補を幾何級数的に並べ、各 f で ``cv2.solvePnP`` を解いて再投影誤差が最小のものを選ぶ。
   非同一平面の点が6点以上あれば DLT（3x4 投影行列）を分解した f も候補に加える。
2. 主点を画像中心に固定し、(log f, rvec, tvec[, k1]) を ``least_squares`` で精緻化する。

床面の点だけでも f を推定できるが、ネットポスト上端など **高さのある点** を加えると安定する。
チェッカーボードで内部パラメータを事前に求めた場合は、f・主点・歪みを固定して外部パラメータだけを解く。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares

from smash_speed.camera import Camera


@dataclass
class Intrinsics:
    """事前に求めた内部パラメータ（チェッカーボードなど）。"""

    f: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    width: int = 0
    height: int = 0

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Intrinsics":
        return cls(**d)


@dataclass
class CalibrationResult:
    """推定結果。"""

    camera: Camera
    names: list[str]
    world: np.ndarray        # (N, 3) [m]
    image: np.ndarray        # (N, 2) [px]
    errors_px: np.ndarray    # (N,) 各点の再投影誤差 [px]
    method: str
    warnings: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)  # クリックの間違いとみなして推定から外した点

    @property
    def used(self) -> np.ndarray:
        """推定に使った点のマスク。"""
        return np.array([n not in self.excluded for n in self.names], bool)

    @property
    def rmse_px(self) -> float:
        """推定に使った点の再投影誤差の RMS [px]（除外した点は含めない）。"""
        e = self.errors_px[self.used]
        return float(np.sqrt(np.mean(e**2))) if len(e) else float("nan")

    @property
    def rmse_all_px(self) -> float:
        return float(np.sqrt(np.mean(self.errors_px**2)))


def dlt(world: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Direct Linear Transform で 3x4 投影行列を推定する（6点以上・非同一平面が必要）。

    数値安定化のため、ワールド座標・画像座標をそれぞれ正規化してから解く。
    """
    n = len(world)
    if n < 6:
        raise ValueError("DLT には6点以上が必要です")

    def norm_mat(pts: np.ndarray) -> np.ndarray:
        c = pts.mean(axis=0)
        s = np.sqrt(pts.shape[1]) / np.mean(np.linalg.norm(pts - c, axis=1))
        T = np.eye(pts.shape[1] + 1)
        T[:-1, :-1] *= s
        T[:-1, -1] = -s * c
        return T

    Tw, Ti = norm_mat(world), norm_mat(image)
    Xw = (Tw @ np.c_[world, np.ones(n)].T).T
    xi = (Ti @ np.c_[image, np.ones(n)].T).T
    A = []
    for X, (u, v, _) in zip(Xw, xi):
        A.append(np.r_[X, np.zeros(4), -u * X])
        A.append(np.r_[np.zeros(4), X, -v * X])
    _, _, vt = np.linalg.svd(np.asarray(A))
    Pn = vt[-1].reshape(3, 4)
    P = np.linalg.inv(Ti) @ Pn @ Tw
    return P / np.linalg.norm(P[2, :3])


def is_coplanar(world: np.ndarray, tol: float = 1e-3) -> bool:
    """点群がほぼ同一平面上にあるか。"""
    c = world - world.mean(axis=0)
    s = np.linalg.svd(c, compute_uv=False)
    return s[-1] < tol * max(s[0], 1e-12)


def _pnp(world: np.ndarray, image: np.ndarray, f: float, cx: float, cy: float,
         dist: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float] | None:
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], float)
    dist = np.zeros(4) if dist is None else dist
    flag = cv2.SOLVEPNP_SQPNP if len(world) >= 3 else cv2.SOLVEPNP_ITERATIVE
    try:
        ok, rvec, tvec = cv2.solvePnP(world.astype(np.float64), image.astype(np.float64), K, dist,
                                      flags=flag)
    except cv2.error:
        return None
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(world, image, K, dist, rvec, tvec)
    proj, _ = cv2.projectPoints(world, rvec, tvec, K, dist)
    err = np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - image) ** 2, axis=1)))
    # カメラの前に点があるか確認
    R = cv2.Rodrigues(rvec)[0]
    if np.any((world @ R.T + tvec.ravel())[:, 2] <= 0):
        return None
    return rvec.ravel(), tvec.ravel(), float(err)


def calibrate(names: list[str], world: np.ndarray, image: np.ndarray, width: int, height: int,
              intrinsics: Intrinsics | None = None, estimate_k1: bool = False,
              warn_px: float = 5.0, reject_outliers: bool = True) -> CalibrationResult:
    """ランドマークの対応からカメラを推定する。

    ``reject_outliers`` が真なら、ほかの点と明らかに合わない点（別の場所をクリックした点）を
    1 点ずつ外して推定し直す。外した点は ``excluded`` に入り、誤差は最終的なカメラで計算する。
    """
    res = _calibrate(names, world, image, width, height, intrinsics, estimate_k1, warn_px)
    if not reject_outliers:
        return res
    world, image, names = np.asarray(world, float), np.asarray(image, float), list(names)
    # 外れ値の基準：30 px 以上、かつ残りの点の誤差の中央値の 6 倍以上
    floor_px = 30.0
    excluded: list[str] = []
    while len(names) - len(excluded) >= 7 and len(excluded) < len(names) // 4:
        keep = np.array([n not in excluded for n in names])
        e = np.linalg.norm(res.camera.project(world) - image, axis=1)
        e_keep = np.where(keep, e, -1.0)
        j = int(np.argmax(e_keep))
        med = float(np.median(e[keep & (np.arange(len(e)) != j)]))
        if e[j] <= max(floor_px, 6.0 * med):
            break
        excluded.append(names[j])
        keep[j] = False
        res = _calibrate([n for n, k in zip(names, keep) if k], world[keep], image[keep], width, height,
                         intrinsics, estimate_k1, warn_px)
    if not excluded:
        return res
    errors = np.linalg.norm(res.camera.project(world) - image, axis=1)
    errors = np.where(np.isfinite(errors), errors, 1e4)
    warns = [f"{n}: 再投影誤差 {errors[names.index(n)]:.0f} px。ほかの点と合わないため、推定から外しました"
             "（別の場所をクリックした可能性があります）。" for n in excluded]
    return CalibrationResult(res.camera, names, world, image, errors, res.method, warns + res.warnings, excluded)


def _calibrate(names: list[str], world: np.ndarray, image: np.ndarray, width: int, height: int,
               intrinsics: Intrinsics | None = None, estimate_k1: bool = False,
               warn_px: float = 5.0) -> CalibrationResult:
    """すべての点を使ってカメラを推定する（``calibrate`` の本体）。

    Args:
        names: ランドマーク名
        world: 実寸の座標 (N, 3) [m]
        image: クリックした画像座標 (N, 2) [px]
        width, height: 画像サイズ [px]
        intrinsics: 既知の内部パラメータ（あれば外部パラメータだけ推定）
        estimate_k1: 放射歪み k1 も推定するか（点が多い場合のみ推奨）
        warn_px: 再投影誤差の警告閾値 [px]
    """
    world = np.asarray(world, float)
    image = np.asarray(image, float)
    n = len(world)
    warnings: list[str] = []
    if n < 4:
        raise ValueError(f"キャリブレーションには4点以上が必要です（現在 {n} 点）")
    coplanar = is_coplanar(world)
    if coplanar:
        warnings.append("全点が同一平面上にあります。高さのある点（ネットポスト上端など）を加えると焦点距離が安定します。")

    if intrinsics is not None:
        dist = np.array([intrinsics.k1, intrinsics.k2, 0, 0])
        r = _pnp(world, image, intrinsics.f, intrinsics.cx, intrinsics.cy, dist)
        if r is None:
            raise RuntimeError("solvePnP が解けませんでした")
        cam = Camera(intrinsics.f, intrinsics.cx, intrinsics.cy, width, height, r[0], r[1],
                     intrinsics.k1, intrinsics.k2)
        method = "pnp_fixed_intrinsics"
    else:
        cx, cy = width / 2, height / 2
        candidates = list(np.geomspace(0.3, 5.0, 60) * max(width, height))
        if n >= 6 and not coplanar:
            try:
                K, _, _, _, _, _, _ = cv2.decomposeProjectionMatrix(dlt(world, image))
                K = K / K[2, 2]
                f_dlt = float(abs(K[0, 0]) + abs(K[1, 1])) / 2
                if np.isfinite(f_dlt) and f_dlt > 0:
                    candidates.append(f_dlt)
            except (ValueError, cv2.error, np.linalg.LinAlgError):
                pass
        best = None
        for f in candidates:
            r = _pnp(world, image, f, cx, cy)
            if r is not None and (best is None or r[2] < best[3]):
                best = (f, r[0], r[1], r[2])
        if best is None:
            raise RuntimeError("どの焦点距離でも solvePnP が解けませんでした。点の対応を確認してください。")
        f0, rvec0, tvec0, _ = best

        n_par = 7 + int(estimate_k1)
        if 2 * n < n_par:
            warnings.append("点が少ないため k1 は推定しません。")
            estimate_k1 = False

        def unpack(p: np.ndarray) -> Camera:
            return Camera(np.exp(p[0]), cx, cy, width, height, p[1:4], p[4:7],
                          p[7] if estimate_k1 else 0.0)

        def resid(p: np.ndarray) -> np.ndarray:
            uv = unpack(p).project(world)
            r = (uv - image).ravel()
            return np.where(np.isfinite(r), r, 1e4)

        p0 = np.r_[np.log(f0), rvec0, tvec0, [0.0] if estimate_k1 else []]
        sol = least_squares(resid, p0, loss="soft_l1", f_scale=warn_px, x_scale="jac")
        cam = unpack(sol.x)
        method = "pnp_focal_search+lm" + ("+k1" if estimate_k1 else "")

    errors = np.linalg.norm(cam.project(world) - image, axis=1)
    for name, e in zip(names, errors):
        if e > warn_px:
            warnings.append(f"{name}: 再投影誤差 {e:.1f} px（閾値 {warn_px} px）。クリック位置を確認してください。")
    if n <= 5 and intrinsics is None:
        warnings.append("点が5点以下です。誤差が小さく見えても過適合の可能性があります。")
    return CalibrationResult(cam, list(names), world, image, errors, method, warnings)


def intrinsics_from_checkerboard(images: list[np.ndarray], pattern: tuple[int, int],
                                 square_m: float) -> tuple[Intrinsics, float]:
    """チェッカーボード画像から内部パラメータ（f, 主点, k1, k2）を推定する（任意）。

    Args:
        images: BGR 画像のリスト
        pattern: 内側の交点の数 (列, 行)
        square_m: マス目の一辺 [m]

    Returns:
        (Intrinsics, 再投影誤差 RMS [px])
    """
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square_m
    obj_pts, img_pts = [], []
    h, w = images[0].shape[:2]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, corners = cv2.findChessboardCorners(gray, pattern)
        if ok:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
            obj_pts.append(objp)
            img_pts.append(corners)
    if len(obj_pts) < 3:
        raise ValueError(f"チェッカーボードを検出できた画像が {len(obj_pts)} 枚しかありません（3枚以上必要）")
    flags = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, (w, h), None, None, flags=flags)
    d = dist.ravel()
    return Intrinsics(float(K[0, 0]), float(K[0, 2]), float(K[1, 2]), float(d[0]), float(d[1]), w, h), float(rms)
