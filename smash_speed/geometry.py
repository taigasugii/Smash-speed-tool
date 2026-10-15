"""3次元化：平面拘束（モード A）と2台カメラの三角測量（モード B）。

モード A では、シャトルは打者の足元と着地点を通る **鉛直な1枚の平面** 内を飛ぶと仮定する。
カメラからシャトルへの視線とその平面の交点を、シャトルの3次元位置とする。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from smash_speed.camera import Camera

EZ = np.array([0.0, 0.0, 1.0])


@dataclass
class FlightPlane:
    """シャトルの飛行平面（鉛直）。

    平面上の点は ``origin + s * direction + z * ez`` で表す。

    Attributes:
        origin: 平面上の床面の点 [m]（通常は打者の足元、z = 0）
        direction: 平面の水平方向の単位ベクトル（打者 → 着地点）[-]
    """

    origin: np.ndarray
    direction: np.ndarray

    def __post_init__(self) -> None:
        self.origin = np.asarray(self.origin, float).reshape(3).copy()
        self.origin[2] = 0.0
        d = np.asarray(self.direction, float).reshape(3).copy()
        d[2] = 0.0
        n = np.linalg.norm(d)
        if n == 0:
            raise ValueError("飛行平面の方向がゼロベクトルです（2点が同じ位置）")
        self.direction = d / n

    @classmethod
    def from_points(cls, hitter_xy: np.ndarray, landing_xy: np.ndarray) -> "FlightPlane":
        """床面上の2点（打者の足元 → 着地点）[m] から平面を作る。"""
        a = np.r_[np.asarray(hitter_xy, float)[:2], 0.0]
        b = np.r_[np.asarray(landing_xy, float)[:2], 0.0]
        return cls(origin=a, direction=b - a)

    @property
    def normal(self) -> np.ndarray:
        """平面の法線（水平）。"""
        return np.cross(EZ, self.direction)

    def to_world(self, s: np.ndarray, z: np.ndarray) -> np.ndarray:
        """平面内座標 (s, z) [m] → ワールド座標 (N, 3) [m]。"""
        s = np.atleast_1d(s)[:, None]
        z = np.atleast_1d(z)[:, None]
        return self.origin + s * self.direction + z * EZ

    def to_plane(self, X: np.ndarray) -> np.ndarray:
        """ワールド座標 (N, 3) → 平面内座標 (N, 2) = (s, z)。平面外の成分は捨てる。"""
        X = np.atleast_2d(X)
        return np.column_stack([(X - self.origin) @ self.direction, X[:, 2]])

    def to_dict(self) -> dict[str, Any]:
        return {"origin_m": self.origin.tolist(), "direction": self.direction.tolist()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FlightPlane":
        return cls(origin=np.array(d["origin_m"]), direction=np.array(d["direction"]))


def intersect_plane(camera: Camera, uv: np.ndarray, plane: FlightPlane) -> np.ndarray:
    """画像座標 (N, 2) [px] の視線と飛行平面の交点 (N, 3) [m] を返す。

    視線が平面とほぼ平行、または交点がカメラの後ろにある場合は NaN。
    """
    c, d = camera.rays(uv)
    n = plane.normal
    denom = d @ n
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = ((plane.origin - c) @ n) / denom
    X = c + lam[:, None] * d
    bad = (np.abs(denom) < 1e-9) | (lam <= 0)
    X[bad] = np.nan
    return X


def floor_point(camera: Camera, uv: np.ndarray, z: float = 0.0) -> np.ndarray:
    """画像座標 (N, 2) [px] の視線と水平面 z = ``z`` [m] の交点 (N, 3) を返す。"""
    c, d = camera.rays(uv)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = (z - c[2]) / d[:, 2]
    X = c + lam[:, None] * d
    X[(lam <= 0) | ~np.isfinite(lam)] = np.nan
    return X


def triangulate(cameras: list[Camera], uvs: list[np.ndarray]) -> np.ndarray:
    """複数カメラの対応点から3次元位置を線形三角測量（DLT）で求める。

    Args:
        cameras: カメラのリスト（2台以上）
        uvs: 各カメラの画像座標 (N, 2) [px]。NaN の行は使わない。

    Returns:
        (N, 3) [m]。有効なカメラが2台未満の点は NaN。
    """
    n = len(uvs[0])
    X = np.full((n, 3), np.nan)
    norms = [cam.undistort_normalized(uv) for cam, uv in zip(cameras, uvs)]
    Rts = [np.hstack([cam.R, cam.tvec[:, None]]) for cam in cameras]
    for i in range(n):
        rows = []
        for xn, Rt in zip(norms, Rts):
            x, y = xn[i]
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            rows.append(x * Rt[2] - Rt[0])
            rows.append(y * Rt[2] - Rt[1])
        if len(rows) < 4:
            continue
        _, _, vt = np.linalg.svd(np.asarray(rows))
        h = vt[-1]
        X[i] = h[:3] / h[3]
    return X
