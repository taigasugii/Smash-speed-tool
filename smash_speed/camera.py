"""ピンホールカメラモデル（放射歪み k1, k2 つき）。

座標系はコート座標（README「座標系」）[m]、画像座標はピクセル [px]（原点は左上、u 右・v 下）。
キャリブレーション・3次元化・軌道フィッティングはすべてこのクラスの投影関数を通して計算する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass
class Camera:
    """内部・外部パラメータを持つカメラ。

    Attributes:
        f: 焦点距離 [px]（正方画素を仮定）
        cx, cy: 主点 [px]
        width, height: 画像サイズ [px]
        rvec: ワールド→カメラの回転（Rodrigues ベクトル）[rad]
        tvec: ワールド→カメラの並進 [m]
        k1, k2: 放射歪み係数（正規化座標に対する）[-]
    """

    f: float
    cx: float
    cy: float
    width: int
    height: int
    rvec: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tvec: np.ndarray = field(default_factory=lambda: np.zeros(3))
    k1: float = 0.0
    k2: float = 0.0

    def __post_init__(self) -> None:
        self.rvec = np.asarray(self.rvec, dtype=float).reshape(3)
        self.tvec = np.asarray(self.tvec, dtype=float).reshape(3)

    # ------------------------------------------------------------------ 行列
    @property
    def K(self) -> np.ndarray:
        """内部パラメータ行列 (3, 3)。"""
        return np.array([[self.f, 0.0, self.cx], [0.0, self.f, self.cy], [0.0, 0.0, 1.0]])

    @property
    def R(self) -> np.ndarray:
        """ワールド→カメラの回転行列 (3, 3)。"""
        return cv2.Rodrigues(self.rvec)[0]

    @property
    def center(self) -> np.ndarray:
        """ワールド座標でのカメラ中心 [m]。"""
        return -self.R.T @ self.tvec

    @property
    def P(self) -> np.ndarray:
        """歪みを無視した投影行列 (3, 4)。"""
        return self.K @ np.hstack([self.R, self.tvec[:, None]])

    @property
    def dist_coeffs(self) -> np.ndarray:
        """OpenCV 形式の歪み係数 (k1, k2, p1, p2)。"""
        return np.array([self.k1, self.k2, 0.0, 0.0])

    # ------------------------------------------------------------------ 投影
    def project(self, X: np.ndarray) -> np.ndarray:
        """ワールド座標の点 (N, 3) [m] を画像座標 (N, 2) [px] に投影する。

        カメラの後ろにある点は NaN を返す。
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Xc = X @ self.R.T + self.tvec
        z = Xc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            xn = Xc[:, :2] / z[:, None]
        r2 = np.sum(xn**2, axis=1, keepdims=True)
        xd = xn * (1.0 + self.k1 * r2 + self.k2 * r2**2)
        uv = xd * self.f + np.array([self.cx, self.cy])
        uv[z <= 0] = np.nan
        return uv

    def undistort_normalized(self, uv: np.ndarray, iterations: int = 10) -> np.ndarray:
        """画像座標 (N, 2) [px] を歪み補正済みの正規化座標 (N, 2) に変換する。"""
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        xd = (uv - np.array([self.cx, self.cy])) / self.f
        if self.k1 == 0.0 and self.k2 == 0.0:
            return xd
        xn = xd.copy()
        for _ in range(iterations):  # 固定点反復で放射歪みを逆変換
            r2 = np.sum(xn**2, axis=1, keepdims=True)
            xn = xd / (1.0 + self.k1 * r2 + self.k2 * r2**2)
        return xn

    def rays(self, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """画像座標 (N, 2) [px] に対応する視線を返す。

        Returns:
            (origin (3,), directions (N, 3))。方向はワールド座標の単位ベクトル。
        """
        xn = self.undistort_normalized(uv)
        d_cam = np.hstack([xn, np.ones((len(xn), 1))])
        d = d_cam @ self.R  # R.T @ d_cam を行ベクトルで
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        return self.center, d

    # ------------------------------------------------------------------ 保存
    def to_dict(self) -> dict[str, Any]:
        """JSON に保存できる辞書を返す。"""
        return {
            "f_px": float(self.f), "cx_px": float(self.cx), "cy_px": float(self.cy),
            "width": int(self.width), "height": int(self.height),
            "rvec": self.rvec.tolist(), "tvec_m": self.tvec.tolist(),
            "k1": float(self.k1), "k2": float(self.k2),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Camera":
        """``to_dict`` の逆。"""
        return cls(
            f=d["f_px"], cx=d["cx_px"], cy=d["cy_px"], width=d["width"], height=d["height"],
            rvec=np.array(d["rvec"]), tvec=np.array(d["tvec_m"]),
            k1=d.get("k1", 0.0), k2=d.get("k2", 0.0),
        )

    def scaled(self, s: float) -> "Camera":
        """画像を s 倍に拡大縮小したときのカメラを返す（検出を縮小画像で行う場合など）。"""
        return Camera(
            f=self.f * s, cx=self.cx * s, cy=self.cy * s,
            width=round(self.width * s), height=round(self.height * s),
            rvec=self.rvec.copy(), tvec=self.tvec.copy(), k1=self.k1, k2=self.k2,
        )


def look_at(eye: np.ndarray, target: np.ndarray, f: float, width: int, height: int,
            up: np.ndarray = np.array([0.0, 0.0, 1.0])) -> Camera:
    """``eye`` [m] から ``target`` [m] を向いたカメラを作る（合成データ用）。"""
    eye = np.asarray(eye, float)
    fwd = np.asarray(target, float) - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.vstack([right, down, fwd])  # カメラ座標: x 右, y 下, z 前
    rvec = cv2.Rodrigues(R)[0].ravel()
    tvec = -R @ eye
    return Camera(f=f, cx=width / 2, cy=height / 2, width=width, height=height, rvec=rvec, tvec=tvec)
