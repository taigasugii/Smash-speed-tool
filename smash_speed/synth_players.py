"""選手の足の動きの合成データ（ROADMAP D2 の移動距離・速さの正解に使う）。

``footwork_path`` は、ホームの位置（自分の半面の中ほど）から打つ所へ素早く動き、少し止まってホームへ戻る、を繰り返す
足元の軌跡。1 回の移動は躍度最小（速さが山の形）で、3 m の移動で最高 5 m/s ほど（シングルスのフットワークに近い）。

``observe_players`` は、その足元をカメラで写したときの ``players.csv`` の形（``players.track_players`` の出力と同じ列）に直す。
足元の画像の位置には、実動画（IMG_4397 の 4K）で測った揺れ（枠の高さの 0.4〜0.6%）と、ときどき大きく外れるコマ
（足を開いた・踏み込んだときに枠の下の辺の中央がずれる）を足す。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from smash_speed.camera import Camera
from smash_speed.geometry import floor_point


def _min_jerk(k: np.ndarray) -> np.ndarray:
    return 10 * k ** 3 - 15 * k ** 4 + 6 * k ** 5


def footwork_path(duration_s: float, fps: float, side: str, rng: np.random.Generator,
                  court: dict[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """足元の軌跡。（時刻 (n,) [s]、コートの位置 (n, 2) [m]）を返す。h の側は x < 0、o の側は x > 0。"""
    c = court or {}
    half_len, half_w = c.get("half_length_m", 6.70), c.get("half_width_singles_m", 2.59)
    sx = -1.0 if side == "h" else 1.0
    home = np.array([sx * 3.4, 0.0])
    t = np.arange(int(round(duration_s * fps))) / fps
    xy = np.empty((len(t), 2))
    xy[:] = home
    pos, now, out = home.copy(), 0.0, True
    while now < duration_s:
        # 打つ所（自分の半面のどこか）と、ホームの近くへの戻りを交互に
        if out:
            target = np.array([sx * rng.uniform(0.9, half_len - 0.4), rng.uniform(-half_w + 0.3, half_w - 0.3)])
        else:
            target = home + rng.normal(0, 0.2, 2)
        dur = 0.4 + float(np.hypot(*(target - pos))) / 4.0
        a = now + rng.uniform(0.15, 0.4)   # 止まっている間
        b = a + dur
        xy[t >= b] = target
        inside = (t >= a) & (t < b)
        xy[inside] = pos + (target - pos) * _min_jerk((t[inside] - a) / dur)[:, None]
        pos, now, out = target, b, not out
    return t, xy


def path_length(xy: np.ndarray) -> float:
    return float(np.hypot(*np.diff(xy, axis=0).T).sum())


def _slow_noise(n: int, fps: float, tau_s: float, rng: np.random.Generator) -> np.ndarray:
    """標準偏差 1、時間の幅 ``tau_s`` で相関する揺れ（1 次の自己回帰）。"""
    a = float(np.exp(-1.0 / (tau_s * fps)))
    e = rng.normal(0, np.sqrt(1 - a * a), n)
    out = np.empty(n)
    out[0] = rng.normal()
    for k in range(1, n):
        out[k] = a * out[k - 1] + e[k]
    return out


def observe_players(t: np.ndarray, feet: dict[str, np.ndarray], camera: Camera, rng: np.random.Generator,
                    jitter: float = 0.005, outlier_rate: float = 0.05, outlier: float = 0.04,
                    drift: float = 0.01, swing: float = 0.12, height_m: float = 1.75) -> pd.DataFrame:
    """足元の軌跡（側 → (n, 2)）を写した ``players.csv`` の形。揺れの大きさは枠の高さに対する割合。

    揺れは 4 つ：コマごとの小さな揺れ（``jitter``）、ときどき大きく外れるコマ（``outlier``）、
    0.2 秒ほど続くゆっくりした揺れ（``drift``。足の開き・体の傾き）、止まって打つたびに枠が横へ広がる分
    （``swing``。腕とラケットを伸ばすと、枠の下の辺の中央が 0.35 秒ほど横へずれる）。
    """
    rows = []
    fps = 1.0 / float(np.median(np.diff(t)))
    for side, xy in feet.items():
        n = len(xy)
        foot = camera.project(np.column_stack([xy, np.zeros(n)]))
        head = camera.project(np.column_stack([xy, np.full(n, height_m)]))
        h = foot[:, 1] - head[:, 1]
        noise = rng.normal(0, jitter, (n, 2)) * h[:, None]
        wild = rng.random(n) < outlier_rate
        noise[wild] += rng.normal(0, outlier, (int(wild.sum()), 2)) * h[wild, None]
        noise += drift * np.column_stack([_slow_noise(n, fps, 0.2, rng), _slow_noise(n, fps, 0.2, rng)]) * h[:, None]
        # 止まったコマ（速さが小さい）の始まりごとに 1 回打つ
        speed = np.r_[0.0, np.hypot(*np.diff(xy, axis=0).T)] * fps
        still = speed < 0.05
        for k in np.flatnonzero(still & ~np.r_[False, still[:-1]]):
            w = (t - t[k]) / 0.35
            bump = np.where(np.abs(w) < 1, 0.5 * (1 + np.cos(np.pi * w)), 0.0)
            noise[:, 0] += rng.choice([-1, 1]) * swing * bump * h
        uv = foot + noise
        X = floor_point(camera, uv)
        w = 0.4 * h
        rows.append(pd.DataFrame({
            "frame": np.arange(n), "t": t, "side": side, "player": 0, "x": X[:, 0], "y": X[:, 1],
            "u": uv[:, 0], "v": uv[:, 1], "x0": uv[:, 0] - w / 2, "y0": uv[:, 1] - h, "x1": uv[:, 0] + w / 2,
            "y1": uv[:, 1], "score": 0.9, "interpolated": False, "airborne": False}))
    return pd.concat(rows, ignore_index=True).sort_values(["frame", "side"]).reset_index(drop=True)
