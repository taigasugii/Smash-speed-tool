"""正解がわかっている合成データの生成（テスト・精度検証用）。

物理モデルで既知の初速のスマッシュ軌道を作り、仮想カメラで画像座標に投影して、
ガウスノイズ・検出漏れ・モーションブラー（露光中の平均位置）を加える。
白い背景に小さな点が動く合成動画と、打球音のクリックを含む音声も生成できる。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from smash_speed.camera import Camera, look_at
from smash_speed.court import draw_court
from smash_speed.geometry import FlightPlane
from smash_speed.physics import PhysicsParams, simulate, simulate_until_ground


@dataclass
class SyntheticShot:
    """合成スマッシュ1本の正解値。

    Attributes:
        t_impact: 打点時刻 [s]
        p0: 打点の位置 [m]
        v0: 打点時刻の速度 [m/s]
        t_land: 着地時刻 [s]
        plane: 飛行平面（正解）
        landing_xy: 着地点 (x, y) [m]
    """

    t_impact: float
    p0: np.ndarray
    v0: np.ndarray
    t_land: float
    plane: FlightPlane
    landing_xy: np.ndarray

    @property
    def speed0(self) -> float:
        return float(np.linalg.norm(self.v0))


def default_camera(cfg: dict[str, Any]) -> Camera:
    """設定ファイルの仮想カメラ（コート側方・高い位置）を作る。"""
    c = cfg["synthetic"]["camera"]
    return look_at(np.array(c["eye_m"]), np.array(c["target_m"]), c["f_px"], c["width"], c["height"])


def make_shot(speed0: float, cfg: dict[str, Any], phys: PhysicsParams,
              elevation_deg: float | None = None, t_impact: float = 0.0) -> SyntheticShot:
    """初速 ``speed0`` [m/s] の合成スマッシュを作る。"""
    s = cfg["synthetic"]["smash"]
    p0 = np.array(s["impact_m"], float)
    target = np.array(s["landing_xy_m"], float)
    d = np.r_[target - p0[:2], 0.0]
    d /= np.linalg.norm(d)
    el = np.deg2rad(s["elevation_deg"] if elevation_deg is None else elevation_deg)
    v0 = speed0 * (np.cos(el) * d + np.sin(el) * np.array([0.0, 0.0, 1.0]))
    traj = simulate_until_ground(p0, v0, t_impact, phys)
    land = traj["pos"][-1]
    plane = FlightPlane(origin=np.r_[p0[:2], 0.0], direction=d)
    return SyntheticShot(t_impact, p0, v0, float(traj["t_land"]), plane, land[:2])


def observe(shot: SyntheticShot, camera: Camera, fps: float, phys: PhysicsParams,
            rng: np.random.Generator, noise_px: float = 1.5, dropout: float = 0.1,
            exposure_s: float = 0.0, phase: float | None = None,
            n_sub: int = 9) -> pd.DataFrame:
    """合成スマッシュを fps [Hz] のカメラで観測した検出結果を作る。

    フレーム時刻は打点時刻に対して ``phase`` × フレーム間隔だけずれる（省略時は一様乱数）。
    打点以降・着地以前のフレームだけを出力する。

    Returns:
        列 ``frame, timestamp, x_px, y_px, visible, confidence, x_m, y_m, z_m``（後ろ3つは正解の3次元位置）
    """
    dt = 1.0 / fps
    if phase is None:
        phase = rng.uniform(0.0, 1.0)
    t_first = shot.t_impact + phase * dt
    times = np.arange(t_first, shot.t_land, dt)
    if exposure_s > 0:
        offs = np.linspace(-exposure_s / 2, exposure_s / 2, n_sub)
        tt = np.clip((times[:, None] + offs[None, :]).ravel(), shot.t_impact, shot.t_land)
        pos_sub, _ = simulate(shot.p0, shot.v0, shot.t_impact, tt, phys)
        uv_sub = camera.project(pos_sub).reshape(len(times), n_sub, 2)
        uv = uv_sub.mean(axis=1)
        pos = pos_sub.reshape(len(times), n_sub, 3).mean(axis=1)
    else:
        pos, _ = simulate(shot.p0, shot.v0, shot.t_impact, times, phys)
        uv = camera.project(pos)
    uv_noisy = uv + rng.normal(0.0, noise_px, uv.shape)
    inside = (
        (uv_noisy[:, 0] >= 0) & (uv_noisy[:, 0] < camera.width)
        & (uv_noisy[:, 1] >= 0) & (uv_noisy[:, 1] < camera.height)
    )
    visible = inside & (rng.uniform(size=len(times)) >= dropout)
    df = pd.DataFrame({
        "frame": np.arange(len(times)), "timestamp": times,
        "x_px": np.where(visible, uv_noisy[:, 0], np.nan),
        "y_px": np.where(visible, uv_noisy[:, 1], np.nan),
        "visible": visible.astype(int), "confidence": visible.astype(float),
        "x_m": pos[:, 0], "y_m": pos[:, 1], "z_m": pos[:, 2],
    })
    return df


def render_video(shot: SyntheticShot, camera: Camera, cfg: dict[str, Any], phys: PhysicsParams,
                 path: str | Path, fps: float = 240.0, pre_s: float = 0.2, post_s: float = 0.2,
                 radius_px: int = 4, exposure_s: float = 0.0, n_sub: int = 8,
                 audio_path: str | Path | None = None, audio_rate: int = 48000,
                 seed: int = 0, speed_of_sound: float | None = 343.0) -> pd.DataFrame:
    """白い背景にコートと小さな点（シャトル）が動く合成動画を書き出す。

    動画の時刻 0 は ``shot.t_impact - pre_s``。打点前のシャトルは描かない。
    ``audio_path`` を与えると、打点時刻にクリック音を入れた WAV（モノラル）も書き出す。
    クリック音は、打点からカメラまでの距離 / 音速 ``speed_of_sound`` [m/s] だけ遅れて入る（None なら遅れなし）。

    Returns:
        各フレームの正解（``frame, timestamp, x_px, y_px, visible``。timestamp は動画の時刻 [s]）
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = camera.width, camera.height
    base = np.full((h, w, 3), 255, np.uint8)
    draw_court(base, camera, cfg, color_line=(170, 170, 170), color_net=(120, 120, 120), thickness=2)
    t0 = shot.t_impact - pre_s
    n_frames = int(np.ceil((shot.t_land - t0 + post_s) * fps))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    disk = np.zeros((2 * radius_px + 1, 2 * radius_px + 1), np.float32)
    cv2.circle(disk, (radius_px, radius_px), radius_px, 1.0, -1, cv2.LINE_AA)
    rows = []
    for k in range(n_frames):
        t_vid = k / fps
        t = t0 + t_vid
        if exposure_s > 0:
            # 露光中の移動が途切れて見えないよう、像の移動 1 px ごとに1回描く（連続したブレ）
            ends, _ = simulate(shot.p0, shot.v0, shot.t_impact,
                               np.clip([t - exposure_s / 2, t + exposure_s / 2], shot.t_impact, shot.t_land), phys)
            span = np.linalg.norm(np.diff(camera.project(ends), axis=0))
            n = int(np.clip(np.nan_to_num(span), n_sub, 2000))
            ts = t + np.linspace(-exposure_s / 2, exposure_s / 2, n)
        else:
            ts = np.array([t])
        ts_in = ts[(ts >= shot.t_impact) & (ts <= shot.t_land)]
        frame = base.copy()
        uv_c = np.array([np.nan, np.nan])
        if len(ts_in):
            pos, _ = simulate(shot.p0, shot.v0, shot.t_impact, ts_in, phys)
            uvs = camera.project(pos)
            r, P = radius_px, 2 * radius_px + 2
            acc = np.zeros((h + 2 * P, w + 2 * P), np.float32)  # 端の余白 P つき
            for uv in uvs:
                if np.isfinite(uv).all():
                    x, y = int(round(uv[0])), int(round(uv[1]))
                    if -r <= x < w + r and -r <= y < h + r:
                        acc[y + P - r:y + P + r + 1, x + P - r:x + P + r + 1] += disk
            acc = acc[P:P + h, P:P + w]
            alpha = np.clip(acc / len(ts), 0, 1)[..., None]  # 露光時間のうち写っていた割合
            frame = (frame * (1 - alpha) + np.array([30, 30, 30]) * alpha).astype(np.uint8)
            uv_c = np.nanmean(uvs, axis=0)
        vis = bool(np.isfinite(uv_c).all() and 0 <= uv_c[0] < w and 0 <= uv_c[1] < h)
        rows.append({"frame": k, "timestamp": t_vid, "x_px": uv_c[0] if vis else np.nan,
                     "y_px": uv_c[1] if vis else np.nan, "visible": int(vis)})
        writer.write(frame)
    writer.release()

    if audio_path is not None:
        from scipy.io import wavfile
        rng = np.random.default_rng(seed)
        n = int(np.ceil(n_frames / fps * audio_rate))
        audio = rng.normal(0.0, 0.003, n)
        delay = 0.0 if speed_of_sound is None else float(np.linalg.norm(camera.center - shot.p0)) / speed_of_sound
        i0 = int(round((pre_s + delay) * audio_rate))
        click = np.exp(-np.arange(int(0.004 * audio_rate)) / (0.0006 * audio_rate))
        click = click * rng.normal(0.0, 1.0, len(click))
        audio[i0:i0 + len(click)] += 0.6 * click
        wavfile.write(str(audio_path), audio_rate, (np.clip(audio, -1, 1) * 32767).astype(np.int16))
    return pd.DataFrame(rows)
