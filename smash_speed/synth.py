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
        delay = 0.0 if speed_of_sound is None else float(np.linalg.norm(camera.center - shot.p0)) / speed_of_sound
        write_click_audio(audio_path, n_frames / fps, [(pre_s + delay, 0.6)], audio_rate, seed)
    return pd.DataFrame(rows)


def write_click_audio(path: str | Path, duration_s: float, clicks: list[tuple[float, float]],
                      rate: int = 48000, seed: int = 0) -> None:
    """雑音の上に、時刻 [s]・振幅のクリック音（打球音の代わり）を入れたモノラル WAV を書き出す。"""
    from scipy.io import wavfile
    rng = np.random.default_rng(seed)
    audio = rng.normal(0.0, 0.003, int(np.ceil(duration_s * rate)))
    for t, amp in clicks:
        i0 = int(round(t * rate))
        click = np.exp(-np.arange(int(0.004 * rate)) / (0.0006 * rate)) * rng.normal(0.0, 1.0, int(0.004 * rate))
        n = max(0, min(len(click), len(audio) - i0))
        audio[i0:i0 + n] += amp * click[:n]
    wavfile.write(str(path), rate, (np.clip(audio, -1, 1) * 32767).astype(np.int16))


def render_cluttered_video(shot: SyntheticShot, camera: Camera, cfg: dict[str, Any], phys: PhysicsParams,
                           path: str | Path, fps: float = 60.0, exposure_s: float | None = None,
                           pre_s: float = 0.3, post_s: float = 0.2, n_sub: int = 8,
                           shuttle_radius_m: float = 0.033, noise: float = 2.0,
                           audio_path: str | Path | None = None, seed: int = 0,
                           speed_of_sound: float | None = 343.0,
                           walkers: list[tuple[tuple[float, float], tuple[float, float]]] | None = None
                           ) -> pd.DataFrame:
    """実際の試合に近い、検出の難しい合成動画を書き出す（検出・軌道の選び方の検証用）。

    ``render_video`` の白い背景と点だけの動画と違い、次のものを描く。

    - 模様のある床とコートのライン、センサーの雑音
    - 打つ人の体と、打点を通って振り抜く腕・ラケット（打った瞬間はシャトルと重なる）
    - 打つ前に飛んでくるシャトル、相手の選手、隣のコートを歩く人
    - 露光中のブレ（シャトルも人も ``n_sub`` 回に分けて描いて平均する）

    Args:
        exposure_s: 露光時間 [s]。省略時はフレーム間隔の半分（屋内のスマホ動画に近い）
        walkers: コートの外を歩く人の（足元の位置 (x, y) [m]、速度 (vx, vy) [m/s]）。省略時は隣のコート側に 1 人

    Returns:
        各フレームの正解（``frame, timestamp, x_px, y_px, visible``。timestamp は動画の時刻 [s]）
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    exposure_s = 0.5 / fps if exposure_s is None else exposure_s
    w, h = camera.width, camera.height
    tex = cv2.GaussianBlur(rng.normal(0.0, 14.0, (h, w)).astype(np.float32), (0, 0), 3)
    base = np.clip(np.full((h, w, 3), (96.0, 112.0, 104.0), np.float32) + tex[..., None], 0, 255).astype(np.uint8)
    draw_court(base, camera, cfg, color_line=(205, 205, 205), color_net=(40, 40, 40), thickness=3)

    d = shot.plane.direction
    nrm = shot.plane.normal
    ez = np.array([0.0, 0.0, 1.0])
    feet = shot.p0 * np.array([1, 1, 0]) - cfg["events"]["contact_offset_m"] * d
    shoulder = feet + 1.45 * ez + 0.2 * nrm
    rel = shot.p0 - shoulder
    reach = float(np.hypot(rel @ d, rel[2]))
    phi0 = float(np.arctan2(rel[2], rel @ d))
    side = float(rel @ nrm)
    receiver0 = np.r_[shot.landing_xy, 0.0] + 0.8 * d
    walkers = [((-5.0, 4.6), (1.4, 0.0))] if walkers is None else walkers

    def px_radius(X: np.ndarray, r_m: float) -> float:
        dist = float(np.linalg.norm(X - camera.center))
        return max(1.0, camera.f * r_m / max(dist, 1e-3))

    def draw_body(img: np.ndarray, foot: np.ndarray, height: float, color: tuple[int, ...]) -> None:
        uv = camera.project(np.array([foot + 0.2 * ez, foot + height * ez]))
        if not np.isfinite(uv).all():
            return
        r = px_radius(foot + 1.0 * ez, 0.22)
        a, b = uv.round().astype(int)
        cv2.line(img, tuple(a), tuple(b), color, int(round(2 * r)), cv2.LINE_AA)
        head = camera.project((foot + (height + 0.12) * ez)[None])[0]
        cv2.circle(img, tuple(head.round().astype(int)), int(round(px_radius(foot + height * ez, 0.12))),
                   (120, 140, 175), -1, cv2.LINE_AA)

    def racket_at(t: float) -> tuple[np.ndarray, np.ndarray]:
        """時刻 t の手首と、ラケット面の中心 [m]（打点時刻に面の中心がシャトルと重なる）。"""
        phi = phi0 - 1.8 * np.tanh((t - shot.t_impact) / 0.07)
        u = np.cos(phi) * d + np.sin(phi) * ez
        head = shoulder + reach * u + side * nrm
        wrist = shoulder + 0.55 * reach * u + side * nrm
        return wrist, head

    def draw_scene(img: np.ndarray, t: float) -> None:
        sway = 0.08 * np.sin(2 * np.pi * 1.3 * t)
        draw_body(img, feet + sway * nrm, 1.55, (55, 45, 150))
        wrist, head = racket_at(t)
        uv = camera.project(np.array([shoulder, wrist, head]))
        if np.isfinite(uv).all():
            s_, w_, h_ = uv.round().astype(int)
            cv2.line(img, tuple(s_), tuple(w_), (130, 150, 190), int(round(2 * px_radius(wrist, 0.05))), cv2.LINE_AA)
            cv2.line(img, tuple(w_), tuple(h_), (40, 40, 40), max(2, int(round(2 * px_radius(head, 0.012)))), cv2.LINE_AA)
            rr = int(round(px_radius(head, 0.13)))
            cv2.circle(img, tuple(h_), rr, (180, 180, 170), -1, cv2.LINE_AA)
            cv2.circle(img, tuple(h_), rr, (30, 30, 30), max(1, rr // 6), cv2.LINE_AA)
        rec = receiver0 + 0.9 * np.sin(2 * np.pi * 0.8 * t) * nrm
        draw_body(img, rec, 1.6, (150, 60, 40))
        for (x0, y0), (vx, vy) in walkers:
            draw_body(img, np.array([x0 + vx * t, y0 + vy * t, 0.0]), 1.65, (60, 90, 60))

    def shuttle_pos(ts: np.ndarray) -> np.ndarray:
        """打つ前は上から落ちてくるシャトル、打った後は運動方程式の軌道 [m]（着地後は NaN）。"""
        out = np.full((len(ts), 3), np.nan)
        after = (ts >= shot.t_impact) & (ts <= shot.t_land)
        if after.any():
            out[after] = simulate(shot.p0, shot.v0, shot.t_impact, ts[after], phys)[0]
        before = ts < shot.t_impact
        out[before] = shot.p0 + (shot.t_impact - ts[before])[:, None] * (4.0 * d + 7.0 * ez)
        return out

    t0 = shot.t_impact - pre_s
    n_frames = int(np.ceil((shot.t_land - t0 + post_s) * fps))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    rows = []
    for k in range(n_frames):
        t = t0 + k / fps
        ts = t + np.linspace(-exposure_s / 2, exposure_s / 2, n_sub)
        dense = t + np.linspace(-exposure_s / 2, exposure_s / 2, 8 * n_sub)
        pos = shuttle_pos(dense)
        uv_s = camera.project(pos)
        acc = np.zeros((h, w, 3), np.float32)
        for j, tj in enumerate(ts):
            img = base.copy()
            draw_scene(img, tj)
            seg = uv_s[8 * j:8 * j + 8]
            ok = np.isfinite(seg).all(axis=1)
            if ok.sum() >= 1:
                X = pos[8 * j:8 * j + 8][ok].mean(axis=0)
                r = int(round(px_radius(X, shuttle_radius_m)))
                pts = seg[ok].round().astype(np.int32)
                if len(pts) >= 2:
                    cv2.polylines(img, [pts], False, (245, 245, 245), 2 * r + 1, cv2.LINE_AA)
                else:
                    cv2.circle(img, tuple(pts[0]), r, (245, 245, 245), -1, cv2.LINE_AA)
            acc += img
        frame = acc / n_sub + rng.normal(0.0, noise, (h, w, 1)).astype(np.float32)
        writer.write(np.clip(frame, 0, 255).astype(np.uint8))
        after = (dense >= shot.t_impact) & np.isfinite(uv_s).all(axis=1)
        uv_c = uv_s[after].mean(axis=0) if after.any() else np.array([np.nan, np.nan])
        vis = bool(np.isfinite(uv_c).all() and 0 <= uv_c[0] < w and 0 <= uv_c[1] < h)
        rows.append({"frame": k, "timestamp": k / fps, "x_px": uv_c[0] if vis else np.nan,
                     "y_px": uv_c[1] if vis else np.nan, "visible": int(vis)})
    writer.release()

    if audio_path is not None:
        delay = 0.0 if speed_of_sound is None else float(np.linalg.norm(camera.center - shot.p0)) / speed_of_sound
        # 打球音のほかに、弱い足音を 2 つ入れる
        write_click_audio(audio_path, n_frames / fps, [(pre_s + delay, 0.6), (pre_s - 0.22, 0.12),
                                                       (pre_s + 0.35, 0.1)], seed=seed)
    return pd.DataFrame(rows)
