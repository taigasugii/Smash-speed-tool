"""出力：速度指標の CSV、グラフ、オーバーレイ動画、複数本の集計表。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from smash_speed.camera import Camera  # noqa: E402
from smash_speed.court import court_segments, draw_court  # noqa: E402
from smash_speed.physics import (  # noqa: E402
    MPS_TO_KMH, FitResult, PhysicsParams, simulate_until_ground,
)

plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def result_row(name: str, fit: FitResult, metrics: dict[str, float], naive: dict[str, Any],
               extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """1本分の結果を CSV の1行にまとめる。"""
    lo, hi = fit.speed0_ci if fit.speed0_ci else (np.nan, np.nan)
    row: dict[str, Any] = {"shot": name, **metrics,
                           "speed_initial_ci_low_mps": lo, "speed_initial_ci_high_mps": hi,
                           "speed_initial_ci_low_kmh": lo * MPS_TO_KMH,
                           "speed_initial_ci_high_kmh": hi * MPS_TO_KMH,
                           "ci_method": fit.ci_method, "ci_sd_fit_mps": fit.speed0_sd_fit,
                           "ci_sd_impact_timing_mps": fit.speed0_sd_timing,
                           "naive_first_interval_mps": naive.get("first_mps", np.nan),
                           "naive_first_interval_kmh": naive.get("first_mps", np.nan) * MPS_TO_KMH,
                           "naive_mean_mps": naive.get("mean_mps", np.nan),
                           "n_frames_used": fit.n_obs, "rms_px": fit.rms_px, "mode": fit.mode,
                           "terminal_velocity_mps": fit.terminal_velocity, "t_impact_s": fit.t_ref}
    if extra:
        row.update(extra)
    return row


def write_rows(rows: list[dict[str, Any]], path: str | Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def summarize(result_csvs: list[Path], out: str | Path) -> pd.DataFrame:
    """複数本の ``result.csv`` を1つの集計表にまとめ、平均・標準偏差の行を加える。"""
    df = pd.concat([pd.read_csv(p) for p in result_csvs], ignore_index=True)
    num = df.select_dtypes("number")
    stats = pd.DataFrame([num.mean().rename("mean"), num.std().rename("std")])
    stats.insert(0, "shot", stats.index)
    out_df = pd.concat([df, stats], ignore_index=True)
    out_df.to_csv(out, index=False, encoding="utf-8-sig")
    return out_df


def plot_speed_time(fit: FitResult, phys: PhysicsParams, naive: dict[str, Any], path: str | Path,
                    t_obs: np.ndarray | None = None) -> None:
    """時刻と速さの関係（物理モデル）と、フレーム間差分の速さを重ねて描く。"""
    traj = simulate_until_ground(fit.p0, fit.v0, fit.t_ref, fit.phys(phys))
    sp = np.linalg.norm(traj["vel"], axis=1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot((traj["t"] - fit.t_ref) * 1000, sp * MPS_TO_KMH, label="物理モデル")
    if len(naive.get("speed", [])):
        ax.plot((naive["t_mid"] - fit.t_ref) * 1000, naive["speed"] * MPS_TO_KMH, "o",
                ms=4, label="フレーム間差分")
    if fit.speed0_ci:
        lo, hi = fit.speed0_ci
        ax.errorbar([0], [fit.speed0 * MPS_TO_KMH], yerr=[[(fit.speed0 - lo) * MPS_TO_KMH],
                    [(hi - fit.speed0) * MPS_TO_KMH]], fmt="s", color="C3", capsize=4,
                    label=f"初速 {fit.speed0 * MPS_TO_KMH:.0f} km/h（95%CI）")
    if t_obs is not None:
        for t in t_obs:
            ax.axvline((t - fit.t_ref) * 1000, color="0.85", lw=0.6, zorder=0)
    ax.set_xlabel("打点からの時間 [ms]")
    ax.set_ylabel("速さ [km/h]")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_trajectory_views(fit: FitResult, phys: PhysicsParams, X_obs: np.ndarray, cfg: dict[str, Any],
                          path: str | Path) -> None:
    """3次元軌道の側面図（x–z）と上面図（x–y）。観測点は平面との交点。"""
    traj = simulate_until_ground(fit.p0, fit.v0, fit.t_ref, fit.phys(phys))
    P = traj["pos"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6.5))
    c = cfg["court"]
    a1.plot(P[:, 0], P[:, 2], label="当てはめた軌道")
    if len(X_obs):
        a1.plot(X_obs[:, 0], X_obs[:, 2], ".", label="観測（平面との交点）")
    a1.plot([0, 0], [0, c["net_height_center_m"]], "k-", lw=3, label="ネット")
    a1.axhline(0, color="k", lw=0.8)
    a1.set_xlim(-c["half_length_m"] - 0.3, c["half_length_m"] + 0.3)
    a1.set_ylim(-0.1, max(3.5, P[:, 2].max() + 0.3))
    a1.set_xlabel("x [m]")
    a1.set_ylabel("z [m]")
    a1.set_title("側面図")
    a1.legend(fontsize=8)
    a1.set_aspect("equal")
    for a, b, kind in court_segments(cfg):
        if kind == "line":
            a2.plot([a[0], b[0]], [a[1], b[1]], color="0.6", lw=0.8)
    a2.plot([0, 0], [-c["half_width_doubles_m"], c["half_width_doubles_m"]], "k-", lw=2)
    a2.plot(P[:, 0], P[:, 1])
    if len(X_obs):
        a2.plot(X_obs[:, 0], X_obs[:, 1], ".")
    a2.plot(*P[-1, :2], "x", color="C3", ms=8)
    a2.set_aspect("equal")
    a2.set_xlabel("x [m]")
    a2.set_ylabel("y [m]")
    a2.set_title("上面図")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_image_fit(t_obs: np.ndarray, uv_obs: np.ndarray, fit: FitResult, phys: PhysicsParams,
                   camera: Camera, path: str | Path, background: np.ndarray | None = None) -> None:
    """画像上の観測点と、当てはめた軌道の投影を比べる。"""
    traj = simulate_until_ground(fit.p0, fit.v0, fit.t_ref, fit.phys(phys))
    curve = camera.project(traj["pos"])
    pred, _ = fit.evaluate(t_obs, phys)
    uv_pred = camera.project(pred)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    if background is not None:
        ax.imshow(background[:, :, ::-1])
    ax.plot(curve[:, 0], curve[:, 1], "-", color="C1", lw=1.2, label="当てはめた軌道")
    ax.plot(uv_obs[:, 0], uv_obs[:, 1], "o", mfc="none", color="C0", ms=6, label="検出")
    ax.plot(uv_pred[:, 0], uv_pred[:, 1], "+", color="C3", ms=7, label="同時刻の予測")
    # 軌道の投影はカメラの近く・後ろで画面の外へ大きく飛ぶので、表示は画像の範囲に限る
    ax.set_xlim(0, camera.width)
    ax.set_ylim(camera.height, 0)
    ax.set_aspect("equal")
    ax.set_title(f"再投影誤差 RMS {fit.rms_px:.2f} px")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def render_overlay(reader, start_s: float, end_s: float, detections: pd.DataFrame, fit: FitResult | None,
                   phys: PhysicsParams, camera: Camera, cfg: dict[str, Any], path: str | Path,
                   metrics: dict[str, float] | None = None, out_width: int = 1920,
                   draw_court_lines: bool = True, fps_out: float | None = None) -> None:
    """元の動画に、検出位置・当てはめた軌道・初速を描き込んだ動画を書き出す。

    Args:
        reader: ``VideoReader``（scale = 1 で開いたもの）
        start_s, end_s: 区間（実時間）[s]
        detections: 検出結果（元解像度の px）
        fit: フィッティング結果（None なら検出位置だけを描く）
        out_width: 出力動画の横幅 [px]
        fps_out: 出力の fps（省略時は撮影 fps。スロー動画は実時間より遅く再生される）
    """
    meta = reader.meta
    s = out_width / meta.width
    size = (out_width, int(round(meta.height * s)))
    fps_out = fps_out or min(meta.capture_fps, 60.0)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps_out, size)
    cam_s = camera.scaled(s)
    curve = None
    if fit is not None:
        traj = simulate_until_ground(fit.p0, fit.v0, fit.t_ref, fit.phys(phys))
        curve_t = traj["t"]
        curve = cam_s.project(traj["pos"])
    det = detections.set_index("frame")
    trail: list[tuple[int, int]] = []
    court_layer = None
    for i, t, img in reader.iter_range(start_s, end_s):
        frame = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        if draw_court_lines:
            if court_layer is None:
                court_layer = np.zeros_like(frame)
                draw_court(court_layer, cam_s, cfg, color_line=(0, 200, 255), color_net=(200, 0, 200), thickness=1)
                court_mask = court_layer.any(axis=2)
            frame[court_mask] = (0.5 * frame[court_mask] + 0.5 * court_layer[court_mask]).astype(np.uint8)
        if curve is not None:
            ok = np.isfinite(curve).all(axis=1)
            cv2.polylines(frame, [curve[ok].round().astype(np.int32)], False, (0, 165, 255), 2, cv2.LINE_AA)
            if fit.t_ref <= t <= curve_t[-1]:
                p, _ = fit.evaluate(np.array([t]), phys)
                q = cam_s.project(p)[0]
                if np.isfinite(q).all():
                    cv2.circle(frame, tuple(q.round().astype(int)), 9, (0, 165, 255), 2, cv2.LINE_AA)
        if i in det.index and det.at[i, "visible"] == 1:
            q = (int(round(det.at[i, "x_px"] * s)), int(round(det.at[i, "y_px"] * s)))
            trail.append(q)
        for k, q in enumerate(trail[-30:]):
            cv2.circle(frame, q, 4, (0, 255, 0), -1 if k == len(trail[-30:]) - 1 else 1, cv2.LINE_AA)
        lines = [f"t = {t:.3f} s"]
        if fit is not None and t >= fit.t_ref:
            v = fit.speed0 * MPS_TO_KMH
            ci = ""
            if fit.speed0_ci:
                ci = f"  (95% CI {fit.speed0_ci[0] * MPS_TO_KMH:.0f}-{fit.speed0_ci[1] * MPS_TO_KMH:.0f})"
            lines.append(f"Initial speed {v:.0f} km/h = {fit.speed0:.1f} m/s{ci}")
            if metrics:
                lines.append(f"Mean speed {metrics['speed_mean_path_kmh']:.0f} km/h   "
                             f"Arrival {metrics['speed_arrival_kmh']:.0f} km/h")
        for k, text in enumerate(lines):
            y = 32 + 30 * k
            cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    make_web_video(Path(path))


def make_web_video(src: Path) -> Path | None:
    """ブラウザや Windows の標準プレーヤーで再生できる H.264 版（``<名前>_web.mp4``）を ffmpeg で作る。"""
    import subprocess

    from smash_speed.video_io import find_tool
    dst = src.with_name(src.stem + "_web.mp4")
    try:
        subprocess.run([find_tool("ffmpeg"), "-v", "error", "-y", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-preset", "fast", "-crf", "20", "-movflags", "+faststart", str(dst)], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return dst
