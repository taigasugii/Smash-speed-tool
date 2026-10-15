"""モード B：カメラ2台による三角測量（高精度モード）。

2台の動画の時刻合わせは **打球音の音声波形の相互相関** で行う（手動の微調整 ``offset_s`` も可）。
軌道は2台の再投影誤差の和を最小にする形で、6自由度（位置・速度）を当てはめる（平面拘束は不要）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, correlate, sosfiltfilt

from smash_speed import events, report
from smash_speed.calibration.io import load_calibration
from smash_speed.config import save_config
from smash_speed.detection.baseline import BaselineParams, detect
from smash_speed.geometry import triangulate
from smash_speed.physics import FitOptions, Observations, PhysicsParams, fit_trajectory, speed_metrics
from smash_speed.repro import run_metadata
from smash_speed.video_io import extract_audio, load_wav, probe


def audio_offset(x1: np.ndarray, x2: np.ndarray, rate: int, max_offset_s: float = 3.0,
                 highpass_hz: float = 1000.0) -> tuple[float, float]:
    """2つの音声の時刻のずれ（x2 の時刻 − x1 の時刻）[s] を相互相関で求める。

    打球音のような鋭い音が効くように高域通過フィルタをかけた包絡線どうしで相関をとり、
    ピークを放物線補間してサンプルより細かく求める。

    Returns:
        (ずれ [s], 相関の鋭さ（ピーク / 2番目のピーク）)
    """
    sos = butter(4, highpass_hz, btype="highpass", fs=rate, output="sos")
    e1 = np.abs(sosfiltfilt(sos, x1))
    e2 = np.abs(sosfiltfilt(sos, x2))
    e1 = (e1 - e1.mean()) / (e1.std() + 1e-12)
    e2 = (e2 - e2.mean()) / (e2.std() + 1e-12)
    c = correlate(e2, e1, mode="full", method="fft")
    lags = np.arange(-len(e1) + 1, len(e2))
    keep = np.abs(lags) <= max_offset_s * rate
    c, lags = c[keep], lags[keep]
    i = int(np.argmax(c))
    frac = 0.0
    if 0 < i < len(c) - 1:
        a, b, d = c[i - 1], c[i], c[i + 1]
        den = a - 2 * b + d
        frac = 0.5 * (a - d) / den if den != 0 else 0.0
    guard = int(0.005 * rate)
    others = np.r_[c[:max(0, i - guard)], c[i + guard + 1:]]
    sharp = float(c[i] / (others.max() + 1e-12)) if len(others) else np.inf
    return float((lags[i] + frac) / rate), sharp


@dataclass
class StereoOptions:
    videos: list[Path]
    calibs: list[Path]
    out_dir: Path
    detections: list[Path] | None = None
    capture_fps: float | None = None
    start_s: float | None = None     # 1台目の時刻 [s]
    end_s: float | None = None
    offset_s: float | None = None    # 2台目の時刻 − 1台目の時刻 [s]（省略時は音声の相互相関）
    impact_time_s: float | None = None
    audio_paths: list[Path] | None = None


def analyze_stereo(o: StereoOptions, cfg: dict[str, Any]) -> dict[str, Any]:
    """2台のカメラから初速・平均速度を求める。時刻はすべて1台目の動画の時刻 [s]。"""
    out = Path(o.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_config({**cfg, "_run": run_metadata({"mode": "stereo"})}, out / "config_used.yaml")
    phys = PhysicsParams.from_config(cfg)
    metas = [probe(v, o.capture_fps, cfg) for v in o.videos]
    cams = [load_calibration(c)[0] for c in o.calibs]

    # --- 時刻合わせ
    wavs = list(o.audio_paths) if o.audio_paths else []
    if not wavs:
        for k, (v, m) in enumerate(zip(o.videos, metas)):
            if m.has_audio:
                wavs.append(extract_audio(v, out / f"audio{k + 1}.wav", cfg=cfg))
    sharp = None
    if o.offset_s is not None:
        offset = o.offset_s
    elif len(wavs) == 2:
        (x1, r1), (x2, r2) = load_wav(wavs[0]), load_wav(wavs[1])
        if r1 != r2:
            raise ValueError("2つの音声のサンプリング周波数が違います")
        offset, sharp = audio_offset(x1, x2, r1)
        offset *= metas[0].time_scale
    else:
        raise ValueError("音声がないため時刻を合わせられません。--offset で指定してください。")

    # --- 検出（2台目は時刻をずらした区間）
    dets = []
    for k, m in enumerate(metas):
        sh = 0.0 if k == 0 else offset
        if o.detections:
            d = pd.read_csv(o.detections[k])
        else:
            s0 = None if o.start_s is None else o.start_s + sh
            s1 = None if o.end_s is None else o.end_s + sh
            d = detect(m, s0, s1, BaselineParams.from_config(cfg), n_tracks=1)
        d = d.copy()
        d["timestamp_cam1"] = d["timestamp"] - sh
        d.to_csv(out / f"detections{k + 1}.csv", index=False)
        dets.append(d[d.visible == 1])
    t_first = min(float(d.timestamp_cam1.min()) for d in dets)

    # --- 打点時刻（1台目の音声 → なければ最初の検出）
    impact = None
    if o.impact_time_s is not None:
        impact = events.ImpactEstimate(o.impact_time_s, cfg["events"]["manual_sigma_s"], "manual")
    c_sound = cfg["events"].get("speed_of_sound_mps", 343.0)
    t_imp_raw = None
    if o.impact_time_s is None and wavs:
        # 最初の検出を三角測量して、1台目のカメラから打点付近までの距離（音の遅れ）を見積もる
        a0, b0 = dets[0].iloc[:3], dets[1]
        j = np.abs(b0.timestamp_cam1.to_numpy()[:, None] - a0.timestamp_cam1.to_numpy()[None, :]).argmin(axis=0)
        X0 = triangulate(cams, [a0[["x_px", "y_px"]].to_numpy(), b0[["x_px", "y_px"]].to_numpy()[j]])
        X0 = X0[np.isfinite(X0).all(axis=1)]
        d1_0 = float(np.linalg.norm(cams[0].center - X0[0])) if len(X0) else float(np.linalg.norm(cams[0].center))
        x, rate = load_wav(wavs[0])
        ons = [(t * metas[0].time_scale, s) for t, s in events.audio_onsets(
            x, rate, cfg["events"]["audio_highpass_hz"], k_mad=cfg["events"]["audio_k_mad"])]
        impact = events.impact_from_audio(ons, t_first, sigma_s=cfg["physics"]["impact_time_sigma_s"],
                                          av_offset_s=-d1_0 / c_sound)
        if impact is not None:
            t_imp_raw = impact.t + d1_0 / c_sound
    if impact is None:
        impact = events.impact_from_first_frame(t_first, 1.0 / metas[0].capture_fps)

    opts = FitOptions.from_config(cfg, t_ref_sigma=impact.sigma)
    audio_sync = o.offset_s is None
    audio_impact = t_imp_raw is not None
    off_raw = offset

    def build_obs(off: float) -> list[Observations]:
        out_obs = []
        for k, (c, d) in enumerate(zip(cams, dets)):
            t = d.timestamp.to_numpy() - (0.0 if k == 0 else off)
            out_obs.append(Observations(c, t, d[["x_px", "y_px"]].to_numpy()))
        return [ob.subset(ob.t >= impact.t - 1e-9) for ob in out_obs]

    obs = build_obs(offset)
    fit = fit_trajectory(obs, impact.t, "free", phys=phys, opts=opts)
    # 打球音の伝搬遅れ：各カメラまでの距離 / 音速。音声の相互相関によるずれと、1台目の音声の打点を補正する
    for _ in range(2):
        d1, d2 = (float(np.linalg.norm(c.center - fit.p0)) for c in cams)
        new_off = off_raw - (d2 - d1) / c_sound if audio_sync else offset
        new_t = t_imp_raw - d1 / c_sound if audio_impact else impact.t
        if abs(new_off - offset) < 1e-5 and abs(new_t - impact.t) < 1e-5:
            break
        offset = new_off
        impact = events.ImpactEstimate(new_t, impact.sigma, impact.method)
        obs = build_obs(offset)
        fit = fit_trajectory(obs, impact.t, "free", phys=phys, opts=opts)
    for k, d in enumerate(dets):
        d["timestamp_cam1"] = d["timestamp"] - (0.0 if k == 0 else offset)
    metrics = speed_metrics(fit, phys)

    # 参考：同時刻の三角測量点（フレーム間差分用）
    t1 = obs[0].t
    idx = np.clip(np.searchsorted(obs[1].t, t1), 1, max(len(obs[1].t) - 1, 1))
    near = np.where(np.abs(obs[1].t[idx - 1] - t1) < np.abs(obs[1].t[np.minimum(idx, len(obs[1].t) - 1)] - t1),
                    idx - 1, np.minimum(idx, len(obs[1].t) - 1))
    ok = np.abs(obs[1].t[near] - t1) < 0.5 / metas[1].capture_fps
    X = triangulate(cams, [obs[0].uv[ok], obs[1].uv[near[ok]]])
    from smash_speed.physics import finite_difference_speeds
    naive = finite_difference_speeds(t1[ok], X)

    (out / "fit_result.json").write_text(json.dumps({**fit.to_dict(), "sync_offset_s": offset,
                                                     "sync_sharpness": sharp, "impact": impact.to_dict()},
                                                    ensure_ascii=False, indent=1), encoding="utf-8")
    row = report.result_row(Path(o.videos[0]).stem, fit, metrics, naive,
                            {"sync_offset_s": offset, "impact_method": impact.method,
                             "impact_sigma_s": impact.sigma, "capture_fps": metas[0].capture_fps})
    report.write_rows([row], out / "result.csv")
    report.plot_speed_time(fit, phys, naive, out / "speed_time.png", t1)
    report.plot_trajectory_views(fit, phys, X, cfg, out / "trajectory.png")
    return row
