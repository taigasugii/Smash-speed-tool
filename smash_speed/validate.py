"""精度検証：fps 別・手法別に初速の推定誤差を比べる（研究の中核）。

2種類の検証を用意する。

- ``synthetic_study``: 正解がわかっている合成データで、fps・速度の算出方法・打点時刻の推定方法・
  飛行平面の誤差を変えたときの誤差を調べる（モンテカルロ）
- ``dataset_study``: 240fps の実動画から求めた初速を「正解」とし、同じ動画から作った
  120 / 60 / 30fps 版での推定値と比べる。スピードガンなどの外部計測値とも比べられる

評価指標は平均絶対誤差（MAE）、平均絶対パーセント誤差（MAPE）、平均誤差（バイアス）、
Bland–Altman の一致限界（バイアス ± 1.96 SD）。系統的なずれは線形回帰で補正できる。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from smash_speed import report  # noqa: F401,E402  （日本語フォントの設定）
from smash_speed import synth  # noqa: E402
from smash_speed.calibration import calibrate, landmark_table, save_calibration  # noqa: E402
from smash_speed.events import impact_from_first_frame, impact_from_hitter_position  # noqa: E402
from smash_speed.geometry import FlightPlane, intersect_plane  # noqa: E402
from smash_speed.physics import (  # noqa: E402
    MPS_TO_KMH, FitOptions, Observations, PhysicsParams, finite_difference_speeds, fit_trajectory,
)


# =============================================================================
# 指標
# =============================================================================
def error_stats(ref: np.ndarray, est: np.ndarray) -> dict[str, float]:
    """推定値と正解の比較指標（単位は入力と同じ）。"""
    ref, est = np.asarray(ref, float), np.asarray(est, float)
    ok = np.isfinite(ref) & np.isfinite(est)
    ref, est = ref[ok], est[ok]
    if len(ref) == 0:
        return {"n": 0, "mae": np.nan, "mape": np.nan, "bias": np.nan, "sd": np.nan,
                "loa_low": np.nan, "loa_high": np.nan, "rmse": np.nan}
    d = est - ref
    sd = float(np.std(d, ddof=1)) if len(d) > 1 else np.nan
    return {"n": int(len(d)), "mae": float(np.mean(np.abs(d))), "mape": float(np.mean(np.abs(d) / ref) * 100),
            "bias": float(np.mean(d)), "sd": sd, "loa_low": float(np.mean(d) - 1.96 * sd),
            "loa_high": float(np.mean(d) + 1.96 * sd), "rmse": float(np.sqrt(np.mean(d**2)))}


def bland_altman(ax, ref: np.ndarray, est: np.ndarray, title: str, unit: str = "km/h") -> None:
    """Bland–Altman プロット（横軸：平均、縦軸：推定 − 正解）。"""
    ref, est = np.asarray(ref, float), np.asarray(est, float)
    ok = np.isfinite(ref) & np.isfinite(est)
    m, d = (ref[ok] + est[ok]) / 2, est[ok] - ref[ok]
    s = error_stats(ref, est)
    ax.scatter(m, d, s=10, alpha=0.6)
    for y, ls in ((s["bias"], "-"), (s["loa_low"], "--"), (s["loa_high"], "--")):
        if np.isfinite(y):
            ax.axhline(y, color="C3", ls=ls, lw=1)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"{title}\nbias {s['bias']:+.1f}, LoA [{s['loa_low']:+.1f}, {s['loa_high']:+.1f}] {unit}",
                 fontsize=9)
    ax.set_xlabel(f"平均 [{unit}]")
    ax.set_ylabel(f"推定 − 正解 [{unit}]")


def fit_linear_correction(est: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """系統的なずれの補正モデル ref ≈ a × est + b を最小二乗で求める。"""
    ok = np.isfinite(est) & np.isfinite(ref)
    a, b = np.polyfit(est[ok], ref[ok], 1)
    return {"a": float(a), "b": float(b)}


def apply_correction(est: np.ndarray, corr: dict[str, float]) -> np.ndarray:
    return corr["a"] * np.asarray(est) + corr["b"]


def _summarize_and_plot(df: pd.DataFrame, group_cols: list[str], out: Path, ref_col: str,
                        est_col: str, title: str) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group_cols, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(group_cols, key)), **error_stats(g[ref_col], g[est_col])})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")

    # 誤差（MAPE）と fps の関係
    others = [c for c in group_cols if c != "fps"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for key, g in summary.groupby(others, sort=True) if others else [((), summary)]:
        key = key if isinstance(key, tuple) else (key,)
        ax.plot(g["fps"], g["mape"], "o-", label=" / ".join(map(str, key)))
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(df["fps"].unique()))
    ax.set_xticklabels([f"{f:g}" for f in sorted(df["fps"].unique())])
    ax.set_xlabel("fps")
    ax.set_ylabel("初速の MAPE [%]")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "mape_vs_fps.png", dpi=150)
    plt.close(fig)

    # Bland–Altman（条件ごと）
    groups = list(df.groupby(group_cols, sort=True))
    ncol = len(df["fps"].unique())
    nrow = int(np.ceil(len(groups) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.0 * nrow), squeeze=False)
    for ax, (key, g) in zip(axes.ravel(), groups):
        key = key if isinstance(key, tuple) else (key,)
        bland_altman(ax, g[ref_col], g[est_col], " / ".join(map(str, key)))
    for ax in axes.ravel()[len(groups):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "bland_altman.png", dpi=130)
    plt.close(fig)
    return summary


# =============================================================================
# 合成データによる検証
# =============================================================================
def synthetic_study(cfg: dict[str, Any], out: Path, n_shots: int = 100,
                    fps_list: list[float] = (30, 60, 120, 240), seed: int = 0,
                    plane_error_deg: float = 2.0, plane_error_m: float = 0.2) -> pd.DataFrame:
    """合成データで fps 別・手法別の初速の誤差を調べる。

    条件:
        - 速度の算出方法: physics（物理モデル）/ naive（最初のフレーム間差分）
        - 打点時刻: audio（真値 + 2 ms の誤差。伝搬遅れは補正済みとみなす）/ first_frame（最初の検出の半フレーム前）/
          hitter_position（軌道が打点を通過する時刻。打点の位置に σ = ``events.contact_sigma_m`` の誤差）
        - 飛行平面: クリック相当の誤差（向き ``plane_error_deg``°、位置 ``plane_error_m`` m の正規誤差）を与え、
          ``refine_plane`` で補正する

    Returns:
        1本・1条件ごとの結果の表（``results.csv`` にも保存）
    """
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    phys = PhysicsParams.from_config(cfg)
    cam = synth.default_camera(cfg)
    s = cfg["synthetic"]
    lo_v, hi_v = s["smash"]["speed_range_mps"]
    sigma_audio = cfg["physics"]["impact_time_sigma_s"]
    contact_sigma = cfg["events"]["contact_sigma_m"]
    rows = []
    for i in range(n_shots):
        v = rng.uniform(lo_v, hi_v)
        el = rng.uniform(-20.0, -5.0)
        shot = synth.make_shot(v, cfg, phys, elevation_deg=el)
        d = shot.plane.direction
        a = np.deg2rad(rng.normal(0, plane_error_deg))
        rot = np.array([d[0] * np.cos(a) - d[1] * np.sin(a), d[0] * np.sin(a) + d[1] * np.cos(a), 0.0])
        plane = FlightPlane(shot.plane.origin + rng.normal(0, plane_error_m) * shot.plane.normal, rot)
        audio_err = rng.normal(0, sigma_audio)
        for fps in fps_list:
            df = synth.observe(shot, cam, fps, phys, rng, noise_px=s["noise_px"], dropout=s["dropout"],
                               exposure_s=min(s["exposure_s"], 1 / fps))
            dv = df[df.visible == 1]
            if len(dv) < 4:
                continue
            obs = Observations(cam, dv.timestamp.to_numpy(), dv[["x_px", "y_px"]].to_numpy())
            t_first = float(dv.timestamp.iloc[0])
            ff = impact_from_first_frame(t_first, 1 / fps)
            impacts = {"audio": (shot.t_impact + audio_err, sigma_audio), "first_frame": (ff.t, ff.sigma),
                       "hitter_position": None}
            fits = {}
            for imp, spec in impacts.items():
                try:
                    if imp == "hitter_position":
                        # 最初の検出から当てはめた軌道を逆にたどり、打点（位置に σ = contact_sigma の誤差）を通る時刻
                        f0 = fits["first_frame"]
                        hp_plane = FlightPlane(f0.plane.origin + rng.normal(0, contact_sigma) * f0.plane.direction,
                                               f0.plane.direction)
                        t_sing = f0.t_ref - 1.0 / (f0.phys(phys).k * f0.speed0)
                        lo = max(t_first - 0.1, f0.t_ref - 0.7 * (f0.t_ref - t_sing))
                        hp = impact_from_hitter_position(lambda t: f0.evaluate(np.array([t]), phys)[0][0], hp_plane,
                                                         lo, t_first + 0.5 / fps, 0.0, contact_sigma)
                        if hp is None:
                            raise ValueError("hitter_position")
                        spec = (hp.t, hp.sigma)
                    t_ref, sig = spec
                    fit = fit_trajectory(obs, t_ref, "plane", plane, phys,
                                         FitOptions(refine_plane=True, t_ref_sigma=sig, ci_method="jacobian"))
                    fits[imp] = fit
                    est = fit.speed0
                    ci = fit.speed0_ci or (np.nan, np.nan)
                except (ValueError, RuntimeError, KeyError):
                    est, ci = np.nan, (np.nan, np.nan)
                rows.append({"shot": i, "fps": fps, "method": "physics", "impact": imp, "true_mps": v,
                             "est_mps": est, "ci_low": ci[0], "ci_high": ci[1], "n_obs": len(dv)})
            X = intersect_plane(cam, obs.uv, plane)
            fd = finite_difference_speeds(obs.t, X)
            rows.append({"shot": i, "fps": fps, "method": "naive", "impact": "-", "true_mps": v,
                         "est_mps": fd["first_mps"], "ci_low": np.nan, "ci_high": np.nan, "n_obs": len(dv)})
    df = pd.DataFrame(rows)
    df["true_kmh"] = df["true_mps"] * MPS_TO_KMH
    df["est_kmh"] = df["est_mps"] * MPS_TO_KMH
    df["covered"] = (df["ci_low"] <= df["true_mps"]) & (df["true_mps"] <= df["ci_high"])
    df.to_csv(out / "results.csv", index=False, encoding="utf-8-sig")
    summary = _summarize_and_plot(df, ["method", "impact", "fps"], out, "true_kmh", "est_kmh",
                                  "合成データ：初速の誤差")
    cov = df[df.method == "physics"].groupby(["impact", "fps"])["covered"].mean().rename("ci95_coverage")
    summary = summary.merge(cov.reset_index(), on=["impact", "fps"], how="left")
    summary.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")

    # 補正モデル（前半で学習・後半で評価）。物理モデルは補正前でもほぼ偏りがないので、主に差分法向け
    corr_rows = []
    half = n_shots // 2
    for (m, imp, fps), g in df.groupby(["method", "impact", "fps"]):
        tr, te = g[g.shot < half], g[g.shot >= half]
        if tr["est_kmh"].notna().sum() < 3 or te.empty:
            continue
        c = fit_linear_correction(tr["est_kmh"].to_numpy(), tr["true_kmh"].to_numpy())
        st = error_stats(te["true_kmh"], apply_correction(te["est_kmh"], c))
        corr_rows.append({"method": m, "impact": imp, "fps": fps, **c, "mape_after": st["mape"],
                          "mape_before": error_stats(te["true_kmh"], te["est_kmh"])["mape"]})
    pd.DataFrame(corr_rows).to_csv(out / "correction.csv", index=False, encoding="utf-8-sig")
    (out / "study.json").write_text(json.dumps({
        "n_shots": n_shots, "fps": list(fps_list), "seed": seed, "plane_error_deg": plane_error_deg,
        "plane_error_m": plane_error_m, "noise_px": s["noise_px"], "dropout": s["dropout"],
        "exposure_s": s["exposure_s"], "camera": cam.to_dict()}, indent=1), encoding="utf-8")
    return df


# =============================================================================
# 合成の撮影セッション（動画 + キャリブレーション + 正解）
# =============================================================================
def make_synthetic_session(cfg: dict[str, Any], out: Path, speed: float = 80.0, fps: float = 240.0,
                           exposure_s: float = 0.0, click_noise_px: float = 0.7, seed: int = 0
                           ) -> dict[str, Any]:
    """合成の動画・音声・キャリブレーション・正解を書き出す（パイプラインの通し確認用）。"""
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    phys = PhysicsParams.from_config(cfg)
    cam = synth.default_camera(cfg)
    shot = synth.make_shot(speed, cfg, phys)
    truth = synth.render_video(shot, cam, cfg, phys, out / "synthetic.mp4", fps=fps, exposure_s=exposure_s,
                               audio_path=out / "synthetic.wav", seed=seed)
    truth.to_csv(out / "truth_detections.csv", index=False)
    table = landmark_table(cfg)
    names, world, image = [], [], []
    for n, X in table.items():
        uv = cam.project(X[None])[0]
        if np.isfinite(uv).all() and 0 <= uv[0] < cam.width and 0 <= uv[1] < cam.height:
            names.append(n)
            world.append(X)
            image.append(uv + rng.normal(0, click_noise_px, 2))
    res = calibrate(names, np.array(world), np.array(image), cam.width, cam.height)
    save_calibration(res, out / "calibration.json", video=str(out / "synthetic.mp4"))
    pre_s = 0.2
    info = {"speed0_mps": speed, "speed0_kmh": speed * MPS_TO_KMH, "fps": fps, "t_impact_video_s": pre_s,
            # 打者の足元は打点の真下から contact_offset_m だけ後ろ（hitter_position 法の仮定に合わせる）
            "hitter_xy": (shot.plane.origin[:2] - cfg["events"]["contact_offset_m"]
                          * shot.plane.direction[:2]).tolist(),
            "landing_xy": shot.landing_xy.tolist(),
            "t_land_video_s": pre_s + shot.t_land - shot.t_impact, "calibration_rmse_px": res.rmse_px}
    (out / "truth.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info


# =============================================================================
# 実データによる検証（240fps を正解とする）
# =============================================================================
def dataset_study(cfg: dict[str, Any], dataset: Path, out: Path, fps_list: list[float] = (30, 60, 120, 240),
                  reference_csv: Path | None = None, detectors: tuple[str, ...] = ("labels", "baseline"),
                  impact: str = "auto") -> pd.DataFrame:
    """ラベル付きの高 fps 動画から、fps 別・検出手法別・算出方法別に初速を比べる。

    ``dataset`` は一覧 CSV（またはそれを ``index.csv`` として含むディレクトリ）で、列は
    ``shot, video, labels, calib, capture_fps, hitter_x, hitter_y, landing_x, landing_y``（``labels`` は
    ``annotate`` の出力 ``labels.csv``）。打点時刻は ``impact="auto"`` なら各動画の音声などから自動で、
    ``impact="labels"`` ならラベルの打点フレームの時刻（低 fps 版では時刻のずれを補正）を使う。

    検出手法:
        - ``labels``: 正解ラベルを間引いたもの（検出が完璧な場合の上限）
        - ``baseline``: 疑似低 fps 動画にベースライン検出をかけたもの
        - ``tracknet``: 設定に TrackNetV3 があれば、その検出

    240fps × labels × physics の初速を正解とする。外部計測値（``reference_csv``: shot, speed_kmh）があれば併記する。
    """
    from smash_speed.pipeline import AnalyzeOptions, analyze
    from smash_speed.train.lowfps import make_lowfps
    from smash_speed.video_io import probe

    index = dataset / "index.csv" if dataset.is_dir() else dataset
    base = index.parent
    items = pd.read_csv(index)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for it in items.itertuples():
        video = base / it.video
        from smash_speed.annotate import normalize_labels
        labels = normalize_labels(pd.read_csv(base / it.labels))
        cap = getattr(it, "capture_fps", None)
        cap = None if cap is None or pd.isna(cap) else float(cap)
        meta = probe(video, cap, cfg)
        hit = labels[labels.event == "hit"]
        t_imp = float(hit.timestamp.iloc[0]) if len(hit) else None
        for fps in fps_list:
            low = make_lowfps(meta, base / it.labels, fps, out / "lowfps" / it.shot, blur=True)
            vid_low = Path(low["video"])
            for det_name in detectors:
                if det_name == "labels":
                    det_csv = Path(low["labels"])
                elif det_name == "baseline":
                    det_csv = None
                else:
                    from smash_speed.detection.tracknet import run_tracknet
                    d = run_tracknet(probe(vid_low, None, cfg), None, None, cfg,
                                     work_dir=out / "tracknet" / it.shot / f"{fps:g}")
                    det_csv = out / "tracknet" / it.shot / f"{fps:g}" / "detections.csv"
                    d.to_csv(det_csv, index=False)
                o = AnalyzeOptions(
                    video=vid_low, calib=base / it.calib, out_dir=out / "runs" / it.shot / f"{fps:g}_{det_name}",
                    detections=det_csv, hitter_xy=(it.hitter_x, it.hitter_y),
                    landing_xy=(it.landing_x, it.landing_y),
                    impact_time_s=(t_imp - low["time_offset_s"]) if (impact == "labels" and t_imp is not None) else None,
                    overlay=False,
                    name=f"{it.shot}_{fps:g}_{det_name}")
                try:
                    r = analyze(o, cfg)
                except (RuntimeError, ValueError) as e:
                    print(f"[失敗] {it.shot} {fps} {det_name}: {e}")
                    continue
                for method, col in (("physics", "speed_initial_kmh"), ("naive", "naive_first_interval_kmh")):
                    rows.append({"shot": it.shot, "fps": fps, "detector": det_name, "method": method,
                                 "est_kmh": r[col]})
    df = pd.DataFrame(rows)
    ref = df[(df.fps == max(fps_list)) & (df.detector == "labels") & (df.method == "physics")]
    df = df.merge(ref[["shot", "est_kmh"]].rename(columns={"est_kmh": "ref_kmh"}), on="shot", how="left")
    if reference_csv is not None:
        ext = pd.read_csv(reference_csv).rename(columns={"speed_kmh": "external_kmh"})
        df = df.merge(ext[["shot", "external_kmh"]], on="shot", how="left")
    df.to_csv(out / "results.csv", index=False, encoding="utf-8-sig")
    _summarize_and_plot(df, ["detector", "method", "fps"], out, "ref_kmh", "est_kmh",
                        f"実データ：{max(fps_list):g}fps を正解とした初速の誤差")
    return df
