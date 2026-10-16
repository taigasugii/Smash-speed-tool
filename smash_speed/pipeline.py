"""``analyze``：動画1本（スマッシュ1本）から初速・平均速度を求める一連の処理。

途中結果はすべて出力先に保存し、既にあるものは再利用する（``force`` で作り直し）::

    out_dir/
      config_used.yaml   使用した設定と実行メタデータ
      video_meta.json    fps・解像度・タイムスタンプ
      detections.csv     フレームごとのシャトルの画像座標
      audio.wav          解析区間の音声
      events.json        打点時刻（方法・不確かさ）、打球音の一覧、着地時刻
      plane.json         飛行平面（モード A）
      fit_result.json    軌道フィッティングの結果
      result.csv         速度指標（1行）
      speed_time.png / trajectory.png / image_fit.png / overlay.mp4
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from smash_speed import events, report
from smash_speed.calibration.io import load_calibration
from smash_speed.camera import Camera
from smash_speed.config import save_config
from smash_speed.detection.baseline import BaselineParams, detect
from smash_speed.geometry import FlightPlane, floor_point, intersect_plane
from smash_speed.physics import (
    FitOptions, FitResult, Observations, PhysicsParams, finite_difference_speeds, fit_trajectory,
    speed_metrics,
)
from smash_speed.repro import run_metadata, set_seed
from smash_speed.video_io import VideoMeta, VideoReader, extract_audio, load_wav, probe


@dataclass
class AnalyzeOptions:
    """``analyze`` の入力。座標は [m]（コート座標）または [px]（元動画の画像座標）。"""

    video: Path
    calib: Path
    out_dir: Path
    capture_fps: float | None = None
    mode: str = "plane"
    start_s: float | None = None
    end_s: float | None = None
    detections: Path | None = None          # 既存の検出 CSV を使う（TrackNet の結果など）
    plane_json: Path | None = None
    hitter_xy: tuple[float, float] | None = None
    landing_xy: tuple[float, float] | None = None
    hitter_px: tuple[float, float] | None = None
    landing_px: tuple[float, float] | None = None
    plane_gui: bool = False
    impact_time_s: float | None = None      # 手動指定の打点時刻（動画の実時間）
    impact_method: str = "auto"             # auto | audio | crossing | first_frame | manual
    av_offset_s: float = 0.0
    audio_path: Path | None = None
    ci_method: str | None = None
    overlay: bool = True
    force: bool = False
    name: str | None = None
    roi: list[float] | None = None          # 検出する範囲 [x0, y0, x1, y1]（元解像度 px）
    extra: dict[str, Any] = field(default_factory=dict)


def _camera_for_video(camera: Camera, meta: VideoMeta) -> Camera:
    """キャリブレーションした画像サイズと動画のサイズが違えば、カメラを拡大縮小する。"""
    if camera.width == meta.width and camera.height == meta.height:
        return camera
    s = meta.width / camera.width
    if abs(meta.height / camera.height - s) > 1e-3:
        raise ValueError(f"キャリブレーションの画像サイズ {camera.width}x{camera.height} と動画 "
                         f"{meta.width}x{meta.height} の縦横比が違います")
    return camera.scaled(s)


def _select_tracks(det: pd.DataFrame, onsets: list[tuple[float, float]] | None = None,
                   hit_uv: np.ndarray | None = None, img_w: float = 1.0
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """検出から、打点後の軌道と、その直前に終わる打点前の軌道を選ぶ。

    手がかり（あるものだけ使う）:
        - 画像上の速さと長さ（速く長く続くほどシャトルらしい）
        - 打球音の直後に始まっているか（強い打球音の 0.04 s 前〜0.2 s 後）
        - 打つ人の打点（足元の上 2.5 m）の近くから始まっているか
    """
    vis = det[det.visible == 1]
    if "track_id" not in vis or vis["track_id"].nunique() <= 1:
        return vis, vis.iloc[0:0]
    strong = []
    if onsets:
        smax = max(s_ for _, s_ in onsets)
        strong = [t for t, s_ in onsets if s_ >= 0.3 * smax]
    scores = {}
    for tid, g in vis.groupby("track_id"):
        if len(g) < 3 or tid < 0:
            continue
        uv = g[["x_px", "y_px"]].to_numpy()
        speed = np.median(np.linalg.norm(np.diff(uv, axis=0), axis=1) / np.diff(g.timestamp))
        score = speed * np.sqrt(len(g))
        t0 = float(g.timestamp.min())
        if strong:
            score *= 1.0 if any(t - 0.04 <= t0 <= t + 0.2 for t in strong) else 0.15
        if hit_uv is not None and np.isfinite(hit_uv).all():
            dd = np.linalg.norm(uv[0] - hit_uv) / img_w
            score *= 1.0 / (1.0 + (dd / 0.12) ** 2)
        scores[tid] = score
    if not scores:
        return vis, vis.iloc[0:0]
    out_id = max(scores, key=scores.get)
    out = vis[vis.track_id == out_id]
    t0 = out.timestamp.min()
    before = vis[(vis.track_id != out_id) & (vis.timestamp < t0)]
    inc = before.iloc[0:0]
    if len(before):
        last_end = before.groupby("track_id").timestamp.max()
        tid = last_end.idxmax()
        if t0 - last_end[tid] < 0.15:
            inc = before[before.track_id == tid]
    return out, inc


def auto_plane(obs: Observations, t_ref: float, phys: PhysicsParams, cfg: dict[str, Any]
               ) -> FlightPlane:
    """飛行平面が与えられないとき、平面の位置と向きを格子探索して再投影誤差が最小のものを選ぶ（試験的）。

    平面は「ネットを y = y_net で横切り、x 軸から yaw だけ回った鉛直面」として探索する。
    """
    W = cfg["court"]["half_width_doubles_m"]
    best = (np.inf, None)
    for y_net in np.arange(-W, W + 1e-9, 0.5):
        for yaw in np.deg2rad(np.arange(-35, 36, 5)):
            plane = FlightPlane(np.array([0.0, y_net, 0.0]), np.array([np.cos(yaw), np.sin(yaw), 0.0]))
            try:
                f = fit_trajectory(obs, t_ref, "plane", plane, phys, FitOptions(ci_method="none"))
            except (ValueError, RuntimeError):
                continue
            if f.success and f.rms_px < best[0]:
                best = (f.rms_px, plane)
    if best[1] is None:
        raise RuntimeError("飛行平面を自動で決められませんでした。--hitter / --landing を指定してください。")
    return best[1]


def _resolve_plane(o: AnalyzeOptions, camera: Camera, cfg: dict[str, Any], reader: VideoReader | None,
                   t_frame: float) -> tuple[FlightPlane | None, str]:
    if o.plane_json and Path(o.plane_json).exists():
        return FlightPlane.from_dict(json.loads(Path(o.plane_json).read_text(encoding="utf-8"))), "file"
    if o.hitter_xy and o.landing_xy:
        return FlightPlane.from_points(o.hitter_xy, o.landing_xy), "court_xy"
    if o.hitter_px and o.landing_px:
        P = floor_point(camera, np.array([o.hitter_px, o.landing_px], float))
        return FlightPlane.from_points(P[0], P[1]), "image_px"
    if o.plane_gui and reader is not None:
        from smash_speed.calibration.gui import run_plane_gui
        _, frame = reader.read_at(t_frame)
        plane = run_plane_gui(frame, camera, cfg)
        if plane is not None:
            return plane, "gui"
    return None, "auto"


def analyze(o: AnalyzeOptions, cfg: dict[str, Any]) -> dict[str, Any]:
    """動画1本を解析して、速度指標の辞書を返す（ファイルは ``o.out_dir`` に保存）。"""
    out = Path(o.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.get("seed", 0))
    save_config({**cfg, "_run": run_metadata({"options": {k: str(v) for k, v in o.__dict__.items()}})},
                out / "config_used.yaml")
    name = o.name or Path(o.video).stem
    phys = PhysicsParams.from_config(cfg)
    log: list[str] = []

    # --- 動画・カメラ
    meta = probe(o.video, o.capture_fps, cfg)
    meta.save(out / "video_meta.json")
    camera, _ = load_calibration(o.calib)
    camera = _camera_for_video(camera, meta)
    dt = 1.0 / meta.capture_fps

    # --- シャトル検出
    det_path = out / "detections.csv"
    if o.detections is not None:
        det = pd.read_csv(o.detections)
        det.to_csv(det_path, index=False)
    elif det_path.exists() and not o.force:
        det = pd.read_csv(det_path)
    else:
        bp = BaselineParams.from_config(cfg)
        if o.roi:
            bp.roi = list(o.roi)
        det = detect(meta, o.start_s, o.end_s, bp, n_tracks=cfg.get("detection", {}).get("n_track_candidates", 12))
        det.to_csv(det_path, index=False)
    # --- 飛行平面（軌道の選択・打球音の伝搬遅れの見積もりにも使うので先に決める）
    reader = VideoReader(meta) if o.plane_gui else None
    plane, plane_src = _resolve_plane(o, camera, cfg, reader, o.start_s or 0.0)
    if reader is not None:
        reader.close()
    ev_cfg = cfg.get("events", {})
    c_sound = ev_cfg.get("speed_of_sound_mps", 343.0)
    hitter_known = plane is not None and plane_src in ("file", "court_xy", "image_px", "gui")
    # 打点の位置の目安：打つ人の足元から飛行方向へ contact_offset 前、高さ hit_height
    hit_X = plane.to_world(np.array([ev_cfg.get("contact_offset_m", 0.3)]),
                           np.array([ev_cfg.get("hit_height_m", 2.5)]))[0] if hitter_known else None
    hit_uv = camera.project(hit_X[None])[0] if hit_X is not None else None

    # --- 打球音
    onsets: list[tuple[float, float]] = []
    wav = o.audio_path
    if wav is None and meta.has_audio:
        wav = out / "audio.wav"
        if not wav.exists() or o.force:
            # 音声はメタデータ上の時刻で切り出し、実時間に換算する
            s0 = max(0.0, (o.start_s or 0.0) - 0.5) / meta.time_scale
            s1 = ((o.end_s if o.end_s is not None else meta.timestamps[-1]) + 0.2) / meta.time_scale
            extract_audio(o.video, wav, start_s=s0, end_s=s1, cfg=cfg)
        audio_t0 = max(0.0, (o.start_s or 0.0) - 0.5)
    else:
        audio_t0 = 0.0
    if wav is not None and Path(wav).exists():
        x, rate = load_wav(wav)
        onsets = [(audio_t0 + t * meta.time_scale, s) for t, s in events.audio_onsets(
            x, rate, ev_cfg.get("audio_highpass_hz", 2000.0), k_mad=ev_cfg.get("audio_k_mad", 10.0))]
        if meta.time_scale != 1.0:
            log.append("撮影 fps を上書きしたため、音声の時刻も同じ倍率で換算しました。音声と映像のずれを確認してください。")

    # --- 打点後の軌道を選ぶ
    out_tr, in_tr = _select_tracks(det, onsets, hit_uv, float(meta.width))
    if len(out_tr) < 4:
        raise RuntimeError(f"シャトルの検出が {len(out_tr)} 点しかありません（区間・ROI・閾値を確認してください）")
    t_first = float(out_tr.timestamp.min())
    if onsets:
        smax = max(s_ for _, s_ in onsets)
        strong = [t for t, s_ in onsets if s_ >= 0.3 * smax]
        if strong and not any(t - 0.04 <= t_first <= t + 0.2 for t in strong):
            log.append(f"使った軌道（{t_first:.2f} s から）は、強い打球音（{', '.join(f'{t:.2f}' for t in strong)} s）の直後に"
                       "始まっていません。打つ前のシャトルや別の物を拾っている可能性があります。"
                       "手順 3 の「検出する範囲」で、打った後にシャトルが飛ぶ範囲（選手の頭より上など）を囲んでください。")
    first_uv = out_tr[["x_px", "y_px"]].to_numpy()[:1]
    if hit_X is not None:
        X_first = hit_X
    elif plane is not None:
        X_first = intersect_plane(camera, first_uv, plane)[0]
    else:  # 平面が未定なら、カメラからコート中央までと同じ距離にあると仮定する
        c0, d0 = camera.rays(first_uv)
        X_first = c0 + np.linalg.norm(c0) * d0[0]
    dist0 = float(np.linalg.norm(camera.center - X_first)) if np.isfinite(X_first).all() else 0.0

    sigma_audio = cfg["physics"].get("impact_time_sigma_s", 0.002)
    impact: events.ImpactEstimate | None = None
    audio_raw_t: float | None = None
    method = o.impact_method
    if o.impact_time_s is not None or method == "manual":
        if o.impact_time_s is None:
            raise ValueError("--impact-time を指定してください")
        impact = events.ImpactEstimate(o.impact_time_s, ev_cfg.get("manual_sigma_s", 0.003), "manual")
    if impact is None and method in ("auto", "audio") and onsets:
        # 音はカメラまで 距離 / 音速 だけ遅れて届くので、その分を引いてから打点を選ぶ
        impact = events.impact_from_audio(onsets, t_first, sigma_s=sigma_audio,
                                          av_offset_s=o.av_offset_s - dist0 / c_sound)
        if impact is None:
            log.append("最初の検出の直前に打球音が見つかりませんでした。")
        else:
            audio_raw_t = impact.t - o.av_offset_s + dist0 / c_sound
    t_last_in = float(in_tr.timestamp.max()) if len(in_tr) else None
    provisional = events.impact_from_first_frame(t_first, dt, t_last_in)
    if impact is None and method in ("auto", "audio", "crossing", "hitter_position", "first_frame"):
        impact = provisional

    # --- 観測
    use = out_tr[out_tr.timestamp >= min(impact.t, provisional.t) - 1e-9]
    use = use[use.timestamp <= impact.t + ev_cfg.get("max_flight_s", 1.2)]
    obs = Observations(camera, use.timestamp.to_numpy(), use[["x_px", "y_px"]].to_numpy())

    opts = FitOptions.from_config(cfg)
    if o.ci_method:
        opts.ci_method = o.ci_method
    if o.mode == "plane" and plane is None:
        log.append("飛行平面が指定されていないため、格子探索で推定しました（試験的）。")
        plane = auto_plane(obs, impact.t, phys, cfg)
        opts.plane_sigma_deg, opts.plane_sigma_m = 10.0, 1.0

    if opts.refine_plane and len(obs.t) <= ev_cfg.get("min_frames_refine_plane", 5):
        opts.refine_plane = False
        log.append(f"使えるコマが {len(obs.t)} 個と少ないため、飛行平面は補正せず、指定どおりに使いました"
                   "（打つ人と着地点の位置が結果に直接効きます）。")

    # --- 当てはめ（外れ値を除いて再当てはめ）
    def run_fit(t_ref: float, sigma: float, o_: Observations) -> FitResult:
        opts.t_ref_sigma = sigma
        return fit_trajectory(o_, t_ref, o.mode, plane, phys, opts)

    fit = run_fit(impact.t, impact.sigma, obs)
    pred, _ = fit.evaluate(obs.t, phys)
    err = np.linalg.norm(camera.project(pred) - obs.uv, axis=1)
    bad = err > 3 * max(opts.loss_scale_px, fit.rms_px)
    if bad.any() and (~bad).sum() >= 4:
        log.append(f"再投影誤差の大きい {int(bad.sum())} 点を除いて当てはめ直しました。")
        obs = obs.subset(~bad)
        fit = run_fit(impact.t, impact.sigma, obs)

    # 打球音の伝搬遅れを、当てはめた打点の位置からの距離で見積もり直す
    sound_delay = None
    if audio_raw_t is not None:
        for _ in range(2):
            ref = hit_X if hit_X is not None else fit.p0
            if hit_X is None and (abs(ref[0]) > 9 or abs(ref[1]) > 6):  # 当てはめた打点がコートから遠すぎる
                ref = X_first
            dist = float(np.linalg.norm(camera.center - ref))
            sound_delay = dist / c_sound
            t_new = audio_raw_t + o.av_offset_s - sound_delay
            if abs(t_new - impact.t) < 1e-5:
                break
            impact = events.ImpactEstimate(t_new, impact.sigma, "audio")
            fit = run_fit(impact.t, impact.sigma, obs)
        log.append(f"打球音の伝搬遅れ {sound_delay * 1000:.1f} ms（カメラ〜打点 {dist:.2f} m）を補正しました。")

    lookback = cfg.get("events", {}).get("impact_lookback_s", 0.1)  # 最初の検出より前に打点を探す幅 [s]

    def search_lo(f: FitResult) -> float:
        # 2乗抵抗の軌道を時間の逆向きに積分すると t_ref - 1/(k|v0|) で発散するので、その手前までに限る
        t_sing = f.t_ref - 1.0 / (f.phys(phys).k * f.speed0)
        return max(t_first - lookback, f.t_ref - 0.7 * (f.t_ref - t_sing))

    # 打点前の軌道が見えていれば、軌道の交点から打点時刻を求め直す
    if impact.method == "first_frame" and method in ("auto", "crossing") and len(in_tr) >= 3:
        f0 = fit

        def outgoing_uv(t: float) -> np.ndarray:
            p, _ = f0.evaluate(np.array([t]), phys)
            return camera.project(p)[0]

        cross = events.impact_from_crossing(in_tr.timestamp.to_numpy(), in_tr[["x_px", "y_px"]].to_numpy(),
                                            outgoing_uv, search_lo(f0), t_first)
        if cross is not None:
            impact = cross
            fit = run_fit(impact.t, impact.sigma, obs)

    # 打者の位置がわかっていれば、軌道が打点（足元の少し前）を通過する時刻を打点とする
    if (impact.method == "first_frame" and method in ("auto", "hitter_position")
            and plane_src in ("file", "court_xy", "image_px", "gui")):
        f0 = fit
        ev = cfg.get("events", {})
        hp = events.impact_from_hitter_position(
            lambda t: f0.evaluate(np.array([t]), phys)[0][0], f0.plane, search_lo(f0), t_first + dt / 2,
            ev.get("contact_offset_m", 0.3), ev.get("contact_sigma_m", 0.4))
        if hp is not None:
            impact = hp
            fit = run_fit(impact.t, impact.sigma, obs)
        else:
            log.append("軌道が打者の位置を通過する時刻を求められませんでした（打点時刻は最初の検出から推定）。")

    metrics = speed_metrics(fit, phys)
    X_obs = intersect_plane(camera, obs.uv, fit.plane) if fit.plane is not None else np.empty((0, 3))
    naive = finite_difference_speeds(obs.t, X_obs)
    t_land_obs = events.landing_from_detections(obs.t, obs.uv)

    # --- 保存
    (out / "events.json").write_text(json.dumps({
        "impact": impact.to_dict(), "first_detection_s": t_first, "audio_onsets": onsets,
        "audio_onset_used_s": audio_raw_t, "sound_delay_s": sound_delay,
        "landing_model_s": metrics["t_land_s"], "landing_detection_s": t_land_obs,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    if fit.plane is not None:
        (out / "plane.json").write_text(json.dumps({**fit.plane.to_dict(), "source": plane_src},
                                                   indent=1), encoding="utf-8")
    (out / "fit_result.json").write_text(json.dumps({**fit.to_dict(), "log": log}, ensure_ascii=False,
                                                    indent=1), encoding="utf-8")
    row = report.result_row(name, fit, metrics, naive, {
        "capture_fps": meta.capture_fps, "impact_method": impact.method, "impact_sigma_s": impact.sigma,
        "plane_source": plane_src, "video": str(o.video), **o.extra})
    report.write_rows([row], out / "result.csv")
    report.plot_speed_time(fit, phys, naive, out / "speed_time.png", obs.t)
    report.plot_trajectory_views(fit, phys, X_obs, cfg, out / "trajectory.png")
    with VideoReader(meta) as r:
        _, bg = r.read_at(t_first)
        report.plot_image_fit(obs.t, obs.uv, fit, phys, camera, out / "image_fit.png", background=bg)
        if o.overlay:
            t0 = o.start_s if o.start_s is not None else max(0.0, impact.t - 0.3)
            t1 = o.end_s if o.end_s is not None else metrics["t_land_s"] + 0.3
            report.render_overlay(r, t0, t1, det, fit, phys, camera, cfg, out / "overlay.mp4", metrics)
    for line in log:
        print("[注意]", line)
    return row
