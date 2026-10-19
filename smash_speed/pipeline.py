"""``analyze``：動画1本（スマッシュ1本）から初速・平均速度を求める一連の処理。

途中結果はすべて出力先に保存し、既にあるものは再利用する（``force`` で作り直し）::

    out_dir/
      config_used.yaml   使用した設定と実行メタデータ
      video_meta.json    fps・解像度・タイムスタンプ
      detections.csv     フレームごとのシャトルの画像座標
      audio.wav          解析区間の音声
      events.json        打点時刻（方法・不確かさ）、打球音の一覧、着地時刻
      candidates.csv     検出の候補（軌道にする前の、動く小さな物の一覧）
      observations.csv   当てはめに使ったシャトルの位置
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

import cv2
import numpy as np
import pandas as pd

from smash_speed import events, report
from smash_speed.calibration.io import load_calibration
from smash_speed.camera import Camera
from smash_speed.config import save_config
from smash_speed.detection import guided
from smash_speed.detection.baseline import DETECTOR_VERSION, BaselineParams, detect, make_roi_mask
from smash_speed.errors import AnalysisError
from smash_speed.geometry import FlightPlane, floor_point, intersect_plane
from smash_speed.physics import (
    FitOptions, FitResult, Observations, PhysicsParams, finite_difference_speeds, fit_trajectory, simulate,
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


def _same_json(path: Path, obj: dict[str, Any]) -> bool:
    """保存済みの JSON が obj と同じか（検出結果を使い回してよいかの判定）。"""
    try:
        return json.loads(path.read_text(encoding="utf-8")) == json.loads(json.dumps(obj, default=str))
    except (OSError, ValueError):
        return False


def _camera_for_video(camera: Camera, meta: VideoMeta) -> Camera:
    """キャリブレーションした画像サイズと動画のサイズが違えば、カメラを拡大縮小する。"""
    if camera.width == meta.width and camera.height == meta.height:
        return camera
    s = meta.width / camera.width
    if abs(meta.height / camera.height - s) > 1e-3:
        raise AnalysisError(f"キャリブレーションの画像サイズ {camera.width}x{camera.height} と動画 "
                            f"{meta.width}x{meta.height} の縦横比が違います",
                            ["この動画で手順 2「コートを合わせる」をやり直してください（別のカメラ位置・解像度の"
                             "キャリブレーションが選ばれています）。"])
    return camera.scaled(s)


def _select_tracks(det: pd.DataFrame, onsets: list[tuple[float, float]] | None = None,
                   hit_uv: np.ndarray | None = None, img_w: float = 1.0, lead_s: float = 0.04
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """検出から、打点後の軌道と、その直前に終わる打点前の軌道を選ぶ。

    手がかり（あるものだけ使う）:
        - 検出のスコア（長さ・速さ・滑らかさ・向きのそろい方。``confidence`` 列から戻す）
        - 打球音の直後に始まっているか（強い打球音の 0.04 s 前〜0.2 s 後。直後ほどよい）
        - 打つ人の打点（足元の上 2.5 m）の近くから始まっているか
    """
    vis = det[det.visible == 1]
    if "event" in det and (det["event"].fillna("") == "hit").any():
        # 手でつけた位置：「打った瞬間」の印のあるコマから後を打点後の軌道とする
        f_hit = int(det.loc[det["event"].fillna("") == "hit", "frame"].iloc[-1])
        return vis[vis.frame >= f_hit], vis[vis.frame < f_hit]
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
        c = float(g["confidence"].iloc[0]) if "confidence" in g else 0.0
        if 0.0 < c < 1.0:
            score = 1000.0 * c / (1.0 - c)  # ベースライン検出の track_score
        else:  # TrackNet・手動ラベルなど
            score = speed * np.sqrt(len(g))
        t0 = float(g.timestamp.min())
        if strong:
            gaps = [t0 - t for t in strong if t - lead_s <= t0 <= t + 0.2]
            # 打った直後から写っている軌道ほどよい（0.06 s 遅れで半分）
            score *= 1.0 / (1.0 + (max(min(gaps, key=abs), 0.0) / 0.06) ** 2) if gaps else 0.15
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


def _strong_onsets(onsets: list[tuple[float, float]], ratio: float = 0.3, n_max: int = 3) -> list[float]:
    """強い打球音（いちばん強い音の ``ratio`` 倍以上）の時刻 [s]。強い順に最大 ``n_max`` 個。"""
    if not onsets:
        return []
    smax = max(s_ for _, s_ in onsets)
    strong = sorted((o_ for o_ in onsets if o_[1] >= ratio * smax), key=lambda o_: -o_[1])
    return [t for t, _ in strong[:n_max]]


def _as_track(points: pd.DataFrame) -> pd.DataFrame:
    """点の表（列 ``frame, timestamp, x_px, y_px``）を、検出結果と同じ形の 1 本の軌道にする。"""
    tr = points[["frame", "timestamp", "x_px", "y_px"]].copy()
    tr["frame"] = tr["frame"].astype(int)
    tr["visible"] = 1
    tr["confidence"] = 1.0
    tr["track_id"] = 0
    return tr.sort_values("frame").reset_index(drop=True)


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
        raise AnalysisError("シャトルが飛んだ面（打つ人 → 着地点）を自動で決められませんでした。",
                            ["手順 3 で「打つ人の足元」と「着地点」を指定してください。"])
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


def _landing_distance(o: AnalyzeOptions, camera: Camera, plane: FlightPlane) -> float | None:
    """打つ人の足元から着地点までの水平距離 [m]（着地点がわからなければ None）。"""
    if o.landing_xy is not None:
        P = np.r_[np.asarray(o.landing_xy, float)[:2], 0.0]
    elif o.landing_px is not None:
        P = floor_point(camera, np.array([o.landing_px], float))[0]
    else:
        return None
    s_land = float(plane.to_plane(P[None])[0, 0])
    return s_land if np.isfinite(s_land) and s_land > 0 else None


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

    # --- 飛行平面（検出する範囲・軌道の選択・打球音の伝搬遅れの見積もりにも使うので先に決める）
    reader = VideoReader(meta) if o.plane_gui else None
    plane, plane_src = _resolve_plane(o, camera, cfg, reader, o.start_s or 0.0)
    if reader is not None:
        reader.close()
    ev_cfg = cfg.get("events", {})
    det_cfg = cfg.get("detection", {})
    c_sound = ev_cfg.get("speed_of_sound_mps", 343.0)
    hitter_known = plane is not None and plane_src in ("file", "court_xy", "image_px", "gui")
    # 打点の位置の目安：打つ人の足元から飛行方向へ contact_offset 前、高さ hit_height
    hit_X = plane.to_world(np.array([ev_cfg.get("contact_offset_m", 0.3)]),
                           np.array([ev_cfg.get("hit_height_m", 2.5)]))[0] if hitter_known else None
    hit_uv = camera.project(hit_X[None])[0] if hit_X is not None else None

    # --- シャトル検出
    det_path = out / "detections.csv"
    cand_path = out / "candidates.csv"
    det_meta_path = out / "detections_meta.json"
    bp = BaselineParams.from_config(cfg)
    if o.roi:
        bp.roi = list(o.roi)
    own_detection = o.detections is None
    if own_detection and not bp.roi and hitter_known and det_cfg.get("auto_roi", True):
        # 検出する範囲が指定されていなければ、打つ人から着地点までの空間を画像に投影した範囲にする
        length = _landing_distance(o, camera, plane)
        poly = guided.corridor_polygon(camera, plane, length if length is not None else 10.0)
        if poly is not None and guided.polygon_area_ratio(poly, meta.width, meta.height) < 0.9:
            bp.roi_polygon = np.round(poly, 1).tolist()
            log.append("検出する範囲を、打つ人から着地点までの空間（床〜高さ 4 m）に自動で絞りました。")
    n_cand = det_cfg.get("n_track_candidates", 40)
    det_key = {"version": DETECTOR_VERSION, "start": o.start_s, "end": o.end_s, "roi": bp.roi,
               "params": {k: v for k, v in vars(bp).items() if k != "extra"} | dict(bp.extra), "n": n_cand}
    cands = pd.DataFrame()
    if not own_detection:
        det = pd.read_csv(o.detections)
        det.to_csv(det_path, index=False)
    elif det_path.exists() and cand_path.exists() and not o.force and _same_json(det_meta_path, det_key):
        det = pd.read_csv(det_path)
        cands = pd.read_csv(cand_path)
    else:  # 検出の方法・設定・区間が前回と違えば、保存済みの検出は使わない
        det, cands = detect(meta, o.start_s, o.end_s, bp, n_tracks=n_cand, return_candidates=True, exclusive=False)
        det.to_csv(det_path, index=False)
        cands.to_csv(cand_path, index=False)
        det_meta_path.write_text(json.dumps(det_key, ensure_ascii=False, default=str), encoding="utf-8")
    unit_px = max(1.0, meta.width / bp.process_width)  # 検出に使った縮小画像の 1 px が元の解像度で何 px か

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
    # 打つ人の位置と打球音がわかっていれば、打点から打ち出した軌道を総当たりで作り、検出の候補を最も多く通るものを
    # 選ぶ（選手・観客・ラケットの動きは、打点から出る物理的な軌道に乗らない）。見つからなければ従来の選び方にする
    search = None
    if own_detection and hitter_known and det_cfg.get("physics_search", True):
        if o.impact_time_s is not None:
            t_hits = [o.impact_time_s]
        else:
            delay0 = float(np.linalg.norm(camera.center - hit_X)) / c_sound
            t_hits = [t + o.av_offset_s - delay0 for t in _strong_onsets(onsets)]
        if t_hits:
            max_flight = ev_cfg.get("max_flight_s", 1.2)
            t0s, t1s = min(t_hits) - 3 * dt, max(t_hits) + max_flight + 2 * dt
            if o.start_s is not None:
                t0s = max(t0s, o.start_s)
            if o.end_s is not None:
                t1s = min(t1s, o.end_s)
            scale = 1.0 / unit_px
            with VideoReader(meta, scale=scale) as r:
                frames = [(i, t, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) for i, t, img in r.iter_range(t0s, t1s)]
            mask = make_roi_mask(bp, frames[0][2].shape[:2], scale) if frames else None
            maps = guided.evidence_maps(frames, mask)
            del frames
            search = guided.search_trajectory(maps, camera, plane, hit_X, t_hits, phys, scale, max_flight)
            if search is not None:
                predict = guided.trajectory_predictor(search.p0, search.v0, search.t_impact, phys, camera, t1s)
                with VideoReader(meta) as r:
                    gray = ((i, t, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                            for i, t, img in r.iter_range(search.t_impact - 1.5 * dt, t1s))
                    rd = guided.redetect_along(gray, predict, unit_px)
                rd = rd[rd.timestamp > search.t_impact]
                # 最初の検出の候補が予測位置の近くにあればそれを使い、候補のないコマだけ探し直した位置で補う
                # （探し直しは予測位置を手がかりにするので、候補があるならそちらの方が偏りがない）
                mc = guided.match_candidates(cands, predict, unit_px) if not cands.empty else rd.iloc[0:0]
                mc = mc[mc.timestamp > search.t_impact]
                pts = pd.concat([mc, rd[~rd.frame.isin(mc.frame)]]).sort_values("timestamp")
                if len(pts) >= 4:
                    search.matched = pts
                else:
                    search = None
    old_tr, old_in = _select_tracks(det, onsets, hit_uv, float(meta.width))
    if search is not None and len(old_tr) >= 4:
        # 従来の選び方の軌道が探索した軌道とほぼ重なれば、そちらを使う（候補を時間方向につないだ軌道なので、
        # 打った直後にラケットと重なった点が混じりにくい）。漏れたコマは当てはめの後で補う
        near = guided.match_candidates(
            old_tr.rename(columns={"x_px": "x", "y_px": "y"}), predict, unit_px)
        if len(near) >= 0.8 * len(old_tr):
            search.matched = old_tr
    if search is not None:
        out_tr = _as_track(search.matched)
        in_tr = old_in if search.matched is old_tr else out_tr.iloc[0:0]
        log.append(f"打点（打つ人の足元の少し前・高さ {ev_cfg.get('hit_height_m', 2.5):.1f} m）から打球音の時刻に打ち出した軌道を"
                   f"探し、シャトルを {len(out_tr)} コマ見つけました（だいたいの初速 {search.speed0 * 3.6:.0f} km/h）。")
    else:
        out_tr, in_tr = old_tr, old_in
    if len(out_tr) < 4 and own_detection:
        # 条件を緩めてもう一度探す（薄い残像・短い軌道も拾う）
        relaxed = BaselineParams(**{k: v for k, v in vars(bp).items() if k != "extra"}, extra=dict(bp.extra))
        relaxed.diff_threshold = max(6, int(round(bp.diff_threshold * 0.6)))
        relaxed.max_candidates = bp.max_candidates * 2
        relaxed.max_streak_area_px = bp.max_streak_area_px * 2
        relaxed.min_track_len = 3
        det2 = detect(meta, o.start_s, o.end_s, relaxed, n_tracks=n_cand, exclusive=False)
        out2, in2 = _select_tracks(det2, onsets, hit_uv, float(meta.width))
        if len(out2) > len(out_tr):
            log.append(f"シャトルが見つかりにくかったため、検出の条件を緩めて探し直しました（{len(out_tr)} → {len(out2)} 点）。")
            det, out_tr, in_tr = det2, out2, in2
            det.to_csv(det_path, index=False)
    if len(out_tr) < 4:
        hints = []
        if o.start_s is not None and o.end_s is not None and o.end_s - o.start_s < 0.5:
            hints.append(f"区間が {o.end_s - o.start_s:.2f} 秒と短すぎます。手順 1 で、打つ 0.3 秒ほど前から"
                         "シャトルが落ちるまで（1 秒前後）にしてください。")
        if bp.roi:
            hints.append("手順 3 の「検出する範囲」が狭すぎる可能性があります。打った後にシャトルが飛ぶ範囲全体を"
                         "囲み直すか、「消す」で範囲の指定をやめてください。")
        else:
            hints.append("手順 3 の「検出する範囲」で、打った後にシャトルが飛ぶ範囲（選手の頭より上など）を囲むと、"
                         "選手の動きと区別しやすくなります。")
        hints.append("手順 1 で動画を 1 コマずつ動かし、打った直後のシャトルが写っているか確認してください。"
                     "写っていない・ほとんど見えない場合は、シャッター速度を速くするか、シャトルが画面を横切る"
                     "向き（コートの横）から撮り直してください。")
        hints.append("それでも見つからない場合は、「研究ツール」の「正解ラベルを付ける」で打った直後の数コマに"
                     "シャトルの位置をつけて保存し、手順 4 の「詳細設定」→「シャトルの見つけ方」で"
                     "「手でつけた位置」を選ぶと解析できます。")
        raise AnalysisError(f"打った後のシャトルが {len(out_tr)} コマ分しか見つかりませんでした（4 コマ以上必要です）。",
                            hints)
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
        # 最初の検出はラケットの残像のこともあるので、打球音がそれより 1 コマ後までなら打点の候補にする
        # （打点より前のコマは、あとで当てはめから外す）
        impact = events.impact_from_audio(onsets, t_first, sigma_s=sigma_audio, tolerance_s=dt + 0.005,
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
    if impact.method in ("audio", "manual"):
        # 打った瞬間が音・手動でわかっていれば、それより前のコマ（ラケットとシャトルが重なって写る）は使わない
        use = out_tr[out_tr.timestamp > impact.t]
        if len(use) < len(out_tr):
            log.append(f"打った瞬間（{impact.t:.3f} s）より前の {len(out_tr) - len(use)} コマは、ラケットと重なって写るため使いませんでした。")
    else:
        use = out_tr[out_tr.timestamp >= min(impact.t, provisional.t) - 1e-9]
    use = use[use.timestamp <= impact.t + ev_cfg.get("max_flight_s", 1.2)]
    if len(use) < 4:
        raise AnalysisError(f"打った後のシャトルが {len(use)} コマ分しか使えませんでした（4 コマ以上必要です）。",
                            ["手順 1 の区間の終わりを、シャトルが床に落ちた後まで延ばしてください。",
                             "手順 3 の「検出する範囲」で、打った後にシャトルが飛ぶ範囲全体を囲んでください。"])
    obs = Observations(camera, use.timestamp.to_numpy(), use[["x_px", "y_px"]].to_numpy())

    opts = FitOptions.from_config(cfg)
    if o.ci_method:
        opts.ci_method = o.ci_method
    if o.mode == "plane" and plane is None:
        log.append("飛行平面が指定されていないため、格子探索で推定しました（試験的）。")
        plane = auto_plane(obs, impact.t, phys, cfg)
        opts.plane_sigma_deg, opts.plane_sigma_m = 10.0, 1.0

    fit_mode = o.mode
    if plane is not None:
        # カメラから打点への視線と飛行平面のなす角。小さい（飛ぶ方向を正面に近い角度から見ている）ほど、
        # 平面の位置のわずかなずれで奥行き＝速さが大きく変わる
        Xc = hit_X if hit_X is not None else plane.origin + np.array([0.0, 0.0, 2.5])
        ray = Xc - camera.center
        graze = float(np.degrees(np.arcsin(min(1.0, abs(ray @ plane.normal) / np.linalg.norm(ray)))))
        free_below = cfg["physics"].get("free_below_graze_deg", 20.0)
        if (o.mode == "plane" and search is not None and graze < free_below
                and ev_cfg.get("use_hit_prior", True) and hit_X is not None):
            # 視線が飛行平面に沿っていると、平面の横のわずかなずれで奥行き（＝速さ）が大きく変わる。
            # 平面に縛らず、「打球音の時刻に打点の近くを出た」ことだけを手がかりに 3 次元の軌道を当てはめる
            fit_mode = "free"
            log.append(f"視線と飛行面の角度が {graze:.0f}° と小さいため、飛行面に縛らずに 3 次元の軌道を当てはめました"
                       "（打球音の時刻に打点の近くを出たことを手がかりにします）。")
        if graze < 35.0:
            log.append(f"カメラがシャトルの飛ぶ方向を斜め前（後ろ）から見ています（視線と飛行面の角度 {graze:.0f}°）。"
                       "この角度では、打つ人・着地点の位置が少しずれるだけで速さが大きく変わります。"
                       "コートの横（ネットの延長線上の少し後ろ）から撮ると安定します。")

    if hit_X is not None and impact.method in ("audio", "manual") and ev_cfg.get("use_hit_prior", True):
        # 打点時刻が別に決まっているときは、その時刻の位置を打つ人の位置の近くに寄せる
        opts.p0_prior_m = hit_X
        opts.p0_prior_sigma_m = (ev_cfg.get("hit_prior_sigma_m", 0.6), ev_cfg.get("hit_height_sigma_m", 0.5))

    want_refine = opts.refine_plane
    if opts.refine_plane and len(obs.t) <= ev_cfg.get("min_frames_refine_plane", 5):
        opts.refine_plane = False
        log.append(f"使えるコマが {len(obs.t)} 個と少ないため、飛行平面は補正せず、指定どおりに使いました"
                   "（打つ人と着地点の位置が結果に直接効きます）。")

    # --- 当てはめ（外れ値を除いて再当てはめ）
    def run_fit(t_ref: float, sigma: float, o_: Observations) -> FitResult:
        opts.t_ref_sigma = sigma
        if search is not None:  # 探索で選んだ軌道を、この打点時刻まで進めて（戻して）初期値にする
            p_, v_ = simulate(search.p0, search.v0, search.t_impact, np.array([t_ref]), phys)
            opts.init_state = (p_[0], v_[0]) if np.isfinite(p_).all() else None
        try:
            return fit_trajectory(o_, t_ref, fit_mode, plane, phys, opts)
        except ValueError as e:
            raise AnalysisError(f"シャトルの軌道を計算できませんでした（{e}）。",
                                ["手順 3 の「打つ人の足元」「着地点」が実際の位置と合っているか確認してください。",
                                 "手順 2 のコートの合わせ込みで、黄色い線が実際のラインに重なっているか確認してください。",
                                 "検出したシャトルの位置が少なすぎる可能性があります。「検出する範囲」を囲んで"
                                 "やり直してください。"]) from e

    fit = run_fit(impact.t, impact.sigma, obs)
    # 打った直後のコマは、振ったラケットの残像がシャトルと重なって写ることがある。ずれが大きいときは、
    # 最初の 2 点のどちらかを外すと大きく改善するか試す（全体のずれが小さいと外れ値の判定が効かないため）
    if fit.rms_px > 5.0 and len(obs.t) >= 5:
        best = None
        for j in range(min(2, len(obs.t))):
            keep = np.ones(len(obs.t), bool)
            keep[j] = False
            try:
                f2 = run_fit(impact.t, impact.sigma, obs.subset(keep))
            except AnalysisError:
                continue
            if f2.rms_px < 0.5 * fit.rms_px and (best is None or f2.rms_px < best[0].rms_px):
                best = (f2, keep, j)
        if best is not None:
            log.append(f"打った直後の {best[2] + 1} コマ目は、ラケットなどと重なって写っているとみなして使いませんでした"
                       f"（画像上のずれ {fit.rms_px:.0f} → {best[0].rms_px:.1f} px）。")
            fit, obs = best[0], obs.subset(best[1])
    pred, _ = fit.evaluate(obs.t, phys)
    err = np.linalg.norm(camera.project(pred) - obs.uv, axis=1)
    bad = err > 3 * max(opts.loss_scale_px, fit.rms_px)
    if bad.any() and (~bad).sum() >= 4:
        log.append(f"再投影誤差の大きい {int(bad.sum())} 点を除いて当てはめ直しました。")
        obs = obs.subset(~bad)
        fit = run_fit(impact.t, impact.sigma, obs)

    # --- 当てはめた軌道に沿って探し直す（最初の検出で漏れたコマを拾う）
    if own_detection and "event" not in det and det_cfg.get("redetect", True):
        n0, rms0 = len(obs.t), fit.rms_px
        t_end = impact.t + ev_cfg.get("max_flight_s", 1.2)
        if o.end_s is not None:
            t_end = min(t_end, o.end_s)
        for _ in range(2):
            predict = guided.fit_predictor(fit, phys, camera, t_end)
            t_lo = impact.t if impact.method in ("audio", "manual") else float(obs.t.min()) - 0.5 * dt
            with VideoReader(meta) as r:
                gray = ((i, t, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                        for i, t, img in r.iter_range(t_lo - 1.5 * dt, t_end + 1.5 * dt))
                rd = guided.redetect_along(gray, predict, unit_px)
            rd = rd[(rd.timestamp > t_lo) & (rd.timestamp <= t_end)]
            if rd.empty:
                break
            # 使っている点はそのままにし、点のないコマを補う：まず最初の検出の候補（予測位置の近く）、
            # なければ探し直した位置（探し直しは予測位置を手がかりにするので、候補の方が偏りがない）
            old_pts = pd.DataFrame({"frame": [meta.frame_at(t) for t in obs.t], "timestamp": obs.t,
                                    "x_px": obs.uv[:, 0], "y_px": obs.uv[:, 1]})
            mc = guided.match_candidates(cands, predict, unit_px) if not cands.empty else rd.iloc[0:0]
            mc = mc[(mc.timestamp > t_lo) & (mc.timestamp <= t_end)]
            add = pd.concat([mc, rd[~rd.frame.isin(mc.frame)]])
            # 軌道にあったのに外したコマ（打つ前・ラケットと重なった・ずれの大きい点）は補わない
            removed = set(out_tr.frame) - set(old_pts.frame)
            add = add[~add.frame.isin(removed)]
            new_pts = pd.concat([old_pts, add[~add.frame.isin(old_pts.frame)][old_pts.columns]])
            new_pts = new_pts.drop_duplicates("frame").sort_values("timestamp")
            if len(new_pts) == len(old_pts):
                break
            obs_new = Observations(camera, new_pts.timestamp.to_numpy(), new_pts[["x_px", "y_px"]].to_numpy())
            if len(obs_new.t) < 4:
                break
            if want_refine and not opts.refine_plane and len(obs_new.t) > ev_cfg.get("min_frames_refine_plane", 5):
                opts.refine_plane = True
            f_new = run_fit(impact.t, impact.sigma, obs_new)
            pred, _ = f_new.evaluate(obs_new.t, phys)
            err = np.linalg.norm(camera.project(pred) - obs_new.uv, axis=1)
            bad = err > 3 * max(opts.loss_scale_px, f_new.rms_px, unit_px)
            if bad.any() and (~bad).sum() >= 4:
                obs_new = obs_new.subset(~bad)
                f_new = run_fit(impact.t, impact.sigma, obs_new)
            if f_new.rms_px > max(1.5 * fit.rms_px, fit.rms_px + 1.5 * unit_px):
                break  # 探し直した方が軌道に合わない（別の物を拾った）なら使わない
            fit, obs = f_new, obs_new
        if len(obs.t) != n0 or abs(fit.rms_px - rms0) > 1e-6:
            log.append(f"当てはめた軌道に沿って、漏れたコマを探し直しました（使えたコマ {n0} → {len(obs.t)}、"
                       f"画像上のずれ {rms0:.1f} → {fit.rms_px:.1f} px）。")
            pts = pd.DataFrame({"frame": [meta.frame_at(t) for t in obs.t], "timestamp": obs.t,
                                "x_px": obs.uv[:, 0], "y_px": obs.uv[:, 1]})
            out_tr = _as_track(pd.concat([pts, out_tr[~out_tr.frame.isin(pts.frame)][pts.columns]]))

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
            lambda t: f0.evaluate(np.array([t]), phys)[0][0], f0.plane or plane, search_lo(f0), t_first + dt / 2,
            ev.get("contact_offset_m", 0.3), ev.get("contact_sigma_m", 0.4))
        if hp is not None:
            impact = hp
            fit = run_fit(impact.t, impact.sigma, obs)
        else:
            log.append("軌道が打者の位置を通過する時刻を求められませんでした（打点時刻は最初の検出から推定）。")

    # 打球音で決めた打点時刻で当てはめが悪いときは、映像だけで決めた打点時刻（軌道が打つ人の位置を通る時刻）も試す。
    # 打球音が別の音（足音・隣のコート・返球）だったり、音と映像がずれていたりすると、音の時刻では軌道が合わない
    if (impact.method == "audio" and method == "auto" and fit.rms_px > 6.0
            and plane_src in ("file", "court_xy", "image_px", "gui")):
        all_use = out_tr[out_tr.timestamp <= impact.t + ev_cfg.get("max_flight_s", 1.2)]
        obs_all = Observations(camera, all_use.timestamp.to_numpy(), all_use[["x_px", "y_px"]].to_numpy())
        try:
            f0 = run_fit(provisional.t, provisional.sigma, obs_all)
            ev = cfg.get("events", {})
            hp = events.impact_from_hitter_position(
                lambda t: f0.evaluate(np.array([t]), phys)[0][0], f0.plane or plane, search_lo(f0), t_first + dt / 2,
                ev.get("contact_offset_m", 0.3), ev.get("contact_sigma_m", 0.4))
            if hp is not None:
                keep = obs_all.t > hp.t
                if keep.sum() >= 4:
                    f_hp = run_fit(hp.t, hp.sigma, obs_all.subset(keep))
                    if f_hp.rms_px < 0.5 * fit.rms_px:
                        log.append(f"打球音（{impact.t:.3f} s）では軌道が合わないため（画像上のずれ {fit.rms_px:.0f} px）、"
                                   f"軌道が打つ人の位置を通る時刻 {hp.t:.3f} s を打った瞬間としました（ずれ {f_hp.rms_px:.1f} px）。"
                                   "打球音が別の音（足音・返球・隣のコート）だった可能性があります。")
                        impact, fit, obs = hp, f_hp, obs_all.subset(keep)
        except AnalysisError:
            pass

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
    pd.DataFrame({"frame": [meta.frame_at(t) for t in obs.t], "timestamp": obs.t, "x_px": obs.uv[:, 0],
                  "y_px": obs.uv[:, 1]}).to_csv(out / "observations.csv", index=False)
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
            used = pd.concat([in_tr, out_tr[out_tr.timestamp.isin(obs.t)]])  # 当てはめに使った軌道だけを描く
            report.render_overlay(r, t0, t1, used, fit, phys, camera, cfg, out / "overlay.mp4", metrics)
    for line in log:
        print("[注意]", line)
    return row
