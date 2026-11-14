"""動画 1 本のすべてのショットを、種類・初速つきで一覧にする（ROADMAP B5 の解析の部分）。

1. 動画全体を 1 回だけ縮小して読み、シャトルを検出する（``detection.baseline``。1 コマ 1 点）。
   学習したモデル（``tracknet``）か、外の検出 CSV（``detections``）を代わりに使うこともできる（F1b）。
2. 打球音（音声があれば）と軌道の向きの変化から、すべての打球の時刻と打った側を出す（``hits.find_hits``）。
3. 打球の間隔で組に分け、組ごとに軌道と初速を当てはめる（``shot_fit.fit_shots``）。ほかの組の検出が
   混ざらないよう、組の前後で検出を切り分ける。``joint_fit=True`` なら、その後に組の中のショットをつないで
   まとめて当てはめ直し、長いショットからシャトルの終端速度も求める（``rally_fit.fit_rallies``）。
4. 打球の間隔・シャトルの着地・シャトルが飛んでいるかから、ラリーに区切る（``rallies.split_rallies``）。
5. 軌道から種類を決める（``shot_types.classify_shots``。ラリーの最初のショットはサーブ）。
6. 選手の位置も出すとき（``players=True``）は、選手の動きで短いラリーを確かめ、ラリーの間に拾って返した・
   となりのコートのシャトルをラリーから外す（``rallies_with_players``）。

結果は ``<出力先>/`` に書き出す。

- ``shots.csv``・``rallies.csv``：ショットとラリー（``shots`` の形）
- ``detections.csv``：全コマのシャトルの検出（GUI で場面に軌道を重ねる・解析し直すのに使う）
- ``shot_list.json``：動画・キャリブレーション・打球の見つけ方・種類ごとの本数と速さなど（GUI が読む）
- ``stats.json``・``landing.svg``：ラリーの統計と、着地点の分布図（``rally_stats``）
- ``report.html``：統計をまとめた 1 枚のページ（``rally_report``。スマホでも見られる）
- ``players.csv``・``people.csv``・``players_preview.jpg``・``players.json``：選手の位置（``players=True`` のときだけ。``players``）。
  このときは ``shots.csv`` の ``hitter_x_m``・``hitter_y_m`` に打つ人の足元も入れ（``hitter.attach_hitters``）、
  当てはめの前に出して打点の手がかりにもする（``hitter_prior``。``shot_fit.fit_shots`` の ``hitter_xy``）
- ``movement.json``・``movement.csv``・``movement.svg``：選手の移動距離・速さ・いた場所（同じく ``players=True`` のとき。``movement``）

``python -m smash_speed.shot_list <動画> --calib <calibration.json> --out <出力先>`` で動かす。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd

from smash_speed import events
from smash_speed.calibration.io import load_calibration
from smash_speed.camera import Camera
from smash_speed.detection.baseline import BaselineParams, detect_frames
from smash_speed.errors import AnalysisError
from smash_speed.hitter import attach_hitters, hitter_at
from smash_speed.hits import Hit, find_hits
from smash_speed.physics import FitOptions, PhysicsParams
from smash_speed.players import PLAYERS_FILE, PlayerOptions, analyze_players
from smash_speed.rallies import RallyOptions, split_rallies
from smash_speed.rally_report import write_report
from smash_speed.rally_stats import write_stats
from smash_speed.repro import run_metadata
from smash_speed.rally_fit import JointOptions, fit_rallies
from smash_speed.fit_batch import gate_session
from smash_speed.shot_fit import (ShotFit, _parallel_map, fit_shots, joint_job, pool_active, pool_executor,
                                  resolve_backend, shot_fits_to_shots, worker_env, worker_pool)
from smash_speed.shot_types import classify_shots
from smash_speed.shots import (RALLIES_FILE, SHOT_TYPE_LABELS, SHOT_TYPES, SHOTS_FILE, Rally, Shot, write_rallies,
                               write_shots)
from smash_speed.video_io import VideoMeta, extract_audio, load_wav, probe

SHOT_LIST_FILE = "shot_list.json"
DETECTIONS_FILE = "detections.csv"
TRACKNET_DIR = "tracknet"         # 学習したモデルの検出の作業場所（``<出力先>/tracknet/detections.csv``）
TRACKNET_THRESHOLD = 0.25         # 学習したモデルの熱の地図の閾値（F1b：打球の見つけ方で 0.25 が良い）
BATCH_RALLIES = 8                 # 当てはめをまとめて解くとき（``fit_backend="batch"``）に、1 つのプロセスで同時に進めるラリーの数
BATCH_PROCESSES = 4               # 同・ラリーを分けて進めるプロセスの数


@dataclass
class ShotListOptions:
    video: Path
    calib: Path
    out_dir: Path
    capture_fps: float | None = None
    start_s: float | None = None
    end_s: float | None = None
    roi: list[float] | None = None    # 検出する範囲 [x0, y0, x1, y1]（元解像度 px）
    audio: Path | None = None         # 別の音声ファイル（動画の時刻と同じ時刻の WAV）
    use_audio: bool = True            # 偽なら打球音を使わない（軌道だけで打球を見つける）
    rally_gap_s: float = 3.0          # 打球の間がこれより空き、シャトルが飛んでいなければ別のラリー [s]
    n_tracks: int | None = None       # 検出で残す軌道の数の上限（None なら全部。実動画では 20 秒で 1000 本ほど）
    filter_tracks: bool = True        # シャトルらしい軌道だけを残す（``shuttle_tracks``）
    workers: int | None = None        # 当てはめを並べて動かすプロセスの数（None なら CPU の数 − 1、16 まで）
    players: bool = False             # 選手の位置も出す（``players.analyze_players``。人の検出のモデルを使うので遅くなる）
    player_options: PlayerOptions | None = None
    hitter_prior: bool = True         # 選手の位置があれば、打つ人の足元を打点の手がかりにする（``fit_shots`` の ``hitter_xy``）
    hit_players: bool = True          # 選手の位置があれば、打球を見つけるのにも使う（``find_hits`` の ``players``。
                                      # どちらの選手にも届かない所の打球を捨て、近い方を打った側にする）
    detections: Path | None = None    # 自前の検出の代わりに使う検出 CSV（``tracknet_train detect`` の出力など。
                                      # 列 frame, x_px, y_px, visible[, track_id]。時刻はコマの番号から付け直す）
    tracknet: Path | None = None      # 学習したモデル（``tracknet_train`` の model.pt）。あればこれで全コマを検出し
                                      # （``tracknet_train.detect_video``。torch は ``.venv-train``）、自前の検出の代わりに使う
    tracknet_threshold: float = TRACKNET_THRESHOLD
    joint_fit: bool = False           # ラリーの中のショットをつないでまとめて当てはめ直す（``rally_fit``。打点・打った時刻を
                                      # となりのショットと合わせ、長いショットからシャトルの終端速度も求める）
    joint_options: JointOptions | None = None
    fit_backend: str = "auto"         # 当てはめの計算の仕方（``shot_fit.run_fits``）："cpu"（1 つずつプロセスで並べる）、
                                      # "batch"（段階ごとにまとめて解く。ROADMAP F3e）、"auto"（batch が使えれば batch）


# =============================================================================
# 検出と打球音
# =============================================================================
def _with_progress(frames: Iterator[tuple[int, float, np.ndarray]], n_total: int,
                   progress: Callable[[str], None], every: float = 0.1) -> Iterator[tuple[int, float, np.ndarray]]:
    """コマを読みながら、1 割ごとに進み具合を知らせる。"""
    step = max(1, int(n_total * every))
    for k, item in enumerate(frames, 1):
        if k % step == 0:
            progress(f"[進行] シャトルを検出中 {min(100, round(100 * k / n_total))}%")
        yield item


def detect_all(meta: VideoMeta, cfg: dict[str, Any], start_s: float | None = None, end_s: float | None = None,
               roi: list[float] | None = None, n_tracks: int | None = None, exclusive: bool = True,
               progress: Callable[[str], None] = print) -> pd.DataFrame:
    """動画の区間の全コマでシャトルを検出する（座標は元の解像度）。

    ``exclusive`` が真なら 1 コマ 1 点（スコアの高い軌道を優先）。偽なら重なった軌道もすべて残す
    （``shuttle_tracks`` で絞り込んでから 1 コマ 1 点にする）。
    """
    from smash_speed.batch import stream_gray

    p = BaselineParams.from_config(cfg)
    if roi:
        p.roi = list(roi)
    scale = min(1.0, p.process_width / meta.width)
    ts = meta.timestamps
    lo = 0.0 if start_s is None else start_s
    hi = float(ts[-1]) if end_s is None else end_s
    n_total = max(1, int(np.sum((ts >= lo) & (ts <= hi))))
    frames = stream_gray(meta, int(round(meta.width * scale)), cfg, start_s, end_s)
    return detect_frames(_with_progress(frames, n_total, progress), scale, p, n_tracks=n_tracks, exclusive=exclusive)


def load_detections(path: Path, meta: VideoMeta, start_s: float | None = None, end_s: float | None = None
                    ) -> pd.DataFrame:
    """外の検出 CSV を、``detect_all(exclusive=False)`` と同じ形（列 ``frame, timestamp, x_px, y_px, visible,
    confidence, track_id``）にして、区間で切る。

    時刻はコマの番号から付け直す（``tracknet_train`` は撮影の fps を知らないので、スロー動画では時刻が違う）。
    ``track_id`` がなければ、見えている点を 1 本の軌道にする。
    """
    det = pd.read_csv(path)
    missing = {"frame", "x_px", "y_px", "visible"} - set(det.columns)
    if missing:
        raise AnalysisError(f"検出のファイル {Path(path).name} に列 {', '.join(sorted(missing))} がありません。",
                            ["学習したモデルの検出（tracknet_train detect）の detections.csv を選んでください。"])
    ts = meta.timestamps
    det = det[(det.frame >= 0) & (det.frame < len(ts))].copy()
    det["frame"] = det.frame.astype(int)
    det["timestamp"] = ts[det.frame.to_numpy()]
    det["visible"] = det.visible.fillna(0).astype(int)
    if "confidence" not in det:
        det["confidence"] = det.visible.astype(float)
    if "track_id" not in det:
        det["track_id"] = np.where(det.visible > 0, 0, -1)
    det["track_id"] = det.track_id.fillna(-1).astype(int)
    lo = -np.inf if start_s is None else start_s
    hi = np.inf if end_s is None else end_s
    det = det[(det.timestamp >= lo - 1e-6) & (det.timestamp <= hi + 1e-6)]
    cols = ["frame", "timestamp", "x_px", "y_px", "visible", "confidence", "track_id"]
    return det[cols + [c for c in det.columns if c not in cols]].sort_values("frame", kind="stable"
                                                                             ).reset_index(drop=True)


def tracknet_detections(o: ShotListOptions, out: Path, progress: Callable[[str], None] = print) -> Path:
    """学習したモデル（``o.tracknet``）で区間の全コマを検出し、``<出力先>/tracknet/detections.csv`` のパスを返す。"""
    from smash_speed.train.tracknet_train import detect_video

    if not Path(o.tracknet).exists():
        raise AnalysisError(f"学習したモデル {o.tracknet} が見つかりません。",
                            ["docs/gpu.md の「エージェントの正解での学習」の手順で作った model.pt を選んでください。"])
    progress("[進行] 学習したモデルでシャトルを検出しています")
    d = out / TRACKNET_DIR
    detect_video(o.video, o.tracknet, d, o.start_s, o.end_s, o.tracknet_threshold,
                 progress=lambda m: progress(f"[進行] {m}"))
    return d / "detections.csv"


# =============================================================================
# シャトルらしい軌道の絞り込み
# =============================================================================
@dataclass
class TrackFilter:
    """シャトルらしい軌道を残す条件（実動画では、選手・ラケット・ほかのコートの人の動きも軌道になる）。

    速さ・広がりは画像の幅を単位にする（解像度によらない）。「コートの空間」は、コートの外側の線から
    ``margin_m`` だけ広げた範囲の、床から ``max_height_m`` までの箱。
    """

    min_points: int = 5               # 軌道の点の数
    min_duration_s: float = 0.13      # 軌道の長さ [s]
    max_image_speed: float = 1.2      # 画像の上の速さ（1 コマごとの移動の中央値）の上限 [画像の幅/s]。
                                      # 超えるものは、別々の物の間を飛び移った軌道（実動画で 1.5〜6）
    min_extent: float = 0.03          # 軌道の点の広がり（外接する四角の対角線）[画像の幅]。止まっている物を落とす
    min_image_speed: float = 0.1      # 画像の上の速さ（1 コマごとの移動の中央値）の下限 [画像の幅/s]。ゆっくり動く物
                                      # （手に持って歩くシャトル・人の動き。IMG_4397 の 97〜99.5 s で 0.07〜0.08）を落とす。
                                      # 飛んでいるシャトルは、ゆっくり落ちる軌道でも 0.15 ほど以上（練習の動画 3 本と IMG_4397）
    min_inside: float = 0.8           # 視線がコートの空間を通る点の割合
    min_speed_mps: float = 3.0        # コートの空間のいちばん奥にあるとしたときの速さの上限 [m/s] がこれより遅い
                                      # 軌道を落とす（人の歩き・手足の動き。飛んでいるシャトルは頂点の近くでも 3 m/s 以上）
    margin_m: tuple[float, float] = (2.0, 1.5)  # コートの空間の、長さ方向・幅方向の広げ幅 [m]
    max_height_m: float = 12.0
    # 残った軌道どうしが同じ時間に重なったら（隣のコートのシャトルなど）、重ならない組で点の数の合計が
    # 最も多いものを選ぶ（``select_tracks``）
    max_overlap_s: float = 0.07       # これまでの重なりは、同じシャトルの続きとして許す [s]
    edge_margin: float = 0.02         # 画面の左右の端とみなす幅 [画像の幅]
    edge_start_weight: float = 0.5    # 画面の左右の端から始まった軌道の点の重み（ラリーのシャトルは打点か画面の上から
                                      # 入ってくる。左右の端から入ってくるのは、ほとんど隣のコートのシャトル）


def court_depth_range(camera: Camera, uv: np.ndarray, cfg: dict[str, Any], flt: TrackFilter
                      ) -> tuple[np.ndarray, np.ndarray]:
    """画像の点の視線がコートの空間を通る区間（カメラからの距離 [m] の始まりと終わり）。通らなければ 始まり ≥ 終わり。"""
    c = cfg.get("court", {})
    half = (c.get("half_length_m", 6.7) + flt.margin_m[0], c.get("half_width_doubles_m", 3.05) + flt.margin_m[1])
    o, d = camera.rays(np.asarray(uv, float).reshape(-1, 2))
    lo, hi = np.zeros(len(d)), np.full(len(d), 1e3)
    for ax, a, b in ((0, -half[0], half[0]), (1, -half[1], half[1]), (2, 0.0, flt.max_height_m)):
        dd = np.where(np.abs(d[:, ax]) < 1e-12, 1e-12, d[:, ax])
        s1, s2 = (a - o[ax]) / dd, (b - o[ax]) / dd
        lo, hi = np.maximum(lo, np.minimum(s1, s2)), np.minimum(hi, np.maximum(s1, s2))
    return lo, hi


def track_table(det: pd.DataFrame, camera: Camera, cfg: dict[str, Any], flt: TrackFilter | None = None
                ) -> pd.DataFrame:
    """軌道ごとの手がかり（点の数・長さ・画像の上の速さと広がり・コートの空間を通る割合・速さの上限）。"""
    flt = flt or TrackFilter()
    W, f = float(camera.width), float(camera.K[0, 0])
    rows = []
    vis = det[(det.visible > 0) & det.x_px.notna()]
    for tid, g in vis.groupby("track_id"):
        g = g.sort_values("timestamp")
        t, uv = g.timestamp.to_numpy(float), g[["x_px", "y_px"]].to_numpy(float)
        lo, hi = court_depth_range(camera, uv, cfg, flt)
        inside = hi > lo
        speed = v_max = 0.0
        if len(t) >= 2:
            step = np.linalg.norm(np.diff(uv, axis=0), axis=1) / np.maximum(np.diff(t), 1e-9)
            speed = float(np.median(step)) / W
            far = np.where(inside, hi, np.nan)
            if inside.any():
                # 隣り合う 2 点のどちらかがコートの空間にあれば、その奥の端までの距離で角速度を速さにする
                v_max = float(np.nanmedian(step / f * np.fmax(far[1:], far[:-1])))
        u0 = float(uv[0, 0]) / W
        rows.append({"track_id": int(tid), "n": len(t), "t_start": float(t[0]), "t_end": float(t[-1]),
                     "duration_s": float(t[-1] - t[0]), "image_speed": speed,
                     "extent": float(np.linalg.norm(uv.max(0) - uv.min(0))) / W, "inside": float(inside.mean()),
                     "speed_max_mps": v_max, "edge_start": bool(min(u0, 1.0 - u0) < flt.edge_margin)})
    tab = pd.DataFrame(rows, columns=["track_id", "n", "t_start", "t_end", "duration_s", "image_speed", "extent",
                                      "inside", "speed_max_mps", "edge_start"])
    tab["keep"] = ((tab.n >= flt.min_points) & (tab.duration_s >= flt.min_duration_s)
                   & (tab.image_speed <= flt.max_image_speed) & (tab.image_speed >= flt.min_image_speed)
                   & (tab.extent >= flt.min_extent)
                   & (tab.inside >= flt.min_inside) & (tab.speed_max_mps >= flt.min_speed_mps))
    return tab


def select_tracks(tab: pd.DataFrame, flt: TrackFilter | None = None) -> pd.Series:
    """``keep`` の軌道から、同じ時間に重ならない組で、点の数（左右の端から始まった軌道は軽くする）の合計が
    最も多いものを選ぶ（重み付きの区間スケジューリング）。``tab`` と同じ並びの真偽の列を返す。

    確からしさの高い方を 1 コマずつ選ぶと、隣のコートのシャトルの軌道が、打球の前後のラリーのシャトルの点を
    数コマだけ奪い、打球の時刻と打った側を間違える（IMG_4397 の 82.1・94.9 s のスマッシュ）。
    """
    flt = flt or TrackFilter()
    out = pd.Series(False, index=tab.index)
    k = tab[tab.keep].sort_values(["t_end", "t_start"], kind="stable")
    if k.empty:
        return out
    t0, t1 = k.t_start.to_numpy(float), k.t_end.to_numpy(float)
    w = k.n.to_numpy(float) * np.where(k.edge_start.to_numpy(bool), flt.edge_start_weight, 1.0)
    # prev[j]：j の前に置ける（終わりが j の始まり + 許す重なり 以下の）最後の軌道の番号 + 1
    prev = np.searchsorted(t1, t0 + flt.max_overlap_s + 1e-9, side="right")
    best = np.zeros(len(k) + 1)
    for j in range(len(k)):
        best[j + 1] = max(best[j], w[j] + best[min(prev[j], j)])
    j = len(k)
    while j > 0:
        if best[j] == best[j - 1]:
            j -= 1
            continue
        out[k.index[j - 1]] = True
        j = min(prev[j - 1], j - 1)
    return out


def one_per_frame(det: pd.DataFrame, keep: pd.Series | None = None) -> pd.DataFrame:
    """1 コマ 1 点にする（確からしさ ``confidence`` の高い軌道を優先）。

    ``keep`` があれば、その行の点だけを使う。点のないコマは見えないコマとして残す。
    """
    frames = det[["frame", "timestamp"]].drop_duplicates("frame")
    vis = det[(det.visible > 0) & (True if keep is None else keep)].sort_values(["frame", "confidence"], ascending=[True, False], kind="stable")
    vis = vis.drop_duplicates("frame").drop(columns="timestamp")
    out = frames.merge(vis, on="frame", how="left")
    out["visible"] = out["visible"].fillna(0).astype(int)
    out["confidence"] = out["confidence"].fillna(0.0)
    out["track_id"] = out["track_id"].fillna(-1).astype(int)
    return out.sort_values("frame", kind="stable").reset_index(drop=True)[det.columns]


def shuttle_tracks(det: pd.DataFrame, camera: Camera, cfg: dict[str, Any], flt: TrackFilter | None = None
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """重なった軌道を含む検出から、シャトルらしい軌道だけを残し（``track_table``）、同じ時間に重なったものから
    選んで（``select_tracks``）1 コマ 1 点にする。

    Returns:
        (絞り込んだ検出, ``track_table`` の表。列 ``selected`` は選んだ軌道)
    """
    tab = track_table(det, camera, cfg, flt)
    tab["selected"] = select_tracks(tab, flt)
    return one_per_frame(det, det.track_id.isin(tab.track_id[tab.selected])), tab


def hit_onsets(meta: VideoMeta, cfg: dict[str, Any], out_dir: Path, audio: Path | None = None,
               start_s: float | None = None, end_s: float | None = None) -> list[tuple[float, float]] | None:
    """打球音の候補（映像の時刻 [s]、強さ）。音声がなければ None。"""
    if audio is not None:
        x, rate = load_wav(audio)
    elif meta.has_audio:
        wav = out_dir / "audio.wav"
        extract_audio(meta.path, wav, cfg=cfg)
        x, rate = load_wav(wav)
    else:
        return None
    ev = cfg.get("events", {})
    ons = [(t * meta.time_scale, s) for t, s in events.audio_onsets(
        x, rate, ev.get("audio_highpass_hz", 2000.0), k_mad=ev.get("audio_k_mad", 10.0))]
    lo = -np.inf if start_s is None else start_s
    hi = np.inf if end_s is None else end_s
    return [(t, s) for t, s in ons if lo <= t <= hi]


# =============================================================================
# ラリーごとの当てはめと種類
# =============================================================================
def rally_groups(hits: list[Hit], gap_s: float) -> list[list[int]]:
    """打球の間隔が ``gap_s`` を超えた所で分けた、打球の番号の組。"""
    groups: list[list[int]] = []
    for i, h in enumerate(hits):
        if not groups or h.t - hits[i - 1].t > gap_s:
            groups.append([])
        groups[-1].append(i)
    return groups


def analyze_hits(det: pd.DataFrame, hits: list[Hit], camera: Camera, phys: PhysicsParams, cfg: dict[str, Any],
                 rally_gap_s: float = 3.0, tail_s: float = 4.0, ci: bool = True, max_kmh: float = 600.0,
                 max_rms_frac: float = 0.01, workers: int = 1, progress: Callable[[str], None] = print,
                 players: pd.DataFrame | None = None, rally_threads: int | None = None, joint: JointOptions | None = None,
                 joint_info: dict[str, Any] | None = None, fit_backend: str = "cpu",
                 batch_processes: int | None = None) -> list[Shot]:
    """打球ごとに軌道と初速を当てはめ、種類を決めた ``Shot`` の列（時刻の順。ラリーの番号つき）。

    Args:
        det: 動画全体の検出結果
        hits: ``find_hits`` の結果（時刻の順）
        tail_s: ラリーの最後の打球の後、これだけの間の検出までを最後のショットに使う [s]
        ci: 初速の信頼区間を出す（遅くなるが、信頼度に使う）
        max_kmh: これより速い初速は、当てはめに失敗したとみなす（最速のスマッシュは 565 km/h）[km/h]
        max_rms_frac: 画像の上のずれ（RMS）が画像の幅のこの割合を超えたら、当てはめに失敗したとみなす
            （4K で 38 px。合成ラリーでは 1〜2 px、実動画の正しい当てはめも数 px）
        workers: 当てはめを並べて動かすプロセスの数（``fit_shots``）
        players: 選手の位置（``players.csv``）。あれば打球ごとの打つ人の足元（``hitter_at``）を打点の手がかりにする
        rally_threads: 同時に当てはめを進めるラリーの数（``workers`` が 2 以上のとき。結果は 1 つずつと同じ）。
            None なら 4（``fit_backend`` が "batch" なら ``BATCH_RALLIES``）
        joint: 与えると、ショットごとの当てはめの後に、打球の組ごとにショットをつないでまとめて当てはめ直す
            （``rally_fit.fit_rallies``。終端速度は動画全体で 1 つ）
        joint_info: 辞書を渡すと、まとめた当てはめの記録（つないだショットの数・終端速度・かかった時間）を入れる
        fit_backend: 当てはめの計算の仕方（``shot_fit.resolve_backend``）。"batch" では、ラリーを打球の数がなるべく
            同じ ``batch_processes`` 組に分けてプロセスごとに進め、それぞれの中でラリーを ``rally_threads`` 個ずつ
            同時に進めて、段階ごとの当てはめをほかのラリーの分とまとめて解く（``_fit_groups``・``fit_batch.BatchGate``）。
            結果はラリーの組み方・プロセスの数によらない
        batch_processes: "batch" で使うプロセスの数（None なら ``BATCH_PROCESSES``。``workers`` まで）
    """
    opts = FitOptions.from_config(cfg, **({} if ci else {"ci_method": "none"}))
    backend = resolve_backend(fit_backend)
    vis_t = det.timestamp.to_numpy(float)
    groups = rally_groups(hits, rally_gap_s)
    shots: list[Shot] = []

    if rally_threads is None:
        rally_threads = BATCH_RALLIES if backend == "batch" else 4
    fit_args = (det, hits, groups, vis_t, camera, phys, opts, cfg, tail_s, max_kmh, max_rms_frac, workers, players)

    def one(g: int) -> tuple[list[ShotFit], dict[str, Any]]:
        return _fit_rally(det, hits, groups, g, vis_t, camera, phys, opts, cfg, tail_s, max_kmh, max_rms_frac,
                          workers, players, backend)

    def to_shots(g: int, fits: list[ShotFit]) -> list[Shot]:
        return _rally_shots([hits[i] for i in groups[g]], fits, camera, phys, max_kmh, max_rms_frac)

    # 当てはめのプロセスはラリーをまたいで使い回す（起動に 1 回 4〜5 秒かかる。ROADMAP F3）。
    # ラリーどうしは関係しないので、``rally_threads`` 個のラリーを同時に進めて同じプロセスの組に仕事を渡す
    # （1 つのラリーの中では段階ごとに全部の当てはめの終わりを待つので、その間ほかのプロセスが空いていた。F3b）
    # まとめて解くときは、ラリーを ``batch_processes`` 組に分けてプロセスごとに進め、それぞれの中で
    # ``rally_threads`` 個ずつ同時に進めて当てはめをまとめる（``_fit_groups``）
    n_proc = max(1, min(batch_processes or BATCH_PROCESSES, workers, len(groups))) if backend == "batch" else 1
    pool_workers = workers if backend == "cpu" else (max(n_proc, workers) if joint is not None else n_proc)
    with worker_pool(pool_workers):
        n_threads = min(rally_threads, len(groups)) if workers > 1 and pool_active() else 1
        fitted: dict[int, tuple[list[ShotFit], dict[str, Any]]] = {}
        by_group: dict[int, list[Shot]] = {}
        if backend == "batch":
            chunks = _balanced_chunks([len(g) for g in groups], n_proc if pool_active() else 1)

            def done(part: dict[int, tuple[list[ShotFit], dict[str, Any]]]) -> None:
                for g in sorted(part):
                    fitted[g] = part[g]
                    by_group[g] = to_shots(g, part[g][0])
                    _rally_progress(progress, g, len(groups), by_group[g], len(by_group))

            left = chunks
            if len(chunks) > 1:
                from concurrent.futures import as_completed
                from concurrent.futures.process import BrokenProcessPool

                got: set[int] = set()
                try:
                    with worker_env():
                        futs = {pool_executor().submit(_fit_groups, fit_args, c, rally_threads): k
                                for k, c in enumerate(chunks)}
                    for fut in as_completed(futs):
                        done(fut.result())
                        got.add(futs[fut])
                    left = []
                except (BrokenProcessPool, OSError):  # プロセスを使えなければ、残りをこのプロセスで
                    left = [c for k, c in enumerate(chunks) if k not in got]
            for c in left:
                done(_fit_groups(fit_args, c, rally_threads))
        elif n_threads > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(n_threads) as ex:
                futs = {ex.submit(one, g): g for g in range(len(groups))}
                for fut in as_completed(futs):
                    g = futs[fut]
                    fitted[g] = fut.result()
                    by_group[g] = to_shots(g, fitted[g][0])
                    _rally_progress(progress, g, len(groups), by_group[g], len(by_group))
        else:
            for g in range(len(groups)):
                fitted[g] = one(g)
                by_group[g] = to_shots(g, fitted[g][0])
                _rally_progress(progress, g, len(groups), by_group[g], g + 1)
        if joint is not None and groups:
            t0 = time.time()
            n_before = sum(s.speed_kmh is not None for g in by_group for s in by_group[g])
            res, vt = fit_rallies([fitted[g][1] for g in range(len(groups))], phys, opts, joint,
                                  mapper=lambda fn, args: _parallel_map(fn, args, workers))
            for g, r in enumerate(res):
                by_group[g] = to_shots(g, r.fits)
            rec = {"n_joint": int(sum(sum(r.joint) for r in res)),
                   "n_links": int(sum(sum(r.links) for r in res)),
                   "n_cut": int(sum(len(r.cut) for r in res)), "terminal_velocity": vt,
                   "n_speed_before": n_before,
                   "n_speed": sum(s.speed_kmh is not None for g in by_group for s in by_group[g]),
                   "elapsed_s": round(time.time() - t0, 1)}
            if joint_info is not None:
                joint_info.update(rec)
            progress(f"[進行] ラリーの中のショットをつないで当てはめ直しました：{rec['n_joint']} 本"
                     f"（{rec['elapsed_s']:.0f} 秒）")
            if vt["value_mps"] is not None:
                progress(f"[進行] シャトルの終端速度：{vt['value_mps']:.2f} ± {vt['sigma_mps']:.2f} m/s"
                         + ("（この値で計算しました）" if vt["used"]
                            else f"（{vt['reason']}ので、設定の {vt['default_mps']} m/s のまま）"))
    for g in range(len(groups)):
        shots += by_group[g]
    split_rallies(shots, det, RallyOptions(max_gap_s=rally_gap_s), cfg.get("court"))
    return classify_shots(shots, phys)


def _balanced_chunks(sizes: list[int], n: int) -> list[list[int]]:
    """番号 0..len(sizes)−1 を、大きさ（打球の数）の合計がなるべく同じ ``n`` 組に分ける（大きい順に、いちばん軽い組へ）。
    組の中は番号の順。空の組は返さない。"""
    load = [0] * max(1, n)
    out: list[list[int]] = [[] for _ in load]
    for g in sorted(range(len(sizes)), key=lambda g: (-sizes[g], g)):
        k = int(np.argmin(load))
        out[k].append(g)
        load[k] += sizes[g]
    return [sorted(c) for c in out if c]


def _fit_groups(fit_args: tuple, gids: list[int], rally_threads: int
                ) -> dict[int, tuple[list[ShotFit], dict[str, Any]]]:
    """ラリー ``gids`` を、``rally_threads`` 個ずつ同時に進め、当てはめをまとめて解く（``fit_backend="batch"``。
    ``analyze_hits`` の中の 1 組。別のプロセスでも動かす）。結果はラリーの組み方・スレッドの数によらない。"""
    det, hits, groups, vis_t, camera, phys, opts, cfg, tail_s, max_kmh, max_rms_frac, workers, players = fit_args
    n_threads = max(1, min(rally_threads, len(gids)))
    out: dict[int, tuple[list[ShotFit], dict[str, Any]]] = {}
    with gate_session(n_threads, len(gids)) as gate:
        def one(g: int) -> tuple[list[ShotFit], dict[str, Any]]:
            gate.enter()        # 同時に進めているほかのラリーの当てはめとまとめて解く（``fit_batch.BatchGate``）
            try:
                return _fit_rally(det, hits, groups, g, vis_t, camera, phys, opts, cfg, tail_s, max_kmh,
                                  max_rms_frac, workers, players, "batch")
            finally:
                gate.leave()

        if n_threads > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(n_threads) as ex:
                for g, r in zip(gids, ex.map(one, gids)):
                    out[g] = r
        else:
            for g in gids:
                out[g] = one(g)
    return out


def _fit_rally(det: pd.DataFrame, hits: list[Hit], groups: list[list[int]], g: int, vis_t: np.ndarray,
               camera: Camera, phys: PhysicsParams, opts: FitOptions, cfg: dict[str, Any], tail_s: float,
               max_kmh: float, max_rms_frac: float, workers: int, players: pd.DataFrame | None,
               backend: str = "cpu") -> tuple[list[ShotFit], dict[str, Any]]:
    """``analyze_hits`` の 1 ラリー分（打球の組 ``groups[g]``）のショットごとの当てはめと、まとめて当てはめ直すときの
    引数（``shot_fit.joint_job``）。"""
    hs = [hits[i] for i in groups[g]]
    lo = hs[0].t - 0.5
    hi = hs[-1].t + tail_s
    if g + 1 < len(groups):
        hi = min(hi, hits[groups[g + 1][0]].t - 0.05)
    part = det[(vis_t >= lo) & (vis_t <= hi)]
    feet = None
    if players is not None and not players.empty:
        feet = [None if e is None else np.array([e.x, e.y])
                for e in (hitter_at(players, h.t, h.side) for h in hs)]
    sound = cfg.get("events", {}).get("speed_of_sound_mps", 343.0)
    fits = fit_shots(part, [h.t for h in hs], camera, phys, [h.side for h in hs], opts, workers=workers,
                     max_rms_px=max_rms_frac * camera.width, max_kmh=max_kmh, t_audio=[h.t_audio for h in hs],
                     speed_of_sound_mps=sound, hitter_xy=feet, backend=backend)
    return fits, joint_job(fits, part, camera, feet, [h.t_audio for h in hs], speed_of_sound_mps=sound,
                           max_rms_px=max_rms_frac * camera.width)


def _rally_shots(hs: list[Hit], fits: list[ShotFit], camera: Camera, phys: PhysicsParams, max_kmh: float,
                 max_rms_frac: float) -> list[Shot]:
    """1 ラリー分の当てはめ ``fits``（打球 ``hs`` と同じ順）を ``Shot`` の列にする。"""
    rally_shots = shot_fits_to_shots(fits, phys, source="shot_list")
    for s, h, sf in zip(rally_shots, hs, fits):
        # ありえない速さ・画像の上で軌道が検出と合わない（別の物の検出が混ざった軌道など）ものは、
        # 速さが出なかったものとして扱う
        why = None
        if s.speed_kmh is not None and s.speed_kmh > max_kmh:
            why = f"ありえない速さ {s.speed_kmh:.0f} km/h"
        elif sf.fit is not None and sf.fit.rms_px > max_rms_frac * camera.width:
            why = "軌道が検出と合わない"
        if why:
            s.note = f"{why}・{s.note}"
            s.speed_kmh = s.speed_ci_low_kmh = s.speed_ci_high_kmh = s.confidence = None
            s.v0_x_mps = s.v0_y_mps = s.v0_z_mps = None
        s.note = f"{h.method}・{s.note}" if s.note else h.method
        if s.confidence is None:
            s.confidence = round(h.confidence, 3)  # 初速の信頼区間がなければ、打球の見つけ方の信頼度
    return rally_shots


def _rally_progress(progress: Callable[[str], None], g: int, n_groups: int, shots: list[Shot], n_done: int) -> None:
    """1 ラリーの当てはめが終わったことを知らせる（並べて動かすと、終わる順はラリーの順と違う）。"""
    n_ok = sum(s.speed_kmh is not None for s in shots)
    progress(f"[進行] ラリー {g + 1}/{n_groups}：{len(shots)} 本（速さが出たもの {n_ok} 本、済み {n_done}/{n_groups}）")


def recorded_at(meta: VideoMeta) -> str | None:
    """動画を撮った日時（ISO 8601。動画のタグ ``creation_time`` など。なければ None）。日付ごとの記録（``history``）に使う。"""
    for k in ("com.apple.quicktime.creationdate", "creation_time"):
        if meta.tags.get(k):
            return str(meta.tags[k])
    return None


def summarize(shots: list[Shot]) -> dict[str, Any]:
    """種類ごとの本数・平均と最高の初速 [km/h]。"""
    out: dict[str, Any] = {}
    for k in SHOT_TYPES:
        mine = [s for s in shots if s.shot_type == k]
        if not mine:
            continue
        v = [s.speed_kmh for s in mine if s.speed_kmh is not None]
        out[k] = {"label": SHOT_TYPE_LABELS[k], "n": len(mine),
                  "mean_kmh": round(float(np.mean(v)), 1) if v else None,
                  "max_kmh": round(float(np.max(v)), 1) if v else None}
    return out


# =============================================================================
# 動画 1 本を解析する
# =============================================================================
def analyze_video(o: ShotListOptions, cfg: dict[str, Any], progress: Callable[[str], None] = print) -> dict[str, Any]:
    """動画 1 本のすべてのショットを解析し、``<出力先>/`` に書き出す。``shot_list.json`` の中身を返す。"""
    from smash_speed.pipeline import _camera_for_video

    t_start = time.time()
    timings: dict[str, float] = {}    # 段階ごとにかかった時間 [s]（F3 の速さの計測。``shot_list.json`` の ``timings_s``）
    t_mark = [t_start]

    def lap(name: str) -> None:
        now = time.time()
        timings[name] = round(timings.get(name, 0.0) + now - t_mark[0], 2)
        t_mark[0] = now

    out = Path(o.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    meta = probe(o.video, o.capture_fps, cfg)
    camera, _ = load_calibration(o.calib)
    camera = _camera_for_video(camera, meta)
    phys = PhysicsParams.from_config(cfg)
    lap("probe")

    onsets = hit_onsets(meta, cfg, out, o.audio, o.start_s, o.end_s) if o.use_audio else None
    if onsets is not None:
        progress(f"[進行] 打球音の候補 {len(onsets)} 個")
    lap("audio")
    det_path = tracknet_detections(o, out, progress) if o.tracknet is not None else o.detections
    if det_path is not None:
        det = load_detections(det_path, meta, o.start_s, o.end_s)
    else:
        det = detect_all(meta, cfg, o.start_s, o.end_s, o.roi, o.n_tracks, exclusive=False, progress=progress)
    lap("shuttle_detect")
    tracks = None
    if o.filter_tracks:
        det, tracks = shuttle_tracks(det, camera, cfg)
        progress(f"[進行] シャトルらしい軌道 {int(tracks.selected.sum())} / {len(tracks)} 本"
                 f"（重なりを除く前 {int(tracks.keep.sum())} 本）")
    else:
        det = one_per_frame(det)
    det.to_csv(out / DETECTIONS_FILE, index=False)
    n_vis = int(det.visible.sum())
    if n_vis < 5:
        raise AnalysisError("シャトルがほとんど見つかりませんでした。",
                            ["コートとシャトルが写っている動画か、検出する範囲が正しいかを確かめてください。"])
    progress(f"[進行] シャトルが見えたコマ {n_vis} / {len(det)}")
    lap("shuttle_tracks")

    # 選手の位置は打球を見つける前に出し、打った時刻にシャトルに届く選手から打球と打った側を確かめる
    # （``find_hits`` の ``players``）。当てはめでは打つ人の足元を打点の手がかりにする（``fit_shots`` の ``hitter_xy``）
    players_info = players = None
    if o.players:
        players_info = players_step(o, cfg, out, progress)
        if "error" not in players_info and (out / PLAYERS_FILE).exists():
            players = pd.read_csv(out / PLAYERS_FILE)
        lap("players")
    hits = find_hits(det, meta.capture_fps, onsets, camera=camera, rally_gap_s=o.rally_gap_s,
                     players=players if o.hit_players else None)
    progress(f"[進行] 打球 {len(hits)} 本")
    lap("hits")
    workers = o.workers or max(1, min(16, (os.cpu_count() or 2) - 1))
    joint_info: dict[str, Any] = {}
    shots = analyze_hits(det, hits, camera, phys, cfg, o.rally_gap_s, workers=workers, progress=progress,
                         players=players if o.hitter_prior else None,
                         joint=(o.joint_options or JointOptions()) if o.joint_fit else None, joint_info=joint_info,
                         fit_backend=o.fit_backend)
    lap("fit")
    rallies = split_rallies(shots, det, RallyOptions(max_gap_s=o.rally_gap_s), cfg.get("court"))
    write_shots(out / SHOTS_FILE, shots)
    write_rallies(out / RALLIES_FILE, rallies)

    info = {
        "video": str(Path(o.video).resolve()), "calib": str(Path(o.calib).resolve()),
        "capture_fps": o.capture_fps, "fps": meta.capture_fps, "width": meta.width, "height": meta.height,
        "start_s": o.start_s, "end_s": o.end_s, "roi": o.roi, "recorded_at": recorded_at(meta),
        "audio": str(Path(o.audio).resolve()) if o.audio else None,
        "detector": "tracknet" if o.tracknet is not None else ("file" if o.detections is not None else "baseline"),
        "detections": str(Path(det_path).resolve()) if det_path is not None else None,
        "tracknet_threshold": o.tracknet_threshold if o.tracknet is not None else None,
        "used_audio": onsets is not None, "n_onsets": None if onsets is None else len(onsets),
        "n_frames": len(det), "n_visible": n_vis,
        "n_tracks": None if tracks is None else len(tracks),
        "n_tracks_kept": None if tracks is None else int(tracks.keep.sum()),
        "n_tracks_selected": None if tracks is None else int(tracks.selected.sum()),
        "n_hits": len(hits), "n_rallies": len(rallies),
        "hit_methods": pd.Series([h.method for h in hits], dtype=object).value_counts().to_dict(),
        "by_type": summarize(shots), "elapsed_s": round(time.time() - t_start, 1),
        "joint_fit": joint_info if o.joint_fit else None,
        "fit_backend": resolve_backend(o.fit_backend),   # 当てはめの計算の仕方（cpu・batch。ROADMAP F3e）
        "meta": run_metadata(),
    }
    if players_info is not None:
        info["players"] = players_info
        if players is not None:
            n = attach_hitters(shots, players)
            write_shots(out / SHOTS_FILE, shots)
            info["players"]["n_hitters"] = n
            info["players"]["hitter_prior"] = o.hitter_prior
            progress(f"[進行] 打つ人の足元：{n} / {len(shots)} 本")
        rallies = rallies_with_players(shots, rallies, det, o, cfg, out, info, progress)
    info["elapsed_s"] = round(time.time() - t_start, 1)
    # レポートは shot_list.json を読むので先に書き、書き出しの時間を足して書き直す
    (out / SHOT_LIST_FILE).write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    write_stats(out, shots, rallies, cfg.get("court"))
    write_report(out, cfg.get("court"), shots, rallies)
    lap("write")
    info["elapsed_s"] = round(time.time() - t_start, 1)
    info["timings_s"] = timings
    (out / SHOT_LIST_FILE).write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    progress(f"[進行] 完了：{len(shots)} 本（{len(rallies)} ラリー）、{info['elapsed_s']:.0f} 秒")
    return info


def rallies_with_players(shots: list[Shot], rallies: list[Rally], det: pd.DataFrame, o: ShotListOptions,
                         cfg: dict[str, Any], out: Path, info: dict[str, Any],
                         progress: Callable[[str], None] = print) -> list[Rally]:
    """選手の位置（``players.csv``）が出たら、ラリーを区切り直して短いラリーを選手の動きで確かめる
    （``rallies.split_rallies`` の ``players``）。ラリーが変わったら ``shots.csv``・``rallies.csv`` を書き直し、``info`` の数を直す。
    ラリーごとの移動距離（``movement.json`` など）は、決まったラリーでいつも書き直す。選手の位置がなければ ``rallies`` をそのまま返す。
    """
    from smash_speed.movement import movement_brief, write_movement

    if "error" in info.get("players", {}) or not (out / PLAYERS_FILE).exists():
        return rallies
    players = pd.read_csv(out / PLAYERS_FILE)
    new = split_rallies(shots, det, RallyOptions(max_gap_s=o.rally_gap_s), cfg.get("court"), players=players)
    n_out = sum(s.rally_id is None for s in shots)
    info["players"]["n_shots_outside_rallies"] = n_out
    if n_out:
        progress(f"[進行] 選手の動きから、ラリーではない打球 {n_out} 本をラリーから外しました（{len(rallies)} → {len(new)} ラリー）")
        write_shots(out / SHOTS_FILE, shots)
        write_rallies(out / RALLIES_FILE, new)
        info["n_rallies"] = len(new)
    # 選手の位置は当てはめの前（この解析のラリーが出る前）に出すので、ラリーごとの移動距離はここで、決まったラリーで書き直す
    if "movement" in info["players"]:
        info["players"]["movement"] = movement_brief(write_movement(out, players, cfg.get("court"), new))
    return new


def players_step(o: ShotListOptions, cfg: dict[str, Any], out: Path, progress: Callable[[str], None] = print
                 ) -> dict[str, Any]:
    """選手の位置を ``players.csv`` に書き、``shot_list.json`` に入れる要約を返す。

    ショットの一覧は選手がいなくても出せるので、モデルが取れない・動かないときは一覧を止めず、``error`` に理由を残す。
    """
    progress("[進行] 選手を検出しています")
    try:
        p = analyze_players(o.video, o.calib, out, cfg, o.player_options, o.start_s, o.end_s, o.capture_fps,
                            progress=progress)
    except Exception as e:  # noqa: BLE001  （モデルの取得の失敗・onnxruntime の失敗など）
        progress(f"[注意] 選手の位置は出せませんでした：{e}")
        return {"error": str(e)}
    progress(f"[進行] 選手の位置：手前 {p['coverage']['h']:.0%}・奥 {p['coverage']['o']:.0%} のコマ")
    return {k: p[k] for k in ("n_frames", "coverage", "provider", "detect_s", "movement") if k in p}


def main(argv: list[str] | None = None) -> None:
    import argparse

    from smash_speed.config import load_config

    ap = argparse.ArgumentParser(description="動画 1 本のすべてのショットを、種類・初速つきで一覧にする")
    ap.add_argument("video", type=Path)
    ap.add_argument("--calib", type=Path, required=True, help="calibration.json")
    ap.add_argument("--out", type=Path, required=True, help="出力先のフォルダ")
    ap.add_argument("--capture-fps", type=float, default=None, help="実際の撮影 fps（スロー動画のとき）")
    ap.add_argument("--start", type=float, default=None, help="解析する区間の始まり [s]")
    ap.add_argument("--end", type=float, default=None, help="解析する区間の終わり [s]")
    ap.add_argument("--audio", type=Path, default=None, help="別の音声ファイル（WAV）")
    ap.add_argument("--no-audio", action="store_true", help="打球音を使わない")
    ap.add_argument("--all-tracks", action="store_true", help="シャトルらしい軌道に絞り込まない")
    ap.add_argument("--players", action="store_true", help="選手の位置も出す（players.csv）")
    ap.add_argument("--no-hitter-prior", action="store_true", help="打つ人の足元を打点の手がかりにしない")
    ap.add_argument("--no-hit-players", action="store_true", help="打球を見つけるのに選手の位置を使わない")
    ap.add_argument("--detections", type=Path, default=None,
                    help="自前の検出の代わりに使う検出 CSV（tracknet_train detect の detections.csv など）")
    ap.add_argument("--tracknet", type=Path, default=None, help="学習したモデル（model.pt）で全コマを検出して使う")
    ap.add_argument("--tracknet-threshold", type=float, default=TRACKNET_THRESHOLD)
    ap.add_argument("--joint-fit", action="store_true",
                    help="ラリーの中のショットをつないでまとめて当てはめ直す（終端速度も長いショットから求める）")
    ap.add_argument("--no-estimate-vt", action="store_true", help="--joint-fit で、終端速度は設定の値のままにする")
    ap.add_argument("--fit-backend", default="auto", choices=["auto", "cpu", "batch"],
                    help="当てはめの計算の仕方（cpu：1 本ずつプロセスで並べる、batch：まとめて解く、auto：batch が使えれば batch）")
    ap.add_argument("--config", type=Path, default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    info = analyze_video(ShotListOptions(a.video, a.calib, a.out, a.capture_fps, a.start, a.end, audio=a.audio,
                                         use_audio=not a.no_audio, filter_tracks=not a.all_tracks,
                                         players=a.players, hitter_prior=not a.no_hitter_prior,
                                         hit_players=not a.no_hit_players, detections=a.detections,
                                         tracknet=a.tracknet, tracknet_threshold=a.tracknet_threshold,
                                         joint_fit=a.joint_fit,
                                         joint_options=JointOptions(estimate_vt=not a.no_estimate_vt),
                                         fit_backend=a.fit_backend), cfg)
    print("、".join(f"{v['label']} {v['n']} 本" + (f"（最高 {v['max_kmh']:.0f} km/h）" if v["max_kmh"] else "")
                   for v in info["by_type"].values()))


if __name__ == "__main__":
    import sys

    if (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    main()
