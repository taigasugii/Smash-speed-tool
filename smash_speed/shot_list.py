"""動画 1 本のすべてのショットを、種類・初速つきで一覧にする（ROADMAP B5 の解析の部分）。

1. 動画全体を 1 回だけ縮小して読み、シャトルを検出する（``detection.baseline``。1 コマ 1 点）。
2. 打球音（音声があれば）と軌道の向きの変化から、すべての打球の時刻と打った側を出す（``hits.find_hits``）。
3. 打球の間隔でラリーに分け（仮の区切り。本当の区切りは C1）、ラリーごとに軌道と初速を当てはめる
   （``shot_fit.fit_shots``）。ほかのラリーの検出が混ざらないよう、ラリーの前後で検出を切り分ける。
4. 軌道から種類を決める（``shot_types.classify_shots``）。

結果は ``<出力先>/`` に書き出す。

- ``shots.csv``・``rallies.csv``：ショットとラリー（``shots`` の形）
- ``detections.csv``：全コマのシャトルの検出（GUI で場面に軌道を重ねる・解析し直すのに使う）
- ``shot_list.json``：動画・キャリブレーション・打球の見つけ方・種類ごとの本数と速さなど（GUI が読む）

``python -m smash_speed.shot_list <動画> --calib <calibration.json> --out <出力先>`` で動かす。
"""

from __future__ import annotations

import json
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
from smash_speed.hits import Hit, find_hits
from smash_speed.physics import FitOptions, PhysicsParams
from smash_speed.repro import run_metadata
from smash_speed.shot_fit import fit_shots, shot_fits_to_shots
from smash_speed.shot_types import classify_shots
from smash_speed.shots import (RALLIES_FILE, SHOT_TYPE_LABELS, SHOT_TYPES, SHOTS_FILE, Shot, make_rallies,
                               write_rallies, write_shots)
from smash_speed.video_io import VideoMeta, extract_audio, load_wav, probe

SHOT_LIST_FILE = "shot_list.json"
DETECTIONS_FILE = "detections.csv"


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
    rally_gap_s: float = 3.0          # 打球の間がこれより空いたら別のラリー [s]
    n_tracks: int = 2000              # 検出で残す軌道の数の上限（長い動画では軌道が多い）


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
               roi: list[float] | None = None, n_tracks: int = 2000,
               progress: Callable[[str], None] = print) -> pd.DataFrame:
    """動画の区間の全コマでシャトルを検出する（1 コマ 1 点。座標は元の解像度）。"""
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
    return detect_frames(_with_progress(frames, n_total, progress), scale, p, n_tracks=n_tracks, exclusive=True)


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
                 max_rms_frac: float = 0.01, progress: Callable[[str], None] = print) -> list[Shot]:
    """打球ごとに軌道と初速を当てはめ、種類を決めた ``Shot`` の列（時刻の順。ラリーの番号つき）。

    Args:
        det: 動画全体の検出結果
        hits: ``find_hits`` の結果（時刻の順）
        tail_s: ラリーの最後の打球の後、これだけの間の検出までを最後のショットに使う [s]
        ci: 初速の信頼区間を出す（遅くなるが、信頼度に使う）
        max_kmh: これより速い初速は、当てはめに失敗したとみなす（最速のスマッシュは 565 km/h）[km/h]
        max_rms_frac: 画像の上のずれ（RMS）が画像の幅のこの割合を超えたら、当てはめに失敗したとみなす
            （4K で 38 px。合成ラリーでは 1〜2 px、実動画の正しい当てはめも数 px）
    """
    opts = FitOptions.from_config(cfg, **({} if ci else {"ci_method": "none"}))
    vis_t = det.timestamp.to_numpy(float)
    groups = rally_groups(hits, rally_gap_s)
    shots: list[Shot] = []
    for g, idx in enumerate(groups):
        hs = [hits[i] for i in idx]
        lo = hs[0].t - 0.5
        hi = hs[-1].t + tail_s
        if g + 1 < len(groups):
            hi = min(hi, hits[groups[g + 1][0]].t - 0.05)
        part = det[(vis_t >= lo) & (vis_t <= hi)]
        fits = fit_shots(part, [h.t for h in hs], camera, phys, [h.side for h in hs], opts)
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
        shots += rally_shots
        n_ok = sum(s.speed_kmh is not None for s in rally_shots)
        progress(f"[進行] ラリー {g + 1}/{len(groups)}：{len(hs)} 本（速さが出たもの {n_ok} 本）")
    make_rallies(shots, max_gap_s=rally_gap_s)
    return classify_shots(shots, phys)


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
    out = Path(o.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    meta = probe(o.video, o.capture_fps, cfg)
    camera, _ = load_calibration(o.calib)
    camera = _camera_for_video(camera, meta)
    phys = PhysicsParams.from_config(cfg)

    onsets = hit_onsets(meta, cfg, out, o.audio, o.start_s, o.end_s) if o.use_audio else None
    if onsets is not None:
        progress(f"[進行] 打球音の候補 {len(onsets)} 個")
    det = detect_all(meta, cfg, o.start_s, o.end_s, o.roi, o.n_tracks, progress)
    det.to_csv(out / DETECTIONS_FILE, index=False)
    n_vis = int(det.visible.sum())
    if n_vis < 5:
        raise AnalysisError("シャトルがほとんど見つかりませんでした。",
                            ["コートとシャトルが写っている動画か、検出する範囲が正しいかを確かめてください。"])
    progress(f"[進行] シャトルが見えたコマ {n_vis} / {len(det)}")

    hits = find_hits(det, meta.capture_fps, onsets, camera=camera, rally_gap_s=o.rally_gap_s)
    progress(f"[進行] 打球 {len(hits)} 本")
    shots = analyze_hits(det, hits, camera, phys, cfg, o.rally_gap_s, progress=progress)
    rallies = make_rallies(shots, max_gap_s=o.rally_gap_s)
    write_shots(out / SHOTS_FILE, shots)
    write_rallies(out / RALLIES_FILE, rallies)

    info = {
        "video": str(Path(o.video).resolve()), "calib": str(Path(o.calib).resolve()),
        "capture_fps": o.capture_fps, "fps": meta.capture_fps, "width": meta.width, "height": meta.height,
        "start_s": o.start_s, "end_s": o.end_s, "roi": o.roi,
        "audio": str(Path(o.audio).resolve()) if o.audio else None,
        "used_audio": onsets is not None, "n_onsets": None if onsets is None else len(onsets),
        "n_frames": len(det), "n_visible": n_vis, "n_hits": len(hits), "n_rallies": len(rallies),
        "hit_methods": pd.Series([h.method for h in hits], dtype=object).value_counts().to_dict(),
        "by_type": summarize(shots), "elapsed_s": round(time.time() - t_start, 1),
        "meta": run_metadata(),
    }
    (out / SHOT_LIST_FILE).write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    progress(f"[進行] 完了：{len(shots)} 本（{len(rallies)} ラリー）、{info['elapsed_s']:.0f} 秒")
    return info


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
    ap.add_argument("--config", type=Path, default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    info = analyze_video(ShotListOptions(a.video, a.calib, a.out, a.capture_fps, a.start, a.end, audio=a.audio,
                                         use_audio=not a.no_audio), cfg)
    print("、".join(f"{v['label']} {v['n']} 本" + (f"（最高 {v['max_kmh']:.0f} km/h）" if v["max_kmh"] else "")
                   for v in info["by_type"].values()))


if __name__ == "__main__":
    import sys

    if (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    main()
