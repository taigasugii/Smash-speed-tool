"""基礎打ちの解析（コート合わせなし）。

コート合わせがなくても、画像の上のシャトルの検出・打球音・（あれば）人の枠だけで、打球の時刻は出せる。
このモジュールは、動画 1 本の全コマの検出と打球（G2a）。続く課題で、打った側（G3）・続いた打ち合いの区切り（G4）・
まとめ（G5）を足す。

出力（``<出力先>/``）：``detections.csv``（1 コマ 1 点に絞った検出）・``onsets.csv``（打球音の候補）・``hits.csv``（打球）・
``drill_detect.json``（件数と所要時間）。
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from smash_speed.errors import AnalysisError
from smash_speed.hits import Hit, find_hits, match_hits
from smash_speed.shot_list import (DETECTIONS_FILE, TRACKNET_THRESHOLD, ShotListOptions, TrackFilter, choose_tracknet,
                                   detect_all, hit_onsets, load_detections, one_per_frame, select_tracks,
                                   tracknet_detections)
from smash_speed.synth_drill import DrillOptions
from smash_speed.video_io import probe

HITS_FILE = "hits.csv"
ONSETS_FILE = "onsets.csv"
DETECT_INFO_FILE = "drill_detect.json"


# =============================================================================
# 画像の上だけの軌道の絞り込み
# =============================================================================
def image_track_table(det: pd.DataFrame, width: int, flt: TrackFilter | None = None) -> pd.DataFrame:
    """``shot_list.track_table`` の、コートの空間を使わない版。点の数・長さ・画像の上の速さと広がりだけで残す軌道を決める。

    列は ``track_table`` と同じ（``inside`` は 1、``speed_max_mps`` は無限大にして、コートの条件は通す）。
    """
    flt = flt or TrackFilter()
    W = float(width)
    rows = []
    for tid, g in det[(det.visible > 0) & det.x_px.notna()].groupby("track_id"):
        g = g.sort_values("timestamp")
        t, uv = g.timestamp.to_numpy(float), g[["x_px", "y_px"]].to_numpy(float)
        speed = 0.0
        if len(t) >= 2:
            speed = float(np.median(np.linalg.norm(np.diff(uv, axis=0), axis=1) / np.maximum(np.diff(t), 1e-9))) / W
        u0 = float(uv[0, 0]) / W
        rows.append({"track_id": int(tid), "n": len(t), "t_start": float(t[0]), "t_end": float(t[-1]),
                     "duration_s": float(t[-1] - t[0]), "image_speed": speed,
                     "extent": float(np.linalg.norm(uv.max(0) - uv.min(0))) / W, "inside": 1.0,
                     "speed_max_mps": float("inf"), "edge_start": bool(min(u0, 1.0 - u0) < flt.edge_margin)})
    tab = pd.DataFrame(rows, columns=["track_id", "n", "t_start", "t_end", "duration_s", "image_speed", "extent",
                                      "inside", "speed_max_mps", "edge_start"])
    tab["keep"] = ((tab.n >= flt.min_points) & (tab.duration_s >= flt.min_duration_s)
                   & (tab.image_speed <= flt.max_image_speed) & (tab.image_speed >= flt.min_image_speed)
                   & (tab.extent >= flt.min_extent))
    return tab


def drill_tracks(det: pd.DataFrame, width: int, flt: TrackFilter | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """重なった軌道を含む検出から、シャトルらしい軌道を画像の上の条件で残し（``image_track_table``）、
    同じ時間に重なったものから選んで（``select_tracks``）1 コマ 1 点にする。``shot_list.shuttle_tracks`` の、コート合わせなし版。"""
    tab = image_track_table(det, width, flt)
    tab["selected"] = select_tracks(tab, flt)
    return one_per_frame(det, det.track_id.isin(tab.track_id[tab.selected])), tab


# =============================================================================
# 打球
# =============================================================================
# 練習の打球の見つけ方（``find_hits`` の既定からの変更。合成の斜めカメラの 30fps で、雑音なし 再現率 0.96・適合率 0.98、
# 既定の雑音 0.86・0.82。変える前は 0.79・0.84 と 0.72・0.70）
DRILL_HIT_KWARGS: dict[str, Any] = {
    "rally_both_sides": False,    # 続いた打ち合いの最初と最後の打球は、前か後ろのシャトルが見えない（サーブ・球出し、落ちる球）
    "min_track_score": 3.0,       # 音があるとき、音のない軌道の変化は、はっきりしたものだけ（斜めのカメラは向きの変化が小さく、偽の変化が多い）
    "min_audio_strength": 20.0,   # 軌道の変化が小さくても、音がはっきりしていれば打球にする
}


def drill_hits(det: pd.DataFrame, fps: float, onsets: list[tuple[float, float]] | None, frame_size: tuple[int, int],
               **kw: Any) -> list[Hit]:
    """絞り込んだ検出と打球音から打球を見つける（``hits.find_hits``。コート合わせがないので、打った側は空欄）。

    練習は続けて打つので、``rally_gap_s`` は既定のまま。打った側をそろえる単位は G3 で決める。
    """
    base = dict(DRILL_HIT_KWARGS)
    if not onsets:
        del base["min_track_score"]   # 音がなければ軌道の変化だけが手がかりなので、しきい値を上げない
    return find_hits(det, fps, onsets, camera=None, frame_size=frame_size, **{**base, **kw})


def hits_table(hits: list[Hit]) -> pd.DataFrame:
    cols = list(Hit(0.0).to_dict())
    return pd.DataFrame([h.to_dict() for h in hits], columns=cols)


# =============================================================================
# 動画 1 本
# =============================================================================
@dataclass
class DrillDetectOptions:
    video: Path
    out_dir: Path
    capture_fps: float | None = None
    start_s: float | None = None
    end_s: float | None = None
    audio: Path | None = None
    use_audio: bool = True
    tracknet: Path | None = None      # 学習したモデル（``shot_list.choose_tracknet`` で決める）
    detections: Path | None = None    # 自前の検出 CSV（あれば全コマの検出はしない）
    tracknet_threshold: float = TRACKNET_THRESHOLD
    tracknet_fallback: bool = False
    roi: list[float] | None = None


def detect(o: DrillDetectOptions, cfg: dict[str, Any], progress: Callable[[str], None] = print) -> dict[str, Any]:
    """動画の全コマのシャトルの検出と打球音から、打球を見つけて ``<出力先>/`` に書き、件数と所要時間を返す。"""
    t0 = time.time()
    timings: dict[str, float] = {}
    mark = [t0]

    def lap(name: str) -> None:
        now = time.time()
        timings[name] = round(now - mark[0], 2)
        mark[0] = now

    out = Path(o.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    meta = probe(o.video, o.capture_fps, cfg)
    lap("probe")
    onsets = hit_onsets(meta, cfg, out, o.audio, o.start_s, o.end_s) if o.use_audio else None
    if onsets is not None:
        pd.DataFrame(onsets, columns=["t", "strength"]).to_csv(out / ONSETS_FILE, index=False)
        progress(f"[進行] 打球音の候補 {len(onsets)} 個")
    lap("audio")

    # コート合わせがないので、ショットの一覧の検出の段階と同じ道具を、カメラなしで使う
    so = ShotListOptions(o.video, Path(), out, o.capture_fps, o.start_s, o.end_s, tracknet=o.tracknet,
                         tracknet_threshold=o.tracknet_threshold)
    tracknet, det_path, note = o.tracknet, o.detections, None
    if tracknet is not None:
        try:
            det_path = tracknet_detections(so, out, progress)
        except (OSError, RuntimeError, ValueError, AnalysisError) as e:
            if not o.tracknet_fallback:
                raise
            note = f"学習したモデルで検出できなかったので、今の検出で探した（{str(e).strip()[-300:]}）"
            progress(f"[進行] {note}")
            tracknet = None
    if det_path is not None:
        det = load_detections(det_path, meta, o.start_s, o.end_s)
    else:
        det = detect_all(meta, cfg, o.start_s, o.end_s, o.roi, exclusive=False, progress=progress)
    lap("shuttle_detect")
    det, tab = drill_tracks(det, meta.width)
    det.to_csv(out / DETECTIONS_FILE, index=False)
    n_vis = int(det.visible.sum())
    if n_vis < 5:
        raise AnalysisError("シャトルがほとんど見つかりませんでした。", ["シャトルが写っている動画かを確かめてください。"])
    progress(f"[進行] シャトルらしい軌道 {int(tab.selected.sum())} / {len(tab)} 本、見えたコマ {n_vis} / {len(det)}")
    lap("shuttle_tracks")

    hits = drill_hits(det, meta.capture_fps, onsets, (meta.width, meta.height))
    hits_table(hits).to_csv(out / HITS_FILE, index=False)
    lap("hits")
    t_lo = float(det.timestamp.min())
    minutes = max((float(det.timestamp.max()) - t_lo) / 60.0, 1e-9)
    info = {"video": str(o.video), "duration_s": round(minutes * 60.0, 2), "n_frames": int(len(det)),
            "n_visible": n_vis, "n_tracks": int(tab.selected.sum()), "n_onsets": None if onsets is None else len(onsets),
            "n_hits": len(hits), "hits_per_min": round(len(hits) / minutes, 2),
            "detector": "tracknet" if tracknet is not None else ("detections" if det_path is not None else "baseline"),
            "detector_note": note, "timings_s": timings, "elapsed_s": round(time.time() - t0, 2)}
    (out / DETECT_INFO_FILE).write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    progress(f"[進行] 打球 {len(hits)} 本（1 分あたり {info['hits_per_min']}）")
    return info


# =============================================================================
# 合成での測り方
# =============================================================================
def synthetic_hits_benchmark(kinds: tuple[str, ...] | None = None, n_scenes: int = 1, fps: float = 30.0,
                             camera_name: str = "oblique", seed: int = 0, use_audio: tuple[bool, ...] = (True, False),
                             opts: DrillOptions | None = None, cfg: dict[str, Any] | None = None
                             ) -> dict[bool, dict[str, Any]]:
    """合成の基礎打ち（``synth_drill``）で、絞り込みと打球の見つけ方を測る。音あり・なしごとに、
    再現率・適合率・正しく見つけた打球の時刻のずれの平均 [s]（``mean_abs_dt_s``）と、練習の種類ごとの数字（``by_kind``）を返す。

    場面を作るのが遅い（1 場面 1 分ほど）ので、既定は短い打ち合い 3 回。同じ場面を音あり・なしで使い回す。
    """
    from smash_speed.config import load_config
    from smash_speed.physics import PhysicsParams
    from smash_speed.synth_drill import MAIN_DRILLS, make_drill_scene, observe_drill
    from smash_speed.synth_rally import named_rally_camera

    cfg = cfg or load_config()
    phys = PhysicsParams.from_config(cfg)
    cam = named_rally_camera(camera_name, cfg)
    opts = opts or DrillOptions(n_streaks=3, streak_hits=(4, 14), pause_s=(3.0, 6.0))
    acc = {a: {"tot": {"tp": 0, "fp": 0, "fn": 0}, "dts": [], "per": {}} for a in use_audio}
    for kind in kinds or MAIN_DRILLS:
        k_tot = {a: {"tp": 0, "fp": 0, "fn": 0} for a in use_audio}
        for i in range(n_scenes):
            rng = np.random.default_rng([seed, len(kind), i])
            ob = observe_drill(make_drill_scene(rng, phys, kind, opts), cam, fps, np.random.default_rng([seed, 5, i]),
                               opts)
            det, _ = drill_tracks(ob.det.drop(columns=["source"], errors="ignore"), cam.width)
            for a in use_audio:
                hits = drill_hits(det, fps, ob.onsets if a else None, (cam.width, cam.height))
                m = match_hits(hits, ob.truth.hits.t.to_numpy())
                for key in k_tot[a]:
                    k_tot[a][key] += m[key]
                    acc[a]["tot"][key] += m[key]
                if m["mean_abs_dt_s"] is not None:
                    acc[a]["dts"].append(m["mean_abs_dt_s"])
        for a in use_audio:
            k = k_tot[a]
            acc[a]["per"][kind] = {**k, "recall": k["tp"] / max(k["tp"] + k["fn"], 1),
                                   "precision": k["tp"] / max(k["tp"] + k["fp"], 1)}
    out = {}
    for a, v in acc.items():
        t = v["tot"]
        out[a] = {"camera": camera_name, "fps": fps, **t, "recall": t["tp"] / max(t["tp"] + t["fn"], 1),
                  "precision": t["tp"] / max(t["tp"] + t["fp"], 1),
                  "mean_abs_dt_s": float(np.mean(v["dts"])) if v["dts"] else None, "by_kind": v["per"]}
    return out


def main(argv: list[str] | None = None) -> None:
    from smash_speed.config import load_config

    ap = argparse.ArgumentParser(description="基礎打ちの動画の検出と打球（コート合わせなし）")
    ap.add_argument("video", type=Path, nargs="?")
    ap.add_argument("--out", type=Path, default=None, help="出力先のフォルダ（既定：outputs/drills/<動画の名前>）")
    ap.add_argument("--capture-fps", type=float, default=None)
    ap.add_argument("--start", type=float, default=None, help="解析する区間の始まり [s]")
    ap.add_argument("--end", type=float, default=None, help="解析する区間の終わり [s]")
    ap.add_argument("--audio", type=Path, default=None, help="別の音声ファイル（WAV）")
    ap.add_argument("--no-audio", action="store_true", help="打球音を使わない")
    ap.add_argument("--detections", type=Path, default=None, help="自前の検出 CSV")
    ap.add_argument("--tracknet", type=Path, default=None, help="学習したモデル（model.pt）で全コマを検出して使う")
    ap.add_argument("--tracknet-threshold", type=float, default=TRACKNET_THRESHOLD)
    ap.add_argument("--baseline-detector", action="store_true", help="学習したモデルを使わず、今の検出で探す")
    ap.add_argument("--bench", action="store_true", help="動画のかわりに、合成の基礎打ちで打球の見つけ方を測る")
    ap.add_argument("--scenes", type=int, default=1)
    ap.add_argument("--config", type=Path, default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.bench:
        for audio, r in synthetic_hits_benchmark(n_scenes=a.scenes, cfg=cfg).items():
            print(f"{'音あり' if audio else '音なし'}：再現率 {r['recall']:.3f}・適合率 {r['precision']:.3f}"
                  f"・時刻のずれ {r['mean_abs_dt_s']:.3f} 秒（{', '.join(f'{k} {v['recall']:.2f}/{v['precision']:.2f}' for k, v in r['by_kind'].items())}）")
        return
    if a.video is None:
        ap.error("動画か --bench を指定してください")
    tracknet, fallback = choose_tracknet(a.tracknet, a.detections, a.baseline_detector, cfg)
    out = a.out or Path("outputs") / "drills" / a.video.stem
    info = detect(DrillDetectOptions(a.video, out, a.capture_fps, a.start, a.end, a.audio, not a.no_audio,
                                     tracknet, a.detections, a.tracknet_threshold, fallback), cfg)
    print(f"打球 {info['n_hits']} 本（1 分あたり {info['hits_per_min']}）、{info['elapsed_s']:.0f} 秒")


if __name__ == "__main__":
    import sys

    if (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    main()
