"""動画 1 本のフォームの解析（ROADMAP E2〜E7 をつなぐ所）。

側方から撮った動画（``docs/pose_features.md`` の撮影の想定）から、ショットごとに次を行う。

1. 打球の時刻の候補を決める：渡された時刻（ショットの一覧など）→ 打球音 → 動画全体の手首の速さの最大、の順。
2. 候補の前後（前 1.3 秒〜後 0.8 秒）のコマの姿勢を推定し（``pose``）、欠けを整え、打点を決め直し、
   前 1.0 秒〜後 0.5 秒に切り出して正規化する（``pose_prep``）。手首がほとんど動いていない候補（相手の打球音など）は捨てる。
3. フォームの特徴量（``pose_features``）を出す。

結果は ``<出力先>/form/`` に書き出す。

- ``form_features.csv``：1 行 1 ショット（ショットの ID・時刻・打点の決め方・特徴量）
- ``poses/<ショットの ID>.npz``：正規化した姿勢（``pose_prep.NormalizedPose``。画面・類似度スコアに使う）
- ``raw/<ショットの ID>.npz``：推定したままの姿勢（``pose.PoseSequence``。画像に重ねる用）
- ``form.json``：動画・モデル・利き手などの条件

ショットの ID は ``<動画の名前>_<打点の時刻（1/100 秒）>``（例 ``clear_01_001023``）。解析し直しても同じ打球なら同じ ID になる。

``python -m smash_speed.form <動画> --out <出力先> [--model rtmpose] [--left-handed] [--times 10.2,15.3]``
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from smash_speed.pose import PoseModel, PoseSequence, clean, estimate_frames, make_model
from smash_speed.pose_features import FEATURE_NAMES, form_features
from smash_speed.pose_prep import AFTER_S, BEFORE_S, prepare_shot, wrist_speed

FORM_DIR = "form"
FEATURES_FILE = "form_features.csv"
INFO_FILE = "form.json"
BASE_COLUMNS = ["shot_id", "video", "t_hit_s", "t_impact_s", "impact_method", "model", "left_handed"]
MARGIN_S = 0.3            # 打点を決め直す分、切り出しより前後に広く推定する [s]

FrameSource = Callable[[float, float], Iterable[tuple[float, np.ndarray]]]


@dataclass
class FormOptions:
    """フォームの解析の設定。"""

    model: str = "rtmpose"
    device: str | None = None            # auto / gpu / cpu（None なら環境変数か auto）
    left_handed: bool = False
    sound_distance_m: float = 7.0        # 打つ人からカメラ（マイク）までの距離 [m]
    wrist_offset_s: float = 0.0          # 音がないとき、手首の速さの最大から打点までの時間 [s]
    before_s: float = BEFORE_S
    after_s: float = AFTER_S
    min_wrist_speed: float = 2.0         # 手首の速さの最大がこれより遅い候補は捨てる [身長/秒]
    min_gap_s: float = 1.0               # 候補どうしをこれより近くしない [s]
    audio_min_strength: float = 20.0     # 打球音の候補にする音の強さ（雑音比）


def shot_id(video: str | Path, t_impact: float) -> str:
    return f"{Path(video).stem}_{int(round(t_impact * 100)):06d}"


def _merge_close(times: list[float], gap: float) -> list[float]:
    out: list[float] = []
    for t in sorted(times):
        if not out or t - out[-1] >= gap:
            out.append(t)
    return out


def video_frames(video: str | Path, capture_fps: float | None = None) -> tuple[FrameSource, dict[str, Any]]:
    """動画の区間のコマを読む関数と、動画の情報。"""
    from smash_speed.video_io import VideoReader, probe

    meta = probe(video, capture_fps=capture_fps)

    def frames(t0: float, t1: float) -> Iterable[tuple[float, np.ndarray]]:
        with VideoReader(meta) as reader:
            for _, t, img in reader.iter_range(t0, t1):
                yield t, img

    info = {"width": meta.width, "height": meta.height, "fps": meta.capture_fps,
            "duration_s": float(meta.timestamps[-1]) if len(meta.timestamps) else 0.0, "has_audio": meta.has_audio,
            "time_scale": meta.time_scale}
    return frames, info


def video_onsets(video: str | Path, time_scale: float = 1.0) -> list[tuple[float, float]]:
    """動画の音声の鋭い音の立ち上がり（時刻 [s]、強さ）。音声がなければ空。"""
    from smash_speed.events import audio_onsets
    from smash_speed.video_io import extract_audio, load_wav

    with tempfile.TemporaryDirectory() as d:
        try:
            wav = extract_audio(video, Path(d) / "a.wav")
        except Exception:
            return []
        x, rate = load_wav(wav)
    return [(t * time_scale, s) for t, s in audio_onsets(x, rate)]


def swing_candidates(frames: FrameSource, model: PoseModel, duration_s: float, info: dict[str, Any],
                     opts: FormOptions) -> list[float]:
    """音も時刻もないとき：動画全体の姿勢を推定し、手首の速さの最大を候補にする。"""
    seq = clean(estimate_frames(frames(0.0, duration_s), model, info.get("width", 0), info.get("height", 0),
                                info.get("fps", 0.0)))
    tm, v = wrist_speed(seq)
    if len(v) == 0:
        return []
    sh = (seq.joint("left_shoulder") + seq.joint("right_shoulder")) / 2
    hp = (seq.joint("left_hip") + seq.joint("right_hip")) / 2
    L = np.linalg.norm(sh - hp, axis=1)
    if not np.isfinite(L).any():  # 人が見つからない
        return []
    torso = float(np.nanmedian(L))
    # 身長 ≈ 胴体の長さ ÷ 0.288。速さを身長/秒にして、決まった値より速い山を拾う
    v = np.nan_to_num(v) * 0.288 / torso
    peaks = [i for i in range(1, len(v) - 1) if v[i] >= opts.min_wrist_speed and v[i] >= v[i - 1] and v[i] > v[i + 1]]
    peaks.sort(key=lambda i: -v[i])
    chosen: list[float] = []
    for i in peaks:
        if all(abs(tm[i] - c) >= opts.min_gap_s for c in chosen):
            chosen.append(float(tm[i]))
    return sorted(chosen)


def analyze_shot(frames: FrameSource, model: PoseModel, t_hit: float, info: dict[str, Any], opts: FormOptions,
                 audio_times: list[float] | None = None) -> tuple[dict[str, Any], PoseSequence, Any] | None:
    """1 本のショットの姿勢と特徴量。手首がほとんど動いていなければ None。"""
    t0 = t_hit - opts.before_s - MARGIN_S
    t1 = t_hit + opts.after_s + MARGIN_S
    raw = estimate_frames(frames(t0, t1), model, info.get("width", 0), info.get("height", 0), info.get("fps", 0.0))
    if len(raw) < 5:
        return None
    seq = clean(raw)
    try:
        norm, imp = prepare_shot(seq, t_hit, opts.left_handed, audio_times, opts.sound_distance_m,
                                 opts.wrist_offset_s, opts.before_s, opts.after_s)
    except ValueError:
        return None
    feats = form_features(norm)
    if not np.isfinite(feats.wrist_speed_max) or feats.wrist_speed_max < opts.min_wrist_speed:
        return None
    row = {"t_hit_s": t_hit, "t_impact_s": imp.t, "impact_method": imp.method, "model": raw.model,
           "left_handed": opts.left_handed, **feats.to_dict()}
    return row, raw, norm


def write_rows(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> Path:
    """1 行 1 ショットの CSV（Excel で開けるよう BOM 付きの UTF-8。分からない値は空欄）。"""
    columns = columns or BASE_COLUMNS + FEATURE_NAMES
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _cell(r.get(k)) for k in columns})
    return path


def _cell(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, float):
        return "" if not np.isfinite(v) else f"{v:.6g}"
    return v


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    """``write_rows`` で書いた CSV を読む（数に読めるものは数に、空欄は None）。"""
    out = []
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            d: dict[str, Any] = {}
            for k, v in r.items():
                if v == "":
                    d[k] = None
                elif v in ("True", "False"):
                    d[k] = v == "True"
                else:
                    try:
                        d[k] = float(v)
                    except ValueError:
                        d[k] = v
            out.append(d)
    return out


def analyze(video: str | Path, out_dir: str | Path, opts: FormOptions | None = None,
            hit_times: list[float] | None = None, frames: FrameSource | None = None,
            info: dict[str, Any] | None = None, onsets: list[tuple[float, float]] | None = None,
            model: PoseModel | None = None, progress: Callable[[str], None] = print) -> list[dict[str, Any]]:
    """動画 1 本のショットのフォームを解析し、``<out_dir>/form/`` に書き出す。

    ``frames``・``info``・``onsets``・``model`` は差し替え用（テストでは合成の骨格を返す偽物を使う）。
    """
    opts = opts or FormOptions()
    t_start = time.time()
    out = Path(out_dir) / FORM_DIR
    if frames is None:
        frames, vinfo = video_frames(video)
        info = {**vinfo, **(info or {})}
    info = info or {}
    if onsets is None:
        onsets = video_onsets(video, info.get("time_scale", 1.0)) if info.get("has_audio") else []
    audio_times = [t for t, s in onsets if s >= opts.audio_min_strength]
    own_model = model is None
    if own_model:
        model = make_model(opts.model, prefer=opts.device)
    try:
        return _analyze(video, out, opts, hit_times, frames, info, audio_times, model, progress, t_start)
    finally:
        # MediaPipe は終わるときに閉じないと、Python の終了の途中で後片付けが失敗してエラーを出す
        if own_model and hasattr(model, "close"):
            model.close()


def _analyze(video, out: Path, opts: FormOptions, hit_times, frames: FrameSource, info: dict[str, Any],
             audio_times: list[float], model: PoseModel, progress: Callable[[str], None], t_start: float
             ) -> list[dict[str, Any]]:
    if hit_times:
        cands, source = sorted(hit_times), "given"
    elif audio_times:
        cands, source = _merge_close(audio_times, opts.min_gap_s), "audio"
    else:
        progress("[進行] 打球音がないので、動画全体の手首の動きからスイングを探します")
        cands, source = swing_candidates(frames, model, info.get("duration_s", 0.0), info, opts), "wrist"
    progress(f"[進行] スイングの候補 {len(cands)} 個")
    rows = []
    for k, t in enumerate(cands):
        res = analyze_shot(frames, model, t, info, opts, audio_times)
        if res is None:
            continue
        row, raw, norm = res
        sid = shot_id(video, row["t_impact_s"])
        if any(r["shot_id"] == sid for r in rows):
            continue
        row = {"shot_id": sid, "video": Path(video).name, **row}
        raw.save(out / "raw" / f"{sid}.npz")
        norm.meta.update(shot_id=sid, video=Path(video).name)
        norm.save(out / "poses" / f"{sid}.npz")
        rows.append(row)
        progress(f"[進行] {k + 1}/{len(cands)} {sid}")
    write_rows(out / FEATURES_FILE, rows)
    (out / INFO_FILE).write_text(json.dumps({
        "video": str(video), "options": asdict(opts), "candidates_from": source, "n_candidates": len(cands),
        "n_shots": len(rows), "elapsed_s": round(time.time() - t_start, 1), "model": getattr(model, "name", ""),
        "provider": getattr(model, "provider", ""), "analyzed_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
        ensure_ascii=False, indent=1), encoding="utf-8")
    progress(f"[完了] {len(rows)} 本のショットのフォームを書き出しました: {out / FEATURES_FILE}")
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="側方から撮った動画のフォームの解析（姿勢推定 → 特徴量）")
    p.add_argument("video")
    p.add_argument("--out", required=True, help="出力先（form/ を作る）")
    p.add_argument("--model", choices=("rtmpose", "mediapipe"), default="rtmpose")
    p.add_argument("--device", choices=("auto", "gpu", "cpu"), default=None)
    p.add_argument("--left-handed", action="store_true", help="左利き（画像を反転して右利きと同じ式で出す）")
    p.add_argument("--distance", type=float, default=7.0, help="打つ人からカメラまでの距離 [m]")
    p.add_argument("--times", default=None, help="打球のおおよその時刻 [s] をカンマ区切りで（なければ音・手首から探す）")
    args = p.parse_args(argv)
    opts = FormOptions(model=args.model, device=args.device, left_handed=args.left_handed,
                       sound_distance_m=args.distance)
    times = [float(x) for x in args.times.split(",")] if args.times else None
    analyze(args.video, args.out, opts, hit_times=times)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
