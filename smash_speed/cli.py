"""コマンドラインの入口（``smash-speed <subcommand>``）。

時刻はすべて動画の実時間 [s]（``--capture-fps`` を指定した場合は換算後）で指定する。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from smash_speed.config import load_config


def _xy(s: str) -> tuple[float, float]:
    a, b = s.split(",")
    return float(a), float(b)


# =============================================================================
# 各サブコマンド
# =============================================================================
def cmd_calibrate(a: argparse.Namespace, cfg: dict) -> int:
    from smash_speed.calibration import calibrate, landmark_table, load_points, save_calibration
    from smash_speed.calibration.solve import Intrinsics
    from smash_speed.court import draw_court
    from smash_speed.video_io import VideoReader, probe
    import cv2

    meta = probe(a.video, a.capture_fps, cfg)
    with VideoReader(meta) as r:
        _, frame = r.read_at(a.time)
    intr = None
    if a.intrinsics:
        intr = Intrinsics.from_dict(json.loads(Path(a.intrinsics).read_text(encoding="utf-8")))
    initial = load_points(a.points) if a.points else None
    if a.no_gui:
        if not initial:
            print("--no-gui には --points が必要です", file=sys.stderr)
            return 2
        table = landmark_table(cfg)
        names = [n for n in table if n in initial]
        res = calibrate(names, np.array([table[n] for n in names]), np.array([initial[n] for n in names]),
                        meta.width, meta.height, intr, a.k1, cfg["calibration"]["reproj_warn_px"])
    else:
        from smash_speed.calibration.gui import run_calibration_gui
        res = run_calibration_gui(frame, cfg, initial, intr, a.k1)
        if res is None:
            print("中止しました")
            return 1
    save_calibration(res, a.out, video=str(a.video), frame_time_s=a.time)
    print(f"保存しました: {a.out}  RMSE {res.rmse_px:.2f} px  f = {res.camera.f:.0f} px  "
          f"カメラ位置 {res.camera.center.round(2)} m")
    for n, e in zip(res.names, res.errors_px):
        print(f"  {n:24s} {e:6.2f} px")
    for w in res.warnings:
        print("[警告]", w)
    preview = Path(a.preview) if a.preview else Path(a.out).with_suffix(".png")
    img = frame.copy()
    draw_court(img, res.camera, cfg, thickness=max(2, meta.width // 960))
    for p in res.image:
        cv2.drawMarker(img, tuple(np.round(p).astype(int)), (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
    cv2.imwrite(str(preview), img)
    print(f"再投影の確認画像: {preview}")
    return 0


def cmd_check_calib(a: argparse.Namespace, cfg: dict) -> int:
    """保存済みのキャリブレーションを別の動画に重ね描きして確認する（カメラ固定の撮影で再利用するとき）。"""
    import cv2

    from smash_speed.calibration import load_calibration
    from smash_speed.court import draw_court
    from smash_speed.pipeline import _camera_for_video
    from smash_speed.video_io import VideoReader, probe

    meta = probe(a.video, a.capture_fps, cfg)
    cam, _ = load_calibration(a.calib)
    cam = _camera_for_video(cam, meta)
    with VideoReader(meta) as r:
        _, frame = r.read_at(a.time)
    draw_court(frame, cam, cfg, thickness=max(2, meta.width // 960))
    cv2.imwrite(str(a.out), frame)
    print(f"保存しました: {a.out}")
    return 0


def cmd_plane(a: argparse.Namespace, cfg: dict) -> int:
    from smash_speed.calibration import load_calibration
    from smash_speed.calibration.gui import run_plane_gui
    from smash_speed.pipeline import _camera_for_video
    from smash_speed.video_io import VideoReader, probe

    meta = probe(a.video, a.capture_fps, cfg)
    cam, _ = load_calibration(a.calib)
    cam = _camera_for_video(cam, meta)
    with VideoReader(meta) as r:
        _, frame = r.read_at(a.time)
    plane = run_plane_gui(frame, cam, cfg)
    if plane is None:
        return 1
    Path(a.out).write_text(json.dumps({**plane.to_dict(), "source": "gui"}, indent=1), encoding="utf-8")
    print(f"保存しました: {a.out}")
    return 0


def cmd_onsets(a: argparse.Namespace, cfg: dict) -> int:
    """打球音の候補時刻を一覧する（長い動画からスマッシュを探すとき）。"""
    import tempfile

    from smash_speed.events import audio_onsets
    from smash_speed.video_io import extract_audio, load_wav, probe

    meta = probe(a.video, a.capture_fps, cfg)
    if a.audio:
        x, rate = load_wav(a.audio)
    elif not meta.has_audio:
        print("この動画には音声がありません（--audio で別の音声ファイルを指定できます）", file=sys.stderr)
        return 1
    else:
        with tempfile.TemporaryDirectory() as d:
            wav = extract_audio(a.video, Path(d) / "a.wav", start_s=a.start / meta.time_scale if a.start else None,
                                end_s=a.end / meta.time_scale if a.end else None, cfg=cfg)
            x, rate = load_wav(wav)
    ev = cfg["events"]
    t0 = a.start or 0.0
    ons = audio_onsets(x, rate, ev["audio_highpass_hz"], k_mad=a.k_mad or ev["audio_k_mad"])
    print("time_s,strength")
    for t, s in ons:
        print(f"{t0 + t * meta.time_scale:.4f},{s:.1f}")
    return 0


def cmd_detect(a: argparse.Namespace, cfg: dict) -> int:
    from smash_speed.video_io import probe

    meta = probe(a.video, a.capture_fps, cfg)
    if a.detector == "baseline":
        from smash_speed.detection.baseline import BaselineParams, detect
        p = BaselineParams.from_config(cfg)
        if a.roi:
            p.roi = [float(v) for v in a.roi.split(",")]
        det = detect(meta, a.start, a.end, p, n_tracks=a.n_tracks)
    else:
        from smash_speed.detection.tracknet import run_tracknet
        det = run_tracknet(meta, a.start, a.end, cfg, work_dir=Path(a.out).parent / "tracknet_work")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    det.to_csv(a.out, index=False)
    print(f"検出 {int(det.visible.sum())} / {len(det)} フレーム → {a.out}")
    return 0


def cmd_analyze(a: argparse.Namespace, cfg: dict) -> int:
    from smash_speed.pipeline import AnalyzeOptions, analyze

    out = Path(a.out) if a.out else Path("outputs") / Path(a.video).stem
    o = AnalyzeOptions(
        video=Path(a.video), calib=Path(a.calib), out_dir=out, capture_fps=a.capture_fps, mode=a.mode,
        start_s=a.start, end_s=a.end, detections=Path(a.detections) if a.detections else None,
        plane_json=Path(a.plane) if a.plane else None,
        hitter_xy=_xy(a.hitter) if a.hitter else None, landing_xy=_xy(a.landing) if a.landing else None,
        hitter_px=_xy(a.hitter_px) if a.hitter_px else None,
        landing_px=_xy(a.landing_px) if a.landing_px else None, plane_gui=a.plane_gui,
        impact_time_s=a.impact_time, impact_method=a.impact_method, av_offset_s=a.av_offset,
        audio_path=Path(a.audio) if a.audio else None, ci_method=a.ci, overlay=not a.no_overlay,
        force=a.force, name=a.name, roi=[float(v) for v in a.roi.split(",")] if a.roi else None,
    )
    row = analyze(o, cfg)
    lo, hi = row["speed_initial_ci_low_kmh"], row["speed_initial_ci_high_kmh"]
    print(f"\n初速      {row['speed_initial_kmh']:.1f} km/h（{row['speed_initial_mps']:.1f} m/s）"
          f"  95%CI {lo:.0f}–{hi:.0f} km/h")
    print(f"平均速度  移動距離ベース {row['speed_mean_path_kmh']:.1f} km/h / 直線距離ベース "
          f"{row['speed_mean_straight_kmh']:.1f} km/h")
    print(f"到達時    {row['speed_arrival_kmh']:.1f} km/h  減速率 {row['deceleration_ratio']:.2f}")
    print(f"参考：フレーム間差分（最初の区間）{row['naive_first_interval_kmh']:.1f} km/h")
    print(f"打点時刻 {row['t_impact_s']:.4f} s（{row['impact_method']}, ±{row['impact_sigma_s'] * 1000:.1f} ms）"
          f"  使用フレーム {row['n_frames_used']}  再投影誤差 {row['rms_px']:.2f} px")
    print(f"出力: {out}")
    return 0


def cmd_find_smashes(a: argparse.Namespace, cfg: dict) -> int:
    from smash_speed.batch import BatchOptions, find_candidates

    o = BatchOptions(video=Path(a.video), calib=Path(a.calib), out_dir=Path(a.out), capture_fps=a.capture_fps,
                     start_s=a.start, end_s=a.end, min_strength=a.min_strength, n_select=a.select,
                     roi=[float(v) for v in a.roi.split(",")] if a.roi else None,
                     audio=Path(a.audio) if a.audio else None)
    cands = find_candidates(o, cfg, progress=lambda m: print(m, flush=True))
    ok = sorted([c for c in cands if c.rough_kmh is not None and not c.doubtful], key=lambda c: -c.rough_kmh)
    print()
    print("速い順（だいたいの初速。打つ人・着地点を確認すると変わります）")
    for c in ok[:15]:
        print(f"  #{c.id:2d}  {c.t_onset:7.2f} s  {c.rough_kmh:6.0f} km/h  {'スマッシュらしい' if c.smash_like else ''}")
    print(f"出力: {a.out}")
    return 0


def cmd_analyze_smashes(a: argparse.Namespace, cfg: dict) -> int:
    from smash_speed.batch import analyze_selected, write_ranking

    ids = [int(v) for v in a.ids.split(",")] if a.ids else None
    cands = analyze_selected(Path(a.batch), cfg, ids, overlay=not a.no_overlay,
                             progress=lambda m: print(m, flush=True))
    df = write_ranking(Path(a.batch), cands)
    print()
    print("速い順")
    for r in df.itertuples():
        head = f"{r.rank}位" if r.rank else "順位なし（ありえない速さ。打つ人・着地点を見直す）"
        print(f"  {head}  {r.t_onset_s:7.2f} s  {r.speed_initial_kmh:6.1f} km/h（{r.ci_low_kmh:.0f}〜{r.ci_high_kmh:.0f}）")
    return 0


def cmd_stereo(a: argparse.Namespace, cfg: dict) -> int:
    from smash_speed.stereo import StereoOptions, analyze_stereo

    o = StereoOptions(videos=[Path(v) for v in a.videos], calibs=[Path(c) for c in a.calibs],
                      detections=[Path(d) for d in a.detections] if a.detections else None,
                      out_dir=Path(a.out), capture_fps=a.capture_fps, start_s=a.start, end_s=a.end,
                      offset_s=a.offset, impact_time_s=a.impact_time)
    row = analyze_stereo(o, cfg)
    print(f"初速 {row['speed_initial_kmh']:.1f} km/h  時刻のずれ（2台目 − 1台目）{row['sync_offset_s'] * 1000:.2f} ms")
    return 0


def cmd_annotate(a: argparse.Namespace, cfg: dict) -> int:
    from smash_speed.annotate import run_annotation
    from smash_speed.video_io import probe

    meta = probe(a.video, a.capture_fps, cfg)
    run_annotation(meta, Path(a.out), a.start, a.end, init_csv=Path(a.init) if a.init else None)
    return 0


def cmd_make_lowfps(a: argparse.Namespace, cfg: dict) -> int:
    from smash_speed.train.lowfps import make_lowfps
    from smash_speed.video_io import probe

    meta = probe(a.video, a.capture_fps, cfg)
    for fps in a.target_fps:
        for n_avg in (a.avg_frames or [None]):
            res = make_lowfps(meta, Path(a.labels) if a.labels else None, fps, Path(a.out),
                              blur=a.blur, n_avg=n_avg)
            print(f"{fps} fps（平均 {res['n_avg']} 枚）: {res['video']}")
    return 0


def cmd_export_tracknet(a: argparse.Namespace, cfg: dict) -> int:
    from smash_speed.train.dataset import export_tracknet_dataset

    n = export_tracknet_dataset(Path(a.index), Path(a.out), cfg)
    print(f"{n} 本のラリーを TrackNetV3 の形式で書き出しました: {a.out}")
    return 0


def cmd_train(a: argparse.Namespace, cfg: dict) -> int:
    from smash_speed.train.finetune import finetune_tracknet

    if a.config:
        cfg = load_config(a.config)
    return finetune_tracknet(cfg, dry_run=a.dry_run)


def cmd_eval_detection(a: argparse.Namespace, cfg: dict) -> int:
    import pandas as pd

    from smash_speed.train.metrics import detection_metrics

    tol = a.tolerance or cfg["detection"]["match_tolerance_px"]
    m = detection_metrics(pd.read_csv(a.pred), pd.read_csv(a.gt), tol)
    print(json.dumps(m, ensure_ascii=False, indent=1))
    return 0


def cmd_validate(a: argparse.Namespace, cfg: dict) -> int:
    from smash_speed import validate

    if a.synthetic:
        validate.synthetic_study(cfg, Path(a.out), n_shots=a.n, fps_list=a.fps, seed=cfg["seed"])
    else:
        if not a.dataset:
            print("--dataset か --synthetic を指定してください", file=sys.stderr)
            return 2
        validate.dataset_study(cfg, Path(a.dataset), Path(a.out), fps_list=a.fps,
                               reference_csv=Path(a.reference) if a.reference else None)
    print(f"出力: {a.out}")
    return 0


def cmd_synth(a: argparse.Namespace, cfg: dict) -> int:
    from smash_speed.validate import make_synthetic_session

    info = make_synthetic_session(cfg, Path(a.out), speed=a.speed, fps=a.fps, exposure_s=a.exposure)
    print(json.dumps(info, ensure_ascii=False, indent=1))
    return 0


def cmd_summarize(a: argparse.Namespace, cfg: dict) -> int:
    from smash_speed.report import summarize

    files = sorted(Path(a.results).glob("**/result.csv"))
    if not files:
        print("result.csv が見つかりません", file=sys.stderr)
        return 1
    df = summarize(files, a.out)
    print(f"{len(files)} 本を集計しました: {a.out}")
    print(df[["shot", "speed_initial_kmh", "speed_mean_path_kmh"]].to_string(index=False))
    return 0


# =============================================================================
# 引数
# =============================================================================
SUBCOMMANDS: dict[str, str] = {
    "calibrate": "コート・ネットをクリックしてカメラパラメータを推定する",
    "check-calib": "保存済みのキャリブレーションを別の動画に重ね描きして確認する",
    "plane": "打者の足元と着地点をクリックして飛行平面を指定する",
    "onsets": "打球音の候補時刻を一覧する（スマッシュを探す）",
    "detect": "シャトルを検出して CSV に保存する",
    "analyze": "動画1本からシャトルの初速・平均速度を算出する",
    "stereo": "2台のカメラで三角測量して初速を算出する",
    "annotate": "シャトル位置・打点・着地の正解ラベルを付ける",
    "make-lowfps": "高 fps 動画から疑似低 fps 動画とラベルを生成する",
    "export-tracknet": "ラベル付きデータを TrackNetV3 の学習データ形式で書き出す",
    "train": "TrackNetV3 をファインチューニングする",
    "eval-detection": "検出結果を正解ラベルと比べて適合率・再現率・F1 を出す",
    "validate": "fps 別に初速の推定精度を検証する",
    "synth": "合成のスマッシュ動画（正解つき）を作る",
    "summarize": "複数本の結果を集計表にまとめる",
    "find-smashes": "動画全体からスマッシュの候補を探し、だいたいの初速を出す（いちばん速いスマッシュ探しの 1 段目）",
    "analyze-smashes": "確認した候補を詳しく解析して速い順に並べる（2 段目）",
}


def build_parser() -> argparse.ArgumentParser:
    """全サブコマンドを登録した ArgumentParser を返す。"""
    parser = argparse.ArgumentParser(prog="smash-speed", description="バドミントンのスマッシュ速度解析ツール")
    parser.add_argument("--config", help="既定設定を上書きする YAML ファイル")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str) -> argparse.ArgumentParser:
        return sub.add_parser(name, help=SUBCOMMANDS[name], description=SUBCOMMANDS[name])

    def video_args(p: argparse.ArgumentParser, time: bool = False, rng: bool = False) -> None:
        p.add_argument("--video", required=True)
        p.add_argument("--capture-fps", type=float, help="実際の撮影 fps（スロー動画でメタデータと違う場合）")
        if time:
            p.add_argument("--time", type=float, default=0.0, help="表示するフレームの時刻 [s]")
        if rng:
            p.add_argument("--start", type=float, help="解析区間の開始 [s]")
            p.add_argument("--end", type=float, help="解析区間の終了 [s]")

    p = add("calibrate")
    video_args(p, time=True)
    p.add_argument("--out", default="calibration.json")
    p.add_argument("--points", help="ランドマークの画像座標 JSON（GUI の初期値、または --no-gui で直接使う）")
    p.add_argument("--no-gui", action="store_true")
    p.add_argument("--intrinsics", help="チェッカーボードで求めた内部パラメータ JSON")
    p.add_argument("--k1", action="store_true", help="放射歪み k1 も推定する")
    p.add_argument("--preview", help="再投影の確認画像の保存先")
    p.set_defaults(func=cmd_calibrate)

    p = add("check-calib")
    video_args(p, time=True)
    p.add_argument("--calib", required=True)
    p.add_argument("--out", default="calib_check.png")
    p.set_defaults(func=cmd_check_calib)

    p = add("plane")
    video_args(p, time=True)
    p.add_argument("--calib", required=True)
    p.add_argument("--out", default="plane.json")
    p.set_defaults(func=cmd_plane)

    p = add("onsets")
    video_args(p, rng=True)
    p.add_argument("--k-mad", type=float)
    p.add_argument("--audio", help="動画の音声の代わりに使う音声ファイル（WAV）")
    p.set_defaults(func=cmd_onsets)

    p = add("detect")
    video_args(p, rng=True)
    p.add_argument("--out", required=True)
    p.add_argument("--detector", choices=["baseline", "tracknet"], default="baseline")
    p.add_argument("--roi", help="x0,y0,x1,y1（元解像度 px）")
    p.add_argument("--n-tracks", type=int, default=4)
    p.set_defaults(func=cmd_detect)

    p = add("analyze")
    video_args(p, rng=True)
    p.add_argument("--calib", required=True)
    p.add_argument("--mode", choices=["plane"], default="plane")
    p.add_argument("--out")
    p.add_argument("--name")
    p.add_argument("--detections", help="既存の検出 CSV（TrackNet の結果やアノテーション）")
    p.add_argument("--plane", help="plane.json")
    p.add_argument("--hitter", help="打者の足元のコート座標 x,y [m]")
    p.add_argument("--landing", help="着地点のコート座標 x,y [m]")
    p.add_argument("--hitter-px", help="打者の足元の画像座標 u,v [px]")
    p.add_argument("--landing-px", help="着地点の画像座標 u,v [px]")
    p.add_argument("--plane-gui", action="store_true", help="飛行平面をクリックで指定する")
    p.add_argument("--impact-time", type=float, help="打点時刻を手動で指定 [s]")
    p.add_argument("--impact-method", default="auto", choices=["auto", "audio", "crossing", "hitter_position", "first_frame", "manual"])
    p.add_argument("--av-offset", type=float, default=0.0, help="音声→映像の時刻のずれ [s]")
    p.add_argument("--audio", help="音声ファイル（動画の音声の代わりに使う）")
    p.add_argument("--ci", choices=["jacobian", "bootstrap", "none"])
    p.add_argument("--no-overlay", action="store_true")
    p.add_argument("--force", action="store_true", help="保存済みの途中結果を使わずに作り直す")
    p.add_argument("--roi", help="シャトルを検出する範囲 x0,y0,x1,y1（元解像度 px）")
    p.set_defaults(func=cmd_analyze)

    p = add("stereo")
    p.add_argument("--videos", nargs=2, required=True)
    p.add_argument("--calibs", nargs=2, required=True)
    p.add_argument("--detections", nargs=2)
    p.add_argument("--capture-fps", type=float)
    p.add_argument("--start", type=float)
    p.add_argument("--end", type=float)
    p.add_argument("--offset", type=float, help="2台目の時刻 − 1台目の時刻 [s]（手動の微調整。省略時は音声の相互相関）")
    p.add_argument("--impact-time", type=float)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_stereo)

    p = add("annotate")
    video_args(p, rng=True)
    p.add_argument("--out", required=True, help="ラベルの保存先ディレクトリ")
    p.add_argument("--init", help="初期値にする検出 CSV")
    p.set_defaults(func=cmd_annotate)

    p = add("make-lowfps")
    video_args(p)
    p.add_argument("--labels", help="元動画のラベル CSV（labels.csv）")
    p.add_argument("--target-fps", type=float, nargs="+", default=[30, 60, 120])
    p.add_argument("--blur", action="store_true", help="連続フレームの平均で露光時間を再現する")
    p.add_argument("--avg-frames", type=int, nargs="+", help="平均する枚数（省略時は 元fps/目標fps）")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_make_lowfps)

    p = add("export-tracknet")
    p.add_argument("--index", required=True, help="データ一覧 CSV（video, labels, session, player, split）")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_export_tracknet)

    p = add("train")
    p.add_argument("--config", dest="config", help="学習設定 YAML（configs/train.yaml）")
    p.add_argument("--dry-run", action="store_true", help="実行するコマンドを表示するだけ")
    p.set_defaults(func=cmd_train)

    p = add("eval-detection")
    p.add_argument("--pred", required=True)
    p.add_argument("--gt", required=True)
    p.add_argument("--tolerance", type=float)
    p.set_defaults(func=cmd_eval_detection)

    p = add("validate")
    p.add_argument("--dataset", help="240fps のラベル付き動画の一覧 CSV またはディレクトリ")
    p.add_argument("--synthetic", action="store_true", help="合成データで検証する")
    p.add_argument("--n", type=int, default=100, help="合成データの本数")
    p.add_argument("--fps", type=float, nargs="+", default=[30, 60, 120, 240])
    p.add_argument("--reference", help="スピードガンなど外部計測値の CSV（shot, speed_kmh）")
    p.add_argument("--out", default="outputs/validation")
    p.set_defaults(func=cmd_validate)

    p = add("synth")
    p.add_argument("--out", required=True)
    p.add_argument("--speed", type=float, default=80.0, help="初速 [m/s]")
    p.add_argument("--fps", type=float, default=240.0)
    p.add_argument("--exposure", type=float, default=0.0, help="露光時間 [s]")
    p.set_defaults(func=cmd_synth)

    p = add("find-smashes")
    video_args(p, rng=True)
    p.add_argument("--calib", required=True)
    p.add_argument("--out", required=True, help="候補の保存先フォルダ")
    p.add_argument("--min-strength", type=float, default=0.25, help="最も強い打球音に対する割合（これ以上を候補にする）")
    p.add_argument("--select", type=int, default=8, help="最初から詳しく解析する候補にしておく数（速い順）")
    p.add_argument("--roi", help="シャトルを検出する範囲 x0,y0,x1,y1（元解像度 px）")
    p.add_argument("--audio", help="音声ファイル（動画の音声の代わりに使う）")
    p.set_defaults(func=cmd_find_smashes)

    p = add("analyze-smashes")
    p.add_argument("--batch", required=True, help="find-smashes の保存先フォルダ")
    p.add_argument("--ids", help="解析する候補の番号（カンマ区切り。省略時は選んだ候補）")
    p.add_argument("--no-overlay", action="store_true")
    p.set_defaults(func=cmd_analyze_smashes)

    p = add("summarize")
    p.add_argument("--results", required=True, help="result.csv を含むディレクトリ")
    p.add_argument("--out", default="summary.csv")
    p.set_defaults(func=cmd_summarize)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI を実行し、終了コードを返す。"""
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config if args.command != "train" else None)
    from smash_speed.errors import AnalysisError

    try:
        return int(args.func(args, cfg) or 0)
    except AnalysisError as e:
        # 想定している失敗：原因と対処だけを出す（GUI はこの行を読んで画面に出す）
        for line in e.report_lines():
            print(line, flush=True)
        return 2
    except Exception as e:  # noqa: BLE001  想定外の失敗：詳細を残しつつ、最後に要点を出す
        import traceback

        traceback.print_exc()
        print(f"[エラー] 予期しないエラーで止まりました（{type(e).__name__}: {e}）", flush=True)
        print("[対処] 上の詳細をコピーして開発者に伝えてください。区間や設定を変えると通ることがあります。", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
