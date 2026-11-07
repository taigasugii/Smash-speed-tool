"""解析用の表（ROADMAP E7）：1 ショット 1 行で、フォームの特徴量・お手本との類似度スコア・打球の結果を並べる。

``<出力先>/form/`` の次のファイルを、ショットの ID でつないで ``form/analysis.csv`` に書き出す。

- ``form_features.csv``（E2・E3、``smash_speed.form``）：ショットの ID・動画・時刻・打点の決め方・特徴量
- ``scores.csv``（E5、``smash_speed.form_score``）：お手本ごとの点。表には 1 つのお手本の点だけを入れる（``--ref-name``、
  なければ最初のお手本）
- ``ball.csv``（E6、``smash_speed.side_clear.analyze_balls``）：初速・打ち出し角・最高点・落ちた所

採点などの正解のラベルは、人があとで ``shot_id`` でつなぐ（ここでは作らない）。モデルの学習・統計の検定もここでは行わない。
列の意味は ``COLUMN_INFO``（``analysis_columns.csv`` にも書き出す）と ``docs/pose_features.md``。

``python -m smash_speed.form_table <出力先> [--ref お手本.npz ...] [--stature 1.70 --hitter-from-net 5.5]``
（``--ref`` があれば先に点を付け直し、``--stature`` があれば先に打球の結果を出し直す）
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from smash_speed.form import BASE_COLUMNS, FEATURES_FILE, FORM_DIR, read_rows, write_rows
from smash_speed.pose_features import FEATURE_INFO, FEATURE_NAMES
from smash_speed.side_clear import BALL_COLUMNS, BALL_FILE

ANALYSIS_FILE = "analysis.csv"
COLUMNS_FILE = "analysis_columns.csv"
SCORE_COLUMNS = ["reference", "score_dtw", "score_cos", "cos_angle_deg", "dtw_deg", "lockstep_deg"]
COLUMNS = BASE_COLUMNS + FEATURE_NAMES + SCORE_COLUMNS + [c for c in BALL_COLUMNS if c != "shot_id"]

# 列の説明（単位）。特徴量は FEATURE_INFO に足りないものを補う
COLUMN_INFO: dict[str, str] = {
    "shot_id": "ショットの ID（<動画の名前>_<打点の時刻の 1/100 秒>）。正解のラベルをつなぐ鍵",
    "video": "動画のファイル名",
    "t_hit_s": "打球の候補の時刻 [s]",
    "t_impact_s": "打点の時刻 [s]（動画の実時間）",
    "impact_method": "打点の決め方（audio 打球音 / wrist 手首の速さ / given 渡された時刻）",
    "model": "姿勢推定のモデル",
    "left_handed": "左利き（反転して出した）",
    "body_height_torso": "身長（胴体の長さの何倍か）",
    "shoulder_width_impact": "打点での肩幅の見かけ ÷ 最大",
    "hip_width_impact": "打点での腰幅の見かけ ÷ 最大",
    "t_twist_max": "捻りが最大の時刻 [s]（打点が 0）",
    "pelvis_speed_dps": "腰の回旋の角速度の最大 [deg/s]",
    "trunk_speed_dps": "胸の回旋の角速度の最大 [deg/s]",
    "elbow_speed_dps": "肘が伸びる角速度の最大 [deg/s]",
    "chain_order": "速さの最大の早い順",
    "lag_pelvis_trunk": "腰 → 肩の最大の時間差 [s]",
    "lag_trunk_elbow": "肩 → 肘の最大の時間差 [s]",
    "lag_elbow_wrist": "肘 → 手首の最大の時間差 [s]",
    "step_speed_max": "腰の中点が前へ動く速さの最大 [身長/s]",
    "reference": "スコアのお手本の名前",
    "score_dtw": "お手本との類似度（DTW、0〜100）",
    "score_cos": "お手本との類似度（姿勢のベクトルのコサイン、0〜100）",
    "cos_angle_deg": "骨の向きのずれの目安 [deg]",
    "dtw_deg": "DTW で合わせた後の角度の差の平均 [deg]",
    "lockstep_deg": "時間をずらさずに比べた角度の差の平均 [deg]",
    "ball_speed_kmh": "シャトルの初速 [km/h]",
    "ball_elevation_deg": "打ち出し角 [deg]（上向きが正）",
    "ball_hit_height_m": "打点の高さ（シャトルの軌道から）[m]",
    "apex_height_m": "最高点の高さ [m]",
    "apex_from_hitter_m": "最高点の、打つ人からの距離 [m]",
    "land_from_hitter_m": "落ちた所の、打つ人からの距離 [m]",
    "land_depth_m": "落ちた所の、相手側のネットからの深さ [m]（打つ人のネットからの距離が分かるときだけ）",
    "ball_flight_s": "飛んでいた時間 [s]",
    "ball_n_obs": "当てはめに使ったシャトルのコマの数",
    "ball_rms_px": "当てはめの画像の上のずれ [px]",
    "ball_in_frame_s": "シャトルが画面の中に見えていた時間（打点から）[s]。短いほど最高点・落ちた所は物理モデルで伸ばした値",
    "ball_note": "打球の結果を出せなかった理由",
}
for _k, (_name, _unit) in FEATURE_INFO.items():
    COLUMN_INFO.setdefault(_k, f"{_name}{f' [{_unit}]' if _unit else ''}")


def _by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(r["shot_id"]): r for r in rows if r.get("shot_id")}


def build_table(form_dir: str | Path, ref_name: str | None = None) -> list[dict[str, Any]]:
    """``form/`` の結果をショットの ID でつないだ行（打点の時刻の順）。"""
    d = Path(form_dir)
    d = d / FORM_DIR if (d / FORM_DIR).is_dir() else d
    rows = read_rows(d / FEATURES_FILE)
    scores: dict[str, dict[str, Any]] = {}
    if (d / "scores.csv").exists():
        srows = read_rows(d / "scores.csv")
        names = [r["reference"] for r in srows]
        use = ref_name if ref_name in names else (names[0] if names else None)
        scores = _by_id([r for r in srows if r["reference"] == use])
    balls = _by_id(read_rows(d / BALL_FILE)) if (d / BALL_FILE).exists() else {}
    out = []
    for r in rows:
        sid = str(r["shot_id"])
        s = {k: v for k, v in scores.get(sid, {}).items() if k != "shot_id"}
        b = {k: v for k, v in balls.get(sid, {}).items() if k != "shot_id"}
        out.append({**r, **s, **b})
    out.sort(key=lambda r: (str(r.get("video")), float(r.get("t_impact_s") or 0.0)))
    return out


def write_table(form_dir: str | Path, ref_name: str | None = None) -> Path:
    d = Path(form_dir)
    d = d / FORM_DIR if (d / FORM_DIR).is_dir() else d
    rows = build_table(d, ref_name)
    path = write_rows(d / ANALYSIS_FILE, rows, COLUMNS)
    with (d / COLUMNS_FILE).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["column", "description"])
        for c in COLUMNS:
            w.writerow([c, COLUMN_INFO.get(c, "")])
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="フォームの解析の結果を 1 ショット 1 行の表にする")
    p.add_argument("out_dir")
    p.add_argument("--ref", action="append", default=[], help="お手本（.npz）。あれば点を付け直す")
    p.add_argument("--ref-name", default=None, help="表に入れるお手本の名前（なければ最初のお手本）")
    p.add_argument("--stature", type=float, default=None, help="打つ人の身長 [m]。あれば打球の結果を出し直す")
    p.add_argument("--hitter-from-net", type=float, default=None, help="打つ人のネットからの距離 [m]")
    args = p.parse_args(argv)
    d = Path(args.out_dir)
    d = d / FORM_DIR if (d / FORM_DIR).is_dir() else d
    if args.ref:
        from smash_speed.form_score import main as score_main
        score_main(["score", str(d), *sum((["--ref", r] for r in args.ref), [])])
    if args.stature:
        from smash_speed.side_clear import analyze_balls
        rows = analyze_balls(d, args.stature, args.hitter_from_net)
        print(f"打球の結果：{sum(1 for r in rows if r.get('ball_speed_kmh') is not None)}/{len(rows)} 本")
    path = write_table(d, args.ref_name)
    print(f"表を書き出しました: {path}（列の説明 {d / COLUMNS_FILE}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
