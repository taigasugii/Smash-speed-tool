"""ショットの種類の正解ラベル（人が画面で直した種類）の保存と読み出し（ROADMAP B6）。

ショットの一覧（``shots.csv``）は解析し直すと置き換わるので、正解ラベルは別のファイル
（出力の場所の ``shot_labels.csv``。動画をまたいで 1 つ）に、動画の名前と打った時刻で持つ。
一覧を読むときは、同じ動画で打った時刻が ``MATCH_TOL_S`` 以内のショットにラベルを付け直す。

1 行には、ラベルのほかに、付けたときのショットの値（初速・打点・着地点・打点での速度など）と
決まりで判定した種類（``predicted``）も残す。将来の学習（B7）は、このファイルだけで特徴量と正解がそろう。
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable

from smash_speed.shots import SHOT_TYPES, Shot, _cell, _parse

LABELS_FILE = "shot_labels.csv"
MATCH_TOL_S = 0.08   # 同じショットとみなす打った時刻の差 [s]（30 fps で 2 コマほど）
# ショットの値のうち、ラベルといっしょに残すもの（学習の特徴量になるもの）
FEATURE_FIELDS = ("side", "speed_kmh", "hit_x_m", "hit_y_m", "hit_z_m", "land_x_m", "land_y_m", "t_land_s",
                  "v0_x_mps", "v0_y_mps", "v0_z_mps", "terminal_velocity_mps", "confidence", "rally_id", "index_in_rally")


@dataclass
class ShotLabel:
    """1 本のショットの正解ラベル。"""

    video_name: str                        # 動画のファイル名（PC を移っても同じになるよう、場所は含めない）
    t_hit_s: float                         # 打った時刻（ラベルを付けたときの一覧の値）
    label: str                             # 人が決めた種類（SHOT_TYPES のどれか）
    predicted: str = ""                    # 決まりで判定した種類（ラベルを付けたとき）
    video: str = ""                        # 動画の場所（参考）
    labeled_at: str = ""                   # 付けた日時
    side: str = ""
    speed_kmh: float | None = None
    hit_x_m: float | None = None
    hit_y_m: float | None = None
    hit_z_m: float | None = None
    land_x_m: float | None = None
    land_y_m: float | None = None
    t_land_s: float | None = None
    v0_x_mps: float | None = None
    v0_y_mps: float | None = None
    v0_z_mps: float | None = None
    terminal_velocity_mps: float | None = None
    confidence: float | None = None
    rally_id: int | None = None
    index_in_rally: int | None = None


LABEL_COLUMNS = [f.name for f in fields(ShotLabel)]


def video_key(path_or_name: str) -> str:
    """動画を見分ける名前（ファイル名を小文字にしたもの。Windows の \\ も区切りとみなす）。"""
    return str(path_or_name or "").replace("\\", "/").split("/")[-1].lower()


def read_labels(path: str | Path) -> list[ShotLabel]:
    """``shot_labels.csv`` を読む。なければ空。"""
    path = Path(path)
    if not path.exists():
        return []
    types = {f.name: f.type for f in fields(ShotLabel)}
    with path.open(newline="", encoding="utf-8-sig") as f:
        return [ShotLabel(**{k: _parse(v, types[k]) for k, v in row.items() if k in types}) for row in csv.DictReader(f)]


def write_labels(path: str | Path, labels: Iterable[ShotLabel]) -> Path:
    """動画の名前・打った時刻の順に書く（途中で止まっても前のファイルが壊れないよう、書いてから置き換える）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(LABEL_COLUMNS)
        for r in sorted(labels, key=lambda r: (r.video_name, r.t_hit_s)):
            d = asdict(r)
            w.writerow([_cell(d[c]) for c in LABEL_COLUMNS])
    tmp.replace(path)
    return path


def _find(labels: list[ShotLabel], video: str, t_hit_s: float, tol: float = MATCH_TOL_S) -> int | None:
    """同じ動画で打った時刻がいちばん近いラベルの番号（``tol`` より離れていれば None）。"""
    key = video_key(video)
    best, best_dt = None, tol
    for k, r in enumerate(labels):
        dt = abs(r.t_hit_s - t_hit_s)
        if r.video_name == key and dt <= best_dt:
            best, best_dt = k, dt
    return best


def set_label(path: str | Path, video: str, shot: Shot, label: str | None) -> list[ShotLabel]:
    """1 本のショットにラベルを付ける（同じショットのラベルがあれば置き換える）。``label`` が空なら消す。"""
    if label and label not in SHOT_TYPES:
        raise ValueError(f"ショットの種類 {label!r} は {SHOT_TYPES} のどれかにしてください")
    labels = read_labels(path)
    k = _find(labels, video, shot.t_hit_s)
    if k is not None:
        labels.pop(k)
    if label:
        d = asdict(shot)
        labels.append(ShotLabel(video_name=video_key(video), t_hit_s=round(float(shot.t_hit_s), 4), label=label,
                                predicted=shot.shot_type, video=str(video),
                                labeled_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                                **{f: d[f] for f in FEATURE_FIELDS}))
    write_labels(path, labels)
    return labels


def match_labels(labels: Iterable[ShotLabel], video: str, shots: list[Shot],
                 tol: float = MATCH_TOL_S) -> dict[int, str]:
    """一覧のショットの番号 → 正解ラベル。1 つのラベルは、打った時刻がいちばん近い 1 本にだけ付ける。"""
    key = video_key(video)
    out: dict[int, tuple[str, float]] = {}
    for r in labels:
        if r.video_name != key or not shots:
            continue
        dts = [abs(s.t_hit_s - r.t_hit_s) for s in shots]
        k = min(range(len(shots)), key=dts.__getitem__)
        if dts[k] <= tol and (k not in out or dts[k] < out[k][1]):
            out[k] = (r.label, dts[k])
    return {k: v[0] for k, v in out.items()}


def label_summary(labels: Iterable[ShotLabel]) -> dict[str, Any]:
    """ラベルの本数（全部・種類ごと・動画の数）と、決まりの判定が合っていた割合。"""
    labels = list(labels)
    by_type: dict[str, int] = {}
    for r in labels:
        by_type[r.label] = by_type.get(r.label, 0) + 1
    judged = [r for r in labels if r.predicted]
    agree = sum(r.predicted == r.label for r in judged)
    return {"n": len(labels), "by_type": by_type, "n_videos": len({r.video_name for r in labels}),
            "rule_accuracy": (agree / len(judged)) if judged else math.nan}
