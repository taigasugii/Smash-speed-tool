"""新しい動画をまとめて解析する（ROADMAP v2 の L3）。

動画のフォルダの中の、まだ解析していない動画を一覧にし、選んだものを順に解析する：

- 基礎打ちの解析（``smash_speed.drill``。コート合わせなしで、打球の数・時刻）。出力は ``<保存先>/drills/<動画の名前>``
- その動画のコート合わせ（``<保存先>/<動画の名前>_calibration.json``。GUI の手順 2 で保存する所）があれば、
  ショットの一覧（``smash-speed shots``）。出力は ``<保存先>/<動画の名前>_shots``（GUI と同じ）

「解析した」は、それぞれの最後に書くファイル（``drill_detect.json``・``shot_list.json``）があるかで決める。途中で止めた動画は
そのファイルがないので「まだ」のままで、もう一度始めると、その動画の最初から解析し直す（終わった動画はとばす）。
GUI の「まとめて解析」から、人が押したときだけ動く（夜のエージェントとは関係しない）。
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

VIDEO_EXTS = (".mp4", ".mov", ".m4v", ".avi", ".mkv")
DRILL_DIR = "drills"
DRILL_DONE = "drill_detect.json"       # ``drill.DETECT_INFO_FILE``
SHOTS_DONE = "shot_list.json"          # ``shot_list.SHOT_LIST_FILE``


def drill_dir(out_root: Path, stem: str) -> Path:
    return Path(out_root) / DRILL_DIR / stem


def shots_dir(out_root: Path, stem: str) -> Path:
    return Path(out_root) / f"{stem}_shots"


def calib_path(out_root: Path, stem: str) -> Path:
    return Path(out_root) / f"{stem}_calibration.json"


@dataclass
class QueueItem:
    """フォルダの中の動画 1 本と、その解析の状態。"""

    path: str
    name: str
    size_mb: float
    mtime: float
    drill_done: bool
    has_calib: bool
    shots_done: bool
    todo: list[str] = field(default_factory=list)   # まだの解析（"drill"・"shots"）

    @property
    def done(self) -> bool:
        return not self.todo

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "done": self.done}


def item_for(video: Path, out_root: Path) -> QueueItem:
    """動画 1 本の状態。"""
    video = Path(video)
    stem = video.stem
    drill = (drill_dir(out_root, stem) / DRILL_DONE).exists()
    cal = calib_path(out_root, stem).exists()
    shots = (shots_dir(out_root, stem) / SHOTS_DONE).exists()
    todo = ([] if drill else ["drill"]) + (["shots"] if cal and not shots else [])
    st = video.stat()
    return QueueItem(str(video), video.name, round(st.st_size / 2 ** 20, 1), st.st_mtime, drill, cal, shots, todo)


def list_videos(videos_dir: Path, out_root: Path) -> list[QueueItem]:
    """フォルダの中の動画（直下だけ）を、撮った・保存した順（ファイルの時刻の古い順）に。"""
    d = Path(videos_dir)
    if not d.is_dir():
        return []
    vids = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    return [item_for(p, out_root) for p in sorted(vids, key=lambda p: (p.stat().st_mtime, p.name))]


def commands(item: QueueItem, out_root: Path, cli: Callable[..., list[str]], model: bool = False,
             python: str | None = None) -> list[list[str]]:
    """動画 1 本のまだの解析のコマンド（基礎打ちの解析 → ショットの一覧の順）。

    Args:
        cli: ``smash-speed`` のサブコマンドのコマンド列を作る関数（GUI の ``server.cli``）
        model: 学習したモデル（設定の ``detection.shot_list_model``）でシャトルを探す（なければ今の検出）
        python: 基礎打ちの解析を動かす Python（省略時はこの Python）
    """
    stem = Path(item.path).stem
    out: list[list[str]] = []
    if "drill" in item.todo:
        cmd = [python or sys.executable, "-m", "smash_speed.drill", item.path, "--out", str(drill_dir(out_root, stem))]
        out.append(cmd + ([] if model else ["--baseline-detector"]))
    if "shots" in item.todo:
        cmd = cli("shots", "--video", item.path, "--calib", calib_path(out_root, stem), "--out", shots_dir(out_root, stem))
        out.append(cmd + ([] if model else ["--baseline-detector"]))
    return out
