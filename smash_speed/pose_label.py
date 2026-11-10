"""手で付けた関節・打点のコマの保存（ROADMAP E1・E1b の正解づくり。GUI の「関節・打点を付ける」）。

保存先は出力の場所の下の ``pose_labels/``：

- ``joints.csv``：手で付けた関節。形は ``pose.LABEL_COLUMNS``（``python -m smash_speed.pose pck`` がそのまま読む）。
  1 コマ 17 行で、（動画, コマ）ごとに付け直すと置き換える。
- ``joints_prefill.csv``：そのコマの下書きに使ったもの（``video,frame,prefill``。手だけで付けたなら空欄）。
  下書きのモデルに近い所で手を止めると、そのモデルの PCK が高めに出るので、比べるときに確かめられるよう残す。
- ``impacts.csv``：手で付けた打点のコマ（``IMPACT_COLUMNS``。1 行 1 スイング）。同じ動画で ``SAME_SWING_S`` 以内に
  付け直すと、同じスイングとして置き換える。

付けるコマの目安（``plan_frames``）：1 本のスイングから 10 コマ（構え・テイクバック・振り出し 2・打点の前後 3・
フォロースルー 2・戻り）。20 本で約 200 コマになる（``docs/pose_features.md`` の「精度（PCK）」）。

``python -m smash_speed.pose_label progress <joints.csv>`` で付けた数を、
``python -m smash_speed.pose_label impacts <impacts.csv> <出力の場所>`` で、フォームの解析（``smash_speed.form``）が
見積もった打点と手で付けた打点の誤差（``pose_prep.impact_errors``）と、音がないときの差 ``wrist_offset_s`` の目安を出す。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from smash_speed.pose import KEYPOINTS, LABEL_COLUMNS, NUM_KEYPOINTS, PoseSequence, clean
from smash_speed.pose_prep import calibrate_wrist_offset, find_impact, impact_errors

LABELS_DIR = "pose_labels"
JOINTS_FILE = "joints.csv"
PREFILL_FILE = "joints_prefill.csv"
IMPACTS_FILE = "impacts.csv"
IMPACT_COLUMNS = ["video", "frame", "t_s"]
TARGET_FRAMES = 200
SAME_SWING_S = 0.3        # 同じ動画でこれより近い打点は、同じスイングの付け直しとみなす [s]
TORSO_JOINTS = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")

# 1 本のスイングで付けるコマ：（打点からの時間 [s]、打点からのコマ数、呼び名）。時間とコマ数の和の所のコマ
PLAN = ((-0.8, 0, "構え"), (-0.5, 0, "テイクバック"), (-0.25, 0, "振り出し"), (-0.1, 0, "振り出し"),
        (0.0, -1, "打つ直前"), (0.0, 0, "打った瞬間"), (0.0, 1, "打った直後"),
        (0.1, 0, "フォロースルー"), (0.25, 0, "フォロースルー"), (0.45, 0, "戻り"))


def plan_frames(t_impact: float, timestamps: Iterable[float]) -> list[dict[str, Any]]:
    """打点の時刻 [s] のスイングで付けるコマ（``PLAN``）。動画の外・重なるコマは除く。"""
    ts = np.asarray(list(timestamps), float)
    if len(ts) == 0:
        return []
    i0 = int(np.argmin(np.abs(ts - t_impact)))
    out: list[dict[str, Any]] = []
    for dt, dk, phase in PLAN:
        i = int(np.argmin(np.abs(ts - (ts[i0] + dt)))) + dk if dt else i0 + dk
        if dt and abs(ts[min(max(i, 0), len(ts) - 1)] - (ts[i0] + dt)) > 0.05:
            continue  # 動画の端で、その時刻がない
        if 0 <= i < len(ts) and all(o["frame"] != i for o in out):
            out.append({"frame": i, "t": float(ts[i]), "phase": phase})
    return out


# --- 関節 ------------------------------------------------------------------------

def _atomic_write(path: Path, header: list[str], rows: Iterable[list[Any]]) -> None:
    """途中で止まっても前のファイルが壊れないよう、別の名前に書いてから置き換える。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    os.replace(tmp, path)


def read_joints(path: str | Path) -> dict[tuple[str, int], dict[str, Any]]:
    """``joints.csv`` を（動画, コマ）ごとに読む：``t_s``、``xy`` (17, 2)（付けていなければ NaN）、``visible`` (17,)。"""
    out: dict[tuple[str, int], dict[str, Any]] = {}
    path = Path(path)
    if not path.exists():
        return out
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("joint") not in KEYPOINTS:
                continue
            key = (r.get("video") or "", int(r["frame"]))
            d = out.setdefault(key, {"t_s": float(r["t_s"]) if r.get("t_s") else float("nan"),
                                     "xy": np.full((NUM_KEYPOINTS, 2), np.nan), "visible": np.zeros(NUM_KEYPOINTS, int)})
            j = KEYPOINTS.index(r["joint"])
            v = int(r.get("visible") or 0)
            if v > 0 and r.get("x") not in (None, "") and r.get("y") not in (None, ""):
                d["xy"][j] = (float(r["x"]), float(r["y"]))
                d["visible"][j] = v
    return out


def write_joints(path: str | Path, data: dict[tuple[str, int], dict[str, Any]]) -> Path:
    """（動画, コマ）の順に、1 コマ 17 行で書く（付けていない関節は visible = 0、位置は空欄）。"""
    path = Path(path)

    def rows():
        for (video, frame) in sorted(data):
            d = data[(video, frame)]
            for j, name in enumerate(KEYPOINTS):
                v = int(d["visible"][j])
                x, y = d["xy"][j]
                if v > 0 and not (np.isfinite(x) and np.isfinite(y)):
                    v = 0
                yield [video, frame, f"{d['t_s']:.6f}", name, f"{x:.2f}" if v else "", f"{y:.2f}" if v else "", v]

    _atomic_write(path, LABEL_COLUMNS, rows())
    return path


def check_joints(joints: Any) -> tuple[np.ndarray, np.ndarray]:
    """画面から来た 17 点（``[[x, y, visible], ...]``、付けていない点は null か visible = 0）を確かめて配列にする。"""
    if not isinstance(joints, list) or len(joints) != NUM_KEYPOINTS:
        raise ValueError(f"関節は {NUM_KEYPOINTS} 点の並びにしてください")
    xy = np.full((NUM_KEYPOINTS, 2), np.nan)
    vis = np.zeros(NUM_KEYPOINTS, int)
    for j, p in enumerate(joints):
        if not p:
            continue
        x, y, v = float(p[0]), float(p[1]), int(p[2])
        if v not in (0, 1, 2) or not (np.isfinite(x) and np.isfinite(y)):
            raise ValueError(f"{KEYPOINTS[j]} の値が正しくありません: {p!r}")
        if v:
            xy[j], vis[j] = (x, y), v
    return xy, vis


def save_frame(labels_dir: str | Path, video: str, frame: int, t_s: float, xy: np.ndarray, visible: np.ndarray,
               prefill: str = "") -> dict[str, Any]:
    """1 コマの関節を保存する（同じ（動画, コマ）は置き換え）。保存後の進み具合を返す。"""
    d = Path(labels_dir)
    data = read_joints(d / JOINTS_FILE)
    data[(video, int(frame))] = {"t_s": float(t_s), "xy": np.asarray(xy, float), "visible": np.asarray(visible, int)}
    write_joints(d / JOINTS_FILE, data)
    pre = read_prefill(d / PREFILL_FILE)
    pre[(video, int(frame))] = prefill or ""
    _atomic_write(d / PREFILL_FILE, ["video", "frame", "prefill"], ([v, f, p] for (v, f), p in sorted(pre.items())))
    return progress(data)


def delete_frame(labels_dir: str | Path, video: str, frame: int) -> dict[str, Any]:
    d = Path(labels_dir)
    data = read_joints(d / JOINTS_FILE)
    if data.pop((video, int(frame)), None) is not None:
        write_joints(d / JOINTS_FILE, data)
    pre = read_prefill(d / PREFILL_FILE)
    if pre.pop((video, int(frame)), None) is not None:
        _atomic_write(d / PREFILL_FILE, ["video", "frame", "prefill"], ([v, f, p] for (v, f), p in sorted(pre.items())))
    return progress(data)


def read_prefill(path: str | Path) -> dict[tuple[str, int], str]:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {(r["video"], int(r["frame"])): r.get("prefill") or "" for r in csv.DictReader(f)}


def progress(data: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    """付けたコマの数。``usable`` は PCK に数えられるコマ（肩・腰の 4 点がそろっている）。"""
    torso = [KEYPOINTS.index(n) for n in TORSO_JOINTS]
    per: dict[str, int] = {}
    usable = 0
    for (video, _), d in data.items():
        per[video] = per.get(video, 0) + 1
        usable += bool((d["visible"][torso] > 0).all())
    return {"frames": len(data), "usable": usable, "target": TARGET_FRAMES, "per_video": per}


# --- 打点 ------------------------------------------------------------------------

def read_impacts(path: str | Path, video: str | None = None) -> list[dict[str, Any]]:
    """``impacts.csv`` を読む（``video`` を渡せばその動画だけ）。動画・時刻の順。"""
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = [{"video": r["video"], "frame": int(r["frame"]), "t_s": float(r["t_s"])} for r in csv.DictReader(f)]
    rows = [r for r in rows if video is None or r["video"] == video]
    return sorted(rows, key=lambda r: (r["video"], r["t_s"]))


def _write_impacts(path: Path, rows: list[dict[str, Any]]) -> None:
    rows = sorted(rows, key=lambda r: (r["video"], r["t_s"]))
    _atomic_write(path, IMPACT_COLUMNS, ([r["video"], r["frame"], f"{r['t_s']:.6f}"] for r in rows))


def add_impact(labels_dir: str | Path, video: str, frame: int, t_s: float,
               same_swing_s: float = SAME_SWING_S) -> list[dict[str, Any]]:
    """打点のコマを足す。同じ動画で ``same_swing_s`` 以内の打点は、同じスイングの付け直しとして置き換える。"""
    path = Path(labels_dir) / IMPACTS_FILE
    rows = [r for r in read_impacts(path) if not (r["video"] == video and abs(r["t_s"] - t_s) <= same_swing_s)]
    rows.append({"video": video, "frame": int(frame), "t_s": float(t_s)})
    _write_impacts(path, rows)
    return read_impacts(path, video)


def delete_impact(labels_dir: str | Path, video: str, frame: int) -> list[dict[str, Any]]:
    path = Path(labels_dir) / IMPACTS_FILE
    rows = read_impacts(path)
    keep = [r for r in rows if not (r["video"] == video and r["frame"] == int(frame))]
    if len(keep) != len(rows):
        _write_impacts(path, keep)
    return read_impacts(path, video)


def form_results(root: str | Path) -> list[dict[str, Any]]:
    """出力の場所の下のフォームの解析の結果（``*/form/``）：動画の名前・利き手・ショットの行。"""
    from smash_speed.form import FEATURES_FILE, FORM_DIR, INFO_FILE, read_rows

    root = Path(root)
    out = []
    for p in sorted(set(root.glob(f"*/{FORM_DIR}/{FEATURES_FILE}")) | set(root.glob(f"{FORM_DIR}/{FEATURES_FILE}"))):
        d = p.parent
        try:
            info = json.loads((d / INFO_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            info = {}
        rows = read_rows(p)
        video = Path(info.get("video") or (rows[0].get("video") if rows else "") or "").name
        left = bool((info.get("options") or {}).get("left_handed"))
        out.append({"dir": d, "video": video, "left_handed": left, "rows": rows})
    return out


def impact_report(impacts: list[dict[str, Any]], root: str | Path, near_s: float = SAME_SWING_S) -> dict[str, Any]:
    """手で付けた打点と、フォームの解析が見積もった打点（``form_features.csv`` の ``t_impact_s``）の誤差。

    見積もりは、手の打点から ``near_s`` 以内のいちばん近いショット。手首の速さが最大の時刻は、推定したままの姿勢
    （``raw/<ショットの ID>.npz``）から解析と同じ決め方で出し直し、手の打点との差の中央値を ``wrist_offset_s`` の目安にする。
    """
    est, manual, wrist, methods, fps = [], [], [], [], []
    unmatched = 0
    results = form_results(root)
    for imp in impacts:
        cands = [(abs(float(r["t_impact_s"]) - imp["t_s"]), r, res) for res in results if res["video"] == imp["video"]
                 for r in res["rows"] if r.get("t_impact_s") is not None]
        cands = [c for c in cands if c[0] <= near_s]
        if not cands:
            unmatched += 1
            continue
        _, row, res = min(cands, key=lambda c: c[0])
        est.append(float(row["t_impact_s"]))
        manual.append(imp["t_s"])
        methods.append(row.get("impact_method") or "")
        t_w = None
        raw = res["dir"] / "raw" / f"{row['shot_id']}.npz"
        if raw.exists():
            seq = PoseSequence.load(raw)
            fps.append(seq.fps)
            got = find_impact(clean(seq), None, "left" if res["left_handed"] else "right",
                              near_s=float(row.get("t_hit_s") or row["t_impact_s"]))
            t_w = got.t_wrist if got is not None else None
        wrist.append(t_w)
    out = impact_errors(est, manual, float(np.median(fps)) if fps else None)
    out.update(n_manual=len(impacts), unmatched=unmatched,
               methods={m: methods.count(m) for m in sorted(set(methods))},
               wrist_offset_s=calibrate_wrist_offset(wrist, manual) if any(w is not None for w in wrist) else None,
               n_wrist=sum(w is not None for w in wrist))
    return out


# --- CLI ------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="手で付けた関節・打点のコマ（GUI の「関節・打点を付ける」）のまとめ")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("progress", help="付けた関節のコマの数")
    g.add_argument("joints")
    c = sub.add_parser("impacts", help="手で付けた打点と、フォームの解析の打点の誤差")
    c.add_argument("impacts")
    c.add_argument("root", help="フォームの解析の結果がある出力の場所")
    c.add_argument("--video", default=None, help="この動画だけ")
    args = p.parse_args(argv)

    if args.cmd == "progress":
        pr = progress(read_joints(args.joints))
        print(f"付けたコマ {pr['frames']} / 目安 {pr['target']}（PCK に数えられるコマ {pr['usable']}）")
        for v, n in sorted(pr["per_video"].items()):
            print(f"  {v}: {n} コマ")
        return 0
    imps = read_impacts(args.impacts, args.video)
    r = impact_report(imps, args.root)
    print(f"手で付けた打点 {r['n_manual']}、解析のショットと組にできた {r['n']}（できなかった {r['unmatched']}）")
    if r["n"]:
        frames = f"（{r['mean_abs_frames']:.2f} コマ）" if r.get("mean_abs_frames") is not None else ""
        print(f"誤差（解析 − 手）: 平均の絶対値 {r['mean_abs_s'] * 1000:.1f} ms{frames}、中央値 {r['median_s'] * 1000:+.1f} ms、"
              f"最大 {r['max_abs_s'] * 1000:.1f} ms")
        print("打点の決め方: " + "、".join(f"{k or '不明'} {v}" for k, v in r["methods"].items()))
    if r["wrist_offset_s"] is not None:
        print(f"音がないときの差 wrist_offset_s の目安: {r['wrist_offset_s']:+.3f} s（{r['n_wrist']} 本の中央値）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
