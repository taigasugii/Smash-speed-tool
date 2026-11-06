"""実動画のシャトルの正解を、エージェント（または人）が画像を見て付けるための道具（ROADMAP F1a）。

今の検出（背景との差、``shot_list.detect_all``）が怪しいコマを選び、元の解像度で候補の周りを切り出して
前後のコマと並べた画像（見本帳）を書く。見本帳を見て 1 コマずつ答え（``answers.csv``）を書き、``apply`` で
正解ラベル（``annotate`` と同じ ``labels.csv``・TrackNetV3 形式の ``<動画名>_ball.csv``）にする。

付けるのは「このコートのラリーのシャトル」の位置。隣のコートのシャトル・ラケット・人は、シャトルではないとする。

選ぶコマ（``reason``。1 コマに複数あれば ``+`` でつなぐ）::

    gap        選んだ軌道が数コマ途切れた所（予測した位置の周りを切り出す）と、その前後
    jump       選んだ軌道の点が急に跳んだ所の前後
    rejected   ``select_tracks`` で負けた軌道・絞り込みで落ちた軌道の点（本当はシャトルかもしれない）
    track_end  選んだ軌道の始まりと終わり（打った瞬間・画面の外へ出る所）
    edge       画面の端（上・左右）に近い点
    normal     選んだ軌道の途中のふつうのコマ（全体の 2〜3 割。学習が偏らないように）

答え（``answers.csv`` の ``id,answer``）::

    ok         印（候補）がシャトル（位置は前後のコマとの差で少し寄せる）
    p X Y      シャトルは (X, Y)（元の解像度の画素。見本帳の 50 px ごとの線から読む。差で寄せる）
    n          このコマにシャトルは写っていない（画面の外・完全に隠れている）
    x          印はシャトルではない（本当のシャトルは切り出しの中にない。コマの正解にはしないが、誤検出として数える）
    o          印は隣のコートのシャトル（位置は残すが、このコートの正解にはしない。学習での扱いは F1b で決める）
    u          分からない（学習にも答え合わせにも使わない）

使い方::

    python -m smash_speed.train.agent_labels pick IMG_5109.MP4 --camera clips:IMG_5109 --out outputs/train_labels/IMG_5109 --n 150
    （見本帳 outputs/train_labels/IMG_5109/sheets/*.jpg を見て answers.csv を書く）
    python -m smash_speed.train.agent_labels apply outputs/train_labels/IMG_5109
    （日を変えて 1 割を付け直す：recheck/sheets を見て recheck/answers.csv を書き、前の答えと比べる）
    python -m smash_speed.train.agent_labels recheck pick outputs/train_labels/IMG_5109
    python -m smash_speed.train.agent_labels recheck compare outputs/train_labels/IMG_5109
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CANDIDATES_FILE = "candidates.csv"
ANSWERS_FILE = "answers.csv"
REASONS = ("gap", "jump", "rejected", "track_end", "edge", "normal")


# =============================================================================
# 怪しいコマを選ぶ
# =============================================================================
def pick_frames(det_all: pd.DataFrame, tab: pd.DataFrame, width: int, height: int, n: int = 150,
                normal_frac: float = 0.25, seed: int = 0, exclude: set[int] | None = None) -> pd.DataFrame:
    """怪しいコマと、ふつうのコマを選ぶ。

    Args:
        det_all: 重なった軌道も残した検出（``detect_all(exclusive=False)``。列 frame, timestamp, x_px, y_px,
            visible, confidence, track_id）
        tab: ``track_table`` の表に ``selected`` の列を足したもの（``shuttle_tracks`` の 2 つ目）
        n: 選ぶコマの数（ふつうのコマを含む）
        exclude: 選ばないコマ（もう付けたコマ）

    Returns:
        列 frame, timestamp, reason, cand_x, cand_y（候補。予測した位置のときは ``predicted`` が真）, predicted
    """
    exclude = exclude or set()
    rng = np.random.default_rng(seed)
    frames = det_all[["frame", "timestamp"]].drop_duplicates("frame").set_index("frame").timestamp
    vis = det_all[(det_all.visible > 0) & det_all.x_px.notna()]
    sel_ids = set(tab.track_id[tab.selected]) if len(tab) else set()
    kept_ids = set(tab.track_id[tab.keep]) if len(tab) else set()
    sel = vis[vis.track_id.isin(sel_ids)].sort_values(["frame", "confidence"], ascending=[True, False])
    sel = sel.drop_duplicates("frame")
    rows: dict[int, dict[str, Any]] = {}

    def add(f: int, reason: str, x: float, y: float, predicted: bool = False) -> None:
        f = int(f)
        if f in exclude or f not in frames.index:
            return
        r = rows.setdefault(f, {"frame": f, "reasons": [], "cand_x": x, "cand_y": y, "predicted": predicted})
        if reason not in r["reasons"]:
            r["reasons"].append(reason)
        # 予測の近くに検出の点があれば、そちらを候補にする（遠い点は別の物のことが多い。切り出しには赤い印で入る）
        if r["predicted"] and not predicted and np.hypot(x - r["cand_x"], y - r["cand_y"]) < 0.05 * W:
            r.update(cand_x=x, cand_y=y, predicted=False)

    W = float(width)
    for tid, g in sel.groupby("track_id"):
        g = g.sort_values("frame")
        f, xy = g.frame.to_numpy(int), g[["x_px", "y_px"]].to_numpy(float)
        add(f[0], "track_end", *xy[0])
        add(f[-1], "track_end", *xy[-1])
        step = np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.maximum(np.diff(f), 1)
        med = float(np.median(step)) if len(step) else 0.0
        for k in range(len(f) - 1):
            df = f[k + 1] - f[k]
            if 2 <= df <= 8:  # 途切れ：真ん中のコマを予測した位置で、前後を検出の点で
                m = (f[k] + f[k + 1]) // 2
                w = (m - f[k]) / df
                add(m, "gap", *(xy[k] + w * (xy[k + 1] - xy[k])), predicted=True)
                add(f[k], "gap", *xy[k])
                add(f[k + 1], "gap", *xy[k + 1])
            if step[k] > max(3.0 * med, 0.03 * W):
                add(f[k], "jump", *xy[k])
                add(f[k + 1], "jump", *xy[k + 1])
    # 選ばれなかった軌道の点（keep で負けたものを先に）
    other = vis[~vis.track_id.isin(sel_ids)]
    if len(other):
        pri = other.track_id.isin(kept_ids).to_numpy()
        for part in (other[pri], other[~pri]):
            for tid, g in part.groupby("track_id"):
                g = g.sort_values("frame")
                for r in g.iloc[[len(g) // 2]].itertuples() if len(g) < 6 else g.iloc[[1, len(g) // 2, -2]].itertuples():
                    add(r.frame, "rejected", r.x_px, r.y_px)
    m = 0.03
    edge = sel[(sel.x_px < m * W) | (sel.x_px > (1 - m) * W) | (sel.y_px < m * height)]
    for r in edge.itertuples():
        add(r.frame, "edge", r.x_px, r.y_px)

    sus = pd.DataFrame(list(rows.values()), columns=["frame", "reasons", "cand_x", "cand_y", "predicted"])
    n_normal = int(round(n * normal_frac))
    n_sus = n - n_normal
    if len(sus) > n_sus:
        # 理由ごとにまんべんなく（少ない理由から順に、それぞれ同じ数まで）
        sus["first"] = [min((REASONS.index(x) for x in r), default=0) for r in sus.reasons]
        order = rng.permutation(len(sus))
        sus = sus.iloc[order]
        sus["rank"] = sus.groupby("first").cumcount()
        sus = sus.sort_values(["rank", "first"], kind="stable").head(n_sus).drop(columns=["first", "rank"])
    pool = sel[~sel.frame.isin(set(sus.frame)) & ~sel.frame.isin(exclude)]
    pool = pool.iloc[rng.permutation(len(pool))[:max(0, n - len(sus))]]
    normal = pd.DataFrame({"frame": pool.frame.to_numpy(int), "reasons": [["normal"]] * len(pool),
                           "cand_x": pool.x_px.to_numpy(float), "cand_y": pool.y_px.to_numpy(float),
                           "predicted": False})
    out = pd.concat([sus, normal], ignore_index=True) if len(normal) else sus
    out["reason"] = ["+".join(r) for r in out.reasons]
    out["timestamp"] = frames.loc[out.frame.to_numpy(int)].to_numpy(float)
    out = out.sort_values("frame").reset_index(drop=True)
    return out[["frame", "timestamp", "reason", "cand_x", "cand_y", "predicted"]]


# =============================================================================
# 見本帳
# =============================================================================
def _crop(img: np.ndarray, cx: float, cy: float, half: int, zoom: float) -> tuple[np.ndarray, int, int]:
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    pad = cv2.copyMakeBorder(img, half, half, half, half, cv2.BORDER_CONSTANT)
    c = pad[y0 + half:y0 + 3 * half, x0 + half:x0 + 3 * half]
    return cv2.resize(c, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST), x0, y0


def _grid(tile: np.ndarray, x0: int, y0: int, zoom: float, labels: bool) -> None:
    """元の画像の 50 px ごとの線（100 px ごとは明るく）と、その座標の数字。"""
    h, w = tile.shape[:2]
    for k in range((x0 // 50) * 50, x0 + int(w / zoom) + 50, 50):
        x = int(round((k - x0) * zoom))
        if 0 <= x < w:
            cv2.line(tile, (x, 0), (x, h), (0, 160, 160) if k % 100 else (0, 255, 255), 1)
            if labels and k % 100 == 0:
                cv2.putText(tile, str(k), (x + 2, h - 4), 0, 0.35, (0, 255, 255), 1)
    for k in range((y0 // 50) * 50, y0 + int(h / zoom) + 50, 50):
        y = int(round((k - y0) * zoom))
        if 0 <= y < h:
            cv2.line(tile, (0, y), (w, y), (0, 160, 160) if k % 100 else (0, 255, 255), 1)
            if labels and k % 100 == 0:
                cv2.putText(tile, str(k), (2, y - 2), 0, 0.35, (0, 255, 255), 1)


def _cross(tile: np.ndarray, u: float, v: float, col: tuple[int, int, int], r0: int = 6, r1: int = 14) -> None:
    """真ん中を空けた十字（シャトルを隠さない）。"""
    u, v = int(round(u)), int(round(v))
    for a, b in (((-r1, 0), (-r0, 0)), ((r0, 0), (r1, 0)), ((0, -r1), (0, -r0)), ((0, r0), (0, r1))):
        cv2.line(tile, (u + a[0], v + a[1]), (u + b[0], v + b[1]), col, 1)


def item_tiles(get_frame, item: pd.Series, det_all: pd.DataFrame, half: int, zoom: float) -> np.ndarray:
    """1 コマ分：前・このコマ・次のコマの切り出し（同じ範囲）。このコマには候補（水色、予測なら紫）と、
    ほかの検出の点（赤）の印を付ける。"""
    f = int(item.frame)
    tiles = []
    for d in (-1, 0, 1):
        try:
            img = get_frame(f + d)
        except IndexError:
            img = None
        if img is None:
            tiles.append(np.zeros((int(2 * half * zoom),) * 2 + (3,), np.uint8))
            continue
        t, x0, y0 = _crop(img, item.cand_x, item.cand_y, half, zoom)
        _grid(t, x0, y0, zoom, labels=d == 0)
        if d == 0:
            pts = det_all[(det_all.frame == f) & (det_all.visible > 0) & det_all.x_px.notna()]
            for p in pts.itertuples():
                if np.hypot(p.x_px - item.cand_x, p.y_px - item.cand_y) > 2:
                    _cross(t, (p.x_px - x0) * zoom, (p.y_px - y0) * zoom, (0, 0, 255), 4, 9)
            _cross(t, (item.cand_x - x0) * zoom, (item.cand_y - y0) * zoom,
                   (255, 0, 255) if item.predicted else (255, 255, 0))
        cv2.putText(t, f"{f + d}", (3, 13), 0, 0.42, (255, 255, 255), 1)
        tiles.append(cv2.copyMakeBorder(t, 0, 0, 0, 2, cv2.BORDER_CONSTANT, value=(80, 80, 80)))
    row = np.hstack(tiles)
    head = np.zeros((18, row.shape[1], 3), np.uint8)
    cv2.putText(head, f"#{int(item.id)}  {item.reason}", (3, 13), 0, 0.45, (255, 255, 255), 1)
    return np.vstack([head, row])


def write_sheets(get_frame, items: pd.DataFrame, det_all: pd.DataFrame, out_dir: Path, half: int = 120,
                 zoom: float = 1.0, per_row: int = 2, rows: int = 4) -> list[Path]:
    """見本帳（1 枚に ``per_row × rows`` コマ）を書く。``items`` には列 ``id`` が要る。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    per = per_row * rows
    for s in range(0, len(items), per):
        part = [item_tiles(get_frame, it, det_all, half, zoom) for _, it in items.iloc[s:s + per].iterrows()]
        blank = np.zeros_like(part[0])
        part += [blank] * (-len(part) % per_row)
        part = [cv2.copyMakeBorder(p, 0, 4, 0, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for p in part]
        img = np.vstack([np.hstack(part[k:k + per_row]) for k in range(0, len(part), per_row)])
        p = out_dir / f"sheet_{s // per:03d}.jpg"
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        paths.append(p)
    return paths


def write_check_sheets(get_frame, labels: pd.DataFrame, out_dir: Path, half: int = 40, zoom: float = 3.0,
                       per_row: int = 6, rows: int = 4) -> list[Path]:
    """見える正解だけを、そのコマを拡大して並べる（付けた位置がシャトルの真ん中に乗っているかを見直す）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    vis = labels[labels.visible == 1]
    paths, per = [], per_row * rows
    for s in range(0, len(vis), per):
        tiles = []
        for r in vis.iloc[s:s + per].itertuples():
            t, x0, y0 = _crop(get_frame(int(r.frame)), r.x_px, r.y_px, half, zoom)
            _cross(t, (r.x_px - x0) * zoom, (r.y_px - y0) * zoom, (255, 255, 0), 8, 18)
            cv2.putText(t, f"{int(r.frame)}", (3, 13), 0, 0.42, (255, 255, 255), 1)
            tiles.append(cv2.copyMakeBorder(t, 0, 2, 0, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % per_row)
        img = np.vstack([np.hstack(tiles[k:k + per_row]) for k in range(0, len(tiles), per_row)])
        p = out_dir / f"check_{s // per:03d}.jpg"
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        paths.append(p)
    return paths


# =============================================================================
# 答え → 正解ラベル
# =============================================================================
def snap(get_frame, f: int, x: float, y: float, r: int = 10, min_diff: float = 12.0) -> tuple[float, float, bool]:
    """(x, y) の近く（半径 r）で、前後のコマとの差（動くもの）の塊の重心に寄せる。塊がなければそのまま。

    Returns:
        (x, y, 寄せたか)
    """
    try:
        cur, prev, nxt = (cv2.cvtColor(get_frame(i), cv2.COLOR_BGR2GRAY).astype(np.float32) for i in (f, f - 1, f + 1))
    except (IndexError, cv2.error, TypeError):
        return x, y, False
    d = np.minimum(np.abs(cur - prev), np.abs(cur - nxt))
    x0, y0 = int(max(0, round(x) - r)), int(max(0, round(y) - r))
    patch = d[y0:int(round(y)) + r + 1, x0:int(round(x)) + r + 1]
    if patch.size == 0 or patch.max() < min_diff:
        return x, y, False
    my, mx = np.unravel_index(np.argmax(patch), patch.shape)
    lab = cv2.connectedComponents((patch > 0.5 * patch[my, mx]).astype(np.uint8))[1]
    sel = lab == lab[my, mx]
    wy, wx = np.nonzero(sel)
    w = patch[sel]
    return float((wx * w).sum() / w.sum() + x0), float((wy * w).sum() / w.sum() + y0), True


def parse_answer(text: str) -> tuple[str, float, float]:
    """``ok`` / ``p X Y`` / ``n`` / ``x`` / ``o`` / ``u`` → (種類, X, Y)。"""
    s = str(text).strip().lower().split()
    if not s:
        raise ValueError("空の答え")
    if s[0] == "p" and len(s) == 3:
        return "p", float(s[1]), float(s[2])
    if s[0] in ("ok", "n", "x", "o", "u") and len(s) == 1:
        return s[0], np.nan, np.nan
    raise ValueError(f"答えの形が違います: {text!r}")


def answers_to_labels(cands: pd.DataFrame, answers: pd.DataFrame, get_frame=None) -> pd.DataFrame:
    """候補と答えから、``labels.csv`` の形（frame, timestamp, x_px, y_px, visible, event, source, reason）にする。

    ``visible``：1 見える・0 写っていない・-1 分からない（使わない。``x``・``o`` もここ。``o`` は位置を残す）。``get_frame`` があれば位置を差で寄せる。
    """
    a = cands.merge(answers[["id", "answer"]], on="id", how="inner")
    rows = []
    for r in a.itertuples():
        kind, x, y = parse_answer(r.answer)
        src = f"agent_{kind}"
        if kind in ("ok", "o"):
            x, y = float(r.cand_x), float(r.cand_y)
        if kind in ("ok", "p", "o"):
            snapped = False
            if get_frame is not None:
                x, y, snapped = snap(get_frame, int(r.frame), x, y)
            vis = -1 if kind == "o" else 1
            src += "_snap" if snapped else ""
        else:
            x = y = np.nan
            vis = 0 if kind == "n" else -1
        rows.append({"frame": int(r.frame), "timestamp": float(r.timestamp), "x_px": x, "y_px": y,
                     "visible": vis, "event": "", "source": src, "reason": r.reason})
    return pd.DataFrame(rows, columns=["frame", "timestamp", "x_px", "y_px", "visible", "event", "source", "reason"])


def merge_labels(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """同じコマは新しい方に置き換えて、コマの順に並べる。"""
    if old is None or old.empty:
        out = new
    else:
        out = pd.concat([old[~old.frame.isin(set(new.frame))], new], ignore_index=True)
    return out.sort_values("frame", kind="stable").reset_index(drop=True)


def save_labels(labels: pd.DataFrame, out_dir: Path, video_name: str) -> None:
    """``labels.csv`` と、使えるコマ（見える・写っていない）だけの TrackNetV3 形式 ``<動画名>_ball.csv``。

    コマはとびとびなので、``_ball.csv`` の Frame は元の動画のコマ番号のまま（区間の先頭を 0 にしない）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    labels.to_csv(out_dir / "labels.csv", index=False)
    use = labels[labels.visible >= 0]
    tn = pd.DataFrame({"Frame": use.frame.astype(int), "Visibility": (use.visible == 1).astype(int),
                       "X": np.where(use.visible == 1, use.x_px, 0.0).round(1),
                       "Y": np.where(use.visible == 1, use.y_px, 0.0).round(1)})
    tn.to_csv(out_dir / f"{video_name}_ball.csv", index=False)


def summary(labels: pd.DataFrame) -> dict[str, Any]:
    """付けた数（見える・写っていない・分からない）と、理由ごとの「候補が正しかった」割合。"""
    out: dict[str, Any] = {"n": len(labels), "visible": int((labels.visible == 1).sum()),
                           "not_visible": int((labels.visible == 0).sum()), "unknown": int((labels.source == "agent_u").sum()),
                           "not_shuttle": int((labels.source == "agent_x").sum()),
                           "other_court": int(labels.source.str.fullmatch(r"agent_o(_snap)?").sum())}
    by = {}
    for reason in REASONS:
        m = labels.reason.str.contains(reason, regex=False)
        if m.any():
            s = labels.source[m]
            by[reason] = {"n": int(m.sum()), "candidate_ok": int(s.str.startswith("agent_ok").sum()),
                          "moved": int(s.str.startswith("agent_p").sum()), "not_visible": int((s == "agent_n").sum()),
                          "not_shuttle": int((s == "agent_x").sum()),
                          "other_court": int(s.str.fullmatch(r"agent_o(_snap)?").sum()), "unknown": int((s == "agent_u").sum())}
    out["by_reason"] = by
    return out


# =============================================================================
# 動画から
# =============================================================================
def _camera(spec: str):
    """``clips:<名前>``（``real_hits_clips.json``）か ``videos:<名前>``（``real_videos_baseline.json``）か、
    キャリブレーションの JSON のパス。"""
    from smash_speed.camera import Camera

    kind, _, name = spec.partition(":")
    if kind == "clips":
        d = json.loads((ROOT / "agent" / "checks" / "real_hits_clips.json").read_text(encoding="utf-8"))
        return Camera.from_dict(next(c for c in d["clips"] if c["name"] == name)["camera"])
    if kind == "videos":
        d = json.loads((ROOT / "agent" / "checks" / "real_videos_baseline.json").read_text(encoding="utf-8"))
        return Camera.from_dict(next(e for e in d["videos"] if e["name"] == name)["calibration"]["camera"])
    from smash_speed.calibration.io import load_calibration

    return load_calibration(Path(spec))[0]


def _video_path(video: str) -> Path:
    p = Path(video)
    if p.exists():
        return p
    cfg = ROOT / ".agent" / "config.json"
    if cfg.exists():
        d = json.loads(cfg.read_text(encoding="utf-8-sig")).get("videos_dir")
        if d and (Path(d) / video).exists():
            return Path(d) / video
    raise FileNotFoundError(f"{video} が見つかりません")


class FrameGetter:
    """番号でコマを読む（近くのコマを使い回す）。"""

    def __init__(self, meta):
        from smash_speed.video_io import VideoReader

        self.r = VideoReader(meta)
        self.cache: dict[int, np.ndarray] = {}

    def __call__(self, i: int) -> np.ndarray:
        if i < 0 or i >= len(self.r.meta.timestamps):
            raise IndexError(i)
        if i not in self.cache:
            if len(self.cache) > 12:
                self.cache.pop(next(iter(self.cache)))
            self.cache[i] = self.r.read_frame(i)
        return self.cache[i]


def pick_video(video: str, camera_spec: str, out_dir: Path, start_s: float | None, end_s: float | None,
               n: int, seed: int, half: int) -> pd.DataFrame:
    """検出（なければ作って ``detections.csv``・``tracks.csv`` に残す）→ コマを選ぶ → 見本帳。"""
    from smash_speed.config import load_config
    from smash_speed.pipeline import _camera_for_video
    from smash_speed.shot_list import detect_all, track_table, select_tracks
    from smash_speed.video_io import probe

    cfg = load_config(None)
    meta = probe(_video_path(video), None, cfg)
    camera = _camera_for_video(_camera(camera_spec), meta)
    out_dir.mkdir(parents=True, exist_ok=True)
    det_csv = out_dir / f"detections_{start_s}_{end_s}.csv"
    if det_csv.exists():
        det_all = pd.read_csv(det_csv)
    else:
        det_all = detect_all(meta, cfg, start_s, end_s, exclusive=False, progress=lambda s: None)
        det_all.to_csv(det_csv, index=False)
    tab = track_table(det_all, camera, cfg)
    tab["selected"] = select_tracks(tab)
    tab.to_csv(out_dir / "tracks.csv", index=False)
    old = pd.read_csv(out_dir / CANDIDATES_FILE) if (out_dir / CANDIDATES_FILE).exists() else None
    done = set(old.frame) if old is not None else set()
    cands = pick_frames(det_all, tab, meta.width, meta.height, n=n, seed=seed, exclude=done)
    start_id = int(old.id.max()) + 1 if old is not None and len(old) else 0
    cands.insert(0, "id", np.arange(start_id, start_id + len(cands)))
    cands["video"] = Path(meta.path).name
    get = FrameGetter(meta)
    write_sheets(get, cands, det_all, out_dir / "sheets" / f"from_{start_id:04d}", half=half)
    all_c = pd.concat([old, cands], ignore_index=True) if old is not None else cands
    all_c.to_csv(out_dir / CANDIDATES_FILE, index=False)
    return cands


def apply_answers(out_dir: Path, snap_pos: bool = True) -> dict[str, Any]:
    """``answers.csv`` を正解ラベルにして ``labels.csv``・``_ball.csv``・``label_summary.json`` を書く。"""
    from smash_speed.config import load_config
    from smash_speed.video_io import probe

    cands = pd.read_csv(out_dir / CANDIDATES_FILE)
    answers = pd.read_csv(out_dir / ANSWERS_FILE, dtype={"answer": str})
    video = str(cands.video.iloc[0])
    meta = probe(_video_path(video), None, load_config(None))
    get = FrameGetter(meta) if snap_pos else None
    labels = answers_to_labels(cands, answers, get)
    save_labels(labels, out_dir, Path(video).stem)
    if get is not None:
        for p in (out_dir / "check").glob("check_*.jpg"):
            p.unlink()
        write_check_sheets(get, labels, out_dir / "check")
    s = summary(labels)
    s["implausible_jumps"] = implausible_jumps(labels, meta.width).to_dict("records")
    (out_dir / "label_summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
    return s


# =============================================================================
# 付け直し・見直し
# =============================================================================
RECHECK_DIR = "recheck"


def pick_recheck(cands: pd.DataFrame, answers: pd.DataFrame, frac: float = 0.1, seed: int = 0) -> pd.DataFrame:
    """答えたコマから ``frac`` を選ぶ（日を変えて付け直し、前の答えと比べる）。前の答えは見本帳に出さない。"""
    done = cands[cands.id.isin(set(answers.id))]
    n = max(1, int(round(frac * len(done))))
    rng = np.random.default_rng(seed)
    return done.iloc[np.sort(rng.choice(len(done), size=min(n, len(done)), replace=False))].reset_index(drop=True)


def compare_answers(first: pd.DataFrame, second: pd.DataFrame) -> dict[str, Any]:
    """同じコマの 2 回の正解ラベル（``answers_to_labels`` の形）を比べる。

    ``source`` の種類（ok・p → 見える、n、x、o、u）が合う割合と、2 回とも見えるコマの位置のずれ [px]。
    ``ok`` と ``p`` はどちらも「見える」として合っているとする（位置のずれで比べる）。
    """
    def kind(s: str) -> str:
        k = s.removeprefix("agent_").removesuffix("_snap")
        return "vis" if k in ("ok", "p") else k

    m = first.merge(second, on="frame", suffixes=("_1", "_2"))
    k1, k2 = m.source_1.map(kind), m.source_2.map(kind)
    both = m[(m.visible_1 == 1) & (m.visible_2 == 1)]
    d = np.hypot(both.x_px_1 - both.x_px_2, both.y_px_1 - both.y_px_2).to_numpy(float)
    pairs = pd.crosstab(k1, k2)
    return {"n": len(m), "agree": int((k1 == k2).sum()), "agree_rate": float((k1 == k2).mean()) if len(m) else float("nan"),
            # 「見える」かどうか（使えるラベルとして）だけで比べたとき
            "visible_agree_rate": float(((k1 == "vis") == (k2 == "vis")).mean()) if len(m) else float("nan"),
            "n_both_visible": len(both), "dist_median_px": float(np.median(d)) if len(d) else float("nan"),
            "dist_max_px": float(d.max()) if len(d) else float("nan"), "n_over_5px": int((d > 5).sum()),
            "pairs": {f"{a}->{b}": int(pairs.loc[a, b]) for a in pairs.index for b in pairs.columns if pairs.loc[a, b]},
            "disagree_frames": [int(f) for f in m.frame[k1 != k2]] + [int(f) for f in both.frame[d > 5]]}


def implausible_jumps(labels: pd.DataFrame, width: int, max_gap: int = 3, max_step_frac: float = 0.06) -> pd.DataFrame:
    """近いコマ（``max_gap`` コマ以内）の見えるラベルどうしで、1 コマあたりの動きが画面の幅の
    ``max_step_frac`` を超える組（シャトルとしてありえない跳び方。どちらかが別の物の見込み）。"""
    v = labels[labels.visible == 1].sort_values("frame")
    f, x, y = v.frame.to_numpy(int), v.x_px.to_numpy(float), v.y_px.to_numpy(float)
    rows = []
    for k in range(len(v) - 1):
        df = f[k + 1] - f[k]
        if 1 <= df <= max_gap:
            step = float(np.hypot(x[k + 1] - x[k], y[k + 1] - y[k]) / df)
            if step > max_step_frac * width:
                rows.append({"frame_a": int(f[k]), "frame_b": int(f[k + 1]), "px_per_frame": round(step, 1)})
    return pd.DataFrame(rows, columns=["frame_a", "frame_b", "px_per_frame"])


def recheck_pick(out_dir: Path, frac: float, seed: int, half: int) -> pd.DataFrame:
    """付け直すコマを選んで ``recheck/candidates.csv`` と見本帳を書く（前の答えは ``answers.csv`` にだけある）。"""
    from smash_speed.config import load_config
    from smash_speed.video_io import probe

    cands = pd.read_csv(out_dir / CANDIDATES_FILE)
    answers = pd.read_csv(out_dir / ANSWERS_FILE, dtype={"answer": str})
    items = pick_recheck(cands, answers, frac, seed)
    rd = out_dir / RECHECK_DIR
    rd.mkdir(parents=True, exist_ok=True)
    items.to_csv(rd / CANDIDATES_FILE, index=False)
    meta = probe(_video_path(str(cands.video.iloc[0])), None, load_config(None))
    dets = sorted(out_dir.glob("detections_*.csv"))
    det_all = pd.read_csv(dets[0]) if dets else pd.DataFrame(columns=["frame", "x_px", "y_px", "visible"])
    write_sheets(FrameGetter(meta), items, det_all, rd / "sheets", half=half)
    return items


def recheck_compare(out_dir: Path) -> dict[str, Any]:
    """``recheck/answers.csv`` を前の答えと比べ、``recheck/compare.json`` に書く。"""
    from smash_speed.config import load_config
    from smash_speed.video_io import probe

    rd = out_dir / RECHECK_DIR
    items = pd.read_csv(rd / CANDIDATES_FILE)
    get = FrameGetter(probe(_video_path(str(items.video.iloc[0])), None, load_config(None)))
    first = answers_to_labels(items, pd.read_csv(out_dir / ANSWERS_FILE, dtype={"answer": str}), get)
    second = answers_to_labels(items, pd.read_csv(rd / ANSWERS_FILE, dtype={"answer": str}), get)
    out = compare_answers(first, second)
    (rd / "compare.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pick", help="怪しいコマを選んで見本帳を書く")
    p.add_argument("video", help="動画（パスか videos_dir の中の名前）")
    p.add_argument("--camera", required=True, help="clips:<名前> / videos:<名前> / キャリブレーションの JSON")
    p.add_argument("--out", required=True)
    p.add_argument("--start", type=float); p.add_argument("--end", type=float)
    p.add_argument("--n", type=int, default=150); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--half", type=int, default=120, help="切り出しの半分の幅 [px]")
    a_ = sub.add_parser("apply", help="answers.csv を正解ラベルにする")
    a_.add_argument("out"); a_.add_argument("--no-snap", action="store_true")
    r_ = sub.add_parser("recheck", help="答えたコマの一部を付け直す（pick：見本帳を書く、compare：前の答えと比べる）")
    r_.add_argument("step", choices=["pick", "compare"]); r_.add_argument("out")
    r_.add_argument("--frac", type=float, default=0.1); r_.add_argument("--seed", type=int, default=0)
    r_.add_argument("--half", type=int, default=120)
    a = ap.parse_args(argv)
    if (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    if a.cmd == "pick":
        c = pick_video(a.video, a.camera, Path(a.out), a.start, a.end, a.n, a.seed, a.half)
        print(f"{len(c)} コマ（{c.reason.str.split('+').explode().value_counts().to_dict()}）→ {a.out}")
    elif a.cmd == "recheck":
        if a.step == "pick":
            c = recheck_pick(Path(a.out), a.frac, a.seed, a.half)
            print(f"{len(c)} コマ → {Path(a.out) / RECHECK_DIR}")
        else:
            print(json.dumps(recheck_compare(Path(a.out)), ensure_ascii=False, indent=1))
    else:
        print(json.dumps(apply_answers(Path(a.out), not a.no_snap), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
