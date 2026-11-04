"""正しい初速を作るときに、コマを見て確かめるための道具（画像は Git に入れない場所に書く）。

``true_speeds.py`` に入れるシャトルの位置・打った時刻・床の点を、コマを並べた画像で決める。

使い方（``--out`` は Git の外か ``outputs/`` の下にする。人が写るので画像は公開しない）::

    # コマを並べる（決まった範囲。50 px ごとの線と、コマ番号・時刻が入る）
    python agent/truth/frames.py sheet IMG_5096 1926 1940 --center 910 270 --half 80 --out outputs/truth/s.jpg
    # シャトルを追う（差分で動くものを追うだけなので、結果は必ず track-sheet で目で確かめる）
    python agent/truth/frames.py track IMG_5096 1933 921 268 --vel 22.5 8.5 --n 170 --out outputs/truth/t.json
    # 追った点を中心にしたコマを並べる（印がシャトル・ぶれの帯の真ん中に乗っているかを見る）
    python agent/truth/frames.py track-sheet IMG_5096 outputs/truth/t.json --step 5 --out outputs/truth/ts.jpg
    # 画像の点 → コートの床の座標（足元・着地点）
    python agent/truth/frames.py floor IMG_5096 808 650 1395 557

動画は ``.agent/config.json`` の ``videos_dir`` か ``--videos`` から探す。キャリブレーションは
``agent/checks/real_videos_baseline.json`` の中身を使う（品質チェックと同じ）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from smash_speed.camera import Camera  # noqa: E402
from smash_speed.geometry import floor_point  # noqa: E402
from smash_speed.video_io import VideoReader, probe  # noqa: E402

BASELINE = ROOT / "agent" / "checks" / "real_videos_baseline.json"


def _entry(name: str) -> dict:
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    return next(e for e in base["videos"] if e["name"] == name)


def camera(name: str) -> Camera:
    return Camera.from_dict(_entry(name)["calibration"]["camera"])


def reader(name: str, videos: str | None) -> VideoReader:
    cfg = ROOT / ".agent" / "config.json"
    d = videos or (json.loads(cfg.read_text(encoding="utf-8")).get("videos_dir") if cfg.exists() else None)
    e = _entry(name)
    for p in ([Path(d) / e["video"]] if d else []) + [Path(e["video_path"])]:
        if p.exists():
            return VideoReader(probe(p))
    raise FileNotFoundError(f"{e['video']} が見つかりません（--videos で場所を指定してください）")


def _grid(img: np.ndarray, x0: int, y0: int, zoom: int) -> None:
    h, w = img.shape[:2]
    for k in range(0, 100000, 50):
        x, y = (k - x0) * zoom, (k - y0) * zoom
        col = (0, 255, 255) if k % 100 == 0 else (0, 128, 128)
        if 0 <= x < w:
            cv2.line(img, (x, 0), (x, h), col, 1)
        if 0 <= y < h:
            cv2.line(img, (0, y), (w, y), col, 1)
        if x >= w and y >= h:
            break


def _save(tiles: list[np.ndarray], ncol: int, out: str) -> None:
    tiles = [cv2.copyMakeBorder(t, 0, 1, 0, 1, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % ncol)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out, np.vstack([np.hstack(tiles[k:k + ncol]) for k in range(0, len(tiles), ncol)]))


def sheet(r: VideoReader, i0: int, i1: int, step: int, cx: int, cy: int, half: int, out: str, ncol: int) -> None:
    zoom = max(1, 160 // half)
    tiles = []
    for i in range(i0, i1 + 1, step):
        f = cv2.copyMakeBorder(r.read_frame(i), half, half, half, half, cv2.BORDER_CONSTANT)
        c = cv2.resize(f[cy:cy + 2 * half, cx:cx + 2 * half], None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
        _grid(c, cx - half, cy - half, zoom)
        cv2.putText(c, f"{i} {r.meta.timestamps[i]:.3f}s", (3, 15), 0, 0.45, (0, 0, 255), 1)
        tiles.append(c)
    _save(tiles, ncol, out)


def track(r: VideoReader, i0: int, u: float, v: float, du: float, dv: float, n: int, win: int) -> dict[str, list | None]:
    """前後のコマとの差（動くもの）を、予測した位置の近くで追う。見失ったコマは None。"""
    gap = 6 if r.meta.capture_fps > 100 else 2
    cache: dict[int, np.ndarray] = {}

    def gray(i: int) -> np.ndarray:
        if i not in cache:
            g = cv2.cvtColor(r.read_frame(i), cv2.COLOR_BGR2GRAY).astype(np.float32)
            cache[i] = cv2.GaussianBlur(g, (3, 3), 0)
        return cache[i]

    res: dict[str, list | None] = {str(i0): [u, v]}
    for i in range(i0 + 1, i0 + n):
        pu, pv = u + du, v + dv
        try:
            d = np.minimum(np.abs(gray(i) - gray(i - gap)), np.abs(gray(i) - gray(i + gap)))
        except IndexError:
            break
        x0, y0 = int(max(0, pu - win)), int(max(0, pv - win))
        patch = d[y0:int(pv + win), x0:int(pu + win)]
        if patch.size == 0 or patch.max() < 12:
            res[str(i)] = None
            u, v = pu, pv
            continue
        yy, xx = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
        w = patch * np.exp(-((xx + x0 - pu) ** 2 + (yy + y0 - pv) ** 2) / (2 * (win / 2) ** 2))
        my, mx = np.unravel_index(np.argmax(w), w.shape)
        lab = cv2.connectedComponents((patch > 0.5 * patch[my, mx]).astype(np.uint8))[1]
        sel = lab == lab[my, mx]
        wy, wx = np.nonzero(sel)
        ww = patch[sel]
        nu, nv = float((wx * ww).sum() / ww.sum() + x0), float((wy * ww).sum() / ww.sum() + y0)
        du, dv = 0.6 * du + 0.4 * (nu - u), 0.6 * dv + 0.4 * (nv - v)
        u, v = nu, nv
        res[str(i)] = [round(u, 1), round(v, 1)]
    return res


def track_sheet(r: VideoReader, marks: dict[str, list | None], step: int, half: int, out: str, ncol: int) -> None:
    keys = sorted(int(k) for k, m in marks.items() if m)[::step]
    zoom = max(1, 120 // half)
    tiles = []
    for i in keys:
        m = marks[str(i)]
        cx, cy = int(m[0]), int(m[1])
        f = cv2.copyMakeBorder(r.read_frame(i), half, half, half, half, cv2.BORDER_CONSTANT)
        c = cv2.resize(f[cy:cy + 2 * half, cx:cx + 2 * half], None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
        o = int((half + m[0] - cx) * zoom), int((half + m[1] - cy) * zoom)
        for a, b in (((-12, 0), (-5, 0)), ((5, 0), (12, 0)), ((0, -12), (0, -5)), ((0, 5), (0, 12))):
            cv2.line(c, (o[0] + a[0], o[1] + a[1]), (o[0] + b[0], o[1] + b[1]), (0, 0, 255), 1)
        cv2.putText(c, str(i), (2, 12), 0, 0.4, (0, 0, 255), 1)
        tiles.append(c)
    _save(tiles, ncol, out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--videos", help="動画のフォルダ（既定は .agent/config.json の videos_dir）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheet", help="決まった範囲のコマを並べる")
    s.add_argument("video"); s.add_argument("i0", type=int); s.add_argument("i1", type=int)
    s.add_argument("--step", type=int, default=1); s.add_argument("--center", type=int, nargs=2, required=True)
    s.add_argument("--half", type=int, default=80); s.add_argument("--cols", type=int, default=4)
    s.add_argument("--out", required=True)
    t = sub.add_parser("track", help="シャトルを追う（目で確かめる前の下書き）")
    t.add_argument("video"); t.add_argument("i0", type=int); t.add_argument("u", type=float); t.add_argument("v", type=float)
    t.add_argument("--vel", type=float, nargs=2, required=True, help="1 コマあたりの動き [px]")
    t.add_argument("--n", type=int, default=100); t.add_argument("--win", type=int, default=25)
    t.add_argument("--out", required=True)
    ts = sub.add_parser("track-sheet", help="追った点を中心にコマを並べる")
    ts.add_argument("video"); ts.add_argument("marks")
    ts.add_argument("--step", type=int, default=1); ts.add_argument("--half", type=int, default=20)
    ts.add_argument("--cols", type=int, default=8); ts.add_argument("--out", required=True)
    fl = sub.add_parser("floor", help="画像の点 → 床の座標 [m]")
    fl.add_argument("video"); fl.add_argument("px", type=float, nargs="+")
    a = ap.parse_args(argv)
    if (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    if a.cmd == "floor":
        X = floor_point(camera(a.video), np.asarray(a.px, float).reshape(-1, 2))
        for p, x in zip(np.asarray(a.px).reshape(-1, 2), X):
            print(f"({p[0]:.0f}, {p[1]:.0f}) px → x = {x[0]:.2f} m, y = {x[1]:.2f} m")
        return 0
    with reader(a.video, a.videos) as r:
        if a.cmd == "sheet":
            sheet(r, a.i0, a.i1, a.step, a.center[0], a.center[1], a.half, a.out, a.cols)
        elif a.cmd == "track":
            res = track(r, a.i0, a.u, a.v, a.vel[0], a.vel[1], a.n, a.win)
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(res), encoding="utf-8")
            print(f"{sum(m is not None for m in res.values())} コマ → {a.out}")
        else:
            track_sheet(r, json.loads(Path(a.marks).read_text(encoding="utf-8")), a.step, a.half, a.out, a.cols)
    return 0


if __name__ == "__main__":
    sys.exit(main())
