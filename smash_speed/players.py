"""選手の検出と追跡（ROADMAP D1）。

1. 人の検出：YOLOX（Megvii、Apache-2.0。COCO の「人」だけを使う）の ONNX を onnxruntime で動かす
   （``accel.make_session``。GPU があれば GPU）。モデルは最初に使うときに ``ckpts/`` へ落とす（yolox_s で約 36 MB）。
2. 足元の位置：人の枠の下の辺の中央を床（z = 0）へ投影し、コートの座標 (x, y) [m] にする。
3. 追跡：コートの手前（h：x < 0）と奥（o：x > 0）に分け、側ごとに近いコマどうしの検出をつないで短い軌跡にする。
   軌跡のうち、時間が重ならない組で「コートの中にいた点の数」の合計が最も多いものを選ぶ
   （コートの外で待っている人・隣のコートの人・審判などを外す）。短い抜けは前後から補う。
4. 跳んでいるコマ：足元の位置で測った枠の高さがふだんより大きく出るコマを見つけ、床の上の位置を前後から補う。

5. 動き：``movement.write_movement`` で、移動距離・速さ・いた場所の分布（``movement.json``・``movement.svg``）も書く。
6. ダブルス（1 側 2 人）：``per_side`` が 2（または 0 = 自動で、その側に 2 人いると分かったとき）の側は、
   ``player_ids.track_pair`` で 2 人を同じ番号のまま追う（すれ違い・重なりの後も入れ替わらないよう、動きと枠の中の色で確かめる）。
   色は検出のときに枠の中から取り（``box_appearance``）、``people.csv`` の ``app`` に入れる。

結果は 1 コマ 1 側 1 人の ``players.csv``（``frame, t, side, player, x, y, u, v, x0, y0, x1, y1, score, interpolated,
airborne``）と、
検出したすべての人の ``people.csv``、確かめ用の ``players_preview.jpg``・``players.json``。

``python -m smash_speed.players <動画> --calib <calibration.json> --out <出力先> [--start 80 --end 100]`` で動かす。
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from smash_speed.camera import Camera
from smash_speed.geometry import floor_point

PLAYERS_FILE = "players.csv"
PLAYERS_PREVIEW_FILE = "players_preview.jpg"
PEOPLE_FILE = "people.csv"     # 検出したすべての人（追跡の条件を変えて選び直すのに使う）
SIDES = ("h", "o")

# 使えるモデル（YOLOX の公式の ONNX。https://github.com/Megvii-BaseDetection/YOLOX 、Apache-2.0）
_YOLOX_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/{name}.onnx"
MODELS: dict[str, str] = {   # 名前 → SHA-256
    "yolox_s": "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
    "yolox_tiny": "427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7",
}
DEFAULT_MODEL = "yolox_s"
MODELS_DIR = Path(__file__).resolve().parents[1] / "ckpts"
PERSON_CLASS = 0   # COCO の「人」
PAD_VALUE = 114    # YOLOX の学習と同じ余白の色


@dataclass
class PlayerOptions:
    """選手の検出と追跡の条件。"""

    model: str = DEFAULT_MODEL
    min_score: float = 0.3            # 人の検出のスコアの下限
    nms_iou: float = 0.45             # 重なった枠をまとめる IoU
    process_width: int = 1280         # 検出に使う画像の横幅 [px]（モデルの入力は 640 なので、これ以上は要らない）
    stride: int = 1                   # 何コマごとに検出するか
    per_side: int = 1                 # 1 側の人数（シングルス 1、ダブルス 2、0 は側ごとに自動で決める。``player_ids``）
    margin_m: tuple[float, float] = (2.0, 1.0)  # 選手とみなす範囲の、コートの外側の線からの広げ幅（長さ方向・幅方向）[m]
    net_margin_m: float = 0.5         # ネットを越えて相手側に足が出たとみなしてよい幅 [m]
    outside_weight: float = 0.3       # コートの線の外（広げた範囲）にいた点の重み（中にいた点は 1）
    # つなぐ範囲は画像の上で決める（奥の選手は足元が数 px ずれるだけでコートの位置が 1〜2 m 動くので、コートの座標では切れやすい）。
    # 単位は枠の高さ（≒身長）：選手の速さ 8 m/s は 1 秒に身長の約 4.5 倍
    max_speed_h: float = 5.0          # 足元の動きの速さの上限 [枠の高さ/s]
    gate_h: float = 0.3               # つなぐ範囲の最小（枠の揺れの分）[枠の高さ]
    max_gap_s: float = 0.5            # 軌跡の途切れをつなぎ、補ってよい長さ [s]
    min_track_s: float = 0.2          # 選ぶ軌跡の長さの下限 [s]（一瞬の誤検出を落とす）
    # 1 人に 2 つの枠（体と、ラケットの腕など）が出ると、軌跡が 2 本に分かれて前後の軌跡と少し重なる。
    # 重なりを許さないと、間の軌跡がまるごと落ちる（IMG_5109_1 の 14.2〜16.6 s、ネットの近くの奥の選手が 1.8 秒抜けた）
    max_overlap_s: float = 0.4        # 選ぶ軌跡どうしの時間の重なりの上限 [s]（重なったコマは重みの大きい方）
    # 跳んでいるコマ（``ground_jumps``）：足が浮くと枠の下の辺が上がり、床へ投影した位置がカメラから 1〜5 m 遠くへずれる。
    # そのとき、足元の位置で測った枠の高さ [m] が、その選手のふだん（中央値）より大きく出る。
    # ふだん 1.5〜1.7 m、腕を上げると 2.2 m ほど、跳んでいる間は 2.7〜3.5 m（IMG_3391・IMG_4397・IMG_5096 のジャンプスマッシュ）
    jump_ratio: float = 1.5           # ふだんの高さのこの倍を超えたコマを、跳んでいる芯とみなす
    jump_edge_ratio: float = 1.2      # 芯の前後で、この倍を超えている間（踏み切り・着地）も跳んでいるとみなす
    jump_tail_ratio: float = 1.05     # その外も、この倍より大きく、外へ向かって下がり続ける間は含める
    max_jump_s: float = 1.2           # これより長い区間は跳んでいるとはみなさない [s]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# =============================================================================
# モデル
# =============================================================================
def model_path(name: str = DEFAULT_MODEL, models_dir: str | Path | None = None) -> Path:
    return Path(models_dir or MODELS_DIR) / f"{name}.onnx"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(name: str = DEFAULT_MODEL, models_dir: str | Path | None = None,
                 progress: Callable[[str], None] = print) -> Path:
    """モデルの ONNX のパスを返す。なければ落として、SHA-256 を確かめる。"""
    if name not in MODELS:
        raise ValueError(f"人の検出のモデルは {', '.join(MODELS)} のどれかにしてください: {name!r}")
    path = model_path(name, models_dir)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    progress(f"[準備] 人の検出のモデル（{name}）を取得中")
    urllib.request.urlretrieve(_YOLOX_URL.format(name=name), tmp)
    digest = _sha256(tmp)
    if digest != MODELS[name]:
        tmp.unlink(missing_ok=True)
        raise OSError(f"人の検出のモデル {name} の中身が想定と違います（SHA-256 {digest}）")
    tmp.replace(path)
    return path


# =============================================================================
# 人の検出
# =============================================================================
def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float]:
    """縦横比を保って ``size`` 四方に縮め、右と下を埋める（YOLOX の前処理）。(1, 3, size, size) と縮めた率を返す。"""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(h * r), int(w * r)
    out = np.full((size, size, 3), PAD_VALUE, np.uint8)
    out[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out.transpose(2, 0, 1)[None].astype(np.float32), r


def decode_yolox(raw: np.ndarray, size: int, strides: tuple[int, ...] = (8, 16, 32)) -> np.ndarray:
    """YOLOX の出力 (N, 5 + 種類の数) の格子の上の値を、入力画像の上の (cx, cy, w, h, 物らしさ, ...) にする。"""
    grids, steps = [], []
    for s in strides:
        n = size // s
        gy, gx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        grids.append(np.stack([gx, gy], -1).reshape(-1, 2))
        steps.append(np.full((n * n, 1), s))
    g = np.concatenate(grids).astype(np.float32)
    st = np.concatenate(steps).astype(np.float32)
    out = np.array(raw, np.float32, copy=True)
    out[:, :2] = (out[:, :2] + g) * st
    out[:, 2:4] = np.exp(out[:, 2:4]) * st
    return out


class PersonDetector:
    """人の枠 (N, 5)：x0, y0, x1, y1 [入力画像の px]、スコア を返す。"""

    def __init__(self, model: str | Path | bytes, min_score: float = 0.3, nms_iou: float = 0.45,
                 prefer: str | None = None):
        from smash_speed.accel import make_session

        self.session = make_session(model, prefer)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.size = int(inp.shape[-1])
        self.min_score = min_score
        self.nms_iou = nms_iou

    @property
    def provider(self) -> str:
        return self.session.get_providers()[0]

    def detect(self, img: np.ndarray) -> np.ndarray:
        x, r = letterbox(img, self.size)
        raw = self.session.run(None, {self.input_name: x})[0][0]
        d = decode_yolox(raw, self.size)
        score = d[:, 4] * d[:, 5 + PERSON_CLASS]
        keep = score >= self.min_score
        d, score = d[keep], score[keep]
        if not len(d):
            return np.zeros((0, 5))
        xywh = np.column_stack([d[:, 0] - d[:, 2] / 2, d[:, 1] - d[:, 3] / 2, d[:, 2], d[:, 3]]) / r
        idx = np.array(cv2.dnn.NMSBoxes(xywh.tolist(), score.tolist(), self.min_score, self.nms_iou), int).ravel()
        b = xywh[idx]
        return np.column_stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3], score[idx]])


# 枠の中の色（ダブルスで 2 人を見分ける手がかり。``player_ids``）：上半身と下半身の 2 か所で、
# 色相 ``APP_HUE_BINS`` 個（鮮やかな画素だけ）＋ 色のない画素の暗い・中くらい・明るい の割合。合計 1 を 0〜255 にして持つ
APP_HUE_BINS = 8
APP_REGIONS = ((0.18, 0.48), (0.52, 0.80))   # 枠の高さに対する縦の範囲（上半身、下半身）。横は中央の 4 割
APP_DIM = len(APP_REGIONS) * (APP_HUE_BINS + 3)


def box_appearance(img: np.ndarray, box: Iterable[float]) -> np.ndarray:
    """BGR の画像の枠（x0, y0, x1, y1 [px]）の中の色の割合（長さ ``APP_DIM``、uint8）。枠が小さすぎれば 0。"""
    x0, y0, x1, y1 = (float(v) for v in box)
    w, h = x1 - x0, y1 - y0
    out = np.zeros(APP_DIM, np.uint8)
    if w < 4 or h < 8:
        return out
    n = APP_HUE_BINS + 3
    for k, (a, b) in enumerate(APP_REGIONS):
        xa, xb = int(round(x0 + 0.3 * w)), int(round(x0 + 0.7 * w))
        ya, yb = int(round(y0 + a * h)), int(round(y0 + b * h))
        crop = img[max(ya, 0):max(yb, 0), max(xa, 0):max(xb, 0)]
        if crop.size < 12:
            continue
        hsv = cv2.cvtColor(np.ascontiguousarray(crop), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        colored = (hsv[:, 1] >= 70) & (hsv[:, 2] >= 60)
        hist = np.zeros(n)
        hist[:APP_HUE_BINS] = np.bincount((hsv[colored, 0].astype(int) * APP_HUE_BINS) // 180,
                                          minlength=APP_HUE_BINS)[:APP_HUE_BINS]
        v = hsv[~colored, 2]
        hist[APP_HUE_BINS:] = [(v < 70).sum(), ((v >= 70) & (v < 170)).sum(), (v >= 170).sum()]
        out[k * n:(k + 1) * n] = np.round(255 * hist / max(hist.sum(), 1.0))
    return out


def encode_appearance(feat: np.ndarray) -> str:
    return np.asarray(feat, np.uint8).tobytes().hex()


def decode_appearance(col: Iterable[Any]) -> np.ndarray:
    """``people.csv`` の ``app``（16 進の文字列）の列を (n, ``APP_DIM``) の 0〜1 に戻す。空・形の違う行は NaN。"""
    col = list(col)
    out = np.full((len(col), APP_DIM), np.nan)
    for i, s in enumerate(col):
        if isinstance(s, str) and len(s) == 2 * APP_DIM:
            v = np.frombuffer(bytes.fromhex(s), np.uint8)
            if v.any():
                out[i] = v / 255.0
    return out


def detect_people(frames: Iterable[tuple[int, float, np.ndarray]], detector: PersonDetector, scale: float = 1.0,
                  appearance: bool = False) -> pd.DataFrame:
    """コマごとに人を検出する。``scale`` は元の解像度に対する画像の縮小率（枠は元の解像度に直す）。

    ``appearance`` が真なら、枠の中の色（``box_appearance``）を ``app`` の列に 16 進の文字列で入れる。
    """
    rows = []
    for i, t, img in frames:
        for x0, y0, x1, y1, s in detector.detect(img):
            row = (i, t, x0 / scale, y0 / scale, x1 / scale, y1 / scale, s)
            rows.append(row + (encode_appearance(box_appearance(img, (x0, y0, x1, y1))),) if appearance else row)
    return pd.DataFrame(rows, columns=["frame", "t", "x0", "y0", "x1", "y1", "score"] + (["app"] if appearance else []))


def add_court_positions(people: pd.DataFrame, camera: Camera) -> pd.DataFrame:
    """枠の下の辺の中央（足元）を床に投影し、``u, v``（画像）と ``x, y``（コート）[m] を足す。"""
    out = people.copy()
    out["u"] = (out["x0"] + out["x1"]) / 2
    out["v"] = out["y1"]
    if len(out):
        X = floor_point(camera, out[["u", "v"]].to_numpy(float))
        out["x"], out["y"] = X[:, 0], X[:, 1]
    else:
        out["x"], out["y"] = np.zeros(0), np.zeros(0)
    return out


# =============================================================================
# 追跡
# =============================================================================
def side_region(people: pd.DataFrame, side: str, cfg: dict[str, Any], o: PlayerOptions
                ) -> tuple[np.ndarray, np.ndarray]:
    """``side`` の側で選手とみなす範囲にいる人（真偽の配列）と、その点の重み（線の中 1、外 ``outside_weight``）。"""
    L = float(cfg["court"]["half_length_m"])
    W = float(cfg["court"]["half_width_doubles_m"])
    x = people["x"].to_numpy(float)
    y = people["y"].to_numpy(float)
    sx = x if side == "o" else -x   # 自分の側を正にした長さ方向の位置
    with np.errstate(invalid="ignore"):
        region = (sx >= -o.net_margin_m) & (sx <= L + o.margin_m[0]) & (np.abs(y) <= W + o.margin_m[1])
        inside = (sx >= -o.net_margin_m) & (sx <= L) & (np.abs(y) <= W)
    region &= np.isfinite(x) & np.isfinite(y)
    weight = np.where(inside, 1.0, o.outside_weight) * people["score"].to_numpy(float)
    return region, weight


def link_tracklets(cand: pd.DataFrame, o: PlayerOptions) -> np.ndarray:
    """近いコマどうしの検出をつないで短い軌跡にし、軌跡の番号（``cand`` の行の順）を返す。

    コマごとに、生きている軌跡の最後の枠と検出の、足元の画像の上の距離（枠の高さで割る）で割り当てる
    （``linear_sum_assignment``）。つなぐ範囲は ``gate_h + max_speed_h × 経過時間``。
    ``max_gap_s`` より長く見えない軌跡は終わりにする。
    """
    ids = np.full(len(cand), -1, int)
    active: list[list[float]] = []   # [軌跡の番号, 最後の時刻, u, v, 枠の高さ]
    next_id = 0
    frames = cand["frame"].to_numpy()
    ts = cand["t"].to_numpy(float)
    height = (cand["y1"] - cand["y0"]).to_numpy(float)
    xy = np.column_stack([cand[["u", "v"]].to_numpy(float), height])
    order = np.argsort(frames, kind="stable")
    bounds = np.flatnonzero(np.diff(frames[order])) + 1
    for rows in np.split(order, bounds):
        if not len(rows):
            continue
        t = ts[rows[0]]
        active = [a for a in active if t - a[1] <= o.max_gap_s]
        used = set()
        if active:
            last = np.array([a[2:] for a in active])
            scale = (last[:, None, 2] + xy[rows][None, :, 2]) / 2
            cost = np.linalg.norm(last[:, None, :2] - xy[rows][None, :, :2], axis=2) / scale
            gate = o.gate_h + o.max_speed_h * (t - np.array([a[1] for a in active]))
            big = 1e6
            c = np.where(cost <= gate[:, None], cost, big)
            for ai, ri in zip(*linear_sum_assignment(c)):
                if c[ai, ri] >= big:
                    continue
                r = rows[ri]
                ids[r] = int(active[ai][0])
                active[ai][1:] = [t, *xy[r]]
                used.add(ri)
        for ri, r in enumerate(rows):
            if ri not in used:
                ids[r] = next_id
                active.append([next_id, t, *xy[r]])
                next_id += 1
    return ids


def choose_tracklets(spans: pd.DataFrame, max_overlap_s: float = 0.0) -> list[int]:
    """時間が重ならない軌跡の組で、重みの合計が最も多いものを選ぶ（区間の重み付きの選び方）。

    ``spans`` は ``id, t0, t1, weight`` の表。選んだ軌跡の番号を時刻の順に返す。
    """
    if spans.empty:
        return []
    s = spans.sort_values("t1").reset_index(drop=True)
    t0 = s["t0"].to_numpy(float)
    t1 = s["t1"].to_numpy(float)
    w = s["weight"].to_numpy(float)
    # prev[k]：k より前に終わる軌跡のうち、最後のもの（k の始まりと max_overlap_s までの重なりは許す）
    prev = np.searchsorted(t1, t0 + max_overlap_s, side="right") - 1
    prev = np.minimum(prev, np.arange(len(s)) - 1)
    best = np.zeros(len(s) + 1)
    take = np.zeros(len(s), bool)
    for k in range(len(s)):
        with_k = w[k] + best[prev[k] + 1]
        take[k] = with_k > best[k]
        best[k + 1] = with_k if take[k] else best[k]
    chosen = []
    k = len(s) - 1
    while k >= 0:
        if take[k]:
            chosen.append(int(s["id"][k]))
            k = prev[k]
        else:
            k -= 1
    return chosen[::-1]


def _fill_gaps(track: pd.DataFrame, frame_t: pd.Series, o: PlayerOptions) -> pd.DataFrame:
    """選んだ点の間の ``max_gap_s`` 以下の抜けを、前後の位置から直線で補う（``interpolated`` を真にする）。"""
    track = track.sort_values("frame").reset_index(drop=True)
    track["interpolated"] = False
    fill = []
    num = ["x", "y", "u", "v", "x0", "y0", "x1", "y1"]
    for a, b in zip(track.index[:-1], track.index[1:]):
        ta, tb = track.at[a, "t"], track.at[b, "t"]
        if tb - ta > o.max_gap_s:
            continue
        gap = frame_t[(frame_t.index > track.at[a, "frame"]) & (frame_t.index < track.at[b, "frame"])]
        for f, t in gap.items():
            k = (t - ta) / (tb - ta)
            row = {c: (1 - k) * track.at[a, c] + k * track.at[b, c] for c in num}
            fill.append({**row, "frame": f, "t": t, "score": np.nan, "interpolated": True})
    if fill:
        track = pd.concat([track, pd.DataFrame(fill)], ignore_index=True).sort_values("frame")
    return track.reset_index(drop=True)


PLAYER_COLUMNS = ["frame", "t", "side", "player", "x", "y", "u", "v", "x0", "y0", "x1", "y1", "score", "interpolated"]


def track_players(people: pd.DataFrame, frame_t: pd.Series, cfg: dict[str, Any], o: PlayerOptions | None = None
                  ) -> pd.DataFrame:
    """人の検出（``add_court_positions`` の後）から、側ごとに ``per_side`` 人の選手の位置を 1 コマずつ出す。

    Args:
        people: ``frame, t, x0, y0, x1, y1, score, u, v, x, y``
        frame_t: 検出したコマの番号 → 時刻 [s]（人が 1 人もいないコマも含める。抜けを補うのに使う）
    """
    o = o or PlayerOptions()
    if o.per_side != 1:
        from smash_speed.player_ids import track_sides

        return track_sides(people, frame_t, cfg, o)[0]
    return _track_singles(people, frame_t, cfg, o, SIDES)


def _track_singles(people: pd.DataFrame, frame_t: pd.Series, cfg: dict[str, Any], o: PlayerOptions,
                   sides: Iterable[str]) -> pd.DataFrame:
    """側ごとに 1 人を、コートの中にいた点の多い軌跡の組で選ぶ（シングルスの追跡）。"""
    out = []
    for side in sides:
        region, weight = side_region(people, side, cfg, o)
        cand = people[region].assign(weight=weight[region]).reset_index(drop=True)
        if "app" in cand:
            cand = cand.drop(columns="app")
        if cand.empty:
            continue
        cand["tid"] = link_tracklets(cand, o)
        chosen = choose_tracklets(tracklet_spans(cand, o), o.max_overlap_s)
        if not chosen:
            continue
        # 同じコマに 2 つ残ったとき（軌跡の境目）は重みの大きい方
        track = cand[cand["tid"].isin(chosen)].sort_values("weight", ascending=False).drop_duplicates("frame")
        out.append(_fill_gaps(track.drop(columns=["tid", "weight"]), frame_t, o).assign(side=side, player=0))
    return _player_table(out)


def tracklet_spans(cand: pd.DataFrame, o: PlayerOptions) -> pd.DataFrame:
    """軌跡（``cand`` の ``tid``）ごとの始まり・終わりの時刻と重みの合計（``min_track_s`` より短いものは除く）。"""
    g = cand.groupby("tid")
    spans = pd.DataFrame({"t0": g["t"].min(), "t1": g["t"].max(), "weight": g["weight"].sum()})
    return spans[spans["t1"] - spans["t0"] >= o.min_track_s].rename_axis("id").reset_index()


def _player_table(out: list[pd.DataFrame]) -> pd.DataFrame:
    if not out:
        return pd.DataFrame(columns=PLAYER_COLUMNS)
    res = pd.concat(out, ignore_index=True)
    res["frame"] = res["frame"].astype(int)
    res["interpolated"] = res["interpolated"].astype(bool)
    return res[PLAYER_COLUMNS].sort_values(["frame", "side", "player"]).reset_index(drop=True)


def body_heights(players: pd.DataFrame, camera: Camera) -> np.ndarray:
    """足元のコートの位置に立っているとしたときの、枠の高さ [m]（枠の上の辺から下の辺まで）。"""
    if players.empty:
        return np.zeros(0)
    X0 = np.column_stack([players["x"].to_numpy(float), players["y"].to_numpy(float), np.zeros(len(players))])
    ppm = camera.project(X0)[:, 1] - camera.project(X0 + [0.0, 0.0, 1.0])[:, 1]   # その位置の 1 m の縦の長さ [px]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ppm > 0, (players["v"] - players["y0"]).to_numpy(float) / ppm, np.nan)


def ground_jumps(players: pd.DataFrame, camera: Camera, o: PlayerOptions | None = None) -> pd.DataFrame:
    """跳んでいるコマを見つけて ``airborne`` を真にし、その ``x, y`` を前後の床にいたコマから直線で補う。

    ``u, v`` と枠は検出のまま残す（``x, y`` だけが床の上の位置の見積もりになる）。
    """
    o = o or PlayerOptions()
    out = players.copy()
    out["airborne"] = False
    if out.empty:
        return out
    height = body_heights(out, camera)
    for _, g in out.groupby(["side", "player"]):
        g = g.sort_values("t")
        h = height[out.index.get_indexer(g.index)]
        ratio = h / np.nanmedian(h) if np.isfinite(h).any() else np.zeros(len(g))
        core = np.nan_to_num(ratio) > o.jump_ratio
        edge = np.nan_to_num(ratio) > o.jump_edge_ratio
        air = np.zeros(len(g), bool)
        # 芯を含む、edge の続く区間
        start = np.flatnonzero(edge & ~np.r_[False, edge[:-1]])
        stop = np.flatnonzero(edge & ~np.r_[edge[1:], False]) + 1
        t = g["t"].to_numpy(float)
        r = np.nan_to_num(ratio)
        for a, b in zip(start, stop):
            if not core[a:b].any():
                continue
            # 足が少しだけ浮いたコマ（低いカメラでは 10 cm で 1 m ずれる）も、高さが下がり続ける間は含める
            while a > 0 and o.jump_tail_ratio < r[a - 1] < r[a]:
                a -= 1
            while b < len(r) and o.jump_tail_ratio < r[b] < r[b - 1]:
                b += 1
            if t[b - 1] - t[a] <= o.max_jump_s:
                air[a:b] = True
        if not air.any() or (~air).sum() < 2:
            continue
        out.loc[g.index[air], "airborne"] = True
        for c in ("x", "y"):
            out.loc[g.index[air], c] = np.interp(t[air], t[~air], g[c].to_numpy(float)[~air])
    return out


def coverage(players: pd.DataFrame, n_frames: int, per_side: int | dict[str, int] = 1) -> dict[str, float]:
    """側ごとの、選手の位置が出たコマの割合（補ったコマも含む）。``per_side`` は 1 側の人数（側ごとに違えば {側: 人数}）。"""
    per = per_side if isinstance(per_side, dict) else {side: per_side for side in SIDES}
    return {side: round(float((players["side"] == side).sum()) / max(1, n_frames * max(1, per.get(side, 1))), 4)
            for side in SIDES}


def players_per_side(players: pd.DataFrame) -> dict[str, int]:
    """``players.csv`` の側ごとの人数（その側の ``player`` の種類の数）。"""
    return {side: int(players.loc[players["side"] == side, "player"].nunique()) for side in SIDES}


# =============================================================================
# 動画 1 本
# =============================================================================
def _with_progress(frames: Iterator[tuple[int, float, np.ndarray]], n_total: int,
                   progress: Callable[[str], None], every: float = 0.1) -> Iterator[tuple[int, float, np.ndarray]]:
    step = max(1, int(n_total * every))
    for k, item in enumerate(frames, 1):
        if k % step == 0:
            progress(f"[進行] 選手を検出中 {min(100, round(100 * k / n_total))}%")
        yield item


def analyze_players(video: str | Path, calib: str | Path, out_dir: str | Path, cfg: dict[str, Any],
                    o: PlayerOptions | None = None, start_s: float | None = None, end_s: float | None = None,
                    capture_fps: float | None = None, prefer: str | None = None,
                    progress: Callable[[str], None] = print) -> dict[str, Any]:
    """動画の区間で選手を検出・追跡し、``players.csv`` と確かめ用の画像を書き出す。"""
    from smash_speed.batch import stream_bgr
    from smash_speed.calibration.io import load_calibration
    from smash_speed.movement import movement_brief, write_movement
    from smash_speed.video_io import probe

    o = o or PlayerOptions()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = probe(video, capture_fps, cfg)
    camera, _ = load_calibration(calib)
    if (camera.width, camera.height) != (meta.width, meta.height):
        camera = camera.scaled(meta.width / camera.width)
    detector = PersonDetector(ensure_model(o.model, progress=progress), o.min_score, o.nms_iou, prefer)
    scale = min(1.0, o.process_width / meta.width)
    ts = meta.timestamps
    lo = 0.0 if start_s is None else start_s
    hi = float(ts[-1]) if end_s is None else end_s
    n_total = max(1, int(np.sum((ts >= lo) & (ts <= hi))) // o.stride)
    frames = ((i, t, img) for i, t, img in stream_bgr(meta, int(round(meta.width * scale)), cfg, start_s, end_s)
              if lo <= t <= hi and i % o.stride == 0)
    seen: dict[int, float] = {}

    def remember(it):
        for i, t, img in it:
            seen[i] = t
            yield i, t, img

    t0 = time.time()
    # 枠の中の色は、ダブルスで 2 人を見分けるのに使う（シングルスと決めてあるときは要らない）
    people = detect_people(_with_progress(remember(frames), n_total, progress), detector, scale,
                           appearance=o.per_side != 1)
    elapsed = time.time() - t0
    people = add_court_positions(people, camera)
    people.to_csv(out_dir / PEOPLE_FILE, index=False, float_format="%.4f")
    frame_t = pd.Series(seen, dtype=float).sort_index()
    doubles_info = None
    if o.per_side != 1:
        from smash_speed.player_ids import track_sides

        tracked, doubles_info = track_sides(people, frame_t, cfg, o)
    else:
        tracked = track_players(people, frame_t, cfg, o)
    players = ground_jumps(tracked, camera, o)
    per_side = players_per_side(players)
    players.to_csv(out_dir / PLAYERS_FILE, index=False, float_format="%.4f")
    info = {
        "video": str(Path(video).resolve()), "calib": str(Path(calib).resolve()),
        "start_s": start_s, "end_s": end_s, "n_frames": len(frame_t), "n_people": len(people),
        "coverage": coverage(players, len(frame_t), per_side),
        "per_side": per_side,        # 側ごとの人数（2 ならダブルスとして追った）
        "doubles": doubles_info,     # ダブルスの追跡の確かめ用の数（``per_side`` が 1 のときは None）
        "airborne": {s: round(float(players["airborne"][players["side"] == s].mean()), 4) if (players["side"] == s).any()
                     else 0.0 for s in SIDES},
        "provider": detector.provider,
        "detect_s": round(elapsed, 1), "options": o.to_dict(),
    }
    if len(frame_t):
        write_preview(meta, players, people, int(frame_t.index[len(frame_t) // 2]), out_dir / PLAYERS_PREVIEW_FILE)
    info["movement"] = movement_brief(write_movement(out_dir, players, cfg.get("court")))
    (out_dir / "players.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def detect_in_windows(meta: Any, camera: Camera, detector: PersonDetector, windows: list[tuple[float, float]],
                      cfg: dict[str, Any] | None = None, o: PlayerOptions | None = None
                      ) -> tuple[pd.DataFrame, pd.Series]:
    """動画のいくつかの短い区間（[始まり, 終わり] [s]）だけで人を検出する（動画は 1 回だけ読む）。

    区間が重なっても、検出はコマごとに 1 回だけ。``add_court_positions`` の後の人と、検出したコマの番号 → 時刻を返す。
    区間ごとの選手は ``track_between`` で選ぶ。
    """
    from smash_speed.batch import stream_bgr

    o = o or PlayerOptions()
    if not windows:
        return add_court_positions(detect_people(iter(()), detector), camera), pd.Series(dtype=float)
    scale = min(1.0, o.process_width / meta.width)
    t_first = min(a for a, _ in windows)
    t_last = max(b for _, b in windows)
    frames = ((i, t, img) for i, t, img in stream_bgr(meta, int(round(meta.width * scale)), cfg,
                                                      max(0.0, t_first - 0.1), t_last + 0.1)
              if i % o.stride == 0 and any(a <= t <= b for a, b in windows))
    seen: dict[int, float] = {}

    def remember(it: Iterator[tuple[int, float, np.ndarray]]) -> Iterator[tuple[int, float, np.ndarray]]:
        for i, t, img in it:
            seen[i] = t
            yield i, t, img

    people = add_court_positions(detect_people(remember(frames), detector, scale), camera)
    return people, pd.Series(seen, dtype=float).sort_index()


def track_between(people: pd.DataFrame, frame_t: pd.Series, start_s: float, end_s: float, camera: Camera,
                  cfg: dict[str, Any], o: PlayerOptions | None = None) -> pd.DataFrame:
    """``detect_in_windows`` の結果のうち、区間 [start_s, end_s] のコマだけで選手を追跡する（跳んだコマは床に直す）。"""
    ft = frame_t[(frame_t >= start_s) & (frame_t <= end_s)]
    return ground_jumps(track_players(people[people["frame"].isin(ft.index)], ft, cfg, o), camera, o)


def write_preview(meta: Any, players: pd.DataFrame, people: pd.DataFrame, frame: int, path: Path,
                  width: int = 1280) -> None:
    """1 コマに、検出した人（細い灰色）と選んだ選手（手前 青・奥 橙）の枠とコートの位置を描いて保存する。"""
    from smash_speed.video_io import VideoReader

    with VideoReader(meta) as rd:
        img = rd.read_frame(frame)
    s = width / img.shape[1]
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    for _, p in people[people["frame"] == frame].iterrows():
        cv2.rectangle(img, (int(p.x0 * s), int(p.y0 * s)), (int(p.x1 * s), int(p.y1 * s)), (160, 160, 160), 1)
    colors = {"h": (255, 120, 0), "o": (0, 140, 255)}
    for _, p in players[players["frame"] == frame].iterrows():
        c = colors[p.side]
        cv2.rectangle(img, (int(p.x0 * s), int(p.y0 * s)), (int(p.x1 * s), int(p.y1 * s)), c, 2)
        cv2.putText(img, f"{p.side}{p.player} ({p.x:.1f}, {p.y:.1f}) m", (int(p.x0 * s), max(12, int(p.y0 * s) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def main(argv: list[str] | None = None) -> None:
    import argparse

    from smash_speed.config import load_config

    ap = argparse.ArgumentParser(description="動画の選手を検出・追跡し、コートの上の位置を players.csv に書き出す")
    ap.add_argument("video", type=Path)
    ap.add_argument("--calib", type=Path, required=True, help="calibration.json")
    ap.add_argument("--out", type=Path, required=True, help="出力先")
    ap.add_argument("--start", type=float, default=None, help="区間の始め [s]")
    ap.add_argument("--end", type=float, default=None, help="区間の終わり [s]")
    ap.add_argument("--capture-fps", type=float, default=None)
    ap.add_argument("--model", choices=list(MODELS), default=DEFAULT_MODEL)
    ap.add_argument("--stride", type=int, default=1, help="何コマごとに検出するか")
    ap.add_argument("--doubles", action="store_true", help="ダブルス（1 側 2 人）")
    ap.add_argument("--auto", action="store_true", help="シングルスかダブルスかを側ごとに自動で決める")
    ap.add_argument("--device", choices=["auto", "gpu", "cpu"], default=None)
    ap.add_argument("--config", type=Path, default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    o = PlayerOptions(model=a.model, stride=max(1, a.stride), per_side=2 if a.doubles else 0 if a.auto else 1)
    info = analyze_players(a.video, a.calib, a.out, cfg, o, a.start, a.end, a.capture_fps, a.device)
    print(f"{info['n_frames']} コマ（{info['provider']}、検出 {info['detect_s']} 秒）："
          f"手前 {info['coverage']['h']:.0%}・奥 {info['coverage']['o']:.0%} のコマで選手の位置が出ました → {a.out}")


if __name__ == "__main__":
    import sys

    if (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    main()
