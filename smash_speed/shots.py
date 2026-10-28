"""ショットとラリーのデータ形式と、``shots.csv``・``rallies.csv`` の読み書き。

1 本のショット（``Shot``）は「打った時刻・打った側・種類・初速・打点・着地点・軌道・信頼度」を持つ。
軌道は点の列ではなく、物理モデルの初期値（打点 p0・初速度 v0・終端速度 v_t）で持ち、
``Shot.trajectory`` で描き直す（CSV の 1 行に収まり、後から細かさを選べる）。
ラリー（``Rally``）はショットの並び（``Shot.rally_id`` と ``Shot.index_in_rally``）と得点を持つ。

打った側はコートの座標系（README「座標系」）の x の符号で表す。ランドマークの名前と同じく
``h`` = x < 0 の側、``o`` = x > 0 の側。分からなければ空文字。

CSV は 1 行 1 ショット（1 ラリー）、ヘッダーは ``SHOT_COLUMNS``・``RALLY_COLUMNS`` の順。
分からない値は空欄にする（読み込むと None）。文字コードは Excel で開けるよう BOM 付きの UTF-8。

``python -m smash_speed.shots <一括解析の出力先>`` で、今のスマッシュの一括解析の結果
（``candidates.json``）から ``shots.csv`` を書き出す。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from smash_speed.physics import MPS_TO_KMH, PhysicsParams, simulate_until_ground

SHOTS_FILE = "shots.csv"
RALLIES_FILE = "rallies.csv"

# ショットの種類（ShuttleSet の分類を、ROADMAP B4 の 8 種類にまとめたもの）。分からなければ unknown
SHOT_TYPES = ("smash", "clear", "drop", "drive", "lob", "net", "push", "serve", "unknown")
SHOT_TYPE_LABELS = {"smash": "スマッシュ", "clear": "クリア", "drop": "ドロップ", "drive": "ドライブ",
                    "lob": "ロブ", "net": "ネット", "push": "プッシュ", "serve": "サーブ", "unknown": "不明"}
SIDES = ("h", "o", "")


@dataclass
class Shot:
    """1 本のショット。時刻 [s]（映像の時刻）、位置 [m]（コートの座標）、速さ [km/h]。"""

    t_hit_s: float                         # 打った時刻
    side: str = ""                         # 打った側：h（x < 0）| o（x > 0）| ""（分からない）
    shot_type: str = "unknown"             # SHOT_TYPES のどれか
    speed_kmh: float | None = None         # 初速
    speed_ci_low_kmh: float | None = None  # 初速の 95% 信頼区間
    speed_ci_high_kmh: float | None = None
    hit_x_m: float | None = None           # 打点
    hit_y_m: float | None = None
    hit_z_m: float | None = None
    land_x_m: float | None = None          # 着地点（床面）
    land_y_m: float | None = None
    t_land_s: float | None = None          # 着地した時刻
    v0_x_mps: float | None = None          # 打点での速度（軌道を描き直すのに使う）[m/s]
    v0_y_mps: float | None = None
    v0_z_mps: float | None = None
    terminal_velocity_mps: float | None = None  # 当てはめた終端速度（空欄なら設定の値）
    confidence: float | None = None        # 信頼度 0〜1（大きいほど確か）
    rally_id: int | None = None            # 属するラリー
    index_in_rally: int | None = None      # ラリーの中で何本目か（1 から）
    source: str = ""                       # どこから来たか（smash_batch | manual | …）
    ref: str = ""                          # 元の結果の場所（出力先からの相対パス・候補の番号など）
    note: str = ""

    @property
    def hit_point(self) -> np.ndarray | None:
        v = (self.hit_x_m, self.hit_y_m, self.hit_z_m)
        return None if any(x is None for x in v) else np.array(v, float)

    @property
    def v0(self) -> np.ndarray | None:
        v = (self.v0_x_mps, self.v0_y_mps, self.v0_z_mps)
        return None if any(x is None for x in v) else np.array(v, float)

    def trajectory(self, phys: PhysicsParams | None = None) -> dict[str, np.ndarray] | None:
        """打点から床に着くまでの軌道（``simulate_until_ground`` と同じ形）。打点・速度がなければ None。"""
        p0, v0 = self.hit_point, self.v0
        if p0 is None or v0 is None:
            return None
        phys = phys or PhysicsParams()
        if self.terminal_velocity_mps:
            phys = PhysicsParams(phys.g, self.terminal_velocity_mps, phys.method, phys.rtol, phys.atol)
        return simulate_until_ground(p0, v0, self.t_hit_s, phys)


@dataclass
class Rally:
    """1 つのラリー。得点はラリーが終わった後の値。"""

    rally_id: int
    t_start_s: float                       # 最初のショットを打った時刻
    t_end_s: float | None = None           # シャトルが落ちた（ラリーが終わった）時刻
    n_shots: int = 0
    winner_side: str = ""                  # 得点した側：h | o | ""
    end_reason: str = ""                   # 終わり方：in（相手のコートに落ちた）| out | net | fault | ""
    score_h: int | None = None             # ラリーの後の得点
    score_o: int | None = None
    note: str = ""


SHOT_COLUMNS = [f.name for f in fields(Shot)]
RALLY_COLUMNS = [f.name for f in fields(Rally)]


# =============================================================================
# 読み書き
# =============================================================================
def _cell(v: Any) -> Any:
    """CSV に書く値。None・NaN は空欄にする。"""
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ""
    return v


def _parse(value: str, typ: Any) -> Any:
    """CSV の文字列を、フィールドの型（``"float | None"`` などの注釈の文字列）に戻す。"""
    t = str(typ)
    if value == "":
        return "" if t == "str" else None
    if t.startswith("int"):
        return int(float(value))
    if t.startswith("float"):
        return float(value)
    return value


def _write(path: Path, columns: list[str], rows: Iterable[Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in rows:
            d = asdict(r)
            w.writerow([_cell(d[c]) for c in columns])
    tmp.replace(path)
    return path


def _read(path: Path, cls: type) -> list[Any]:
    types = {f.name: f.type for f in fields(cls)}
    out = []
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            # 知らない列（後から足した列・人が足した列）は読み飛ばし、ない列は既定値にする
            out.append(cls(**{k: _parse(v, types[k]) for k, v in row.items() if k in types}))
    return out


def write_shots(path: str | Path, shots: Iterable[Shot]) -> Path:
    """ショットを打った時刻の順に ``shots.csv`` に書く。"""
    shots = sorted(shots, key=lambda s: s.t_hit_s)
    for s in shots:
        validate_shot(s)
    return _write(Path(path), SHOT_COLUMNS, shots)


def read_shots(path: str | Path) -> list[Shot]:
    return _read(Path(path), Shot)


def write_rallies(path: str | Path, rallies: Iterable[Rally]) -> Path:
    return _write(Path(path), RALLY_COLUMNS, sorted(rallies, key=lambda r: r.t_start_s))


def read_rallies(path: str | Path) -> list[Rally]:
    return _read(Path(path), Rally)


def validate_shot(s: Shot) -> None:
    """種類・打った側・信頼度が決まった値の中にあるかを確かめる。"""
    if s.shot_type not in SHOT_TYPES:
        raise ValueError(f"ショットの種類 {s.shot_type!r} は {SHOT_TYPES} のどれかにしてください")
    if s.side not in SIDES:
        raise ValueError(f"打った側 {s.side!r} は h・o・空文字のどれかにしてください")
    if s.confidence is not None and not 0.0 <= s.confidence <= 1.0:
        raise ValueError(f"信頼度 {s.confidence} は 0〜1 にしてください")


# =============================================================================
# ラリーとショットの対応
# =============================================================================
def side_of(x_m: float | None) -> str:
    """コートの x 座標から、どちらの側か（h | o）。分からなければ空文字。"""
    if x_m is None or not math.isfinite(x_m) or x_m == 0.0:
        return ""
    return "h" if x_m < 0 else "o"


def shots_of(rally: Rally, shots: Iterable[Shot]) -> list[Shot]:
    """ラリーに属するショットを、ラリーの中の順に返す。"""
    mine = [s for s in shots if s.rally_id == rally.rally_id]
    return sorted(mine, key=lambda s: (s.index_in_rally if s.index_in_rally is not None else 10**9, s.t_hit_s))


def make_rallies(shots: list[Shot], max_gap_s: float = 3.0) -> list[Rally]:
    """打った時刻の間隔が ``max_gap_s`` を超えたら別のラリーとして、ショットにラリーの番号と順番を付ける。

    打球の間隔だけで分ける仮の区切り（着地・シャトルが飛んでいるかも使う区切りは ``rallies.split_rallies``）。
    得点は分からないので空欄。
    ``shots`` の ``rally_id``・``index_in_rally`` を書き換える。
    """
    rallies: list[Rally] = []
    prev: float | None = None
    for s in sorted(shots, key=lambda s: s.t_hit_s):
        if prev is None or s.t_hit_s - prev > max_gap_s:
            rallies.append(Rally(rally_id=len(rallies) + 1, t_start_s=s.t_hit_s))
        r = rallies[-1]
        r.n_shots += 1
        s.rally_id, s.index_in_rally = r.rally_id, r.n_shots
        r.t_end_s = s.t_land_s if s.t_land_s is not None else s.t_hit_s
        prev = s.t_hit_s
    return rallies


# =============================================================================
# 今のスマッシュの一括解析の結果から
# =============================================================================
def _f(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def confidence_from_ci(kmh: float | None, lo: float | None, hi: float | None) -> float | None:
    """初速の 95% 信頼区間の幅から信頼度を出す：1 −（幅 ÷ 初速）を 0〜1 に収めたもの。

    幅が初速の 10% なら 0.9、初速と同じ幅なら 0。信頼区間がなければ None。
    """
    if not kmh or lo is None or hi is None or kmh <= 0:
        return None
    return float(np.clip(1.0 - (hi - lo) / kmh, 0.0, 1.0))


def shot_from_fit(fit: dict[str, Any], t_hit_s: float | None = None, phys: PhysicsParams | None = None,
                  **kw: Any) -> Shot:
    """``fit_result.json`` の中身（``FitResult.to_dict``）から、打点・初速・軌道・着地点を埋めたショットを作る。"""
    p0, v0 = np.asarray(fit["p0_m"], float), np.asarray(fit["v0_mps"], float)
    vt = _f(fit.get("terminal_velocity_mps"))
    t0 = float(fit["t_ref_s"]) if t_hit_s is None else t_hit_s
    ci = fit.get("speed0_ci_mps") or [None, None]
    kmh = float(np.linalg.norm(v0)) * MPS_TO_KMH
    lo = _f(ci[0]) * MPS_TO_KMH if _f(ci[0]) is not None else None
    hi = _f(ci[1]) * MPS_TO_KMH if _f(ci[1]) is not None else None
    s = Shot(t_hit_s=t0, side=side_of(float(p0[0])), speed_kmh=kmh, speed_ci_low_kmh=lo, speed_ci_high_kmh=hi,
             hit_x_m=float(p0[0]), hit_y_m=float(p0[1]), hit_z_m=float(p0[2]),
             v0_x_mps=float(v0[0]), v0_y_mps=float(v0[1]), v0_z_mps=float(v0[2]), terminal_velocity_mps=vt,
             confidence=confidence_from_ci(kmh, lo, hi))
    for k, v in kw.items():
        setattr(s, k, v)
    tr = s.trajectory(phys)
    if tr is not None and np.isfinite(tr["t_land"]):
        s.land_x_m, s.land_y_m = float(tr["pos"][-1, 0]), float(tr["pos"][-1, 1])
        s.t_land_s = float(tr["t_land"])
    return s


def shots_from_candidates(out_dir: str | Path, phys: PhysicsParams | None = None) -> list[Shot]:
    """一括解析の出力先（``candidates.json``）の、詳しく解析できたスマッシュをショットにする。

    種類は smash。初速・信頼区間は ``ranking.csv`` と同じ値（``Candidate.result``）を使い、
    打点・速度・着地点は各候補の ``fit_result.json`` から埋める（なければ空欄）。
    """
    from smash_speed.batch import load_candidates

    out_dir = Path(out_dir)
    cands, _ = load_candidates(out_dir)
    shots: list[Shot] = []
    for c in cands:
        r = c.result or {}
        if r.get("kmh") is None:
            continue
        fit_path = out_dir / c.label / "fit_result.json"
        common = dict(shot_type="smash", source="smash_batch", ref=f"{c.label}",
                      note="" if c.confirmed else "打つ人・着地点は未確認")
        if fit_path.exists():
            s = shot_from_fit(json.loads(fit_path.read_text(encoding="utf-8")), phys=phys, **common)
        else:
            t0 = c.t_impact if c.t_impact is not None else c.t_onset
            s = Shot(t_hit_s=float(t0), side=side_of((c.hitter or {}).get("x")), **common)
            if c.landing:
                s.land_x_m, s.land_y_m = _f(c.landing.get("x")), _f(c.landing.get("y"))
        # 初速は順位表と同じ値にそろえる
        s.speed_kmh, s.speed_ci_low_kmh, s.speed_ci_high_kmh = _f(r["kmh"]), _f(r.get("ci_low_kmh")), \
            _f(r.get("ci_high_kmh"))
        s.confidence = confidence_from_ci(s.speed_kmh, s.speed_ci_low_kmh, s.speed_ci_high_kmh)
        shots.append(s)
    return sorted(shots, key=lambda s: s.t_hit_s)


def export_batch(out_dir: str | Path, phys: PhysicsParams | None = None) -> Path:
    """一括解析の結果を ``<出力先>/shots.csv`` に書き出す。"""
    return write_shots(Path(out_dir) / SHOTS_FILE, shots_from_candidates(out_dir, phys))


def main(argv: list[str] | None = None) -> None:
    from smash_speed.config import load_config

    ap = argparse.ArgumentParser(description="スマッシュの一括解析の結果から shots.csv を書き出す")
    ap.add_argument("out_dir", type=Path, help="一括解析の出力先（candidates.json がある所）")
    ap.add_argument("--config", type=Path, default=None)
    a = ap.parse_args(argv)
    path = export_batch(a.out_dir, PhysicsParams.from_config(load_config(a.config)))
    print(f"{len(read_shots(path))} 本のショットを {path} に書き出しました")


if __name__ == "__main__":
    main()
