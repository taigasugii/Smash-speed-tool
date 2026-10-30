"""お手本との類似度スコア（ROADMAP E5）。

正規化した姿勢（``pose_prep.NormalizedPose``）どうしを比べて 0〜100 点にする。2 つの点の付け方を出す。

1. **DTW スコア**（``score_dtw``）：関節角度の時間の列（肘・肩・両膝・胸と腰の回旋）を、動的時間伸縮（DTW）で
   時間をずらして合わせ、合わせた後の 1 コマあたりの角度の差の平均 d [deg] を 100·exp(−d / ``DTW_SCALE_DEG``) にする。
   スイングの速さ・タイミングの少しの違いでは下がらず、形の違いで下がる。
2. **コサインスコア**（``score_cos``）：DTW で合わせたコマどうしで、姿勢のベクトル（体の 10 本の骨（上腕・前腕・
   太もも・すね・肩の線・腰の線）の向きの単位ベクトルを並べたもの）のコサイン類似度をとり、その平均を 100 倍する
   （0 未満は 0）。骨ごとの向きのずれ Δ の cos の平均と同じなので、``cos_angle_deg`` = arccos(平均) も出す。

お手本（``Reference``）は、1 本のショットか、何本かのショットを 1 本目に DTW で合わせて平均したもの。差し替えて
点の付け方を変えられる。``reference_sensitivity`` で、お手本ごとの点の違い（平均・ばらつき・順位の相関）を出す。

``python -m smash_speed.form_score make-ref <お手本.npz> <姿勢.npz> [...]`` でお手本を作り、
``python -m smash_speed.form_score score <出力先> --ref <お手本.npz> [--ref ...]`` で ``form/scores.csv`` に書き出す。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from smash_speed.pose import KP
from smash_speed.pose_features import _series, angle_deg
from smash_speed.pose_prep import NormalizedPose

DTW_SCALE_DEG = 20.0      # 角度の差の平均がこれだけあると 100·e⁻¹ ≈ 37 点
BAND_S = 0.2              # DTW でずらしてよい時間の幅（Sakoe–Chiba の帯）[s]
WINDOW = (-1.0, 0.5)      # 比べる区間（打点を 0）[s]
STEP_S = 1 / 60           # 比べる前に、この間隔の時刻にそろえる（fps の違う動画どうしでも比べられる）
ANGLE_NAMES = ("elbow", "shoulder", "front_knee", "back_knee", "trunk_turn", "pelvis_turn")
BONES = (("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"), ("left_shoulder", "left_elbow"),
         ("left_elbow", "left_wrist"), ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
         ("left_hip", "left_knee"), ("left_knee", "left_ankle"), ("right_shoulder", "left_shoulder"),
         ("right_hip", "left_hip"))


def _resample(t: np.ndarray, x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """時刻 t の値 x (T, ...) を grid の時刻に直線で写す（欠けは前後から埋める。区間の外は端の値）。"""
    x = np.asarray(x, float)
    flat = x.reshape(len(t), -1)
    out = np.empty((len(grid), flat.shape[1]))
    for j in range(flat.shape[1]):
        ok = np.isfinite(flat[:, j])
        out[:, j] = np.interp(grid, t[ok], flat[ok, j]) if ok.sum() >= 2 else np.nan
    return out.reshape((len(grid),) + x.shape[1:])


def angle_series(n: NormalizedPose) -> np.ndarray:
    """関節角度の時間の列 (T, 6) [deg]（``ANGLE_NAMES`` の順）。欠けを埋め、特徴量と同じくなめらかにする。"""
    J = n.joint
    cols = [angle_deg(J("right_shoulder"), J("right_elbow"), J("right_wrist")),
            angle_deg(J("right_elbow"), J("right_shoulder"), J("right_hip")),
            angle_deg(J("left_hip"), J("left_knee"), J("left_ankle")),
            angle_deg(J("right_hip"), J("right_knee"), J("right_ankle"))]
    cols = [_series(n, c) for c in cols]
    # 胸・腰の回り：特徴量の arccos(幅 ÷ 最大) は横向きの近くで幅の揺れを大きくする（1 px の揺れで 10° ほど）ので、
    # ここでは幅に比例する 90 × (1 − 幅 ÷ 最大) を使う（横向きで 0、正面で 90 なのは同じ）
    for part in ("shoulder", "hip"):
        w = _series(n, np.abs(n.joint(f"left_{part}")[:, 0] - n.joint(f"right_{part}")[:, 0]))
        cols.append(None if w is None or np.max(w) <= 0 else 90.0 * (1.0 - w / np.max(w)))
    return np.column_stack([c if c is not None else np.full(len(n.t), np.nan) for c in cols])


def pose_vectors(n: NormalizedPose) -> np.ndarray:
    """10 本の骨の向きの単位ベクトルを並べたもの (T, 20)。体の大きさ・腰の位置によらない。"""
    out = []
    for a, b in BONES:
        d = n.xy[:, KP[b]] - n.xy[:, KP[a]]
        L = np.linalg.norm(d, axis=1, keepdims=True)
        out.append(np.where(L > 1e-9, d / np.where(L > 1e-9, L, 1.0), np.nan))
    return np.concatenate(out, axis=1)


@dataclass
class Reference:
    """お手本：時刻をそろえた関節角度と姿勢のベクトル。"""

    t: np.ndarray            # (T,) [s]（打点が 0）
    angles: np.ndarray       # (T, 6) [deg]
    vectors: np.ndarray      # (T, 20)
    name: str = ""
    sources: list[str] = field(default_factory=list)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, t=self.t, angles=self.angles, vectors=self.vectors,
                            info=json.dumps({"name": self.name, "sources": self.sources}, ensure_ascii=False))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Reference":
        with np.load(path, allow_pickle=False) as z:
            info = json.loads(str(z["info"]))
            return cls(t=z["t"], angles=z["angles"], vectors=z["vectors"], name=info.get("name") or Path(path).stem,
                       sources=info.get("sources", []))


def prepare(n: NormalizedPose, window: tuple[float, float] = WINDOW, step: float = STEP_S
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """比べるための（時刻、角度 (T, 6)、姿勢のベクトル (T, 20)）。決まった間隔の時刻にそろえる。"""
    lo, hi = max(window[0], float(n.t[0])), min(window[1], float(n.t[-1]))
    grid = np.arange(np.ceil(lo / step) * step, hi + 1e-9, step)
    return grid, _resample(n.t, angle_series(n), grid), _resample(n.t, pose_vectors(n), grid)


def reference_from(n: NormalizedPose, name: str = "") -> Reference:
    t, a, v = prepare(n)
    return Reference(t=t, angles=a, vectors=v, name=name or str(n.meta.get("shot_id", "")),
                     sources=[str(n.meta.get("shot_id", ""))])


def dtw(a: np.ndarray, b: np.ndarray, band: int | None = None) -> tuple[float, list[tuple[int, int]]]:
    """2 つの列 (Ta, D)・(Tb, D) の DTW。コマどうしの距離は、欠けていない列の差の 2 乗平均の平方根。

    Returns:
        合わせた道の距離の平均（1 歩あたり）と、道（(i, j) の並び）
    """
    Ta, Tb = len(a), len(b)
    diff = a[:, None, :] - b[None, :, :]
    ok = np.isfinite(diff)
    cnt = ok.sum(axis=2)
    cost = np.sqrt(np.where(cnt > 0, np.nansum(np.where(ok, diff, 0.0) ** 2, axis=2) / np.maximum(cnt, 1), np.nan))
    cost = np.where(np.isfinite(cost), cost, np.nanmax(cost) if np.isfinite(cost).any() else 1.0)
    band = max(Ta, Tb) if band is None else max(band, abs(Ta - Tb))
    D = np.full((Ta + 1, Tb + 1), np.inf)
    D[0, 0] = 0.0
    steps = np.zeros((Ta + 1, Tb + 1), int)
    for i in range(1, Ta + 1):
        j0, j1 = max(1, i - band), min(Tb, i + band)
        for j in range(j0, j1 + 1):
            c = cost[i - 1, j - 1]
            prev = (D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
            k = int(np.argmin(prev))
            D[i, j] = c + prev[k]
            steps[i, j] = k
    path = []
    i, j = Ta, Tb
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        k = steps[i, j]
        i, j = (i - 1, j - 1) if k == 0 else (i - 1, j) if k == 1 else (i, j - 1)
    path.reverse()
    return float(D[Ta, Tb] / len(path)), path


def _cos(u: np.ndarray, v: np.ndarray) -> float:
    ok = np.isfinite(u) & np.isfinite(v)
    if ok.sum() < 6:
        return np.nan
    u, v = u[ok], v[ok]
    den = np.linalg.norm(u) * np.linalg.norm(v)
    return float(u @ v / den) if den > 0 else np.nan


@dataclass
class Score:
    score_dtw: float
    score_cos: float
    cos_angle_deg: float    # 骨の向きのずれの目安 arccos(コサイン類似度の平均) [deg]
    dtw_deg: float          # 合わせた後の角度の差の平均 [deg]
    lockstep_deg: float     # 時間をずらさずに比べたときの角度の差の平均 [deg]（参考）
    reference: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {k: (round(v, 4) if isinstance(v, float) and np.isfinite(v) else (None if isinstance(v, float) else v))
                for k, v in self.__dict__.items()}


def score(n: NormalizedPose, ref: Reference, scale_deg: float = DTW_SCALE_DEG, band_s: float = BAND_S) -> Score:
    """お手本との類似度スコア。"""
    t, a, v = prepare(n)
    step = float(np.median(np.diff(ref.t))) if len(ref.t) > 1 else STEP_S
    d, path = dtw(a, ref.angles, band=int(round(band_s / step)))
    cos = [_cos(v[i], ref.vectors[j]) for i, j in path]
    cos = [c for c in cos if np.isfinite(c)]
    # 時間をずらさない比べ方（同じ時刻どうし）
    common = np.intersect1d(np.round(t / step).astype(int), np.round(ref.t / step).astype(int))
    ia = np.searchsorted(np.round(t / step).astype(int), common)
    ib = np.searchsorted(np.round(ref.t / step).astype(int), common)
    diff = a[ia] - ref.angles[ib]
    lock = float(np.sqrt(np.nanmean(diff**2))) if len(common) else np.nan
    mc = float(np.mean(cos)) if cos else np.nan
    return Score(score_dtw=float(100.0 * np.exp(-d / scale_deg)),
                 score_cos=float(100.0 * max(0.0, mc)) if cos else np.nan,
                 cos_angle_deg=float(np.degrees(np.arccos(np.clip(mc, -1, 1)))) if cos else np.nan,
                 dtw_deg=d, lockstep_deg=lock, reference=ref.name)


def make_reference(poses: list[NormalizedPose], name: str = "reference") -> Reference:
    """何本かのショットの平均のお手本。2 本目からは 1 本目に DTW で合わせて、1 本目の時刻ごとに平均する。"""
    if not poses:
        raise ValueError("お手本にするショットがありません")
    t0, a0, v0 = prepare(poses[0])
    sum_a, sum_v = np.nan_to_num(a0), np.nan_to_num(v0)
    cnt_a, cnt_v = np.isfinite(a0).astype(float), np.isfinite(v0).astype(float)
    for n in poses[1:]:
        _, a, v = prepare(n)
        _, path = dtw(a0, a, band=int(round(BAND_S / STEP_S)))
        for i, j in path:
            sum_a[i] += np.nan_to_num(a[j]); cnt_a[i] += np.isfinite(a[j])
            sum_v[i] += np.nan_to_num(v[j]); cnt_v[i] += np.isfinite(v[j])
    with np.errstate(invalid="ignore", divide="ignore"):
        A = np.where(cnt_a > 0, sum_a / cnt_a, np.nan)
        V = np.where(cnt_v > 0, sum_v / cnt_v, np.nan)
    return Reference(t=t0, angles=A, vectors=V, name=name, sources=[str(n.meta.get("shot_id", "")) for n in poses])


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    return r


def reference_sensitivity(table: dict[str, dict[str, float]]) -> dict[str, Any]:
    """お手本ごとの点（``{お手本: {ショット: 点}}``）の違い：お手本ごとの平均・標準偏差と、
    お手本どうしの順位の相関（Spearman）。お手本で点が変わっても順位が同じなら、比べる目的には差し支えない。"""
    refs = list(table)
    shots = sorted(set.intersection(*(set(v) for v in table.values()))) if refs else []
    out: dict[str, Any] = {"n_shots": len(shots), "per_reference": {}, "rank_correlation": {}}
    for r in refs:
        x = np.array([table[r][s] for s in shots], float)
        out["per_reference"][r] = {"mean": float(np.mean(x)) if len(x) else None,
                                   "sd": float(np.std(x)) if len(x) else None}
    for i, r1 in enumerate(refs):
        for r2 in refs[i + 1:]:
            x = np.array([table[r1][s] for s in shots]); y = np.array([table[r2][s] for s in shots])
            rho = float(np.corrcoef(_rank(x), _rank(y))[0, 1]) if len(shots) >= 3 else None
            out["rank_correlation"][f"{r1} / {r2}"] = rho
    return out


def score_dir(form_dir: str | Path, refs: list[Reference]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """``form/poses/*.npz`` をすべてのお手本と比べる。1 行 1 ショット × お手本。"""
    d = Path(form_dir)
    rows, table = [], {r.name: {} for r in refs}
    for p in sorted((d / "poses").glob("*.npz")):
        n = NormalizedPose.load(p)
        for r in refs:
            s = score(n, r)
            rows.append({"shot_id": p.stem, **s.to_dict()})
            table[r.name][p.stem] = s.score_dtw
    return rows, reference_sensitivity(table)


def main(argv: list[str] | None = None) -> int:
    from smash_speed.form import FORM_DIR, write_rows

    p = argparse.ArgumentParser(description="お手本との類似度スコア")
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-ref", help="ショット（正規化した姿勢の .npz）からお手本を作る")
    m.add_argument("out")
    m.add_argument("poses", nargs="+")
    m.add_argument("--name", default=None)
    s = sub.add_parser("score", help="出力先のショットをお手本と比べ、form/scores.csv に書き出す")
    s.add_argument("out_dir")
    s.add_argument("--ref", action="append", required=True)
    args = p.parse_args(argv)
    if args.cmd == "make-ref":
        ref = make_reference([NormalizedPose.load(x) for x in args.poses], args.name or Path(args.out).stem)
        ref.save(args.out)
        print(f"お手本を作りました（{len(args.poses)} 本の平均）: {args.out}")
        return 0
    d = Path(args.out_dir)
    d = d / FORM_DIR if (d / FORM_DIR).is_dir() else d
    rows, sens = score_dir(d, [Reference.load(r) for r in args.ref])
    write_rows(d / "scores.csv", rows, ["shot_id", "reference", "score_dtw", "score_cos", "cos_angle_deg", "dtw_deg",
                                        "lockstep_deg"])
    (d / "scores_sensitivity.json").write_text(json.dumps(sens, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(rows)} 行を書き出しました: {d / 'scores.csv'}")
    for r, v in sens["per_reference"].items():
        print(f"  {r}: 平均 {v['mean']:.1f} 点、ばらつき {v['sd']:.1f}" if v["mean"] is not None else f"  {r}: —")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
