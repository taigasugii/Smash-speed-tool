"""ショットの種類の判定（ルール）。

1 本のショットの軌道（打点・初速度・終端速度）から、種類を決める手がかり（``ShotFeatures``）を出し、
決まった順の条件（``classify_features``）でスマッシュ／クリア／ドロップ／ドライブ／ロブ／ネット／プッシュ／サーブに分ける。
分類は ShuttleSet の体系を ROADMAP B4 の 8 種類にまとめたもの（``smash_speed.shots.SHOT_TYPES``）。

手がかりは、初速・打ち出し角・打点の高さ・打つ位置の深さ（ネットからの距離）・ネットの上を通る高さ・
軌道のいちばん高い所・着地点の深さ（相手側のネットからの距離）。着地点は「誰も打たなかったら床に着いた所」
（打ち返されたショットも、軌道を床まで延ばして決める）。

判定の順番（しきい値は ``RuleThresholds``）：

1. ラリーの最初のショット → サーブ
2. 打点が高い（頭より上で打った）：下向きで速い → スマッシュ、深く飛ぶ → クリア、それ以外 → ドロップ
3. 打点が低い〜中くらい：ネットの近くに落ちる → ネット、高く上がって深く飛ぶ → ロブ、
   それ以外（低い軌道）は速ければドライブ、遅ければプッシュ

``python -m smash_speed.shot_types --bench`` で、合成ラリーでの混同行列を出す。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable

import numpy as np
import pandas as pd

from smash_speed.physics import PhysicsParams, simulate_until_ground
from smash_speed.shots import Shot

# 判定の対象（unknown を除く）
CLASSES = ("smash", "clear", "drop", "drive", "lob", "net", "push", "serve")


@dataclass
class ShotFeatures:
    """種類を決める手がかり。速さ [m/s]、角度 [度]、長さ・高さ [m]。分からない値は nan。"""

    speed_mps: float          # 初速
    elevation_deg: float      # 打ち出し角（上向きが正）
    hit_z_m: float            # 打点の高さ
    hit_depth_m: float        # 打つ位置の深さ（ネットからの距離 |x|）
    apex_z_m: float           # 軌道のいちばん高い所
    net_z_m: float            # ネットの真上を通る高さ（ネットを越えなければ nan）
    land_depth_m: float       # 着地点の深さ（相手側のネットからの距離。自分の側に落ちれば負）
    first_in_rally: bool = False  # ラリーの最初のショット


@dataclass(frozen=True)
class RuleThresholds:
    """判定のしきい値。合成ラリー（``synth_rally.SHOT_PROFILES``）の範囲から決めた。"""

    overhead_z_m: float = 1.95        # これより高い打点は、頭より上で打ったショット（スマッシュ・クリア・ドロップ）
    smash_speed_mps: float = 36.0     # 頭より上で、下向きでこれより速ければスマッシュ
    smash_max_elev_deg: float = 0.0   # スマッシュの打ち出し角の上限
    clear_land_m: float = 3.8         # 頭より上で、これより深く飛べばクリア（浅ければドロップ）
    net_land_m: float = 1.75          # 頭より下で、これより浅く（ネットの近くに）落ちればネット
    deep_land_m: float = 4.0          # 頭より下で、これより深く飛び、高く上がればロブ
    lob_elev_deg: float = 28.0        # ロブの打ち出し角の下限（または lob_apex_m より高く上がる）
    lob_apex_m: float = 3.0
    drive_speed_mps: float = 20.0     # 低い軌道で、これより速ければドライブ（遅ければプッシュ）


def features_from_trajectory(p0: np.ndarray, v0: np.ndarray, phys: PhysicsParams,
                             first_in_rally: bool = False) -> ShotFeatures:
    """打点 ``p0`` [m] と打点での速度 ``v0`` [m/s] から、軌道を床まで延ばして手がかりを出す。"""
    p0, v0 = np.asarray(p0, float), np.asarray(v0, float)
    tr = simulate_until_ground(p0, v0, 0.0, phys, t_max=8.0, n_dense=4)
    pos = tr["pos"]
    sign = np.sign(p0[0]) if p0[0] != 0 else -np.sign(v0[0]) or -1.0
    # ネットの真上（x = 0）を通る高さ：x の符号が変わる所で線形に補う
    net_z = np.nan
    cross = np.flatnonzero(np.sign(pos[:-1, 0]) != np.sign(pos[1:, 0]))
    if len(cross):
        i = int(cross[0])
        w = pos[i, 0] / (pos[i, 0] - pos[i + 1, 0])
        net_z = float(pos[i, 2] + w * (pos[i + 1, 2] - pos[i, 2]))
    land_depth = float(-sign * pos[-1, 0]) if np.isfinite(tr["t_land"]) else np.nan
    return ShotFeatures(
        speed_mps=float(np.linalg.norm(v0)),
        elevation_deg=float(np.degrees(np.arctan2(v0[2], np.linalg.norm(v0[:2])))),
        hit_z_m=float(p0[2]), hit_depth_m=float(abs(p0[0])), apex_z_m=float(pos[:, 2].max()),
        net_z_m=net_z, land_depth_m=land_depth, first_in_rally=first_in_rally)


def shot_features(shot: Shot, phys: PhysicsParams | None = None,
                  first_in_rally: bool | None = None) -> ShotFeatures | None:
    """``Shot`` の打点・速度から手がかりを出す。打点・速度がなければ None。

    ``first_in_rally`` を省略すると ``shot.index_in_rally == 1`` で決める。
    """
    p0, v0 = shot.hit_point, shot.v0
    if p0 is None or v0 is None:
        return None
    phys = phys or PhysicsParams()
    if shot.terminal_velocity_mps:
        phys = PhysicsParams(phys.g, shot.terminal_velocity_mps, phys.method, phys.rtol, phys.atol)
    if first_in_rally is None:
        first_in_rally = shot.index_in_rally == 1
    return features_from_trajectory(p0, v0, phys, first_in_rally)


def classify_features(f: ShotFeatures, th: RuleThresholds | None = None) -> str:
    """手がかりから種類（``CLASSES`` のどれか）を決める。"""
    th = th or RuleThresholds()
    if f.first_in_rally:
        return "serve"
    land = f.land_depth_m if np.isfinite(f.land_depth_m) else 0.0
    if f.hit_z_m >= th.overhead_z_m:
        if f.elevation_deg <= th.smash_max_elev_deg and f.speed_mps >= th.smash_speed_mps:
            return "smash"
        return "clear" if land >= th.clear_land_m else "drop"
    if land < th.net_land_m:
        return "net"
    if land >= th.deep_land_m and (f.elevation_deg >= th.lob_elev_deg or f.apex_z_m >= th.lob_apex_m):
        return "lob"
    return "drive" if f.speed_mps >= th.drive_speed_mps else "push"


def classify_shots(shots: Iterable[Shot], phys: PhysicsParams | None = None,
                   th: RuleThresholds | None = None, overwrite: bool = False) -> list[Shot]:
    """ショットの種類を決めた ``Shot`` の列を返す（元の列は変えない）。

    種類が決まっている（unknown でない）ショットは、``overwrite`` でなければそのまま。
    打点・速度のないショットは unknown のまま。ラリーの最初かどうかは ``index_in_rally`` で決める。
    """
    out = []
    for s in shots:
        if s.shot_type != "unknown" and not overwrite:
            out.append(s)
            continue
        f = shot_features(s, phys)
        out.append(replace(s, shot_type="unknown" if f is None else classify_features(f, th)))
    return out


def confusion_matrix(true: Iterable[str], pred: Iterable[str], labels: Iterable[str] = CLASSES) -> pd.DataFrame:
    """混同行列（行が正解、列が判定。unknown の列も付ける）。"""
    labels = list(labels)
    cols = labels + ["unknown"]
    m = pd.DataFrame(0, index=labels, columns=cols)
    for t, p in zip(true, pred):
        if t in m.index:
            m.loc[t, p if p in cols else "unknown"] += 1
    return m


def mistakes(m: pd.DataFrame, top: int = 5) -> list[tuple[str, str, int]]:
    """多い間違い（正解, 判定, 本数）を多い順に。"""
    out = [(t, p, int(m.loc[t, p])) for t in m.index for p in m.columns if t != p and m.loc[t, p] > 0]
    return sorted(out, key=lambda x: -x[2])[:top]


def scores(m: pd.DataFrame) -> dict[str, Any]:
    """混同行列から、正解率と種類ごとの再現率・適合率。"""
    n = int(m.values.sum())
    diag = {k: int(m.loc[k, k]) for k in m.index}
    out = {"n": n, "accuracy": sum(diag.values()) / n if n else None, "by_type": {}}
    for k in m.index:
        n_true, n_pred = int(m.loc[k].sum()), int(m[k].sum())
        out["by_type"][k] = {"n": n_true, "recall": diag[k] / n_true if n_true else None,
                             "precision": diag[k] / n_pred if n_pred else None}
    return out


# =============================================================================
# 合成ラリーでの測り方
# =============================================================================
def truth_benchmark(n_rallies: int = 100, seed: int = 0, phys: PhysicsParams | None = None) -> dict[str, Any]:
    """合成ラリーの正解の軌道（打点・初速度）から判定する。ルールそのものの限界を見る（当てはめの誤差なし）。

    Returns:
        ``matrix``（混同行列）、``scores``、``rows``（1 ショット 1 行：正解・判定・手がかり）
    """
    from smash_speed.synth_rally import make_rally

    phys = phys or PhysicsParams()
    rng = np.random.default_rng([seed, 1])
    rows = []
    for r in range(n_rallies):
        rally = make_rally(rng, phys, n_shots=int(rng.integers(4, 13)), server_side="h" if r % 2 == 0 else "o")
        for i, s in enumerate(rally.segments):
            f = features_from_trajectory(s.p0, s.v0, phys, i == 0)
            rows.append({"rally": r, "index": i, "profile": s.profile, "true": s.shot_type,
                         "pred": classify_features(f), **asdict(f)})
    return _summary(pd.DataFrame(rows))


def synthetic_benchmark(n_rallies: int = 10, fps: float = 120.0, seed: int = 0, hits: str = "audio",
                        phys: PhysicsParams | None = None, **kwargs: Any) -> dict[str, Any]:
    """合成ラリーを観測し、打球を見つけて軌道を当てはめた結果（``shot_fit.synthetic_benchmark``）から判定する。

    ラリーの最初のショットは「見つけた最初の打球」（正解の順番は使わない）。
    当てはめられなかったショット（判定は unknown）も数える。見つからなかった打球は数えない。
    """
    from smash_speed.shot_fit import synthetic_benchmark as fit_benchmark

    phys = phys or PhysicsParams()
    r = fit_benchmark(n_rallies, fps, seed, hits, keep_shots=True, **kwargs)
    rows = []
    for _, row in r["rows"].iterrows():
        f = shot_features(row["shot"], phys)
        rows.append({"rally": row.rally, "index": row["index"], "profile": row.profile, "true": row.shot_type,
                     "pred": "unknown" if f is None else classify_features(f), "err_pct": row.err_pct,
                     **(asdict(f) if f is not None else {})})
    return _summary(pd.DataFrame(rows))


def _summary(df: pd.DataFrame) -> dict[str, Any]:
    m = confusion_matrix(df.true, df.pred) if len(df) else confusion_matrix([], [])
    return {"matrix": m, "scores": scores(m), "rows": df}


def format_matrix(m: pd.DataFrame) -> str:
    """混同行列を、種類の日本語の名前で表にした文字列（unknown の列は 1 本もなければ省く）。"""
    from smash_speed.shots import SHOT_TYPE_LABELS

    if not m["unknown"].any():
        m = m.drop(columns="unknown")
    return m.rename(index=SHOT_TYPE_LABELS, columns=SHOT_TYPE_LABELS).to_string()


def main(argv: list[str] | None = None) -> None:
    import argparse
    import time
    from pathlib import Path

    from smash_speed.shots import SHOT_TYPE_LABELS, read_shots, write_shots

    ap = argparse.ArgumentParser(description="ショットの種類をルールで判定する")
    ap.add_argument("shots_csv", nargs="?", help="種類を決める shots.csv（上書きせず、*_typed.csv に書き出す）")
    ap.add_argument("--bench", action="store_true", help="合成ラリーで混同行列を出す")
    ap.add_argument("--rallies", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=float, nargs="+", default=[30.0, 120.0])
    ap.add_argument("--hits", default="audio", choices=["true", "audio", "track"])
    ap.add_argument("--truth", action="store_true", help="当てはめをせず、正解の軌道から判定する")
    a = ap.parse_args(argv)
    if a.shots_csv:
        src = Path(a.shots_csv)
        shots = classify_shots(read_shots(src), overwrite=True)
        dst = write_shots(src.with_name(src.stem + "_typed.csv"), shots)
        counts = pd.Series([s.shot_type for s in shots]).value_counts()
        print(f"{dst}：" + "、".join(f"{SHOT_TYPE_LABELS[k]} {v} 本" for k, v in counts.items()))
        return
    if not a.bench:
        ap.error("shots.csv か --bench を指定してください")
    if a.truth:
        runs = [("正解の軌道", lambda: truth_benchmark(a.rallies, a.seed))]
    else:
        runs = [(f"{fps:.0f} fps 打球の時刻 {a.hits}",
                 lambda fps=fps: synthetic_benchmark(a.rallies, fps, a.seed, a.hits)) for fps in a.fps]
    for name, run in runs:
        t0 = time.time()
        r = run()
        sc = r["scores"]
        print(f"{name}：正解率 {sc['accuracy']:.3f}（{sc['n']} 本、{time.time() - t0:.0f} 秒）")
        print(format_matrix(r["matrix"]))
        print("多い間違い：" + "、".join(f"{SHOT_TYPE_LABELS[t]} → {SHOT_TYPE_LABELS[p]} {n} 本"
                                    for t, p, n in mistakes(r["matrix"])))


if __name__ == "__main__":
    import sys

    if (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    main()
