"""ショットの種類の判定（学習）。正解ラベルから軽い分類器を学習し、ルールの判定と比べる（ROADMAP B7）。

正解ラベル（``shot_labels.csv``、B6）の 1 行には、付けたときのショットの打点・打点での速度・終端速度が残っているので、
ルールと同じ手がかり（``shot_types.ShotFeatures``）をそこから出し直して特徴量にする。

分類器は、追加の依存を入れずに numpy だけで書いた勾配ブースティング（多クラスのソフトマックス、
2 次の近似で深さの浅い木を足していく。特徴量は分位点で 32 区画ほどに分けてから分ける所を探す）。
数百本のラベルなら 1 秒ほどで学習できる。

ルールと比べるときは、ラリー（動画・ラリーの番号）をまとめて k 個の組に分け、1 組ずつ外して学習・判定する
（同じラリーのショットが学習と判定の両方に入らないように）。ルールは同じショットの手がかりにそのまま当てる。

実際のラベルはまだ少ないので、``synthetic_labels`` で合成ラリーのショットに「実動画のずれ」（打点が低く・初速が遅く
出るなど）と雑音を入れたラベルを作り、学習がルールのずれを取り戻せるかを確かめる（``--bench``）。

「打球ではない」のラベル（``shot_labels.NOT_A_HIT``）は種類ではないので分類器には入れず、``hit_check`` で
ラベルを付けたショットのうち何本が打球ではなかったか（打った瞬間の見つけ間違いの割合）を数える。

``python -m smash_speed.shot_learn <shot_labels.csv>`` でルールと比べ、``--out`` に学習したモデル（JSON）を書く。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from smash_speed.physics import PhysicsParams
from smash_speed.shot_labels import NOT_A_HIT, ShotLabel, read_labels
from smash_speed.shot_types import (CLASSES, ShotFeatures, classify_features, confusion_matrix, features_from_trajectory,
                                    scores, shot_features)
from smash_speed.shots import Shot

MIN_LABELS = 200          # ルールと比べて意味のある数字になるラベルの本数の目安（ROADMAP B7）
FEATURE_NAMES = ("speed_mps", "elevation_deg", "hit_z_m", "hit_depth_m", "apex_z_m", "net_z_m", "land_depth_m",
                 "first_in_rally")
NO_NET_Z = -1.0           # ネットを越えないショットの「ネットの上の高さ」（nan の代わり）


# =============================================================================
# 特徴量
# =============================================================================
def label_shot(r: ShotLabel) -> Shot:
    """ラベルに残った値から ``Shot`` を作り直す（種類は決まりの判定）。"""
    return Shot(t_hit_s=r.t_hit_s, side=r.side, shot_type=r.predicted or "unknown", speed_kmh=r.speed_kmh,
                hit_x_m=r.hit_x_m, hit_y_m=r.hit_y_m, hit_z_m=r.hit_z_m, land_x_m=r.land_x_m, land_y_m=r.land_y_m,
                t_land_s=r.t_land_s, v0_x_mps=r.v0_x_mps, v0_y_mps=r.v0_y_mps, v0_z_mps=r.v0_z_mps,
                terminal_velocity_mps=r.terminal_velocity_mps, confidence=r.confidence, rally_id=r.rally_id,
                index_in_rally=r.index_in_rally)


def feature_vector(f: ShotFeatures) -> np.ndarray:
    """手がかりを数の並び（``FEATURE_NAMES`` の順）にする。分からない着地点はルールと同じく 0。"""
    d = asdict(f)
    v = np.array([float(d[k]) for k in FEATURE_NAMES], float)
    v[FEATURE_NAMES.index("net_z_m")] = d["net_z_m"] if np.isfinite(d["net_z_m"]) else NO_NET_Z
    v[FEATURE_NAMES.index("land_depth_m")] = d["land_depth_m"] if np.isfinite(d["land_depth_m"]) else 0.0
    return v


def label_table(labels: Iterable[ShotLabel], phys: PhysicsParams | None = None) -> pd.DataFrame:
    """ラベルを 1 本 1 行の表にする：正解・ルールの判定・ラリーの組・手がかり。

    打点・速度のないラベル（当てはめられなかったショット）と、種類が ``CLASSES`` にないラベル
    （「打球ではない」・unknown）は入れない。
    """
    phys = phys or PhysicsParams()
    rows = []
    for r in labels:
        if r.label not in CLASSES:
            continue
        f = shot_features(label_shot(r), phys)
        if f is None:
            continue
        rows.append({"video_name": r.video_name, "t_hit_s": r.t_hit_s, "group": f"{r.video_name}#{r.rally_id}",
                     "true": r.label, "rule": classify_features(f),
                     **dict(zip(FEATURE_NAMES, feature_vector(f)))})
    return pd.DataFrame(rows, columns=["video_name", "t_hit_s", "group", "true", "rule", *FEATURE_NAMES])


def hit_check(labels: Iterable[ShotLabel]) -> dict[str, Any]:
    """ラベルを付けたショットのうち「打球ではない」の本数と割合（打った瞬間の見つけ間違いの割合の目安）。

    人がラベルを付けたショットは一覧に出たショットなので、「打球ではない」の割合は、見つけた打球のうち
    間違いだったものの割合（誤検出の割合）になる。見落とした打球は一覧に出ないので、ここでは数えられない。

    Returns:
        ``n``（ラベルの本数）・``n_not_hit``・``rate``（ラベルがなければ nan）・``by_video``（動画ごとの同じもの）
    """
    by_video: dict[str, dict[str, Any]] = {}
    for r in labels:
        v = by_video.setdefault(r.video_name, {"n": 0, "n_not_hit": 0})
        v["n"] += 1
        v["n_not_hit"] += r.label == NOT_A_HIT
    for v in by_video.values():
        v["rate"] = v["n_not_hit"] / v["n"]
    n = sum(v["n"] for v in by_video.values())
    n_not = sum(v["n_not_hit"] for v in by_video.values())
    return {"n": n, "n_not_hit": n_not, "rate": (n_not / n) if n else float("nan"), "by_video": by_video}


# =============================================================================
# 勾配ブースティング
# =============================================================================
@dataclass(frozen=True)
class BoostParams:
    """学習の設定。ラベルが数百本なので、浅い木・少ない回数・葉の最小本数で過学習を抑える。"""

    n_rounds: int = 80
    learning_rate: float = 0.15
    max_depth: int = 3
    min_leaf: int = 5         # 葉に入るショットの最少本数
    reg_lambda: float = 1.0   # 葉の値の L2 正則化
    n_bins: int = 32          # 特徴量ごとに分ける所の候補の数


def _leaf(g: np.ndarray, h: np.ndarray, lam: float) -> float:
    return float(-g.sum() / (h.sum() + lam))


def _grow(X: np.ndarray, B: np.ndarray, edges: list[np.ndarray], idx: np.ndarray, g: np.ndarray, h: np.ndarray,
          depth: int, p: BoostParams) -> list:
    """回帰木を 1 本作る。葉は ``[値]``、分かれ目は ``[特徴量, しきい値, 左, 右]``（x <= しきい値 が左）。"""
    G, H = g[idx].sum(), h[idx].sum()
    if depth >= p.max_depth or len(idx) < 2 * p.min_leaf:
        return [_leaf(g[idx], h[idx], p.reg_lambda)]
    base = G * G / (H + p.reg_lambda)
    best = (1e-9, -1, -1)
    for j in range(X.shape[1]):
        nb = len(edges[j]) + 1
        if nb < 2:
            continue
        b = B[idx, j]
        gl = np.cumsum(np.bincount(b, weights=g[idx], minlength=nb))[:-1]
        hl = np.cumsum(np.bincount(b, weights=h[idx], minlength=nb))[:-1]
        cl = np.cumsum(np.bincount(b, minlength=nb))[:-1]
        ok = (cl >= p.min_leaf) & (len(idx) - cl >= p.min_leaf)
        if not ok.any():
            continue
        gain = gl ** 2 / (hl + p.reg_lambda) + (G - gl) ** 2 / (H - hl + p.reg_lambda) - base
        gain[~ok] = -np.inf
        k = int(np.argmax(gain))
        if gain[k] > best[0]:
            best = (float(gain[k]), j, k)
    _, j, k = best
    if j < 0:
        return [_leaf(g[idx], h[idx], p.reg_lambda)]
    left = B[idx, j] <= k
    return [j, float(edges[j][k]), _grow(X, B, edges, idx[left], g, h, depth + 1, p),
            _grow(X, B, edges, idx[~left], g, h, depth + 1, p)]


def _tree_predict(tree: list, X: np.ndarray) -> np.ndarray:
    if len(tree) == 1:
        return np.full(len(X), tree[0])
    j, thr, left, right = tree
    out = np.empty(len(X))
    m = X[:, j] <= thr
    out[m] = _tree_predict(left, X[m])
    out[~m] = _tree_predict(right, X[~m])
    return out


def _softmax(F: np.ndarray) -> np.ndarray:
    e = np.exp(F - F.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


@dataclass
class ShotTypeModel:
    """学習した分類器。``classes`` は学習に出てきた種類だけ（ほかの種類には判定しない）。"""

    classes: list[str]
    prior: list[float]                         # 最初の値（種類ごとの割合の対数）
    trees: list[list[list]] = field(default_factory=list)  # 回ごとに、種類ごとの木
    learning_rate: float = 0.15
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    n_train: int = 0

    def scores(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, float))
        F = np.tile(np.asarray(self.prior, float), (len(X), 1))
        for round_trees in self.trees:
            for k, tree in enumerate(round_trees):
                F[:, k] += self.learning_rate * _tree_predict(tree, X)
        return F

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _softmax(self.scores(X))

    def predict(self, X: np.ndarray) -> list[str]:
        return [self.classes[i] for i in np.argmax(self.scores(X), axis=1)]

    def classify(self, f: ShotFeatures) -> str:
        return self.predict(feature_vector(f)[None])[0]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "ShotTypeModel":
        if list(d.get("feature_names", FEATURE_NAMES)) != list(FEATURE_NAMES):
            raise ValueError("モデルの特徴量が今のコードと違います。学習し直してください")
        return cls(**d)


def fit_model(X: np.ndarray, y: Iterable[str], params: BoostParams | None = None) -> ShotTypeModel:
    """手がかりの表 ``X``（``FEATURE_NAMES`` の順）と正解 ``y`` から学習する。"""
    p = params or BoostParams()
    X = np.asarray(X, float)
    y = list(y)
    classes = [c for c in CLASSES if c in set(y)]
    if not classes:
        raise ValueError("学習に使えるラベルがありません")
    Y = np.array([[t == c for c in classes] for t in y], float)
    prior = np.log(Y.mean(axis=0))
    model = ShotTypeModel(classes, prior.tolist(), [], p.learning_rate, n_train=len(y))
    if len(classes) == 1:
        return model
    # 分ける所の候補：特徴量ごとの分位点（同じ値はまとめる）
    qs = np.linspace(0, 1, p.n_bins + 1)[1:-1]
    edges = [np.unique(np.quantile(X[:, j], qs)) for j in range(X.shape[1])]
    B = np.stack([np.searchsorted(edges[j], X[:, j], side="left") for j in range(X.shape[1])], axis=1)
    F = np.tile(prior, (len(X), 1))
    idx = np.arange(len(X))
    for _ in range(p.n_rounds):
        P = _softmax(F)
        round_trees = []
        for k in range(len(classes)):
            g = P[:, k] - Y[:, k]
            h = np.maximum(P[:, k] * (1 - P[:, k]), 1e-6)
            tree = _grow(X, B, edges, idx, g, h, 0, p)
            round_trees.append(tree)
            F[:, k] += p.learning_rate * _tree_predict(tree, X)
        model.trees.append(round_trees)
    return model


def save_model(path: str | Path, model: ShotTypeModel) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.to_json(), ensure_ascii=False), encoding="utf-8")
    return path


def load_model(path: str | Path) -> ShotTypeModel:
    return ShotTypeModel.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def classify_shots_model(shots: Iterable[Shot], model: ShotTypeModel, phys: PhysicsParams | None = None) -> list[str]:
    """学習したモデルでの判定（打点・速度のないショットは unknown）。``Shot`` は変えない。"""
    out = []
    for s in shots:
        f = shot_features(s, phys)
        out.append("unknown" if f is None else model.classify(f))
    return out


# =============================================================================
# ルールとの比べ方
# =============================================================================
def compare_with_rules(table: pd.DataFrame, k: int = 5, seed: int = 0,
                       params: BoostParams | None = None) -> dict[str, Any]:
    """ラリーの組ごとに k 個に分けて 1 組ずつ外して学習し、外した組での判定をルールと比べる。

    Returns:
        ``n``、``rule``・``model``（それぞれ ``matrix``・``scores``）、``rows``（``table`` に ``model`` の列を足したもの）
    """
    if len(table) == 0:
        raise ValueError("比べるラベルがありません")
    groups = table["group"].to_numpy()
    uniq = np.unique(groups)
    k = max(2, min(k, len(uniq)))
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(rng.permutation(uniq), np.arange(len(uniq)) % k))
    folds = np.array([fold_of[g] for g in groups])
    X = table[list(FEATURE_NAMES)].to_numpy(float)
    y = table["true"].to_numpy()
    pred = np.empty(len(table), dtype=object)
    for i in range(k):
        test = folds == i
        if not test.any():
            continue
        if (~test).sum() == 0:
            pred[test] = table["rule"].to_numpy()[test]
            continue
        model = fit_model(X[~test], y[~test], params)
        pred[test] = model.predict(X[test])
    rows = table.assign(model=pred)
    out: dict[str, Any] = {"n": len(rows), "k": k, "rows": rows}
    for name in ("rule", "model"):
        m = confusion_matrix(rows["true"], rows[name])
        out[name] = {"matrix": m, "scores": scores(m)}
    return out


# =============================================================================
# 合成ラベル（学習がルールのずれを取り戻せるかを確かめる）
# =============================================================================
@dataclass(frozen=True)
class RealShift:
    """合成の正解の軌道に入れる「実動画のずれ」。カメラの高さ・キャリブレーションのずれで打点が低く出る、
    当てはめで初速が遅く出る、などのいつも同じ向きのずれ（ルールのしきい値を外れる）と、ショットごとの雑音。"""

    hit_z_shift_m: float = -0.35      # 打点の高さのずれ
    speed_scale: float = 0.85         # 初速の倍率
    elevation_shift_deg: float = 0.0  # 打ち出し角のずれ
    hit_xy_sigma_m: float = 0.25      # 打点の位置の雑音
    speed_sigma: float = 0.06         # 初速の雑音（割合）
    elevation_sigma_deg: float = 3.0  # 打ち出し角の雑音


def _velocity_like(v0: np.ndarray, speed: float, elev_deg: float) -> np.ndarray:
    """水平の向きは ``v0`` のまま、速さと打ち出し角を置き換えた速度。"""
    h = np.asarray(v0[:2], float)
    h = h / (np.linalg.norm(h) or 1.0)
    e = np.radians(elev_deg)
    return speed * np.array([h[0] * np.cos(e), h[1] * np.cos(e), np.sin(e)])


def synthetic_labels(n_rallies: int = 60, seed: int = 0, shift: RealShift | None = None,
                     phys: PhysicsParams | None = None, video_name: str = "synthetic.mp4",
                     not_hit_share: float = 0.0) -> list[ShotLabel]:
    """合成ラリーの正解の種類に、ずれと雑音を入れた軌道の値を付けたラベル（``shot_labels.csv`` と同じ形）。

    ``not_hit_share`` > 0 なら、打った瞬間の見つけ間違いをまねて、ショットの飛んでいる途中（打ってから 0.3 秒後）に
    「打球ではない」のラベルを足す（ラベル全体のうちおよそこの割合。値はそのショットの途中の位置・速度）。
    """
    from smash_speed.synth_rally import make_rally

    phys = phys or PhysicsParams()
    shift = shift or RealShift()
    rng = np.random.default_rng([seed, 7])
    rng_not = np.random.default_rng([seed, 11])   # 打球ではないを足すか（ほかのショットの値は変えない）
    out = []
    for r in range(n_rallies):
        rally = make_rally(rng, phys, n_shots=int(rng.integers(4, 13)), server_side="h" if r % 2 == 0 else "o",
                           t_start=100.0 * r)
        for i, s in enumerate(rally.segments):
            p0 = s.p0 + np.array([*rng.normal(0, shift.hit_xy_sigma_m, 2), shift.hit_z_shift_m])
            p0[2] = max(p0[2], 0.1)
            speed = s.speed0 * shift.speed_scale * (1 + rng.normal(0, shift.speed_sigma))
            elev = s.elevation_deg + shift.elevation_shift_deg + rng.normal(0, shift.elevation_sigma_deg)
            v0 = _velocity_like(s.v0, max(speed, 0.5), elev)
            f = features_from_trajectory(p0, v0, phys, i == 0)
            out.append(ShotLabel(video_name=video_name, t_hit_s=round(float(s.t_hit), 4), label=s.shot_type,
                                 predicted=classify_features(f), side=s.side, speed_kmh=float(np.linalg.norm(v0) * 3.6),
                                 hit_x_m=float(p0[0]), hit_y_m=float(p0[1]), hit_z_m=float(p0[2]),
                                 v0_x_mps=float(v0[0]), v0_y_mps=float(v0[1]), v0_z_mps=float(v0[2]),
                                 terminal_velocity_mps=phys.terminal_velocity, rally_id=r, index_in_rally=i + 1))
            if not_hit_share > 0 and rng_not.uniform() < not_hit_share / (1 - not_hit_share):
                # 飛んでいる途中の 1 点を「打った」と間違えたもの：位置・速度は 0.3 秒後の軌道の値
                dt = 0.3
                p1 = p0 + v0 * dt + 0.5 * np.array([0.0, 0.0, -9.81]) * dt ** 2
                v1 = v0 * np.exp(-9.81 * dt / phys.terminal_velocity) + np.array([0.0, 0.0, -9.81 * dt])
                p1[2] = max(p1[2], 0.1)
                f1 = features_from_trajectory(p1, v1, phys, False)
                out.append(ShotLabel(video_name=video_name, t_hit_s=round(float(s.t_hit) + dt, 4), label=NOT_A_HIT,
                                     predicted=classify_features(f1), side=s.side,
                                     speed_kmh=float(np.linalg.norm(v1) * 3.6), hit_x_m=float(p1[0]),
                                     hit_y_m=float(p1[1]), hit_z_m=float(p1[2]), v0_x_mps=float(v1[0]),
                                     v0_y_mps=float(v1[1]), v0_z_mps=float(v1[2]),
                                     terminal_velocity_mps=phys.terminal_velocity, rally_id=r, index_in_rally=None))
    return out


def benchmark(n_rallies: int = 60, seed: int = 0, shift: RealShift | None = None, k: int = 5) -> dict[str, Any]:
    """合成ラベルで、ルールと学習を比べる（``compare_with_rules`` の結果）。"""
    return compare_with_rules(label_table(synthetic_labels(n_rallies, seed, shift)), k=k, seed=seed)


def main(argv: list[str] | None = None) -> None:
    import argparse
    import time

    from smash_speed.shot_types import format_matrix, mistakes
    from smash_speed.shots import SHOT_TYPE_LABELS

    ap = argparse.ArgumentParser(description="ショットの種類の判定を正解ラベルから学習し、ルールと比べる")
    ap.add_argument("labels_csv", nargs="?", help="正解ラベル（出力の場所の shot_labels.csv）")
    ap.add_argument("--out", help="すべてのラベルで学習したモデルを書く場所（JSON）")
    ap.add_argument("--bench", action="store_true", help="合成ラベル（ずれと雑音入り）で比べる")
    ap.add_argument("--rallies", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-k", type=int, default=5, help="比べるときに分ける組の数")
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.bench:
        table = label_table(synthetic_labels(a.rallies, a.seed))
        print(f"合成ラベル（{a.rallies} ラリー、打点 {RealShift().hit_z_shift_m:+.2f} m・初速 ×{RealShift().speed_scale}）")
    elif a.labels_csv:
        labels = read_labels(a.labels_csv)
        table = label_table(labels)
        print(f"{a.labels_csv}：ラベル {len(labels)} 本（使えるもの {len(table)} 本、動画 {table['video_name'].nunique()} 本）")
        hc = hit_check(labels)
        if hc["n"]:
            print(f"打球ではない：{hc['n_not_hit']} 本（ラベルを付けたショットの {100 * hc['rate']:.1f}%。"
                  "打った瞬間の見つけ間違いの割合の目安。種類の学習には使いません）")
        if len(table) < MIN_LABELS:
            print(f"※ {MIN_LABELS} 本より少ないので、数字は目安です")
    else:
        ap.error("shot_labels.csv か --bench を指定してください")
    if len(table) < 10 or table["true"].nunique() < 2:
        print("ラベルが少なすぎて比べられません（2 種類以上・10 本以上いります）")
        return
    r = compare_with_rules(table, a.k, a.seed)
    for name, title in (("rule", "ルール"), ("model", "学習")):
        m = r[name]["matrix"]
        print(f"\n{title}：正解率 {r[name]['scores']['accuracy']:.3f}（{r['n']} 本、{r['k']} 組で 1 組ずつ外して判定）")
        print(format_matrix(m))
        print("多い間違い：" + ("、".join(f"{SHOT_TYPE_LABELS[t]} → {SHOT_TYPE_LABELS[p]} {n} 本"
                                     for t, p, n in mistakes(m)) or "なし"))
    if a.out:
        model = fit_model(table[list(FEATURE_NAMES)].to_numpy(float), table["true"])
        print(f"\nモデル：{save_model(a.out, model)}（{len(table)} 本で学習）")
    print(f"（{time.time() - t0:.1f} 秒）")


if __name__ == "__main__":
    import sys

    if (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    main()
