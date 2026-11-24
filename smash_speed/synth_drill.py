"""基礎打ち（決まった練習を続けて打ち合う）の合成データ（ROADMAP G1）。

練習の動画の多くは 1 対 1 の基礎打ち：ドライブ・プッシュとレシーブ・スマッシュとレシーブ・クリアなどを、ネット・アウト・
取れなかったで途切れるまで続け、拾って球を出し直す（3〜10 秒）。試合のラリーと違い、同じ種類のショットが交互に続く。

1. 場面（``make_drill_scene``）：練習の種類（``DRILLS``）ごとのショットを交互に打ち合う「続いた打ち合い」（``make_streak``。
   ``synth_rally.SyntheticRally`` と同じ形）を、とぎれの間をはさんで並べる。打つ間隔は、シャトルが打てる高さまで下りてきた
   間のどこで打つかで決める（ばらつき・時間とともに遅れる疲れ）。
2. 観測（``observe_drill``）：``synth_clutter.observe_cluttered`` と同じ形の検出・打球音の候補と、``players.csv`` の形の人の枠を、
   場面から直接作る（動画は描かない）。画面の外の点は検出に出ない。人の枠は画面の端で切る（見える部分が 3 割未満なら出さない）。
3. 正解：打球の時刻・打った側（``DrillTruth.hits``）、続いた打ち合いの区切り（``DrillTruth.streaks``）。

ネットの近くの横から斜めに撮るカメラは ``synth_rally.oblique_rally_camera``（名前は "oblique"）。

``python -m smash_speed.synth_drill --bench`` で、練習の種類ごとの打つ間隔を出す。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from smash_speed.config import load_config
from smash_speed.physics import PhysicsParams, simulate_until_ground
from smash_speed.synth_clutter import ClutterOptions, ClutterScene, observe_cluttered
from smash_speed.synth_players import observe_players
from smash_speed.synth_rally import (HALF_LENGTH_M, SHOT_PROFILES, FlightSegment, SyntheticRally, _segment, aim_shot,
                                     fault_shot, make_typed_shot, named_rally_camera, sample_target)


@dataclass(frozen=True)
class DrillSpec:
    """練習の種類。``pattern`` は 1 本目から順に打つショット（``SHOT_PROFILES`` のキー）で、足りなければ繰り返す。"""

    name: str
    pattern: tuple[str, ...]
    label: str
    slack_m: float = 0.0   # 打てる高さの下の端を下げる量 [m]（低い所まで引きつけて打つ）
    q_mean: float = 0.3    # 打てる間の、どのあたりで打つか（0 は下りてきてすぐ、1 は打てる間の終わり）
    slack_only: tuple[str, ...] = ()  # slack_m を使うショット（空ならすべて。スマッシュは低くすると打てなくなるので外す）


DRILLS: dict[str, DrillSpec] = {
    "drive": DrillSpec("drive", ("drive", "drive"), "ドライブ", slack_m=0.5, q_mean=0.35),
    "push_receive": DrillSpec("push_receive", ("push", "push"), "プッシュとレシーブ", slack_m=0.6, q_mean=0.6),
    "smash": DrillSpec("smash", ("smash", "lob"), "スマッシュ", slack_m=0.8, q_mean=1.0, slack_only=("lob",)),
    "smash_receive": DrillSpec("smash_receive", ("smash", "lob"), "スマッシュレシーブ", slack_m=0.8, q_mean=1.0, slack_only=("lob",)),
    "clear": DrillSpec("clear", ("clear", "clear"), "クリアの打ち合い"),
}
# 完了の条件で 3 回ずつ作る 4 種類
MAIN_DRILLS = ("drive", "push_receive", "smash_receive", "clear")


@dataclass
class DrillOptions:
    """基礎打ちの場面の作り方。時間は秒。"""

    n_streaks: int = 6                          # 続いた打ち合いの数
    streak_hits: tuple[int, int] = (3, 40)      # 続いた打ち合いの本数の範囲（つなげなくなればそれより短い）
    pause_s: tuple[float, float] = (3.0, 10.0)  # とぎれた後、拾って球を出し直すまで
    ends: dict[str, float] = field(default_factory=lambda: {"net": 0.35, "out": 0.3, "in": 0.35})
    jitter: float = 0.15                        # 打つ時刻を窓の中のどこにするか（0〜1）のばらつき
    fatigue: float = 0.0                        # 疲れ：最後の打ち合いの ``tired``（``make_streak``。0 でなし、1 でかなり遅く打つ）
    clutter: ClutterOptions = field(default_factory=lambda: ClutterOptions(neighbor=False))


# =============================================================================
# 場面
# =============================================================================
@dataclass
class DrillScene(ClutterScene):
    """基礎打ちの場面。``rallies`` が続いた打ち合い（1 つ 1 つが 1 本のつながり）。"""

    kind: str = "drive"

    def streaks(self) -> list[tuple[float, float, int, dict[str, int]]]:
        """続いた打ち合いの（はじめの打球・最後の打球の時刻、本数、打った側ごとの本数）。"""
        out = []
        for r in self.rallies:
            sides = [s.side for s in r.segments]
            out.append((r.segments[0].t_hit, r.segments[-1].t_hit, len(sides),
                        {"h": sides.count("h"), "o": sides.count("o")}))
        return out


def _window(seg: FlightSegment, hit_z_m: tuple[float, float], phys: PhysicsParams,
            min_flight_s: float = 0.25) -> list[tuple[float, np.ndarray]]:
    """ショット ``seg`` を相手が打てる（相手側のコートの上で下りてきて、高さが範囲にある）時刻と位置の並び。"""
    tr = simulate_until_ground(seg.p0, seg.v0, seg.t_hit, phys, t_max=8.0, n_dense=2000)
    t, pos, vel = tr["t"], tr["pos"], tr["vel"]
    ok = ((np.sign(pos[:, 0]) == -np.sign(seg.p0[0])) & (np.abs(pos[:, 0]) <= HALF_LENGTH_M)
          & (np.abs(pos[:, 1]) <= 3.0) & (vel[:, 2] < 0) & (t - seg.t_hit >= min_flight_s)
          & (pos[:, 2] >= hit_z_m[0]) & (pos[:, 2] <= hit_z_m[1]))
    return [(float(t[i]), pos[i].copy()) for i in np.nonzero(ok)[0]]


def make_streak(rng: np.random.Generator, phys: PhysicsParams, kind: str, n_hits: int, t_start: float = 0.0,
                end: str = "in", jitter: float = 0.15, tired: float = 0.0, max_tries: int = 12) -> SyntheticRally:
    """練習 ``kind`` の続いた打ち合い（最大 ``n_hits`` 本）を作る。1 本目は h の側が打つ。

    次の打球は、前のショットが打てる高さまで下りてきた間（``_window``）の、先頭から ``jitter`` 程度のばらつきの所で打つ。
    ``tired``（0〜）が大きいほど、打てる高さの下の端を ``0.5 * tired`` m 下げ、打てる間の遅い方で打つ（疲れて遅く打つ）。つなげなくなれば、そこで床に落ちて終わる。
    ``end`` は最後のショットの落ち方（"in" は取れなかった、"out"・"net" は ``synth_rally.fault_shot``）。
    """
    spec = DRILLS[kind]
    pattern = spec.pattern
    for _ in range(max_tries):  # つなげずに 1〜2 本で終わったときは、作り直す
        streak = _build_streak(rng, phys, spec, n_hits, t_start, jitter, tired, max_tries)
        if len(streak.segments) >= min(n_hits, 3):
            break
    segs = streak.segments
    if end != "in":
        last = segs[-1]
        v0 = fault_shot(end, last.profile, last.p0, rng, phys)
        if v0 is None:
            end = "in"
        else:
            segs[-1] = _segment(last.profile, last.p0, v0, last.t_hit, phys)
    return SyntheticRally(segs, end)


def _build_streak(rng: np.random.Generator, phys: PhysicsParams, spec: DrillSpec, n_hits: int, t_start: float,
                  jitter: float, tired: float, max_tries: int) -> SyntheticRally:
    pattern = spec.pattern
    segs = [make_typed_shot(pattern[0], rng, phys, side="h", t_hit=t_start)]
    while len(segs) < n_hits:
        prev = segs[-1]
        profile = pattern[len(segs) % len(pattern)]
        pr = SHOT_PROFILES[profile]
        use = not spec.slack_only or profile in spec.slack_only
        z_lo = max(0.15, pr.hit_z_m[0] - (spec.slack_m if use else 0.0) - 0.5 * tired)
        nxt = None
        win = _window(prev, (z_lo, pr.hit_z_m[1]), phys)
        for _ in range(max_tries if win else 0):
            q = float(np.clip(rng.normal(spec.q_mean + 0.3 * tired, jitter), 0.0, 1.0))
            t_hit, p = win[int(round(q * (len(win) - 1)))]
            lo, hi = pr.hit_depth_m
            if not lo - 0.5 <= abs(p[0]) <= hi + 0.5:
                continue
            v0 = aim_shot(profile, p, sample_target(profile, np.sign(p[0]), rng), rng, phys)
            if v0 is not None:
                nxt = (t_hit, p, profile, v0)
                break
        if nxt is None:
            break
        t_hit, p, profile, v0 = nxt
        prev.t_end, prev.intercepted = t_hit, True
        segs.append(_segment(profile, p, v0, t_hit, phys))
    return SyntheticRally(segs)


def make_drill_scene(rng: np.random.Generator, phys: PhysicsParams, kind: str = "drive",
                     opts: DrillOptions | None = None, t_start: float = 0.5) -> DrillScene:
    """練習 ``kind`` の場面：続いた打ち合いを、拾って球を出し直す間をはさんで ``opts.n_streaks`` 個並べる。"""
    o = opts or DrillOptions()
    if kind not in DRILLS:
        raise ValueError(f"知らない練習の種類です: {kind}（{', '.join(DRILLS)} のどれか）")
    ends = list(o.ends)
    p_end = np.array([o.ends[e] for e in ends], float)
    rallies: list[SyntheticRally] = []
    t = t_start
    for k in range(o.n_streaks):
        n = int(rng.integers(o.streak_hits[0], o.streak_hits[1] + 1))
        end = str(rng.choice(ends, p=p_end / p_end.sum()))
        r = make_streak(rng, phys, kind, n, t, end, o.jitter, o.fatigue * k / max(o.n_streaks - 1, 1))
        rallies.append(r)
        t = r.t_end + float(rng.uniform(*o.pause_s))
    return DrillScene(rallies, [], 7.5, phys, kind)


# =============================================================================
# 観測と正解
# =============================================================================
@dataclass
class DrillTruth:
    """基礎打ちの正解。``hits`` は列 ``t, side``（打球の時刻 [s]・打った側）、``streaks`` は列 ``t_start, t_end, n_hits, n_h, n_o, end``。"""

    kind: str
    hits: pd.DataFrame
    streaks: pd.DataFrame

    def tempo_s(self) -> float:
        """打球の間隔の平均 [s]。スマッシュのように短い間隔（打った後）と長い間隔（レシーブの後）が交互になる練習は、
        中央値がどちらかの山に寄るので、練習の速さの目安にはこちらを使う。"""
        iv = self.intervals()
        return float(iv.mean()) if len(iv) else float("nan")

    def intervals(self) -> np.ndarray:
        """続いた打ち合いの中の、打球の間隔 [s]（とぎれをまたがない）。"""
        out = []
        for _, s in self.streaks.iterrows():
            t = self.hits.t[(self.hits.t >= s.t_start) & (self.hits.t <= s.t_end)].to_numpy()
            out.append(np.diff(t))
        return np.concatenate(out) if out else np.array([])


@dataclass
class DrillObservation:
    """``observe_drill`` の結果。検出・打球音は ``synth_clutter.observe_cluttered`` と、人の枠は ``players.csv`` と同じ形。"""

    det: pd.DataFrame
    onsets: list[tuple[float, float]]
    players: pd.DataFrame
    frames: pd.DataFrame
    truth: DrillTruth


def drill_truth(scene: DrillScene) -> DrillTruth:
    hits = pd.DataFrame([(t, side) for t, _, side, _ in scene.hits()], columns=["t", "side"])
    rows = [(a, b, n, c["h"], c["o"], r.end) for (a, b, n, c), r in zip(scene.streaks(), scene.rallies)]
    return DrillTruth(scene.kind, hits, pd.DataFrame(rows, columns=["t_start", "t_end", "n_hits", "n_h", "n_o", "end"]))


def clip_boxes(players: pd.DataFrame, width: int, height: int, min_visible: float = 0.3) -> pd.DataFrame:
    """人の枠を画面で切る。見える部分が枠の ``min_visible`` 未満の行は、検出されなかったことにして外す。"""
    p = players.copy()
    area = (p.x1 - p.x0) * (p.y1 - p.y0)
    p["x0"], p["x1"] = p.x0.clip(0, width), p.x1.clip(0, width)
    p["y0"], p["y1"] = p.y0.clip(0, height), p.y1.clip(0, height)
    seen = (p.x1 - p.x0) * (p.y1 - p.y0)
    return p[seen >= min_visible * area].reset_index(drop=True)


def observe_drill(scene: DrillScene, camera, fps: float, rng: np.random.Generator,
                  opts: DrillOptions | None = None) -> DrillObservation:
    """場面をカメラで観測した、検出・打球音の候補・人の枠と、正解。"""
    o = opts or DrillOptions()
    det, onsets, frames = observe_cluttered(scene, camera, fps, rng, o.clutter)
    t = frames.timestamp.to_numpy()
    feet = {side: scene.feet(side, t)[:, :2] for side in ("h", "o")}
    players = clip_boxes(observe_players(t, feet, camera, rng), camera.width, camera.height)
    return DrillObservation(det, onsets, players, frames, drill_truth(scene))


# =============================================================================
# 測り方
# =============================================================================
def interval_summary(kinds: tuple[str, ...] = MAIN_DRILLS, n_scenes: int = 3, seed: int = 0,
                     cfg: dict[str, Any] | None = None, opts: DrillOptions | None = None) -> dict[str, dict[str, Any]]:
    """練習の種類ごとに、``n_scenes`` 回作った場面の打球の間隔の中央値・本数などを出す。"""
    cfg = cfg or load_config()
    phys = PhysicsParams.from_config(cfg)
    out: dict[str, dict[str, Any]] = {}
    for kind in kinds:
        meds, means, lens, all_iv = [], [], [], []
        for i in range(n_scenes):
            rng = np.random.default_rng(seed + 1000 * i + len(kind))
            tr = drill_truth(make_drill_scene(rng, phys, kind, opts))
            iv = tr.intervals()
            all_iv.append(iv)
            meds.append(float(np.median(iv)))
            means.append(tr.tempo_s())
            lens.append(int(tr.streaks.n_hits.median()))
        out[kind] = {"median_interval_s": meds, "median_streak_hits": lens,
                     "pooled_median_s": float(np.median(np.concatenate(all_iv))),
                     "mean_interval_s": means}
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="基礎打ちの合成データの打球の間隔を種類ごとに出す")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--scenes", type=int, default=3)
    ap.add_argument("--camera", default="oblique")
    ap.add_argument("--fps", type=float, default=30.0)
    a = ap.parse_args(argv)
    cfg = load_config()
    res = interval_summary(tuple(DRILLS), a.scenes, cfg=cfg)
    for kind, r in res.items():
        print(f"{DRILLS[kind].label}：打つ間隔の中央値 {r['pooled_median_s']:.2f} 秒"
              f"（場面ごと {', '.join(f'{m:.2f}' for m in r['median_interval_s'])}）、"
              f"平均 {', '.join(f'{m:.2f}' for m in r['mean_interval_s'])}、"
              f"続いた本数の中央値 {r['median_streak_hits']}")
    if a.bench:
        phys = PhysicsParams.from_config(cfg)
        cam = named_rally_camera(a.camera, cfg)
        rng = np.random.default_rng(0)
        scene = make_drill_scene(rng, phys, "drive")
        ob = observe_drill(scene, cam, a.fps, rng)
        print(f"{a.camera} 30fps の例：検出 {int((ob.det.visible > 0).sum())} 点・打球音 {len(ob.onsets)} 件・"
              f"人の枠 {len(ob.players)} 行（全 {2 * len(ob.frames)} 行のうち）")


if __name__ == "__main__":
    main()
