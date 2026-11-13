"""ダブルスの合成ラリー（ROADMAP D4）：4 人の足の動き・誰が打ったか・陣形の正解と、人の検出の形の観測。

``synth_rally.make_session`` のラリー（シャトルの軌道と打った側）に、1 側 2 人の選手を足す（``make_doubles_scene``）。

- 誰が打つか：打点に近い方（行き先の位置で比べる）。サーブと受ける人はラリーごとに引く。
- 打った後の動き：攻めた（スマッシュ・ドロップ・プッシュ・ネット）ら前後（後ろから打てば打った人が後ろ・パートナーが前、
  前で打てば逆）、上げた（ロブ・クリア・ロングサーブ）ら横並び（打った人が自分のいた側）。ドライブ・ショートサーブはそのまま。
  ローテーションで 2 人がすれ違う・画像の上で重なる。
- 選手ではない人：コートの後ろで待つ人・隣のコートの選手・コートの横を歩く人（``extras``）。

``observe_people`` は ``players.detect_people`` と同じ形（``people.csv``。枠・スコア・枠の中の色 ``app``）の観測。

- 枠の揺れは ``synth_players.observe_players`` と同じ。打つ人の枠は打つ前後に高く・広くなる（腕とラケット）。
- 重なり：カメラから遠い人の枠が近い人の枠に隠れると、検出が抜ける・2 人で 1 つの枠にまとまる。
- 色：人ごとの色の割合に、コマごとの揺れと、重なった相手の色の混ざりを足す。``kits`` で、4 人とも違う色（distinct）・
  ペアで同じ上（same_shirt）・ペアで上下とも同じ（same）を選ぶ。
- 正解の人の名前は ``who`` の列（追跡は使わない。答え合わせ用）。

``doubles_benchmark`` が、追跡（番号の入れ替わり・選手ではない人の混ざり）・誰が打ったか・陣形を正解と比べる。
``python -m smash_speed.synth_doubles`` で表を出す。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import ndtr

from smash_speed.camera import Camera
from smash_speed.physics import PhysicsParams
from smash_speed.players import APP_DIM, add_court_positions, encode_appearance
from smash_speed.synth_players import _slow_noise, footwork_path
from smash_speed.synth_rally import SyntheticRally, make_session, observe_session
from smash_speed.synth_rally_players import PLAY_PACE_MPS, REACH_M, WALK_PACE_MPS, _follow, _sx, _Waypoint

PEOPLE = ("h0", "h1", "o0", "o1")
ATTACK = ("smash", "drop", "push", "net")
LIFT = ("lob", "clear")
NEIGHBOR_DY_M = 7.0     # 隣のコートの中心までの距離 [m]（隣の選手がときどき選手とみなす範囲に入る近さ）
KITS = ("distinct", "same_shirt", "same")


@dataclass
class DoublesScene:
    """ダブルスの合成ラリーの正解。"""

    session: list[SyntheticRally]
    hitters: list[list[str]]                       # ラリーごと・ショットごとの打った人（``PEOPLE`` の名前）
    plan: dict[str, list[_Waypoint]]
    extras: dict[str, Any] = field(default_factory=dict)   # 選手ではない人の作り方（``extra_feet`` が使う）

    def hits(self) -> list[tuple[float, str, str, np.ndarray, str]]:
        """すべての打球の（時刻・打った側・打った人・打点・ショットの種類）。時刻の順。"""
        return [(s.t_hit, s.side, who, s.p0, s.shot_type)
                for r, names in zip(self.session, self.hitters) for s, who in zip(r.segments, names)]

    def feet(self, times: np.ndarray) -> dict[str, np.ndarray]:
        """各時刻の 4 人の足元の正解（人 → (n, 2) [m]）。"""
        t = np.asarray(times, float)
        return {name: _follow(t, self.plan[name][0].xy.copy(), self.plan[name]) for name in PEOPLE}


def _formation_targets(side: str, feet: np.ndarray, seg: Any, partner_y: float, rng: np.random.Generator
                       ) -> tuple[np.ndarray, np.ndarray] | None:
    """打った後の、打った人とパートナーの行き先。陣形を変えないショットなら None。"""
    sx = _sx(side)
    if seg.shot_type in ATTACK:
        if abs(feet[0]) >= 3.0:    # 後ろから攻めた：打った人が後ろ、パートナーが前
            return (np.array([sx * rng.uniform(4.3, 5.2), float(np.clip(0.6 * feet[1], -1.5, 1.5))]),
                    np.array([sx * rng.uniform(1.7, 2.3), float(np.clip(0.3 * feet[1] + rng.normal(0, 0.3), -1.2, 1.2))]))
        return (np.array([sx * rng.uniform(1.6, 2.2), float(np.clip(0.5 * feet[1], -1.5, 1.5))]),
                np.array([sx * rng.uniform(4.4, 5.0), float(rng.normal(0, 0.4))]))
    if seg.shot_type in LIFT or seg.profile == "serve_long":
        ya = float(np.sign(feet[1])) if abs(feet[1]) >= 0.3 else (-float(np.sign(partner_y)) or 1.0)
        return (np.array([sx * rng.uniform(3.4, 4.0), ya * rng.uniform(1.2, 1.7)]),
                np.array([sx * rng.uniform(3.4, 4.0), -ya * rng.uniform(1.2, 1.7)]))
    return None


def make_doubles_scene(rng: np.random.Generator, phys: PhysicsParams, n_rallies: int = 5,
                       rest_s: tuple[float, float] = (4.0, 8.0), n_shots: tuple[int, int] = (4, 14),
                       extras: bool = True, **kw: Any) -> DoublesScene:
    """続けて打った ``n_rallies`` 個のラリーと、4 人の動きの計画・誰が打ったかの正解。"""
    session = make_session(rng, phys, n_rallies, n_shots=n_shots, rest_s=rest_s, **kw)
    plan: dict[str, list[_Waypoint]] = {name: [] for name in PEOPLE}
    cur: dict[str, np.ndarray] = {}
    hitters: list[list[str]] = []
    for r in session:
        serve = r.segments[0]
        s_side, r_side = serve.side, "o" if serve.side == "h" else "h"
        server, receiver = int(rng.integers(2)), int(rng.integers(2))
        sp = serve.p0[:2] + np.clip(rng.normal(0, 0.15, 2), -0.3, 0.3)
        sy = float(np.sign(sp[1])) or 1.0
        pos = {f"{s_side}{server}": sp,
               f"{s_side}{1 - server}": np.array([_sx(s_side) * rng.uniform(4.2, 4.9), -sy * rng.uniform(0.0, 0.6)]),
               f"{r_side}{receiver}": np.array([_sx(r_side) * rng.uniform(2.4, 3.0), -sy * rng.uniform(0.6, 1.2)]),
               f"{r_side}{1 - receiver}": np.array([_sx(r_side) * rng.uniform(4.2, 4.9), sy * rng.uniform(0.2, 0.8)])}
        hold = rng.uniform(1.0, 1.8)
        for name in PEOPLE:
            plan[name].append(_Waypoint(r.t_start - hold, pos[name], WALK_PACE_MPS, on_time=True))
            plan[name].append(_Waypoint(r.t_start, pos[name], WALK_PACE_MPS))
            cur[name] = pos[name]
        names = []
        for k, seg in enumerate(r.segments):
            side = seg.side
            if k == 0:
                hit = server
            else:
                d = [float(np.hypot(*(cur[f"{side}{i}"] - seg.p0[:2]))) for i in (0, 1)]
                hit = int(rng.integers(2)) if abs(d[0] - d[1]) < 0.4 else int(np.argmin(d))
                feet = seg.p0[:2] + np.clip(rng.normal(0, REACH_M, 2), -0.9, 0.9)
                feet[0] = _sx(side) * max(0.4, abs(feet[0]))
                plan[f"{side}{hit}"].append(_Waypoint(seg.t_hit, feet, PLAY_PACE_MPS, on_time=True))
                cur[f"{side}{hit}"] = feet
            me, mate = f"{side}{hit}", f"{side}{1 - hit}"
            names.append(me)
            targets = _formation_targets(side, cur[me], seg, float(cur[mate][1]), rng)
            if targets is not None:
                t_arrive = seg.t_hit + rng.uniform(0.7, 1.0)
                for name, xy in zip((me, mate), targets):
                    plan[name].append(_Waypoint(t_arrive, xy, PLAY_PACE_MPS))
                    cur[name] = xy
        hitters.append(names)
    ex: dict[str, Any] = {}
    if extras:
        ex = {"queue_side": str(rng.choice(["h", "o"])), "queue_y": rng.uniform(-2.0, 2.0, 2).tolist(),
              "seed": int(rng.integers(1 << 30))}
    return DoublesScene(session, hitters, plan, ex)


def extra_feet(scene: DoublesScene, times: np.ndarray, fps: float) -> dict[str, np.ndarray]:
    """選手ではない人の足元（名前 → (n, 2)。写っていない間は NaN）。待つ人 2 人・隣のコートの 2 人・横を歩く人。"""
    if not scene.extras:
        return {}
    rng = np.random.default_rng(scene.extras["seed"])
    n = len(times)
    out: dict[str, np.ndarray] = {}
    sx = _sx(scene.extras["queue_side"])
    for k, y in enumerate(scene.extras["queue_y"]):    # コートの後ろの線のすぐ外で待つ（少し揺れる）
        base = np.array([sx * rng.uniform(6.9, 7.5), y])
        out[f"queue{k}"] = base + 0.12 * np.column_stack([_slow_noise(n, fps, 2.0, rng), _slow_noise(n, fps, 2.0, rng)])
    dur = float(times[-1] - times[0]) + 1.0
    for side in ("h", "o"):                            # 隣のコートの選手
        _, xy = footwork_path(dur, fps, side, rng)
        xy = xy[:n] if len(xy) >= n else np.vstack([xy, np.repeat(xy[-1:], n - len(xy), axis=0)])
        out[f"nb_{side}"] = xy + np.array([0.0, NEIGHBOR_DY_M])
    walk = np.full((n, 2), np.nan)                     # コートの横（手前の線の外）を端から端へ歩く人
    t0 = float(rng.uniform(times[0], max(times[0] + 1.0, times[-1] - 12.0)))
    m = (times >= t0) & (times <= t0 + 12.0)
    walk[m] = np.column_stack([-7.5 + 15.0 * (times[m] - t0) / 12.0, np.full(int(m.sum()), -3.7)])
    out["walker"] = walk
    return out


# =============================================================================
# 観測
# =============================================================================
def make_kits(rng: np.random.Generator, names: list[str], kits: str = "distinct") -> dict[str, np.ndarray]:
    """人ごとの色の割合（長さ ``APP_DIM``。上半身・下半身それぞれ合計 1）。"""
    if kits not in KITS:
        raise ValueError(f"kits は {', '.join(KITS)} のどれかにしてください: {kits!r}")
    half = APP_DIM // 2
    draw = lambda: rng.dirichlet(np.full(half, 0.3))  # noqa: E731
    out: dict[str, np.ndarray] = {}
    pair: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in names:
        side = name[0] if name in PEOPLE else ""
        if side and kits != "distinct":
            if side not in pair:
                pair[side] = (draw(), draw())
            out[name] = np.r_[pair[side][0], pair[side][1] if kits == "same" else draw()]
        else:
            out[name] = np.r_[draw(), draw()]
    return out


def observe_people(times: np.ndarray, feet: dict[str, np.ndarray], camera: Camera, rng: np.random.Generator,
                   kits: dict[str, np.ndarray], hits: list[tuple[float, str, float]] | None = None,
                   jitter: float = 0.005, outlier_rate: float = 0.05, outlier: float = 0.04, drift: float = 0.01,
                   swing: float = 0.12, miss_rate: float = 0.02, occlude: tuple[float, float] = (0.3, 0.7),
                   merge_rate: float = 0.3, app_noise: tuple[float, float] = (0.1, 0.3)) -> pd.DataFrame:
    """足元の軌跡（人 → (n, 2)。NaN は写っていない）を写した人の検出（``people.csv`` の形 + 正解の ``who``）。

    ``hits`` は（打った時刻・打った人・打点の高さ [m]）。打つ人の枠は前後 0.2 秒、高く・広くなる。
    ``occlude`` は「遠い人の枠が近い人の枠に隠れた割合」がこの範囲で 0 → 1 に検出が抜けやすくなる幅、
    ``merge_rate`` は半分より多く隠れたときに 2 人が 1 つの枠にまとまる割合。
    """
    t = np.asarray(times, float)
    n = len(t)
    fps = 1.0 / float(np.median(np.diff(t)))
    names = list(feet)
    box = np.full((len(names), n, 4), np.nan)
    depth = np.full((len(names), n), np.inf)
    for k, name in enumerate(names):
        xy = np.asarray(feet[name], float)
        ok = np.isfinite(xy).all(axis=1)
        X = np.column_stack([np.nan_to_num(xy), np.zeros(n)])
        tall = 1.75 + 0.08 * rng.normal()
        foot = camera.project(X)
        head = camera.project(X + [0.0, 0.0, tall])
        h = foot[:, 1] - head[:, 1]
        noise = rng.normal(0, jitter, (n, 2)) * h[:, None]
        wild = rng.random(n) < outlier_rate
        noise[wild] += rng.normal(0, outlier, (int(wild.sum()), 2)) * h[wild, None]
        noise += drift * np.column_stack([_slow_noise(n, fps, 0.2, rng), _slow_noise(n, fps, 0.2, rng)]) * h[:, None]
        up = 0.03 * _slow_noise(n, fps, 0.5, rng)       # 姿勢で枠の高さが少し変わる
        wide = np.zeros(n)
        for th, who, z in hits or []:
            if who != name:
                continue
            w = (t - th) / 0.2
            bump = np.where(np.abs(w) < 1, 0.5 * (1 + np.cos(np.pi * w)), 0.0)
            noise[:, 0] += rng.choice([-1, 1]) * swing * bump * h
            up += bump * max(0.0, min(z, 2.7) - tall) / tall
            wide += 0.35 * bump
        uv = foot + noise
        hh = h * (1 + up)
        ww = 0.4 * h * (1 + wide)
        b = np.column_stack([uv[:, 0] - ww / 2, uv[:, 1] - hh, uv[:, 0] + ww / 2, uv[:, 1]])
        ok &= (h > 8) & (uv[:, 0] > 0) & (uv[:, 0] < camera.width) & (uv[:, 1] > 0) & (uv[:, 1] < camera.height * 1.05)
        b[~ok] = np.nan
        box[k] = b
        depth[k] = np.where(ok, np.linalg.norm(X - camera.center, axis=1), np.inf)
    area = (box[:, :, 2] - box[:, :, 0]) * (box[:, :, 3] - box[:, :, 1])
    cover = np.zeros((len(names), n))               # 自分より近い人に隠れた割合（いちばん大きいもの）
    by = np.full((len(names), n), -1)
    for a in range(len(names)):
        for b_ in range(len(names)):
            if a == b_:
                continue
            with np.errstate(invalid="ignore"):
                iw = np.minimum(box[a, :, 2], box[b_, :, 2]) - np.maximum(box[a, :, 0], box[b_, :, 0])
                ih = np.minimum(box[a, :, 3], box[b_, :, 3]) - np.maximum(box[a, :, 1], box[b_, :, 1])
                c = np.where((depth[b_] < depth[a]) & (iw > 0) & (ih > 0), iw * ih / area[a], 0.0)
            c = np.nan_to_num(c)
            by[a] = np.where(c > cover[a], b_, by[a])
            cover[a] = np.maximum(cover[a], c)
    present = np.isfinite(box[:, :, 0])
    lost = np.zeros_like(present)
    merged = np.zeros_like(present)
    for k in range(len(names)):
        u = ndtr(_slow_noise(n, fps, 0.3, rng))     # 0.3 秒ほど続く運（隠れは何コマか続く）
        p_occ = np.clip((cover[k] - occlude[0]) / (occlude[1] - occlude[0]), 0.0, 1.0)
        merged[k] = present[k] & (cover[k] > 0.5) & (ndtr(_slow_noise(n, fps, 0.3, rng)) < merge_rate)
        lost[k] = present[k] & ((u < p_occ) | merged[k] | (rng.random(n) < miss_rate))
    rows = []
    half = APP_DIM // 2
    for k, name in enumerate(names):
        b = box[k].copy()
        for j in np.flatnonzero(merged.any(axis=0)):   # 自分に隠れた人とまとまった枠
            for a in np.flatnonzero(merged[:, j] & (by[:, j] == k)):
                if present[k, j]:
                    b[j] = [min(b[j, 0], box[a, j, 0]), min(b[j, 1], box[a, j, 1]),
                            max(b[j, 2], box[a, j, 2]), max(b[j, 3], box[a, j, 3])]
        keep = np.flatnonzero(present[k] & ~lost[k])
        if not len(keep):
            continue
        eps = rng.uniform(*app_noise, len(keep))[:, None]
        app = (1 - eps) * kits[name] + eps * np.c_[rng.dirichlet(np.ones(half), len(keep)),
                                                    rng.dirichlet(np.ones(half), len(keep))]
        other = by[k, keep]
        mix = np.where(other >= 0, 0.8 * cover[k, keep], 0.0)[:, None]
        others = np.array([kits[names[q]] if q >= 0 else kits[name] for q in other])
        app = (1 - mix) * app + mix * others
        codes = [encode_appearance(np.round(255 * np.r_[a[:half] / a[:half].sum(), a[half:] / a[half:].sum()]))
                 for a in app]
        rows.append(pd.DataFrame({"frame": keep, "t": t[keep], "x0": b[keep, 0], "y0": b[keep, 1], "x1": b[keep, 2],
                                  "y1": b[keep, 3], "score": np.where(cover[k, keep] > 0.1, 0.6, 0.9),
                                  "app": codes, "who": name}))
    if not rows:
        return add_court_positions(pd.DataFrame(columns=["frame", "t", "x0", "y0", "x1", "y1", "score", "app", "who"]),
                                   camera)
    people = pd.concat(rows, ignore_index=True).sort_values(["frame", "x0"]).reset_index(drop=True)
    return add_court_positions(people, camera)


def observe_doubles(scene: DoublesScene, camera: Camera, fps: float, phys: PhysicsParams, rng: np.random.Generator,
                    kits: str = "distinct", **kw: Any) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """シャトルの検出（``synth_rally.observe_session``）と、同じコマの人の検出（``observe_people``）、4 人の足元の正解。"""
    det = observe_session(scene.session, camera, fps, phys, rng)
    t = det["timestamp"].to_numpy(float)
    feet = scene.feet(t)
    everyone = {**feet, **extra_feet(scene, t, fps)}
    marks = [(th, who, float(p0[2])) for th, _, who, p0, _ in scene.hits()]
    people = observe_people(t, everyone, camera, rng, make_kits(rng, list(everyone), kits), marks, **kw)
    return det, people, feet


def truth_players(times: np.ndarray, feet: dict[str, np.ndarray], camera: Camera) -> pd.DataFrame:
    """足元の正解を ``players.csv`` の形にしたもの（揺れなし。陣形の正解を出すのに使う）。"""
    rows = []
    n = len(times)
    for name in PEOPLE:
        xy = feet[name]
        X = np.column_stack([xy, np.zeros(n)])
        foot, head = camera.project(X), camera.project(X + [0.0, 0.0, 1.75])
        h = foot[:, 1] - head[:, 1]
        rows.append(pd.DataFrame({"frame": np.arange(n), "t": times, "side": name[0], "player": int(name[1]),
                                  "x": xy[:, 0], "y": xy[:, 1], "u": foot[:, 0], "v": foot[:, 1],
                                  "x0": foot[:, 0] - 0.2 * h, "y0": head[:, 1], "x1": foot[:, 0] + 0.2 * h,
                                  "y1": foot[:, 1], "score": 1.0, "interpolated": False, "airborne": False}))
    return pd.concat(rows, ignore_index=True)


# =============================================================================
# 答え合わせ
# =============================================================================
def who_is_tracked(players: pd.DataFrame, people: pd.DataFrame) -> pd.DataFrame:
    """追跡した選手の行（補っていないコマ）に、その検出の正解の人（``who``）を付ける。"""
    real = players[~players["interpolated"].astype(bool)]
    key = people[["frame", "x0", "y0", "who"]].copy()
    key["x0"], key["y0"] = key["x0"].round(3), key["y0"].round(3)
    left = real.assign(x0=real["x0"].round(3), y0=real["y0"].round(3))
    return left.merge(key.drop_duplicates(["frame", "x0", "y0"]), on=["frame", "x0", "y0"], how="left")


def tracking_scores(players: pd.DataFrame, people: pd.DataFrame, n_frames: int) -> dict[str, Any]:
    """番号の付き方の答え合わせ（側ごと）。

    ``id_accuracy``：その側の選手を追ったコマのうち、番号が正解の人と合っていた割合（番号と人の対応は合う方）。
    ``switches``：1 つの番号が追う人が入れ替わった回数。``extra_share``：選手ではない人を追ったコマの割合。
    ``coverage``：選手ごとの、追えたコマの割合の平均。``mapping``：番号 → 正解の人。
    """
    tr = who_is_tracked(players, people)
    out: dict[str, Any] = {}
    for side in ("h", "o"):
        g = tr[tr["side"] == side]
        names = [f"{side}0", f"{side}1"]
        mine = g[g["who"].isin(names)]
        best, mapping = -1, {0: names[0], 1: names[1]}
        for perm in ((0, 1), (1, 0)):
            m = {0: names[perm[0]], 1: names[perm[1]]}
            ok = int((mine["who"] == mine["player"].map(m)).sum())
            if ok > best:
                best, mapping = ok, m
        switches = 0
        for _, gp in mine.sort_values("frame").groupby("player"):
            w = gp["who"].to_numpy()
            switches += int(np.sum(w[1:] != w[:-1]))
        out[side] = {"n": int(len(g)), "id_accuracy": round(best / max(1, len(mine)), 4), "switches": switches,
                     "extra_share": round(1.0 - len(mine) / max(1, len(g)), 4),
                     "coverage": round(float(np.mean([(mine["who"] == nm).sum() / max(1, n_frames) for nm in names])), 4),
                     "mapping": mapping}
    return out


def local_who(tr: pd.DataFrame, side: str, player: int, t: float, window_s: float = 0.5) -> str:
    """時刻 ``t`` の前後で、その番号が追っていた正解の人（いちばん多いもの。なければ空文字）。"""
    g = tr[(tr["side"] == side) & (tr["player"] == player) & ((tr["t"] - t).abs() <= window_s)]
    w = g["who"].dropna()
    return "" if w.empty else str(w.mode().iloc[0])


def doubles_benchmark(n_sessions: int = 2, n_rallies: int = 5, fps: float = 30.0, seed: int = 0,
                      cameras: tuple[str, ...] = ("back_high", "corner_low", "side"), kits: str = "distinct",
                      extras: bool = True, hit_noise_m: float = 0.5) -> dict[str, Any]:
    """合成のダブルスで、追跡・誰が打ったか・陣形を正解と比べる（カメラごと）。

    打球の時刻と打った側は正解を使う（打球の見つけ方・当てはめは別のチェックで見ている）。誰が打ったかは、
    当てはめの前（画像の手がかりだけ：``hitter_pre``）と、打点が ``hit_noise_m`` の誤差で分かったとき（``hitter_post``）の 2 通り。
    ``hitter_*`` は「追っていた人が正解の人か」、``hitter_*_id`` は番号の入れ替わりも含めて合っていたか。
    """
    from smash_speed.config import load_config
    from smash_speed.doubles import PairTracks, formation_table, pick_hitter
    from smash_speed.hits import _visible_points
    from smash_speed.player_ids import track_sides
    from smash_speed.players import PlayerOptions, ground_jumps
    from smash_speed.synth_clutter import clutter_camera

    cfg = load_config()
    phys = PhysicsParams.from_config(cfg)
    res: dict[str, Any] = {}
    for cam_name in cameras:
        camera = clutter_camera(cam_name, cfg)
        acc: dict[str, list[float]] = {k: [] for k in ("id_accuracy", "extra_share", "coverage", "formation",
                                                        "formation_stable")}
        n_hits = {k: 0 for k in ("n", "pre", "pre_id", "post", "post_id")}
        switches, minutes, auto_ok = 0, 0.0, 0
        for k in range(n_sessions):
            rng = np.random.default_rng(seed + k)
            scene = make_doubles_scene(rng, phys, n_rallies, extras=extras)
            det, people, feet = observe_doubles(scene, camera, fps, phys, rng, kits)
            frame_t = pd.Series(det["timestamp"].to_numpy(float), index=det["frame"].to_numpy())
            players, info = track_sides(people.drop(columns="who"), frame_t, cfg, PlayerOptions(per_side=0))
            auto_ok += int(info["per_side"] == {"h": 2, "o": 2})
            if info["per_side"] != {"h": 2, "o": 2}:
                players, info = track_sides(people.drop(columns="who"), frame_t, cfg, PlayerOptions(per_side=2))
            players = ground_jumps(players, camera)
            sc = tracking_scores(players, people, len(frame_t))
            minutes += float(frame_t.iloc[-1] - frame_t.iloc[0]) / 60.0
            for side in ("h", "o"):
                for m in ("id_accuracy", "extra_share", "coverage"):
                    acc[m].append(sc[side][m])
                switches += sc[side]["switches"]
            # 誰が打ったか
            tr = who_is_tracked(players, people)
            tracks = PairTracks(players, camera)
            vis = _visible_points(det)
            for t, side, who, p0, _ in scene.hits():
                n_hits["n"] += 1
                for mode in ("pre", "post"):
                    pts = None if mode == "pre" else [p0[:2] + rng.normal(0, hit_noise_m, 2)]
                    pick = pick_hitter(tracks, side, t, vis, pts)
                    if pick is None:
                        continue
                    n_hits[mode] += int(local_who(tr, side, pick.player, t) == who)
                    n_hits[mode + "_id"] += int(sc[side]["mapping"][pick.player] == who)
            # 陣形（ラリーの中のコマ）
            est = formation_table(players)
            true = formation_table(truth_players(frame_t.to_numpy(), feet, camera))
            j = est.merge(true, on=["side", "frame"], suffixes=("", "_true"))
            in_rally = np.zeros(len(j), bool)
            for r in scene.session:
                in_rally |= ((j["t"] >= r.t_start) & (j["t"] <= r.t_end)).to_numpy()
            j = j[in_rally]
            if len(j):
                acc["formation"].append(float((j["formation"] == j["formation_true"]).mean()))
                st = j[j["formation_true"] != "transition"]
                acc["formation_stable"].append(float((st["formation"] == st["formation_true"]).mean()) if len(st) else 1.0)
        n = max(1, n_hits["n"])
        res[cam_name] = {
            "n_sessions": n_sessions, "auto_doubles": auto_ok, "minutes": round(minutes, 2),
            "id_accuracy": round(float(np.mean(acc["id_accuracy"])), 4),
            "switches": switches, "switches_per_min": round(switches / max(minutes, 1e-9) / 4, 3),
            "extra_share": round(float(np.mean(acc["extra_share"])), 4),
            "coverage": round(float(np.mean(acc["coverage"])), 4),
            "n_hits": n_hits["n"], **{f"hitter_{m}": round(n_hits[m] / n, 4) for m in ("pre", "pre_id", "post", "post_id")},
            "formation": round(float(np.mean(acc["formation"])), 4),
            "formation_stable": round(float(np.mean(acc["formation_stable"])), 4),
        }
    return res


def singles_auto_benchmark(n_sessions: int = 2, n_rallies: int = 4, fps: float = 30.0, seed: int = 0,
                           cameras: tuple[str, ...] = ("back_high", "corner_low", "side")) -> dict[str, Any]:
    """シングルスの合成ラリー（1 側 1 人）に待つ人・隣のコートの人を足して、自動がシングルスのままにするか、
    その結果がシングルスの追跡（``per_side`` = 1）と同じかを確かめる。カメラごとの {合った回数, 回数}。"""
    from smash_speed.config import load_config
    from smash_speed.players import PlayerOptions, track_players
    from smash_speed.synth_clutter import clutter_camera

    cfg = load_config()
    phys = PhysicsParams.from_config(cfg)
    res: dict[str, Any] = {}
    for cam_name in cameras:
        camera = clutter_camera(cam_name, cfg)
        ok = same = 0
        for k in range(n_sessions):
            rng = np.random.default_rng(seed + k)
            scene = make_doubles_scene(rng, phys, n_rallies)
            det = observe_session(scene.session, camera, fps, phys, rng)
            t = det["timestamp"].to_numpy(float)
            feet = scene.feet(t)
            everyone = {"h0": feet["h0"], "o0": feet["o0"], **extra_feet(scene, t, fps)}
            people = observe_people(t, everyone, camera, rng, make_kits(rng, list(everyone)))
            frame_t = pd.Series(t, index=det["frame"].to_numpy())
            auto = track_players(people.drop(columns="who"), frame_t, cfg, PlayerOptions(per_side=0))
            one = track_players(people.drop(columns="who"), frame_t, cfg, PlayerOptions(per_side=1))
            ok += int(auto.groupby("side")["player"].nunique().max() == 1)
            same += int(auto.equals(one))
        res[cam_name] = {"n": n_sessions, "singles": ok, "same_as_singles": same}
    return res


def format_benchmark(res: dict[str, Any]) -> str:
    """``doubles_benchmark`` の結果の表（カメラごとに 1 行）。"""
    lines = ["カメラ        番号の正しさ  入れ替わり/人・分  選手以外  追えた  打った人(前/後)  番号込み(前/後)  陣形(全体/決まった形)"]
    for name, r in res.items():
        lines.append(f"{name:<12} {r['id_accuracy']:>10.1%} {r['switches_per_min']:>14.2f} {r['extra_share']:>9.1%} "
                     f"{r['coverage']:>6.1%}   {r['hitter_pre']:.0%} / {r['hitter_post']:.0%}       "
                     f"{r['hitter_pre_id']:.0%} / {r['hitter_post_id']:.0%}       "
                     f"{r['formation']:.0%} / {r['formation_stable']:.0%}   （打球 {r['n_hits']} 本、"
                     f"自動でダブルス {r['auto_doubles']}/{r['n_sessions']}）")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="合成のダブルスで、追跡・誰が打ったか・陣形を正解と比べる")
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--rallies", type=int, default=5)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kits", choices=list(KITS), nargs="+", default=list(KITS))
    a = ap.parse_args(argv)
    for kits in a.kits:
        print(f"== 服の色：{kits} ==")
        print(format_benchmark(doubles_benchmark(a.sessions, a.rallies, a.fps, a.seed, kits=kits)))
    print("== シングルス（自動でシングルスのままか）==")
    print(singles_auto_benchmark(a.sessions, fps=a.fps, seed=a.seed))


if __name__ == "__main__":
    import sys

    if (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    main()
