"""ショットごとのフットワーク（スプリットステップ・反応の速さ・移動・戻り・ジャンプ）と、ショットの出来の目安の点数。

入力は、ショットの一覧（``shots.csv``。ラリー・打った側・打点・着地点）と選手の位置（``players.csv``。
``players.ground_jumps`` の後）と、キャリブレーションのカメラ。1 行は「そのショットを打った選手」の、
相手の 1 本前のショット（受けた球）を打たれてから自分が打つまでと、打った後の動き。

- **スプリットステップ**：相手が打つ少し前〜直後（``split_window_s``）に、足が小さく床から浮いたか。
  枠の下の辺（足）・上の辺（頭）が、そのコマの前後 0.15〜0.45 秒の中央値をつないだ基準（短い跳びは基準に入らない）より
  上がった分を、その位置の 1 m の縦の長さ [px] で割って「浮いた高さ」にする（足と頭の小さい方。足だけ・頭だけの
  動きは踏み込み・伸び上がりなので数えない）。``split_min_m``〜``split_max_m`` の山があればスプリットステップ、その時刻と
  相手の打った時刻の差（負なら打つ前）を出す。
- **反応の速さ**：相手が打ってから、打つ所（自分が打った時の足元）へ向かって ``react_mps`` より速く
  ``react_hold_s`` 続けて動き出すまで [s]。打つ所へ向かう向き（速度と打つ所への向きのなす角の cos が
  ``toward_cos`` 以上）だけを数える。負なら相手が打つ前に動き出した（読んでいた）。相手が打つ ``react_lead_s`` 前から
  もう打つ所へ向かって動いていたら、前の動きの続きなので空欄（練習の続けて打つ球に多い）。
- **移動**：相手が打ってから自分が打つまでの移動距離と最高の速さ（``movement`` と同じならし方。足元の揺れが床の上で
  大きい奥の選手（``players.foot_noise_m``）ほど最高の速さを強くならす。K3）。
- **準備の時間**：打つ所（打った時の足元）から ``arrive_m`` 以内に着いてから打つまで [s]。
- **戻り**：打った時の足元から、そのラリーでいちばん長くいた所（``base``。ラリーの間の足元の中央値）までの距離と、
  次に相手が打つまでに ``recover_m`` 以内へ戻るのにかかった時間 [s]（戻れなければ空欄）。
- **ジャンプ**：打つ前後（``jump_window_s``）の浮いた高さの最大と、``jump_min_m`` を超えたか（跳んで打ったか）。
  ``players.ground_jumps`` が跳んでいるとしたコマがあれば、それも跳んだとする。

浮いた高さは小さい（スプリットステップは 5〜12 cm、4K の奥の選手で数 px）ので、確かでないことがある。
窓の中に、枠が画面の端に付いたコマ（体が切れている）、位置が出ていないコマ・補ったコマ・足元が不確かなコマ
（``players.uncertain_feet``。ネットの帯・ほかの選手の枠）が多いときは
``uncertain`` を真にして、理由を ``uncertain_reason`` に書く。まとめ（``footwork.json``）は確かな行だけで出す。

**ショットの出来の目安**（``quality``、0〜100）は、決まった項目の点（それぞれ 0〜1）の平均。学習はしない。
項目と、その種類で使うかどうかは ``QUALITY_TERMS``・``quality_terms`` に書いた（``docs/footwork.md`` にも）。

``write_footwork`` が ``footwork.csv``（1 ショット 1 行）と ``footwork.json``（選手ごと・ラリーごとのまとめ）を書く。
``python -m smash_speed.footwork <ショットの一覧の出力先>`` で書き直せる（``players.csv`` と ``shot_list.json`` の
キャリブレーションを使う）。
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from smash_speed.camera import Camera
from smash_speed.movement import MovementOptions, _smooth, smooth_track
from smash_speed.shots import Shot

FOOTWORK_CSV = "footwork.csv"
FOOTWORK_FILE = "footwork.json"


@dataclass
class FootworkOptions:
    """フットワークの条件。"""

    # 浮いた高さの基準にする、前後の範囲（そのコマから inner〜outer 秒離れた所の中央値）[s]
    lift_gap_s: tuple[float, float] = (0.15, 0.45)   # スプリットステップ（0.2〜0.3 秒浮く）
    jump_gap_s: tuple[float, float] = (0.3, 0.75)    # ジャンプ（0.5 秒ほど浮く）
    split_window_s: tuple[float, float] = (-0.45, 0.2)  # 相手が打った時刻からの、スプリットステップを探す範囲 [s]
    split_min_m: float = 0.04       # これより高く浮いたらスプリットステップ [m]
    split_max_m: float = 0.25       # これより高いのはスプリットステップではない（ジャンプ）[m]
    split_max_depth: float = 0.03
    split_max_mps: float = 2.5      # 相手が打った時にこれより速く動いていたら、スプリットステップは決めない（空欄）[m/s]   # 枠の高さが前後の基準からこの割合より変わったコマは浮いたとみなさない
    react_mps: float = 1.0          # これより速く動いたら動き出したとみなす [m/s]
    react_hold_s: float = 0.1       # その速さが続く長さ [s]
    react_lead_s: float = 0.2       # 相手が打つこれだけ前からの動き出しも数える（読んで動いた）[s]
    react_max_s: float = 1.0        # 相手が打ってからこれより遅い動き出しは探さない [s]
    toward_cos: float = 0.3         # 打つ所へ向かっているとみなす向き（cos）
    arrive_m: float = 0.4           # 打つ所からこれ以内に来たら「着いた」[m]
    recover_m: float = 0.6          # ラリーでいちばんいた所からこれ以内に来たら「戻った」[m]
    recover_max_s: float = 1.5      # 次に相手が打たないときに戻りを探す長さ [s]
    no_prev_s: float = 1.0          # 受けた球がない（サーブ）ときに移動・準備を見る、打つ前の長さ [s]
    jump_window_s: tuple[float, float] = (-0.35, 0.15)  # 打った時刻からの、ジャンプを探す範囲 [s]
    jump_min_m: float = 0.1         # これより高く浮いたらジャンプ [m]（IMG_5109 13.8 s の跳んだスマッシュが 14 cm）
    # いちばん高く浮いたコマの枠の高さが、前後の基準からこの割合より変わっていたら、カメラへ近づいた・離れた動きと
    # 跳んだのとが分けられないので、ジャンプは出さない（空欄）
    jump_max_depth: float = 0.03
    edge_px: float = 3.0            # 枠が画面の端からこれ以内なら、体が切れているかもしれない [px]
    max_missing: float = 0.3        # 窓の中の、位置が出ていないコマの割合の上限
    max_interpolated: float = 0.3   # 同・補ったコマの割合の上限
    # 足元の確かさ（I5。``players.uncertain_feet``・``foot_noise_m``）を使う（K3）。``foot_noise_m`` のない古い
    # ``players.csv`` でも、カメラと枠があればここで出し、奥の選手ほど最高の速さを強くならす（``movement``）
    use_foot_noise: bool = True
    max_foot_uncertain: float = 0.3  # 窓の中の「足元が不確か」のコマの割合の上限
    max_prev_gap_s: float = 2.5     # ラリーが分からないとき、受けた球とみなす 1 本前のショットまでの間隔の上限 [s]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# =============================================================================
# 選手ごとの位置・浮いた高さ
# =============================================================================
def _odd(n: float) -> int:
    return max(1, int(round(n)) | 1)


def _gap_baseline(t: np.ndarray, v: np.ndarray, inner_s: float, outer_s: float) -> np.ndarray:
    """各コマの値の基準：前（``t − outer_s``〜``t − inner_s``）と後（``t + inner_s``〜``t + outer_s``）の中央値を、
    それぞれの時刻の真ん中で直線につないだ所。真ん中（``± inner_s``）は見ないので、短い跳びは基準に入らず、
    一定の速さの動きは基準が追う。片方しかなければその中央値。"""
    out = np.full(len(t), np.nan)
    lo_a, lo_b = np.searchsorted(t, t - outer_s), np.searchsorted(t, t - inner_s)
    hi_a, hi_b = np.searchsorted(t, t + inner_s, "right"), np.searchsorted(t, t + outer_s, "right")
    for i in range(len(t)):
        L, R = v[lo_a[i]:lo_b[i]], v[hi_a[i]:hi_b[i]]
        L, R = L[np.isfinite(L)], R[np.isfinite(R)]
        if len(L) >= 2 and len(R) >= 2:
            tl, tr = np.mean(t[lo_a[i]:lo_b[i]]), np.mean(t[hi_a[i]:hi_b[i]])
            ml, mr = float(np.median(L)), float(np.median(R))
            out[i] = ml + (mr - ml) * (t[i] - tl) / (tr - tl) if tr > tl else (ml + mr) / 2
        elif len(L) >= 2 or len(R) >= 2:
            out[i] = float(np.median(L if len(L) >= 2 else R))
    return out


def _lift(g: pd.DataFrame, camera: Camera, inner_s: float, outer_s: float, with_depth: bool = False) -> Any:
    """枠の下の辺（足）と上の辺（頭）が、前後の基準（``_gap_baseline``）より上がった高さの小さい方 [m]。

    跳ぶと足も頭も同じだけ上がる。足だけ（踏み込み・足を上げる）、頭だけ（伸び上がる）の動きは小さい方で落ちる。
    画像の上の差 [px] を、基準の床の位置での 1 m の縦の長さ [px] で割って高さにする。"""
    t = g["t"].to_numpy(float)
    base = {c: _gap_baseline(t, g[c].to_numpy(float), inner_s, outer_s) for c in ("x", "y", "v", "y0")}
    X0 = np.column_stack([base["x"], base["y"], np.zeros(len(g))])
    ok = np.isfinite(X0).all(axis=1)
    ppm = np.full(len(g), np.nan)
    if ok.any():
        X = X0[ok]
        ppm[ok] = camera.project(X)[:, 1] - camera.project(X + [0.0, 0.0, 1.0])[:, 1]   # 1 m の縦の長さ [px]
    with np.errstate(invalid="ignore", divide="ignore"):
        foot = np.where(ppm > 0, (base["v"] - g["v"].to_numpy(float)) / ppm, np.nan)
        head = np.where(ppm > 0, (base["y0"] - g["y0"].to_numpy(float)) / ppm, np.nan)
    lift = np.fmin(foot, head)
    if with_depth:
        # 枠の高さの、基準からの変わり方。跳んでも枠の高さはほぼ変わらず、カメラへ近づく・離れると変わる
        h = (g["v"] - g["y0"]).to_numpy(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            depth = np.abs(h / (base["v"] - base["y0"]) - 1)
        return lift, depth
    # 1 コマだけの外れ（足を開いた・枠の揺れ）を落とす
    return median_filter(np.nan_to_num(lift), 3, mode="nearest") if len(lift) >= 3 else lift


def player_tracks(players: pd.DataFrame, camera: Camera | None = None, image_size: tuple[int, int] | None = None,
                  o: FootworkOptions | None = None, mo: MovementOptions | None = None
                  ) -> dict[tuple[str, int], pd.DataFrame]:
    """選手ごとの、ならした位置・速さ・速度・浮いた高さ・画面の端の印（1 コマ 1 行）。

    浮いた所（``split_min_m`` より高い）は、足元の床の位置が浮いた分だけ遠くへずれる（低いカメラでは 10 cm で
    1 m ほど）ので、前後の床にいたコマから補ってからならす。カメラがなければ浮いた高さは空欄。
    """
    o = o or FootworkOptions()
    out: dict[tuple[str, int], pd.DataFrame] = {}
    if players is None or players.empty:
        return out
    if image_size is None and camera is not None:
        image_size = (int(camera.width), int(camera.height))
    for (side, player), g in players.groupby(["side", "player"]):
        g = g.dropna(subset=["x", "y"]).sort_values("t").reset_index(drop=True)
        if len(g) < 3:
            continue
        t = g["t"].to_numpy(float)
        step = np.diff(t)
        fps = 1.0 / float(np.median(step)) if len(step) and np.median(step) > 0 else 30.0
        has_box = all(c in g for c in ("u", "v", "x0", "y0", "x1", "y1"))
        if camera is not None and has_box:
            lift, depth = _lift(g, camera, *o.lift_gap_s, with_depth=True)
            # 枠の高さが変わったコマ（カメラへ近づく・離れる動き）は浮いたとみなさない
            lift = median_filter(np.nan_to_num(np.where(depth <= o.split_max_depth, lift, 0.0)), 3, mode="nearest")
            jlift, jdepth = _lift(g, camera, *o.jump_gap_s, with_depth=True)
            jlift = median_filter(np.nan_to_num(jlift), 3, mode="nearest")
        else:
            lift = jlift = jdepth = np.full(len(g), np.nan)
        fixed = g.copy()
        if not o.use_foot_noise:
            fixed = fixed.drop(columns=[c for c in ("foot_noise_m",) if c in fixed])
        elif "foot_noise_m" not in fixed and camera is not None and has_box:
            from smash_speed.players import foot_noise_m

            fixed["foot_noise_m"] = foot_noise_m(fixed, camera)
        hop = np.nan_to_num(lift) > o.split_min_m
        if hop.any() and (~hop).sum() >= 2:
            for c in ("x", "y"):
                fixed.loc[hop, c] = np.interp(t[hop], t[~hop], g[c].to_numpy(float)[~hop])
        s = smooth_track(fixed, mo)
        tr = pd.DataFrame({"t": t, "x": s["x"].to_numpy(float), "y": s["y"].to_numpy(float),
                           "speed": s["speed"].to_numpy(float), "peak_speed": s["peak_speed"].to_numpy(float),
                           "segment": s["segment"].to_numpy(int), "lift": lift, "jump_lift": jlift,
                           "jump_depth": jdepth})
        # 動き出しは弱くならした位置の速度で見る（強くならすと動き出しが 0.1 秒ほど早く出る）
        mo2 = mo or MovementOptions()
        px = np.empty(len(tr))
        py = np.empty(len(tr))
        for _, idx in tr.groupby("segment").groups.items():
            k = np.asarray(idx)
            px[k] = _smooth(fixed["x"].to_numpy(float)[k], fps, mo2.median_s, mo2.peak_smooth_s)
            py[k] = _smooth(fixed["y"].to_numpy(float)[k], fps, mo2.median_s, mo2.peak_smooth_s)
        tr["vx"], tr["vy"] = _velocity(tr.assign(x=px, y=py))
        tr["airborne"] = g["airborne"].astype(bool).to_numpy() if "airborne" in g else False
        tr["interpolated"] = g["interpolated"].astype(bool).to_numpy() if "interpolated" in g else False
        tr["foot_uncertain"] = (g["foot_uncertain"].astype(bool).to_numpy()
                                if o.use_foot_noise and "foot_uncertain" in g else False)
        if has_box and image_size is not None:
            w, h = image_size
            e = o.edge_px
            tr["edge"] = ((g["x0"] <= e) | (g["y0"] <= e) | (g["x1"] >= w - e) | (g["y1"] >= h - e)).to_numpy()
        else:
            tr["edge"] = False
        tr.attrs["fps"] = fps
        out[(str(side), int(player))] = tr
    return out


def _velocity(tr: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    vx, vy = np.zeros(len(tr)), np.zeros(len(tr))
    t = tr["t"].to_numpy(float)
    for _, idx in tr.groupby("segment").groups.items():
        k = np.asarray(idx)
        if len(k) >= 2:
            vx[k] = np.gradient(tr["x"].to_numpy(float)[k], t[k])
            vy[k] = np.gradient(tr["y"].to_numpy(float)[k], t[k])
    return vx, vy


def truth_track(t: np.ndarray, xy: np.ndarray, lift: np.ndarray | None = None) -> pd.DataFrame:
    """正解の足元（ならさない）を ``player_tracks`` と同じ形にする（合成データの正解に使う）。"""
    lift = np.zeros(len(t)) if lift is None else lift
    tr = pd.DataFrame({"t": t, "x": xy[:, 0], "y": xy[:, 1], "segment": 0, "lift": lift, "jump_lift": lift,
                       "jump_depth": 0.0})
    tr["vx"], tr["vy"] = _velocity(tr)
    tr["speed"] = tr["peak_speed"] = np.hypot(tr["vx"], tr["vy"])
    tr["airborne"] = tr["interpolated"] = tr["edge"] = False
    tr.attrs["fps"] = 1.0 / float(np.median(np.diff(t)))
    return tr


# =============================================================================
# 1 本ごと
# =============================================================================
def _window(tr: pd.DataFrame, t0: float, t1: float) -> pd.DataFrame:
    return tr[(tr["t"] >= t0) & (tr["t"] <= t1)]


def _at(tr: pd.DataFrame, t: float) -> np.ndarray | None:
    if tr.empty:
        return None
    k = int(np.argmin(np.abs(tr["t"].to_numpy(float) - t)))
    if abs(float(tr["t"].iloc[k]) - t) > 0.2:
        return None
    return tr[["x", "y"]].iloc[k].to_numpy(float)


def _distance(w: pd.DataFrame) -> float:
    total = 0.0
    for _, s in w.groupby("segment"):
        if len(s) > 1:
            total += float(np.hypot(np.diff(s["x"].to_numpy(float)), np.diff(s["y"].to_numpy(float))).sum())
    return total


def _r(v: Any, nd: int = 3) -> float | None:
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


def reaction_time(tr: pd.DataFrame, t_prev: float, t_hit: float, target: np.ndarray,
                  o: FootworkOptions) -> float | None:
    """相手が打ってから、打つ所へ向かって動き出すまで [s]。動き出しが見つからなければ None。"""
    w = _window(tr, t_prev - o.react_lead_s, min(t_prev + o.react_max_s, t_hit))
    if len(w) < 2:
        return None
    d = target - w[["x", "y"]].to_numpy(float)
    dist = np.hypot(*d.T)
    v = w[["vx", "vy"]].to_numpy(float)
    sp = np.hypot(*v.T)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (v * d).sum(axis=1) / (sp * dist)
    ok = (sp >= o.react_mps) & (np.nan_to_num(cos) >= o.toward_cos) & (dist > o.arrive_m)
    n = max(1, int(round(o.react_hold_s * tr.attrs.get("fps", 30.0))))
    t = w["t"].to_numpy(float)
    for k in np.flatnonzero(ok):
        if ok[k:k + n].all() and k + n <= len(ok):
            # 探す範囲の始め（相手が打つ ``react_lead_s`` 前）から動いていたら、前の動きの続きで、動き出しは分からない
            return None if k == 0 else float(t[k] - t_prev)
    return None


def shot_footwork(tr: pd.DataFrame, t_hit: float, t_prev: float | None, t_next: float | None,
                  base: np.ndarray | None, o: FootworkOptions | None = None) -> dict[str, Any]:
    """1 本の、打った選手のフットワーク（列は ``FOOTWORK_COLUMNS`` の数字の所）。

    Args:
        tr: その選手の ``player_tracks``（または ``truth_track``）
        t_prev: 受けた球（相手の 1 本前のショット）を打たれた時刻。サーブなら None
        t_next: 次に相手が打った時刻（なければ None）
        base: そのラリーでいちばんいた所（戻りの基準）
    """
    o = o or FootworkOptions()
    row: dict[str, Any] = {}
    target = _at(tr, t_hit)
    # スプリットステップ
    split = split_dt = split_h = None
    if t_prev is not None:
        w = _window(tr, t_prev + o.split_window_s[0], t_prev + o.split_window_s[1])
        lift = w["lift"].to_numpy(float) if len(w) else np.zeros(0)
        moving = w["speed"].max() > o.split_max_mps if len(w) else False
        if len(lift) and np.isfinite(lift).any() and not moving:
            k = int(np.nanargmax(lift))
            split = bool(o.split_min_m < lift[k] <= o.split_max_m)
            if split:
                split_dt, split_h = float(w["t"].iloc[k]) - t_prev, float(lift[k])
    row.update(split_step=split, split_dt_s=_r(split_dt), split_height_m=_r(split_h))
    # 反応・移動・準備
    t0 = t_prev if t_prev is not None else t_hit - o.no_prev_s
    react = reaction_time(tr, t_prev, t_hit, target, o) if t_prev is not None and target is not None else None
    w = _window(tr, t0, t_hit)
    dist = _distance(w) if len(w) > 1 else None
    peak = float(w["peak_speed"].max()) if len(w) else None
    prep = None
    if target is not None and len(w):
        far = np.hypot(*(w[["x", "y"]].to_numpy(float) - target).T) > o.arrive_m
        prep = t_hit - t0 if not far.any() else t_hit - float(w["t"].to_numpy(float)[np.flatnonzero(far)[-1]])
    row.update(reaction_s=_r(react), move_distance_m=_r(dist, 2), move_peak_mps=_r(peak, 2), prep_s=_r(prep))
    # 戻り
    rec_m = rec_s = None
    if base is not None and target is not None:
        rec_m = float(np.hypot(*(target - base)))
        if rec_m <= o.recover_m:
            rec_s = 0.0
        else:
            w = _window(tr, t_hit, t_next if t_next is not None else t_hit + o.recover_max_s)
            back = np.hypot(*(w[["x", "y"]].to_numpy(float) - base).T) <= o.recover_m if len(w) else np.zeros(0, bool)
            if back.any():
                rec_s = float(w["t"].to_numpy(float)[np.flatnonzero(back)[0]]) - t_hit
    row.update(base_x_m=_r(base[0] if base is not None else None, 2), base_y_m=_r(base[1] if base is not None else None, 2),
               recovery_m=_r(rec_m, 2), recovery_s=_r(rec_s))
    # ジャンプ
    w = _window(tr, t_hit + o.jump_window_s[0], t_hit + o.jump_window_s[1])
    lift = w["jump_lift"].to_numpy(float) if len(w) else np.zeros(0)
    jump_h = None
    if len(lift) and np.isfinite(lift).any():
        k = int(np.nanargmax(lift))
        if float(np.nan_to_num(w["jump_depth"].to_numpy(float)[k])) <= o.jump_max_depth:
            jump_h = float(lift[k])
    air = bool(w["airborne"].any()) if len(w) else False
    jump = None if jump_h is None and not air else bool(air or (jump_h is not None and jump_h > o.jump_min_m))
    row.update(jump=jump, jump_height_m=_r(jump_h, 2))
    return row


def _uncertain(tr: pd.DataFrame, t0: float, t1: float, has_lift: bool, o: FootworkOptions) -> list[str]:
    w = _window(tr, t0, t1)
    fps = tr.attrs.get("fps", 30.0)
    expected = max(1.0, (t1 - t0) * fps + 1)
    why = []
    if len(w) < expected * (1 - o.max_missing):
        why.append("位置が出ていないコマが多い")
    elif len(w) and w["interpolated"].mean() > o.max_interpolated:
        why.append("補ったコマが多い")
    if len(w) and w["edge"].any():
        why.append("体が画面の端で切れている")
    elif len(w) and "foot_uncertain" in w and w["foot_uncertain"].mean() > o.max_foot_uncertain:
        why.append("足元が不確か（ネット・ほかの選手に隠れている）")
    if not has_lift:
        why.append("跳んだ高さが測れない（カメラなし）")
    return why


FOOTWORK_COLUMNS = ["shot_index", "rally_id", "index_in_rally", "t_hit_s", "side", "player", "shot_type",
                    "prev_t_hit_s", "split_step", "split_dt_s", "split_height_m", "reaction_s", "move_distance_m",
                    "move_peak_mps", "prep_s", "base_x_m", "base_y_m", "recovery_m", "recovery_s", "jump",
                    "jump_height_m", "uncertain", "uncertain_reason", "quality", "q_height", "q_speed", "q_depth",
                    "q_prep"]


def _player_of(shot: Shot, tracks: dict[tuple[str, int], pd.DataFrame]) -> tuple[str, int] | None:
    """打った選手。1 側に 1 人ならその人、2 人なら打った時の足元（``hitter_x_m``）か打点に近い方。"""
    mine = [k for k in tracks if k[0] == shot.side]
    if len(mine) <= 1:
        return mine[0] if mine else None
    ref = ((shot.hitter_x_m, shot.hitter_y_m) if shot.hitter_x_m is not None and shot.hitter_y_m is not None
           else (shot.hit_x_m, shot.hit_y_m))
    if ref[0] is None or ref[1] is None:
        return None
    best, d_best = None, math.inf
    for k in mine:
        p = _at(tracks[k], shot.t_hit_s)
        if p is not None and float(np.hypot(p[0] - ref[0], p[1] - ref[1])) < d_best:
            best, d_best = k, float(np.hypot(p[0] - ref[0], p[1] - ref[1]))
    return best


def _neighbours(shots: list[Shot], order: list[int], j: int, o: FootworkOptions) -> tuple[float | None, float | None]:
    """``order[j]`` のショットの、受けた球（相手の 1 本前）と次に相手が打った時刻。"""
    s = shots[order[j]]

    def same(k: int) -> bool:
        u = shots[order[k]]
        if s.rally_id is not None or u.rally_id is not None:
            return u.rally_id == s.rally_id
        return abs(u.t_hit_s - s.t_hit_s) <= o.max_prev_gap_s

    prev = nxt = None
    if j > 0 and same(j - 1) and shots[order[j - 1]].side not in ("", s.side):
        prev = shots[order[j - 1]].t_hit_s
    if j + 1 < len(order) and same(j + 1) and shots[order[j + 1]].side not in ("", s.side):
        nxt = shots[order[j + 1]].t_hit_s
    return prev, nxt


def _base(tr: pd.DataFrame, shots: list[Shot], s: Shot) -> np.ndarray | None:
    """そのラリーで、その選手がいちばんいた所（ラリーの最初から最後のショットまでの足元の中央値）。"""
    if s.rally_id is not None:
        ts = [u.t_hit_s for u in shots if u.rally_id == s.rally_id]
        t0, t1 = min(ts) - 0.5, max(ts) + 0.5
    else:
        t0, t1 = s.t_hit_s - 3.0, s.t_hit_s + 3.0
    w = _window(tr, t0, t1)
    return None if w.empty else np.median(w[["x", "y"]].to_numpy(float), axis=0)


def footwork_table(shots: list[Shot], tracks: dict[tuple[str, int], pd.DataFrame], court: dict[str, Any] | None = None,
                   o: FootworkOptions | None = None) -> pd.DataFrame:
    """ショットごとのフットワークと出来の目安（``FOOTWORK_COLUMNS``。``shot_index`` は ``shots`` の中の番号）。"""
    o = o or FootworkOptions()
    order = sorted(range(len(shots)), key=lambda i: shots[i].t_hit_s)
    rows = []
    for j, i in enumerate(order):
        s = shots[i]
        row: dict[str, Any] = {"shot_index": i, "rally_id": s.rally_id, "index_in_rally": s.index_in_rally,
                               "t_hit_s": round(s.t_hit_s, 3), "side": s.side, "shot_type": s.shot_type}
        key = _player_of(s, tracks) if s.side else None
        prev, nxt = _neighbours(shots, order, j, o)
        row["prev_t_hit_s"] = _r(prev)
        if key is None:
            row.update(player=None, uncertain=True,
                       uncertain_reason="打った側が分からない" if not s.side else "選手の位置がない")
        else:
            tr = tracks[key]
            row["player"] = key[1]
            row.update(shot_footwork(tr, s.t_hit_s, prev, nxt, _base(tr, shots, s), o))
            t0 = (prev if prev is not None else s.t_hit_s - o.no_prev_s) + min(0.0, o.split_window_s[0])
            why = _uncertain(tr, t0, s.t_hit_s + o.jump_window_s[1], bool(np.isfinite(tr["lift"]).any()), o)
            row.update(uncertain=bool(why), uncertain_reason="・".join(why))
        q = quality_terms(s, row, court)
        row.update({f"q_{k}": _r(v, 2) for k, v in q.items()})
        row["quality"] = quality_score(q)
        rows.append(row)
    df = pd.DataFrame(rows, columns=FOOTWORK_COLUMNS)
    return df.sort_values("shot_index").reset_index(drop=True) if len(df) else df


# =============================================================================
# ショットの出来の目安
# =============================================================================
# 項目ごとに、点が 0 になる値と 1 になる値（その間は直線。外は 0 か 1）。種類ごとに使う項目だけを書く。
# どれも「練習で目指すこと」の目安で、試合の勝ち負けとは別。値は一般的な目安で、選手のレベルで変わる。
QUALITY_TERMS: dict[str, dict[str, tuple[float, float]]] = {
    # 打点の高さ [m]：上から打つ球は高い所で、低い球（ドライブ・プッシュ・ネット）はネットより下がる前に打つほど良い
    "height": {"smash": (2.0, 2.8), "clear": (1.9, 2.7), "drop": (1.9, 2.7),
               "drive": (0.8, 1.5), "push": (0.8, 1.5), "net": (0.6, 1.4)},
    # 初速 [km/h]：速さで押す種類だけ
    "speed": {"smash": (150.0, 300.0), "drive": (80.0, 160.0), "push": (50.0, 110.0)},
    # 落ちた所のネットからの深さ [m]：奥へ運ぶ球は深いほど、前へ落とす球は浅いほど良い（負の幅は浅いほど良い）
    "depth": {"clear": (4.0, 6.2), "lob": (4.0, 6.2), "drop": (3.0, 1.5), "net": (2.5, 0.8)},
}
PREP_FULL_S = 0.4   # 打つ所に着いてからこれだけ待てたら準備の点は満点 [s]（どの種類でも使う）
MIN_QUALITY_TERMS = 2  # これより少ない項目しか出ないショットは点を出さない


def _ramp(v: float | None, lo: float, hi: float) -> float | None:
    if v is None or not np.isfinite(v):
        return None
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))


def quality_terms(s: Shot, fw: dict[str, Any], court: dict[str, Any] | None = None) -> dict[str, float | None]:
    """ショットの出来の項目（0〜1。使わない・出ない項目は None）：打点の高さ・初速・落ちた所の深さ・準備の時間。"""
    t = s.shot_type
    out: dict[str, float | None] = {}
    r = QUALITY_TERMS["height"].get(t)
    out["height"] = _ramp(s.hit_z_m, *r) if r else None
    r = QUALITY_TERMS["speed"].get(t)
    out["speed"] = _ramp(s.speed_kmh, *r) if r else None
    r = QUALITY_TERMS["depth"].get(t)
    depth = abs(s.land_x_m) if s.land_x_m is not None and s.side and (s.land_x_m < 0) != (s.side == "h") else None
    out["depth"] = _ramp(depth, *r) if r else None
    prep = fw.get("prep_s")
    out["prep"] = None if prep is None or fw.get("uncertain") else _ramp(prep, 0.0, PREP_FULL_S)
    return out


def quality_score(q: dict[str, float | None]) -> float | None:
    """出た項目の点の平均 × 100（``MIN_QUALITY_TERMS`` より少なければ None）。"""
    v = [x for x in q.values() if x is not None]
    return round(100 * float(np.mean(v)), 1) if len(v) >= MIN_QUALITY_TERMS else None


# =============================================================================
# まとめ・書き出し
# =============================================================================
def _med(s: pd.Series, nd: int = 3) -> float | None:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return None if s.empty else round(float(s.median()), nd)


def _mean(s: pd.Series, nd: int = 2) -> float | None:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return None if s.empty else round(float(s.mean()), nd)


def _brief(g: pd.DataFrame) -> dict[str, Any]:
    ok = g[~g["uncertain"].astype(bool)]
    split = ok["split_step"].dropna().astype(bool)
    smash = ok[ok["shot_type"] == "smash"]
    jumps = smash["jump"].dropna().astype(bool)
    return {
        "n_shots": int(len(g)), "n_certain": int(len(ok)),
        "split_rate": None if split.empty else round(float(split.mean()), 3),
        "n_split_checked": int(len(split)),
        "split_dt_median_s": _med(ok["split_dt_s"]),
        "reaction_median_s": _med(ok["reaction_s"]),
        "move_distance_mean_m": _mean(ok["move_distance_m"]),
        "move_peak_max_mps": None if ok["move_peak_mps"].dropna().empty else round(float(ok["move_peak_mps"].max()), 2),
        "prep_median_s": _med(ok["prep_s"]),
        "recovery_median_s": _med(ok["recovery_s"]),
        "n_smash": int(len(smash)), "n_jump_smash": int(jumps.sum()),
        "jump_height_median_m": _med(smash.loc[smash["jump"].fillna(False).astype(bool), "jump_height_m"], 2),
        "quality_mean": _mean(g["quality"], 1),
    }


def footwork_summary(table: pd.DataFrame, o: FootworkOptions | None = None) -> dict[str, Any]:
    """選手ごと（``h0``・``o0`` など）とラリーごとのまとめ。数字は確かな行だけ（出来の目安の平均はすべての行）。"""
    o = o or FootworkOptions()
    res: dict[str, Any] = {"n_shots": int(len(table)), "options": o.to_dict(), "players": {}, "rallies": []}
    if table.empty:
        return res
    t = table.dropna(subset=["player"])
    for (side, player), g in t.groupby(["side", "player"]):
        res["players"][f"{side}{int(player)}"] = {"side": side, "player": int(player), **_brief(g)}
    for rid, g in t.dropna(subset=["rally_id"]).groupby("rally_id"):
        res["rallies"].append({"rally_id": int(rid), "players": {
            f"{side}{int(p)}": _brief(gg) for (side, p), gg in g.groupby(["side", "player"])}})
    return res


def _csv_value(v: Any) -> Any:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ""
    return v


def write_footwork(out_dir: str | Path, shots: list[Shot], players: pd.DataFrame, camera: Camera | None = None,
                   court: dict[str, Any] | None = None, image_size: tuple[int, int] | None = None,
                   o: FootworkOptions | None = None) -> dict[str, Any]:
    """``footwork.csv``・``footwork.json`` を書き、まとめを返す。"""
    o = o or FootworkOptions()
    out_dir = Path(out_dir)
    table = footwork_table(shots, player_tracks(players, camera, image_size, o), court, o)
    table.astype(object).map(_csv_value).to_csv(out_dir / FOOTWORK_CSV, index=False, encoding="utf-8")
    summary = footwork_summary(table, o)
    (out_dir / FOOTWORK_FILE).write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary


def read_footwork(out_dir: str | Path) -> pd.DataFrame | None:
    p = Path(out_dir) / FOOTWORK_CSV
    return pd.read_csv(p) if p.exists() else None


def footwork_brief(summary: dict[str, Any]) -> dict[str, Any]:
    """``shot_list.json`` に入れる短いまとめ（選手ごと）。"""
    keys = ("n_shots", "n_certain", "split_rate", "reaction_median_s", "prep_median_s", "recovery_median_s",
            "n_smash", "n_jump_smash", "quality_mean")
    return {k: {kk: v.get(kk) for kk in keys} for k, v in summary.get("players", {}).items()}


# =============================================================================
# 合成データでの確かめ
# =============================================================================
def synthetic_benchmark(n_sessions: int = 3, n_rallies: int = 5, fps: float = 30.0, seed: int = 0,
                        camera: Camera | None = None, o: FootworkOptions | None = None) -> dict[str, Any]:
    """合成ラリー（``synth_rally_players``）に、スプリットステップ・スマッシュのジャンプ（``session_hops``）を入れて写し、
    正解のショット（打った時刻・打点・着地点は正解）で測った値を、正解の足元で同じ手順で測った値と比べる。

    Returns:
        スプリットステップの再現率・適合率、その時刻のずれ、ジャンプの再現率・適合率と高さのずれ、
        反応の速さ・移動距離・準備・戻りの時間の、正解との差の中央値・90% 点、確かでないとした行の割合
    """
    from smash_speed.config import load_config
    from smash_speed.physics import PhysicsParams
    from smash_speed.players import ground_jumps
    from smash_speed.synth_players import observe_players
    from smash_speed.synth_rally import rally_camera
    from smash_speed.synth_rally_players import hop_lift, make_player_session, session_feet, session_hops

    o = o or FootworkOptions()
    cfg = load_config()
    phys = PhysicsParams.from_config(cfg)
    camera = camera or rally_camera(cfg)
    pairs: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    n_uncertain = n_rows = 0
    for k in range(n_sessions):
        rng = np.random.default_rng([seed, 31, k])
        session, _ = make_player_session(rng, phys, n_rallies, p_pickup=0.0, p_neighbor=0.0)
        shots: list[Shot] = []
        for i, r in enumerate(session, start=1):
            shots += r.to_shots(rally_id=i, phys=phys)
        t = np.arange(0.0, session[-1].t_end + 2.0, 1.0 / fps)
        feet = session_feet(session, [], t, rng, cfg["court"])
        hops = session_hops(session, rng)
        lift = hop_lift(hops, t)
        obs = ground_jumps(observe_players(t, feet, camera, rng, lift=lift), camera)
        table = footwork_table(shots, player_tracks(obs, camera, o=o), cfg["court"], o)
        truth = footwork_table(shots, {(s, 0): truth_track(t, feet[s], lift[s]) for s in ("h", "o")}, cfg["court"], o)
        n_rows += len(table)
        n_uncertain += int(table["uncertain"].astype(bool).sum())
        # 正解のスプリットステップ・ジャンプ（ショットの通しの番号 → 跳んだか）
        split_true = {hp.seg_index: hp for hp in hops if hp.kind == "split"}
        jump_true = {hp.seg_index: hp for hp in hops if hp.kind == "jump"}
        order = sorted(range(len(shots)), key=lambda i: shots[i].t_hit_s)
        pos = {i: j for j, i in enumerate(order)}
        for a, b in zip(table.to_dict("records"), truth.to_dict("records")):
            j = pos[a["shot_index"]]
            info = {"split": split_true.get(j - 1) if a["prev_t_hit_s"] is not None else None,
                    "jump": jump_true.get(j), "has_prev": a["prev_t_hit_s"] is not None,
                    "shot_type": a["shot_type"]}
            pairs.append((a, b, info))
    return _score(pairs, n_rows, n_uncertain)


def _score(pairs: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]], n_rows: int, n_uncertain: int
           ) -> dict[str, Any]:
    tp = fp = fn = 0
    dt_err, jh_err = [], []
    jtp = jfp = jfn = 0
    diffs: dict[str, list[float]] = {"reaction_s": [], "move_distance_m": [], "prep_s": [], "recovery_s": []}
    for a, b, info in pairs:
        if a["uncertain"]:
            continue
        if info["has_prev"] and a["split_step"] is not None:
            hp = info["split"]
            if a["split_step"] and hp is not None:
                tp += 1
                dt_err.append(abs(a["split_dt_s"] - b["split_dt_s"]) if b["split_dt_s"] is not None else np.nan)
            elif a["split_step"]:
                fp += 1
            elif hp is not None:
                fn += 1
        if info["shot_type"] == "smash" and a["jump"] is not None:
            hp = info["jump"]
            if a["jump"] and hp is not None:
                jtp += 1
                if a["jump_height_m"] is not None:
                    jh_err.append(a["jump_height_m"] - hp.height_m)
            elif a["jump"]:
                jfp += 1
            elif hp is not None:
                jfn += 1
        for c in diffs:
            if a[c] is not None and b[c] is not None and np.isfinite([a[c], b[c]]).all():
                diffs[c].append(float(a[c]) - float(b[c]))

    def q(v: list[float]) -> dict[str, Any]:
        x = np.abs(np.asarray(v, float))
        x = x[np.isfinite(x)]
        return {"n": int(len(x)), "median_abs": _r(np.median(x)) if len(x) else None,
                "p90_abs": _r(np.percentile(x, 90)) if len(x) else None,
                "bias": _r(float(np.nanmean(v))) if len(v) else None}

    return {
        "n_rows": n_rows, "uncertain_share": round(n_uncertain / max(1, n_rows), 3),
        "split": {"recall": round(tp / max(1, tp + fn), 3), "precision": round(tp / max(1, tp + fp), 3),
                  "tp": tp, "fp": fp, "fn": fn, "dt": q(dt_err)},
        "jump": {"recall": round(jtp / max(1, jtp + jfn), 3), "precision": round(jtp / max(1, jtp + jfp), 3),
                 "tp": jtp, "fp": jfp, "fn": jfn, "height": q(jh_err)},
        **{c: q(v) for c, v in diffs.items()},
    }


def main(argv: list[str] | None = None) -> None:
    import argparse

    from smash_speed.calibration.io import load_calibration
    from smash_speed.config import load_config
    from smash_speed.players import PLAYERS_FILE
    from smash_speed.shots import SHOTS_FILE, read_shots

    ap = argparse.ArgumentParser(description="ショットごとのフットワークと出来の目安を footwork.csv・footwork.json に書く")
    ap.add_argument("out_dir", type=Path, nargs="?", help="ショットの一覧の出力先（shots.csv・players.csv があるところ）")
    ap.add_argument("--calib", type=Path, default=None, help="calibration.json（省くと shot_list.json のもの）")
    ap.add_argument("--bench", action="store_true", help="合成データで正解と比べる")
    ap.add_argument("--config", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.bench:
        print(json.dumps(synthetic_benchmark(), ensure_ascii=False, indent=1))
        return
    if a.out_dir is None:
        ap.error("出力先を指定してください")
    cfg = load_config(a.config)
    info_path = a.out_dir / "shot_list.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    calib = a.calib or (Path(info["calib"]) if info.get("calib") else None)
    camera = load_calibration(calib)[0] if calib is not None and Path(calib).exists() else None
    size = (int(info["width"]), int(info["height"])) if info.get("width") and info.get("height") else None
    summary = write_footwork(a.out_dir, read_shots(a.out_dir / SHOTS_FILE), pd.read_csv(a.out_dir / PLAYERS_FILE),
                             camera, cfg.get("court"), size)
    for k, p in summary["players"].items():
        print(k, json.dumps(p, ensure_ascii=False))


if __name__ == "__main__":
    import sys

    if (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    main()
