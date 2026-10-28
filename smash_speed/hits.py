"""動画の中のすべての打球（打った時刻と打った側）を見つける。

2 つの手がかりを組み合わせる。

- 軌道の向きが変わる所（``track_breaks``）：シャトルの検出結果の点の列で、ある時刻の前の数点と後の数点に
  それぞれ 2 次式を当てはめ、その時刻での速度（画像上 [px/s]）の差が、検出位置のばらつきと速さから見て
  大きすぎる所を打球とする。打球のあと止まった（床に落ちた）所は除く。検出が長く途切れた所の前後で
  向きが変わっていれば、その間にも打球を置く。検出が始まった所（サーブ・画面に入った所）と、画面の端から
  出て戻ってきた所は弱い手がかり（音で確かめられたときだけ打球にする。音のない動画では検出の始まりだけ使う）。
- 打球音（``events.audio_onsets``）：時刻は正確だが、隣のコートの音・足音も拾う。

組み合わせ方（``find_hits``）：

- 軌道の変化の近く（``match_s`` 以内）に打球音があれば、時刻は打球音にする（いちばん確か）。
  音が届くまでの遅れは、音と軌道の両方がある打球の時刻の差の中央値から見積もって引く。
- 打球音のない軌道の変化は、変化が十分大きければ打球にする（音のない動画・音が埋もれた打球）。
- 軌道の変化と組にならなかった打球音は、近くにしきい値を 0.8 倍にして探した小さな変化があれば打球にする
  （遅いネット前の打ち合い）。それもなく、前後でシャトルが見えていて向きが変わっていなければ捨てる
  （隣のコートの音など）。シャトルが見えていなければ、ラリーの途中（前後 1 秒以内にシャトルが見えている）の
  強い音だけ打球にする（信頼度は低い）。

打った側は、打球から次の打球までにシャトルが画像上で動いた向きを、コートの x 軸（キャリブレーションの
カメラで投影した向き）と比べた票と、「1 ラリーの中ではふつう交互」という決まりから、いちばんありそうな並びを選ぶ。
カメラがなければ打った側は空欄。

``python -m smash_speed.hits --bench`` で、合成ラリー（``synth_rally``）での再現率・適合率を出す。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from smash_speed.camera import Camera
from smash_speed.shots import Shot, make_rallies


@dataclass
class Hit:
    """見つけた打球 1 つ。

    Attributes:
        t: 打った時刻 [s]（映像の時刻）
        side: 打った側 h | o | ""（分からない）
        method: 手がかり audio+track | track | track_gap | track_start | audio（``TrackBreak.kind`` を参照）
        score: 軌道の変化の大きさ（しきい値の何倍か。軌道の手がかりがなければ 0）
        audio_strength: 打球音の強さ（雑音比。音がなければ 0）
        confidence: 信頼度 0〜1
        t_track: 軌道から求めた時刻 [s]（なければ None）
        t_audio: 打球音の時刻 [s]（映像の時刻。音が届くまでの遅れを引く前。打球音を使っていなければ None）
    """

    t: float
    side: str = ""
    method: str = ""
    score: float = 0.0
    audio_strength: float = 0.0
    confidence: float = 0.0
    t_track: float | None = None
    t_audio: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrackBreak:
    """軌道の向きが変わった所（打球の候補）。"""

    t: float            # 前後の 2 次式が最も近づく時刻 [s]
    score: float        # 速度の差 ÷ しきい値
    kind: str           # track（見えている所）| track_gap（途切れた所）| track_start（検出の始まり）
                        # | track_edge（画面の端から出入りした所）
    v_before: np.ndarray | None = None
    v_after: np.ndarray | None = None


# =============================================================================
# 軌道の向きが変わる所
# =============================================================================
def _visible_points(det: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    d = det[(det["visible"] > 0) & det["x_px"].notna() & det["y_px"].notna()]
    return d["timestamp"].to_numpy(float), d[["x_px", "y_px"]].to_numpy(float)


def _split_tracklets(t: np.ndarray, max_gap_s: float) -> list[np.ndarray]:
    """検出が ``max_gap_s`` より長く途切れた所で分けた、点の番号の列。"""
    if not len(t):
        return []
    cuts = np.flatnonzero(np.diff(t) > max_gap_s) + 1
    return np.split(np.arange(len(t)), cuts)


def _fit(t: np.ndarray, uv: np.ndarray, t_ref: float) -> np.ndarray:
    """``t - t_ref`` の多項式（4 点以上で 2 次、それより少なければ 1 次）の係数 (2, deg+1)。"""
    deg = 2 if len(t) >= 4 else 1
    return np.array([np.polyfit(t - t_ref, uv[:, j], deg) for j in range(2)])


def _pos(c: np.ndarray, dt: float | np.ndarray) -> np.ndarray:
    return np.stack([np.polyval(cj, dt) for cj in c], axis=-1)


def _vel(c: np.ndarray, dt: float) -> np.ndarray:
    return np.array([np.polyval(np.polyder(cj), dt) for cj in c])


def _vel_sigma(t: np.ndarray, t_eval: float, noise_px: float) -> float:
    """``_fit`` で求めた速度の、``t_eval`` での標準偏差 [px/s]（位置の誤差 ``noise_px`` の白色雑音として）。"""
    deg = 2 if len(t) >= 4 else 1
    s = t - t_eval
    A = np.vander(s, deg + 1, increasing=True)
    cov = np.linalg.pinv(A.T @ A)
    return float(noise_px * np.sqrt(max(cov[1, 1], 0.0)))


def _crossing_time(cb: np.ndarray, ca: np.ndarray, ref_b: float, ref_a: float, lo: float, hi: float) -> float:
    """前の式（時刻の原点 ``ref_b``）と後の式（``ref_a``）が最も近づく時刻 [s]（``lo``〜``hi`` の間）。"""
    tt = np.linspace(lo, hi, 101)
    gap = np.linalg.norm(_pos(cb, tt - ref_b) - _pos(ca, tt - ref_a), axis=1)
    return float(tt[int(np.argmin(gap))])


def estimate_noise_px(t: np.ndarray, uv: np.ndarray, floor_px: float = 0.5) -> float:
    """検出位置のばらつき [px]：連続する点の 2 階差分の大きさの中央値から（打球の所の外れ値に強い）。"""
    if len(t) < 5:
        return 2.0
    # 検出漏れで間隔が違う所も混ぜてしまうが、中央値なのでほとんど影響しない
    d2 = uv[2:] - 2 * uv[1:-1] + uv[:-2]
    # 2 階差分の各成分の標準偏差は sqrt(6)·σ。中央値 / 0.6745 で標準偏差にする
    s = np.median(np.abs(d2), axis=0) / 0.6745 / np.sqrt(6)
    return float(max(floor_px, np.mean(s)))


def track_breaks(det: pd.DataFrame, fps: float, window_s: float = 0.1, min_points: int = 3,
                 min_window_points: int = 3, max_points: int = 30, max_gap_s: float = 0.25,
                 k_sigma: float = 4.0, rel: float = 0.5, still_px_s: float = 60.0, min_interval_s: float = 0.25,
                 noise_px: float | None = None, frame_size: tuple[int, int] | None = None,
                 edge_margin: float = 0.02, threshold_scale: float = 1.0) -> list[TrackBreak]:
    """シャトルの検出結果から、軌道の向きが変わった所（打球の候補）を見つける。

    Args:
        det: 検出結果（列 ``timestamp, x_px, y_px, visible``）
        fps: フレームレート [Hz]
        window_s: 前後それぞれに使う時間の長さ [s]（点の数は ``min_window_points``〜``max_points``）
        min_points: 打球の前後それぞれに要る点の数
        max_gap_s: 検出がこれより長く途切れたら、軌道を分ける [s]
        k_sigma: しきい値（速度の差が、速度の誤差の何倍を超えたら打球か）
        rel: しきい値（速度の差が、前後の速さの大きい方の何倍を超えたら打球か）
        still_px_s: 打球のあとの速さがこれより遅ければ、止まった（床に落ちた）とみなして除く [px/s]
        min_interval_s: これより近い候補は、変化の大きい方だけ残す [s]
        noise_px: 検出位置のばらつき [px]（省略時は検出結果から見積もる）
        frame_size: 画像の大きさ (幅, 高さ) [px]。与えると、画面の端から出入りした所（高いロブなど）は
            打球ではなく ``track_edge``（音で確かめられたときだけ打球にする弱い手がかり）にする
        edge_margin: 画面の端とみなす幅（画像の長い辺に対する割合）
        threshold_scale: しきい値に掛ける値（打球音の近くで小さな変化を探すときは 1 より小さくする）

    Returns:
        時刻の順の ``TrackBreak``。score はしきい値の何倍か（1 以上）
    """
    t, uv = _visible_points(det)
    if noise_px is None:
        noise_px = estimate_noise_px(t, uv)
    # 低 fps でも前後それぞれ ``min_window_points`` 点以上（2 次式）。1 次式にすると、減速の速い打球の直後を
    # 打球と間違える。点が少ない 2 次式は速度のばらつきが大きく、遅いネットの打球を見落とす
    k = int(np.clip(round(window_s * fps), max(min_points, min_window_points), max_points))
    out: list[TrackBreak] = []
    tracks = _split_tracklets(t, max_gap_s)

    def at_edge(idx: np.ndarray, forward: bool) -> bool:
        """軌道の終わり（forward）・始まりを 2 コマ先（前）まで延ばすと、画面の端に掛かるか。"""
        if frame_size is None:
            return False
        sel = idx[-k:] if forward else idx[:k]
        t_ref = t[sel[-1]] if forward else t[sel[0]]
        c = _fit(t[sel], uv[sel], t_ref)
        p = _pos(c, (2.0 if forward else -2.0) / fps)
        m = edge_margin * max(frame_size)
        return bool(p[0] < m or p[0] > frame_size[0] - m or p[1] < m or p[1] > frame_size[1] - m)

    for n_tr, idx in enumerate(tracks):
        tt, pp = t[idx], uv[idx]
        if len(tt) < min_points:
            continue
        # 検出の始まり（前の軌道から長く途切れたか、最初の軌道）
        if n_tr == 0 or tt[0] - t[tracks[n_tr - 1][-1]] > max_gap_s:
            ca = _fit(tt[:k], pp[:k], tt[0])
            kind = "track_edge" if at_edge(idx, forward=False) else "track_start"
            out.append(TrackBreak(float(tt[0]), 1.0, kind, None, _vel(ca, 0.0)))
        cand: list[tuple[TrackBreak, int]] = []
        for j in range(min_points - 1, len(tt) - min_points):
            b = slice(max(0, j + 1 - k), j + 1)
            a = slice(j + 1, min(len(tt), j + 1 + k))
            tm = 0.5 * (tt[j] + tt[j + 1])
            cb, ca = _fit(tt[b], pp[b], tm), _fit(tt[a], pp[a], tm)
            vb, va = _vel(cb, 0.0), _vel(ca, 0.0)
            jump = float(np.linalg.norm(va - vb))
            sig = np.hypot(_vel_sigma(tt[b], tm, noise_px), _vel_sigma(tt[a], tm, noise_px))
            thr = threshold_scale * max(k_sigma * sig, rel * max(np.linalg.norm(va), np.linalg.norm(vb)))
            if jump <= thr or np.linalg.norm(va) < still_px_s:
                continue
            cand.append((TrackBreak(float(tm), jump / thr, "track", vb, va), j))
        kept = _keep_strongest([c for c, _ in cand], min_interval_s)
        for c, j in cand:
            if any(c is kc for kc in kept):
                c.t = _refine_split(tt, pp, j, k, min_points)
        # 時刻を決め直すと近づくことがあるので、もう一度間引く
        out += _keep_strongest(kept, min_interval_s)
    # 途切れた所：前の軌道の終わりと次の軌道の始まりで向きが変わっていれば、その間に打球があった
    for prev, nxt in zip(tracks[:-1], tracks[1:]):
        if len(prev) < min_points or len(nxt) < min_points:
            continue
        t0, t1 = t[prev[-1]], t[nxt[0]]
        if t1 - t0 > 4 * max_gap_s:
            continue
        pb, pa = prev[-k:], nxt[:k]
        cb, ca = _fit(t[pb], uv[pb], t0), _fit(t[pa], uv[pa], t1)
        vb, va = _vel(cb, 0.0), _vel(ca, 0.0)
        cos = float(vb @ va / (np.linalg.norm(vb) * np.linalg.norm(va) + 1e-9))
        # 画面の外に出て戻ってきた所は、向きが変わって見えても打球とは限らない（弱い手がかりにする）
        edge = at_edge(prev, forward=True) and at_edge(nxt, forward=False)
        if (cos < 0.0 or edge) and np.linalg.norm(va) >= still_px_s:
            tc = _crossing_time(cb, ca, t0, t1, t0, t1)
            # 「検出の始まり」として入れた所を、途切れた所の打球に置き換える
            out = [b for b in out if not (b.kind in ("track_start", "track_edge") and abs(b.t - t1) < 1e-9)]
            out.append(TrackBreak(tc, 1.0 - cos, "track_edge" if edge else "track_gap", vb, va))
    return sorted(out, key=lambda b: b.t)


def _refine_split(tt: np.ndarray, pp: np.ndarray, j: int, k: int, min_points: int, extra: int = 3) -> float:
    """候補 ``j``（点 j と j+1 の間）の前後 ``k + extra`` 点の範囲を 2 つに分けて 2 次式を当てはめ、
    残差がいちばん小さい分け目の間で、前後の式が最も近づく時刻を打球の時刻にする
    （速度の差が最大の所は、窓が長いと打球から少しずれる）。"""
    lo, hi = max(0, j - k - extra), min(len(tt), j + k + extra + 2)
    best, best_s = np.inf, j
    for s in range(lo + min_points - 1, hi - min_points):
        sse, dof = 0.0, 0
        for sl in (slice(lo, s + 1), slice(s + 1, hi)):
            c = _fit(tt[sl], pp[sl], tt[s])
            sse += float(np.sum((_pos(c, tt[sl] - tt[s]) - pp[sl]) ** 2))
            dof += 2 * (sl.stop - sl.start - c.shape[1])
        sse /= max(dof, 1)  # 自由度で割った残差の分散（式の次数が違う分け目どうしを比べる）
        if sse < best:
            best, best_s = sse, s
    s = best_s
    b, a = slice(max(lo, s + 1 - k), s + 1), slice(s + 1, min(hi, s + 1 + k))
    cb, ca = _fit(tt[b], pp[b], tt[s]), _fit(tt[a], pp[a], tt[s])
    return _crossing_time(cb, ca, tt[s], tt[s], tt[s], tt[s + 1])


def _keep_strongest(cands: list[TrackBreak], min_interval_s: float) -> list[TrackBreak]:
    """``min_interval_s`` 以内に並んだ候補から、変化の大きいものだけを残す。"""
    keep: list[TrackBreak] = []
    for c in sorted(cands, key=lambda c: -c.score):
        if all(abs(c.t - k.t) >= min_interval_s for k in keep):
            keep.append(c)
    return sorted(keep, key=lambda c: c.t)


# =============================================================================
# 打球音と組み合わせる
# =============================================================================
def find_hits(det: pd.DataFrame | None, fps: float, onsets: list[tuple[float, float]] | None = None,
              camera: Camera | None = None, av_offset_s: float = 0.0, match_s: float = 0.08,
              min_track_score: float = 1.2, min_audio_strength: float = 30.0, quiet_window_s: float = 0.1,
              rally_window_s: float = 1.0,
              rally_gap_s: float = 3.0, min_lag_pairs: int = 3, lag_tol_s: float = 0.03, weak_scale: float = 0.8,
              **break_kw) -> list[Hit]:
    """打球の時刻と打った側を出す。

    Args:
        det: シャトルの検出結果（列 ``timestamp, x_px, y_px, visible``）。なければ打球音だけで決める
        fps: フレームレート [Hz]
        onsets: ``audio_onsets`` の結果（音声の時刻 [s]、強さ）。なければ軌道だけで決める
        camera: キャリブレーションのカメラ（打った側を決めるのに使う。なければ側は空欄）
        av_offset_s: 映像の時刻 = 音声の時刻 + av_offset_s
        match_s: 打球音と軌道の変化を同じ打球とみなす時刻の差 [s]（音が届くまでの遅れを含む）
        min_track_score: 打球音のない軌道の変化を打球にする、変化の大きさ（しきい値の何倍か）
        min_audio_strength: シャトルが見えていない所の打球音を打球にする、音の強さ（雑音比）
        quiet_window_s: 打球音の前後これだけの間シャトルが見えていて変化がなければ、その音を捨てる [s]
        rally_window_s: 軌道の変化のない打球音は、前後それぞれこれだけの間にシャトルが見えていなければ捨てる [s]
        rally_gap_s: 打球の間がこれより空いたら別のラリー（打った側を交互にそろえる単位）[s]
        min_lag_pairs: 音の遅れを見積もるのに要る、音と軌道の両方がある打球の数（足りなければ遅れは 0）
        lag_tol_s: 遅れを引いた音の時刻と軌道の時刻がこれより離れていたら、軌道の時刻を使う [s]
        weak_scale: 軌道の変化と組にならなかった打球音の近くでは、しきい値をこの倍まで下げて変化を探す
        **break_kw: ``track_breaks`` に渡す

    Returns:
        時刻の順の ``Hit``
    """
    if camera is not None:
        break_kw.setdefault("frame_size", (camera.width, camera.height))
    breaks = track_breaks(det, fps, **break_kw) if det is not None else []
    audio = [(t + av_offset_s, s) for t, s in (onsets or [])]
    t_vis = _visible_points(det)[0] if det is not None else np.zeros(0)
    used: set[int] = set()
    hits: list[Hit] = []
    for b in breaks:
        near = [i for i, (ta, _) in enumerate(audio) if i not in used and abs(ta - b.t) <= match_s]
        if near:
            i = max(near, key=lambda i: (audio[i][1], -abs(audio[i][0] - b.t)))
            used.add(i)
            ta, s = audio[i]
            conf = 0.95 if b.kind in ("track", "track_gap") else 0.8
            hits.append(Hit(ta, method="audio+track", score=b.score, audio_strength=s, confidence=conf, t_track=b.t,
                            t_audio=ta))
            continue
        if b.kind == "track_edge":
            continue  # 画面の端から出入りしただけかもしれないので、音で確かめられたときだけ
        if b.kind == "track_start":
            # 音で確かめられない検出の始まりは、音のない動画のときだけ打球にする（画面に入っただけかもしれない）
            if not audio:
                hits.append(Hit(b.t, method=b.kind, score=b.score, confidence=0.4, t_track=b.t))
            continue
        if b.kind == "track_gap" or b.score >= min_track_score:
            conf = float(np.clip(0.5 + 0.1 * b.score, 0.5, 0.85)) if b.kind == "track" else 0.5
            hits.append(Hit(b.t, method=b.kind, score=b.score, confidence=conf, t_track=b.t))
    weak: list[TrackBreak] | None = None
    for i, (ta, s) in enumerate(audio):
        if i in used:
            continue
        if any(abs(h.t - ta) <= match_s for h in hits):
            continue
        if det is not None:
            # 音の近くに小さな向きの変化があれば打球（遅いネット前の打ち合いは、音がないと見落としやすい）
            if weak is None:
                weak = [b for b in track_breaks(det, fps, threshold_scale=weak_scale, **break_kw)
                        if b.kind == "track"]
            near = [b for b in weak if abs(b.t - ta) <= match_s and not any(abs(h.t - ta) <= match_s for h in hits)]
            if near:
                b = max(near, key=lambda b: b.score)
                hits.append(Hit(ta, method="audio+track", score=b.score * weak_scale, audio_strength=s,
                                confidence=0.8, t_track=b.t, t_audio=ta))
                continue
            # 前後でシャトルが見えていたのに向きが変わっていない → 打球ではない
            seen_before = np.any((t_vis >= ta - quiet_window_s) & (t_vis < ta - 0.02))
            seen_after = np.any((t_vis > ta + 0.02) & (t_vis <= ta + quiet_window_s))
            if seen_before and seen_after:
                continue
            # ラリーの外（前にも後にもしばらくシャトルが見えていない）の音は、打球ではない
            if not (np.any((t_vis >= ta - rally_window_s) & (t_vis < ta))
                    and np.any((t_vis > ta) & (t_vis <= ta + rally_window_s))):
                continue
        if s < min_audio_strength:
            continue
        hits.append(Hit(ta, method="audio", audio_strength=s, confidence=0.4 if det is not None else 0.6, t_audio=ta))
    # 音はシャトルからカメラ（マイク）まで届くのに遅れる（15 m で約 44 ms）。音と軌道の両方がある打球の
    # 時刻の差の中央値を遅れとして、音から決めた時刻から引く
    pairs = [h.t - h.t_track for h in hits if h.method == "audio+track" and h.t_track is not None
             and h.score >= 1.0 and abs(h.t - h.t_track) <= match_s]
    if len(pairs) >= min_lag_pairs:
        lag = float(np.median(pairs))
        for h in hits:
            if h.method.startswith("audio"):
                h.t -= lag
            # 音の時刻が軌道の時刻と合わない（打球の直前の別の音とまとめて拾った、など）なら軌道の時刻にする
            if h.method == "audio+track" and h.score >= 1.0 and abs(h.t - h.t_track) > lag_tol_s:
                h.t, h.method, h.t_audio = h.t_track, "track", None
    hits.sort(key=lambda h: h.t)
    if camera is not None and det is not None:
        assign_sides(hits, det, camera, rally_gap_s)
    return hits


# =============================================================================
# 打った側
# =============================================================================
def court_x_direction(camera: Camera, height_m: float = 1.5) -> np.ndarray:
    """コートの +x の向き（h 側 → o 側）を画像に投影した単位ベクトル（ネットの上 ``height_m`` で測る）。"""
    a, b = camera.project(np.array([[-1.0, 0.0, height_m], [1.0, 0.0, height_m]]))
    d = b - a
    return d / (np.linalg.norm(d) + 1e-12)


def side_votes(hits: list[Hit], det: pd.DataFrame, camera: Camera) -> list[float]:
    """打球ごとの「h 側が打った」票（+）・「o 側が打った」票（−）。打球から次の打球までの画像上の動きの向きから。"""
    t, uv = _visible_points(det)
    ex = court_x_direction(camera)
    votes = []
    for i, h in enumerate(hits):
        t_next = hits[i + 1].t if i + 1 < len(hits) else np.inf
        m = (t > h.t + 0.01) & (t < t_next - 0.01)
        if m.sum() < 2:
            votes.append(0.0)
            continue
        d = uv[m][-1] - uv[m][0]
        # h 側が打てば +x の向きに飛ぶ。動いた量 [px] で重みを付け、1 本で大きくなりすぎないよう抑える
        votes.append(float(np.tanh((d @ ex) / 200.0)))
    return votes


def assign_sides(hits: list[Hit], det: pd.DataFrame, camera: Camera, rally_gap_s: float = 3.0,
                 p_alternate: float = 0.9) -> None:
    """``hits`` の打った側を決める。

    1 ラリーの中では打つ側はふつう交互（打球の見落とし・見つけすぎがあると同じ側が続く）なので、
    「交互になる確率 ``p_alternate``」と各打球の票から、いちばんありそうな並びを動的計画法で選ぶ。
    """
    votes = side_votes(hits, det, camera)
    start = 0
    for i in range(1, len(hits) + 1):
        if i < len(hits) and hits[i].t - hits[i - 1].t <= rally_gap_s:
            continue
        seq = _most_likely_sides(votes[start:i], p_alternate)
        for k, side in zip(range(start, i), seq):
            hits[k].side = side
        start = i


def _most_likely_sides(votes: list[float], p_alternate: float) -> list[str]:
    """票（+ なら h、− なら o。大きさは確かさ）の列から、いちばんありそうな側の並び（ビタビ）。"""
    if not any(votes):
        return [""] * len(votes)
    # 状態 0 = h、1 = o。票 v から「h が打った」確率を 0.5 + 0.45 v とする
    emit = np.log(np.array([[0.5 + 0.45 * v, 0.5 - 0.45 * v] for v in votes]))
    trans = np.log(np.array([[1 - p_alternate, p_alternate], [p_alternate, 1 - p_alternate]]))
    score = emit[0].copy()
    back = np.zeros((len(votes), 2), int)
    for n in range(1, len(votes)):
        cand = score[:, None] + trans      # 前の状態 × 今の状態
        back[n] = np.argmax(cand, axis=0)
        score = cand[back[n], [0, 1]] + emit[n]
    st = int(np.argmax(score))
    path = [st]
    for n in range(len(votes) - 1, 0, -1):
        st = int(back[n, st])
        path.append(st)
    return ["ho"[p] for p in reversed(path)]


def hits_to_shots(hits: list[Hit], rally_gap_s: float = 3.0, source: str = "hits") -> list[Shot]:
    """見つけた打球を ``shots.csv`` の形（``Shot``。種類・初速はまだ分からない）にする。

    ラリーは打球の間隔で仮に分ける（``shots.make_rallies``。着地も使う区切りは、軌道を当てはめた後の
    ``rallies.split_rallies``）。
    """
    shots = [Shot(t_hit_s=h.t, side=h.side, confidence=round(h.confidence, 3), source=source, note=h.method)
             for h in hits]
    make_rallies(shots, max_gap_s=rally_gap_s)
    return shots


# =============================================================================
# 正解との比べ方と、合成ラリーでの測り方
# =============================================================================
def match_hits(pred: list[Hit], true_t: np.ndarray, true_side: list[str] | None = None,
               tol_s: float = 0.05) -> dict[str, Any]:
    """見つけた打球と正解の打球を、時刻の差 ``tol_s`` 以内で 1 対 1 に組み合わせて数える。

    Returns:
        ``tp, fp, fn, recall, precision, mean_abs_dt_s, side_correct``（側の正解がなければ side_correct は None）
    """
    true_t = np.asarray(true_t, float)
    pairs = sorted(((abs(p.t - tt), i, j) for i, p in enumerate(pred) for j, tt in enumerate(true_t)
                    if abs(p.t - tt) <= tol_s))
    used_p, used_t, matched = set(), set(), []
    for d, i, j in pairs:
        if i in used_p or j in used_t:
            continue
        used_p.add(i)
        used_t.add(j)
        matched.append((i, j, d))
    tp = len(matched)
    side_ok = None
    if true_side is not None:
        side_ok = sum(pred[i].side == true_side[j] for i, j, _ in matched)
    return {"tp": tp, "fp": len(pred) - tp, "fn": len(true_t) - tp,
            "recall": tp / len(true_t) if len(true_t) else 1.0,
            "precision": tp / len(pred) if pred else 1.0,
            "mean_abs_dt_s": float(np.mean([d for *_, d in matched])) if matched else None,
            "side_correct": side_ok}


def synthetic_benchmark(n_rallies: int = 20, fps: float = 120.0, seed: int = 0, use_audio: bool = True,
                        n_false_clicks: int = 3, noise_px: float = 1.5, dropout: float = 0.1,
                        camera: Camera | None = None, tol_s: float = 0.05, cfg: dict[str, Any] | None = None,
                        verbose: bool = False, **find_kw) -> dict[str, Any]:
    """合成ラリーで ``find_hits`` の再現率・適合率・打った側の正解率を測る。

    観測は ``synth_rally.observe_rally``（検出位置の雑音 ``noise_px``・検出漏れ ``dropout``）、
    打球音は打つたびのクリック音（打点からカメラまでの音の遅れつき）に、打球ではない音（隣のコートなど）を
    1 ラリーあたり ``n_false_clicks`` 個まぜて ``audio_onsets`` で拾う。
    """
    import tempfile
    from pathlib import Path

    from scipy.io import wavfile

    from smash_speed.config import load_config
    from smash_speed.events import audio_onsets
    from smash_speed.physics import PhysicsParams
    from smash_speed.synth import write_click_audio
    from smash_speed.synth_rally import make_rally, observe_rally, rally_camera

    cfg = cfg or load_config()
    phys = PhysicsParams.from_config(cfg)
    camera = camera or rally_camera(cfg)
    rng = np.random.default_rng(seed)
    rally_rng = np.random.default_rng([seed, 1])  # 音のあり・なしで同じラリーになるよう、ラリーは別の乱数で作る
    tot = {"tp": 0, "fp": 0, "fn": 0, "side_correct": 0, "n_true": 0}
    dts: list[float] = []
    for r in range(n_rallies):
        rally = make_rally(rally_rng, phys, n_shots=int(rally_rng.integers(4, 13)),
                           server_side="h" if r % 2 == 0 else "o")
        det = observe_rally(rally, camera, fps, phys, np.random.default_rng([seed, 2, r]), noise_px=noise_px,
                            dropout=dropout)
        true_t = np.array([s.t_hit for s in rally.segments])
        true_side = [s.side for s in rally.segments]
        onsets = None
        if use_audio:
            t0 = rally.t_start - 0.3
            clicks = [(s.t_hit - t0 + float(np.linalg.norm(camera.center - s.p0)) / 343.0,
                       0.6 if s.shot_type in ("smash", "drive", "clear") else 0.35) for s in rally.segments]
            dur = rally.t_end - t0 + 0.3
            clicks += [(float(rng.uniform(0.0, dur - 0.01)), float(rng.uniform(0.1, 0.4)))
                       for _ in range(n_false_clicks)]
            with tempfile.TemporaryDirectory() as td:
                wav = Path(td) / "a.wav"
                write_click_audio(wav, dur, clicks, 48000, int(rng.integers(1 << 30)))
                rate, x = wavfile.read(wav)
            onsets = [(t + t0, s) for t, s in audio_onsets(x.astype(float) / 32768.0, rate)]
        hits = find_hits(det, fps, onsets, camera=camera, **find_kw)
        m = match_hits(hits, true_t, true_side, tol_s)
        if verbose and (m["fp"] or m["fn"] or m["side_correct"] != m["tp"]):
            print(f"ラリー {r}：正解", np.round(true_t, 3).tolist(), true_side)
            print("  見つけた", [(round(h.t, 3), h.side, h.method, round(h.score, 1)) for h in hits])
        for key in ("tp", "fp", "fn", "side_correct"):
            tot[key] += m[key]
        tot["n_true"] += len(true_t)
        if m["mean_abs_dt_s"] is not None:
            dts += [m["mean_abs_dt_s"]] * m["tp"]
    tp, fp, fn = tot["tp"], tot["fp"], tot["fn"]
    return {"n_rallies": n_rallies, "n_hits": tot["n_true"], "fps": fps, "use_audio": use_audio,
            "tp": tp, "fp": fp, "fn": fn,
            "recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1),
            "side_accuracy": tot["side_correct"] / max(tp, 1),
            "mean_abs_dt_ms": 1000 * float(np.mean(dts)) if dts else None}


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="合成ラリーで、打球を見つける精度（再現率・適合率）を測る")
    ap.add_argument("--bench", action="store_true", help="合成ラリーで測る")
    ap.add_argument("--rallies", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=float, nargs="+", default=[30.0, 120.0, 240.0])
    a = ap.parse_args(argv)
    for fps in a.fps:
        for use_audio in (True, False):
            r = synthetic_benchmark(a.rallies, fps, a.seed, use_audio)
            print(f"{fps:5.0f} fps 音{'あり' if use_audio else 'なし'}：打球 {r['n_hits']} 本、"
                  f"再現率 {r['recall']:.3f}、適合率 {r['precision']:.3f}、打った側 {r['side_accuracy']:.3f}、"
                  f"時刻のずれ {r['mean_abs_dt_ms']:.1f} ms")


if __name__ == "__main__":
    main()
