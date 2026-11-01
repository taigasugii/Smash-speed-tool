"""実動画に近い条件の合成ラリー：検出の抜け・誤検出、選手とラケット、隣のコートのシャトル、低いカメラ、画面の外での打球。

``synth_rally.observe_session`` は「シャトルだけが、決まった割合で抜けて見える」観測で、打球の見つけ方（B2）・初速（B3）・
種類（B4）・ラリーの区切り（C1）は、横から撮ったこの観測ではほぼ 100% 合う。実動画では次のことが起きる。

- 打つ瞬間にシャトルが体・ラケットに隠れる、まとめて何コマも抜ける（照明・背景と同じ色）
- 振ったラケット・走る選手の頭が、動く小さな物として検出される（打点の近くで、シャトルの軌道とつながることもある）
- 隣のコートのシャトルが、同じ時間に別の軌道として見える。打球音も入る
- カメラが低い・角にあると、高いショット（クリア・ロブ・ロングサーブ）が画面の上に出て、打球が画面の外で起きる
- 柔らかいショット（ネット・ショートサーブ）の打球音は小さく、拾えないことがある。足音・声も音の候補になる

1. 場面（``ClutterScene``）：続けて打ったラリー（``synth_rally.make_session``）に、2 人の選手（打点の間を動く足元と、
   打つ前後に振るラケット）と、隣のコートのラリー（y を ``ClutterOptions.neighbor_dy_m`` ずらす）を足す。
2. 検出（``observe_cluttered``）：``detection.baseline.detect_frames(exclusive=False)`` と同じ形（軌道の番号つき・
   同じコマに何点もある）の検出と、打球音の候補を、場面から直接作る（動画を描いて検出するより速い）。
3. 測り方（``clutter_benchmark``）：解析の流れ（``shot_list.shuttle_tracks`` → ``hits.find_hits`` →
   ``shot_list.analyze_hits``。中でラリーの区切りと種類の判定もする）を通し、打球・初速・種類・ラリーの区切りを正解と比べる。

カメラは ``clutter_camera`` の名前で選ぶ（横・横の低い所・角の低い所・後ろの高い所）。既定の合成データ
（``synth_rally``）・設定の既定値は変えない。

``python -m smash_speed.synth_clutter --bench`` で、雑音なし・ありを並べて測る。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from smash_speed.camera import Camera, look_at
from smash_speed.physics import MPS_TO_KMH, PhysicsParams
from smash_speed.shots import Rally, Shot
from smash_speed.synth_rally import SyntheticRally, make_session, rally_camera, rally_positions

EZ = np.array([0.0, 0.0, 1.0])

# 合成ラリー用のカメラ（1920×1080）。"side" は設定の rally_camera（横・遠く・高い）
CAMERAS: dict[str, dict[str, Any]] = {
    # 横の近く・低い所（三脚・ベンチの高さ）。高いショットは画面の上に、奥の打点は画面の横に出る
    "side_low": {"eye_m": [0.0, -7.5, 1.2], "target_m": [0.0, 0.0, 2.0], "f_px": 1150.0},
    # コートの角の後ろの低い所（実動画に多い位置）。手前の打点は画面の下・横に出やすい
    "corner_low": {"eye_m": [-9.5, -6.5, 1.5], "target_m": [0.5, 0.5, 2.0], "f_px": 1000.0},
    # 後ろ・高い所（ROADMAP B2 で弱かった位置。遠くへ飛ぶ・近づくシャトルが画面の上で加速して見える）
    "back_high": {"eye_m": [-12.0, 0.0, 4.5], "target_m": [0.0, 0.0, 1.5], "f_px": 1300.0},
}


def clutter_camera(name: str, cfg: dict[str, Any]) -> Camera:
    """合成ラリー用のカメラ（``CAMERAS`` の名前か "side"）。"""
    if name == "side":
        return rally_camera(cfg)
    c = CAMERAS[name]
    return look_at(np.array(c["eye_m"]), np.array(c["target_m"]), c["f_px"], 1920, 1080)


@dataclass
class ClutterOptions:
    """足す雑音の大きさ。``ClutterOptions.clean()`` は、今までの合成ラリーと同じ（シャトルだけ・抜けは一様）。"""

    noise_px: float = 2.0                     # 検出の位置の雑音（標準偏差）[px]
    dropout: float = 0.12                     # シャトルが 1 コマずつ抜ける割合
    occlusion_s: tuple[float, float] = (0.03, 0.06)  # 打つ前・後にシャトルが体・ラケットに隠れる時間の上限 [s]（一様に引く）
    burst_per_s: float = 0.3                  # まとめて抜ける区間の数（1 秒あたり）
    burst_s: tuple[float, float] = (0.07, 0.25)  # まとめて抜ける長さ [s]
    racket_p: float = 0.6                     # 打つ前後に振ったラケットが検出に出る割合（打球ごと）
    racket_link_p: float = 0.3                # ラケットの点が、飛んできたシャトルの軌道とつながる割合
    body_p: float = 0.5                       # 走った選手の頭が検出に出る割合（速さ 1.5 m/s 以上で走った区間ごと）
    body_piece_s: tuple[float, float] = (0.15, 0.5)  # 頭の軌道の 1 つの長さ [s]（体と重なって途切れる）
    body_break_s: tuple[float, float] = (0.1, 0.4)   # 途切れる長さ [s]
    neighbor: bool = True                     # 隣のコートのラリー
    neighbor_dy_m: float = 7.5                # 隣のコートの中心の y [m]（コートの外側の線の間が約 1.4 m）
    neighbor_dropout: float = 0.3
    audio_strength: tuple[float, float] = (15.0, 60.0)  # 打球音の強さ（雑音比）
    audio_soft_scale: float = 0.6             # ネット・ショートサーブ・ドロップの打球音の強さの倍率
    audio_miss_soft_p: float = 0.25           # それらの打球音が拾えない割合
    audio_miss_p: float = 0.03                # ほかのショットの打球音が拾えない割合
    neighbor_audio: tuple[float, float] = (8.0, 25.0)  # 隣のコートの打球音の強さ
    other_sounds_per_s: float = 0.4           # 足音・声など（打球ではない音）の数（1 秒あたり）
    other_sounds: tuple[float, float] = (5.0, 35.0)    # その強さ

    @classmethod
    def clean(cls) -> "ClutterOptions":
        """今までの合成ラリー（``synth_rally.observe_session``・打球音はすべて強さ 30）と同じ条件。"""
        return cls(noise_px=1.5, dropout=0.1, occlusion_s=(0.0, 0.0), burst_per_s=0.0, racket_p=0.0,
                   racket_link_p=0.0, body_p=0.0, neighbor=False, audio_strength=(30.0, 30.0), audio_soft_scale=1.0,
                   audio_miss_soft_p=0.0, audio_miss_p=0.0, other_sounds_per_s=0.0)

    @classmethod
    def only(cls, part: str) -> "ClutterOptions":
        """``PARTS`` のうち 1 つだけを足した条件（ほかは ``clean``）。どれがどれだけ効くかを分けて見る。"""
        full, out = cls(), cls.clean()
        return replace(out, **{k: getattr(full, k) for k in PARTS[part]})


# 雑音の種類ごとの ``ClutterOptions`` の項目（``ClutterOptions.only``）
PARTS: dict[str, tuple[str, ...]] = {
    "detection": ("noise_px", "dropout", "occlusion_s", "burst_per_s"),
    "racket": ("racket_p", "racket_link_p"),
    "body": ("body_p",),
    "neighbor": ("neighbor",),
    "audio": ("audio_strength", "audio_soft_scale", "audio_miss_soft_p", "audio_miss_p", "other_sounds_per_s"),
}


# =============================================================================
# 場面
# =============================================================================
@dataclass
class ClutterScene:
    """合成ラリーの場面：ラリーの並び・選手・隣のコートのラリー。時刻 [s]、位置 [m]。"""

    rallies: list[SyntheticRally]
    neighbor: list[SyntheticRally] = field(default_factory=list)
    neighbor_dy_m: float = 7.5
    phys: PhysicsParams = field(default_factory=PhysicsParams)

    @property
    def t_start(self) -> float:
        return self.rallies[0].t_start

    @property
    def t_end(self) -> float:
        return self.rallies[-1].t_end

    def hits(self) -> list[tuple[float, np.ndarray, str, str]]:
        """すべての打球の（時刻・打点・打った側・``SHOT_PROFILES`` のキー）。時刻の順。"""
        return [(s.t_hit, s.p0, s.side, s.profile) for r in self.rallies for s in r.segments]

    def neighbor_hits(self) -> list[tuple[float, np.ndarray]]:
        return [(s.t_hit, s.p0 + np.array([0.0, self.neighbor_dy_m, 0.0])) for r in self.neighbor
                for s in r.segments]

    def shuttle(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """各時刻のシャトルの位置（飛んでいなければ NaN）、何本目のショットか（通し番号。-1 は飛んでいない）、何個目のラリーか。"""
        pos = np.full((len(times), 3), np.nan)
        shot = np.full(len(times), -1)
        rally = np.full(len(times), -1)
        n = 0
        for k, r in enumerate(self.rallies):
            p, w = rally_positions(r, times, self.phys)
            m = w >= 0
            pos[m], shot[m], rally[m] = p[m], w[m] + n, k
            n += len(r.segments)
        return pos, shot, rally

    def neighbor_shuttle(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """隣のコートのシャトルの位置と、通し番号（-1 は飛んでいない）。"""
        pos = np.full((len(times), 3), np.nan)
        shot = np.full(len(times), -1)
        n = 0
        for r in self.neighbor:
            p, w = rally_positions(r, times, self.phys)
            m = w >= 0
            pos[m], shot[m] = p[m] + np.array([0.0, self.neighbor_dy_m, 0.0]), w[m] + n
            n += len(r.segments)
        return pos, shot

    def feet(self, side: str, times: np.ndarray) -> np.ndarray:
        """選手（``side`` の側）の足元の位置。自分の打点の真下（少し後ろ）の間をまっすぐ動き、打たない間はホームに戻る。"""
        sign = -1.0 if side == "h" else 1.0
        home = np.array([sign * 3.2, 0.0])
        kt, kp = [], []
        for t, p, s, _ in self.hits():
            if s != side:
                continue
            if kt and t - kt[-1] > 1.2:  # 次に打つまで間があれば、いったんホームに戻る
                kt.append(0.5 * (kt[-1] + t))
                kp.append(home)
            kt.append(t)
            kp.append(np.array([p[0] + sign * 0.4, p[1]]))
        if not kt:
            kt, kp = [self.t_start], [home]
        kp = np.array(kp)
        xy = np.column_stack([np.interp(times, kt, kp[:, 0]), np.interp(times, kt, kp[:, 1])])
        return np.column_stack([xy, np.zeros(len(times))])

    def racket(self, t_hit: float, p0: np.ndarray, times: np.ndarray) -> np.ndarray:
        """打球 ``t_hit`` の前後 0.15 s に振ったラケットの面の中心（打点の時刻に打点を通る弧）。範囲の外は NaN。"""
        out = np.full((len(times), 3), np.nan)
        m = np.abs(times - t_hit) <= 0.15
        if not m.any():
            return out
        d = np.array([-np.sign(p0[0]) or 1.0, 0.0, 0.0])  # 打つ向き（相手側へ）
        phi = 2.2 * np.tanh((times[m] - t_hit) / 0.06)    # 振りの角度。打点で 0
        r = 0.65                                          # 肩からラケットの面までの長さ [m]
        shoulder = p0 - r * EZ
        out[m] = shoulder + r * (np.cos(phi)[:, None] * EZ + np.sin(phi)[:, None] * d)
        return out


def make_scene(rng: np.random.Generator, phys: PhysicsParams, n_rallies: int = 5,
               rest_s: tuple[float, float] = (1.0, 4.0), opts: ClutterOptions | None = None) -> ClutterScene:
    """続けて打った ``n_rallies`` 個のラリーと、同じ時間に隣のコートで打ったラリーの場面。"""
    o = opts or ClutterOptions()
    rallies = make_session(rng, phys, n_rallies, rest_s=rest_s)
    neighbor: list[SyntheticRally] = []
    if o.neighbor:
        dur = rallies[-1].t_end - rallies[0].t_start
        t0 = rallies[0].t_start + float(rng.uniform(-1.0, 2.0))
        neighbor = make_session(rng, phys, max(1, int(np.ceil(n_rallies * 1.2))), rest_s=(2.0, 6.0), t_start=t0)
        neighbor = [r for r in neighbor if r.t_start < rallies[0].t_start + dur]
    return ClutterScene(rallies, neighbor, o.neighbor_dy_m, phys)


# =============================================================================
# 検出と打球音
# =============================================================================
def _runs(idx: np.ndarray, max_gap: int = 2) -> list[np.ndarray]:
    """コマの番号の並びを、``max_gap`` コマより長く抜けた所で分ける（検出の軌道のつなぎ方 ``max_gap_frames`` と同じ）。"""
    if not len(idx):
        return []
    cut = np.flatnonzero(np.diff(idx) > max_gap + 1) + 1
    return np.split(idx, cut)


def observe_cluttered(scene: ClutterScene, camera: Camera, fps: float, rng: np.random.Generator,
                      opts: ClutterOptions | None = None, pre_s: float = 0.3, post_s: float = 0.5,
                      min_track_len: int = 4) -> tuple[pd.DataFrame, list[tuple[float, float]], pd.DataFrame]:
    """場面を fps [Hz] のカメラで観測した、検出（重なった軌道を含む）と打球音の候補と、コマごとの正解。

    Returns:
        - 検出：列 ``frame, timestamp, x_px, y_px, visible, confidence, track_id, source``
          （``detect_frames(exclusive=False)`` と同じ形。source は shuttle・neighbor・racket・body。
          点のないコマは visible = 0 の 1 行）
        - 打球音の候補：（映像の時刻 [s]、強さ）。打球音はカメラまでの距離 / 音速だけ遅れる
        - 正解：列 ``frame, timestamp, x_m, y_m, z_m, shot_index, rally_index``（shot_index は場面の通し番号）
    """
    o = opts or ClutterOptions()
    dt = 1.0 / fps
    times = np.arange(scene.t_start - pre_s + rng.uniform(0.0, 1.0) * dt, scene.t_end + post_s, dt)
    n = len(times)
    W, H = camera.width, camera.height

    def project(pos: np.ndarray, noise: float) -> tuple[np.ndarray, np.ndarray]:
        uv = camera.project(np.nan_to_num(pos)) + rng.normal(0.0, noise, (len(pos), 2))
        ok = (np.isfinite(pos).all(axis=1) & np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < W)
              & (uv[:, 1] >= 0) & (uv[:, 1] < H))
        return uv, ok

    rows: list[pd.DataFrame] = []
    next_id = 0

    def add(frames: np.ndarray, uv: np.ndarray, source: str, conf: float, split_at: np.ndarray | None = None,
            track_id: int | None = None) -> list[tuple[np.ndarray, int]]:
        """見えたコマ ``frames`` を、抜けた所（と ``split_at`` の値が変わる所）で軌道に分けて足す。"""
        nonlocal next_id
        pieces: list[np.ndarray] = []
        if split_at is None:
            pieces = _runs(frames)
        else:
            for v in np.unique(split_at[frames]):
                pieces += _runs(frames[split_at[frames] == v])
        out = []
        for p in pieces:
            if len(p) < min_track_len:
                continue
            tid = next_id if track_id is None else track_id
            next_id += track_id is None
            rows.append(pd.DataFrame({"frame": p, "timestamp": times[p], "x_px": uv[p, 0], "y_px": uv[p, 1],
                                      "visible": 1, "confidence": conf, "track_id": tid, "source": source}))
            out.append((p, tid))
        return out

    # シャトル：1 コマずつの抜け・まとめて抜ける区間・打つ前後に隠れる区間
    pos, shot, rally = scene.shuttle(times)
    uv, ok = project(pos, o.noise_px)
    ok &= rng.uniform(size=n) >= o.dropout
    for _ in range(rng.poisson(o.burst_per_s * (times[-1] - times[0]))):
        a = rng.uniform(times[0], times[-1])
        ok &= ~((times >= a) & (times < a + rng.uniform(*o.burst_s)))
    hits = scene.hits()
    for t, *_ in hits:
        ok &= ~((times >= t - rng.uniform(0.0, o.occlusion_s[0])) & (times <= t + rng.uniform(0.0, o.occlusion_s[1])))
    shuttle_tracks = add(np.flatnonzero(ok), uv, "shuttle", 0.6, split_at=shot)
    # ラケット：打つ前後に振った面の中心。ときどき、飛んできたシャトルの軌道とつながる
    for k, (t, p0, _, _) in enumerate(hits):
        if rng.uniform() >= o.racket_p:
            continue
        r_uv, r_ok = project(scene.racket(t, p0, times), o.noise_px)
        r_ok &= rng.uniform(size=n) >= 0.2
        incoming = [tid for f, tid in shuttle_tracks if shot[f[-1]] == k - 1 and t - times[f[-1]] < 0.12]
        link = incoming[-1] if incoming and rng.uniform() < o.racket_link_p else None
        add(np.flatnonzero(r_ok), r_uv, "racket", 0.5, track_id=link)
    # 選手の頭：速く（1.5 m/s 以上）走ったとき。体と重なって、短い軌道に途切れる
    for side in ("h", "o"):
        feet = scene.feet(side, times)
        speed = np.r_[0.0, np.linalg.norm(np.diff(feet, axis=0), axis=1) / dt]
        h_uv, h_ok = project(feet + 1.7 * EZ, o.noise_px)
        for f in _runs(np.flatnonzero((speed >= 1.5) & h_ok), 0):
            if rng.uniform() >= o.body_p:
                continue
            t = times[f[0]] + rng.uniform(0.0, o.body_break_s[1])
            while t < times[f[-1]]:
                t1 = t + rng.uniform(*o.body_piece_s)
                add(f[(times[f] >= t) & (times[f] < t1)], h_uv, "body", 0.4)
                t = t1 + rng.uniform(*o.body_break_s)
    # 隣のコートのシャトル
    if o.neighbor and scene.neighbor:
        n_pos, n_shot = scene.neighbor_shuttle(times)
        n_uv, n_ok = project(n_pos, o.noise_px)
        n_ok &= rng.uniform(size=n) >= o.neighbor_dropout
        add(np.flatnonzero(n_ok), n_uv, "neighbor", 0.6, split_at=n_shot)

    det = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["frame", "timestamp", "x_px", "y_px", "visible", "confidence", "track_id", "source"])
    empty = np.setdiff1d(np.arange(n), det.frame.to_numpy(int))
    det = pd.concat([det, pd.DataFrame({"frame": empty, "timestamp": times[empty], "x_px": np.nan, "y_px": np.nan,
                                        "visible": 0, "confidence": 0.0, "track_id": -1, "source": ""})],
                    ignore_index=True)
    det = det.sort_values(["frame", "track_id"], kind="stable").reset_index(drop=True)
    det[["frame", "visible", "track_id"]] = det[["frame", "visible", "track_id"]].astype(int)

    # 打球音：柔らかいショットは拾えないことがある。隣のコートの打球音、足音・声も入る
    onsets: list[tuple[float, float]] = []
    for t, p0, _, profile in hits:
        soft = profile in ("net", "serve_short", "drop")
        if rng.uniform() < (o.audio_miss_soft_p if soft else o.audio_miss_p):
            continue
        s = rng.uniform(*o.audio_strength) * (o.audio_soft_scale if soft else 1.0)
        onsets.append((t + float(np.linalg.norm(camera.center - p0)) / 343.0, float(s)))
    for t, p0 in scene.neighbor_hits() if o.neighbor else []:
        if times[0] <= t <= times[-1]:
            onsets.append((t + float(np.linalg.norm(camera.center - p0)) / 343.0, float(rng.uniform(*o.neighbor_audio))))
    for _ in range(rng.poisson(o.other_sounds_per_s * (times[-1] - times[0]))):
        onsets.append((float(rng.uniform(times[0], times[-1])), float(rng.uniform(*o.other_sounds))))
    onsets.sort()

    truth = pd.DataFrame({"frame": np.arange(n), "timestamp": times, "x_m": pos[:, 0], "y_m": pos[:, 1],
                          "z_m": pos[:, 2], "shot_index": shot, "rally_index": rally})
    return det, onsets, truth


# =============================================================================
# 測り方
# =============================================================================
def truth_shots(scene: ClutterScene) -> tuple[list[Shot], list[Rally]]:
    """場面の正解のショット（``rally_id`` は 1 から）とラリー。"""
    shots: list[Shot] = []
    rallies: list[Rally] = []
    for i, r in enumerate(scene.rallies, start=1):
        shots += r.to_shots(rally_id=i, phys=scene.phys)
        rallies.append(r.to_rally(rally_id=i))
    return shots, rallies


def evaluate(pred: list[Shot], pred_rallies: list[Rally], truth: list[Shot], true_rallies: list[Rally],
             hits: list[Any] | None = None, tol_s: float = 0.05) -> dict[str, Any]:
    """推定したショット・ラリーを正解と比べる。

    Returns:
        ``hits``（``hits.match_hits``）、``rows``（正解と組になったショット 1 本 1 行：種類・正解と推定の初速・誤差 [%]・
        判定した種類）、``rallies``（``rallies.rally_scores``）
    """
    from smash_speed.hits import match_hits
    from smash_speed.rallies import match_to_truth, rally_scores

    out: dict[str, Any] = {}
    if hits is not None:
        out["hits"] = match_hits(hits, np.array([s.t_hit_s for s in truth]), [s.side for s in truth], tol_s)
    rows = []
    for p, j in zip(pred, match_to_truth(pred, truth, tol_s)):
        if j is None:
            continue
        t = truth[j]
        rows.append({"rally": t.rally_id, "index": t.index_in_rally, "profile": t.ref, "true": t.shot_type,
                     "pred": p.shot_type, "true_kmh": t.speed_kmh, "est_kmh": p.speed_kmh,
                     "err_pct": None if p.speed_kmh is None else 100 * (p.speed_kmh / t.speed_kmh - 1),
                     "dt_hit_ms": 1000 * (p.t_hit_s - t.t_hit_s)})
    out["rows"] = pd.DataFrame(rows, columns=["rally", "index", "profile", "true", "pred", "true_kmh", "est_kmh",
                                              "err_pct", "dt_hit_ms"])
    out["rallies"] = rally_scores(pred, pred_rallies, truth, true_rallies, tol_s)
    return out


def analyze_scene(det: pd.DataFrame, onsets: list[tuple[float, float]] | None, camera: Camera, fps: float,
                  phys: PhysicsParams, cfg: dict[str, Any], rally_gap_s: float = 3.0, workers: int = 1
                  ) -> tuple[list[Any], list[Shot], list[Rally]]:
    """解析の流れ（``shot_list.analyze_video`` の検出の後）：軌道の絞り込み → 打球 → 当てはめ・区切り・種類。"""
    from smash_speed.hits import find_hits
    from smash_speed.rallies import RallyOptions, split_rallies
    from smash_speed.shot_list import analyze_hits, shuttle_tracks

    det1, _ = shuttle_tracks(det.drop(columns=["source"], errors="ignore"), camera, cfg)
    hits = find_hits(det1, fps, onsets, camera=camera, rally_gap_s=rally_gap_s)
    shots = analyze_hits(det1, hits, camera, phys, cfg, rally_gap_s, ci=False, workers=workers,
                         progress=lambda _: None)
    rallies = split_rallies([Shot(**vars(s)) for s in shots], det1, RallyOptions(max_gap_s=rally_gap_s),
                            cfg.get("court"))
    return hits, shots, rallies


def hits_benchmark(camera_name: str = "side", n_sessions: int = 3, n_rallies: int = 5, fps: float = 30.0,
                   seed: int = 0, opts: ClutterOptions | None = None, use_audio: bool = True,
                   rest_s: tuple[float, float] = (1.0, 4.0), cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """雑音を足した合成ラリーで、軌道の絞り込み（``shot_list.shuttle_tracks``）と打球の見つけ方（``find_hits``）だけを測る
    （当てはめをしないので速い）。場面は ``clutter_benchmark`` と同じ。

    Returns:
        ``recall``・``precision``・``side_accuracy``（打球）、``shuttle_kept``（シャトルの検出のうち、選んだ軌道に残った割合）、
        ``selected``（選んだ軌道の、元の物ごとの数：shuttle・neighbor・racket・body）
    """
    from smash_speed.config import load_config
    from smash_speed.hits import find_hits, match_hits
    from smash_speed.shot_list import shuttle_tracks

    cfg = cfg or load_config()
    phys = PhysicsParams.from_config(cfg)
    camera = clutter_camera(camera_name, cfg)
    o = opts or ClutterOptions()
    tot = {"tp": 0, "fp": 0, "fn": 0, "side_correct": 0}
    kept, n_shuttle = 0, 0
    selected: dict[str, int] = {}
    for k in range(n_sessions):
        scene = make_scene(np.random.default_rng([seed, 7, k]), phys, n_rallies, rest_s,
                           replace(ClutterOptions(), neighbor=o.neighbor, neighbor_dy_m=o.neighbor_dy_m))
        det, onsets, _ = observe_cluttered(scene, camera, fps, np.random.default_rng([seed, 8, k]), o)
        det1, tab = shuttle_tracks(det.drop(columns=["source"]), camera, cfg)
        sel = set(tab.track_id[tab.selected])
        src = det[det.visible > 0].groupby("track_id").source.first()
        for s in src[src.index.isin(sel)]:
            selected[s] = selected.get(s, 0) + 1
        sh = det[det.source == "shuttle"]
        kept += int(sh.track_id.isin(sel).sum())
        n_shuttle += len(sh)
        hits = find_hits(det1, fps, onsets if use_audio else None, camera=camera)
        truth, _ = truth_shots(scene)
        m = match_hits(hits, np.array([s.t_hit_s for s in truth]), [s.side for s in truth])
        for key in tot:
            tot[key] += m[key]
    tp, fp, fn = tot["tp"], tot["fp"], tot["fn"]
    return {"camera": camera_name, "fps": fps, "recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1),
            "side_accuracy": tot["side_correct"] / max(tp, 1), "tp": tp, "fp": fp, "fn": fn,
            "shuttle_kept": kept / max(n_shuttle, 1), "selected": selected}


def clutter_benchmark(camera_name: str = "side", n_sessions: int = 1, n_rallies: int = 5, fps: float = 30.0,
                      seed: int = 0, opts: ClutterOptions | None = None, use_audio: bool = True,
                      rest_s: tuple[float, float] = (1.0, 4.0), cfg: dict[str, Any] | None = None,
                      workers: int = 1) -> dict[str, Any]:
    """雑音を足した合成ラリーで、打球・初速・種類・ラリーの区切りを測る。

    場面は ``opts`` によらず同じ乱数で作る（雑音なし・ありを同じラリーで比べられる。隣のコートのラリーは
    ``opts.neighbor`` のときだけ足す）。

    Returns:
        ``hits``（再現率・適合率・打った側の正解率・時刻のずれ）、``speed``（全体と種類ごとの初速の平均絶対誤差 [%]・
        当てはめられなかった本数）、``types``（混同行列・正解率など ``shot_types.scores``）、``rallies``（区切りの再現率・
        適合率・ちょうど分かれたラリーの割合）、``rows``（ショット 1 本 1 行）
    """
    from smash_speed.config import load_config
    from smash_speed.shot_types import confusion_matrix, scores

    cfg = cfg or load_config()
    phys = PhysicsParams.from_config(cfg)
    camera = clutter_camera(camera_name, cfg)
    o = opts or ClutterOptions()
    tot = {"tp": 0, "fp": 0, "fn": 0, "side_correct": 0}
    dts: list[float] = []
    rows, rs = [], {}
    n_shots = 0
    for k in range(n_sessions):
        scene = make_scene(np.random.default_rng([seed, 7, k]), phys, n_rallies, rest_s,
                           replace(ClutterOptions(), neighbor=o.neighbor, neighbor_dy_m=o.neighbor_dy_m))
        det, onsets, _ = observe_cluttered(scene, camera, fps, np.random.default_rng([seed, 8, k]), o)
        truth, true_rallies = truth_shots(scene)
        hits, shots, rallies = analyze_scene(det, onsets if use_audio else None, camera, fps, phys, cfg,
                                             workers=workers)
        ev = evaluate(shots, rallies, truth, true_rallies, hits)
        for key in tot:
            tot[key] += ev["hits"][key]
        if ev["hits"]["mean_abs_dt_s"] is not None:
            dts += [ev["hits"]["mean_abs_dt_s"]] * ev["hits"]["tp"]
        rows.append(ev["rows"].assign(session=k))
        for key, v in ev["rallies"].items():
            rs.setdefault(key, []).append(v)
        n_shots += len(truth)
    df = pd.concat(rows, ignore_index=True)
    tp, fp, fn = tot["tp"], tot["fp"], tot["fn"]
    e = df.err_pct.dropna().abs()
    by_type = {}
    for name, g in df.groupby("true"):
        ee = g.err_pct.dropna().abs()
        by_type[name] = {"n": int(len(g)), "n_failed": int(g.err_pct.isna().sum()),
                         "mape": float(ee.mean()) if len(ee) else None}
    m = confusion_matrix(df.true, df.pred)
    btp, bfp, bfn = (sum(rs[f"boundary_{x}"]) for x in ("tp", "fp", "fn"))
    return {
        "camera": camera_name, "fps": fps, "n_sessions": n_sessions, "n_rallies": n_sessions * n_rallies,
        "n_shots": n_shots,
        "hits": {"recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1),
                 "side_accuracy": tot["side_correct"] / max(tp, 1), "tp": tp, "fp": fp, "fn": fn,
                 "mean_abs_dt_ms": 1000 * float(np.mean(dts)) if dts else None},
        "speed": {"mape": float(e.mean()) if len(e) else None, "median": float(e.median()) if len(e) else None,
                  "n": int(len(df)), "n_failed": int(df.err_pct.isna().sum()), "by_type": by_type},
        "types": {"matrix": m, "scores": scores(m)},
        "rallies": {"boundary_recall": btp / max(btp + bfn, 1), "boundary_precision": btp / max(btp + bfp, 1),
                    "exact": float(np.dot(rs["exact"], rs["n_true"]) / max(sum(rs["n_true"]), 1)),
                    "n_pred": int(sum(rs["n_pred"]))},
        "rows": df,
    }


def format_result(r: dict[str, Any]) -> str:
    """``clutter_benchmark`` の結果を 1 行の文字列にする。"""
    from smash_speed.shots import SHOT_TYPE_LABELS

    h, s, t, ra = r["hits"], r["speed"], r["types"]["scores"], r["rallies"]
    worst = sorted(((k, v["mape"]) for k, v in s["by_type"].items() if v["mape"] is not None),
                   key=lambda kv: -kv[1])[:2]
    return (f"打球 再現率 {h['recall']:.3f}・適合率 {h['precision']:.3f}・打った側 {h['side_accuracy']:.3f}、"
            f"初速のずれ {s['mape']:.1f}%（{s['n']} 本・失敗 {s['n_failed']}、"
            + "・".join(f"{SHOT_TYPE_LABELS[k]} {v:.1f}%" for k, v in worst) + "）、"
            f"種類 {t['accuracy']:.3f}、区切り 再現率 {ra['boundary_recall']:.3f}・適合率 {ra['boundary_precision']:.3f}・"
            f"ちょうど {ra['exact']:.2f}")


def main(argv: list[str] | None = None) -> None:
    import argparse
    import time

    ap = argparse.ArgumentParser(description="雑音・人・隣のコート・画面の外を足した合成ラリーで、解析の精度を測る")
    ap.add_argument("--bench", action="store_true", help="合成ラリーで測る")
    ap.add_argument("--cameras", nargs="+", default=["side", "side_low", "corner_low", "back_high"],
                    choices=["side", *CAMERAS])
    ap.add_argument("--sessions", type=int, default=2)
    ap.add_argument("--rallies", type=int, default=5, help="1 つの動画のラリーの数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=float, nargs="+", default=[30.0])
    ap.add_argument("--clean", action="store_true", help="雑音なし（今までの合成ラリーと同じ条件）も測る")
    ap.add_argument("--parts", action="store_true", help="雑音の種類を 1 つずつ足した条件も測る")
    ap.add_argument("--hits-only", action="store_true", help="軌道の絞り込みと打球の見つけ方だけ測る（速い）")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    if not a.bench:
        ap.error("--bench を指定してください")
    conds = (([("雑音なし", ClutterOptions.clean())] if a.clean else [])
             + ([(f"{p}だけ", ClutterOptions.only(p)) for p in PARTS] if a.parts else [])
             + [("雑音あり", ClutterOptions())])
    for fps in a.fps:
        for cam in a.cameras:
            for name, o in conds:
                t0 = time.time()
                if a.hits_only:
                    r = hits_benchmark(cam, a.sessions, a.rallies, fps, a.seed, o)
                    print(f"{cam:10s} {fps:4.0f} fps {name}：打球 再現率 {r['recall']:.3f}・適合率 {r['precision']:.3f}・"
                          f"打った側 {r['side_accuracy']:.3f}、シャトルの検出が残った割合 {r['shuttle_kept']:.2f}、"
                          f"選んだ軌道 {r['selected']}（{time.time() - t0:.0f} 秒）", flush=True)
                    continue
                r = clutter_benchmark(cam, a.sessions, a.rallies, fps, a.seed, o, workers=a.workers)
                print(f"{cam:10s} {fps:4.0f} fps {name}：{format_result(r)}（{r['n_shots']} 本、{time.time() - t0:.0f} 秒）",
                      flush=True)


if __name__ == "__main__":
    import sys

    if (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    main()
