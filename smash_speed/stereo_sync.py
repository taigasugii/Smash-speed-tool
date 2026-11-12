"""2 台のカメラの時刻合わせ（音と映像の両方から）と、その確かめ。

2 台目の動画の時刻 t2 と 1 台目の動画の時刻 t1 の関係を ``t2 = offset + (1 + drift) * t1`` で表す（``SyncModel``）。
drift はスマホの時計の進み方の違い（数十 ppm。10 分で数十 ms になる）。

1. **音**（``audio_coarse_offset`` → ``audio_pairs`` → ``fit_audio_sync``）：打球音・手をたたく音の包絡線の相互相関で
   大まかなずれを出し、鋭い音 1 つずつの時刻の差をサンプルより細かく測る。音は打った所から 2 台へ別々の距離を
   飛ぶので、打点が分かった後で距離の差 / 音速を引く。
2. **映像**（``video_sync``）：ずれが正しければ、同じ時刻の 2 台の視線は 1 点（シャトル）で交わる。fps の高い方の
   検出を時刻で補間し、2 本の視線のすき間（いちばん近づく所の距離）が小さくなるずれを探す。
   フレームの時刻そのものを合わせるので、スマホごとの「音と映像のずれ」に左右されない。最後の値はこちらを使う。
3. **確かめ**：映像のずれを区間ごとに出したばらつきと、音から出したずれとの差（``SyncReport``）。

fps が違っても・可変フレームレートでも、フレームごとの時刻だけを使うので同じに動く。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
from scipy.signal import butter, correlate, sosfiltfilt

from smash_speed.camera import Camera

SPEED_OF_SOUND_MPS = 343.0


@dataclass
class SyncModel:
    """2 台の時刻の関係 ``t2 = offset_s + (1 + drift) * t1``（どちらも動画の時刻 [s]）。"""

    offset_s: float = 0.0
    drift: float = 0.0

    def to_cam2(self, t1: Any) -> Any:
        return self.offset_s + (1.0 + self.drift) * np.asarray(t1, float)

    def to_cam1(self, t2: Any) -> Any:
        return (np.asarray(t2, float) - self.offset_s) / (1.0 + self.drift)

    def shifted(self, d_s: float) -> "SyncModel":
        return SyncModel(self.offset_s + d_s, self.drift)

    def to_dict(self) -> dict[str, float]:
        return {"offset_s": float(self.offset_s), "drift_ppm": float(self.drift * 1e6)}


@dataclass
class SyncReport:
    """時刻合わせの結果と確かめ。時刻 [s]・[ms]。"""

    model: SyncModel
    method: str = ""                       # video | audio | manual
    sigma_s: float = 0.0                   # ずれの不確かさ（1σ）
    n_pairs: int = 0                       # 映像の時刻合わせに使った視線の組の数
    gap_cm: float | None = None            # 合わせた後の 2 台の視線のすき間の中央値 [cm]（キャリブレーションの食い違いも入る）
    sharpness: float | None = None         # 合わない所のすき間 / 合った所のすき間（大きいほど確か）
    chunk_offsets_ms: list[float] = field(default_factory=list)   # 区間ごとの映像のずれ − モデル [ms]
    audio: dict[str, Any] = field(default_factory=dict)           # 音からのずれ・映像との差・ばらつき
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["model"] = self.model.to_dict()
        return d


# =============================================================================
# 音
# =============================================================================
def sharp_envelope(x: np.ndarray, rate: int, highpass_hz: float = 1000.0) -> np.ndarray:
    """鋭い音が立つように高域通過フィルタをかけた波形の大きさ（float32。長い録音でもメモリを食わないように）。"""
    sos = butter(4, highpass_hz, btype="highpass", fs=rate, output="sos")
    return np.abs(sosfiltfilt(sos, np.asarray(x, np.float32))).astype(np.float32)


def _parabola(c: np.ndarray, i: int) -> float:
    """最大の所 i の両隣から放物線で出した、サンプルより細かい位置のずれ。"""
    if not 0 < i < len(c) - 1:
        return 0.0
    a, b, d = float(c[i - 1]), float(c[i]), float(c[i + 1])
    den = a - 2 * b + d
    return 0.5 * (a - d) / den if den != 0 else 0.0


def audio_coarse_offset(e1: np.ndarray, e2: np.ndarray, rate: int, max_offset_s: float | None = None,
                        coarse_hz: float = 1000.0) -> tuple[float, float]:
    """録音全体の大まかなずれ（2 台目の時刻 − 1 台目の時刻）[s] と相関の鋭さ（ピーク / ほかの所の最大）。

    包絡線（``sharp_envelope``）を ``coarse_hz`` まで間引いて（区間の最大）相互相関をとる。細かいずれは
    ``audio_pairs`` で音 1 つずつから出す。
    """
    n = max(1, int(round(rate / coarse_hz)))

    def decimate(e: np.ndarray) -> np.ndarray:
        m = len(e) // n
        d = e[:m * n].reshape(m, n).max(axis=1).astype(float)
        return (d - d.mean()) / (d.std() + 1e-12)

    a, b = decimate(e1), decimate(e2)
    c = correlate(b, a, mode="full", method="fft")
    lags = np.arange(-len(a) + 1, len(b))
    if max_offset_s is not None:
        keep = np.abs(lags) <= max_offset_s * rate / n
        c, lags = c[keep], lags[keep]
    i = int(np.argmax(c))
    guard = int(0.05 * rate / n)
    others = np.r_[c[:max(0, i - guard)], c[i + guard + 1:]]
    sharp = float(c[i] / (others.max() + 1e-12)) if len(others) and others.max() > 0 else float("inf")
    return float((lags[i] + _parabola(c, i)) * n / rate), sharp


def audio_pairs(e1: np.ndarray, e2: np.ndarray, rate: int, onsets1: list[tuple[float, float]], coarse_s: float,
                search_s: float = 0.08, pre_s: float = 0.003, post_s: float = 0.02, min_corr: float = 0.4
                ) -> list[tuple[float, float, float]]:
    """1 台目の鋭い音（``events.audio_onsets``）1 つずつについて、2 台目で同じ音が聞こえた時刻との差を測る。

    音の前 ``pre_s``・後 ``post_s`` の包絡線を、2 台目の ``coarse_s`` だけずらした所の前後 ``search_s`` の中で
    動かして相関がいちばん高い所を探す（打点から 2 台への距離の差で数十 ms 変わるので広く探す）。

    Returns:
        (1 台目の音の時刻 [s], 2 台目の時刻 − 1 台目の時刻 [s], 相関 0〜1) のリスト（相関が ``min_corr`` 以上のもの）
    """
    out = []
    n0, n1, ns = int(pre_s * rate), int(post_s * rate), int(search_s * rate)
    for t1, _ in onsets1:
        i = int(round(t1 * rate))
        j = int(round((t1 + coarse_s) * rate))
        if i - n0 < 0 or i + n1 > len(e1) or j - n0 - ns < 0 or j + n1 + ns > len(e2):
            continue
        a = e1[i - n0:i + n1].astype(float)
        b = e2[j - n0 - ns:j + n1 + ns].astype(float)
        a = a - a.mean()
        c = correlate(b - b.mean(), a, mode="valid")
        # 窓ごとの大きさで割って相関係数にする（大きい別の音に引かれないように）
        w = np.sqrt(np.maximum(np.convolve((b - b.mean()) ** 2, np.ones(len(a)), mode="valid"), 1e-20))
        c = c / (w * (np.linalg.norm(a) + 1e-12))
        k = int(np.argmax(c))
        if c[k] < min_corr:
            continue
        out.append((float(t1), float((j - i + k + _parabola(c, k) - ns) / rate), float(c[k])))
    return out


def robust_line(t: np.ndarray, y: np.ndarray, min_span_s: float = 60.0, min_n: int = 6
                ) -> tuple[float, float, float]:
    """外れ値に強い直線 y = a + b t（点が少ない・時間の幅が短ければ b = 0 で中央値）。

    Returns:
        (a, b, ばらつき（残差の MAD から出した標準偏差）)
    """
    t, y = np.asarray(t, float), np.asarray(y, float)
    if len(y) == 0:
        return 0.0, 0.0, float("nan")
    b = 0.0
    if len(y) >= min_n and np.ptp(t) >= min_span_s:
        # Theil–Sen：すべての組の傾きの中央値
        i, j = np.triu_indices(len(t), 1)
        ok = np.abs(t[j] - t[i]) > 0.25 * np.ptp(t)
        if ok.sum() >= 3:
            b = float(np.median((y[j] - y[i])[ok] / (t[j] - t[i])[ok]))
    a = float(np.median(y - b * t))
    r = y - a - b * t
    sd = float(1.4826 * np.median(np.abs(r - np.median(r)))) if len(y) > 1 else float("nan")
    return a, b, sd


def fit_audio_sync(pairs: list[tuple[float, float, float]], delay_diff_s: np.ndarray | None = None
                   ) -> tuple[SyncModel, float, int]:
    """音の組（``audio_pairs``）から時刻の関係を出す。

    Args:
        delay_diff_s: 組ごとの「音が 2 台目に届くまでの時間 − 1 台目に届くまでの時間」[s]（打点が分かっているとき。
            分からない組は NaN で、使わない）。None なら引かない（大まかな値）

    Returns:
        (モデル, 残差のばらつき [s], 使った組の数)
    """
    if not pairs:
        return SyncModel(), float("nan"), 0
    t = np.array([p[0] for p in pairs])
    d = np.array([p[1] for p in pairs])
    if delay_diff_s is not None:
        dd = np.asarray(delay_diff_s, float)
        ok = np.isfinite(dd)
        t, d = t[ok], d[ok] - dd[ok]
    a, b, sd = robust_line(t, d)
    return SyncModel(a, b), sd, len(t)


# =============================================================================
# 映像（2 台の視線のすき間）
# =============================================================================
def visible_points(det: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """検出結果から、見えているコマの時刻 [s] と画像の位置 [px]（時刻の順）。"""
    vis = det[(det.visible > 0) & det.x_px.notna() & det.y_px.notna()].sort_values("timestamp", kind="stable")
    return vis.timestamp.to_numpy(float), vis[["x_px", "y_px"]].to_numpy(float)


def despike(t: np.ndarray, uv: np.ndarray, ratio: float = 3.0, min_px: float = 8.0, max_gap: float = 1.6
            ) -> np.ndarray:
    """前後のコマから飛び離れた点（シャトルではない物の検出）を除く印（残す点が True）。

    続いた 3 コマで、真ん中の点が前後のどちらからも「前後の点どうしの距離 × ``ratio`` + ``min_px``」より離れていれば除く。
    なめらかに速く動く点は残る（真ん中は前後の間にある）。補間（``TrackInterpolator``）は 1 点の飛びで前後数コマが
    崩れるので、時刻合わせの前にかける。
    """
    keep = np.ones(len(t), bool)
    if len(t) < 3:
        return keep
    dt = np.diff(t)
    step = float(np.median(dt))
    a = np.linalg.norm(uv[1:-1] - uv[:-2], axis=1)
    b = np.linalg.norm(uv[1:-1] - uv[2:], axis=1)
    c = np.linalg.norm(uv[2:] - uv[:-2], axis=1)
    near = (dt[:-1] <= max_gap * step) & (dt[1:] <= max_gap * step)
    keep[1:-1] = ~(near & (np.minimum(a, b) > ratio * c + min_px))
    return keep


class TrackInterpolator:
    """1 台のカメラの検出を、時刻で補間する（続けて見えているコマの中だけ）。

    続けて 4 コマ以上見えている区間は 3 次のスプライン、2〜3 コマは直線。コマの間が ``max_gap`` × ふつうの間隔より
    空いた所はまたがない。打った瞬間は軌道が折れるので、その 1 区間の補間は合わないが、使う側が外れ値として扱う。
    """

    def __init__(self, t: np.ndarray, uv: np.ndarray, max_gap: float = 1.6):
        self.runs: list[tuple[float, float, Any]] = []
        if len(t) < 2:
            return
        dt = np.diff(t)
        step = float(np.median(dt))
        cuts = np.r_[0, np.flatnonzero((dt > max_gap * step) | (dt <= 0)) + 1, len(t)]
        for i0, i1 in zip(cuts[:-1], cuts[1:]):
            if i1 - i0 < 2:
                continue
            tt, pp = t[i0:i1], uv[i0:i1]
            f = CubicSpline(tt, pp) if i1 - i0 >= 4 else (lambda x, tt=tt, pp=pp: np.column_stack(
                [np.interp(x, tt, pp[:, 0]), np.interp(x, tt, pp[:, 1])]))
            self.runs.append((float(tt[0]), float(tt[-1]), f))
        self._lo = np.array([r[0] for r in self.runs])
        self._hi = np.array([r[1] for r in self.runs])

    def __call__(self, t: np.ndarray) -> np.ndarray:
        """時刻 ``t`` の画像の位置 (N, 2)。補間できない時刻は NaN。"""
        t = np.asarray(t, float)
        out = np.full((len(t), 2), np.nan)
        if not self.runs:
            return out
        k = np.searchsorted(self._lo, t, side="right") - 1
        ok = (k >= 0) & (t <= self._hi[np.clip(k, 0, len(self.runs) - 1)])
        for r in np.unique(k[ok]):
            m = ok & (k == r)
            out[m] = self.runs[r][2](t[m])
        return out


def ray_gap(cam_a: Camera, uv_a: np.ndarray, cam_b: Camera, uv_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2 台の視線がいちばん近づく所のすき間 [m] と、その真ん中の点 (N, 3) [m]（カメラの後ろなら NaN）。"""
    ca, da = cam_a.rays(uv_a)
    cb, db = cam_b.rays(uv_b)
    w = ca - cb
    b = np.sum(da * db, axis=1)
    d, e = da @ w, db @ w
    den = 1.0 - b * b
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (b * e - d) / den
        u = (e - b * d) / den
    pa, pb = ca + s[:, None] * da, cb + u[:, None] * db
    gap = np.linalg.norm(pa - pb, axis=1)
    bad = (den < 1e-9) | (s <= 0) | (u <= 0) | ~np.isfinite(gap)
    gap[bad] = np.nan
    mid = (pa + pb) / 2
    mid[bad] = np.nan
    return gap, mid


class RayPairs:
    """2 台の検出から、同じ時刻の視線の組を作る（fps の高い方を補間する）。時刻のずれを変えながら使う。"""

    def __init__(self, dets: list[pd.DataFrame], cams: list[Camera]):
        (t1, uv1), (t2, uv2) = visible_points(dets[0]), visible_points(dets[1])
        (t1, uv1), (t2, uv2) = ((t[m], uv[m]) for t, uv in ((t1, uv1), (t2, uv2)) for m in [despike(t, uv)])
        self.cams = cams
        step = [float(np.median(np.diff(t))) if len(t) > 1 else np.inf for t in (t1, t2)]
        # 補間するのはコマの間隔の短い方（補間の誤差が小さい）
        self.interp_cam2 = step[1] <= step[0]
        self.t1, self.uv1, self.t2, self.uv2 = t1, uv1, t2, uv2
        self.f = TrackInterpolator(t2, uv2) if self.interp_cam2 else TrackInterpolator(t1, uv1)
        self.step_s = float(min(step))

    def pairs(self, sync: SyncModel, mask: np.ndarray | None = None
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ずれ ``sync`` のときの (1 台目の時刻, 1 台目の位置, 2 台目の位置)。補間できない所は除く。

        ``mask`` は補間しない方のカメラの点の選び方。
        """
        if self.interp_cam2:
            t, uv = (self.t1, self.uv1) if mask is None else (self.t1[mask], self.uv1[mask])
            other = self.f(sync.to_cam2(t))
            ok = np.isfinite(other).all(axis=1)
            return t[ok], uv[ok], other[ok]
        t, uv = (self.t2, self.uv2) if mask is None else (self.t2[mask], self.uv2[mask])
        t1 = sync.to_cam1(t)
        other = self.f(t1)
        ok = np.isfinite(other).all(axis=1)
        return t1[ok], other[ok], uv[ok]

    def base_times(self, sync: SyncModel) -> np.ndarray:
        """補間しない方のカメラの点の、1 台目の時刻。"""
        return self.t1 if self.interp_cam2 else sync.to_cam1(self.t2)

    def gaps(self, sync: SyncModel, mask: np.ndarray | None = None) -> np.ndarray:
        _, a, b = self.pairs(sync, mask)
        if len(a) == 0:
            return np.empty(0)
        g, _ = ray_gap(self.cams[0], a, self.cams[1], b)
        return g[np.isfinite(g)]

    def cost(self, sync: SyncModel, mask: np.ndarray | None = None, scale_m: float = 0.1, min_pairs: int = 8
             ) -> float:
        """すき間の、外れ値に強い平均（0〜1。``scale_m`` より大きいすき間は 1 で頭打ち）。組が足りなければ inf。

        シャトルではない物の検出はすき間が何 m にもなるので、頭打ちにしないと 1 つで全体が決まってしまう。
        """
        g = self.gaps(sync, mask)
        if len(g) < min_pairs:
            return float("inf")
        return float(np.mean(g * g / (g * g + scale_m**2)))

    def points(self, sync: SyncModel, max_gap_m: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
        """同じ時刻の視線の交わる点（三角測量）の (1 台目の時刻, 3 次元の位置)。時刻の順。

        視線のすき間が ``max_gap_m`` より大きい組（どちらかがシャトルではない）は除く。
        """
        t, a, b = self.pairs(sync)
        if len(t) == 0:
            return t, np.empty((0, 3))
        g, mid = ray_gap(self.cams[0], a, self.cams[1], b)
        ok = np.isfinite(g) & (g <= max_gap_m)
        order = np.argsort(t[ok], kind="stable")
        return t[ok][order], mid[ok][order]


def _best_offset(rp: RayPairs, sync0: SyncModel, search_s: float, mask: np.ndarray | None = None,
                 step_s: float | None = None) -> tuple[float, float, float] | None:
    """``sync0`` のまわり ±``search_s`` で、すき間がいちばん小さくなるずれの直し [s]・そのすき間・鋭さ。"""
    step = step_s or max(rp.step_s / 4, 0.0005)
    grid = np.arange(-search_s, search_s + step / 2, step)
    c = np.array([rp.cost(sync0.shifted(d), mask) for d in grid])
    if not np.isfinite(c).any():
        return None
    i = int(np.nanargmin(np.where(np.isfinite(c), c, np.inf)))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    res = minimize_scalar(lambda d: rp.cost(sync0.shifted(d), mask), bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-5})
    d, cmin = (float(res.x), float(res.fun)) if res.fun <= c[i] else (float(grid[i]), float(c[i]))
    far = c[np.isfinite(c) & (np.abs(grid - d) > max(2 * step, 0.25 * search_s))]
    sharp = float(np.median(far) / max(cmin, 1e-6)) if len(far) else float("nan")
    return d, cmin, sharp


def video_sync(dets: list[pd.DataFrame], cams: list[Camera], sync0: SyncModel, search_s: float = 0.08,
               chunk_s: float = 10.0, min_chunk_pairs: int = 30, fit_drift: bool = True) -> SyncReport:
    """映像から時刻のずれを出す（``sync0`` のまわりを探す）。区間ごとのずれから、時計の進み方の違いとばらつきも出す。

    Args:
        dets: 2 台の検出結果（列 ``timestamp, x_px, y_px, visible``。時刻はそれぞれの動画の時刻）
        sync0: 大まかなずれ（音・手で指定）
        search_s: 探す幅 [s]
        chunk_s: ずれを別々に出す区間の長さ [s]
    """
    rp = RayPairs(dets, cams)
    best = _best_offset(rp, sync0, search_s)
    if best is None:
        return SyncReport(sync0, "none", float("nan"),
                          warnings=["2 台の両方にシャトルが写っているコマが足りず、映像から時刻を合わせられませんでした。"])
    model = sync0.shifted(best[0])
    # 区間ごとのずれ（時計の進み方の違い・ばらつき）
    tb = rp.base_times(model)
    edges = np.arange(tb.min(), tb.max() + chunk_s, chunk_s) if len(tb) else np.empty(0)
    local = max(4 * rp.step_s, 0.004)
    tc, dc = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (tb >= lo) & (tb < hi)
        if m.sum() < min_chunk_pairs:
            continue
        r = _best_offset(rp, model, local, m, step_s=local / 8)
        if r is not None and abs(r[0]) < local * 0.95:
            tc.append(float(np.mean(tb[m])))
            dc.append(r[0])
    tc, dc = np.array(tc), np.array(dc)
    sd = float("nan")
    if len(dc) >= 3:
        a, b, sd = robust_line(tc, dc) if fit_drift else (float(np.median(dc)), 0.0, float(np.std(dc)))
        # t2 = offset + (1 + drift) t1 の直し：区間のずれ d(t1) = a + b t1
        model = SyncModel(model.offset_s + a, model.drift + b)
        resid = dc - a - b * tc
        sigma = float(max(sd, 1e-5) / np.sqrt(len(dc)))
    else:
        # 区間が足りないときは、点を 4 つの組に分けて出したずれのばらつきから
        idx = np.arange(len(tb))
        parts = [_best_offset(rp, model, local, idx % 4 == k, step_s=local / 8) for k in range(4)]
        vals = [p[0] for p in parts if p is not None]
        resid = np.array(vals) - (np.mean(vals) if vals else 0.0)
        sigma = float(np.std(vals) / 2) if len(vals) >= 3 else rp.step_s / 4
    g = rp.gaps(model)
    # ばらつきからの見積もりは区間が少ないと小さく出すぎるので、コマの間隔の 1/40 を下限にする（合成データでの誤差ほど）
    rep = SyncReport(model, "video", sigma_s=float(max(sigma, rp.step_s / 40)), n_pairs=int(len(g)),
                     gap_cm=float(np.median(g) * 100) if len(g) else None, sharpness=best[2],
                     chunk_offsets_ms=[round(float(v) * 1000, 3) for v in resid])
    if best[2] == best[2] and best[2] < 1.5:
        rep.warnings.append("映像の時刻合わせがはっきり決まりませんでした（シャトルが両方に写るコマが少ないか、"
                            "キャリブレーションが合っていません）。")
    if abs(best[0]) > 0.9 * search_s:
        rep.warnings.append("時刻のずれが探す幅の端にあります。--sync-search を大きくしてやり直してください。")
    if rep.gap_cm is not None and rep.gap_cm > 15:
        rep.warnings.append(f"2 台の視線が {rep.gap_cm:.0f} cm 食い違っています。コートの合わせ（キャリブレーション）を"
                            "見直してください。")
    return rep
