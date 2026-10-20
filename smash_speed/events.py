"""イベント検出：打点（インパクト）時刻と着地時刻の推定。

打点時刻は初速の定義そのものに関わる。スマッシュ直後のシャトルは 1000 m/s^2 程度で減速するため、
打点時刻が 10 ms ずれると初速は 10 m/s ほど変わる。低 fps では打点がフレームの間に入るので、
次の手がかりをフレーム時刻より細かい精度で組み合わせる。

- ``audio``: 打球音の立ち上がり（48 kHz なら 0.02 ms 刻み）。最も精度が高い
- ``crossing``: 打点前に飛んできたシャトルの軌道と、打点後の軌道が交わる時刻
- ``first_frame``: 最初に検出されたフレームの直前（フレーム間隔の一様分布を仮定）
- ``manual``: 手動指定
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import butter, sosfiltfilt


@dataclass
class ImpactEstimate:
    """打点時刻の推定値。

    Attributes:
        t: 打点時刻 [s]（動画の実時間）
        sigma: 不確かさ（標準偏差）[s]
        method: 推定方法
    """

    t: float
    sigma: float
    method: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audio_onsets(x: np.ndarray, rate: int, highpass_hz: float = 2000.0, smooth_s: float = 0.0005,
                 k_mad: float = 10.0, min_interval_s: float = 0.15) -> list[tuple[float, float]]:
    """打球音などの鋭い音の立ち上がり時刻を検出する。

    高域通過フィルタ → 包絡線 → 中央値 + k × MAD を超えた区間ごとに、閾値を横切った時刻を
    サンプル間の線形補間で求める。

    Args:
        x: 波形
        rate: サンプリング周波数 [Hz]
        highpass_hz: 高域通過フィルタの遮断周波数 [Hz]（足音・声を除く）
        smooth_s: 包絡線の平滑化の幅 [s]
        k_mad: 閾値（雑音の MAD の何倍か）
        min_interval_s: これより近い立ち上がりはまとめる [s]

    Returns:
        (立ち上がり時刻 [s], 強さ（雑音比）) のリスト
    """
    if len(x) < rate * 0.01:
        return []
    sos = butter(4, highpass_hz, btype="highpass", fs=rate, output="sos")
    y = np.abs(sosfiltfilt(sos, x))
    n = max(1, int(round(smooth_s * rate)))
    env = np.convolve(y, np.ones(n) / n, mode="same")
    med = np.median(env)
    mad = np.median(np.abs(env - med)) + 1e-12
    thr = med + k_mad * mad
    above = env > thr
    rises = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    out: list[tuple[float, float]] = []
    gap = int(min_interval_s * rate)
    # 立ち上がりごとの強さ（次の立ち上がりまでの包絡線の最大値。``gap`` より先は見ない）
    ends = np.minimum(np.r_[rises[1:], len(env)], rises + gap)
    own = np.array([env[i:max(j, i + 1)].max() for i, j in zip(rises, ends)])
    k = 0
    while k < len(rises):
        # min_interval_s 以内に続く立ち上がりを 1 つにまとめる。時刻は、群の中でいちばん強い音の半分以上の
        # 強さがある最初の立ち上がりにする（打球音のあとの反響は後ろなので打球音が残り、打つ直前の
        # 振り・足音などの弱い音が打球音の直前にあっても、その時刻を打点にしない）
        g = k + 1
        while g < len(rises) and rises[g] - rises[k] < gap:
            g += 1
        grp = np.arange(k, g)
        peak = own[grp].max()
        i = rises[grp[np.argmax(own[grp] >= 0.5 * peak)]]
        a, b = env[i - 1], env[i]
        frac = (thr - a) / (b - a) if b != a else 0.0
        t = (i - 1 + frac) / rate
        out.append((float(t), float((peak - med) / mad)))
        k = g
    return out


def impact_from_audio(onsets: list[tuple[float, float]], t_first_detection: float,
                      max_lead_s: float = 0.2, tolerance_s: float = 0.005,
                      sigma_s: float = 0.002, av_offset_s: float = 0.0) -> ImpactEstimate | None:
    """最初の検出時刻の直前にある打球音を打点とする。

    Args:
        onsets: ``audio_onsets`` の結果（音声の時刻 [s]）
        t_first_detection: 打点後に最初にシャトルが検出された時刻 [s]
        max_lead_s: 打球音が最初の検出よりどれだけ前まで探すか [s]
        tolerance_s: 最初の検出より後でも許す幅（露光の中央時刻とのずれなど）[s]
        sigma_s: 打点時刻の不確かさ [s]
        av_offset_s: 音声 → 映像の時刻のずれ（映像の時刻 = 音声の時刻 + av_offset_s）[s]
    """
    cands = [(t + av_offset_s, s) for t, s in onsets
             if t_first_detection - max_lead_s <= t + av_offset_s <= t_first_detection + tolerance_s]
    if not cands:
        return None
    t, _ = max(cands, key=lambda c: (c[1], c[0]))  # 最も強い音（同程度なら遅い方）
    return ImpactEstimate(float(t), sigma_s, "audio")


def impact_from_first_frame(t_first_detection: float, frame_interval: float,
                            t_last_before: float | None = None) -> ImpactEstimate:
    """最初の検出の直前のフレーム間隔に打点が一様に分布すると仮定する。

    ``t_last_before`` は打点前にシャトルが最後に見えた時刻 [s]（あれば区間をそこまでに狭める）。
    """
    lo = t_first_detection - frame_interval
    if t_last_before is not None:
        lo = max(lo, t_last_before)
    width = t_first_detection - lo
    return ImpactEstimate((lo + t_first_detection) / 2, width / np.sqrt(12), "first_frame")


def impact_from_crossing(t_in: np.ndarray, uv_in: np.ndarray, outgoing_uv, t_lo: float, t_hi: float,
                         sigma_floor_s: float = 0.003) -> ImpactEstimate | None:
    """打点前（飛んできた）と打点後（打ち返した）の画像上の軌道が最も近づく時刻を打点とする。

    Args:
        t_in, uv_in: 打点前の観測（時刻 [s]、画像座標 [px]）。最後の数点を2次式で外挿する
        outgoing_uv: 時刻 [s] → 打点後の軌道の画像座標 [px] の関数（当てはめた軌道を逆向きに外挿）
        t_lo, t_hi: 探索範囲 [s]
        sigma_floor_s: 不確かさの下限 [s]
    """
    if len(t_in) < 3:
        return None
    tt, uv = np.asarray(t_in)[-5:], np.asarray(uv_in)[-5:]
    deg = 2 if len(tt) >= 4 else 1
    c = [np.polyfit(tt - tt[-1], uv[:, j], deg) for j in range(2)]

    def gap(t: float) -> float:
        a = np.array([np.polyval(cj, t - tt[-1]) for cj in c])
        return float(np.linalg.norm(a - outgoing_uv(t)))

    res = minimize_scalar(gap, bounds=(t_lo, t_hi), method="bounded", options={"xatol": 1e-5})
    if not res.success:
        return None
    # 交点での相対速度から、1 px の位置誤差が何秒に相当するかで不確かさを見積もる
    h = 1e-4
    rel_speed = abs(gap(res.x + h) - gap(res.x - h)) / (2 * h) + 1e-9
    sigma = max(sigma_floor_s, (res.fun + 2.0) / rel_speed)
    return ImpactEstimate(float(res.x), float(sigma), "crossing")


def impact_from_hitter_position(position_at, plane, t_lo: float, t_hi: float,
                                contact_offset_m: float = 0.3, sigma_m: float = 0.4
                                ) -> ImpactEstimate | None:
    """当てはめた軌道を逆向きにたどり、打者の位置（打点）を通過する時刻を打点とする。

    打点は打者の足元（飛行平面の原点）から飛行方向へ ``contact_offset_m`` [m] 前にあると仮定する。
    検出漏れで最初の数フレームが欠けても影響を受けない（低 fps で長い残像が写らない場合に有効）。

    Args:
        position_at: 時刻 [s] → 当てはめた軌道の位置 [m] の関数
        plane: 飛行平面（原点 = 打者の足元）
        t_lo, t_hi: 探索範囲 [s]
        contact_offset_m: 打点の、足元からの前方へのずれ [m]
        sigma_m: 打点の位置の不確かさ [m]
    """
    from scipy.optimize import brentq

    def ds(t: float) -> float:
        return float(plane.to_plane(position_at(t)[None])[0, 0] - contact_offset_m)

    try:
        a, b = ds(t_lo), ds(t_hi)
        if a > 0:  # 探索範囲の最初でもう打点を過ぎている
            return None
        # 最初の検出の時点でまだ打点に届いていない（打点の仮定位置より手前）なら、最初の検出時刻を打点とする
        t = t_hi if b < 0 else brentq(ds, t_lo, t_hi, xtol=1e-6)
    except ValueError:
        return None
    h = 1e-4
    v_s = abs(ds(t + h) - ds(t - h)) / (2 * h)
    return ImpactEstimate(float(t), float(sigma_m / max(v_s, 1e-6)), "hitter_position")


def landing_from_detections(t: np.ndarray, uv: np.ndarray, still_px: float = 3.0,
                            n_still: int = 3) -> float | None:
    """画像上でシャトルが止まった最初の時刻 [s]（``n_still`` フレーム続けて動きが ``still_px`` 以下）。"""
    if len(t) <= n_still:
        return None
    step = np.linalg.norm(np.diff(uv, axis=0), axis=1)
    for i in range(len(step) - n_still + 1):
        if np.all(step[i:i + n_still] <= still_px):
            return float(t[i])
    return None


def wrist_speed_peak(frames: list[np.ndarray], times: np.ndarray, right_handed: bool = True
                     ) -> float | None:
    """MediaPipe Pose で利き腕の手首の速さが最大になる時刻 [s] を返す（任意。未導入なら None）。

    姿勢推定は 2D の画像座標で行い、隣接フレームの手首の移動量 / 時間差が最大の区間の中央時刻を返す。
    """
    try:
        import mediapipe as mp  # noqa: F401
        from mediapipe.python.solutions import pose as mp_pose
    except ImportError:
        return None
    idx = mp_pose.PoseLandmark.RIGHT_WRIST if right_handed else mp_pose.PoseLandmark.LEFT_WRIST
    pts = []
    with mp_pose.Pose(static_image_mode=False, model_complexity=1) as pose:
        for img in frames:
            r = pose.process(img[:, :, ::-1])
            if r.pose_landmarks is None:
                pts.append((np.nan, np.nan))
                continue
            lm = r.pose_landmarks.landmark[idx]
            pts.append((lm.x * img.shape[1], lm.y * img.shape[0]))
    pts = np.asarray(pts)
    sp = np.linalg.norm(np.diff(pts, axis=0), axis=1) / np.diff(times)
    if not np.isfinite(sp).any():
        return None
    i = int(np.nanargmax(sp))
    return float((times[i] + times[i + 1]) / 2)
