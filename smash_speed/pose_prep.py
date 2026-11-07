"""姿勢の前処理（ROADMAP E1b）：ショット単位の切り出し、打点の時刻、顔のぼかし、左利きの反転、正規化。

1. 打点の時刻（``find_impact``）：利き腕の手首の速さ（画像の上）が最大の時刻を目安にし、その近く（±0.15 秒）に
   打球音があれば「打球音の時刻 − カメラまでの距離 ÷ 音速」を打点にする。音がなければ手首の最大の時刻に
   ``wrist_offset_s`` を足したもの（手首は打点の少し前に最も速くなる。手で付けた打点と比べて ``calibrate_wrist_offset`` で決める）。
2. 切り出し（``shot_window``）：打点の前 1.0 秒〜後 0.5 秒。
3. 左利きは画像を左右に反転し、関節の左右の名前を入れ替える（``mirror``）。右利きと同じ式で特徴量を出せる。
4. 正規化（``normalize``）：腰の中点を原点、上を +y、打つ向きを +x にし、胴体の長さ（肩の中点〜腰の中点の、区間の中央値）で割る。
   打つ向きは鼻が耳の中点より前にある向きから決める（側方の撮影なので、顔の向きで分かる）。

顔のぼかし（``blur_faces``）は、画面に出す・書き出す映像に使う（鼻・目・耳の周りをぼかす）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from smash_speed.pose import FLIP_INDEX, KP, PoseSequence

SPEED_OF_SOUND_MPS = 343.0
BEFORE_S = 1.0
AFTER_S = 0.5
AUDIO_SEARCH_S = 0.15     # 手首の速さの最大から、この範囲の打球音を同じスイングの音とみなす
FACE_JOINTS = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")


def shot_window(t_hit: float, before_s: float = BEFORE_S, after_s: float = AFTER_S) -> tuple[float, float]:
    """ショットを切り出す区間 [s]。"""
    return t_hit - before_s, t_hit + after_s


def _smooth(x: np.ndarray, n: int = 3) -> np.ndarray:
    """欠けを無視した移動平均（n コマ、中心）。"""
    if n <= 1:
        return x
    out = np.full_like(x, np.nan)
    h = n // 2
    for i in range(len(x)):
        w = x[max(0, i - h):i + h + 1]
        ok = np.isfinite(w)
        if ok.any():
            out[i] = w[ok].mean()
    return out


def wrist_speed(seq: PoseSequence, hand: str = "right", smooth: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """利き腕の手首の画像の上の速さ [px/s]（隣り合うコマの中点の時刻 (T−1,)、速さ (T−1,)）。"""
    p = seq.joint(f"{hand}_wrist")
    p = np.column_stack([_smooth(p[:, 0], smooth), _smooth(p[:, 1], smooth)])
    dt = np.diff(seq.t)
    v = np.linalg.norm(np.diff(p, axis=0), axis=1) / dt
    return (seq.t[1:] + seq.t[:-1]) / 2, v


@dataclass
class Impact:
    """打点の時刻の見積もり。``method`` は audio（打球音）| wrist（手首の速さ）。"""

    t: float
    method: str
    t_wrist: float | None = None
    t_audio: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"t_impact_s": self.t, "impact_method": self.method, "t_wrist_peak_s": self.t_wrist,
                "t_audio_s": self.t_audio}


def find_impact(seq: PoseSequence, audio_times: list[float] | None = None, hand: str = "right",
                sound_distance_m: float = 7.0, wrist_offset_s: float = 0.0, near_s: float | None = None,
                search_s: float = AUDIO_SEARCH_S) -> Impact | None:
    """打点の時刻 [s]。

    Args:
        seq: 姿勢の列（ショットの前後を含む）
        audio_times: 打球音の立ち上がりの時刻 [s]（``events.audio_onsets`` の時刻）。なければ手首だけで決める
        sound_distance_m: 打点からカメラ（マイク）までの距離 [m]。音が届くまでの遅れを引くのに使う
        wrist_offset_s: 手首の速さが最大の時刻から打点までの時間 [s]（音がないときに使う）
        near_s: 打点のおおよその時刻 [s]（分かれば、この ±0.3 秒の中で手首の最大を探す）
    """
    tm, v = wrist_speed(seq, hand)
    ok = np.isfinite(v)
    if near_s is not None:
        ok &= np.abs(tm - near_s) <= 0.3
    if not ok.any():
        return None
    i = int(np.flatnonzero(ok)[np.argmax(v[ok])])
    t_w = float(tm[i])
    # 最大の前後のコマの速さに放物線を当て、コマの間の時刻まで細かくする
    if 0 < i < len(v) - 1 and np.isfinite(v[i - 1:i + 2]).all():
        a, b, c = v[i - 1], v[i], v[i + 1]
        den = a - 2 * b + c
        if den < 0:
            t_w += float(np.clip(0.5 * (a - c) / den, -0.5, 0.5)) * float(tm[i + 1] - tm[i])
    if audio_times:
        cand = [ta for ta in audio_times if abs(ta - t_w) <= search_s]
        if cand:
            ta = min(cand, key=lambda x: abs(x - t_w))
            return Impact(t=ta - sound_distance_m / SPEED_OF_SOUND_MPS, method="audio", t_wrist=t_w, t_audio=ta)
    return Impact(t=t_w + wrist_offset_s, method="wrist", t_wrist=t_w)


def calibrate_wrist_offset(t_wrist: list[float], t_manual: list[float]) -> float:
    """手で付けた打点の時刻と手首の最大の時刻の差の中央値（``wrist_offset_s`` に使う）[s]。"""
    d = [m - w for w, m in zip(t_wrist, t_manual) if w is not None and m is not None]
    return float(np.median(d)) if d else 0.0


def impact_errors(t_est: list[float | None], t_manual: list[float], fps: float | None = None) -> dict[str, Any]:
    """見積もった打点の時刻と手で付けた打点の時刻の誤差（見積もり − 手）。

    Returns:
        ``errors_s``（ショットごと。見積もれなければ None）、``mean_abs_s``・``median_s``・``max_abs_s``、
        ``fps`` があれば ``mean_abs_frames``、``n``
    """
    errs = [None if e is None else float(e - m) for e, m in zip(t_est, t_manual)]
    val = np.array([e for e in errs if e is not None])
    out: dict[str, Any] = {"errors_s": errs, "n": int(len(val))}
    if len(val):
        out.update(mean_abs_s=float(np.mean(np.abs(val))), median_s=float(np.median(val)),
                   max_abs_s=float(np.max(np.abs(val))))
        if fps:
            out["mean_abs_frames"] = out["mean_abs_s"] * fps
    return out


def mirror(seq: PoseSequence) -> PoseSequence:
    """画像を左右に反転し、関節の左右を入れ替える（左利き → 右利きと同じ形）。2 回で元に戻る。"""
    xy = seq.xy[:, FLIP_INDEX].copy()
    xy[..., 0] = seq.width - xy[..., 0]
    meta = dict(seq.meta)
    meta["mirrored"] = not meta.get("mirrored", False)
    return seq.copy(xy=xy, score=seq.score[:, FLIP_INDEX].copy(), meta=meta)


def facing_sign(seq: PoseSequence) -> float:
    """打つ向きが画像の右なら +1、左なら −1（鼻が耳の中点より前にある向き。分からなければ +1）。"""
    ears = np.nanmean(np.stack([seq.joint("left_ear")[:, 0], seq.joint("right_ear")[:, 0]]), axis=0)
    d = seq.joint("nose")[:, 0] - ears
    d = d[np.isfinite(d)]
    if len(d) == 0 or np.median(d) == 0:
        return 1.0
    return float(np.sign(np.median(d)))


def _mid(seq: PoseSequence, a: str, b: str) -> np.ndarray:
    return (seq.joint(a) + seq.joint(b)) / 2


@dataclass
class NormalizedPose:
    """正規化した姿勢。

    Attributes:
        t: 時刻 (T,) [s]（打点が 0）
        xy: 関節の位置 (T, 17, 2)。腰の中点が原点、+x が打つ向き、+y が上、単位は胴体の長さ
        hip: 腰の中点の動き (T, 2)。最初のコマの腰の中点が原点、向き・単位は ``xy`` と同じ
        score: 信頼度 (T, 17)
        torso_px: 胴体の長さ [px]（区間の中央値）
        facing: 元の画像での打つ向き（+1 右、−1 左）
        fps: コマの速さ [Hz]
        meta: 元の列の情報
    """

    t: np.ndarray
    xy: np.ndarray
    hip: np.ndarray
    score: np.ndarray
    torso_px: float
    facing: float
    fps: float
    meta: dict[str, Any] = field(default_factory=dict)

    def joint(self, name: str) -> np.ndarray:
        return self.xy[:, KP[name]]

    def save(self, path) -> None:
        import json
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, t=self.t, xy=self.xy, hip=self.hip, score=self.score,
                            info=json.dumps({"torso_px": self.torso_px, "facing": self.facing, "fps": self.fps,
                                             "meta": self.meta}, ensure_ascii=False))

    @classmethod
    def load(cls, path) -> "NormalizedPose":
        import json
        with np.load(path, allow_pickle=False) as z:
            info = json.loads(str(z["info"]))
            return cls(t=z["t"], xy=z["xy"], hip=z["hip"], score=z["score"], torso_px=info["torso_px"],
                       facing=info["facing"], fps=info["fps"], meta=info.get("meta", {}))


def normalize(seq: PoseSequence, t_impact: float = 0.0, facing: float | None = None) -> NormalizedPose:
    """腰の中点を原点・胴体の長さを 1・打つ向きを +x・上を +y にする。時刻は打点を 0 にする。"""
    hip = _mid(seq, "left_hip", "right_hip")
    sh = _mid(seq, "left_shoulder", "right_shoulder")
    L = np.linalg.norm(sh - hip, axis=1)
    L = L[np.isfinite(L) & (L > 0)]
    if len(L) == 0:
        raise ValueError("肩と腰が見つかるコマがないため、正規化できません")
    torso = float(np.median(L))
    s = facing_sign(seq) if facing is None else float(facing)
    flip = np.array([s, -1.0])
    xy = (seq.xy - hip[:, None, :]) * flip / torso
    first = hip[np.isfinite(hip).all(axis=1)]
    hip_path = (hip - first[0]) * flip / torso if len(first) else np.full_like(hip, np.nan)
    fps = seq.fps or (1.0 / float(np.median(np.diff(seq.t))) if len(seq) > 1 else 0.0)
    meta = dict(seq.meta)
    # 打点での腰の中点の元の画像の位置と画像の大きさ（側方の撮影で、体から画像の縮尺を決めるのに使う。E6）
    ok = np.isfinite(hip).all(axis=1)
    if ok.any():
        meta["raw_hip_px"] = [float(np.interp(t_impact, seq.t[ok], hip[ok, c])) for c in range(2)]
    meta.setdefault("image_size", [seq.width, seq.height])
    return NormalizedPose(t=seq.t - t_impact, xy=xy, hip=hip_path, score=seq.score.copy(), torso_px=torso,
                          facing=s, fps=fps, meta=meta)


def prepare_shot(seq: PoseSequence, t_hit: float, left_handed: bool = False, audio_times: list[float] | None = None,
                 sound_distance_m: float = 7.0, wrist_offset_s: float = 0.0, before_s: float = BEFORE_S,
                 after_s: float = AFTER_S) -> tuple[NormalizedPose, Impact]:
    """1 本のショットの前処理：打点を決め直し、前後を切り出し、左利きなら反転し、正規化する。

    ``t_hit`` はおおよその打点の時刻 [s]（ショットの一覧の打球の時刻など）。``seq`` はその前後を含む姿勢の列。
    打点が見積もれなければ ``t_hit`` を使う。
    """
    if left_handed:
        seq = mirror(seq)
    imp = find_impact(seq, audio_times, "right", sound_distance_m, wrist_offset_s, near_s=t_hit)
    if imp is None:
        imp = Impact(t=t_hit, method="given")
    t0, t1 = shot_window(imp.t, before_s, after_s)
    norm = normalize(seq.slice(t0, t1), imp.t)
    norm.meta["left_handed"] = left_handed
    return norm, imp


def blur_faces(image: np.ndarray, xy: np.ndarray, score: np.ndarray, min_score: float = 0.2,
               scale: float = 1.8, ksize: int = 31) -> np.ndarray:
    """鼻・目・耳の周りをぼかした画像を返す（元の画像は変えない）。

    ``xy`` (P, 17, 2) か (17, 2)：写っている人ごとの関節。顔の点の広がりの ``scale`` 倍（小さすぎるときは
    肩幅・胴体から決めた大きさ）の楕円をぼかす。
    """
    out = image.copy()
    xy = np.asarray(xy, float).reshape(-1, len(KP), 2)
    score = np.asarray(score, float).reshape(-1, len(KP))
    mask = np.zeros(image.shape[:2], np.uint8)
    r_max = 0.0
    for p, s in zip(xy, score):
        pts = np.array([p[KP[n]] for n in FACE_JOINTS if s[KP[n]] >= min_score and np.isfinite(p[KP[n]]).all()])
        if len(pts) == 0:
            continue
        c = pts.mean(axis=0)
        r = float(np.max(np.linalg.norm(pts - c, axis=1))) if len(pts) > 1 else 0.0
        sh = [p[KP[n]] for n in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")]
        if np.isfinite(sh).all():
            r = max(r, 0.18 * float(np.linalg.norm((sh[0] + sh[1]) / 2 - (sh[2] + sh[3]) / 2)))
        r = max(r * scale, 6.0)
        r_max = max(r_max, r)
        cv2.ellipse(mask, (int(round(c[0])), int(round(c[1]))), (int(r), int(r * 1.25)), 0, 0, 360, 255, -1)
    if r_max > 0:
        # 顔が大きく写るほど強くぼかす（目鼻の形が分からない強さ）
        k = max(ksize, int(r_max)) | 1
        blurred = cv2.GaussianBlur(image, (k, k), 0)
        out[mask > 0] = blurred[mask > 0]
    return out
