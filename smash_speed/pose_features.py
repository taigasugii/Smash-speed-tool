"""フォームの特徴量（ROADMAP E2 打点の瞬間の姿勢、E3 動きの順番）。

入力は正規化した姿勢（``pose_prep.NormalizedPose``：腰の中点が原点、+x が打つ向き、+y が上、単位は胴体の長さ、
時刻は打点が 0。左利きは反転済みで、ラケットを持つ腕はいつも ``right``）。式と理由は ``docs/pose_features.md``。

- E2（``impact_features``）：打点の高さ・前後の位置（身長比）、肘・肩・膝の角度、手首の最大の速さ（身長/秒）、
  肩幅・腰幅の見かけの変化から出す回旋の角度、捻りの最大と戻りの速さ、非利き腕を引き込む時刻、打った後の戻りの速さ。
- E3（``sequence_features``）：腰（骨盤の回旋）→ 肩（胸の回旋）→ 肘（伸展）→ 手首（速さ）の最大の時刻とその順番、
  腰の中点の前への移動量・沈み込み・前へ動く速さの最大の時刻（体重移動）。

画像の上の 2 次元の値なので、カメラの向き（奥行き）の動きは見えない。肩幅・腰幅は、側方から見ると
体が横向き（打つ向きに肩の線がそろう）で最も広く、正面を向くと 0 に近づくので、回旋の角度の代わりに使う。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np
from scipy.signal import savgol_filter

from smash_speed.pose import KP
from smash_speed.pose_prep import NormalizedPose

# 身長に対する長さの比（Winter, Biomechanics and Motor Control of Human Movement の体節の比）
TORSO_RATIO = 0.288      # 肩の高さ 0.818 − 大転子（股関節）の高さ 0.530
THIGH_RATIO = 0.245      # 大転子 → 膝
SHANK_RATIO = 0.246      # 膝 → 足首
ANKLE_RATIO = 0.039      # 床 → 足首
SMOOTH_S = 0.05          # 速さ・角速度を出す前のなめらかにする幅 [s]
SWING_WINDOW = (-0.5, 0.1)  # 速さの最大を探す区間（打点を 0）[s]
CHAIN = ("pelvis", "trunk", "elbow", "wrist")
CHAIN_LABELS = {"pelvis": "腰", "trunk": "肩", "elbow": "肘", "wrist": "手首"}


@dataclass
class FormFeatures:
    """1 本のショットのフォームの特徴量。出せないものは NaN。時刻は打点を 0 とする [s]。"""

    # E2 打点の瞬間
    body_height_torso: float = np.nan     # 身長（胴体の長さの何倍か。ほかの値を身長比にするのに使う）
    wrist_height_ratio: float = np.nan    # 打点での手首の高さ ÷ 身長
    wrist_forward_ratio: float = np.nan   # 打点での手首の、腰の中点からの前後の位置 ÷ 身長（前が正）
    elbow_angle_deg: float = np.nan       # 打点での肘の角度（肩・肘・手首。180° でまっすぐ）
    shoulder_angle_deg: float = np.nan    # 打点での肩の角度（上腕と体の脇の線のなす角。180° で真上）
    front_knee_angle_deg: float = np.nan  # 打点での前の脚（非利き腕の側）の膝の角度
    back_knee_angle_deg: float = np.nan   # 打点での後ろの脚（利き腕の側）の膝の角度
    wrist_speed_max: float = np.nan       # 手首の速さの最大 [身長/秒]
    shoulder_width_impact: float = np.nan  # 打点での肩幅の見かけ ÷ 区間の最大（1 で横向き、0 で正面）
    hip_width_impact: float = np.nan
    trunk_turn_impact_deg: float = np.nan  # 打点での胸の回旋（横向きから何度回ったか）
    twist_max_deg: float = np.nan         # 捻り（腰の回旋 − 胸の回旋）の最大
    t_twist_max: float = np.nan
    twist_release_dps: float = np.nan     # 捻りが戻る速さの最大 [deg/s]
    t_free_arm_pull: float = np.nan       # 非利き腕の手首を下へ引き込む速さが最大の時刻
    recovery_speed: float = np.nan        # 打った後（0〜0.5 秒）の腰の中点の水平の速さの最大 [身長/秒]
    # E3 動きの順番
    t_pelvis_peak: float = np.nan         # 腰（骨盤）の回旋の角速度が最大の時刻
    t_trunk_peak: float = np.nan          # 肩（胸）の回旋の角速度が最大の時刻
    t_elbow_peak: float = np.nan          # 肘が伸びる角速度が最大の時刻
    t_wrist_peak: float = np.nan          # 手首の速さが最大の時刻
    pelvis_speed_dps: float = np.nan      # それぞれの最大の値 [deg/s]
    trunk_speed_dps: float = np.nan
    elbow_speed_dps: float = np.nan
    chain_order: str = ""                 # 最大の早い順（例 "pelvis>trunk>elbow>wrist"）
    chain_in_order: float = np.nan        # 腰 → 肩 → 肘 → 手首の順なら 1、違えば 0
    lag_pelvis_trunk: float = np.nan      # 隣り合う最大の時間差 [s]
    lag_trunk_elbow: float = np.nan
    lag_elbow_wrist: float = np.nan
    step_forward_ratio: float = np.nan    # 打点までの腰の中点の前への移動量 ÷ 身長
    hip_drop_ratio: float = np.nan        # 打点の前後の腰の中点の沈み込み ÷ 身長
    t_step_peak: float = np.nan           # 腰の中点が前へ動く速さが最大の時刻（体重移動）
    step_speed_max: float = np.nan        # その速さ [身長/秒]

    def to_dict(self) -> dict[str, Any]:
        return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in asdict(self).items()}


FEATURE_NAMES = [f.name for f in fields(FormFeatures)]
# 表・画面に出す名前と単位（人が読む順）
FEATURE_INFO: dict[str, tuple[str, str]] = {
    "wrist_height_ratio": ("打点の高さ（身長比）", ""),
    "wrist_forward_ratio": ("打点の前後（身長比、前が＋）", ""),
    "elbow_angle_deg": ("打点の肘の角度", "°"),
    "shoulder_angle_deg": ("打点の肩の角度", "°"),
    "front_knee_angle_deg": ("前の膝の角度", "°"),
    "back_knee_angle_deg": ("後ろの膝の角度", "°"),
    "wrist_speed_max": ("手首の速さの最大（身長/秒）", ""),
    "trunk_turn_impact_deg": ("打点での胸の回り", "°"),
    "twist_max_deg": ("捻りの最大", "°"),
    "twist_release_dps": ("捻りが戻る速さ", "°/秒"),
    "t_free_arm_pull": ("反対の腕を引く時刻", "秒"),
    "recovery_speed": ("打った後に戻る速さ（身長/秒）", ""),
    "t_pelvis_peak": ("腰が最も速く回る時刻", "秒"),
    "t_trunk_peak": ("肩が最も速く回る時刻", "秒"),
    "t_elbow_peak": ("肘が最も速く伸びる時刻", "秒"),
    "t_wrist_peak": ("手首が最も速い時刻", "秒"),
    "chain_in_order": ("腰→肩→肘→手首の順", ""),
    "step_forward_ratio": ("踏み込み（身長比）", ""),
    "hip_drop_ratio": ("沈み込み（身長比）", ""),
    "t_step_peak": ("体重移動が最も速い時刻", "秒"),
}


# --- 計算の部品 ----------------------------------------------------------------------

def _fill(x: np.ndarray, max_missing: float = 0.5) -> np.ndarray | None:
    """欠け（NaN）を前後から直線で埋める。半分より多く欠けていれば None。"""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.mean() < 1 - max_missing or ok.sum() < 2:
        return None
    if ok.all():
        return x.copy()
    idx = np.arange(len(x))
    return np.interp(idx, idx[ok], x[ok])


def _smooth(x: np.ndarray, fps: float) -> np.ndarray:
    n = max(5, int(round(SMOOTH_S * fps)) | 1)
    if len(x) <= n:
        return x
    return savgol_filter(x, n, 2)


def _series(n: NormalizedPose, x: np.ndarray) -> np.ndarray | None:
    """欠けを埋めてなめらかにした時間の列。"""
    f = _fill(x)
    return None if f is None else _smooth(f, n.fps)


def _deriv(n: NormalizedPose, x: np.ndarray) -> np.ndarray:
    return np.gradient(x, n.t)


def _at(n: NormalizedPose, x: np.ndarray, t: float = 0.0) -> float:
    """時刻 t の値（前後のコマから直線で）。"""
    if x is None or not (n.t[0] <= t <= n.t[-1]):
        return np.nan
    return float(np.interp(t, n.t, x))


def _peak(n: NormalizedPose, v: np.ndarray | None, window: tuple[float, float] = SWING_WINDOW
          ) -> tuple[float, float]:
    """区間の中で値が最大になる時刻と値（前後のコマに放物線を当ててコマの間まで細かくする）。"""
    if v is None:
        return np.nan, np.nan
    m = (n.t >= window[0]) & (n.t <= window[1]) & np.isfinite(v)
    if not m.any():
        return np.nan, np.nan
    idx = np.flatnonzero(m)
    i = int(idx[np.argmax(v[idx])])
    t, val = float(n.t[i]), float(v[i])
    if 0 < i < len(v) - 1:
        a, b, c = v[i - 1], v[i], v[i + 1]
        den = a - 2 * b + c
        if den < 0:
            d = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
            t += d * float(n.t[i + 1] - n.t[i])
            val = float(b - 0.25 * (a - c) * d)
    return t, val


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """点 b での角 a-b-c [deg]（(T, 2) どうし）。"""
    u, v = a - b, c - b
    cos = np.sum(u * v, axis=-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def body_height(n: NormalizedPose) -> float:
    """身長（胴体の長さの何倍か）。胴体・太もも・すねの長さの和を、身長に対する比の和で割る。

    太もも・すねは体の向きや曲げ方で見かけが縮むので、区間の中の長い方（90 パーセンタイル）を使う。
    """
    segs = []
    for side in ("left", "right"):
        hip, knee, ankle = n.joint(f"{side}_hip"), n.joint(f"{side}_knee"), n.joint(f"{side}_ankle")
        segs.append((np.linalg.norm(knee - hip, axis=1), np.linalg.norm(ankle - knee, axis=1)))
    thigh = np.concatenate([s[0] for s in segs])
    shank = np.concatenate([s[1] for s in segs])
    thigh, shank = thigh[np.isfinite(thigh)], shank[np.isfinite(shank)]
    if len(thigh) == 0 or len(shank) == 0:
        return np.nan
    return float((1.0 + np.percentile(thigh, 90) + np.percentile(shank, 90))
                 / (TORSO_RATIO + THIGH_RATIO + SHANK_RATIO))


def turn_angles(n: NormalizedPose, a: str) -> np.ndarray | None:
    """肩（腰）の線の見かけの幅から出す回旋の角度 [deg]：横向き（幅が最大）で 0、正面（幅 0）で 90。

    幅 w と区間の最大 W から arccos(w / W)。正面を越えて回った・向きの左右は区別できない。
    """
    w = _series(n, np.abs(n.joint(f"left_{a}")[:, 0] - n.joint(f"right_{a}")[:, 0]))
    if w is None:
        return None
    W = float(np.max(w))
    if W <= 0:
        return None
    return np.degrees(np.arccos(np.clip(w / W, 0.0, 1.0)))


# --- 特徴量 -----------------------------------------------------------------------

def form_features(n: NormalizedPose, hand: str = "right") -> FormFeatures:
    """E2・E3 の特徴量をすべて出す。``hand`` はラケットを持つ腕（反転済みなら right）。"""
    f = FormFeatures()
    other = "left" if hand == "right" else "right"
    H = body_height(n)
    f.body_height_torso = H

    sh, el, wr = n.joint(f"{hand}_shoulder"), n.joint(f"{hand}_elbow"), n.joint(f"{hand}_wrist")
    hip_same = n.joint(f"{hand}_hip")
    ankles = (n.joint("left_ankle") + n.joint("right_ankle")) / 2
    wx, wy = _series(n, wr[:, 0]), _series(n, wr[:, 1])

    # --- 打点の瞬間（E2）
    if np.isfinite(H) and wx is not None and wy is not None:
        ay = _series(n, ankles[:, 1])
        if ay is not None:
            f.wrist_height_ratio = (_at(n, wy) - _at(n, ay)) / H + ANKLE_RATIO
        f.wrist_forward_ratio = _at(n, wx) / H
    elbow = _series(n, angle_deg(sh, el, wr))
    f.elbow_angle_deg = _at(n, elbow)
    f.shoulder_angle_deg = _at(n, _series(n, angle_deg(el, sh, hip_same)))
    for attr, side in (("front_knee_angle_deg", other), ("back_knee_angle_deg", hand)):
        knee = angle_deg(n.joint(f"{side}_hip"), n.joint(f"{side}_knee"), n.joint(f"{side}_ankle"))
        setattr(f, attr, _at(n, _series(n, knee)))

    wrist_v = None
    if wx is not None and wy is not None:
        wrist_v = np.hypot(_deriv(n, wx), _deriv(n, wy))
        f.t_wrist_peak, vmax = _peak(n, wrist_v)
        f.wrist_speed_max = vmax / H if np.isfinite(H) else np.nan

    th_s, th_h = turn_angles(n, "shoulder"), turn_angles(n, "hip")
    if th_s is not None:
        f.shoulder_width_impact = float(np.cos(np.radians(_at(n, th_s))))
        f.trunk_turn_impact_deg = _at(n, th_s)
    if th_h is not None:
        f.hip_width_impact = float(np.cos(np.radians(_at(n, th_h))))
    if th_s is not None and th_h is not None:
        twist = th_h - th_s
        f.t_twist_max, f.twist_max_deg = _peak(n, twist, (SWING_WINDOW[0], 0.0))
        if np.isfinite(f.t_twist_max):
            _, f.twist_release_dps = _peak(n, -_deriv(n, twist), (f.t_twist_max, SWING_WINDOW[1]))

    fy = _series(n, n.joint(f"{other}_wrist")[:, 1] + n.hip[:, 1])
    if fy is not None:
        f.t_free_arm_pull, _ = _peak(n, -_deriv(n, fy), (-0.6, SWING_WINDOW[1]))

    hx, hy = _series(n, n.hip[:, 0]), _series(n, n.hip[:, 1])
    if hx is not None and hy is not None and np.isfinite(H):
        vh = np.hypot(_deriv(n, hx), _deriv(n, hy))
        _, rec = _peak(n, vh, (0.0, 0.5))
        f.recovery_speed = rec / H

    # --- 動きの順番（E3）
    if th_h is not None:
        f.t_pelvis_peak, f.pelvis_speed_dps = _peak(n, np.abs(_deriv(n, th_h)))
    if th_s is not None:
        f.t_trunk_peak, f.trunk_speed_dps = _peak(n, np.abs(_deriv(n, th_s)))
    if elbow is not None:
        f.t_elbow_peak, f.elbow_speed_dps = _peak(n, _deriv(n, elbow))
    times = {"pelvis": f.t_pelvis_peak, "trunk": f.t_trunk_peak, "elbow": f.t_elbow_peak, "wrist": f.t_wrist_peak}
    if all(np.isfinite(v) for v in times.values()):
        order = sorted(CHAIN, key=lambda k: times[k])
        f.chain_order = ">".join(order)
        f.chain_in_order = float(tuple(order) == CHAIN)
        f.lag_pelvis_trunk = times["trunk"] - times["pelvis"]
        f.lag_trunk_elbow = times["elbow"] - times["trunk"]
        f.lag_elbow_wrist = times["wrist"] - times["elbow"]

    if hx is not None and np.isfinite(H):
        pre = n.t <= 0
        if pre.any():
            f.step_forward_ratio = (_at(n, hx) - float(np.min(hx[pre]))) / H
        f.t_step_peak, vmax = _peak(n, _deriv(n, hx), (-0.6, SWING_WINDOW[1]))
        f.step_speed_max = vmax / H
    if hy is not None and np.isfinite(H):
        m = (n.t >= -0.6) & (n.t <= SWING_WINDOW[1])
        if m.any():
            f.hip_drop_ratio = (float(hy[n.t <= -0.6].mean() if (n.t <= -0.6).any() else hy[m][0])
                                - float(np.min(hy[m]))) / H
    return f
