"""合成の骨格の動き（側方から撮ったフォアハンドのオーバーヘッドのスイング。ROADMAP E 章の正解用）。

関節の位置が分かっている 3 次元の骨格を、スイングの角度・タイミングを決めた値（``SwingParams``）から作り、
側方のカメラ（``side_camera``）で画像に写す。姿勢推定の精度（E1 の PCK）・特徴量（E2・E3）・
類似度スコア（E5）の計算が、正解と合うかを確かめるのに使う。実際の人の動きをまねたものではなく、
「どの関節がいつ・どれだけ動いたか」が式で分かることを優先した単純な形にしている。

座標はコートと同じく z が上 [m]。打つ向きを +x、打つ人の右手側を −y とする（右利き）。
カメラはラケットを持つ腕の側（−y）に置き、+y を向く。画像の右が +x（打つ向き）になる。

体の動き（時刻はインパクトを 0 とする [s]）：

- 腰の中点：前へ ``step_m`` だけ踏み込み（中心 ``t_step``）、打った後に ``recover_m`` だけ戻る（中心 ``t_recover``）。
  踏み込みの中心で腰を ``dip_m`` だけ沈める。
- 骨盤・胸の向き：横向き（打つ向きから −90°）から、ロジスティック関数で正面へ回る。角速度が最大になるのは
  ``t_pelvis``・``t_trunk``。向き ψ は打つ向き（+x）から測る（左回りが正）。
- ラケットを持つ腕（右腕）：上腕と前腕は、肩を通る鉛直な面（x-z の面）の中で動く（腕の面を打つ向きに固定した単純化）。
  上腕の向きは真下から前回りに測った角度 φ（90° が前、180° が真上、230° が後ろ上）で、``t_arm`` を中心に
  ``arm_start_deg`` から ``arm_impact_deg`` へ回る。肘の角度（肩・肘・手首のなす角）は ``t_elbow`` を中心に
  ``elbow_start_deg`` から ``elbow_impact_deg`` へ伸びる。
- 反対の腕（左腕）：前に上げた所（``free_arm_start_deg``）から、``t_pull`` を中心に体の横へ引き込む。
- 足首は腰の中点から前後に決まった距離に置き、膝は腰と足首の間で前に曲がる位置（2 本の棒の逆運動学）にする。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from smash_speed.camera import Camera, look_at
from smash_speed.pose import KP, NUM_KEYPOINTS, PoseSequence

# 身長に対する長さ・高さの比（人体計測の平均的な値。単純化のため丸めている）
ANKLE_Z = 0.039
HIP_Z = 0.50              # 少し膝を曲げて構えた腰の高さ
TORSO = 0.288             # 腰の中点 → 肩の中点
SHOULDER_HALF = 0.10      # 肩の関節の間の半分
HIP_HALF = 0.055          # 股関節の間の半分
UPPER_ARM = 0.186
FOREARM = 0.146
THIGH = 0.245
SHANK = 0.246
NOSE_UP = 0.11            # 肩の中点 → 鼻（上）
NOSE_FWD = 0.05           # 頭の中心 → 鼻（前）
EAR_HALF = 0.07
EYE_HALF = 0.03


@dataclass(frozen=True)
class SwingParams:
    """合成のスイングの決め方。時刻はインパクトを 0 とする [s]、角度 [deg]、長さ [m]。"""

    height_m: float = 1.70
    t_pelvis: float = -0.15        # 骨盤の回転の角速度が最大の時刻
    t_trunk: float = -0.10         # 胸（肩）の回転の角速度が最大の時刻
    t_elbow: float = -0.05         # 肘が伸びる角速度が最大の時刻
    t_arm: float = -0.03           # 上腕を振り上げる角速度が最大の時刻
    t_pull: float = -0.12          # 反対の腕を引き込む速さが最大の時刻
    t_step: float = -0.12          # 前への踏み込みの速さが最大の時刻
    t_recover: float = 0.25        # 打った後に戻る速さが最大の時刻
    tau_rot: float = 0.04          # 回転の速さ（ロジスティック関数の幅）[s]
    tau_arm: float = 0.03
    tau_step: float = 0.06
    pelvis_start_deg: float = -80.0
    pelvis_end_deg: float = 0.0
    trunk_start_deg: float = -90.0
    trunk_end_deg: float = 0.0
    arm_start_deg: float = 230.0
    arm_impact_deg: float = 160.0
    elbow_start_deg: float = 70.0
    elbow_impact_deg: float = 160.0
    free_arm_start_deg: float = 150.0
    free_arm_end_deg: float = 20.0
    step_m: float = 0.40
    recover_m: float = 0.60
    dip_m: float = 0.08

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sig(t: np.ndarray, center: float, tau: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(t - center) / tau))


def _ramp(t: np.ndarray, a: float, b: float, center: float, tau: float, t_end: float | None = None) -> np.ndarray:
    """a から b へロジスティック関数で移る値。``t_end`` を渡すと、その時刻で b になるよう全体を伸ばす。"""
    s = _sig(t, center, tau)
    if t_end is not None:
        s0, s1 = _sig(np.array(-10.0), center, tau), _sig(np.array(t_end), center, tau)
        s = (s - s0) / (s1 - s0)
    return a + (b - a) * s


def angles(p: SwingParams, t: np.ndarray) -> dict[str, np.ndarray]:
    """時刻 t [s] の各部の角度 [deg] と腰の位置 [m]（正解）。肘・上腕はインパクトでちょうど決めた値になる。"""
    t = np.asarray(t, float)
    return {
        "pelvis": _ramp(t, p.pelvis_start_deg, p.pelvis_end_deg, p.t_pelvis, p.tau_rot),
        "trunk": _ramp(t, p.trunk_start_deg, p.trunk_end_deg, p.t_trunk, p.tau_rot),
        "arm": _ramp(t, p.arm_start_deg, p.arm_impact_deg, p.t_arm, p.tau_arm, t_end=0.0),
        "elbow": _ramp(t, p.elbow_start_deg, p.elbow_impact_deg, p.t_elbow, p.tau_arm, t_end=0.0),
        "free_arm": _ramp(t, p.free_arm_start_deg, p.free_arm_end_deg, p.t_pull, p.tau_arm),
        "hip_x": p.step_m * _sig(t, p.t_step, p.tau_step) + p.recover_m * _sig(t, p.t_recover, p.tau_step),
        "hip_dip": p.dip_m * np.exp(-0.5 * ((t - p.t_step - 0.05) / 0.08) ** 2),
    }


def _yaw(psi_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """向き ψ の前の向き f と左の向き l（水平の単位ベクトル (T, 3)）。"""
    r = np.radians(psi_deg)
    z = np.zeros_like(r)
    f = np.stack([np.cos(r), np.sin(r), z], axis=-1)
    left = np.stack([-np.sin(r), np.cos(r), z], axis=-1)
    return f, left


def _sagittal(phi_deg: np.ndarray) -> np.ndarray:
    """x-z の面の中で、真下から前回りに φ 回した向き (T, 3)。"""
    r = np.radians(phi_deg)
    return np.stack([np.sin(r), np.zeros_like(r), -np.cos(r)], axis=-1)


def _knee(hip: np.ndarray, ankle: np.ndarray, thigh: float, shank: float) -> np.ndarray:
    """腰と足首から、前（+x）に曲がった膝の位置（届かないときは伸ばし切る）。"""
    d = ankle - hip
    L = np.linalg.norm(d, axis=-1, keepdims=True)
    L = np.minimum(L, thigh + shank - 1e-6)
    u = d / np.linalg.norm(d, axis=-1, keepdims=True)
    a = (thigh**2 - shank**2 + L**2) / (2 * L)
    h = np.sqrt(np.maximum(thigh**2 - a**2, 0.0))
    # 脚の向きに直交し、x-z の面の中で前を向く向き
    perp = np.stack([-u[:, 2], np.zeros(len(u)), u[:, 0]], axis=-1)
    perp *= np.where(perp[:, :1] < 0, -1.0, 1.0)
    return hip + a * u + h * perp


def joints_3d(p: SwingParams, t: np.ndarray, origin: np.ndarray = np.zeros(3)) -> np.ndarray:
    """時刻 t [s]（インパクトが 0）の関節の 3 次元の位置 (T, 17, 3) [m]。``origin`` は構えた時の腰の中点の真下の床。"""
    t = np.atleast_1d(np.asarray(t, float))
    H = p.height_m
    a = angles(p, t)
    T = len(t)
    X = np.zeros((T, NUM_KEYPOINTS, 3))
    hip_mid = np.zeros((T, 3))
    hip_mid[:, 0] = a["hip_x"]
    hip_mid[:, 2] = HIP_Z * H - a["hip_dip"]
    hip_mid += origin
    _, lp = _yaw(a["pelvis"])
    ft, lt = _yaw(a["trunk"])
    X[:, KP["left_hip"]] = hip_mid + HIP_HALF * H * lp
    X[:, KP["right_hip"]] = hip_mid - HIP_HALF * H * lp
    sh_mid = hip_mid + np.array([0.0, 0.0, TORSO * H])
    X[:, KP["left_shoulder"]] = sh_mid + SHOULDER_HALF * H * lt
    X[:, KP["right_shoulder"]] = sh_mid - SHOULDER_HALF * H * lt
    # ラケットを持つ腕（右）
    up = _sagittal(a["arm"])
    fore = _sagittal(a["arm"] + (180.0 - a["elbow"]))
    X[:, KP["right_elbow"]] = X[:, KP["right_shoulder"]] + UPPER_ARM * H * up
    X[:, KP["right_wrist"]] = X[:, KP["right_elbow"]] + FOREARM * H * fore
    # 反対の腕（左）：肘を軽く曲げたまま引き込む
    up_l = _sagittal(a["free_arm"])
    fore_l = _sagittal(a["free_arm"] - 20.0)
    X[:, KP["left_elbow"]] = X[:, KP["left_shoulder"]] + UPPER_ARM * H * up_l
    X[:, KP["left_wrist"]] = X[:, KP["left_elbow"]] + FOREARM * H * fore_l
    # 頭：胸の半分だけ打つ向きへ回す
    fh, lh = _yaw(0.5 * a["trunk"])
    head = sh_mid + np.array([0.0, 0.0, NOSE_UP * H])
    X[:, KP["nose"]] = head + NOSE_FWD * H * fh
    X[:, KP["left_eye"]] = head + NOSE_FWD * 0.8 * H * fh + EYE_HALF * H * lh + [0, 0, 0.02 * H]
    X[:, KP["right_eye"]] = head + NOSE_FWD * 0.8 * H * fh - EYE_HALF * H * lh + [0, 0, 0.02 * H]
    X[:, KP["left_ear"]] = head + EAR_HALF * H * lh
    X[:, KP["right_ear"]] = head - EAR_HALF * H * lh
    # 足：右足（後ろ）と左足（前）。足首は腰の中点に対して決まった位置
    for side, dx in (("left", 0.20), ("right", -0.25)):
        hip = X[:, KP[f"{side}_hip"]]
        ankle = hip.copy()
        ankle[:, 0] = hip_mid[:, 0] + dx * H / 1.7
        ankle[:, 2] = ANKLE_Z * H + origin[2]
        X[:, KP[f"{side}_ankle"]] = ankle
        X[:, KP[f"{side}_knee"]] = _knee(hip, ankle, THIGH * H, SHANK * H)
    return X


def side_camera(distance_m: float = 7.0, height_m: float = 1.3, target_x_m: float = 0.0, yaw_deg: float = 0.0,
                width: int = 1920, height: int = 1080, hfov_deg: float = 69.0) -> Camera:
    """側方のカメラ（ROADMAP E 章の撮影の想定）。打つ人の右手側 ``distance_m`` [m]、高さ ``height_m`` [m]、水平。

    ``yaw_deg`` はカメラの向きを打つ向きに直交からずらす角度（置き方のずれを調べる用）。
    ``hfov_deg`` は横の画角（スマホの標準のカメラは 1080p の横向きで約 65〜75°）。
    """
    f = (width / 2) / np.tan(np.radians(hfov_deg) / 2)
    eye = np.array([target_x_m, -distance_m, height_m])
    r = np.radians(yaw_deg)
    fwd = np.array([np.sin(r), np.cos(r), 0.0])
    return look_at(eye, eye + fwd, f, width, height)


def swing_times(fps: float = 60.0, before_s: float = 1.0, after_s: float = 0.5, phase: float = 0.0) -> np.ndarray:
    """インパクトを 0 とした、コマの時刻 [s]。``phase``（0〜1）でコマの位置をインパクトからずらす。"""
    n0 = int(np.floor(before_s * fps))
    n1 = int(np.floor(after_s * fps))
    return (np.arange(-n0, n1 + 1) + phase) / fps


def project_swing(p: SwingParams, camera: Camera, t: np.ndarray, noise_px: float = 0.0,
                  rng: np.random.Generator | None = None, t_offset: float = 0.0,
                  model: str = "synthetic") -> PoseSequence:
    """スイングを画像に写した姿勢の列（偽物の姿勢推定の結果）。``noise_px`` は関節の位置に足す揺れ（標準偏差）。"""
    X = joints_3d(p, t)
    T = len(t)
    xy = camera.project(X.reshape(-1, 3)).reshape(T, NUM_KEYPOINTS, 2)
    if noise_px > 0:
        rng = rng or np.random.default_rng(0)
        xy = xy + rng.normal(0.0, noise_px, xy.shape)
    score = np.ones((T, NUM_KEYPOINTS))
    fps = 1.0 / float(np.median(np.diff(t))) if T > 1 else 0.0
    return PoseSequence(t=np.asarray(t, float) + t_offset, xy=xy, score=score, width=camera.width,
                        height=camera.height, fps=fps, model=model)


def vary(p: SwingParams, **changes: float) -> SwingParams:
    """一部の値を変えたスイング。"""
    return replace(p, **changes)


def wrist_speed_peak_time(p: SwingParams, t: np.ndarray | None = None) -> float:
    """3 次元の手首の速さが最大になる時刻 [s]（正解。インパクトの前 0.5 秒〜後 0.1 秒）。"""
    t = np.arange(-0.5, 0.1, 0.0005) if t is None else t
    X = joints_3d(p, t)[:, KP["right_wrist"]]
    v = np.linalg.norm(np.gradient(X, t, axis=0), axis=1)
    return float(t[int(np.argmax(v))])
