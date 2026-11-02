"""クリックなしのキャリブレーションの候補：画像からコートの候補をいくつか出し、人は選ぶ（確かめる）だけにする。

1. 全自動（``AutoCourtCalibrator``）と同じ方法で、ラインの組から BWF のコートの形に合う候補を集める
   （全自動はいちばんよい 1 つだけを使うが、ラインの取り違えで 2 番目以降が正しいことがある）
2. 候補ごとに、四隅の予想位置を「おおよそのクリック」として半自動と同じ方法（``CourtSnapper.refine``）で
   ラインに吸い付けて詰める
3. 詰めたあとのラインらしさで並べ、同じコートになったものをまとめて、上から ``n`` 個を返す

どちらの端が打つ人の側（x < 0）かは画像だけでは決まらないので、カメラに近い端（画面の下の端）を
打つ人の側にする。カメラのおおよその位置（手順 ① の上から見た図）が x > 0 の側なら入れ替える。
座標は解析の座標系（ネット直下のコート中央が原点）[m]。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from smash_speed.calibration.auto_court_calibrator import (COURT_WIDTH_M, NET_X_M, AutoCourtCalibrator,
                                                           _apply_h)
from smash_speed.calibration.snap import HALF_L, HALF_W, CourtSnapper, SnapResult

# 解析の座標 → 全自動の座標（角が原点）
_TO_AUTO = np.array([[1.0, 0, NET_X_M], [0, 1.0, COURT_WIDTH_M / 2], [0, 0, 1.0]])
# コートを 180° 回す（打つ人の側を入れ替える）
_ROT180 = np.diag([-1.0, -1.0, 1.0])
_CORNERS = np.array([[-HALF_L, -HALF_W], [-HALF_L, HALF_W], [HALF_L, -HALF_W], [HALF_L, HALF_W]])


@dataclass
class Proposal:
    """キャリブレーションの候補 1 つ。

    Attributes:
        result: ラインに吸い付けた結果（``points`` を手順 3 の点としてそのまま使える）
        score: ラインらしさ（大きいほどよい。候補どうしの比べ合いにだけ使う）
        n_snapped: 2 本のラインとも測れたランドマークの数
    """

    result: SnapResult
    score: float
    n_snapped: int


def _orient(Hm: np.ndarray, shape: tuple[int, int], cam_xy: tuple[float, float] | None) -> np.ndarray:
    """カメラに近い端（画面の下の端）が打つ人の側（x < 0）になるように向きを決める。"""
    ends = _apply_h(Hm, np.array([[-HALF_L / 2, 0.0], [HALF_L / 2, 0.0]]))
    near_is_neg = ends[0, 1] >= ends[1, 1]
    if cam_xy is not None and cam_xy[0] > 0:
        near_is_neg = not near_is_neg
    return Hm if near_is_neg else Hm @ _ROT180


def candidate_homographies(frame: np.ndarray, max_candidates: int = 6) -> list[np.ndarray]:
    """全自動と同じ方法で集めたコートの候補（解析の座標 → 元の解像度の画像 [px] のホモグラフィ）。"""
    cal = AutoCourtCalibrator(debug=False)
    prep = cal._preprocess(frame)
    shape = prep["gray"].shape
    segs, seg_w, seg_c = cal._detect_segments(prep["gray"], prep["bgr"], None)
    if len(segs) < 6:
        return []
    vps = cal._find_vanishing_points(segs, seg_w, shape)
    if len(vps) < 2:
        return []
    cands: list[dict] = []
    for m in [np.ones(len(segs), bool)] + cal._color_groups(segs, seg_w, seg_c):
        if m.sum() >= 6:
            cands += cal._match_court(segs[m], seg_w[m], vps, shape)
    lab_s = cv2.cvtColor(cv2.bilateralFilter(prep["bgr"], 7, 25, 5), cv2.COLOR_BGR2LAB)
    for c in cands:
        c["verify"] = cal._verify(c["H"], lab_s, ref=cal._last_strength_ref)
    cands.sort(key=lambda c: -c["verify"]["score"])
    uniq: list[dict] = []
    for c in cands:
        if c["verify"]["score"] > 0 and not any(cal._same_court(c["H"], u["H"], shape) for u in uniq):
            uniq.append(c)
        if len(uniq) >= max_candidates:
            break
    S = np.diag([1 / prep["scale"], 1 / prep["scale"], 1.0])
    out = []
    for c in uniq:
        Hm = S @ c["H"] @ _TO_AUTO
        out.append(Hm / Hm[2, 2])
    return out


def propose_calibrations(frame: np.ndarray, table: dict[str, np.ndarray], n: int = 3,
                         cam_xy: tuple[float, float] | None = None,
                         max_candidates: int = 6) -> list[Proposal]:
    """クリックなしで、キャリブレーションの候補をよい順に最大 ``n`` 個返す（見つからなければ空）。

    Args:
        table: ランドマーク名 → 実寸座標 [m]
        cam_xy: カメラのおおよその位置（解析の座標系、上から見た [m]）。どちらの端を打つ人の側にするかだけに使う
        max_candidates: ラインに吸い付けて詰める候補の数
    """
    h, w = frame.shape[:2]
    k = max(h, w) / 1920.0
    snapper = CourtSnapper()
    lab = snapper.prepare(frame)
    img1 = cv2.GaussianBlur(lab, (0, 0), 1.0 * k) * snapper.LAB_W
    full = snapper._model_points()
    found: list[Proposal] = []
    for H0 in candidate_homographies(frame, max_candidates):
        H0 = _orient(H0, (h, w), cam_xy)
        P = _apply_h(H0, _CORNERS)
        if not np.isfinite(P).all():
            continue
        try:
            res = snapper.refine(frame, H0, _CORNERS, P, table, sigma=np.full(4, 4.0), lab=lab)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if res.color is None:
            continue
        score = snapper._court_score(img1, res.homography, full, res.color, k)
        found.append(Proposal(res, float(score), sum(s == "snapped" for s in res.status.values())))
    # 詰めたら同じコートになったものは、よい方だけ残す
    found.sort(key=lambda p: -p.score)
    inv = np.linalg.inv(_TO_AUTO)
    out: list[Proposal] = []
    for p in found:
        if not any(AutoCourtCalibrator._same_court(p.result.homography @ inv, q.result.homography @ inv, (h, w))
                   for q in out):
            out.append(p)
    return out[:n]
