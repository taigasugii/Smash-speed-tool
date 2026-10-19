"""コートのラインからカメラを自動で推定する（クリック不要のキャリブレーション）。

色に頼らず、床の明るさの変化（エッジ）だけでバドミントンのラインを探す。体育館の床にはバスケット・
バレーなどのラインや床板の継ぎ目も混ざるので、次の順に絞り込む。

1. 前処理と線分検出：グレースケール → CLAHE → Bilateral Filter → Canny、LSD で線分を抽出し、
   Canny のエッジに乗っている線分だけを残す。``line_color_hint`` があればその色の線分を重く見る。
2. 消失点：線分の交点を RANSAC で調べ、主要な方向（消失点）を 3 つまで求める。床の 2 方向の組
   （長手・短手）ごとに、どちらの消失点にも ±``angle_tol_deg`` 以内で向かわない線分（斜めの線・
   曲線の断片）と、地平線より上の線分（天井・壁）を捨てる。残りを 2 群に分け、同じ直線上の断片をつなぐ。
3. コートの同定：地平線と平行な補助線の上では、同じ方向のラインの位置は実寸の 1 次関数になる
   （無限遠が無限遠に写るため）。各群で「2 本のラインと 2 つの実寸」の組から 1 次関数を作り、
   ほかのラインがいくつ BWF の位置（0.46 m・1.98 m などの間隔）に乗るかで候補を選ぶ。
   2 群の候補を組み合わせるとホモグラフィが決まるので、全ラインを画像に描いて、検出した線分に
   どれだけ重なるかで最良の組を選ぶ（他競技のラインは間隔が合わないので選ばれない）。
4. 測り直し：原寸の画像で、各ラインに垂直な輝度の断面からラインの中心をサブピクセルで求め、
   選手で隠れた部分を外れ値として除いて直線を当てはめる。ライン同士の交点から ``solvePnP``
   （内部パラメータが未知なら焦点距離も推定）とホモグラフィを求め、再投影誤差から信頼度を出す。

座標系（このモジュール内）：原点 P1 をコートの角（バックバウンダリーラインとサイドラインの外側の縁の
交点）、X を長手、Y を短手、Z を鉛直上向きとする [m]。解析で使うコート中央原点の座標系とは
``x = X - 6.70``・``y = Y - 3.05`` の関係にあり、X = 0 の側を打つ人の側（``h``）とみなす。
コートは 180° 回しても同じ形なので、どちらの端が打つ人の側かは画像だけでは決まらない。既定では
カメラに近い端（横から撮った場合は画面の左の端）を X = 0 とし、``rotate_180`` で入れ替えられる。
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------- BWF の公称寸法
COURT_LENGTH_M = 13.40
COURT_WIDTH_M = 6.10
NET_X_M = 6.70
NET_POST_HEIGHT_M = 1.55
NET_CENTER_HEIGHT_M = 1.524
LINE_WIDTH_M = 0.04

# 短手方向に伸びるライン（X = 一定）：名前、外側の縁の位置 [m]、ラインの中心が縁からどちらにあるか
# （ラインはそれが区切る区域に含まれる。例：ショートサービスラインはサービスコートの側にある）
X_LINES: Tuple[Tuple[str, float, int], ...] = (
    ("back_h", 0.00, +1), ("long_service_h", 0.76, +1), ("short_service_h", 4.72, -1),
    ("short_service_o", 8.68, +1), ("long_service_o", 12.64, -1), ("back_o", 13.40, -1))
# 長手方向に伸びるライン（Y = 一定）
Y_LINES: Tuple[Tuple[str, float, int], ...] = (
    ("side_n", 0.00, +1), ("singles_n", 0.46, +1), ("center", 3.05, 0),
    ("singles_p", 5.64, -1), ("side_p", 6.10, -1))

# 画像の外にある点・ラインの長さの範囲 [m]（センターラインはショートサービスラインの間にはない）
_CENTER_SPANS = ((0.0, 4.72), (8.68, 13.40))

# 四隅（P1〜P4）と、対応する解析用ランドマーク名
CORNERS: Tuple[Tuple[str, float, float, str], ...] = (
    ("P1", 0.00, 0.00, "back_corner_h_n"), ("P2", 13.40, 0.00, "back_corner_o_n"),
    ("P3", 13.40, 6.10, "back_corner_o_p"), ("P4", 0.00, 6.10, "back_corner_h_p"))


def _center_value(edge: float, side: int) -> float:
    return edge + side * LINE_WIDTH_M / 2


X_CENTERS = np.array([_center_value(e, s) for _, e, s in X_LINES])
Y_CENTERS = np.array([_center_value(e, s) for _, e, s in Y_LINES])
X_EDGES = np.array([e for _, e, _ in X_LINES])
Y_EDGES = np.array([e for _, e, _ in Y_LINES])


def landmark_name(ix: int, iy: int) -> Optional[str]:
    """X ラインと Y ラインの交点の名前（解析用ランドマークの名前と同じ）。交点がなければ None。"""
    xname, yname = X_LINES[ix][0], Y_LINES[iy][0]
    kind, end = xname.rsplit("_", 1)                        # back / long_service / short_service, h / o
    if yname == "center":
        if kind == "short_service":
            return f"short_service_center_{end}"
        return f"{kind}_center_{end}"
    ykind, yside = yname.split("_")                         # side / singles, n / p
    base = {("back", "side"): "back_corner", ("back", "singles"): "back_singles",
            ("long_service", "side"): "long_service", ("long_service", "singles"): "long_service_singles",
            ("short_service", "side"): "short_service_side",
            ("short_service", "singles"): "short_service_singles"}[(kind, ykind)]
    return f"{base}_{end}_{yside}"


def to_analysis_xyz(XYZ: np.ndarray) -> np.ndarray:
    """このモジュールの座標（角が原点）→ 解析で使う座標（ネット直下のコート中央が原点）。"""
    XYZ = np.asarray(XYZ, float)
    return XYZ - np.array([NET_X_M, COURT_WIDTH_M / 2, 0.0])[: XYZ.shape[-1]]


@dataclass
class CalibrationResult:
    """自動キャリブレーションの結果。座標はこのモジュールの座標系（角 P1 が原点）[m]、画像は元の解像度 [px]。"""

    success: bool
    homography_matrix: Optional[np.ndarray] = None      # (3, 3) 床面 (X, Y, 1) [m] → 画像 [px]
    rvec: Optional[np.ndarray] = None                   # (3, 1)
    tvec: Optional[np.ndarray] = None                   # (3, 1)
    corners_2d: Optional[np.ndarray] = None             # (4, 2) P1, P2, P3, P4
    keypoints_2d: Optional[Dict[str, Tuple[float, float]]] = None  # 測った交点（ライン外側の縁の角）
    reprojection_error_rms: float = float("inf")       # [px]
    confidence_score: float = 0.0                       # 0〜1
    debug_image: Optional[np.ndarray] = None
    camera_matrix: Optional[np.ndarray] = None          # (3, 3) 使った（推定した）内部パラメータ
    dist_coeffs: Optional[np.ndarray] = None
    message: str = ""
    image_size: Tuple[int, int] = (0, 0)                # (width, height)
    lines_2d: Dict[str, np.ndarray] = field(default_factory=dict)       # ライン名 → 画像の直線 (a, b, c)
    line_points: Dict[str, np.ndarray] = field(default_factory=dict)    # ライン名 → 測ったラインの中心点 (N, 2)
    coverage: float = 0.0                               # 見えているはずのラインのうち確認できた割合

    @property
    def camera_center(self) -> Optional[np.ndarray]:
        """カメラの位置 [m]（このモジュールの座標系）。"""
        if self.rvec is None or self.tvec is None:
            return None
        R = cv2.Rodrigues(np.asarray(self.rvec, float))[0]
        return (-R.T @ np.asarray(self.tvec, float).reshape(3)).ravel()


# ============================================================================ 幾何の小道具
def _hline(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """2 点を通る直線（同次座標、(a, b) が単位ベクトル）。"""
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    one = np.ones(p.shape[:-1] + (1,))
    l = np.cross(np.concatenate([p, one], -1), np.concatenate([q, one], -1))
    return l / np.maximum(np.linalg.norm(l[..., :2], axis=-1, keepdims=True), 1e-12)


def _fit_line(pts: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
    """点群に直線を当てはめる（重みつき全最小二乗）。"""
    w = np.ones(len(pts)) if w is None else np.asarray(w, float)
    m = (pts * w[:, None]).sum(0) / w.sum()
    d = pts - m
    cov = (d * w[:, None]).T @ d
    _, vec = np.linalg.eigh(cov)
    n = vec[:, 0]                                         # 最小固有値の方向 = 法線
    return np.array([n[0], n[1], -n @ m])


def _intersect(l1: np.ndarray, l2: np.ndarray) -> np.ndarray:
    x = np.cross(l1, l2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return x[..., :2] / x[..., 2:3]


def _apply_h(H: np.ndarray, XY: np.ndarray) -> np.ndarray:
    XY = np.atleast_2d(np.asarray(XY, float))
    p = np.c_[XY, np.ones(len(XY))] @ H.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return p[:, :2] / p[:, 2:3]


def _focal_from_h(H: np.ndarray, c: np.ndarray) -> Optional[float]:
    """主点を c としたときの床のホモグラフィから焦点距離を求める（Zhang の拘束、正方画素）。"""
    T = np.array([[1, 0, -c[0]], [0, 1, -c[1]], [0, 0, 1]], float)
    A = T @ H
    h1, h2 = A[:, 0], A[:, 1]
    # (h1x h2x + h1y h2y) / f^2 + h1z h2z = 0、(|h1xy|^2 - |h2xy|^2) / f^2 + h1z^2 - h2z^2 = 0
    a = np.array([h1[0] * h2[0] + h1[1] * h2[1], h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2])
    b = np.array([h1[2] * h2[2], h1[2] ** 2 - h2[2] ** 2])
    den = float(a @ a)
    if den <= 0:
        return None
    x = -float(a @ b) / den                               # x = 1 / f^2
    if not np.isfinite(x) or x <= 0:
        return None
    return float(1.0 / math.sqrt(x))


def _court_segments_world() -> list[tuple[str, np.ndarray, np.ndarray]]:
    """全ラインの中心線（名前、始点、終点）[m]。"""
    segs = []
    for (name, _, _), X in zip(X_LINES, X_CENTERS):
        segs.append((name, np.array([X, 0.0]), np.array([X, COURT_WIDTH_M])))
    for (name, _, _), Y in zip(Y_LINES, Y_CENTERS):
        spans = _CENTER_SPANS if name == "center" else ((0.0, COURT_LENGTH_M),)
        for x0, x1 in spans:
            segs.append((name, np.array([x0, Y]), np.array([x1, Y])))
    return segs


# ============================================================================ 本体
class AutoCourtCalibrator:
    """バドミントンコートのラインを自動で見つけ、カメラの姿勢とホモグラフィを推定する。

    Args:
        process_max_side: 線分検出・候補探しをする画像の長辺 [px]（測り直しは元の解像度で行う）
        angle_tol_deg: 消失点の方向からこの角度以上ずれた線分を捨てる [deg]
        min_segment_frac: 画像の対角線に対するこの割合より短い線分は使わない
        max_hypotheses: 各方向で残すラインの対応づけの候補の数
        max_pairs: 2 次元で照合する（短手・長手の）候補の組の数
        fov_deg: 内部パラメータが未知で、床だけから焦点距離が決まらないときに仮定する水平画角 [deg]
        orientation: どちらの端を X = 0（打つ人の側）にするか。"near" = カメラに近い端
        max_roll_deg: カメラの左右の傾きの上限 [deg]（地平線がこれ以上傾く消失点の組は床とみなさない）
        debug: 途中経過の画像（``debug_image``）を作るか
    """

    def __init__(self, process_max_side: int = 1920, angle_tol_deg: float = 5.0,
                 min_segment_frac: float = 0.015, max_hypotheses: int = 150, max_pairs: int = 3000, fov_deg: float = 60.0,
                 orientation: str = "near", max_roll_deg: float = 30.0, debug: bool = True,
                 seed: int = 0) -> None:
        self.process_max_side = process_max_side
        self.angle_tol_deg = angle_tol_deg
        self.min_segment_frac = min_segment_frac
        self.max_hypotheses = max_hypotheses
        self.max_pairs = max_pairs
        self.fov_deg = fov_deg
        self.orientation = orientation
        self.max_roll_deg = max_roll_deg
        self.debug = debug
        self.rng = np.random.default_rng(seed)
        self._samples = self._model_samples()

    # ------------------------------------------------------------------ 入口
    def calibrate(self, frame: np.ndarray, camera_matrix: Optional[np.ndarray] = None,
                  dist_coeffs: Optional[np.ndarray] = None,
                  line_color_hint: Optional[str] = None) -> CalibrationResult:
        """1 枚の画像（BGR）からコートを見つけ、カメラを推定する。

        失敗しても例外は出さず、``success = False`` と理由（``message``）を返す。途中まで取れていれば
        ``keypoints_2d``（手動で直すときの初期位置）も入れる。
        """
        H_img, W_img = frame.shape[:2]
        prep = self._preprocess(frame)
        shape = prep["gray"].shape
        segs, seg_w, seg_c = self._detect_segments(prep["gray"], prep["bgr"], line_color_hint)
        ref = self._last_strength_ref
        empty = CalibrationResult(False, image_size=(W_img, H_img), camera_matrix=camera_matrix,
                                  dist_coeffs=dist_coeffs)
        if len(segs) < 6:
            empty.message = f"直線がほとんど見つかりません（{len(segs)} 本）"
            return empty
        vps = self._find_vanishing_points(segs, seg_w, shape)
        if len(vps) < 2:
            empty.message = "床のラインの向き（消失点）が 2 つ見つかりません"
            return empty
        # 全部の線分で 1 回、帯の色ごとの線分で 1 回ずつ、コートの候補を探す
        cands: list[dict] = []
        for m in [np.ones(len(segs), bool)] + self._color_groups(segs, seg_w, seg_c):
            if m.sum() >= 6:
                cands += self._match_court(segs[m], seg_w[m], vps, shape)
        lab_s = cv2.cvtColor(cv2.bilateralFilter(prep["bgr"], 7, 25, 5), cv2.COLOR_BGR2LAB)
        for c in cands:
            c["verify"] = self._verify(c["H"], lab_s, ref=ref)
        cands.sort(key=lambda c: -c["verify"]["score"])
        uniq: list[dict] = []
        for c in cands:
            if not any(self._same_court(c["H"], u["H"], shape) for u in uniq):
                uniq.append(c)
        cands = [c for c in uniq if c["verify"]["score"] > 0]
        if not cands:
            empty.message = "BWF のコートの形に合うラインの組が見つかりません"
            empty.debug_image = self._debug_image(prep, segs, None, None) if self.debug else None
            return empty

        lab_full = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        lab_full = cv2.cvtColor(cv2.GaussianBlur(lab_full, (0, 0), 0.8), cv2.COLOR_BGR2LAB).astype(np.float32)
        S = np.diag([1 / prep["scale"], 1 / prep["scale"], 1.0])
        Si = np.linalg.inv(S)
        best = None
        for cand in cands[:4]:
            col = cand["verify"]["color"]
            fit = self._refine_lines(lab_full, S @ cand["H"], col)
            if fit is None:
                continue
            fit = self._refine_lines(lab_full, fit["H"], col, narrow=True) or fit
            v = self._verify(Si @ fit["H"], lab_s, ref=ref)
            fit["verify"] = v
            if best is None or v["score"] > best[0]:
                best = (v["score"], fit, cand)
        if best is None:
            res = self._partial_result(S @ cands[0]["H"], W_img, H_img, camera_matrix, dist_coeffs)
            res.message = "ラインを原寸で測り直せませんでした（ラインが細すぎるか、隠れています）"
            res.debug_image = self._debug_image(prep, segs, cands[0], None) if self.debug else None
            return res
        _, fit, cand = best
        lines, pts = fit["lines"], fit["points"]
        res = self._solve_from_lines(lines, pts, W_img, H_img, camera_matrix, dist_coeffs, fit["H"])
        if res.success and self.orientation == "near":
            res = self._orient_near(res)
        # 実際のラインで確かめられた割合が低い、または別の候補も同じくらい合うなら信頼度を下げる
        found = fit["verify"]["found"]
        res.coverage = float(found)
        if res.success:
            res.confidence_score = round(res.confidence_score * float(np.interp(found, [0.15, 0.45], [0.5, 1.0])), 3)
        others = [c for c in cands if not self._same_court(c["H"], Si @ fit["H"], shape)]
        if others and others[0]["verify"]["score"] > 0.85 * fit["verify"]["score"]:
            res.confidence_score = round(res.confidence_score * 0.75, 3)
            res.message = (res.message + " 別のラインの組でも同じくらい説明できます。").strip()
        if self.debug:
            res.debug_image = self._debug_image(prep, segs, cand, res)
        return res

    def rotate_180(self, result: CalibrationResult) -> CalibrationResult:
        """打つ人の側（X = 0 の端）を反対の端に入れ替えた結果を返す（コートを 180° 回す）。"""
        if not result.lines_2d:
            return result
        swap = {}
        for names in ([n for n, _, _ in X_LINES], [n for n, _, _ in Y_LINES]):
            swap.update({a: b for a, b in zip(names, names[::-1])})
        lines = {swap[k]: v for k, v in result.lines_2d.items()}
        pts = {swap[k]: v for k, v in result.line_points.items()}
        W, H = result.image_size
        Hm = result.homography_matrix
        R180 = np.array([[-1, 0, COURT_LENGTH_M], [0, -1, COURT_WIDTH_M], [0, 0, 1]], float)
        out = self._solve_from_lines(lines, pts, W, H, result.camera_matrix, result.dist_coeffs,
                                     None if Hm is None else Hm @ R180, fixed_K=result.camera_matrix)
        out.debug_image = result.debug_image
        out.message = result.message
        out.confidence_score = result.confidence_score
        return out

    # ------------------------------------------------------------------ a. 前処理と線分
    def _preprocess(self, frame: np.ndarray) -> dict:
        h, w = frame.shape[:2]
        s = min(1.0, self.process_max_side / max(h, w))
        bgr = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if s < 1.0:
            bgr = cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        gray = cv2.bilateralFilter(gray, 7, 25, 5)
        return {"bgr": bgr, "gray": gray, "scale": s}

    @staticmethod
    def _color_mask(bgr: np.ndarray, hint: str) -> Optional[np.ndarray]:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        ranges = {
            "white": [((0, 0, 160), (180, 60, 255))],
            "yellow": [((15, 70, 90), (40, 255, 255))],
            "orange": [((5, 90, 90), (20, 255, 255))],
            "green": [((40, 50, 40), (90, 255, 255))],
            "blue": [((90, 70, 40), (135, 255, 255))],
            "red": [((0, 90, 60), (8, 255, 255)), ((170, 90, 60), (180, 255, 255))],
        }.get(hint.lower())
        if ranges is None:
            return None
        m = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in ranges:
            m |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        return cv2.dilate(m, np.ones((3, 3), np.uint8))

    def _detect_segments(self, gray: np.ndarray, bgr: np.ndarray,
                         hint: Optional[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """線分 (N, 4)、重み (N,)、帯の色 (N, 3)。Canny のエッジに乗っている LSD の線分だけを残す。"""
        h, w = gray.shape
        diag = math.hypot(w, h)
        edges = cv2.Canny(gray, 25, 75)
        edges = cv2.dilate(edges, np.ones((5, 5), np.uint8))
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        found = lsd.detect(gray)[0]
        if found is None:
            return np.zeros((0, 4)), np.zeros(0), np.zeros((0, 3))
        seg = found.reshape(-1, 4).astype(float)
        L = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
        seg = seg[L >= self.min_segment_frac * diag]
        if not len(seg):
            return seg, np.zeros(0), np.zeros((0, 3))
        s = np.linspace(0, 1, 16)
        pts = seg[:, None, :2] + s[None, :, None] * (seg[:, None, 2:] - seg[:, None, :2])
        ui = np.clip(np.round(pts[..., 0]).astype(int), 0, w - 1)
        vi = np.clip(np.round(pts[..., 1]).astype(int), 0, h - 1)
        support = (edges[vi, ui] > 0).mean(axis=1)
        keep = support >= 0.6
        seg, ui, vi = seg[keep], ui[keep], vi[keep]
        # 帯状のライン（両側が同じ床）らしさを、明るさと色み（Lab の a, b）の両方で測る。特定の色は仮定しない。
        # 床板の継ぎ目や明るさの段差は明るさだけが変わり、塗ったラインは色みも床と違うことが多い
        lab = cv2.cvtColor(cv2.bilateralFilter(bgr, 7, 25, 5), cv2.COLOR_BGR2LAB)
        strength, color, offset = self._stripe_strength(lab, seg)
        # 線分を帯の中心へ移す
        dvec = seg[:, 2:] - seg[:, :2]
        dvec /= np.maximum(np.linalg.norm(dvec, axis=1, keepdims=True), 1e-9)
        nvec = np.c_[-dvec[:, 1], dvec[:, 0]]
        seg = seg + np.c_[nvec, nvec] * offset[:, None]
        ref = max(float(np.percentile(strength, 97)) * 0.5, 1e-6) if len(strength) else 1.0
        self._last_strength_ref = 2 * ref * 0.8          # はっきりしたラインの帯の強さの目安（_verify で使う）
        weight = 0.1 + 0.9 * np.clip(strength / ref, 0.0, 1.0)
        if hint:
            mask = self._color_mask(bgr, hint)
            if mask is not None:
                frac = (mask[vi, ui] > 0).mean(axis=1)
                Ls = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
                # 指定した色の線分が十分にあるときだけ使う（色の指定が違っていても動くように）
                if (Ls * (frac > 0.3)).sum() > 0.25 * Ls.sum():
                    weight = weight * (0.3 + 1.7 * frac)
        return seg, weight, color

    # Lab の差の重み（明るさの差は色みの差より軽く見る）
    LAB_W = np.array([0.35, 1.0, 1.0], np.float32)

    @classmethod
    def _stripe_strength(cls, lab: np.ndarray, seg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """線分の近くに「帯」（中央と両側の色の差）がどれだけはっきりあるか (N,) と、その差のベクトル (N, 3)。

        差のベクトル（Lab、明るさは軽く重みづけ）の向きが、ラインの塗料の色と床の色の関係を表す。

        LSD の線分は帯の縁にあるので、垂直方向に少しずらしながら、幅を変えて最大の応答を探す。帯の中心までの
        ずれ (N,) [px] も返す（近くの太いラインは両側の縁が別々の線分になるので、中心に寄せてまとめる）。
        """
        if not len(seg):
            return np.zeros(0), np.zeros((0, 3)), np.zeros(0)
        g = lab.astype(np.float32)
        p, q = seg[:, :2], seg[:, 2:]
        d = (q - p) / np.maximum(np.linalg.norm(q - p, axis=1, keepdims=True), 1e-9)
        n = np.c_[-d[:, 1], d[:, 0]]
        s = np.linspace(0.1, 0.9, 9)
        pts = p[:, None] + s[None, :, None] * (q - p)[:, None]                  # (N, 9, 2)
        off = np.arange(-24, 24.5, 0.5)
        mx = (pts[..., 0:1] + off * n[:, None, 0:1]).reshape(-1, len(off)).astype(np.float32)
        my = (pts[..., 1:2] + off * n[:, None, 1:2]).reshape(-1, len(off)).astype(np.float32)
        prof = cv2.remap(g, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) * cls.LAB_W
        cs = np.concatenate([np.zeros((len(prof), 1, 3), np.float32), np.cumsum(prof, axis=1)], axis=1)
        K = len(off)
        idx = np.arange(K)
        best = np.zeros(len(prof))
        bcol = np.zeros((len(prof), 3), np.float32)
        boff = np.zeros(len(prof))
        for hw in (1, 2, 4, 6, 9, 13, 18):                                     # 半幅 0.5〜9 px

            def box(a, b):
                a, b = np.clip(a, 0, K), np.clip(b, 0, K)
                return (cs[:, b] - cs[:, a]) / np.maximum(b - a, 1)[None, :, None]

            cen = box(idx - hw, idx + hw + 1)
            left = box(idx - 3 * hw - 1, idx - hw)
            right = box(idx + hw + 1, idx + 3 * hw + 2)
            r = np.maximum(0.0, np.linalg.norm(cen - 0.5 * (left + right), axis=2)
                           - 0.6 * np.linalg.norm(left - right, axis=2))
            valid = (idx - 3 * hw - 1 >= 0) & (idx + 3 * hw + 2 <= K) & (np.abs(off) <= hw * 0.5 + 3)
            r = np.where(valid[None], r, 0)
            j = np.argmax(r, axis=1)
            rb = r[np.arange(len(r)), j]
            better = rb > best
            best = np.where(better, rb, best)
            bcol[better] = (cen - 0.5 * (left + right))[np.arange(len(r)), j][better]
            boff[better] = off[j][better]
        strength = np.median(best.reshape(len(seg), -1), axis=1)
        color = np.median(bcol.reshape(len(seg), -1, 3), axis=1)
        offset = np.median(boff.reshape(len(seg), -1), axis=1)
        return strength, color, offset

    def _color_groups(self, seg: np.ndarray, weight: np.ndarray, sig: np.ndarray,
                      max_groups: int = 6) -> list[np.ndarray]:
        """はっきりした帯の線分を、帯と床の色の差の向きでまとめる（コートのラインは 1 色で塗られているため）。

        色の値は仮定せず、多い向きから順に最大 max_groups 個の群を作る。各群は、その向きに近い線分のマスク。
        """
        L = np.linalg.norm(seg[:, 2:] - seg[:, :2], axis=1)
        strong = weight >= 0.5
        if strong.sum() < 6:
            return []
        u = sig / np.maximum(np.linalg.norm(sig, axis=1, keepdims=True), 1e-9)
        ws = L * weight
        left = strong.copy()
        groups = []
        total = ws[strong].sum()
        while left.sum() >= 6 and len(groups) < max_groups:
            d = self._dominant_direction(sig[left], ws[left])
            near = (u @ d) > 0.9
            if (ws * near * left).sum() < 0.03 * total:
                break
            groups.append(near & (weight >= 0.2))
            left &= (u @ d) <= 0.8
        return groups

    # ------------------------------------------------------------------ b. 消失点
    def _vp_residual(self, vps: np.ndarray, mids: np.ndarray, dirs: np.ndarray) -> np.ndarray:
        """線分の向きと「線分の中点 → 消失点」の向きの角度 [rad]、(V, N)。消失点は無限遠でもよい。"""
        d = vps[:, None, :2] - mids[None, :, :] * vps[:, None, 2:3]
        nd = np.linalg.norm(d, axis=2)
        cos = np.abs(np.einsum("vnk,nk->vn", d, dirs)) / np.maximum(nd, 1e-12)
        return np.arccos(np.clip(cos, 0.0, 1.0))

    def _find_vanishing_points(self, segs: np.ndarray, seg_w: np.ndarray,
                               shape: tuple[int, int], k: int = 3) -> list[np.ndarray]:
        """主要な消失点を RANSAC で k 個まで求める（同次座標、ノルム 1）。"""
        p, q = segs[:, :2], segs[:, 2:]
        L = np.linalg.norm(q - p, axis=1)
        dirs = (q - p) / L[:, None]
        mids = (p + q) / 2
        lines = _hline(p, q)
        w = L * seg_w
        remaining = np.ones(len(segs), bool)
        vps: list[np.ndarray] = []
        tol = math.radians(1.5)
        for _ in range(k):
            idx = np.flatnonzero(remaining)
            if len(idx) < 4:
                break
            pr = w[idx] / w[idx].sum()
            n_pairs = 1500
            a = self.rng.choice(idx, n_pairs, p=pr)
            b = self.rng.choice(idx, n_pairs, p=pr)
            ok = a != b
            V = np.cross(lines[a[ok]], lines[b[ok]])
            V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
            best_v, best_s = None, 0.0
            for c0 in range(0, len(V), 300):
                Vc = V[c0:c0 + 300]
                r = self._vp_residual(Vc, mids[idx], dirs[idx])
                sc = ((r < tol) * w[idx][None]).sum(1)
                j = int(np.argmax(sc))
                if sc[j] > best_s:
                    best_s, best_v = float(sc[j]), Vc[j]
            if best_v is None:
                break
            v = best_v
            for _ in range(3):                       # 内点で最小二乗に詰める
                r = self._vp_residual(v[None], mids, dirs)[0]
                inl = remaining & (r < math.radians(2.0))
                if inl.sum() < 2:
                    break
                M = (lines[inl] * w[inl, None]).T @ lines[inl]
                _, vec = np.linalg.eigh(M)
                v = vec[:, 0]
            r = self._vp_residual(v[None], mids, dirs)[0]
            if (w * (remaining & (r < math.radians(2.0)))).sum() < 0.03 * w.sum():
                break
            vps.append(v / np.linalg.norm(v))
            remaining &= r >= math.radians(self.angle_tol_deg)
        return vps

    # ------------------------------------------------------------------ 線の群とその位置
    def _merge_collinear(self, segs: np.ndarray, w: np.ndarray, tol_px: float) -> list[dict]:
        """同じ直線上にある線分をつなぐ。各直線は (a, b, c)、長さ、端点を持つ。"""
        p, q = segs[:, :2], segs[:, 2:]
        L = np.linalg.norm(q - p, axis=1)
        order = np.argsort(-L)
        groups: list[dict] = []
        glines = np.zeros((0, 3))
        for i in order:
            if len(groups):
                d = np.maximum(np.abs(glines @ np.r_[p[i], 1.0]), np.abs(glines @ np.r_[q[i], 1.0]))
                j = int(np.argmin(d))
                if d[j] < tol_px:
                    g = groups[j]
                    g["idx"].append(i)
                    pts = np.r_[segs[g["idx"], :2], segs[g["idx"], 2:]]
                    ww = np.r_[L[g["idx"]] * w[g["idx"]], L[g["idx"]] * w[g["idx"]]]
                    g["line"] = _fit_line(pts, ww)
                    glines[j] = g["line"]
                    continue
            groups.append({"idx": [i], "line": _hline(p[i], q[i])})
            glines = np.vstack([glines, groups[-1]["line"]])
        out = []
        for g in groups:
            ii = g["idx"]
            out.append({"line": g["line"], "length": float((L[ii] * w[ii]).sum()),
                        "pts": np.r_[segs[ii, :2], segs[ii, 2:]]})
        return out

    @staticmethod
    def _line_mid(m: dict) -> np.ndarray:
        """つないだ直線の上で、元の線分の重心にいちばん近い点。"""
        c = m["pts"].mean(0)
        l = m["line"]
        return c - (l @ np.r_[c, 1.0]) * l[:2]

    @staticmethod
    def _transversal(vp: np.ndarray, horizon: np.ndarray, shape: tuple[int, int],
                     floor_pt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ラインの位置を測る補助線（点 c0、向き d、同次の直線）。

        ラインとできるだけ直角に交わるように、床の上の点 c0 から消失点へ向かう方向に垂直にとる
        （浅い角度で交わると、位置のわずかな違いが補助線の上で大きく引き伸ばされるため）。
        """
        c0 = floor_pt
        vd = vp[:2] - c0 * vp[2]
        vd /= max(np.linalg.norm(vd), 1e-12)
        d = np.array([-vd[1], vd[0]])
        return c0, d, _hline(c0, c0 + d)

    @staticmethod
    def _coord(lines: np.ndarray, T: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        """直線が補助線と交わる位置 [px]（補助線に沿った符号つきの距離）。平行なら inf。"""
        c0, d, Tl = T
        x = np.cross(np.atleast_2d(lines), Tl)
        t = np.full(len(x), np.inf)
        ok = np.abs(x[:, 2]) > 1e-9 * np.linalg.norm(x[:, :2], axis=1)
        t[ok] = ((x[ok, :2] / x[ok, 2:3]) - c0) @ d
        return t

    # ------------------------------------------------------------------ c. コートの同定
    def _model_samples(self) -> dict:
        """コートの全ラインの中心線上の点（0.1 m おき）。kind 0 = 長手方向のライン、1 = 短手方向。"""
        XY, kind, seg_id = [], [], []
        for k, (name, a, b) in enumerate(_court_segments_world()):
            n = max(2, int(np.ceil(np.linalg.norm(b - a) / 0.1)) + 1)
            s = np.linspace(0, 1, n)[:, None]
            XY.append(a + s * (b - a))
            kind += [0 if a[1] == b[1] else 1] * n
            seg_id += [k] * n
        XY = np.vstack(XY)
        seg_id = np.array(seg_id)
        last = np.r_[seg_id[1:] != seg_id[:-1], True]
        return {"XY": XY, "kind": np.array(kind), "seg": seg_id, "last": last}

    def _hyp_1d(self, t: np.ndarray, wt: np.ndarray, mids: np.ndarray, model: np.ndarray, t_h: float,
                tol: float, vp: np.ndarray, T: tuple) -> list[dict]:
        """1 方向のラインの位置 t（補助線上）を、実寸の位置 model に対応づける候補。

        t = (r t_h m + q) / (r m + 1)（t_h = 地平線の位置。無限遠なら t = p m + q）を 2 本の対応から作り、
        ほかに何本 BWF の位置に乗るかで並べる。乗っているかは、予想したライン（消失点を通る）と、
        検出したラインの中点との距離 [px] で判定する（補助線の上の距離は、補助線と浅い角度で交わる
        ラインほど引き伸ばされるため）。
        """
        ok = np.isfinite(t)
        t, wt, mids = t[ok], wt[ok], mids[ok]
        src = np.flatnonzero(ok)
        n, M = len(t), len(model)
        if n < 2:
            return []
        if n > 26:                                      # 長い線を優先して候補を絞る
            keep = np.sort(np.argsort(-wt)[:26])
            t, wt, src, mids = t[keep], wt[keep], src[keep], mids[keep]
            n = 26
        o = np.argsort(t)
        t, wt, src, mids = t[o], wt[o], src[o], mids[o]
        di = np.array(list(itertools.combinations(range(n), 2)))
        mi = np.array(list(itertools.combinations(range(M), 2)))
        I = np.repeat(di, len(mi) * 2, axis=0)
        Mi = np.tile(np.r_[mi, mi[:, ::-1]], (len(di), 1))
        t1, t2 = t[I[:, 0]], t[I[:, 1]]
        m1, m2 = model[Mi[:, 0]], model[Mi[:, 1]]
        affine = not np.isfinite(t_h) or abs(t_h) > 1e7
        with np.errstate(divide="ignore", invalid="ignore"):
            if affine:
                p = (t2 - t1) / (m2 - m1)
                qv = t1 - p * m1
                pred = p[:, None] * model[None] + qv[:, None]
                good = np.isfinite(p)
                coef = np.c_[np.zeros(len(p)), p, qv]
            else:
                # t = q + r m (t_h - t) を 2 本で解く
                a1, a2 = m1 * (t_h - t1), m2 * (t_h - t2)
                det = a2 - a1
                r = (t2 - t1) / det
                qv = t1 - r * a1
                den = r[:, None] * model[None] + 1
                pred = (r[:, None] * t_h * model[None] + qv[:, None]) / den
                # 分母が 0 になる所（補助線と平行に写るライン）はコートの途中にあってもよい
                # （カメラがコートの長さの範囲内に立っている場合など）
                good = np.isfinite(r) & (np.abs(det) > 1e-9)
                coef = np.c_[r, r * t_h, qv]
        pred = np.where(np.isfinite(pred), pred, 1e12)
        span = np.abs(pred[np.arange(len(pred)), Mi[:, 1]] - pred[np.arange(len(pred)), Mi[:, 0]])
        good &= span > 3 * tol
        if not good.any():
            return []
        pred, coef = pred[good], coef[good]
        # 予想したライン (H, M, 3) と検出したラインの中点 (n,) との距離 (H, M, n)
        c0, d, _ = T
        P = c0[None, None] + np.clip(pred, -1e9, 1e9)[..., None] * d
        lp = np.cross(np.broadcast_to(vp, P.shape[:-1] + (3,)), np.concatenate([P, np.ones(P.shape[:-1] + (1,))], -1))
        lp /= np.maximum(np.linalg.norm(lp[..., :2], axis=-1, keepdims=True), 1e-12)
        diff = np.abs(lp @ np.c_[mids, np.ones(n)].T)
        j = np.argmin(diff, axis=2)
        dmin = np.take_along_axis(diff, j[:, :, None], 2)[..., 0]
        # 隣のモデルのラインまでの距離（同じ検出ラインの所で測る）の 0.4 倍までしか許さない
        other = np.take_along_axis(diff.transpose(0, 2, 1), j[:, :, None].repeat(M, 2), 1)   # (H, M, M): [h, k, k']
        other[:, np.arange(M), np.arange(M)] = np.inf
        gap = other.min(2)
        inl = dmin < np.minimum(tol, 0.4 * gap)
        wn = np.minimum(1.0, wt / max(wt.max(), 1e-9) * 2)
        score = (inl * (0.4 + 0.6 * wn[j])).sum(1)
        assign = np.where(inl, src[j], -1)
        out: dict[tuple, dict] = {}
        for k in np.argsort(-score):
            if score[k] < 1.5 or len(out) >= self.max_hypotheses:
                break
            key = tuple(assign[k])
            if key in out:
                continue
            out[key] = {"score": float(score[k]), "assign": assign[k], "pred": pred[k], "coef": coef[k]}
        return list(out.values())

    def _match_court(self, segs: np.ndarray, seg_w: np.ndarray, vps: list[np.ndarray],
                     shape: tuple[int, int]) -> list[dict]:
        """消失点の組ごとにコートの候補を作り、画像の線分との重なりで並べる。"""
        h, w = shape
        diag = math.hypot(w, h)
        p, q = segs[:, :2], segs[:, 2:]
        L = np.linalg.norm(q - p, axis=1)
        dirs = (q - p) / L[:, None]
        mids = (p + q) / 2
        all_c: list[dict] = []
        for va, vb in itertools.combinations(vps, 2):
            horizon = np.cross(va, vb)
            nh = np.linalg.norm(horizon[:2])
            if nh < 1e-12:
                continue
            horizon = horizon / nh
            # 地平線は横向き（カメラの傾きは ±max_roll_deg まで）で、床はその下にある。壁の縦線と横線の組
            # （地平線が縦になる）や、画像の下を通る地平線は床ではない
            if abs(horizon[0]) > math.sin(math.radians(self.max_roll_deg)):
                continue
            bottom = np.array([w / 2, h - 1, 1.0])
            if horizon[1] < 0:
                horizon = -horizon
            sgn = 1.0                                   # 地平線より下（v が大きい側）が正
            if horizon @ bottom < 0.05 * h:
                continue
            dist_e = np.minimum(np.c_[p, np.ones(len(p))] @ horizon, np.c_[q, np.ones(len(q))] @ horizon) * sgn
            below = dist_e > 0.01 * h
            r = self._vp_residual(np.array([va, vb]), mids, dirs)
            tol = math.radians(self.angle_tol_deg)
            grp = np.where(r[0] < r[1], 0, 1)
            keep = below & (r.min(0) < tol)
            if keep.sum() < 4:
                continue
            # 群の中で、線分の長さに応じて許す角度を絞る（長い線分ほど消失点に正確に向かう）
            rmin = r.min(0)
            keep &= rmin < np.maximum(math.radians(1.0), np.minimum(tol, 8.0 / np.maximum(L, 1)))
            # 床の上の基準点（補助線を通す点）：地平線と画像の下端の中間
            hp = bottom[:2] - horizon[:2] * (horizon @ bottom) * 0.5
            masks, merged = [], []
            for g in (0, 1):
                m = keep & (grp == g)
                masks.append(m)
                merged.append(self._merge_collinear(segs[m], seg_w[m], tol_px=0.0035 * diag) if m.sum() else [])
            if min(len(merged[0]), len(merged[1])) < 2:
                continue
            emaps = []
            for m in masks:
                img = np.full((h, w), 255, np.uint8)
                for s in segs[m & (seg_w >= 0.35)]:
                    cv2.line(img, tuple(np.round(s[:2]).astype(int)), tuple(np.round(s[2:]).astype(int)), 0, 1)
                emaps.append(cv2.distanceTransform(img, cv2.DIST_L2, 3))
            for g_long in (0, 1):                      # どちらの群を長手方向のライン（Y = 一定）とみなすか
                g_short = 1 - g_long
                vp_long, vp_short = (va, vb) if g_long == 0 else (vb, va)
                Tl = self._transversal(vp_long, horizon, shape, hp)
                Ts = self._transversal(vp_short, horizon, shape, hp)
                ll = np.array([m["line"] for m in merged[g_long]])
                ls = np.array([m["line"] for m in merged[g_short]])
                tl, ts = self._coord(ll, Tl), self._coord(ls, Ts)
                th_l = float(self._coord(horizon[None], Tl)[0])
                th_s = float(self._coord(horizon[None], Ts)[0])
                wl = np.array([m["length"] for m in merged[g_long]])
                ws = np.array([m["length"] for m in merged[g_short]])
                ml = np.array([self._line_mid(m) for m in merged[g_long]])
                ms = np.array([self._line_mid(m) for m in merged[g_short]])
                hy = self._hyp_1d(tl, wl, ml, Y_CENTERS, th_l, 0.004 * diag, vp_long, Tl)
                hx = self._hyp_1d(ts, ws, ms, X_CENTERS, th_s, 0.004 * diag, vp_short, Ts)
                if not hy or not hx:
                    continue
                cands = self._score_2d(hx, hy, Ts, Tl, vp_short, vp_long, th_s, th_l, horizon, sgn,
                                       emaps[g_short], emaps[g_long], shape)
                all_c += cands
        all_c.sort(key=lambda c: -c["score"])
        uniq: list[dict] = []
        for c in all_c:
            if not any(self._same_court(c["H"], u["H"], shape) for u in uniq):
                uniq.append(c)
            if len(uniq) >= 20:
                break
        return uniq

    def _score_2d(self, hx: list[dict], hy: list[dict], Ts, Tl, vp_short, vp_long, th_s, th_l,
                  horizon, sgn, emap_short, emap_long, shape) -> list[dict]:
        """短手・長手の候補の全組について、コートを描いて線分との重なりを点数にする。"""
        h, w = shape
        S = self._samples
        XY, kind = S["XY"], S["kind"]

        def line_of(t, c0, d, vp):                                            # 位置 t の直線（消失点を通る）
            P = c0[None] + t[..., None] * d
            P3 = np.concatenate([P, np.ones(P.shape[:-1] + (1,))], -1)
            return np.cross(np.broadcast_to(vp, P3.shape), P3)

        def map_vals(hyps, model, vals, th):
            """候補の 1 次元の写像 t = (r t_h m + q) / (r m + 1)（または p m + q）で実寸の値を写す。"""
            c = np.array([hh["coef"] for hh in hyps])                     # (K, 3) = r, p, q
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                return (c[:, 1:2] * vals[None] + c[:, 2:3]) / (c[:, 0:1] * vals[None] + 1)

        tX = map_vals(hx, X_CENTERS, XY[:, 0], th_s)       # (Kx, N)
        tY = map_vals(hy, Y_CENTERS, XY[:, 1], th_l)       # (Ky, N)
        lX = line_of(tX, Ts[0], Ts[1], vp_short)            # (Kx, N, 3)
        lY = line_of(tY, Tl[0], Tl[1], vp_long)             # (Ky, N, 3)
        # 1 次元での点数の合計が高い組から順に、決まった数だけ 2 次元で調べる
        sx = np.array([hh["score"] for hh in hx])
        sy = np.array([hh["score"] for hh in hy])
        tot = sx[:, None] + sy[None, :]
        order = np.argsort(-tot, axis=None)[: self.max_pairs]
        I, J = np.unravel_index(order, tot.shape)
        score = np.full(len(I), -np.inf)
        for c0 in range(0, len(I), 300):
            ii, jj = I[c0:c0 + 300], J[c0:c0 + 300]
            x = np.cross(lX[ii], lY[jj])                    # (C, N, 3)
            with np.errstate(divide="ignore", invalid="ignore"):
                uv = x[..., :2] / x[..., 2:3]
            score[c0:c0 + 300] = self._support(uv, emap_long, emap_short, horizon, sgn, shape)
        # 鏡像（カメラが床の下）を除く：コート中央でのヤコビアンの符号
        cands = []
        for k in np.argsort(-score)[:60]:
            i, j = I[k], J[k]
            if score[k] <= 0:
                break
            grid = np.array([[X, Y] for X in (0.0, 4.0, 9.4, 13.4) for Y in (0.0, 3.0, 6.1)])
            gx = map_vals([hx[i]], X_CENTERS, grid[:, 0], th_s)[0]
            gy = map_vals([hy[j]], Y_CENTERS, grid[:, 1], th_l)[0]
            pg = _intersect(line_of(gx, Ts[0], Ts[1], vp_short), line_of(gy, Tl[0], Tl[1], vp_long))
            if not np.isfinite(pg).all():
                continue
            Hm, _ = cv2.findHomography(grid, pg, 0)
            if Hm is None:
                continue
            Jac = Hm[:2, :2] * (Hm[2] @ [6.7, 3.05, 1]) - np.outer(Hm[:2] @ [6.7, 3.05, 1], Hm[2, :2])
            if np.linalg.det(Jac) >= 0:
                continue
            f = _focal_from_h(Hm, np.array([w / 2, h / 2]))
            sc = float(score[k]) * (1.0 if f is not None and 0.2 * max(w, h) < f < 8 * max(w, h) else 0.6)
            cands.append({"score": sc, "H": Hm, "f": f})
        return cands

    def _support(self, uv: np.ndarray, emap_long: np.ndarray, emap_short: np.ndarray, horizon: np.ndarray,
                 sgn: float, shape: tuple[int, int]) -> np.ndarray:
        """コートの全ラインのサンプル点の画像座標 (C, N, 2) が、検出した線分にどれだけ重なるか (C,)。

        重なった長さ − 0.5 × 重ならなかった長さ（画面の中の床の部分だけ数える）[px]。
        """
        h, w = shape
        S = self._samples
        u, v = uv[..., 0], uv[..., 1]
        fin = np.isfinite(u) & np.isfinite(v)
        inside = fin & (u >= 2) & (v >= 2) & (u <= w - 3) & (v <= h - 3)
        with np.errstate(invalid="ignore"):
            hdist = (u * horizon[0] + v * horizon[1] + horizon[2]) * sgn
        inside &= hdist > 3
        ui = np.clip(np.nan_to_num(u, nan=0, posinf=0, neginf=0).round().astype(int), 0, w - 1)
        vi = np.clip(np.nan_to_num(v, nan=0, posinf=0, neginf=0).round().astype(int), 0, h - 1)
        dist = np.where(S["kind"][None] == 0, emap_long[vi, ui], emap_short[vi, ui])
        # 画像上でのサンプル間隔を重みにする（遠くのラインに点が詰まりすぎないように）
        uvf = np.where(inside[..., None], uv, 0.0)
        step = np.linalg.norm(np.diff(uvf, axis=1), axis=-1)
        step = np.concatenate([step, step[:, -1:]], axis=1)
        step = np.where(S["last"][None], np.roll(step, 1, axis=1), step)
        wgt = np.clip(step, 0, 6) * inside
        sup = dist < 2.0 + 0.002 * math.hypot(w, h)
        return (wgt * sup).sum(1) - 0.5 * (wgt * ~sup).sum(1)

    def _verify(self, H: np.ndarray, lab: np.ndarray, window: float = 0.003, ref: Optional[float] = None) -> dict:
        """候補のコートの各ラインの位置で、実際に「帯」があるか、その色がそろっているかを調べる。

        バドミントンのラインは 1 色で塗られているので、本物のコートなら帯の色がどのラインでも同じになる。
        床板の継ぎ目や他競技のラインを拾った候補は、帯がなかったり色がばらばらだったりする。

        Returns:
            score（確認できた長さ − 0.4 × 確認できなかった長さ [px]）、found（確認できた割合）、
            color（ラインと床の色の差の向き。Lab に ``LAB_W`` の重みをつけた単位ベクトル）
        """
        h, w = lab.shape[:2]
        diag = math.hypot(w, h)
        S = self._samples
        XY = S["XY"]
        uv = _apply_h(H, XY)
        kind = S["kind"]
        nrm_w = np.where(kind[:, None] == 0, [[0.0, 1.0]], [[1.0, 0.0]])        # ラインに垂直な実寸の向き
        e1 = _apply_h(H, XY + nrm_w * LINE_WIDTH_M / 2)
        e0 = _apply_h(H, XY - nrm_w * LINE_WIDTH_M / 2)
        ok = np.isfinite(uv).all(1) & np.isfinite(e1).all(1) & np.isfinite(e0).all(1)
        R = 3.0 + window * diag
        ok &= (uv[:, 0] > R + 6) & (uv[:, 1] > R + 6) & (uv[:, 0] < w - R - 7) & (uv[:, 1] < h - R - 7)
        if ok.sum() < 10:
            return {"score": -1e9, "found": 0.0, "color": None}
        seg = S["seg"]
        tang = np.zeros_like(uv)
        for k in np.unique(seg):
            m = seg == k
            if m.sum() >= 2:
                tang[m] = np.gradient(np.nan_to_num(uv[m]), axis=0)
        tn = np.linalg.norm(tang, axis=1)
        ok &= tn > 1e-6
        nrm = np.c_[-tang[:, 1], tang[:, 0]] / np.maximum(tn, 1e-9)[:, None]
        width = np.abs(((e1 - e0) * nrm).sum(1))
        step = np.r_[np.linalg.norm(np.diff(np.nan_to_num(uv), axis=0), axis=1), 0.0]
        step = np.where(S["last"], np.r_[0.0, step[:-1]], step)
        wgt = np.clip(step, 0, 6)
        idx = np.flatnonzero(ok)
        hw = np.clip(width[idx] / 2, 0.75, 6.0)
        off = np.arange(-R - 6, R + 6.25, 0.5)
        mx = (uv[idx, 0:1] + off[None] * nrm[idx, 0:1]).astype(np.float32)
        my = (uv[idx, 1:2] + off[None] * nrm[idx, 1:2]).astype(np.float32)
        prof = cv2.remap(lab.astype(np.float32), mx, my, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE) * self.LAB_W
        K = len(off)
        cs = np.concatenate([np.zeros((len(idx), 1, 3), np.float32), np.cumsum(prof, axis=1)], axis=1)
        hk = np.round(hw / 0.5).astype(int)[:, None]
        ki = np.arange(K)[None]

        def box(a, b):
            a, b = np.clip(a, 0, K), np.clip(b, 0, K)
            return (np.take_along_axis(cs, b[..., None], 1) - np.take_along_axis(cs, a[..., None], 1)) / \
                np.maximum(b - a, 1)[..., None]

        cen = box(ki - hk, ki + hk + 1)
        left = box(ki - 3 * hk - 2, ki - hk)
        right = box(ki + hk + 1, ki + 3 * hk + 3)
        diff = cen - 0.5 * (left + right)
        mag = np.linalg.norm(diff, axis=2) - 0.6 * np.linalg.norm(left - right, axis=2)
        mag = np.where(np.abs(off)[None] <= R, mag, -np.inf)
        j = np.argmax(mag, axis=1)
        best = mag[np.arange(len(idx)), j]
        sig = diff[np.arange(len(idx)), j]                                     # 帯と床の色の差 (Lab, 重みつき)
        found = best > 5.0
        n_vis = float(wgt[idx].sum())
        if found.sum() < 5:
            return {"score": -0.4 * n_vis, "found": 0.0, "color": None}
        # 帯のはっきりさ（画像の中のはっきりしたラインに対する割合）で重みをつける。床板の継ぎ目のような
        # 薄い帯はどこにでもあるので、それだけで説明できる候補の点数は低くなる
        ref = float(np.percentile(best[found], 90)) if ref is None else ref
        cred = np.clip(best / max(ref, 1e-6), 0.0, 1.0)
        dom = self._dominant_direction(sig[found], (wgt[idx] * cred)[found])
        same = found & (self._cos(sig, dom) > 0.85)
        good = float((wgt[idx] * same * cred).sum())
        n_same = float((wgt[idx] * same).sum())
        return {"score": good - 0.4 * (n_vis - good), "found": n_same / max(n_vis, 1e-9), "color": dom}

    @staticmethod
    def _cos(v: np.ndarray, d: np.ndarray) -> np.ndarray:
        return (v @ d) / np.maximum(np.linalg.norm(v, axis=-1) * np.linalg.norm(d), 1e-9)

    @classmethod
    def _dominant_direction(cls, v: np.ndarray, w: np.ndarray, cos_r: float = 0.9) -> np.ndarray:
        """色の差のベクトルの中で、いちばん多い向き（単位ベクトル）。"""
        u = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
        if len(u) > 400:
            pick = np.linspace(0, len(u) - 1, 400).astype(int)
        else:
            pick = np.arange(len(u))
        dens = ((u[pick] @ u.T) > cos_r) @ w
        d = u[pick[int(np.argmax(dens))]]
        for _ in range(3):
            near = (u @ d) > cos_r
            m = (u[near] * w[near, None]).sum(0)
            d = m / max(np.linalg.norm(m), 1e-9)
        return d

    @staticmethod
    def _same_court(H1: np.ndarray, H2: np.ndarray, shape: tuple[int, int]) -> bool:
        c = np.array([[0, 0], [13.4, 0], [13.4, 6.1], [0, 6.1], [6.7, 3.05]])
        a, b = _apply_h(H1, c), _apply_h(H2, c)
        d = np.linalg.norm(a - b, axis=1)
        return bool(np.nanmax(d) < 0.02 * math.hypot(*shape))

    # ------------------------------------------------------------------ 原寸での測り直し
    def _refine_lines(self, lab: np.ndarray, H: np.ndarray, color: Optional[np.ndarray] = None,
                      narrow: bool = False) -> Optional[dict]:
        """各ラインに垂直な色（Lab）の断面からラインの中心を測り、直線を当てはめる（元の解像度）。

        ``color``（コートのラインと床の色の差の向き）があれば、それと違う帯（床板の継ぎ目・他競技のライン）は使わない。
        """
        h, w = lab.shape[:2]
        diag = math.hypot(w, h)
        lines, pts_all, n_vis_total, n_in_total = {}, {}, 0, 0
        resid = []
        cw = self.LAB_W
        for name, a, b in self._segments_by_line():
            n = max(8, int(np.linalg.norm(b - a) / 0.05))
            s = np.linspace(0.0, 1.0, n)[:, None]
            XY = a + s * (b - a)
            along = (b - a) / np.linalg.norm(b - a)
            nrm_w = np.array([-along[1], along[0]])
            uv = _apply_h(H, XY)
            e1 = _apply_h(H, XY + nrm_w * LINE_WIDTH_M / 2)
            e0 = _apply_h(H, XY - nrm_w * LINE_WIDTH_M / 2)
            ok = np.isfinite(uv).all(1) & np.isfinite(e1).all(1) & np.isfinite(e0).all(1)
            if ok.sum() < 6:
                continue
            uv, e1, e0 = uv[ok], e1[ok], e0[ok]
            tang = np.gradient(uv, axis=0)
            tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)
            nrm = np.c_[-tang[:, 1], tang[:, 0]]
            width = np.abs(((e1 - e0) * nrm).sum(1))                        # ライン幅 [px]
            R = (np.maximum(3.0, 1.5 * width + 2.0) if narrow else
                 np.maximum(6.0, 2.0 * width + 0.006 * diag))
            inside = (uv[:, 0] > R + 2) & (uv[:, 1] > R + 2) & (uv[:, 0] < w - R - 3) & (uv[:, 1] < h - R - 3)
            if inside.sum() < 6:
                continue
            uv, nrm, width, R = uv[inside], nrm[inside], width[inside], R[inside]
            n_vis = len(uv)
            Rm = float(R.max())
            pad = 3 * max(1.0, float(width.max())) + 1
            off = np.arange(-Rm - pad, Rm + pad + 0.25, 0.5)
            mapx = (uv[:, 0:1] + off[None] * nrm[:, 0:1]).astype(np.float32)
            mapy = (uv[:, 1:2] + off[None] * nrm[:, 1:2]).astype(np.float32)
            prof = cv2.remap(lab, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) * cw
            # 中央の帯（ライン幅）と両側の帯の色の差。線の太さは場所ごとに違うので帯幅も変える
            cs = np.concatenate([np.zeros((len(prof), 1, 3), np.float32), np.cumsum(prof, axis=1)], axis=1)
            K = prof.shape[1]
            idx = np.arange(K)[None]
            hw = np.clip(np.round(np.maximum(width, 1.0) / 0.5 / 2).astype(int), 1, None)[:, None]

            def box(c0, c1):
                c0 = np.clip(c0, 0, K)
                c1 = np.clip(c1, 0, K)
                return (np.take_along_axis(cs, c1[..., None], 1) - np.take_along_axis(cs, c0[..., None], 1)) / \
                    np.maximum(c1 - c0, 1)[..., None]

            cen = box(idx - hw, idx + hw + 1)
            left = box(idx - 3 * hw - 1, idx - hw)
            right = box(idx + hw + 1, idx + 3 * hw + 2)
            # 両側が床なら左右の色はほぼ同じ（段差のエッジを線と間違えない）
            resp = np.linalg.norm(cen - 0.5 * (left + right), axis=2) - 0.6 * np.linalg.norm(left - right, axis=2)
            if color is not None:
                # 帯と床の色の差の向きが、コートのラインと同じものだけ（細いラインは床の色と混ざっても向きは変わらない）
                dv = cen - 0.5 * (left + right)
                cosv = (dv @ np.asarray(color, np.float32)) / np.maximum(np.linalg.norm(dv, axis=2), 1e-6)
                resp = np.where(cosv > 0.8, resp, -np.inf)
            resp = np.where(np.abs(off)[None] <= R[:, None], resp, -np.inf)
            j = np.argmax(resp, axis=1)
            val = resp[np.arange(len(resp)), j]
            pos = val[np.isfinite(val) & (val > 0)]
            thr = max(4.0, 0.35 * float(np.median(pos))) if len(pos) else 4.0
            good = np.isfinite(val) & (val > thr) & (j > 0) & (j < K - 1)
            if good.sum() < 6:
                continue
            gi = np.flatnonzero(good)
            jj = j[gi]
            y0, y1, y2 = resp[gi, jj - 1], resp[gi, jj], resp[gi, jj + 1]
            y0 = np.where(np.isfinite(y0), y0, y1)
            y2 = np.where(np.isfinite(y2), y2, y1)
            den = y0 - 2 * y1 + y2
            with np.errstate(divide="ignore", invalid="ignore"):
                sub = np.where(den < 0, 0.5 * (y0 - y2) / np.where(den == 0, -1, den), 0.0)
            dpos = off[jj] + np.clip(sub, -1, 1) * 0.5
            pts = uv[gi] + dpos[:, None] * nrm[gi]
            fit = self._ransac_line(pts, max(1.0, 0.15 * float(np.median(width[gi])) + 0.8))
            if fit is None:
                continue
            line, inl = fit
            if inl.sum() < max(6, 0.15 * n_vis):
                continue
            key = name
            if key in pts_all:                                # センターラインは 2 本の区間をまとめる
                allp = np.r_[pts_all[key], pts[inl]]
                lines[key] = _fit_line(allp)
                pts_all[key] = allp
            else:
                lines[key] = line
                pts_all[key] = pts[inl]
            resid.append(np.abs(np.c_[pts[inl], np.ones(inl.sum())] @ line))
            n_vis_total += n_vis
            n_in_total += int(inl.sum())
        nx = sum(1 for nm, _, _ in X_LINES if nm in lines)
        ny = sum(1 for nm, _, _ in Y_LINES if nm in lines)
        if nx < 2 or ny < 2:
            return None
        Hn = self._fit_h_to_lines(H, lines, pts_all)
        r = np.concatenate(resid)
        return {"lines": lines, "points": pts_all, "H": Hn,
                "score": n_in_total / max(n_vis_total, 1) * min(1.0, (nx + ny) / 7) * (nx + ny),
                "coverage": n_in_total / max(n_vis_total, 1), "line_rms": float(np.sqrt(np.mean(r ** 2)))}

    def _segments_by_line(self) -> list[tuple[str, np.ndarray, np.ndarray]]:
        return _court_segments_world()

    @staticmethod
    def _ransac_line(pts: np.ndarray, tol: float, iters: int = 200) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if len(pts) < 4:
            return None
        rng = np.random.default_rng(1)
        best, best_n = None, 0
        for _ in range(iters):
            i, j = rng.choice(len(pts), 2, replace=False)
            if np.linalg.norm(pts[i] - pts[j]) < 1e-6:
                continue
            l = _hline(pts[i], pts[j])
            n_in = int((np.abs(np.c_[pts, np.ones(len(pts))] @ l) < tol).sum())
            if n_in > best_n:
                best, best_n = l, n_in
        if best is None:
            return None
        for _ in range(2):
            inl = np.abs(np.c_[pts, np.ones(len(pts))] @ best) < tol
            if inl.sum() < 2:
                return None
            best = _fit_line(pts[inl])
        inl = np.abs(np.c_[pts, np.ones(len(pts))] @ best) < tol
        return best, inl

    @staticmethod
    def _fit_h_to_lines(H0: np.ndarray, lines: dict, pts: dict) -> np.ndarray:
        """測ったラインの中心点が、ホモグラフィで写したモデルのラインに乗るように H を詰める。"""
        from scipy.optimize import least_squares

        world = {n: ("x", X) for (n, _, _), X in zip(X_LINES, X_CENTERS)}
        world.update({n: ("y", Y) for (n, _, _), Y in zip(Y_LINES, Y_CENTERS)})
        items = [(pts[n], world[n]) for n in lines]
        scale = H0[2, 2] if abs(H0[2, 2]) > 1e-12 else 1.0
        h0 = (H0 / scale).ravel()[:8]

        def resid(hv: np.ndarray) -> np.ndarray:
            Hm = np.r_[hv, 1.0].reshape(3, 3)
            try:
                Hi = np.linalg.inv(Hm)
            except np.linalg.LinAlgError:
                return np.full(sum(len(p) for p, _ in items), 1e3)
            out = []
            for p, (ax, val) in items:
                lw = np.array([1.0, 0.0, -val]) if ax == "x" else np.array([0.0, 1.0, -val])
                li = Hi.T @ lw
                li = li / np.linalg.norm(li[:2])
                out.append(np.c_[p, np.ones(len(p))] @ li)
            return np.concatenate(out)

        sol = least_squares(resid, h0, loss="huber", f_scale=1.5, x_scale="jac", max_nfev=200)
        return np.r_[sol.x, 1.0].reshape(3, 3)

    # ------------------------------------------------------------------ d. PnP とホモグラフィ
    def _solve_from_lines(self, lines: dict, pts: dict, W: int, Hh: int,
                          camera_matrix: Optional[np.ndarray], dist_coeffs: Optional[np.ndarray],
                          H_lines: Optional[np.ndarray], fixed_K: Optional[np.ndarray] = None) -> CalibrationResult:
        """測ったライン同士の交点から、ホモグラフィ・姿勢（solvePnP）・再投影誤差・信頼度を求める。"""
        diag = math.hypot(W, Hh)
        c = np.array([W / 2, Hh / 2])
        names, world_c, world_e, img = [], [], [], []
        for ix, (xn, _, _) in enumerate(X_LINES):
            for iy, (yn, _, _) in enumerate(Y_LINES):
                if xn not in lines or yn not in lines:
                    continue
                nm = landmark_name(ix, iy)
                if nm is None:
                    continue
                p = _intersect(lines[xn], lines[yn])
                if not np.isfinite(p).all():
                    continue
                if not (-0.03 * W <= p[0] <= 1.03 * W and -0.03 * Hh <= p[1] <= 1.03 * Hh):
                    continue
                names.append(nm)
                world_c.append([X_CENTERS[ix], Y_CENTERS[iy]])
                world_e.append([X_EDGES[ix], Y_EDGES[iy]])
                img.append(p)
        res = CalibrationResult(False, image_size=(W, Hh), lines_2d=dict(lines), line_points=dict(pts),
                                camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        if len(names) < 4:
            if H_lines is not None:
                res.homography_matrix = H_lines
                res.corners_2d = _apply_h(H_lines, np.array([[X, Y] for _, X, Y, _ in CORNERS]))
            res.message = f"ラインの交点が {len(names)} 個しか取れません（4 個以上必要）"
            return res
        world_c, world_e, img = np.array(world_c), np.array(world_e), np.array(img)
        Hm, _ = cv2.findHomography(world_c, img, 0)
        if H_lines is not None:
            Hm = H_lines
        # ラインの中心どうしの交点 → ラインの外側の縁の角（交点の近くでの H の差分を足す）
        img_e = img + (_apply_h(Hm, world_e) - _apply_h(Hm, world_c))

        dist = np.zeros(4) if dist_coeffs is None else np.asarray(dist_coeffs, float).ravel()
        K = fixed_K if fixed_K is not None else camera_matrix
        obj = np.c_[world_c, np.zeros(len(world_c))].astype(np.float64)
        if K is None:
            f0 = _focal_from_h(Hm, c)
            if f0 is None or not (0.2 * max(W, Hh) < f0 < 8 * max(W, Hh)):
                f0 = (W / 2) / math.tan(math.radians(self.fov_deg / 2))
            sol = self._pnp_focal(obj, img, f0, c, dist)
        else:
            sol = self._pnp(obj, img, np.asarray(K, float), dist)
        if sol is None:
            res.homography_matrix = Hm
            res.message = "solvePnP が解けませんでした"
            return res
        K, rvec, tvec, rms = sol
        R = cv2.Rodrigues(rvec)[0]
        C = -R.T @ tvec.ravel()
        if C[2] <= 0:
            res.homography_matrix = Hm
            res.message = "カメラが床の下になる解しか得られません"
            return res
        rms_n = rms * 1920.0 / max(W, Hh)                 # 横 1920 px に換算した誤差で評価する
        conf_r = float(np.interp(rms_n, [0.0, 1.0, 3.0, 10.0, 20.0], [1.0, 0.97, 0.9, 0.3, 0.0]))
        cov = sum(len(v) for v in pts.values())
        nl = len(lines)
        conf_n = float(np.interp(len(names), [4, 6, 8], [0.5, 0.8, 1.0]))
        conf_l = float(np.interp(nl, [4, 6, 8], [0.6, 0.85, 1.0]))
        corners = _apply_h(Hm, np.array([[X, Y] for _, X, Y, _ in CORNERS]))
        res.success = True
        res.homography_matrix = Hm
        res.rvec, res.tvec = rvec.reshape(3, 1), tvec.reshape(3, 1)
        res.camera_matrix = K
        res.dist_coeffs = dist.reshape(-1, 1) if dist_coeffs is not None else None
        res.corners_2d = corners
        res.keypoints_2d = {n: (float(p[0]), float(p[1])) for n, p in zip(names, img_e)}
        res.reprojection_error_rms = float(rms)
        res.confidence_score = round(conf_r * conf_n * conf_l, 3)
        res.coverage = float(cov)
        res.message = f"{nl} 本のライン・{len(names)} 個の交点から推定（誤差 {rms:.2f} px）"
        return res

    @staticmethod
    def _pnp(obj: np.ndarray, img: np.ndarray, K: np.ndarray, dist: np.ndarray):
        try:
            ok, rvec, tvec = cv2.solvePnP(obj, img.astype(np.float64), K, dist, flags=cv2.SOLVEPNP_IPPE)
            if not ok:
                ok, rvec, tvec = cv2.solvePnP(obj, img.astype(np.float64), K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                return None
            rvec, tvec = cv2.solvePnPRefineLM(obj, img.astype(np.float64), K, dist, rvec, tvec)
        except cv2.error:
            return None
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
        return K, rvec, tvec, rms

    def _pnp_focal(self, obj: np.ndarray, img: np.ndarray, f0: float, c: np.ndarray, dist: np.ndarray):
        """焦点距離も未知のとき：f0 の周りを探して solvePnP し、(f, 姿勢) を最小二乗で詰める。"""
        from scipy.optimize import least_squares

        best = None
        for f in f0 * np.geomspace(0.6, 1.6, 15):
            K = np.array([[f, 0, c[0]], [0, f, c[1]], [0, 0, 1]], float)
            r = self._pnp(obj, img, K, dist)
            if r is not None and (best is None or r[3] < best[3]):
                best = r
        if best is None:
            return None
        K, rvec, tvec, _ = best

        def resid(p: np.ndarray) -> np.ndarray:
            Kp = np.array([[math.exp(p[0]), 0, c[0]], [0, math.exp(p[0]), c[1]], [0, 0, 1]])
            proj, _ = cv2.projectPoints(obj, p[1:4], p[4:7], Kp, dist)
            return (proj.reshape(-1, 2) - img).ravel()

        if len(obj) >= 5:
            sol = least_squares(resid, np.r_[math.log(K[0, 0]), rvec.ravel(), tvec.ravel()], x_scale="jac")
            f = math.exp(sol.x[0])
            if 0.2 * f0 < f < 5 * f0:
                K = np.array([[f, 0, c[0]], [0, f, c[1]], [0, 0, 1]])
                rvec, tvec = sol.x[1:4].reshape(3, 1), sol.x[4:7].reshape(3, 1)
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
        return K, rvec, tvec, rms

    def _orient_near(self, res: CalibrationResult) -> CalibrationResult:
        """カメラに近い端（横から撮った場合は画面の左の端）を X = 0（打つ人の側）にする。"""
        C = res.camera_center
        if C is None:
            return res
        if abs(C[0] - NET_X_M) > 2.0:
            flip = C[0] > NET_X_M
        else:
            ends = _apply_h(res.homography_matrix, np.array([[0.0, 3.05], [13.4, 3.05]]))
            flip = bool(np.isfinite(ends).all() and ends[0, 0] > ends[1, 0])
        return self.rotate_180(res) if flip else res

    def _partial_result(self, H: np.ndarray, W: int, Hh: int, K, dist) -> CalibrationResult:
        """測り直しに失敗したときの、手動で直すための初期位置（候補のホモグラフィで描いた点）。"""
        kp = {}
        for ix in range(len(X_LINES)):
            for iy in range(len(Y_LINES)):
                nm = landmark_name(ix, iy)
                p = _apply_h(H, np.array([[X_EDGES[ix], Y_EDGES[iy]]]))[0]
                if nm and np.isfinite(p).all() and 0 <= p[0] < W and 0 <= p[1] < Hh:
                    kp[nm] = (float(p[0]), float(p[1]))
        return CalibrationResult(False, homography_matrix=H, keypoints_2d=kp,
                                 corners_2d=_apply_h(H, np.array([[X, Y] for _, X, Y, _ in CORNERS])),
                                 image_size=(W, Hh), camera_matrix=K, dist_coeffs=dist)

    # ------------------------------------------------------------------ e. 可視化
    def visualize(self, frame: np.ndarray, result: CalibrationResult) -> np.ndarray:
        """コートのライン、四隅、推定したカメラで描いた 3 次元のワイヤーフレーム（ネットポスト込み）を重ねる。"""
        img = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        t = max(2, round(max(w, h) / 960))
        if result.homography_matrix is None:
            cv2.putText(img, "auto calibration failed", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2 * t / 2,
                        (0, 0, 255), t, cv2.LINE_AA)
            return img

        def project(P3: np.ndarray) -> np.ndarray:
            if result.rvec is not None and result.camera_matrix is not None:
                d = np.zeros(4) if result.dist_coeffs is None else result.dist_coeffs
                pr, _ = cv2.projectPoints(P3.astype(np.float64), result.rvec, result.tvec, result.camera_matrix, d)
                return pr.reshape(-1, 2)
            return _apply_h(result.homography_matrix, P3[:, :2])

        def poly(P3: np.ndarray, color, th) -> None:
            uv = project(P3)
            ok = np.isfinite(uv).all(1) & (np.abs(uv) < 1e5).all(1)
            if ok.sum() >= 2:
                cv2.polylines(img, [uv[ok].round().astype(np.int32)], False, color, th, cv2.LINE_AA)

        s = np.linspace(0, 1, 30)[:, None]
        for _, a, b in _court_segments_world():
            # 外側の縁ではなくラインの中心を描く（画像のラインの真ん中に重なる）
            P = np.c_[a + s * (b - a), np.zeros(len(s))]
            poly(P, (0, 255, 255), t)
        if result.rvec is not None:
            for Y in (0.0, COURT_WIDTH_M):                 # ネットポスト
                poly(np.array([[NET_X_M, Y, 0.0], [NET_X_M, Y, NET_POST_HEIGHT_M]]), (255, 0, 255), t + 1)
            yy = np.linspace(0, COURT_WIDTH_M, 40)
            top = NET_POST_HEIGHT_M - (NET_POST_HEIGHT_M - NET_CENTER_HEIGHT_M) * (1 - ((yy - 3.05) / 3.05) ** 2)
            poly(np.c_[np.full(40, NET_X_M), yy, top], (255, 0, 255), t)
            poly(np.c_[np.full(40, NET_X_M), yy, np.zeros(40)], (255, 0, 255), max(1, t - 1))
            for X, Y in ((0, 0), (13.4, 0), (13.4, 6.1), (0, 6.1)):   # 四隅の上 2 m までの柱（奥行きの確認用）
                poly(np.array([[X, Y, 0.0], [X, Y, 2.0]]), (255, 160, 0), max(1, t - 1))
        if result.keypoints_2d:
            for p in result.keypoints_2d.values():
                cv2.circle(img, (round(p[0]), round(p[1])), 3 * t, (0, 255, 0), -1, cv2.LINE_AA)
        if result.corners_2d is not None:
            for (nm, *_), p in zip(CORNERS, result.corners_2d):
                if np.isfinite(p).all() and -w < p[0] < 2 * w and -h < p[1] < 2 * h:
                    q = (round(p[0]), round(p[1]))
                    cv2.circle(img, q, 7 * t, (0, 0, 255), t, cv2.LINE_AA)
                    cv2.putText(img, nm, (q[0] + 8 * t, q[1] - 8 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * t,
                                (0, 0, 255), t, cv2.LINE_AA)
        txt = (f"auto: {'OK' if result.success else 'NG'}  RMS {result.reprojection_error_rms:.2f}px  "
               f"confidence {result.confidence_score:.2f}")
        cv2.putText(img, txt, (20, 20 + 18 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * t, (0, 0, 0), 3 * t, cv2.LINE_AA)
        cv2.putText(img, txt, (20, 20 + 18 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * t, (255, 255, 255), t, cv2.LINE_AA)
        return img

    def _debug_image(self, prep: dict, segs: np.ndarray, cand: Optional[dict],
                     res: Optional[CalibrationResult]) -> np.ndarray:
        img = (prep["bgr"] * 0.6).astype(np.uint8)
        for s in segs:
            cv2.line(img, tuple(np.round(s[:2]).astype(int)), tuple(np.round(s[2:]).astype(int)), (255, 200, 0), 1)
        if cand is not None:
            for _, a, b in _court_segments_world():
                uv = _apply_h(cand["H"], np.array([a, b]))
                if np.isfinite(uv).all():
                    cv2.line(img, tuple(np.round(uv[0]).astype(int)), tuple(np.round(uv[1]).astype(int)),
                             (0, 0, 255), 1, cv2.LINE_AA)
        if res is not None and res.line_points:
            s = prep["scale"]
            for p in res.line_points.values():
                for q in p[::3]:
                    cv2.circle(img, (round(q[0] * s), round(q[1] * s)), 1, (0, 255, 0), -1)
        return img
