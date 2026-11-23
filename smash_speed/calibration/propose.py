"""クリックなしのキャリブレーションの候補：画像からコートの候補をいくつか出し、人は選ぶ（確かめる）だけにする。

1. 全自動（``AutoCourtCalibrator``）と同じ方法で、ラインの組から BWF のコートの形に合う候補を集める
   （全自動はいちばんよい 1 つだけを使うが、ラインの取り違えで 2 番目以降が正しいことがある）
2. 候補ごとに、四隅の予想位置を「おおよそのクリック」として半自動と同じ方法（``CourtSnapper.refine``）で
   ラインに吸い付けて詰める
3. 詰めたあとのラインらしさ（＋測れたランドマークの数）で並べ、同じコートになったものをまとめて、上から ``n`` 個を返す。
   上位の候補は、ラインを 1 本取り違えた形（いちばん遠いラインを隣のラインに付け替えたもの）も吸い付けて比べる
   （ゆがみのある広角の動画では、正しいコートと 1 本取り違えたコートの点数がほぼ同じになる。ROADMAP v2 の J3）

手順 ① のカメラのおおよその位置が分かるときは、消失点から決めたカメラの向きと、その位置の近くで
線分がコートのラインに重なる高さ・位置からも候補を作る（``vanishing_candidates``。全自動のラインの組の
探し方は、実動画では他の競技のラインを選んで外れることが多い）。また、詰めた後のカメラの位置が
手順 ① の位置から離れる候補は除く。

どちらの端が打つ人の側（x < 0）かは画像だけでは決まらないので、カメラに近い端（画面の下の端）を
打つ人の側にする。カメラのおおよその位置（手順 ① の上から見た図）が x > 0 の側なら入れ替える。
座標は解析の座標系（ネット直下のコート中央が原点）[m]。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from smash_speed.calibration.auto_court_calibrator import (COURT_WIDTH_M, NET_X_M, AutoCourtCalibrator,
                                                           _apply_h)
from smash_speed.calibration.oneclick import (_COLORS, OneClickCalibrator, _court_score_colors, _homography,
                                              rotation_hypotheses)
from smash_speed.calibration.snap import HALF_L, HALF_W, LINES, CourtSnapper, SnapResult

# 解析の座標 → 全自動の座標（角が原点）
_TO_AUTO = np.array([[1.0, 0, NET_X_M], [0, 1.0, COURT_WIDTH_M / 2], [0, 0, 1.0]])
# コートを 180° 回す（打つ人の側を入れ替える）
_ROT180 = np.diag([-1.0, -1.0, 1.0])
_CORNERS = np.array([[-HALF_L, -HALF_W], [-HALF_L, HALF_W], [HALF_L, -HALF_W], [HALF_L, HALF_W]])
# 並べるときの、2 本のラインとも測れたランドマーク 1 点あたりの点数の上乗せ。1 本取り違えたコートは、ほかの競技の
# ラインや継ぎ目に乗る分でラインらしさの合計が正しいコートとほぼ同じになるが、測れるランドマークが少なくなる
POINT_BONUS = 1.0
# ラインの取り違えを試す上位の候補の数と、試す候補の点数（``rank_score``）がいちばんよい候補からこれ以内のときだけ試す
# （点数の差が大きいときは迷っていない）
N_SHIFT = 2
SHIFT_MARGIN = 5.0
# 取り違えを直した候補で、動かさなかったラインが画像の上で動いてよい量 [px]（横 1920 px に換算）
SHIFT_KEEP_PX = 6.0
# 消失点からの向きの候補のうち、世界の上向きが画像の上からこれより傾くもの（縦の消失点を床の向きに取った、壁を床と
# みなす向き）は使わない [°]。体育館の壁・客席の手すりの線分が床のラインより多い動画（練習の動画 IMG_5288）では、
# 壁を床とみなした向きの方が線分の点数が高くなり、正しい向きの候補が残らない（H1）
MAX_TILT_DEG = 60.0


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

    @property
    def rank_score(self) -> float:
        """並べる点数（ラインらしさ + 測れたランドマークの数 × ``POINT_BONUS``）。"""
        return self.score + POINT_BONUS * self.n_snapped


def _line_moved_px(a: SnapResult, b: SnapResult, line: tuple[str, float]) -> float:
    """ライン ``line`` = (向き, 位置) の、2 つの吸い付けの結果での画像の上の位置の違い（ラインに沿った点の中央値）[px]。"""
    ax, c = line
    s = np.linspace(-HALF_W, HALF_W, 21) if ax == "x" else np.linspace(-HALF_L, HALF_L, 41)
    XY = np.c_[np.full(len(s), c), s] if ax == "x" else np.c_[s, np.full(len(s), c)]
    d = np.linalg.norm(a.project(XY) - b.project(XY), axis=1)
    d = d[np.isfinite(d)]
    return float(np.median(d)) if len(d) else float("inf")


def line_shift_alternatives(res: SnapResult) -> list[tuple[np.ndarray, tuple[str, float]]]:
    """ラインを 1 本取り違えたかもしれない所を直した、はじめのホモグラフィ（吸い付ける前）と、動かさなかったライン
    (向き, 位置) の組のリスト。

    x・y の向きごとに、測れたラインのうち点のいちばん多いラインを動かさず、そこからいちばん遠い測れたラインを、
    隣のライン（コートの同じ向きのラインの並びで 1 つ内側・外側）に付け替える（床の座標をその向きに伸び縮みさせる）。
    """
    out = []
    for ax in ("x", "y"):
        names = [n for n in res.lines_found if LINES[n][0] == ax]
        if len(names) < 2:
            continue
        allc = sorted({float(LINES[n][1]) for n in LINES if LINES[n][0] == ax})
        c0 = float(LINES[max(names, key=lambda n: res.lines_found[n])][1])
        c1 = float(LINES[max(names, key=lambda n: abs(float(LINES[n][1]) - c0))][1])
        if c1 == c0:
            continue
        i = allc.index(c1)
        for j in (i - 1, i + 1):
            if not 0 <= j < len(allc) or allc[j] == c0:
                continue
            # 付け替えた床の座標（c1 にあったラインが allc[j]）→ 今の床の座標：old = a·new + b（c0 は動かさない）
            a = (c1 - c0) / (allc[j] - c0)
            T = np.eye(3)
            r = 0 if ax == "x" else 1
            T[r, r], T[r, 2] = a, c0 - a * c0
            out.append((res.homography @ T, (ax, c0)))
    return out


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


def _floor_segments(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """床の線分（元の解像度 [px] (N, 4)）、重み (N,)、消失点（元の解像度の画像座標 [px]）。"""
    cal = AutoCourtCalibrator(debug=False)
    prep = cal._preprocess(frame)
    segs, seg_w, _ = cal._detect_segments(prep["gray"], prep["bgr"], None)
    if len(segs) < 6:
        return np.zeros((0, 4)), np.zeros(0), []
    vps = []
    for v in cal._find_vanishing_points(segs, seg_w, prep["gray"].shape):
        v = np.asarray(v, float)
        if len(v) == 3:
            if abs(v[2]) < 1e-12:
                continue
            v = v[:2] / v[2]
        vps.append(v[:2] / prep["scale"])
    L = np.linalg.norm(segs[:, 2:] - segs[:, :2], axis=1) / prep["scale"]
    return segs / prep["scale"], seg_w * L, vps


# 床の 2 方向のラインの中心の位置 [m]（"x"：x = 一定のライン、"y"：y = 一定のライン）
_LINE_POS = {a: np.array(sorted(c for ax, c, _ in LINES.values() if ax == a)) for a in ("x", "y")}


def _axis_lines(segs: np.ndarray, wts: np.ndarray, hyp, axis: int, tol_deg: float = 2.0,
                max_segments: int = 1000) -> dict[str, np.ndarray]:
    """カメラの向き ``hyp`` で、床の ``axis`` 方向（0 = x、1 = y）に伸びる線分を選び、真上から見た座標にする。

    カメラから見た床の点は「カメラの真下からの水平の向き ÷ 下向きの成分」に高さを掛けた所にあるので、
    向きが分かれば、線分の位置はカメラの高さ z と位置 C を除いて決まる（床の上の位置 = C + z・u）。

    Returns:
        u (M,)：線分の位置（伸びる向きと直交する座標）、du (M,)：1 px ずれたときの u の変化、
        a (M, 2)：線分の伸びる範囲（伸びる向きの座標、小さい順）、w (M,)：重み。重みの大きい ``max_segments`` 本まで
    """
    Kinv = np.linalg.inv(hyp.K)
    v = hyp.K @ hyp.R[:, axis]                               # この向きの消失点（同次座標）
    p, q = segs[:, :2], segs[:, 2:]
    d = (q - p) / np.maximum(np.linalg.norm(q - p, axis=1, keepdims=True), 1e-9)
    mids = (p + q) / 2
    to_v = v[None, :2] - mids * v[2]
    cos = np.abs(np.sum(to_v * d, axis=1)) / np.maximum(np.linalg.norm(to_v, axis=1), 1e-12)
    sel = np.flatnonzero(np.arccos(np.clip(cos, 0, 1)) < np.radians(tol_deg))
    sel = sel[np.argsort(-wts[sel])[:max_segments]]
    m = len(sel)
    nrm = np.c_[-d[sel, 1], d[sel, 0]]
    pts = np.vstack([p[sel], q[sel], p[sel] + nrm, q[sel] + nrm])
    ray = (np.c_[pts, np.ones(len(pts))] @ Kinv.T) @ hyp.R    # 世界の向き（R の列 = 世界の軸のカメラでの向き）
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = ray[:, :2] / -ray[:, 2:3]
    uv = uv.reshape(4, m, 2)
    ok = (ray[:, 2] < -1e-9).reshape(4, m).all(axis=0) & np.isfinite(uv).all(axis=(0, 2))   # 床を見下ろしている
    u = uv[..., 1 - axis]
    du = np.maximum(np.abs(u[2] - u[0]), np.abs(u[3] - u[1]))
    a = np.sort(uv[:2, :, axis].T, axis=1)
    return {"u": 0.5 * (u[0] + u[1])[ok], "du": du[ok], "a": a[ok], "w": wts[sel][ok]}


def _axis_score(L: dict[str, np.ndarray], lines: np.ndarray, z: np.ndarray, C: np.ndarray,
                sigma_m: float = 0.04) -> np.ndarray:
    """カメラの高さ ``z`` (Z,)・位置 ``C`` (G,、等間隔) ごとに、線分の床の上の位置 C + z・u がコートのラインに合う度合い (Z, G)。

    ライン 1 本ごとに、近くの線分の重みの和の平方根を足す（1 本のラインに線分が集まるより、多くのラインに合う方がよい）。
    線分の重みを「C がいくつならラインに重なるか」（c − z・u）のヒストグラムにし、不確かさの大きさの組ごとにぼかして速く求める。
    """
    from scipy.ndimage import gaussian_filter1d

    u, du = L["u"], L["du"]
    out = np.zeros((len(z), len(C)))
    if not len(u):
        return out
    wn = L["w"] / L["w"].max()
    step = float(C[1] - C[0]) if len(C) > 1 else 0.03
    lo = C[0] - lines.max()                                   # ヒストグラムの範囲（C − c の範囲）
    n = int(np.ceil((C[-1] - lines.min() - lo) / step)) + 1
    classes = sigma_m * 2.0 ** np.arange(6)                   # ぼかしの大きさの組 [m]
    for i, zi in enumerate(z):
        s = np.sqrt(sigma_m ** 2 + (zi * du) ** 2)
        idx = np.round((-zi * u - lo) / step).astype(int)
        cls = np.clip(np.round(np.log2(s / sigma_m)).astype(int), 0, len(classes))
        ok = (idx >= 0) & (idx < n) & (cls < len(classes))   # 不確かさが大きすぎる線分は使わない
        D = np.zeros(n)
        for c in np.unique(cls[ok]):
            m = ok & (cls == c)
            hist = np.bincount(idx[m], weights=wn[m], minlength=n)
            sg = classes[c] / step
            D += gaussian_filter1d(hist, sg, mode="constant", truncate=3.0) * sg * np.sqrt(2 * np.pi)
        for c in lines:
            j = np.round((C - c - lo) / step).astype(int)
            out[i] += np.sqrt(np.maximum(D[np.clip(j, 0, n - 1)], 0.0))
    return out


def _peaks(s: np.ndarray, k: int, min_sep: int) -> np.ndarray:
    """1 次元の値 ``s`` の大きい山の位置を ``k`` 個まで（``min_sep`` 個より近い山はまとめる）。"""
    out: list[int] = []
    for j in np.argsort(-s):
        if all(abs(int(j) - o) >= min_sep for o in out):
            out.append(int(j))
            if len(out) >= k:
                break
    return np.array(out, int)


# ラインの伸びる範囲 [m]（"x" のラインは y の範囲、"y" のラインは x の範囲）
_LINE_SPANS = {a: [(c, spans) for ax, c, spans in LINES.values() if ax == a] for a in ("x", "y")}


def _joint_score(L: dict[str, np.ndarray], lines, z: float, Cn: np.ndarray, Ca: np.ndarray,
                 sigma_m: float = 0.04, pad_m: float = 0.3) -> np.ndarray:
    """組 (Cn, Ca)（ラインと直交する向き・伸びる向きのカメラの位置、(B,)）ごとに、線分がコートのラインの
    「位置が合い、しかもラインのある範囲に乗っている」度合い (B,)。コートの外の他の競技のライン・床板の継ぎ目を除く。"""
    if not len(L["u"]):
        return np.zeros(len(Cn))
    wn = L["w"] / L["w"].max()
    pos = Cn[:, None] + z * L["u"][None, :]                                    # (B, M)
    a0 = Ca[:, None] + z * L["a"][None, :, 0]
    a1 = Ca[:, None] + z * L["a"][None, :, 1]
    ln = np.maximum(a1 - a0, 1e-6)
    s2 = sigma_m ** 2 + (z * L["du"]) ** 2
    out = np.zeros(len(Cn))
    for c, spans in lines:
        g = np.exp(-0.5 * (pos - c) ** 2 / s2[None, :])
        inside = sum(np.clip(np.minimum(a1, hi + pad_m) - np.maximum(a0, lo - pad_m), 0, None) for lo, hi in spans) / ln
        out += np.sqrt((g * np.minimum(inside, 1.0)) @ wn)
    return out


def _tilt_deg(R: np.ndarray) -> float:
    """向き ``R``（列 = 世界の軸のカメラでの向き）で、世界の上向きが画像の上（カメラの −y）からどれだけ傾くか [°]。"""
    return float(np.degrees(np.arccos(np.clip(-R[1, 2], -1.0, 1.0))))


def vanishing_candidates(frame: np.ndarray, cam_xy: tuple[float, float], max_candidates: int = 6,
                         cam_sigma_m: float = 3.0, heights_m: np.ndarray | None = None,
                         n_peaks: int = 6, n_check: int = 40,
                         progress: Callable[[float, str], None] | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """消失点から決めたカメラの向きと、手順 ① のカメラの位置から、コートの候補を作る。

    1. 床の線分から消失点を求め、2 方向が直交する条件で焦点距離とカメラの向きの候補を作る（1 点クリックと同じ）
    2. 向きの候補ごとに、線分を真上から見た座標（カメラの高さ z と位置 C を除いた位置 u）にする。
       床の上の位置は C + z・u なので、x・y の向きごとに「線分がコートのラインに重なる」位置を、高さごとに 1 次元で探す
    3. 高さごとに x・y の山の組を作り、線分がラインのある範囲（コートの中）に乗っているかも見て点を付け直す
       （手順 ① の位置から離れるほど減らす）
    4. 点の高い ``n_check`` 個を、カメラの位置・向き・焦点距離を少し動かして画像のラインに合わせ（``_polish``）、
       線分の点数の上位と、合わせた後の点数の上位を交互に ``max_candidates`` 個返す

    Args:
        progress: 進み具合（0〜1, 説明）を受け取る関数

    Returns:
        (解析の座標 → 元の解像度の画像 [px] のホモグラフィ, 内部パラメータ K) のよい順の一覧
    """
    h, w = frame.shape[:2]
    segs, wts, vps = _floor_segments(frame)
    if len(vps) < 2:
        return []
    hyps = [hy for hy in rotation_hypotheses(vps, (h, w)) if _tilt_deg(hy.R) < MAX_TILT_DEG]
    z = np.arange(0.3, 4.01, 0.05) if heights_m is None else np.asarray(heights_m, float)
    span = 2.0 * cam_sigma_m
    step = 0.03
    cam = np.asarray(cam_xy, float)
    grid = [np.arange(cam[i] - span, cam[i] + span + 1e-9, step) for i in (0, 1)]
    cands = []
    for hi, hyp in enumerate(hyps):
        # Ls[0]：x 方向に伸びる線分（y = 一定のラインに合わせる）、Ls[1]：y 方向に伸びる線分
        Ls = [_axis_lines(segs, wts, hyp, axis) for axis in (0, 1)]
        s1d = []
        for axis in (0, 1):
            G = grid[1 - axis]
            s = _axis_score(Ls[axis], _LINE_POS["y" if axis == 0 else "x"], z, G)
            s1d.append(s - 0.5 * ((G[None, :] - cam[1 - axis]) / cam_sigma_m) ** 2)
        for i, zi in enumerate(z):
            jy = _peaks(s1d[0][i], n_peaks, round(0.2 / step))
            jx = _peaks(s1d[1][i], n_peaks, round(0.2 / step))
            Cy = np.repeat(grid[1][jy], len(jx))
            Cx = np.tile(grid[0][jx], len(jy))
            sc = _joint_score(Ls[0], _LINE_SPANS["y"], zi, Cy, Cx) + _joint_score(Ls[1], _LINE_SPANS["x"], zi, Cx, Cy)
            sc -= 0.5 * ((Cx - cam[0]) ** 2 + (Cy - cam[1]) ** 2) / cam_sigma_m ** 2
            for b in np.argsort(-sc)[:2]:
                cands.append((float(sc[b]), hi, np.array([Cx[b], Cy[b], zi])))
    cands.sort(key=lambda c: -c[0])
    picked: list[tuple[int, np.ndarray]] = []
    for s, hi, Cw in cands:
        if any(hi == qh and np.linalg.norm(Cw - qc) < 0.3 for qh, qc in picked):
            continue
        picked.append((hi, Cw))
        if len(picked) >= n_check:
            break
    # 4. 線分の点数は、隣のコート・継ぎ目の重なりでも上がるので、カメラの位置と向きを少し動かして詰めてから、
    #    描いたコートの全ラインに沿った画像のラインらしさ（1 点クリックと同じ。ラインがない所は 0）でも並べる
    snapper = CourtSnapper()
    k = max(h, w) / 1920.0
    lab = snapper.prepare(frame)
    # 大きくぼかした画像で大まかに合わせてから、ぼかしを小さくして詰める（ずれが大きいと細かいぼかしでは引き寄せられない）
    levels = [(cv2.GaussianBlur(lab, (0, 0), b * k) * snapper.LAB_W, b * k) for b in (6.0, 3.0, 1.5)]
    model = [(XY[::3], d) for XY, d in snapper._model_points()]
    polished = []
    for j, (hi, Cw) in enumerate(picked):
        if progress is not None:
            progress(j / len(picked), "カメラの位置の候補を画像に合わせています")
        polished.append(_polish(snapper, levels, model, hyps[hi], Cw))
    # 微調整は、レンズのゆがみがあると正しい候補を崩すこともあるので、線分の点数の上位（微調整しない）と
    # 微調整した後の点数の上位を交互に取る
    out: list[tuple[np.ndarray, np.ndarray]] = []
    by_q = np.argsort([-q for q, _ in polished])
    for i, j in zip(range(len(picked)), by_q):
        if len(out) < max_candidates:
            out.append((polished[j][1], hyps[picked[j][0]].K))
        if len(out) < max_candidates:
            out.append((_homography(hyps[picked[i][0]], picked[i][1]), hyps[picked[i][0]].K))
    return out


def _polish(snapper: CourtSnapper, levels: list[tuple[np.ndarray, float]], model, hyp, Cw: np.ndarray,
            max_evals: int = 250) -> tuple[float, np.ndarray]:
    """カメラの位置・向き・焦点距離を少しずつ動かして、描いたコートのラインらしさを最大にする（ラインに吸い付ける前の下ごしらえ）。

    線分から決めた位置は数十 px ずれていることがあり、そのままの点数では候補を比べられない。
    向きと焦点距離も動かすのは、レンズのゆがみで消失点からの向きが 2〜3°、焦点距離が 1〜2% ずれることがあるため。
    ``levels`` = [(ぼかした画像, ぼかしの大きさ [px]), ...] の順に合わせ、最後の画像での点数を返す。
    """
    from scipy.optimize import minimize

    def H_of(x: np.ndarray) -> np.ndarray:
        dR, _ = cv2.Rodrigues(np.radians(x[3:6]))
        K = hyp.K.copy()
        K[0, 0] = K[1, 1] = hyp.K[0, 0] * np.exp(x[6])
        return _homography(type(hyp)(K, dR @ hyp.R), Cw + x[:3])

    x = np.zeros(7)
    q = 0.0
    for img, blur_px in levels:
        def f(x: np.ndarray) -> float:
            if not (0.2 < Cw[2] + x[2] < 6.0 and abs(x[6]) < 0.2):
                return 0.0
            snapper.front = np.zeros(2)
            return -_court_score_colors(snapper, img, H_of(x), model, _COLORS, blur_px)

        steps = (0.15, 0.15, 0.05, 0.5, 0.5, 0.5, 0.02)
        simplex = np.vstack([x] + [x + np.eye(7)[i] * st for i, st in enumerate(steps)])
        r = minimize(f, x, method="Nelder-Mead", options={"initial_simplex": simplex, "maxfev": max_evals,
                                                          "xatol": 1e-3, "fatol": 1e-3})
        x, q = r.x, float(-r.fun)
    return q, H_of(x)


def camera_from_homography(Hm: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    """床のホモグラフィから、焦点距離（主点は画像の中心・正方画素）とカメラの位置（上から見た [m]）を求める。

    ``K⁻¹ H`` の 1・2 列目（床の x・y 方向の単位ベクトル）が直交する条件から焦点距離が決まる。
    決まらない（ありえないカメラ）ときは None。
    """
    h, w = shape
    A = np.array([[1.0, 0, -w / 2], [0, 1.0, -h / 2], [0, 0, 1.0]]) @ Hm
    a1, a2 = A[:, 0], A[:, 1]
    if abs(a1[2] * a2[2]) < 1e-15:
        return None
    f2 = -(a1[0] * a2[0] + a1[1] * a2[1]) / (a1[2] * a2[2])
    if not f2 > 0:
        return None
    f = float(np.sqrt(f2))
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    C = OneClickCalibrator._camera_xy(Hm, K)
    return None if C is None else (K, C)


def floor_depths(Hm: np.ndarray, K: np.ndarray, XY: np.ndarray) -> np.ndarray | None:
    """内部パラメータ ``K`` のカメラで床のホモグラフィ ``Hm`` に写る床の点 ``XY`` の、カメラからの奥行きの符号つきの大きさ。

    負ならカメラの後ろ（ホモグラフィでは画像の中に写ってしまう）。カメラが床より上にある向きに符号を決める。
    """
    B = np.linalg.inv(K) @ Hm
    n = float(np.linalg.norm(B[:, 0]))
    if not np.isfinite(B).all() or n < 1e-12:
        return None
    B = B / n
    R = np.c_[B[:, 0], B[:, 1], np.cross(B[:, 0], B[:, 1])]
    if float(-(R.T @ B[:, 2])[2]) < 0:                     # カメラの高さが負なら符号を逆にする
        B = -B
    return np.c_[XY, np.ones(len(XY))] @ B[2]


def _support_points(Hm: np.ndarray, K: np.ndarray | None, shape: tuple[int, int],
                    table: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """吸い付けの手がかりの点（実寸、画像）。ふつうはコートの四隅。

    四隅のどれかがカメラの後ろにある（ネットの近くの横・コートの中から撮った）ときは、四隅の代わりに、カメラの前にあって
    画像に写る床のランドマークを使う。``refine`` は 1 点目をカメラの前の点とみなすので、後ろの角が 1 点目だと、
    写っているラインを全部「後ろ」として測らなくなる（H1）。
    """
    P = _apply_h(Hm, _CORNERS)
    if K is None:
        cam = camera_from_homography(Hm, shape)
        if cam is None:
            return _CORNERS, P
        K = cam[0]
    dep = floor_depths(Hm, K, _CORNERS)
    if dep is None or (dep > 0).all():
        return _CORNERS, P
    h, w = shape
    L = np.array([v[:2] for v in table.values() if abs(float(v[2])) < 1e-6], float)
    uv = _apply_h(Hm, L)
    ok = (floor_depths(Hm, K, L) > 0) & np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < w) &         (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return (L[ok], uv[ok]) if ok.sum() >= 4 else (_CORNERS, P)


def _orient_by_camera(Hm: np.ndarray, shape: tuple[int, int], cam_xy: tuple[float, float]) -> np.ndarray:
    """コートを 180° 回した方が、カメラの位置が手順 ① の位置に近くなるなら回す。"""
    d = []
    for H in (Hm, Hm @ _ROT180):
        cam = camera_from_homography(H, shape)
        d.append(np.inf if cam is None else float(np.hypot(*(cam[1] - np.asarray(cam_xy)))))
    return Hm @ _ROT180 if d[1] < d[0] else Hm


def propose_calibrations(frame: np.ndarray, table: dict[str, np.ndarray], n: int = 3,
                         cam_xy: tuple[float, float] | None = None,
                         max_candidates: int = 6, max_cam_dist_m: float = 6.0,
                         progress: Callable[[float, str], None] | None = None) -> list[Proposal]:
    """クリックなしで、キャリブレーションの候補をよい順に最大 ``n`` 個返す（見つからなければ空）。

    Args:
        table: ランドマーク名 → 実寸座標 [m]
        cam_xy: カメラのおおよその位置（解析の座標系、上から見た [m]、手順 ① の上から見た図）。
            どちらの端を打つ人の側にするかと、消失点からの候補（``vanishing_candidates``）に使う
        max_candidates: ラインに吸い付けて詰める候補の数（全自動・消失点からの候補それぞれ）
        max_cam_dist_m: ``cam_xy`` があるとき、カメラの位置がこれより離れる候補は除く [m]
        progress: 進み具合（0〜1, 説明）を受け取る関数（1 コマ 1〜1.5 分かかるので、画面に出す）
    """
    last = [-1.0, ""]

    def report(frac: float, msg: str) -> None:
        # 同じ説明が続くときは 5% ごとにだけ知らせる（画面のログが流れすぎないように）
        frac = float(np.clip(frac, 0.0, 1.0))
        if progress is not None and (msg != last[1] or frac >= last[0] + 0.05 or frac == 1.0):
            last[:] = [frac, msg]
            progress(frac, msg)

    h, w = frame.shape[:2]
    k = max(h, w) / 1920.0
    snapper = CourtSnapper()
    lab = snapper.prepare(frame)
    img1 = cv2.GaussianBlur(lab, (0, 0), 1.0 * k) * snapper.LAB_W
    full = snapper._model_points()
    found: list[Proposal] = []
    report(0.0, "コートのラインの組を探しています")
    # (はじめのホモグラフィ, 内部パラメータ)。全自動の候補は内部パラメータが分からないので None
    starts: list[tuple[np.ndarray, np.ndarray | None]] = [
        (_orient(H0, (h, w), cam_xy), None) for H0 in candidate_homographies(frame, max_candidates)]
    if cam_xy is not None:
        starts = [(_orient_by_camera(H0, (h, w), cam_xy), K) for H0, K in starts]
    if cam_xy is not None:
        report(0.1, "線の向きからカメラの向きを求めています")
        starts += vanishing_candidates(frame, cam_xy, max_candidates,
                                       progress=lambda f, m: report(0.1 + 0.25 * f, m))
    def snap_one(H0: np.ndarray, K: np.ndarray | None) -> Proposal | None:
        W, P = _support_points(H0, K, (h, w), table)
        if not np.isfinite(P).all():
            return None
        try:
            res = snapper.refine(frame, H0, W, P, table, sigma=np.full(len(W), 4.0), lab=lab)
        except (ValueError, np.linalg.LinAlgError):
            return None
        if res.color is None:
            return None
        # 消失点からの候補は、詰めた後もカメラ K で撮った床として正しいものだけ（コートを細い帯につぶした解などを除く）
        if K is not None and OneClickCalibrator._metric_error(res.homography, K) > 0.3:
            return None
        # 手順 ① のカメラの位置が分かれば、そこから離れたカメラになる候補（他の競技のラインに合ったものなど）を除く
        if cam_xy is not None:
            cam = camera_from_homography(res.homography, (h, w))
            if cam is None or not (0.25 * w < cam[0][0, 0] < 3.0 * w) or \
                    np.hypot(*(cam[1] - np.asarray(cam_xy))) > max_cam_dist_m:
                return None
        # 点数は、その候補の κ（レンズのゆがみ）で描いたライン（``refine`` の後の ``snapper.kappa``）で測る
        score = snapper._court_score(img1, res.homography, full, res.color, k)
        return Proposal(res, float(score), sum(s == "snapped" for s in res.status.values()))

    def distinct(props: list[Proposal]) -> list[Proposal]:
        # 詰めたら同じコートになったものは、よい方だけ残す
        inv = np.linalg.inv(_TO_AUTO)
        out: list[Proposal] = []
        for p in sorted(props, key=lambda p: -p.rank_score):
            if not any(AutoCourtCalibrator._same_court(p.result.homography @ inv, q.result.homography @ inv, (h, w))
                       for q in out):
                out.append(p)
        return out

    # かかる時間の大半は吸い付け（1 つ約 5 秒）なので、残りの割合はその数で配る
    n_steps = len(starts) + 4 * N_SHIFT
    for i, (H0, K) in enumerate(starts):
        report(0.35 + 0.65 * i / n_steps, f"候補をコートのラインに合わせています（{i + 1} / {len(starts)}）")
        p = snap_one(H0, K)
        if p is not None:
            found.append(p)
    # 上位の候補は、ラインを 1 本取り違えた形も吸い付けて比べる
    top = distinct(found)[:N_SHIFT]
    alts = [(Ha, c0, p) for p in top if p.rank_score >= top[0].rank_score - SHIFT_MARGIN
            for Ha, c0 in line_shift_alternatives(p.result)]
    for i, (Ha, c0, orig) in enumerate(alts):
        report(0.35 + 0.65 * (len(starts) + i) / n_steps, f"ラインの取り違えを確かめています（{i + 1} / {len(alts)}）")
        p = snap_one(Ha, None)
        # 動かさなかったライン（点のいちばん多いライン）が、画像の上で元の候補と同じ所にあるものだけ（ほかの組の線へ
        # 滑った吸い付けを除く）
        # また、測れたランドマークが元の候補より少ないものは使わない（付け替えた形の方がよく合うときだけ）
        if p is not None and _line_moved_px(orig.result, p.result, c0) <= SHIFT_KEEP_PX * k and \
                p.n_snapped >= orig.n_snapped:
            found.append(p)
    report(1.0, f"候補を {len(found)} 個見つけました")
    return distinct(found)[:n]
