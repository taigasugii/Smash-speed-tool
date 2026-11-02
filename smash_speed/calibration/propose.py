"""クリックなしのキャリブレーションの候補：画像からコートの候補をいくつか出し、人は選ぶ（確かめる）だけにする。

1. 全自動（``AutoCourtCalibrator``）と同じ方法で、ラインの組から BWF のコートの形に合う候補を集める
   （全自動はいちばんよい 1 つだけを使うが、ラインの取り違えで 2 番目以降が正しいことがある）
2. 候補ごとに、四隅の予想位置を「おおよそのクリック」として半自動と同じ方法（``CourtSnapper.refine``）で
   ラインに吸い付けて詰める
3. 詰めたあとのラインらしさで並べ、同じコートになったものをまとめて、上から ``n`` 個を返す

手順 ① のカメラのおおよその位置が分かるときは、消失点から決めたカメラの向きと、その位置の近くで
線分がコートのラインに重なる高さ・位置からも候補を作る（``vanishing_candidates``。全自動のラインの組の
探し方は、実動画では他の競技のラインを選んで外れることが多い）。また、詰めた後のカメラの位置が
手順 ① の位置から離れる候補は除く。

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
from smash_speed.calibration.oneclick import (_COLORS, OneClickCalibrator, _court_score_colors, _homography,
                                              rotation_hypotheses)
from smash_speed.calibration.snap import HALF_L, HALF_W, LINES, CourtSnapper, SnapResult

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


def vanishing_candidates(frame: np.ndarray, cam_xy: tuple[float, float], max_candidates: int = 6,
                         cam_sigma_m: float = 3.0, heights_m: np.ndarray | None = None,
                         n_peaks: int = 6, n_check: int = 40) -> list[tuple[np.ndarray, np.ndarray]]:
    """消失点から決めたカメラの向きと、手順 ① のカメラの位置から、コートの候補を作る。

    1. 床の線分から消失点を求め、2 方向が直交する条件で焦点距離とカメラの向きの候補を作る（1 点クリックと同じ）
    2. 向きの候補ごとに、線分を真上から見た座標（カメラの高さ z と位置 C を除いた位置 u）にする。
       床の上の位置は C + z・u なので、x・y の向きごとに「線分がコートのラインに重なる」位置を、高さごとに 1 次元で探す
    3. 高さごとに x・y の山の組を作り、線分がラインのある範囲（コートの中）に乗っているかも見て点を付け直す
       （手順 ① の位置から離れるほど減らす）
    4. 点の高い ``n_check`` 個を、カメラの位置・向き・焦点距離を少し動かして画像のラインに合わせ（``_polish``）、
       線分の点数の上位と、合わせた後の点数の上位を交互に ``max_candidates`` 個返す

    Returns:
        (解析の座標 → 元の解像度の画像 [px] のホモグラフィ, 内部パラメータ K) のよい順の一覧
    """
    h, w = frame.shape[:2]
    segs, wts, vps = _floor_segments(frame)
    if len(vps) < 2:
        return []
    hyps = rotation_hypotheses(vps, (h, w))
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
    polished = [_polish(snapper, levels, model, hyps[hi], Cw) for hi, Cw in picked]
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


def _orient_by_camera(Hm: np.ndarray, shape: tuple[int, int], cam_xy: tuple[float, float]) -> np.ndarray:
    """コートを 180° 回した方が、カメラの位置が手順 ① の位置に近くなるなら回す。"""
    d = []
    for H in (Hm, Hm @ _ROT180):
        cam = camera_from_homography(H, shape)
        d.append(np.inf if cam is None else float(np.hypot(*(cam[1] - np.asarray(cam_xy)))))
    return Hm @ _ROT180 if d[1] < d[0] else Hm


def propose_calibrations(frame: np.ndarray, table: dict[str, np.ndarray], n: int = 3,
                         cam_xy: tuple[float, float] | None = None,
                         max_candidates: int = 6, max_cam_dist_m: float = 6.0) -> list[Proposal]:
    """クリックなしで、キャリブレーションの候補をよい順に最大 ``n`` 個返す（見つからなければ空）。

    Args:
        table: ランドマーク名 → 実寸座標 [m]
        cam_xy: カメラのおおよその位置（解析の座標系、上から見た [m]、手順 ① の上から見た図）。
            どちらの端を打つ人の側にするかと、消失点からの候補（``vanishing_candidates``）に使う
        max_candidates: ラインに吸い付けて詰める候補の数（全自動・消失点からの候補それぞれ）
        max_cam_dist_m: ``cam_xy`` があるとき、カメラの位置がこれより離れる候補は除く [m]
    """
    h, w = frame.shape[:2]
    k = max(h, w) / 1920.0
    snapper = CourtSnapper()
    lab = snapper.prepare(frame)
    img1 = cv2.GaussianBlur(lab, (0, 0), 1.0 * k) * snapper.LAB_W
    full = snapper._model_points()
    found: list[Proposal] = []
    # (はじめのホモグラフィ, 内部パラメータ)。全自動の候補は内部パラメータが分からないので None
    starts: list[tuple[np.ndarray, np.ndarray | None]] = [
        (_orient(H0, (h, w), cam_xy), None) for H0 in candidate_homographies(frame, max_candidates)]
    if cam_xy is not None:
        starts = [(_orient_by_camera(H0, (h, w), cam_xy), K) for H0, K in starts]
    if cam_xy is not None:
        starts += vanishing_candidates(frame, cam_xy, max_candidates)
    for H0, K in starts:
        P = _apply_h(H0, _CORNERS)
        if not np.isfinite(P).all():
            continue
        try:
            res = snapper.refine(frame, H0, _CORNERS, P, table, sigma=np.full(4, 4.0), lab=lab)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if res.color is None:
            continue
        # 消失点からの候補は、詰めた後もカメラ K で撮った床として正しいものだけ（コートを細い帯につぶした解などを除く）
        if K is not None and OneClickCalibrator._metric_error(res.homography, K) > 0.3:
            continue
        # 手順 ① のカメラの位置が分かれば、そこから離れたカメラになる候補（他の競技のラインに合ったものなど）を除く
        if cam_xy is not None:
            cam = camera_from_homography(res.homography, (h, w))
            if cam is None or not (0.25 * w < cam[0][0, 0] < 3.0 * w) or                     np.hypot(*(cam[1] - np.asarray(cam_xy))) > max_cam_dist_m:
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
