"""1 点クリックのキャリブレーション：コートの角を 1 点クリックするだけで、コートのラインに合わせる。

床のラインと床板の継ぎ目は、コートの長手・短手の 2 方向に平行に並ぶので、線分を集めると床の 2 方向の
消失点がよく求まる。消失点が 2 つあれば、主点を画像の中心と仮定して焦点距離とカメラの向きが決まる
（2 方向が直交する条件）。残る未知数はカメラの位置（3 つ）だが、クリックした角はカメラから見てその
画素の方向にあるので、カメラはその視線の上にあり、未知数はカメラの高さ 1 つになる。

1. 消失点の組（どちらが長手か・向き）ごとに、カメラの高さを 0.3〜4 m で動かし、描いたコートのラインの
   「ラインらしさ」が大きいものを候補にする。手順 ① で指定したカメラの位置（上から見た図）から
   大きく離れた候補は捨てる
2. 上位の候補を、半自動と同じ方法（``CourtSnapper.refine``）でラインに吸い付けて詰める
3. 詰めたあとのラインらしさ・クリックした角とのずれ・カメラの位置で、最もよい候補を選ぶ

体育館にコートが何面も並んでいても、クリックした角を通る解だけを探すので隣のコートには合わない。
座標は解析の座標系（ネット直下のコート中央が原点）[m]。
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import cv2
import numpy as np

from smash_speed.calibration.auto_court_calibrator import AutoCourtCalibrator, _apply_h
from smash_speed.calibration.snap import HALF_L, HALF_W, CourtSnapper, SnapResult


@dataclass
class _Hyp:
    K: np.ndarray
    R: np.ndarray              # 列 = 世界の x・y・z 軸のカメラ座標での向き


def floor_vanishing_points(frame: np.ndarray) -> list[np.ndarray]:
    """床の線分から求めた消失点（元の解像度の画像座標 [px]）。"""
    cal = AutoCourtCalibrator(debug=False)
    prep = cal._preprocess(frame)
    segs, seg_w, _ = cal._detect_segments(prep["gray"], prep["bgr"], None)
    if len(segs) < 6:
        return []
    out = []
    for v in cal._find_vanishing_points(segs, seg_w, prep["gray"].shape):
        v = np.asarray(v, float)
        if len(v) == 3:
            if abs(v[2]) < 1e-12:
                continue
            v = v[:2] / v[2]
        out.append(v[:2] / prep["scale"])
    return out


def rotation_hypotheses(vps: list[np.ndarray], shape: tuple[int, int]) -> list[_Hyp]:
    """消失点の組から、焦点距離とカメラの向きの候補を作る（主点は画像の中心、正方画素を仮定）。"""
    h, w = shape
    c = np.array([w / 2, h / 2])
    out = []
    for va, vb in itertools.permutations(vps, 2):
        a, b = va - c, vb - c
        f2 = -float(a @ b)                     # 2 方向が直交する条件：(a·b) + f² = 0
        if f2 <= 0:
            continue
        f = math.sqrt(f2)
        if not (0.25 * w < f < 3.0 * w):
            continue
        K = np.array([[f, 0, c[0]], [0, f, c[1]], [0, 0, 1]])
        for s1, s2 in itertools.product((1, -1), (1, -1)):
            r1 = s1 * np.r_[a / f, 1.0]
            r1 /= np.linalg.norm(r1)
            r2 = s2 * np.r_[b / f, 1.0]
            r2 = r2 - (r2 @ r1) * r1
            r2 /= np.linalg.norm(r2)
            r3 = np.cross(r1, r2)
            if r3[1] >= 0:                     # 世界の上向きが、画像の上（-v）を向くもの
                continue
            out.append(_Hyp(K, np.c_[r1, r2, r3]))
    return out


# コートのラインと床の色の差の向きの候補（重みをつけた Lab の単位ベクトル）。ラインは白・黄・緑・青など
# 体育館によって違い、床より暗いことも明るいこともあるので、明暗・赤緑・黄青とその組み合わせを全部試す
_COLORS = np.array([v for v in itertools.product((-1, 0, 1), repeat=3)
                    if any(v) and sum(map(abs, v)) in (1, 3)], np.float32)
_COLORS /= np.linalg.norm(_COLORS, axis=1, keepdims=True)


def _court_score_colors(snapper: CourtSnapper, img: np.ndarray, Hm: np.ndarray, model, colors: np.ndarray,
                        blur_px: float) -> float:
    """``CourtSnapper._court_score`` を色の候補ごとに求め、最大のものを返す。"""
    dv, sd, ids, nl = snapper.line_profiles(img, Hm, model, blur_px)
    if not len(ids):
        return 0.0
    resp = np.clip(dv @ colors.T - 0.5 * sd[:, None], 0.0, 20.0)
    return float(np.max(snapper.per_line_mean(resp, ids, nl)))


def _homography(hyp: _Hyp, C: np.ndarray) -> np.ndarray:
    t = -hyp.R @ C
    Hm = hyp.K @ np.c_[hyp.R[:, 0], hyp.R[:, 1], t]
    return Hm / Hm[2, 2]


class OneClickCalibrator:
    """コートの角を 1 点クリックするだけで、コートのラインに合わせる。

    Args:
        heights_m: 探すカメラの高さ [m]
        cam_sigma_m: 手順 ① で指定したカメラの位置の不確かさ [m]（上から見た図のクリックはおおまか）
        max_cam_dist_m: 指定したカメラの位置からこれ以上離れた候補は捨てる [m]
        n_refine: ラインに吸い付けて詰める候補の数
        n_restarts: 選んだ候補を、手がかりをずらして詰め直す回数（1 回目はずらさない）
    """

    def __init__(self, heights_m: np.ndarray | None = None, cam_sigma_m: float = 3.0, max_cam_dist_m: float = 8.0,
                 n_refine: int = 6, n_restarts: int = 3) -> None:
        self.heights_m = np.arange(0.3, 4.01, 0.02) if heights_m is None else np.asarray(heights_m, float)
        self.cam_sigma_m = cam_sigma_m
        self.max_cam_dist_m = max_cam_dist_m
        self.n_refine = n_refine
        self.n_restarts = n_restarts
        self.snapper = CourtSnapper(click_sigma_px=6.0)

    def calibrate(self, frame: np.ndarray, name: str, uv: tuple[float, float], table: dict[str, np.ndarray],
                  cam_xy: tuple[float, float] | None = None) -> SnapResult:
        """``name``（床のランドマーク）を画像の ``uv`` でクリックしたとして、全ランドマークを求める。

        Args:
            cam_xy: カメラのおおよその位置（解析の座標系、上から見た [m]）。分かれば候補を絞れる
        """
        if name not in table or abs(float(table[name][2])) > 1e-6:
            raise ValueError("床の点（コートの角など）を 1 つクリックしてください")
        h, w = frame.shape[:2]
        k = max(h, w) / 1920.0
        vps = floor_vanishing_points(frame)
        hyps = rotation_hypotheses(vps, (h, w))
        if not hyps:
            raise ValueError("床のラインの向きが分かりませんでした。4 点クリックで合わせてください")
        lab = self.snapper.prepare(frame)
        img3 = cv2.GaussianBlur(lab, (0, 0), 3.0 * k) * self.snapper.LAB_W
        model = [(XY[::3], d) for XY, d in self.snapper._model_points()]
        Pw = np.r_[np.asarray(table[name][:2], float), 0.0]
        self.snapper.front = Pw[:2]
        p = np.asarray(uv, float)
        cands = []
        for hi, hyp in enumerate(hyps):
            ray = hyp.R.T @ (np.linalg.inv(hyp.K) @ np.r_[p, 1.0])     # 世界の向き（カメラ → クリックした角）
            if ray[2] >= -1e-6:                                         # 床を見下ろしていない
                continue
            for z in self.heights_m:
                C = Pw + ray * (z / -ray[2])
                if abs(C[0]) < HALF_L - 0.5 and abs(C[1]) < HALF_W - 0.5:   # コートの中から撮ることはない
                    continue
                pen = 0.0
                if cam_xy is not None:
                    dc = float(np.hypot(C[0] - cam_xy[0], C[1] - cam_xy[1]))
                    if dc > self.max_cam_dist_m:
                        continue
                    pen = 0.5 * (dc / self.cam_sigma_m) ** 2
                Hm = _homography(hyp, C)
                s = _court_score_colors(self.snapper, img3, Hm, model, _COLORS, 3.0 * k)
                cands.append((s - pen, hi, float(z), Hm, C))
        if not cands:
            raise ValueError("クリックした角に合うカメラの位置が見つかりませんでした。4 点クリックで合わせてください")
        cands.sort(key=lambda c: -c[0])
        picked = []
        for c in cands:
            if any(c[1] == q[1] and abs(c[2] - q[2]) < 0.15 for q in picked):
                continue
            picked.append(c)
            if len(picked) >= self.n_refine:
                break
        # 詰める：クリックした角は正確（1 倍）、そのほかは候補の位置（4 倍の不確かさ）を手がかりにする
        quad = np.array([[-HALF_L, -HALF_W], [-HALF_L, HALF_W], [-1.98, -HALF_W], [-1.98, HALF_W]])
        Wp = np.vstack([Pw[:2], quad])
        sg = np.r_[1.0, 4.0, 4.0, 4.0, 4.0]
        img1 = cv2.GaussianBlur(lab, (0, 0), 1.0 * k) * self.snapper.LAB_W
        full = self.snapper._model_points()

        def quality(res: SnapResult, K: np.ndarray) -> float:
            """詰めた結果のよさ：ラインらしさ − クリックした角とのずれ − 指定したカメラの位置とのずれ。"""
            # ありえないカメラ（コートを細い帯につぶして、全ラインを 1 本の濃い線に重ねた解など）は除く：
            # 床の 2 方向の単位長さが、カメラから見て同じ長さで直交しているはず
            m = self._metric_error(res.homography, K)
            if m > 0.3:
                return -np.inf
            d_click = float(np.linalg.norm(_apply_h(res.homography, Pw[None, :2])[0] - p)) / k
            q = self.snapper._court_score(img1, res.homography, full, res.color, k) - 0.5 * (d_click / 6.0) ** 2
            q -= 0.5 * (m / 0.05) ** 2
            if cam_xy is not None:
                Cr = self._camera_xy(res.homography, K)
                if Cr is not None:
                    q -= 0.5 * (float(np.hypot(*(Cr - np.asarray(cam_xy)))) / self.cam_sigma_m) ** 2
            return q

        def run(Hm: np.ndarray, quad_img: np.ndarray, col: np.ndarray) -> SnapResult | None:
            Pp = np.vstack([p, quad_img])
            if not np.isfinite(Pp).all():
                return None
            res = self.snapper.refine(frame, Hm, Wp, Pp, table, sigma=sg, lab=lab, color=col)
            return res if res.color is not None else None

        best = None
        for s0, hi, z, Hm, C in picked:
            col = self.snapper.color_near(lab, Hm, Pw[:2])
            if col is None:
                continue
            res = run(Hm, _apply_h(Hm, quad), col)
            if res is not None:
                q = quality(res, hyps[hi].K)
                if best is None or q > best[0]:
                    best = (q, res, hyps[hi].K)
        if best is None:
            raise ValueError("コートのラインに合わせられませんでした。4 点クリックで合わせてください")
        # 選んだ候補の近くから、手がかりの点を少しずつずらして詰め直し、いちばんよいものを残す
        # （候補のずれが大きいと、途中の解（一部のラインだけ合った状態）で止まることがある）
        rng = np.random.default_rng(0)
        for i in range(self.n_restarts):
            q_img = _apply_h(best[1].homography, quad)
            if i:
                q_img = q_img + rng.normal(0, 12.0 * k, q_img.shape)
            res = run(best[1].homography, q_img, best[1].color)
            if res is None:
                continue
            q = quality(res, best[2])
            if q > best[0] + 1e-6:
                best = (q, res, best[2])
        res = best[1]
        res.message = "1 点から合わせました。" + res.message
        return res

    @staticmethod
    def _metric_error(Hm: np.ndarray, K: np.ndarray) -> float:
        """ホモグラフィがカメラ ``K`` で撮った床として正しいか（0 なら正しい）。

        ``K⁻¹ H`` の 1・2 列目は、床の x・y 方向の単位ベクトルをカメラ座標で表したもの（の定数倍）なので、
        長さが等しく直交する。長さの比の対数と、なす角の余弦の大きい方を返す。
        """
        A = np.linalg.inv(K) @ Hm
        a1, a2 = A[:, 0], A[:, 1]
        n1, n2 = float(np.linalg.norm(a1)), float(np.linalg.norm(a2))
        if n1 < 1e-12 or n2 < 1e-12:
            return np.inf
        return max(abs(math.log(n1 / n2)), abs(float(a1 @ a2)) / (n1 * n2))

    @staticmethod
    def _camera_xy(Hm: np.ndarray, K: np.ndarray) -> np.ndarray | None:
        """床のホモグラフィと内部パラメータから、カメラの位置（上から見た [m]）。"""
        A = np.linalg.inv(K) @ Hm
        for lam in (1.0 / np.linalg.norm(A[:, 0]), -1.0 / np.linalg.norm(A[:, 0])):
            r1, r2, t = lam * A[:, 0], lam * A[:, 1], lam * A[:, 2]
            R = np.c_[r1, r2, np.cross(r1, r2)]
            C = -R.T @ t
            if np.isfinite(C).all() and C[2] > 0:     # カメラは床より上
                return C[:2]
        return None
