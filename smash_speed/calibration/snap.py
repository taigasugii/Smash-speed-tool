"""半自動キャリブレーション：おおよその 4 点から、コートのラインに吸い付けて全ランドマークを求める。

クリックした 4 点（だいたいの位置でよい）から床のホモグラフィを作り、次の 2 段階で合わせる。

1. コート全体の位置合わせ：コートの形を保ったまま、描いた全ラインに沿った「ラインらしさ」
   （ラインの色の帯が両側の床と違うか）の合計が最大になるように動かす。クリックした点の近くの
   ラインから合わせ始め、範囲を広げながら合わせ直す
2. 1 本ずつの測定：各ラインの予想位置に垂直な色の断面からラインの中心を測り、測った点が
   モデルのラインに乗るようにホモグラフィを詰め直す（探す幅を狭めながら数回）

最後に、全ランドマーク（床の交点）の位置を返す。

床板の継ぎ目・他競技のライン・遠くで隣り合うライン（シングルスとダブルスのサイドラインは
0.46 m しか離れていない）に吸い付かないように、次のようにしている。

- 1 本ずつでなくコート全体で動かす（1 本ずつ探すと隣のラインに飛び移りやすい）
- 1 本ずつ測るときの探す幅は、画像上で隣のライン（平行なライン）までの距離の半分より狭くする
- ラインと床の色の差の向きを動画ごとに測り、それと同じ向きの帯だけを使う（ラインは白・黄・緑など
  体育館によって違い、床より暗いこともある）
- 各ラインは RANSAC で直線を当てはめ、選手で隠れた所などの外れ値を除く
- ホモグラフィで説明できないライン（別の線に吸い付いたもの）は外して当てはめ直す

座標は解析の座標系（ネット直下のコート中央が原点、x が長手、y が短手）[m]。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares

from smash_speed.calibration.auto_court_calibrator import (LINE_WIDTH_M, X_CENTERS, X_LINES, Y_CENTERS, Y_LINES,
                                                           AutoCourtCalibrator, _apply_h)

HALF_L, HALF_W = 6.70, 3.05
SHORT_SERVICE = 1.98

# ライン名 → ("x" なら x = 一定（短手方向に伸びる）、"y" なら y = 一定)、中心の位置 [m]、伸びる範囲 [m]
LINES: dict[str, tuple[str, float, tuple[tuple[float, float], ...]]] = {}
for (_n, _, _), _c in zip(X_LINES, X_CENTERS):
    LINES[_n] = ("x", float(_c) - HALF_L, ((-HALF_W, HALF_W),))
for (_n, _, _), _c in zip(Y_LINES, Y_CENTERS):
    LINES[_n] = ("y", float(_c) - HALF_W,
                 ((-HALF_L, -SHORT_SERVICE), (SHORT_SERVICE, HALF_L)) if _n == "center" else ((-HALF_L, HALF_L),))


def _world_line(ax: str, c: float) -> np.ndarray:
    return np.array([1.0, 0.0, -c]) if ax == "x" else np.array([0.0, 1.0, -c])


def _image_line(Hm: np.ndarray, ax: str, c: float) -> np.ndarray:
    li = np.linalg.inv(Hm).T @ _world_line(ax, c)
    return li / np.linalg.norm(li[:2])


@dataclass
class SnapResult:
    """吸着の結果。

    Attributes:
        homography: 床 (x, y, 1) [m] → 画像 [px]
        points: ランドマーク名 → 画像座標 [px]（画像の中にある床のランドマークすべて）
        status: ランドマーク名 → "snapped"（交わる 2 本のラインとも測れた）/ "predicted"（予想位置のまま）
        lines_found: 測れたライン名 → 使った点の数
        line_rms_px: 測った点とモデルのラインとのずれの RMS [px]
        message: 説明
    """

    homography: np.ndarray
    points: dict[str, tuple[float, float]] = field(default_factory=dict)
    status: dict[str, str] = field(default_factory=dict)
    lines_found: dict[str, int] = field(default_factory=dict)
    line_rms_px: float = float("nan")
    message: str = ""
    color: np.ndarray | None = None


class CourtSnapper:
    """おおよそのホモグラフィから、コートのラインに吸い付けて精密なホモグラフィを求める。

    Args:
        radii_px: くり返しごとの探す幅の上限 [px]（横 1920 px に換算した値）
        click_sigma_px: クリックした点の位置の不確かさ [px]（1920 px 換算。ラインが少ないときの支え）
    """

    LAB_W = AutoCourtCalibrator.LAB_W

    def __init__(self, radii_px: tuple[float, ...] = (6.0, 4.0, 3.0), click_sigma_px: float = 8.0,
                 search_px: float = 25.0) -> None:
        self.radii_px = radii_px
        self.click_sigma_px = click_sigma_px
        self.search_px = search_px
        self.front: np.ndarray | None = None     # カメラの前にあると分かっている床の点 [m]（クリックした点）

    # ------------------------------------------------------------------ 入口
    def snap(self, frame: np.ndarray, clicks: dict[str, tuple[float, float]],
             table: dict[str, np.ndarray]) -> SnapResult:
        """``clicks``（床のランドマーク名 → 画像座標、4 点以上）から全ランドマークを求める。

        Args:
            frame: BGR 画像（元の解像度）
            clicks: クリックした点。床にない点（支柱の先端など）は使わない
            table: ランドマーク名 → 実寸座標 [m]
        """
        floor = {n: np.asarray(p, float) for n, p in clicks.items()
                 if n in table and abs(float(table[n][2])) < 1e-6 and p is not None}
        if len(floor) < 4:
            raise ValueError(f"床の点が 4 点以上必要です（今 {len(floor)} 点）")
        names = list(floor)
        W = np.array([table[n][:2] for n in names], float)
        P = np.array([floor[n] for n in names], float)
        H0, _ = cv2.findHomography(W, P, 0)
        if H0 is None:
            raise ValueError("クリックした点からコートの形が決まりません（一直線に並んでいませんか）")
        return self.refine(frame, H0, W, P, table)

    def refine(self, frame: np.ndarray, H0: np.ndarray, W: np.ndarray, P: np.ndarray,
               table: dict[str, np.ndarray], sigma: np.ndarray | None = None,
               lab: np.ndarray | None = None, grow_from: np.ndarray | None = None,
               color: np.ndarray | None = None) -> SnapResult:
        """おおよそのホモグラフィ ``H0`` を、コートのラインに合わせて詰める。

        Args:
            W, P: 手がかりの点（実寸 [m] (N, 2) と画像 [px] (N, 2)）。ラインが少ないときの支えと、
                別の解に落ちないための罰に使う
            sigma: 各点の位置の不確かさの倍率（``click_sigma_px`` に掛ける）。省略すると全部 1
            lab: 前処理した画像（同じ画像で何度も呼ぶとき用）
            grow_from: 位置合わせを始める範囲を決める実寸の点 (M, 2)。省略すると ``W``（正確な点だけを渡す）
            color: ラインと床の色の差の向き。省略すると画像全体の各ラインから測る
        """
        h, w = frame.shape[:2]
        k = max(h, w) / 1920.0
        if lab is None:
            lab = self.prepare(frame)
        sg = np.ones(len(W)) if sigma is None else np.asarray(sigma, float)
        self.front = np.asarray(W[0], float)
        Hm = H0 / H0[2, 2]
        if color is None:
            color = self._line_color(self._measure(lab, Hm, 12.0 * k))
        if color is not None:
            # 1. コート全体の形を保ったまま、全ラインの応答の合計が最大になるように動かす（隣のラインに飛び移らない）
            Hm = self._align(lab, Hm, color, k, W, P, sg, W if grow_from is None else np.atleast_2d(grow_from))
        found: dict[str, np.ndarray] = {}
        for r in self.radii_px:
            # 2. 各ラインの近くで中心を測り、ホモグラフィを詰める
            meas = self._measure(lab, Hm, r * k, color)
            fits = self._fit_lines(meas, k)
            if not fits:
                break
            Hn, keep = self._fit_h(Hm, fits, W, P, k, sg)
            if Hn is None:
                break
            Hm, found = Hn, {n: fits[n] for n in keep}
        res = SnapResult(Hm)
        res.color = color
        res.lines_found = {n: len(p) for n, p in found.items()}
        if found:
            r_all = np.concatenate([np.c_[p, np.ones(len(p))] @ _image_line(Hm, *LINES[n][:2]) for n, p in found.items()])
            res.line_rms_px = float(np.sqrt(np.mean(r_all ** 2)))
        for n, X in table.items():
            if abs(float(X[2])) > 1e-6:
                continue
            uv = _apply_h(Hm, np.asarray(X[:2], float)[None])[0]
            if not self.in_front(Hm, np.asarray(X[:2], float)[None])[0]:
                continue
            if not (np.isfinite(uv).all() and -0.02 * w <= uv[0] <= 1.02 * w and -0.02 * h <= uv[1] <= 1.02 * h):
                continue
            res.points[n] = (round(float(uv[0]), 1), round(float(uv[1]), 1))
            res.status[n] = "snapped" if self._both_lines(n, X, found) else "predicted"
        ns = sum(1 for s in res.status.values() if s == "snapped")
        res.message = f"{len(found)} 本のラインに合わせました（{ns} 点を測定、ずれ {res.line_rms_px:.1f} px）"
        return res

    def in_front(self, Hm: np.ndarray, XY: np.ndarray) -> np.ndarray:
        """床の点 ``XY`` がカメラの前にあるか（後ろの点も、ホモグラフィでは画像の中に写ってしまう）。

        ``front``（カメラの前にあると分かっている点）と、同次座標の 3 番目の符号が同じなら前にある。
        """
        q = np.c_[XY, np.ones(len(XY))] @ Hm[2]
        if self.front is None:
            return np.ones(len(XY), bool)
        ref = float(np.r_[self.front, 1.0] @ Hm[2])
        return q * np.sign(ref) > 0

    @staticmethod
    def prepare(frame: np.ndarray) -> np.ndarray:
        """測定に使う画像（軽くぼかした Lab、float32）。"""
        k = max(frame.shape[:2]) / 1920.0
        return cv2.cvtColor(cv2.GaussianBlur(frame, (0, 0), 0.8 * k), cv2.COLOR_BGR2LAB).astype(np.float32)

    # ------------------------------------------------------------------ コート全体の位置合わせ
    def _model_points(self, box: tuple[float, float, float, float] | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
        """各ラインの中心線上の点 (N, 2) と、ラインに垂直な向き（床の上）[m]。交点のすぐ近くは除く。

        ``box`` = (x0, x1, y0, y1) [m] を与えると、その範囲の中の点だけにする。
        """
        out = []
        for name, (ax, c, spans) in LINES.items():
            s = np.concatenate([np.arange(a + 0.15, b - 0.15, 0.08) for a, b in spans])
            cross = np.array(sorted({v for ax2, v, _ in LINES.values() if ax2 != ax}))
            s = s[np.min(np.abs(s[:, None] - cross[None]), axis=1) > 0.15]
            XY = np.c_[np.full(len(s), c), s] if ax == "x" else np.c_[s, np.full(len(s), c)]
            if box is not None:
                XY = XY[(XY[:, 0] >= box[0]) & (XY[:, 0] <= box[1]) & (XY[:, 1] >= box[2]) & (XY[:, 1] <= box[3])]
            if len(XY):
                out.append((XY, np.array([1.0, 0.0]) if ax == "x" else np.array([0.0, 1.0])))
        return out

    @staticmethod
    def _pack(model) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """ライン別の点の一覧を、1 つの配列（点・ラインの幅の向き・ライン番号）にまとめる。"""
        XY = np.concatenate([m[0] for m in model])
        D = np.concatenate([np.repeat(m[1][None], len(m[0]), 0) for m in model]) * (LINE_WIDTH_M / 2)
        ids = np.concatenate([np.full(len(m[0]), i) for i, m in enumerate(model)])
        return XY, XY + D, ids, len(model)

    def line_profiles(self, img: np.ndarray, Hm: np.ndarray, model, blur_px: float):
        """描いたライン上の各点での「中央の帯 − 両側の床」の色の差と、両側どうしの差。

        Returns:
            (dv (N, 3), side_diff (N,), ライン番号 (N,), ラインの数)。画像の外の点は除く
        """
        key = id(model)
        cache = self.__dict__.setdefault("_pack_cache", {})
        if key not in cache or cache[key][0] is not model:
            cache[key] = (model, self._pack(model))
        XY, XYe, ids, nl = cache[key][1]
        h, w = img.shape[:2]
        both = np.vstack([XY, XYe])
        q = np.c_[both, np.ones(len(both))] @ Hm.T
        with np.errstate(divide="ignore", invalid="ignore"):
            q = q[:, :2] / q[:, 2:3]
        n0 = len(XY)
        uv, e1 = q[:n0], q[n0:]
        ok = np.isfinite(q[:n0]).all(1) & np.isfinite(e1).all(1) & (uv[:, 0] > 2) & (uv[:, 1] > 2) & \
            (uv[:, 0] < w - 3) & (uv[:, 1] < h - 3) & self.in_front(Hm, XY)
        uv, e1, ids = uv[ok], e1[ok], ids[ok]
        if not len(uv):
            return np.zeros((0, 3), np.float32), np.zeros(0), ids, nl
        nv = e1 - uv
        half = np.linalg.norm(nv, axis=1)
        nv = nv / np.maximum(half[:, None], 1e-9)
        side = (3.0 * np.maximum(half, 0.75) + blur_px + 1.0)[:, None]
        pts = np.concatenate([uv, uv + side * nv, uv - side * nv]).astype(np.float32)
        v = cv2.remap(img, pts[:, 0:1], pts[:, 1:2], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[:, 0]
        n = len(uv)
        c, l, r = v[:n], v[n:2 * n], v[2 * n:]
        return c - 0.5 * (l + r), np.linalg.norm(l - r, axis=1), ids, nl

    @staticmethod
    def per_line_mean(resp: np.ndarray, ids: np.ndarray, nl: int, min_pts: int = 8) -> np.ndarray:
        """点ごとの値 (N,) または (N, M) を、ラインごとに平均して足す（点が ``min_pts`` 未満のラインは除く）。"""
        cnt = np.bincount(ids, minlength=nl)
        use = cnt >= min_pts
        if resp.ndim == 1:
            sm = np.bincount(ids, weights=resp, minlength=nl)
            return float((sm[use] / cnt[use]).sum())
        out = np.zeros(resp.shape[1])
        for j in range(resp.shape[1]):
            sm = np.bincount(ids, weights=resp[:, j], minlength=nl)
            out[j] = (sm[use] / cnt[use]).sum()
        return out

    def _court_score(self, img: np.ndarray, Hm: np.ndarray, model, color: np.ndarray, blur_px: float) -> float:
        """ホモグラフィで描いた全ラインに沿った、ラインらしさの平均（ライン 1 本ごとに平均してから足す）。"""
        dv, sd, ids, nl = self.line_profiles(img, Hm, model, blur_px)
        if not len(ids):
            return 0.0
        resp = np.clip(dv @ color - 0.5 * sd, 0.0, 20.0)
        return self.per_line_mean(resp, ids, nl)

    def _align(self, lab: np.ndarray, H0: np.ndarray, color: np.ndarray, k: float,
               W: np.ndarray, P: np.ndarray, sg: np.ndarray, seed: np.ndarray) -> np.ndarray:
        """コートのラインの画像上の位置を、ラインらしさの合計が最大になるように動かす。

        クリックした点の近く（実寸で 1 m の範囲）のラインから合わせ始め、範囲を広げながら合わせ直す
        （クリックから遠いラインは予想位置のずれが大きく、最初から全部に合わせると別の線に引っ張られる）。
        各段階では、その範囲の四隅の画像上の位置（8 個の値）を、ぼかした画像から細かい画像へ
        Nelder–Mead で動かす。コートを細い帯につぶして全ラインを 1 本の濃い線に重ねる解に落ちないように、
        クリックした点（``W`` の実寸 → ``P`` の画像座標）から離れるほど罰を与える。
        """
        from scipy.optimize import minimize

        Hm = H0
        sig = 2.0 * self.click_sigma_px * k * sg
        lo, hi = seed.min(0), seed.max(0)
        pyr = {bl: cv2.GaussianBlur(lab, (0, 0), bl * k) * self.LAB_W for bl in (4.0, 2.0, 1.0)}
        for stage, grow in enumerate((1.0, 2.5, 5.0, 14.0)):
            box = (max(-HALF_L, lo[0] - grow), min(HALF_L, hi[0] + grow),
                   max(-HALF_W, lo[1] - grow), min(HALF_W, hi[1] + grow))
            model = self._model_points(box)
            if not model:
                continue
            quad = np.array([[box[0], box[2]], [box[1], box[2]], [box[1], box[3]], [box[0], box[3]]], float)
            if not np.isfinite(_apply_h(Hm, quad)).all():
                continue
            sched = (((4.0, self.search_px), (2.0, 0.4 * self.search_px), (1.0, 0.15 * self.search_px)) if stage == 0
                     else ((2.0, 0.4 * self.search_px), (1.0, 0.15 * self.search_px)))
            for blur, step in sched:
                img = pyr[blur]
                base = _apply_h(Hm, quad)
                J = self._corner_scale(Hm, quad, lab.shape[:2])

                def to_h(x: np.ndarray) -> np.ndarray:
                    q = (base + x.reshape(4, 2) * J[:, None]).astype(np.float32)
                    return cv2.getPerspectiveTransform(quad.astype(np.float32), q).astype(float)

                def f(x: np.ndarray) -> float:
                    Hc = to_h(x)
                    d = np.linalg.norm(_apply_h(Hc, W) - P, axis=1)
                    return -self._court_score(img, Hc, model, color, blur * k) + 0.5 * float(np.sum((d / sig) ** 2))

                simplex = np.vstack([np.zeros(8), np.eye(8) * step * k])
                best = None
                for start in (self._starts(step * k) if blur == 4.0 else [np.zeros(8)]):
                    sol = minimize(f, start, method="Nelder-Mead",
                                   options={"initial_simplex": simplex + start, "xatol": 0.2 * k, "fatol": 1e-3,
                                            "maxfev": 1200})
                    if best is None or sol.fun < best.fun:
                        best = sol
                Hm = to_h(best.x)
                Hm = Hm / Hm[2, 2]
        return Hm

    @staticmethod
    def _starts(step: float) -> list[np.ndarray]:
        """Nelder–Mead の出発点（今の位置と、全体を少しずらした位置）。"""
        out = [np.zeros(8)]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            out.append(np.tile([dx * step, dy * step], 4))
        return out

    @staticmethod
    def _corner_scale(Hm: np.ndarray, quad: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        """四隅を 1 動かしたとき、画像の中の（四隅で囲んだ範囲の）点が動く量の逆数（四隅ごと）。

        四隅が画像のずっと外にあると、四隅の小さな動きでも画像の中のラインが大きく動くので、動きの大きさをそろえる。
        """
        h, w = shape
        out = np.ones(4)
        q = _apply_h(Hm, quad).astype(np.float32)
        g = np.array([[x, y] for x in np.linspace(quad[0, 0], quad[1, 0], 9) for y in np.linspace(quad[0, 1], quad[2, 1], 5)])
        uv = _apply_h(Hm, g)
        inside = np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < w) & (uv[:, 1] < h)
        if inside.sum() < 3:
            return out
        for i in range(4):
            dq = q.copy()
            dq[i] += np.float32([1.0, 0.0])
            H2 = cv2.getPerspectiveTransform(quad.astype(np.float32), dq).astype(float)
            mv = np.linalg.norm(_apply_h(H2, g[inside]) - uv[inside], axis=1).max()
            out[i] = 1.0 / max(float(mv), 1e-3)
        return np.clip(out, 0.0, 50.0)

    # ------------------------------------------------------------------ ラインの測定
    def _samples(self, Hm: np.ndarray, name: str, shape: tuple[int, int]):
        """ライン上の点（交点のすぐ近くは除く）と、隣の平行なラインまでの画像上の距離。"""
        ax, c, spans = LINES[name]
        s = np.concatenate([np.arange(a + 0.12, b - 0.12, 0.04) for a, b in spans])
        cross = sorted({v for n2, (ax2, v, _) in LINES.items() if ax2 != ax})
        s = s[np.min(np.abs(s[:, None] - np.array(cross)[None]), axis=1) > 0.12]
        XY = np.c_[np.full(len(s), c), s] if ax == "x" else np.c_[s, np.full(len(s), c)]
        uv = _apply_h(Hm, XY)
        other = [v for n2, (ax2, v, _) in LINES.items() if ax2 == ax and n2 != name]
        gap = np.full(len(s), np.inf)
        for v in other:
            XY2 = XY.copy()
            XY2[:, 0 if ax == "x" else 1] = v
            gap = np.minimum(gap, np.linalg.norm(_apply_h(Hm, XY2) - uv, axis=1))
        # ラインの幅 [px]
        d = np.array([1.0, 0.0]) if ax == "x" else np.array([0.0, 1.0])
        width = np.linalg.norm(_apply_h(Hm, XY + d * LINE_WIDTH_M / 2) - _apply_h(Hm, XY - d * LINE_WIDTH_M / 2), axis=1)
        h, w = shape
        ok = np.isfinite(uv).all(1) & (uv[:, 0] > 4) & (uv[:, 1] > 4) & (uv[:, 0] < w - 5) & (uv[:, 1] < h - 5) & \
            self.in_front(Hm, XY)
        return uv[ok], gap[ok], width[ok]

    def _measure(self, lab: np.ndarray, Hm: np.ndarray, rmax: float, color: np.ndarray | None = None) -> dict:
        """各ラインの予想位置のまわりで、ラインの中心を測る。

        1 点ずつ最大の応答を選ぶと、床板の継ぎ目などに飛びやすい。そこで先に、ラインの両端での
        ずれ (a, b) の組を総当たりして、直線に沿った応答の合計が最大になる直線を選び、
        その直線の近く（±2 px 程度）で 1 点ずつ中心を測る。
        """
        out = {}
        for name in LINES:
            uv, gap, width = self._samples(Hm, name, lab.shape[:2])
            # ありえないほど太いライン（カメラのすぐ近くの床など、外れたホモグラフィで起きる）は使わない
            sane = np.isfinite(width) & (width < 0.03 * max(lab.shape[:2]))
            uv, gap, width = uv[sane], gap[sane], width[sane]
            if len(uv) < 8:
                continue
            tang = np.gradient(uv, axis=0)
            tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)
            nrm = np.c_[-tang[:, 1], tang[:, 0]]
            R = np.clip(np.minimum(rmax, 0.45 * gap), 1.5, None)
            hw = np.clip(width / 2, 0.75, None)                  # ラインの半分の幅 [px]
            step = 0.5
            Rm = float(R.max())
            pad = 3 * float(hw.max()) + 2
            off = np.arange(-Rm - pad, Rm + pad + step / 2, step)
            mapx = (uv[:, 0:1] + off[None] * nrm[:, 0:1]).astype(np.float32)
            mapy = (uv[:, 1:2] + off[None] * nrm[:, 1:2]).astype(np.float32)
            prof = cv2.remap(lab, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) * self.LAB_W
            K = prof.shape[1]
            cs = np.concatenate([np.zeros((len(prof), 1, 3), np.float32), np.cumsum(prof, axis=1)], axis=1)
            idx = np.arange(K)[None]
            hb = np.clip(np.round(hw / step).astype(int), 1, None)[:, None]

            def box(c0, c1):
                c0, c1 = np.clip(c0, 0, K), np.clip(c1, 0, K)
                return (np.take_along_axis(cs, c1[..., None], 1) - np.take_along_axis(cs, c0[..., None], 1)) /                     np.maximum(c1 - c0, 1)[..., None]

            cen = box(idx - hb, idx + hb + 1)
            left = box(idx - 3 * hb - 1, idx - hb)
            right = box(idx + hb + 1, idx + 3 * hb + 2)
            dv = cen - 0.5 * (left + right)
            resp = np.linalg.norm(dv, axis=2) - 0.6 * np.linalg.norm(left - right, axis=2)
            if color is not None:
                # 床との色の差の向きが、コートのライン（ほかのラインで測った向き）と同じ帯だけを使う
                cosv = (dv @ color) / np.maximum(np.linalg.norm(dv, axis=2), 1e-6)
                resp = np.where(cosv > 0.8, resp, -np.inf)
            resp = np.where(np.abs(off)[None] <= R[:, None], resp, -np.inf)
            # 直線に沿った応答の合計（1 点の強い応答に引っ張られないよう上限をつける）
            rpos = np.where(np.isfinite(resp), np.maximum(resp, 0.0), 0.0)
            cap = np.percentile(rpos[rpos > 0], 90) if (rpos > 0).any() else 1.0
            rpos = np.minimum(rpos, cap)
            t = np.linspace(0.0, 1.0, len(uv))
            cand = np.arange(-Rm, Rm + step / 2, step)
            a, b = np.meshgrid(cand, cand, indexing="ij")
            a, b = a.ravel(), b.ravel()
            best, best_s = (0.0, 0.0), -1.0
            for i0 in range(0, len(a), 4096):
                o = a[i0:i0 + 4096, None] + (b[i0:i0 + 4096, None] - a[i0:i0 + 4096, None]) * t[None]
                jj = np.clip(np.round((o - off[0]) / step).astype(int), 0, K - 1)
                sc = rpos[np.arange(len(uv))[None], jj].sum(1)
                m = int(np.argmax(sc))
                if sc[m] > best_s:
                    best_s, best = float(sc[m]), (float(a[i0 + m]), float(b[i0 + m]))
            center = best[0] + (best[1] - best[0]) * t
            band = max(2.0, 0.6 * float(np.median(hw)) + 1.0)
            resp2 = np.where(np.abs(off[None] - center[:, None]) <= band, resp, -np.inf)
            j = np.argmax(resp2, axis=1)
            val = resp2[np.arange(len(resp2)), j]
            good = np.isfinite(val) & (val > 3.0) & (j > 0) & (j < K - 1)
            if good.sum() < 6:
                continue
            gi = np.flatnonzero(good)
            jj = j[gi]
            y0, y1, y2 = resp2[gi, jj - 1], resp2[gi, jj], resp2[gi, jj + 1]
            y0 = np.where(np.isfinite(y0), y0, y1)
            y2 = np.where(np.isfinite(y2), y2, y1)
            den = y0 - 2 * y1 + y2
            with np.errstate(divide="ignore", invalid="ignore"):
                sub = np.where(den < 0, 0.5 * (y0 - y2) / np.where(den == 0, -1, den), 0.0)
            dpos = off[jj] + np.clip(sub, -1, 1) * step
            out[name] = {"pts": uv[gi] + dpos[:, None] * nrm[gi], "val": val[gi], "dv": dv[gi, jj], "n": len(uv),
                         "width": float(np.median(width)), "support": best_s / max(len(uv), 1) / cap}
        return out

    def color_near(self, lab: np.ndarray, Hm: np.ndarray, XY: np.ndarray, radius_m: float = 1.5) -> np.ndarray | None:
        """実寸の点 ``XY`` のまわり（``radius_m`` 以内）のラインだけで測った、ラインと床の色の差の向き。

        体育館には他競技のライン（別の色）もあるので、確かにコートのラインがある所（クリックした角）で測る。
        """
        k = max(lab.shape[:2]) / 1920.0
        meas = self._measure(lab, Hm, 8.0 * k)
        Hi = np.linalg.inv(Hm)
        near = {}
        for n, m in meas.items():
            w = _apply_h(Hi, m["pts"])
            keep = np.linalg.norm(w - np.asarray(XY, float)[None], axis=1) <= radius_m
            if keep.sum() >= 4:
                near[n] = {"dv": m["dv"][keep], "val": m["val"][keep]}
        return self._line_color(near)

    @staticmethod
    def _line_color(meas: dict) -> np.ndarray | None:
        """ラインと床の色の差の向き（各ラインの強い応答の向きの中央値）。

        コートのラインは体育館によって白・黄・緑などで、床より明るいとは限らないので、動画ごとに測る。
        """
        dv = [m["dv"][m["val"] >= np.median(m["val"])] for m in meas.values() if len(m["val"])]
        if not dv:
            return None
        dv = np.concatenate(dv)
        u = dv / np.maximum(np.linalg.norm(dv, axis=1, keepdims=True), 1e-6)
        m = np.median(u, axis=0)
        return (m / max(float(np.linalg.norm(m)), 1e-6)).astype(np.float32)

    @staticmethod
    def _fit_lines(meas: dict, k: float) -> dict[str, np.ndarray]:
        fits = {}
        for name, m in meas.items():
            pts = m["pts"]
            tol = max(1.2 * k, 0.25 * m["width"] + 0.8)
            fit = AutoCourtCalibrator._ransac_line(pts, tol)
            if fit is None:
                continue
            _, inl = fit
            if inl.sum() < max(8, 0.2 * m["n"]):
                continue
            fits[name] = pts[inl]
        return fits

    def _fit_h(self, H0: np.ndarray, fits: dict[str, np.ndarray], W: np.ndarray, P: np.ndarray, k: float,
               sg: np.ndarray):
        """測った点がモデルのラインに乗り、クリックした点とも大きくずれないように H を詰める。

        ほかのラインと合わないライン（別の線に吸い付いたもの）を外しながら当てはめ直す。
        """
        keep = list(fits)
        h0 = (H0 / H0[2, 2]).ravel()[:8]
        sig_c = (self.click_sigma_px * k * sg)[:, None]
        Hm = None
        for _ in range(4):
            if not keep:
                return None, []
            items = [(fits[n], LINES[n][:2]) for n in keep]
            # 長いラインほど点が多いので、1 本あたりの重みをそろえる（点の数の平方根で割る）
            wts = [1.0 / math.sqrt(len(p) / 20.0) if len(p) > 20 else 1.0 for p, _ in items]

            def resid(hv: np.ndarray) -> np.ndarray:
                Hc = np.r_[hv, 1.0].reshape(3, 3)
                try:
                    Hi = np.linalg.inv(Hc)
                except np.linalg.LinAlgError:
                    return np.full(sum(len(p) for p, _ in items) + 2 * len(W), 1e3)
                out = []
                for (p, (ax, c)), wt in zip(items, wts):
                    li = Hi.T @ _world_line(ax, c)
                    li = li / np.linalg.norm(li[:2])
                    out.append(wt * (np.c_[p, np.ones(len(p))] @ li))
                out.append(((_apply_h(Hc, W) - P) / sig_c).ravel())
                return np.concatenate(out)

            sol = least_squares(resid, h0, loss="huber", f_scale=1.5 * k, x_scale="jac", max_nfev=300)
            Hm = np.r_[sol.x, 1.0].reshape(3, 3)
            med = {n: float(np.median(np.abs(np.c_[fits[n], np.ones(len(fits[n]))] @ _image_line(Hm, *LINES[n][:2]))))
                   for n in keep}
            worst = max(med, key=med.get)
            if med[worst] <= max(2.0 * k, 0.4 * float(np.median(list(med.values()))) + 1.5 * k):
                return Hm, keep
            keep.remove(worst)
            h0 = sol.x
        return Hm, keep

    @staticmethod
    def _both_lines(name: str, X: np.ndarray, found: dict) -> bool:
        """ランドマーク（2 本のラインの外側の縁の角）をつくる 2 本のラインが、両方とも測れたか。"""
        x, y = float(X[0]), float(X[1])
        xs = [n for n, (ax, c, _) in LINES.items() if ax == "x" and abs(abs(c) - abs(x)) < 0.05 and np.sign(c) == np.sign(x)]
        ys = [n for n, (ax, c, _) in LINES.items() if ax == "y" and abs(c - y) < 0.05]
        return any(n in found for n in xs) and any(n in found for n in ys)


def snap_court(frame: np.ndarray, clicks: dict[str, tuple[float, float]], table: dict[str, np.ndarray]) -> SnapResult:
    """``CourtSnapper().snap`` の短縮形。"""
    return CourtSnapper().snap(frame, clicks, table)


__all__ = ["CourtSnapper", "SnapResult", "snap_court"]
