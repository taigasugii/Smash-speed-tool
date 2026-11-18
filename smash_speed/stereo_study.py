"""2 台のカメラの合成データと、置き方・fps・時刻合わせの誤差による初速の誤差の調べ（撮り方の勧めの根拠）。

合成ラリー（``synth_rally.make_session``）を 2 台の仮想カメラで別々に観測する。それぞれのカメラは自分の時計
（録画を始めた時刻・時計の進み方）とコマの時刻（fps・ゆらぎ）を持ち、検出には雑音・見落としが入る。
カメラはコートの点の画像の位置（雑音つき）からキャリブレーションし直す（実際の「コート合わせ」の誤差の代わり）。

``python -m smash_speed.stereo_study --out outputs/stereo_study`` で表（``study.csv``・``study.md``）を作る。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from smash_speed.camera import Camera, look_at
from smash_speed.physics import MPS_TO_KMH, PhysicsParams, simulate
from smash_speed.stereo_sync import SPEED_OF_SOUND_MPS, SyncModel, video_sync
from smash_speed.stereo_truth import TruthOptions, solve_session
from smash_speed.synth_rally import SyntheticRally, make_session, rally_positions

# スマホの置き方（カメラの位置 [m]）。コートは x が長い方（ネットが x = 0、奥のラインが ±6.7）、y が幅（±3.05）。
# 1 倍のレンズ（横の画角 約 69°、1920 px で f ≈ 1400 px）を基本に、入りきらない近い置き方は 0.5 倍（f ≈ 700 px）
PLACEMENTS: dict[str, dict[str, Any]] = {
    # 後ろの両角（同じ側のコートの後ろ、左右）
    "back_corners": {"eyes": [(-10.0, -3.6, 2.0), (-10.0, 3.6, 2.0)], "target": (1.0, 0.0, 2.2), "f": 950.0,
                     "label": "後ろの両角（コートの後ろ 3.3 m・左右 7 m 離す）"},
    # 後ろの角と、横（ネットの延長）
    "back_side": {"eyes": [(-10.0, -3.6, 2.0), (0.0, -8.0, 2.0)], "target": (0.0, 0.0, 2.2), "f": 800.0,
                  "label": "後ろの角と、横（ネットの延長・サイドラインから 5 m）"},
    # 対角の角（コートをはさんで向かい合う）
    "diagonal": {"eyes": [(-10.0, -3.6, 2.0), (10.0, 3.6, 2.0)], "target": (0.0, 0.0, 2.2), "f": 950.0,
                 "label": "対角の角（コートをはさんで向かい合う）"},
    # 同じ横の側の両角
    "same_side": {"eyes": [(-10.0, -3.6, 2.0), (10.0, -3.6, 2.0)], "target": (0.0, 0.0, 2.2), "f": 950.0,
                  "label": "同じ横の側の、手前と奥の角"},
    # 後ろの真ん中と、横
    "back_center_side": {"eyes": [(-10.5, 0.0, 2.0), (0.0, -8.0, 2.0)], "target": (0.0, 0.0, 2.2), "f": 800.0,
                         "label": "後ろの真ん中と、横（直角）"},
    # 悪い例：2 台を並べて置く
    "narrow": {"eyes": [(-10.0, -0.5, 2.0), (-10.0, 0.5, 2.0)], "target": (1.0, 0.0, 2.2), "f": 950.0,
               "label": "悪い例：後ろに 1 m だけ離して並べる"},
}


def placement_cameras(name: str, height_m: float | None = None, width: int = 1920, height: int = 1080,
                      f_px: float | None = None) -> list[Camera]:
    """置き方の名前（``PLACEMENTS``）から 2 台の仮想カメラを作る。

    ``height_m`` でスマホの高さ、``f_px`` でレンズ（1920 px の幅での焦点距離。0.5 倍 ≈ 700、1 倍 ≈ 1400）を変えられる。
    """
    p = PLACEMENTS[name]
    cams = []
    for eye in p["eyes"]:
        e = np.array(eye, float)
        if height_m is not None:
            e[2] = height_m
        cams.append(look_at(e, np.array(p["target"], float), (f_px or p["f"]) * width / 1920, width, height))
    return cams


@dataclass
class CameraClock:
    """1 台のカメラの時計とコマ。"""

    fps: float = 60.0
    start_s: float = 0.0             # 録画を始めた本当の時刻 [s]（動画の時刻 0）
    drift: float = 0.0               # 時計の進み方のずれ（動画の時刻 = (本当の時刻 − start) × (1 + drift)）
    jitter_s: float = 0.0            # コマの時刻のゆらぎ（可変フレームレート。標準偏差）[s]
    rolling_shutter_s: float = 0.0   # 画像の上の行から下の行までの読み出しの時間 [s]
    av_offset_s: float = 0.0         # 音が映像より遅れて記録される時間 [s]

    def video_time(self, t_true: Any) -> Any:
        return (np.asarray(t_true, float) - self.start_s) * (1.0 + self.drift)


def true_sync(c1: CameraClock, c2: CameraClock) -> SyncModel:
    """2 台の時計の本当の関係（t2 = offset + (1 + drift) t1）。"""
    r = (1.0 + c2.drift) / (1.0 + c1.drift)
    return SyncModel(float(c2.video_time(c1.start_s)), r - 1.0)


def observe_camera(rallies: list[SyntheticRally], camera: Camera, clock: CameraClock, phys: PhysicsParams,
                   rng: np.random.Generator, noise_px: float = 1.5, dropout: float = 0.1, outlier: float = 0.0,
                   post_s: float = 0.5) -> pd.DataFrame:
    """合成ラリーを 1 台のカメラで観測した検出結果（時刻はそのカメラの動画の時刻）。

    ``outlier`` の割合のコマは、シャトルではない所（画像の中のでたらめな点）を検出したことにする。
    読み出しの時間があれば、画像の下の行ほど遅い時刻のシャトルが写る。
    列は ``synth_rally.observe_session`` と同じ（``x_m`` などは正解）。
    """
    dt = 1.0 / clock.fps
    t0 = max(rallies[0].t_start - 0.3, clock.start_s)
    n = int(np.ceil((rallies[-1].t_end + post_s - t0) / dt))
    t_true = t0 + rng.uniform(0.0, dt) + np.arange(n) * dt
    if clock.jitter_s:
        t_true = np.sort(t_true + rng.normal(0.0, clock.jitter_s, n))

    def positions(tt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos = np.full((len(tt), 3), np.nan)
        which = np.full(len(tt), -1)
        ridx = np.full(len(tt), -1)
        for k, r in enumerate(rallies):
            p, w = rally_positions(r, tt, phys)
            m = w >= 0
            pos[m], which[m], ridx[m] = p[m], w[m], k
        return pos, which, ridx

    pos, which, ridx = positions(t_true)
    uv = camera.project(np.nan_to_num(pos))
    if clock.rolling_shutter_s:
        # 写る行に応じた時刻で観測し直す（1 回で十分近い）
        t_row = t_true + (np.nan_to_num(uv[:, 1]) / camera.height - 0.5) * clock.rolling_shutter_s
        pos, which, ridx = positions(t_row)
        uv = camera.project(np.nan_to_num(pos))
    uv = uv + rng.normal(0.0, noise_px, uv.shape)
    inside = (np.isfinite(pos).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < camera.width)
              & (uv[:, 1] >= 0) & (uv[:, 1] < camera.height))
    visible = inside & (rng.uniform(size=n) >= dropout)
    if outlier:
        bad = visible & (rng.uniform(size=n) < outlier)
        uv[bad] = rng.uniform([0, 0], [camera.width, camera.height], (int(bad.sum()), 2))
    return pd.DataFrame({
        "frame": np.arange(n), "timestamp": clock.video_time(t_true),
        "x_px": np.where(visible, uv[:, 0], np.nan), "y_px": np.where(visible, uv[:, 1], np.nan),
        "visible": visible.astype(int), "confidence": visible.astype(float),
        "x_m": pos[:, 0], "y_m": pos[:, 1], "z_m": pos[:, 2], "shot_index": which, "rally_index": ridx,
    })


def hit_sounds(rallies: list[SyntheticRally], camera: Camera, clock: CameraClock, rng: np.random.Generator,
               sigma_s: float = 0.0002) -> list[float]:
    """打球音がそのカメラに届く時刻（動画の時刻。打点からの距離 / 音速だけ遅れ、音と映像のずれが足される）。"""
    out = []
    for r in rallies:
        for s in r.segments:
            d = float(np.linalg.norm(camera.center - s.p0)) / SPEED_OF_SOUND_MPS
            out.append(float(clock.video_time(s.t_hit + d)) + clock.av_offset_s + float(rng.normal(0.0, sigma_s)))
    return out


def recalibrated(camera: Camera, cfg: dict[str, Any], rng: np.random.Generator, noise_px: float) -> Camera:
    """コートの点の画像の位置（雑音つき）からキャリブレーションし直したカメラ（「コート合わせ」の誤差の代わり）。"""
    from smash_speed.calibration import calibrate, landmark_table

    if noise_px <= 0:
        return camera
    table = landmark_table(cfg)
    names, W = [], []
    for n, X in table.items():
        uv = camera.project(X[None])[0]
        if np.isfinite(uv).all() and 0 <= uv[0] < camera.width and 0 <= uv[1] < camera.height:
            names.append(n)
            W.append(X)
    W = np.array(W)
    res = calibrate(names, W, camera.project(W) + rng.normal(0.0, noise_px, (len(W), 2)), camera.width, camera.height)
    return res.camera


# =============================================================================
# 1 つの条件を動かす
# =============================================================================
@dataclass
class StudyCase:
    """調べる条件 1 つ。"""

    name: str
    placement: str = "back_corners"
    fps: tuple[float, float] = (60.0, 60.0)
    height_m: float | None = None
    f_px: float | None = None           # レンズ（None なら置き方ごとの値）
    noise_px: float = 1.5
    dropout: float = 0.1
    outlier: float = 0.02
    calib_noise_px: float = 1.0
    sync_error_s: float = 0.0           # 時刻合わせにわざと入れる誤差（映像で合わせ直さないとき）
    refine_sync: bool = True            # 映像で時刻を合わせる（大まかなずれ ±30 ms から始める）
    drift_ppm: float = 30.0
    jitter_s: float = 0.0005
    rolling_shutter_s: float = 0.0      # 本当の読み出しの時間（解析では知らないものとする）
    av_offset_s: tuple[float, float] = (0.004, -0.003)   # スマホごとの音と映像のずれ
    terminal_velocity: float | None = None               # 本当の終端速度（解析は設定の値のまま）
    hits: str = "rough"                 # rough（本当の時刻を 1 コマほどずらして渡す）| found（3 次元の点から見つける）
    n_rallies: int = 6
    seeds: tuple[int, ...] = (0, 1)
    extra: dict[str, Any] = field(default_factory=dict)


def run_case(case: StudyCase, cfg: dict[str, Any], seed: int, uncertainty: bool = False) -> pd.DataFrame:
    """条件 1 つ・乱数 1 つで合成ラリーを作り、2 台から解析して、正解のショットごとの結果の表を返す。"""
    rng = np.random.default_rng(seed)
    phys = PhysicsParams.from_config(cfg)
    world = phys if case.terminal_velocity is None else PhysicsParams(
        phys.g, case.terminal_velocity, phys.method, phys.rtol, phys.atol)
    rallies = make_session(rng, world, n_rallies=case.n_rallies)
    cams = placement_cameras(case.placement, case.height_m, f_px=case.f_px)
    clocks = [CameraClock(case.fps[0], rallies[0].t_start - 2.0, 0.0, case.jitter_s, case.rolling_shutter_s,
                          case.av_offset_s[0]),
              CameraClock(case.fps[1], rallies[0].t_start - 3.37, case.drift_ppm * 1e-6, case.jitter_s,
                          case.rolling_shutter_s, case.av_offset_s[1])]
    dets = [observe_camera(rallies, c, k, world, rng, case.noise_px, case.dropout, case.outlier)
            for c, k in zip(cams, clocks)]
    est = [recalibrated(c, cfg, rng, case.calib_noise_px) for c in cams]
    truth_sync = true_sync(*clocks)
    sync, sigma, gap = truth_sync.shifted(case.sync_error_s), abs(case.sync_error_s), None
    if case.refine_sync:
        # 音だけで合わせた大まかなずれ（打点から 2 台への距離の差で ±30 ms ほどずれる）から、映像で合わせる
        rep = video_sync(dets, est, SyncModel(truth_sync.offset_s + rng.uniform(-0.03, 0.03)))
        sync, sigma, gap = rep.model, rep.sigma_s, rep.gap_cm
    segs = [s for r in rallies for s in r.segments]
    t_true = np.array([float(clocks[0].video_time(s.t_hit)) for s in segs])
    onsets = [hit_sounds(rallies, c, k, rng) for c, k in zip(cams, clocks)]
    if case.hits == "found":
        from smash_speed.stereo_sync import RayPairs
        from smash_speed.stereo_truth import find_hits_3d
        hit_times = find_hits_3d(*RayPairs(dets, est).points(sync), phys)
    else:
        step = 1.0 / min(case.fps)
        hit_times = list(t_true + rng.uniform(-0.5, 0.5, len(t_true)) * step)
    o = TruthOptions(uncertainty=uncertainty, **case.extra)
    shots, info = solve_session(dets, est, sync, hit_times, phys, o, onsets, sigma)

    # 正解のショットと、時刻の近い順に 1 対 1 で組にする
    mid = truth_sync.to_cam2(np.mean(t_true))
    rows = []
    used: set[int] = set()
    for i, s in enumerate(segs):
        row: dict[str, Any] = {"case": case.name, "seed": seed, "shot_type": s.shot_type, "true_kmh": s.speed0 * MPS_TO_KMH,
                               "sync_err_ms": float(sync.to_cam2(np.mean(t_true)) - mid) * 1000, "gap_cm": gap,
                               "first_in_rally": any(s is r.segments[0] for r in rallies)}
        j = min((j for j in range(len(shots)) if j not in used), key=lambda j: abs(shots[j].t_hit - t_true[i]),
                default=None)
        if j is not None and abs(shots[j].t_hit - t_true[i]) <= 0.15:
            used.add(j)
            e = shots[j]
            row["fitted"] = e.fit is not None
            if e.fit is not None and e.ok(o):
                # 本当の軌道の、解析の打った時刻での速さ（打った時刻の誤差を除いた誤差）
                _, v = simulate(s.p0, s.v0, s.t_hit, np.array([s.t_hit + (e.t_hit - t_true[i])]), world)
                same = float(np.linalg.norm(v[0]) * MPS_TO_KMH)
                row.update({"kmh": e.kmh, "err_pct": 100 * (e.kmh - row["true_kmh"]) / row["true_kmh"],
                            "err_same_t_pct": 100 * (e.kmh - same) / same if np.isfinite(same) else np.nan,
                            "dt_ms": (e.t_hit - t_true[i]) * 1000, "hit_err_cm": float(np.linalg.norm(e.fit.p0 - s.p0)) * 100,
                            "method": e.hit_method, "sigma_kmh": e.sigma_kmh, "n1": e.n_obs[0], "n2": e.n_obs[1],
                            "angle_err_deg": float(np.degrees(np.arctan2(e.fit.v0[2], np.linalg.norm(e.fit.v0[:2])))
                                                   - s.elevation_deg)})
                for k, v0 in e.naive_kmh.items():
                    row[f"naive_{k}_err_pct"] = 100 * (v0 - row["true_kmh"]) / row["true_kmh"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df["n_extra"] = len(shots) - len(used)
    return df


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    """条件 1 つの表のまとめ（誤差は初速の % の絶対値）。"""
    got = df[df.kmh.notna()] if "kmh" in df else df.iloc[:0]
    out: dict[str, Any] = {"n_shots": int(len(df)), "got_pct": round(100 * len(got) / max(len(df), 1), 1)}
    if not len(got):
        return out
    a = got.err_pct.abs()
    sm = got[got.shot_type == "smash"]
    fast = got[got.true_kmh >= 150]
    out.update({
        "median_pct": round(float(a.median()), 2), "mape_pct": round(float(a.mean()), 2),
        "p90_pct": round(float(a.quantile(0.9)), 2), "bias_pct": round(float(got.err_pct.mean()), 2),
        "smash_mape_pct": round(float(sm.err_pct.abs().mean()), 2) if len(sm) else None,
        "fast_mape_pct": round(float(fast.err_pct.abs().mean()), 2) if len(fast) else None,
        "same_t_median_pct": round(float(got.err_same_t_pct.abs().median()), 2),
        "dt_median_ms": round(float(got.dt_ms.abs().median()), 2),
        "hit_err_median_cm": round(float(got.hit_err_cm.median()), 1),
        "angle_err_median_deg": round(float(got.angle_err_deg.abs().median()), 2),
        "sync_err_ms": round(float(df.sync_err_ms.abs().mean()), 3),
        "gap_cm": None if df.gap_cm.isna().all() else round(float(df.gap_cm.mean()), 1),
    })
    for k in ("nearest", "interpolated"):
        c = f"naive_{k}_err_pct"
        if c in got and got[c].notna().any():
            out[f"naive_{k}_median_pct"] = round(float(got[c].abs().median()), 1)
    if "sigma_kmh" in got and got.sigma_kmh.notna().any():
        z = ((got.kmh - got.true_kmh) / got.sigma_kmh).abs().dropna()
        out["within_1sigma_pct"] = round(100 * float((z <= 1).mean()), 0)
        out["within_2sigma_pct"] = round(100 * float((z <= 2).mean()), 0)
    return out


def _run(args: tuple[StudyCase, dict[str, Any], int, bool]) -> pd.DataFrame:
    return run_case(*args)


def study_cases() -> list[StudyCase]:
    """撮り方の勧めを決めるための条件の一覧。"""
    base = StudyCase("基本（後ろの両角・60 fps）")
    cases = [base]
    for p in PLACEMENTS:
        if p != "back_corners":
            cases.append(replace(base, name=f"置き方：{PLACEMENTS[p]['label']}", placement=p))
    for h in (1.2, 3.0):
        cases.append(replace(base, name=f"高さ {h} m", height_m=h))
    cases.append(replace(base, name="レンズ 0.5 倍（超広角、f 700 px）", f_px=700.0))
    cases.append(replace(base, name="レンズ 1 倍（f 1400 px。手前のコートが入りきらない）", f_px=1400.0))
    for f in ((30.0, 30.0), (120.0, 120.0), (240.0, 240.0), (30.0, 240.0), (60.0, 120.0)):
        cases.append(replace(base, name=f"fps {f[0]:.0f}+{f[1]:.0f}", fps=f))
    for e in (0.0, 0.001, 0.002, 0.005, 0.010, 0.020):
        cases.append(replace(base, name=f"時刻合わせの誤差 {e * 1000:.0f} ms（映像で合わせない）", sync_error_s=e,
                             refine_sync=False))
    for n in (3.0,):
        cases.append(replace(base, name=f"検出の雑音 {n} px", noise_px=n))
    for c in (0.0, 3.0, 6.0):
        cases.append(replace(base, name=f"コート合わせの誤差 {c} px", calib_noise_px=c))
    cases.append(replace(base, name="読み出しの時間 10 ms（直さない）", rolling_shutter_s=0.010))
    cases.append(replace(base, name="読み出しの時間 10 ms・240 fps", rolling_shutter_s=0.010, fps=(240.0, 240.0)))
    cases.append(replace(base, name="終端速度が本当は 6.3 m/s", terminal_velocity=6.3))
    cases.append(replace(base, name="打球を自動で見つける", hits="found"))
    cases.append(replace(base, name="打球を自動で見つける・120 fps", hits="found", fps=(120.0, 120.0)))
    cases.append(replace(base, name="打球を自動で見つける・30 fps", hits="found", fps=(30.0, 30.0)))
    return cases


def run_study(cfg: dict[str, Any], out: Path, cases: list[StudyCase] | None = None, workers: int = 4,
              only: str | None = None) -> pd.DataFrame:
    """条件の一覧を動かし、``study.csv``（条件ごとのまとめ）・``study_shots.csv``・``study.md`` を書く。"""
    from concurrent.futures import ProcessPoolExecutor

    cases = cases or study_cases()
    if only:
        cases = [c for c in cases if only in c.name]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(c, cfg, s, c.name.startswith("基本")) for c in cases for s in c.seeds]   # 不確かさは基本の条件だけ（時間がかかる）
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            dfs = list(ex.map(_run, jobs))
    else:
        dfs = [_run(j) for j in jobs]
    shots = pd.concat(dfs, ignore_index=True)
    shots.to_csv(out / "study_shots.csv", index=False)
    rows = [{"case": c.name, **summarize(shots[shots.case == c.name])} for c in cases]
    df = pd.DataFrame(rows)
    df.to_csv(out / "study.csv", index=False)
    cols = [("case", "条件"), ("got_pct", "初速が出た割合 %"), ("median_pct", "誤差の中央値 %"), ("mape_pct", "平均 %"),
            ("p90_pct", "90% 点 %"), ("fast_mape_pct", "150 km/h 以上の平均 %"), ("dt_median_ms", "打った時刻 ms"),
            ("hit_err_median_cm", "打点 cm"), ("naive_nearest_median_pct", "素朴な三角測量 %")]
    lines = ["| " + " | ".join(c[1] for c in cols) + " |", "|---|" + "---:|" * (len(cols) - 1)]
    for r in rows:
        lines.append("| " + " | ".join("" if r.get(k) is None or r.get(k) != r.get(k) else str(r.get(k))
                                       for k, _ in cols) + " |")
    (out / "study.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "study.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return df


def main(argv: list[str] | None = None) -> None:
    from smash_speed.config import load_config

    ap = argparse.ArgumentParser(description="2 台のカメラの置き方・fps・時刻合わせの誤差による、初速の誤差を合成データで調べる")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--quick", action="store_true", help="基本の条件だけ（動くかの確かめ）")
    ap.add_argument("--only", help="名前にこの文字を含む条件だけ")
    ap.add_argument("--config", type=Path, default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    cases = study_cases()[:1] if a.quick else None
    df = run_study(cfg, a.out, cases, a.workers, a.only)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
