"""動画の中のスマッシュを一括で探し、いちばん速いものを見つける（2 段階）。

1. 候補探し（自動・数分）：
   動画全体の打球音を探し、強い音ごとに直後のシャトルを検出して、だいたいの初速を出す。
   打つ人・着地点は分からないので、「最初に写ったシャトルの位置を打点の高さ（hit_height）とみなし、
   そこを通る縦の面の向きを格子探索する」大まかな推定にする。動画は 1 回だけ縮小して読み、
   音の前後の区間だけを検出に回す（4K の動画を区間ごとに読み直すより 10 倍以上速い）。
2. 確認と詳しい解析：
   利用者が上位の候補の「打つ人の足元」「着地点」を確認（推定値を直す）し、その値で通常の解析を行う。
   候補探しで検出したシャトルの位置をそのまま使うので、検出のやり直しはしない。

結果は ``<出力先>/candidates.json`` にまとめ、GUI はこのファイルを読み書きする。
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import cv2
import numpy as np
import pandas as pd

from smash_speed import events
from smash_speed.calibration.io import load_calibration
from smash_speed.camera import Camera
from smash_speed.detection.baseline import BaselineParams, detect_frames
from smash_speed.errors import AnalysisError
from smash_speed.geometry import FlightPlane
from smash_speed.physics import FitOptions, Observations, PhysicsParams, fit_trajectory, simulate_until_ground
from smash_speed.video_io import VideoMeta, extract_audio, find_tool, load_wav, probe

CANDIDATES_FILE = "candidates.json"


@dataclass
class Candidate:
    """スマッシュの候補 1 つ。"""

    id: int
    t_onset: float                    # 打球音の時刻（映像の時刻）[s]
    strength: float                   # 打球音の強さ
    start_s: float                    # 詳しい解析の区間 [s]
    end_s: float
    status: str = "pending"           # ok | no_track | off_court | fit_failed
    smash_like: bool = False          # 下向きで速い（スマッシュらしい）
    doubtful: bool = False            # 推定が怪しい（ありえない速さ・ずれが大きいなど）
    note: str = ""
    n_frames: int = 0
    t_impact: float | None = None
    rough_kmh: float | None = None    # 候補探しでのだいたいの初速
    rough_rms_px: float | None = None
    hitter: dict[str, float] | None = None   # 打つ人の足元の推定（確認後は確認した値）[m]
    landing: dict[str, float] | None = None
    confirmed: bool = False           # 利用者が打つ人・着地点を確認した
    selected: bool = False            # 詳しく解析する
    thumb: str | None = None          # サムネイル（出力先からの相対パス）
    detections: str | None = None     # 検出結果 CSV（同）
    result: dict[str, Any] | None = None  # 詳しい解析の結果

    @property
    def label(self) -> str:
        return f"smash_{self.id:02d}_{self.t_onset:.1f}s"


@dataclass
class BatchOptions:
    video: Path
    calib: Path
    out_dir: Path
    capture_fps: float | None = None
    start_s: float | None = None
    end_s: float | None = None
    min_strength: float = 0.25        # 最も強い打球音に対する割合。これ以上の音を候補にする
    merge_s: float = 0.08             # これより近い音は 1 つにまとめる（反響など。速い打ち合いの返球は 0.15 s 前後で来るので、まとめすぎない）[s]
    pre_s: float = 0.25               # 検出に使う区間：音の pre_s 前〜post_s 後 [s]
    post_s: float = 0.85
    n_select: int = 8                 # 最初から「詳しく解析する」にしておく候補の数（速い順）。だいたいの初速は ±30% ほどずれるので多めに
    roi: list[float] | None = None    # 検出する範囲 [x0, y0, x1, y1]（元解像度 px）
    audio: Path | None = None         # 別の音声ファイル（動画の時刻と同じ時刻の WAV）
    extra: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# 動画を 1 回だけ読む
# =============================================================================
def stream_gray(meta: VideoMeta, width: int, cfg: dict[str, Any] | None = None,
                start_s: float | None = None, end_s: float | None = None
                ) -> Iterator[tuple[int, float, np.ndarray]]:
    """ffmpeg で動画を縮小したグレースケールで先頭から順に読み、(番号, 実時間 [s], 画像) を返す。"""
    w = int(width)
    h = int(round(meta.height * w / meta.width / 2)) * 2
    ts = meta.timestamps
    i0 = 0 if start_s is None else max(0, int(np.searchsorted(ts, start_s)) - 1)
    i1 = len(ts) - 1 if end_s is None else min(len(ts) - 1, int(np.searchsorted(ts, end_s, side="right")))
    # 途中から読む（-ss）と、動画によってはフレーム番号が 1 つずれる（先頭のフレームの時刻が不規則な動画など）。
    # 縮小しながらの読み込みは実時間の数倍速いので、いつも先頭から読み、i0 より前は捨てる
    cmd = [find_tool("ffmpeg", cfg), "-v", "error", "-nostdin", "-i", str(meta.path),
           "-vf", f"scale={w}:{h}:flags=area", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=w * h * 4)
    n = w * h
    k = 0
    try:
        while True:
            buf = proc.stdout.read(n)
            if buf is None or len(buf) < n:
                break
            if k >= len(ts):
                break
            if k >= i0:
                yield k, float(ts[k]), np.frombuffer(buf, np.uint8).reshape(h, w)
            if k >= i1:
                break
            k += 1
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


# =============================================================================
# 候補探し
# =============================================================================
def strong_onsets(onsets: list[tuple[float, float]], min_strength: float, merge_s: float
                  ) -> list[tuple[float, float]]:
    """強い打球音だけを残し、近すぎる音は強い方の 1 つにまとめる。"""
    if not onsets:
        return []
    smax = max(s for _, s in onsets)
    keep = sorted([(t, s) for t, s in onsets if s >= min_strength * smax], key=lambda o: -o[1])
    out: list[tuple[float, float]] = []
    for t, s in keep:
        if all(abs(t - u) >= merge_s for u, _ in out):
            out.append((t, s))
    return sorted(out)


def _ray_at_height(camera: Camera, uv: np.ndarray, z: float) -> np.ndarray | None:
    c, d = camera.rays(np.asarray(uv, float)[None])
    c, d = np.asarray(c).reshape(3), np.asarray(d).reshape(3)
    if abs(d[2]) < 1e-6:
        return None
    s = (z - c[2]) / d[2]
    return c + s * d if s > 0 else None


def _approx_positions(p0: np.ndarray, v0: np.ndarray, tau: np.ndarray, k: float, g: float) -> np.ndarray:
    """2 乗抵抗の近似解（向きは一定・重力は別に足す）。打った直後（v ≫ 終端速度）でよく合う。

    速さ v(τ) = v0 / (1 + k v0 τ)、進んだ距離 = ln(1 + k v0 τ) / k。
    """
    sp = float(np.linalg.norm(v0))
    u = v0 / max(sp, 1e-9)
    dist = np.log1p(k * sp * tau) / k
    pos = p0 + dist[:, None] * u
    pos[:, 2] -= 0.5 * g * tau**2
    return pos


MAX_PLAUSIBLE_KMH = 420.0  # これより速い推定は失敗とみなす（男子の公式最速が約 565 km/h、一般は 300 km/h 前後まで）


def _back_extrapolate(t: np.ndarray, uv: np.ndarray, t_ref: float) -> tuple[np.ndarray, float]:
    """画像上の軌道を時刻 t_ref まで戻す。戻すコマ数も返す。

    打った直後は 1 コマごとに移動量が一定の比で変わる（遠ざかると減速して見える）とみなし、
    最初の 3 点の移動量の比で、等比級数として逆向きにたどる（戻すのは 3 コマ分まで）。
    """
    if len(t) < 2:
        return uv[0], 0.0
    dt = float(t[1] - t[0])
    n = float(np.clip((t[0] - t_ref) / max(dt, 1e-9), 0.0, 3.0))
    v1 = uv[1] - uv[0]
    r = 1.0
    if len(t) >= 3 and np.linalg.norm(v1) > 1e-6:
        r = float(np.clip(np.linalg.norm(uv[2] - uv[1]) / np.linalg.norm(v1), 0.3, 1.2))
    back, step, k = np.zeros(2), v1 / r, 0.0
    while k < n - 1e-9:
        frac = min(1.0, n - k)
        back += step * frac
        step = step / r
        k += 1.0
    return uv[0] - back, n


def _quick_plane_fit(camera: Camera, t: np.ndarray, uv: np.ndarray, t_ref: float, plane: FlightPlane,
                     X0: np.ndarray, k: float, g: float) -> tuple[float, np.ndarray] | None:
    """近似解で、面の中の打点・初速を当てはめる。戻り値は（画像上のずれの RMS [px], θ）。"""
    from scipy.optimize import least_squares

    from smash_speed.geometry import intersect_plane

    X = intersect_plane(camera, uv, plane)
    ok = np.isfinite(X).all(axis=1)
    if ok.sum() < 3:
        return None
    s0, z0 = plane.to_plane(X0[None])[0]
    sx = plane.to_plane(X[ok])
    tt = t[ok] - t_ref
    v_guess = (sx[min(2, len(sx) - 1)] - sx[0]) / max(tt[min(2, len(tt) - 1)] - tt[0], 1e-3)
    th0 = np.array([s0, z0, v_guess[0] * 1.5, v_guess[1] * 1.5])
    tau = t - t_ref

    def resid(th: np.ndarray) -> np.ndarray:
        p0 = plane.to_world(np.array([th[0]]), np.array([th[1]]))[0]
        v0 = th[2] * plane.direction + th[3] * np.array([0.0, 0.0, 1.0])
        q = camera.project(_approx_positions(p0, v0, tau, k, g))
        r = (q - uv).ravel()
        prior = 3.0 * (p0 - X0) / np.array([1.0, 1.0, 0.8])  # 打点は最初のシャトルの近く
        return np.concatenate([np.where(np.isfinite(r), r, 1e3), prior])

    sol = least_squares(resid, th0, loss="soft_l1", f_scale=3.0, x_scale=[1.0, 1.0, 30.0, 30.0], max_nfev=200)
    r = sol.fun[:2 * len(t)]
    return float(np.sqrt(np.mean(r**2))), sol.x


def rough_estimate(obs_t: np.ndarray, obs_uv: np.ndarray, t_onset: float, camera: Camera,
                   cfg: dict[str, Any], yaw_step_deg: float = 5.0) -> dict[str, Any]:
    """打つ人・着地点が分からないまま、だいたいの初速を出す（``_rough_estimate``）。

    打った直後のコマは振ったラケットと重なって写ることがあるので、打点が決まらない・合わないときは
    最初の 1 点を除いてもう一度試す。
    """
    r = _rough_estimate(obs_t, obs_uv, t_onset, camera, cfg, yaw_step_deg)
    if len(obs_t) >= 5 and (r["status"] in ("off_court", "fit_failed") or (r.get("rough_rms_px") or 0) > 10):
        r2 = _rough_estimate(obs_t[1:], obs_uv[1:], t_onset, camera, cfg, yaw_step_deg)
        if r2["status"] == "ok" and (r["status"] != "ok" or r2["rough_rms_px"] < 0.5 * r["rough_rms_px"]):
            r2["note"] = ("。".join(x for x in [r2.get("note", ""), "打った直後の 1 コマ目はラケットなどと重なっているとみなして除きました"] if x))
            return r2
    return r


def _rough_estimate(obs_t: np.ndarray, obs_uv: np.ndarray, t_onset: float, camera: Camera,
                    cfg: dict[str, Any], yaw_step_deg: float = 5.0) -> dict[str, Any]:
    """打つ人・着地点が分からないまま、だいたいの初速を出す。

    最初に写ったシャトルを打点の高さ（``events.hit_height_m``）にあるとみなして打点の位置 X0 を決め、
    X0 を通る縦の面の向き（相手コートの方向から ±30°）を、2 乗抵抗の近似解で速く格子探索する。
    画像上のずれが最小の向きで、最後に運動方程式をきちんと当てはめる。
    打球音の伝搬遅れは X0 までの距離で補正する。
    """
    ev = cfg.get("events", {})
    c = cfg["court"]
    c_sound = ev.get("speed_of_sound_mps", 343.0)
    dt = float(np.median(np.diff(obs_t))) if len(obs_t) >= 2 else 1 / 30
    # 打点：画像上の軌道を打った瞬間まで戻した位置を、打点の高さにあるとみなす。
    # 最初に写ったシャトルは、打ってから数コマ後でもうかなり先にいることがあるので、そのまま使うと
    # 打点が前に寄りすぎ、速さが何倍にもなる。音の伝搬遅れは距離に依存するので 2 回繰り返す。
    # （高さを決め打ちするので、カメラが横にあって視線が浅いと奥行きがずれやすい。その場合は確認で直す）
    t_impact = t_onset - 0.015
    X0 = None
    for _ in range(2):
        u_imp, _n = _back_extrapolate(obs_t, obs_uv, t_impact)
        X0 = _ray_at_height(camera, u_imp, ev.get("hit_height_m", 2.5))
        if X0 is None:
            return {"status": "off_court", "note": "打点の位置を決められません（軌道がカメラより低い所から始まっています）"}
        t_impact = t_onset - float(np.linalg.norm(camera.center - X0)) / c_sound
    if abs(X0[0]) > c["half_length_m"] + 1.5 or abs(X0[1]) > c["half_width_doubles_m"] + 1.5:
        return {"status": "off_court", "note": f"打点が合わせたコートの外（x {X0[0]:.1f}, y {X0[1]:.1f} m）。別のコートの音の可能性"}
    use = obs_t > t_impact
    if use.sum() < 4:
        return {"status": "no_track", "note": "打った後のシャトルが 4 コマ未満"}
    t, uv = obs_t[use], obs_uv[use]
    phys = PhysicsParams.from_config(cfg)
    toward = -np.sign(X0[0]) if abs(X0[0]) > 0.3 else 1.0  # 相手コートの方向
    early = t - t_impact <= 0.35  # 近似解が合うのは序盤だけ

    def plane_of(yaw_deg: float) -> FlightPlane:
        y = np.radians(yaw_deg)
        return FlightPlane(np.array([X0[0], X0[1], 0.0]), np.array([toward * np.cos(y), np.sin(y), 0.0]))

    scores = {}
    for yaw in list(np.arange(-30, 30 + 1e-9, yaw_step_deg)):
        r = _quick_plane_fit(camera, t[early], uv[early], t_impact, plane_of(yaw), X0, phys.k, phys.g)
        if r is not None:
            scores[float(yaw)] = r[0]
    if not scores:
        return {"status": "fit_failed", "note": "軌道を当てはめられませんでした", "t_impact": t_impact}
    plane = plane_of(min(scores, key=scores.get))
    obs = Observations(camera, t, uv)
    phys_c = PhysicsParams(phys.g, phys.terminal_velocity, phys.method, rtol=1e-7, atol=1e-9)
    opts = FitOptions(ci_method="none", loss="huber", loss_scale_px=cfg["physics"].get("loss_scale_px", 2.0))
    try:
        f = fit_trajectory(obs, t_impact, "plane", plane, phys_c, opts)
    except (ValueError, RuntimeError):
        return {"status": "fit_failed", "note": "軌道を当てはめられませんでした", "t_impact": t_impact}
    off = ev.get("contact_offset_m", 0.3)
    hit = plane.origin[:2] - off * plane.direction[:2]
    traj = simulate_until_ground(f.p0, f.v0, f.t_ref, f.phys(phys_c), n_dense=200)
    land = traj["pos"][-1, :2]
    kmh = f.speed0 * 3.6
    if kmh > 1.5 * MAX_PLAUSIBLE_KMH:  # シャトルではない物の動きか、推定の大失敗
        return {"status": "fit_failed", "t_impact": t_impact,
                "note": f"シャトルの軌道ではなさそうです（{kmh:.0f} km/h になってしまう）"}
    gap = (float(t[0]) - t_impact) / dt  # 打ってから最初に写るまでのコマ数
    doubt = []
    if kmh > MAX_PLAUSIBLE_KMH:
        doubt.append(f"{kmh:.0f} km/h はありえない速さです（推定の失敗）")
    if f.rms_px > 10.0:
        doubt.append(f"画像上のずれが大きい（{f.rms_px:.0f} px）")
    if gap > 3.5:
        doubt.append(f"打ってから {gap:.0f} コマ後まで写っていません")
    return {"status": "ok", "t_impact": t_impact, "rough_kmh": float(kmh), "rough_rms_px": float(f.rms_px),
            "n_frames": int(len(t)), "doubtful": bool(doubt), "note": "。".join(doubt),
            "smash_like": bool(f.v0[2] < 0 and kmh >= 120.0 and not doubt),
            "hitter": {"x": round(float(hit[0]), 3), "y": round(float(hit[1]), 3)},
            "landing": {"x": round(float(land[0]), 3), "y": round(float(land[1]), 3)}}


def _thumbnail(frame: np.ndarray, uv: np.ndarray, scale: float, path: Path) -> None:
    img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    pts = (uv * scale).round().astype(int)
    for k, p in enumerate(pts):
        cv2.circle(img, tuple(p), 6 if k == 0 else 4, (0, 165, 255) if k == 0 else (0, 220, 0), 2, cv2.LINE_AA)
    if len(pts) >= 2:
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, (0, 220, 0), 1, cv2.LINE_AA)
    s = 640 / img.shape[1]
    cv2.imwrite(str(path), cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 82])


def find_candidates(o: BatchOptions, cfg: dict[str, Any],
                    progress: Callable[[str], None] = print) -> list[Candidate]:
    """動画全体からスマッシュの候補を探し、``candidates.json`` に保存する。"""
    from smash_speed.pipeline import _camera_for_video, _select_tracks

    out = Path(o.out_dir).resolve()  # GUI のサーバーは作業フォルダが違うので、保存するパスは絶対パスにする
    (out / "thumbs").mkdir(parents=True, exist_ok=True)
    (out / "det").mkdir(parents=True, exist_ok=True)
    meta = probe(o.video, o.capture_fps, cfg)
    camera, _ = load_calibration(o.calib)
    camera = _camera_for_video(camera, meta)
    if not meta.has_audio and o.audio is None:
        raise AnalysisError("この動画には音声がありません。打球音から候補を探すには音声が必要です。",
                            ["音声つきで撮った動画を使うか、手順 1 で 1 本ずつ区間を選んで解析してください。"])

    # --- 打球音
    if o.audio is not None:
        x, rate = load_wav(o.audio)
    else:
        wav = out / "audio.wav"
        extract_audio(o.video, wav, cfg=cfg)
        x, rate = load_wav(wav)
    ev = cfg.get("events", {})
    ons = [(t * meta.time_scale, s) for t, s in events.audio_onsets(
        x, rate, ev.get("audio_highpass_hz", 2000.0), k_mad=ev.get("audio_k_mad", 10.0))]
    t_lo = o.start_s if o.start_s is not None else 0.0
    t_hi = o.end_s if o.end_s is not None else float(meta.timestamps[-1])
    ons = [(t, s) for t, s in ons if t_lo <= t <= t_hi]
    strong = strong_onsets(ons, o.min_strength, o.merge_s)
    progress(f"[進行] 打球音の候補 {len(strong)} 個（音の候補 {len(ons)} 個のうち強いもの）")
    t_end = float(meta.timestamps[-1])
    cands = [Candidate(id=k + 1, t_onset=round(float(t), 4), strength=round(float(s), 1),
                       start_s=round(max(0.0, t - 0.3), 3), end_s=round(min(t_end, t + 1.0), 3))
             for k, (t, s) in enumerate(strong)]
    if not cands:
        raise AnalysisError("強い打球音が見つかりませんでした。",
                            ["音声が小さい・雑音が多い動画では見つからないことがあります。手順 1 で区間を選んで解析してください。"])

    # --- 動画を 1 回だけ読み、音の前後の区間ごとに検出する
    p = BaselineParams.from_config(cfg)
    if o.roi:
        p.roi = list(o.roi)
    scale = min(1.0, p.process_width / meta.width)
    n_cand = cfg.get("detection", {}).get("n_track_candidates", 40)
    windows = {c.id: (c.t_onset - o.pre_s, c.t_onset + o.post_s) for c in cands}
    by_id = {c.id: c for c in cands}
    bufs: dict[int, list[tuple[int, float, np.ndarray]]] = {k: [] for k in windows}
    done = 0

    def finish(cid: int) -> None:
        nonlocal done
        c = by_id[cid]
        frames = bufs.pop(cid)
        done += 1
        if len(frames) < 5:
            c.status, c.note = "no_track", "区間のコマが足りません"
            return
        det = detect_frames(iter(frames), scale, p, n_tracks=n_cand, exclusive=False)
        # この打球音の直後（1 コマ + 20 ms 前〜0.2 s 後）に始まる軌道だけから選ぶ（飛んできたシャトルなど、
        # 打つ前から続く長い軌道を選ばないように。最初のコマはラケットの残像のこともあるので少し前まで含める）
        vis = det[det.visible == 1]
        lead = 1.0 / meta.capture_fps + 0.02
        starts = vis.groupby("track_id").timestamp.min()
        ok_ids = starts[(starts >= c.t_onset - lead) & (starts <= c.t_onset + 0.2)].index
        # 打つ人の位置が分からないので、打球音の直後に始まる上位 3 本の軌道それぞれで大まかに推定し、
        # 打点がコートの中・ありえる速さ・画像上のずれが小さいものを選ぶ
        pool = vis[vis.track_id.isin(ok_ids)]
        tried = []
        for _ in range(3):
            out_tr, _ = _select_tracks(pool, [(c.t_onset, c.strength)], None, float(meta.width), lead_s=lead)
            if len(out_tr) < 4 or "track_id" not in out_tr or out_tr.track_id.nunique() != 1:
                break
            t0 = float(out_tr.timestamp.min())
            if c.t_onset - lead <= t0 <= c.t_onset + 0.2:
                r = rough_estimate(out_tr.timestamp.to_numpy(), out_tr[["x_px", "y_px"]].to_numpy(), c.t_onset, camera, cfg)
                tried.append((r, out_tr))
            pool = pool[pool.track_id != out_tr.track_id.iloc[0]]

        def rank(item: tuple[dict[str, Any], pd.DataFrame]) -> tuple[int, float]:
            r = item[0]
            return (0 if r["status"] == "ok" and not r.get("doubtful") else 1 if r["status"] == "ok" else 2,
                    r.get("rough_rms_px") or 1e9)


        if not tried:
            c.status, c.note = "no_track", "打球音の直後に始まるシャトルの軌道が見つかりません"
        else:
            r, out_tr = min(tried, key=rank)
            for k, v in r.items():
                setattr(c, k, v)
            # 詳しい解析では、サムネイルに描いたのと同じ軌道（ここで選んだ軌道）だけを使う
            # （打つ人の位置が変わると、検出の全候補から別の軌道が選ばれてしまうことがあるため）
            rel = f"det/{c.label}.csv"
            out_tr.to_csv(out / rel, index=False)
            det.to_csv(out / f"det/{c.label}_all.csv", index=False)
            c.detections = rel
            # サムネイル：打った 2 コマ後の画像に、使ったシャトルの位置を描く
            idx = {i: g for i, _, g in frames}
            f_show = int(out_tr.frame.iloc[min(2, len(out_tr) - 1)])
            if f_show in idx:
                rel_t = f"thumbs/{c.label}.jpg"
                _thumbnail(idx[f_show], out_tr[["x_px", "y_px"]].to_numpy(), scale, out / rel_t)
                c.thumb = rel_t
        msg = f"{c.rough_kmh:.0f} km/h" if c.rough_kmh is not None else c.note
        progress(f"[進行] {done}/{len(cands)}  {c.t_onset:7.2f} s  {msg}")

    t_first = min(a for a, _ in windows.values())
    t_last = max(b for _, b in windows.values())
    for i, t, g in stream_gray(meta, int(round(meta.width * scale)), cfg, max(0.0, t_first - 0.1), t_last + 0.1):
        for cid, (a, b) in list(windows.items()):
            if a <= t <= b:
                bufs[cid].append((i, t, g))
            elif t > b:
                windows.pop(cid)
                finish(cid)
    for cid in list(windows):
        finish(cid)

    # 速い順に上位 n_select 個（スマッシュらしいもの）を最初から選んでおく
    ranked = sorted([c for c in cands if c.status == "ok" and c.smash_like], key=lambda c: -(c.rough_kmh or 0))
    for c in ranked[:o.n_select]:
        c.selected = True
    save_candidates(out, cands, {"video": str(Path(o.video).resolve()), "calib": str(Path(o.calib).resolve()),
                                 "capture_fps": o.capture_fps,
                                 "n_onsets": len(ons), "fps": meta.capture_fps, "roi": o.roi,
                                 "audio": str(Path(o.audio).resolve()) if o.audio else None})
    n_ok = sum(c.status == "ok" for c in cands)
    progress(f"[進行] 完了：候補 {len(cands)} 個のうち、シャトルの軌道が見つかったもの {n_ok} 個"
             f"（スマッシュらしいもの {len(ranked)} 個）")
    return cands


# =============================================================================
# 保存・読み込み・詳しい解析
# =============================================================================
def _num(v: Any) -> float | None:
    """JSON に書けない NaN・無限大は None にする（ブラウザの JSON.parse が読めないため）。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def save_candidates(out_dir: Path, cands: list[Candidate], info: dict[str, Any] | None = None) -> None:
    path = Path(out_dir) / CANDIDATES_FILE
    old = json.loads(path.read_text(encoding="utf-8")).get("info", {}) if path.exists() else {}
    rows = []
    for c in cands:
        d = asdict(c)
        for k in ("rough_kmh", "rough_rms_px", "t_impact"):
            d[k] = _num(d[k]) if d[k] is not None else None
        rows.append(d)
    data = {"info": {**old, **(info or {})}, "candidates": rows}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def load_candidates(out_dir: Path) -> tuple[list[Candidate], dict[str, Any]]:
    data = json.loads((Path(out_dir) / CANDIDATES_FILE).read_text(encoding="utf-8"))
    return [Candidate(**c) for c in data["candidates"]], data.get("info", {})


def analyze_selected(out_dir: Path, cfg: dict[str, Any], ids: list[int] | None = None,
                     overlay: bool = True, progress: Callable[[str], None] = print) -> list[Candidate]:
    """選んだ候補を、確認した打つ人・着地点で詳しく解析し、結果を ``candidates.json`` と ``ranking.csv`` に書く。"""
    from smash_speed.pipeline import AnalyzeOptions, analyze

    out_dir = Path(out_dir).resolve()
    cands, info = load_candidates(out_dir)
    todo = [c for c in cands if (c.id in ids if ids else c.selected) and c.status == "ok"]
    if not todo:
        raise AnalysisError("詳しく解析する候補が選ばれていません。", ["候補の一覧で「詳しく解析する」に印をつけてください。"])
    for k, c in enumerate(todo, 1):
        progress(f"[進行] {k}/{len(todo)}  {c.t_onset:.2f} s を解析しています")
        if not (c.hitter and c.landing):
            c.result = {"error": "打つ人・着地点がありません"}
            continue
        o = AnalyzeOptions(video=Path(info["video"]), calib=Path(info["calib"]), out_dir=out_dir / c.label,
                           capture_fps=info.get("capture_fps"), start_s=c.start_s, end_s=c.end_s,
                           detections=out_dir / c.detections if c.detections else None,
                           audio_path=Path(info["audio"]) if info.get("audio") else None,
                           hitter_xy=(c.hitter["x"], c.hitter["y"]), landing_xy=(c.landing["x"], c.landing["y"]),
                           overlay=overlay, force=True, name=c.label)
        try:
            row = analyze(o, cfg)
            c.result = {"kmh": _num(row["speed_initial_kmh"]), "ci_low_kmh": _num(row["speed_initial_ci_low_kmh"]),
                        "ci_high_kmh": _num(row["speed_initial_ci_high_kmh"]), "rms_px": _num(row["rms_px"]),
                        "n_frames": int(row["n_frames_used"]), "impact_method": row["impact_method"],
                        "confirmed": c.confirmed, "dir": str(out_dir / c.label)}
            progress(f"[進行] {c.t_onset:.2f} s → {row['speed_initial_kmh']:.1f} km/h")
        except AnalysisError as e:
            c.result = {"error": e.message, "hints": e.hints}
            progress(f"[進行] {c.t_onset:.2f} s → 解析できませんでした（{e.message}）")
        save_candidates(out_dir, cands)
    write_ranking(out_dir, cands)
    return cands


def write_ranking(out_dir: Path, cands: list[Candidate]) -> pd.DataFrame:
    rows = [{"rank": 0, "id": c.id, "t_onset_s": c.t_onset, "speed_initial_kmh": c.result["kmh"],
             "ci_low_kmh": c.result.get("ci_low_kmh"), "ci_high_kmh": c.result.get("ci_high_kmh"),
             "rms_px": c.result.get("rms_px"), "n_frames": c.result.get("n_frames"),
             "rough_kmh": c.rough_kmh, "confirmed": c.confirmed, "dir": c.result.get("dir")}
            for c in cands if c.result and "kmh" in c.result]
    df = pd.DataFrame(rows, columns=["rank", "id", "t_onset_s", "speed_initial_kmh", "ci_low_kmh", "ci_high_kmh",
                                     "rms_px", "n_frames", "rough_kmh", "confirmed", "dir"])
    df["plausible"] = df["speed_initial_kmh"] <= MAX_PLAUSIBLE_KMH
    df = df.sort_values(["plausible", "speed_initial_kmh"], ascending=[False, False]).reset_index(drop=True)
    df["rank"] = np.where(df["plausible"], np.arange(1, len(df) + 1), 0)  # ありえない速さは順位なし（0）
    df.to_csv(Path(out_dir) / "ranking.csv", index=False, encoding="utf-8-sig")
    return df
