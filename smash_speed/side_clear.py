"""側方から撮ったクリアの打球の結果（ROADMAP E6）：初速・打ち出し角・最高点・落ちた所の深さ。

フォームの撮影（``docs/pose_features.md``：ラケットを持つ腕の側から、打つ向きに直交、距離 6〜8 m、高さ 1.2〜1.5 m、水平）では
コートの線がほとんど写らないので、コートの合わせ込み（キャリブレーション）の代わりに、打つ人の体から画像の縮尺を決める。

- 縮尺 [px/m] = 姿勢から出した身長 [px] ÷ 本当の身長 [m]（人が入れる）
- 床の高さ = 打点の前後の足首の画像の高さ（足首は床から身長の 0.039 倍の高さ）
- カメラは水平で、シャトルは打つ人を通り、カメラの向きに直交する鉛直な面を飛ぶとみなす

この面に物理モデル（``physics.fit_trajectory`` の plane モード）を当てはめ、打点の時刻の速度から初速・打ち出し角を、
当てはめた軌道を床まで伸ばして最高点・落ちた所を出す。クリアは画面の上・右へ出ていくので、最高点と落ちた所は
多くの場合、画面の外を物理モデルで伸ばした値になる。

カメラの置き方・身長の誤差・打つ向きのずれ（クロスに打つと面が斜めになる）でどこまで正しく出るかを、
合成のクリア（``synth_rally.make_typed_shot``）で ``side_benchmark`` で測る（``python -m smash_speed.side_clear --bench``）。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from smash_speed.camera import Camera
from smash_speed.geometry import FlightPlane
from smash_speed.physics import FitOptions, Observations, PhysicsParams, fit_trajectory, simulate_until_ground
from smash_speed.synth_pose import side_camera

NET_TO_BACK_M = 6.70     # ネット〜バックバウンダリーライン [m]


@dataclass
class SideSetup:
    """側方のカメラの縮尺と位置（画像の座標 → 打つ人を通る面の上の [m]）。

    Attributes:
        px_per_m: 打つ人の面での縮尺 [px/m]
        ground_v_px: 打つ人の面での床の画像の高さ [px]
        hitter_u_px: 打つ人（腰の中点）の画像の横の位置 [px]
        width, height: 画像の大きさ [px]
        f_px: 焦点距離 [px]（面の外の点の写り方にだけ効く。分からなければ横の画角 69° から）
        facing: 打つ向きが画像の右なら +1、左なら −1
    """

    px_per_m: float
    ground_v_px: float
    hitter_u_px: float
    width: int = 1920
    height: int = 1080
    f_px: float | None = None
    facing: float = 1.0

    def camera(self) -> Camera:
        """打つ人の足元を原点、打つ向きを +x、カメラの側を −y、上を +z とするカメラ。"""
        f = self.f_px or (self.width / 2) / np.tan(np.radians(69.0) / 2)
        d = f / self.px_per_m                              # 打つ人の面までの距離
        cy = self.height / 2
        h = (self.ground_v_px - cy) / self.px_per_m        # 水平なカメラの高さ（床の写る位置から）
        cx = self.width / 2
        x_cam = (cx - self.to_right(np.array([self.hitter_u_px, 0.0]))[0]) / self.px_per_m
        cam = side_camera(distance_m=d, height_m=h, target_x_m=x_cam, width=self.width, height=self.height)
        cam.f = f
        cam.cx, cam.cy = cx, cy
        return cam

    def to_right(self, uv: np.ndarray) -> np.ndarray:
        """打つ向きが画像の左なら、画像の座標を左右に反転する（``camera`` はいつも右へ打つ形）。"""
        uv = np.array(uv, float)
        if self.facing < 0:
            uv[..., 0] = self.width - uv[..., 0]
        return uv


def setup_from_pose(norm, stature_m: float, width: int, height: int, f_px: float | None = None) -> SideSetup:
    """正規化した姿勢（``pose_prep.NormalizedPose``）と本当の身長 [m] から縮尺を決める。

    身長 [px] = 身長（胴体の何倍か、``pose_features.body_height``）× 胴体の長さ [px]。床は打点の足首の高さから。
    """
    from smash_speed.pose_features import ANKLE_RATIO, body_height

    H = body_height(norm)
    if not np.isfinite(H):
        raise ValueError("脚が見えないため、身長から縮尺を決められません")
    px_per_m = H * norm.torso_px / stature_m
    i = int(np.argmin(np.abs(norm.t)))
    raw = norm.meta.get("raw_hip_px")
    if raw is None:
        raise ValueError("元の画像の腰の位置（meta の raw_hip_px）がありません")
    hip_u, hip_v = raw
    ankle_y = np.nanmean([norm.joint("left_ankle")[i, 1], norm.joint("right_ankle")[i, 1]])  # 胴体の長さ、上が +
    ground_v = hip_v - (ankle_y - ANKLE_RATIO * H) * norm.torso_px
    return SideSetup(px_per_m=float(px_per_m), ground_v_px=float(ground_v), hitter_u_px=float(hip_u), width=width,
                     height=height, f_px=f_px, facing=norm.facing)


@dataclass
class ClearResult:
    """クリアの打球の結果。位置は打つ人の足元からの [m]。"""

    speed_kmh: float
    elevation_deg: float
    hit_height_m: float
    apex_height_m: float
    apex_from_hitter_m: float
    land_from_hitter_m: float
    land_depth_m: float | None     # 落ちた所の、相手側のネットからの深さ（打つ人のネットからの距離が分かれば）
    flight_time_s: float
    n_obs: int
    rms_px: float
    in_frame_until_s: float        # 画面の中で見えていた最後の時刻（打点から）[s]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_clear(t: np.ndarray, uv: np.ndarray, t_hit: float, camera: Camera, phys: PhysicsParams | None = None,
              hitter_from_net_m: float | None = None, hit_height_m: float = 2.5, refine_plane: bool = True
              ) -> ClearResult | None:
    """打った後のシャトルの画像の位置 (N, 2) [px]（時刻 t [s]）から、クリアの打球の結果を出す。

    飛行の面は打つ人を通り打つ向き（+x）に伸びる鉛直な面。``refine_plane`` なら面の向き・位置も少し動かして当てはめる。
    """
    phys = phys or PhysicsParams()
    keep = (t >= t_hit) & np.isfinite(uv).all(axis=1)
    if keep.sum() < 5:
        return None
    obs = Observations(camera, t[keep], uv[keep])
    plane = FlightPlane(origin=np.zeros(3), direction=np.array([1.0, 0.0, 0.0]))
    opts = FitOptions(ci_method="none", refine_plane=refine_plane, plane_sigma_deg=10.0, plane_sigma_m=0.5,
                      p0_prior_m=np.array([0.0, 0.0, hit_height_m]), p0_prior_sigma_m=(1.0, 0.8))
    try:
        fit = fit_trajectory(obs, t_hit, mode="plane", plane=plane, phys=phys, opts=opts)
    except (ValueError, RuntimeError):
        return None
    tr = simulate_until_ground(fit.p0, fit.v0, fit.t_ref, fit.phys(phys), t_max=8.0, n_dense=2000)
    pos = tr["pos"]
    k = int(np.argmax(pos[:, 2]))
    plane_dir = fit.plane.direction if fit.plane is not None else plane.direction
    along = pos[:, :2] @ plane_dir[:2]   # 打つ人の足元からの、打つ向きの距離
    land = float(along[-1])
    v = fit.v0
    return ClearResult(
        speed_kmh=float(np.linalg.norm(v) * 3.6),
        elevation_deg=float(np.degrees(np.arctan2(v[2], np.linalg.norm(v[:2])))),
        hit_height_m=float(fit.p0[2]), apex_height_m=float(pos[k, 2]), apex_from_hitter_m=float(along[k]),
        land_from_hitter_m=land,
        land_depth_m=None if hitter_from_net_m is None else land - hitter_from_net_m,
        flight_time_s=float(tr["t_land"] - t_hit), n_obs=int(keep.sum()), rms_px=float(fit.rms_px),
        in_frame_until_s=float(t[keep][-1] - t_hit))


# --- 動画のショットごと ------------------------------------------------------------------

BALL_FILE = "ball.csv"
BALL_COLUMNS = ["shot_id", "ball_speed_kmh", "ball_elevation_deg", "ball_hit_height_m", "apex_height_m",
                "apex_from_hitter_m", "land_from_hitter_m", "land_depth_m", "ball_flight_s", "ball_n_obs", "ball_rms_px",
                "ball_in_frame_s", "ball_note"]


def _detect_video(video: str):
    """動画の区間のシャトルの検出（``detection.baseline``）を返す関数。"""
    from smash_speed.detection.baseline import detect
    from smash_speed.video_io import probe

    meta = probe(video)
    return lambda t0, t1: detect(meta, t0, t1)


def analyze_balls(out_dir: str | Path, stature_m: float, hitter_from_net_m: float | None = None,
                  video: str | None = None, detect_fn=None, phys: PhysicsParams | None = None,
                  after_s: float = 0.7) -> list[dict[str, Any]]:
    """フォームの解析の結果（``form/``）のショットごとに、打った後のシャトルを検出してクリアの結果を出し、
    ``form/ball.csv`` に書き出す。

    Args:
        stature_m: 打つ人の身長 [m]（画像の縮尺を決める）
        hitter_from_net_m: 打つ人のネットからの距離 [m]（分かれば落ちた所の深さを出す）
        detect_fn: (始め, 終わり) [s] → 検出の表（列 ``timestamp, x_px, y_px, visible``）。差し替え用
    """
    from smash_speed.form import FEATURES_FILE, FORM_DIR, read_rows, write_rows
    from smash_speed.pose_prep import NormalizedPose

    d = Path(out_dir)
    d = d / FORM_DIR if (d / FORM_DIR).is_dir() else d
    rows = read_rows(d / FEATURES_FILE)
    if detect_fn is None:
        info = json.loads((d / "form.json").read_text(encoding="utf-8"))
        detect_fn = _detect_video(video or info["video"])
    out = []
    for r in rows:
        sid, t_imp = r["shot_id"], float(r["t_impact_s"])
        row: dict[str, Any] = {"shot_id": sid}
        try:
            n = NormalizedPose.load(d / "poses" / f"{sid}.npz")
            w, h = n.meta.get("image_size") or (1920, 1080)
            setup = setup_from_pose(n, stature_m, int(w), int(h))
        except (OSError, ValueError) as e:
            out.append({**row, "ball_note": f"縮尺を決められません: {e}"})
            continue
        det = detect_fn(t_imp - 0.02, t_imp + after_s)
        det = det[det["visible"] > 0]
        uv = setup.to_right(det[["x_px", "y_px"]].to_numpy(float))
        res = fit_clear(det["timestamp"].to_numpy(float), uv, t_imp, setup.camera(), phys, hitter_from_net_m)
        if res is None:
            out.append({**row, "ball_note": f"シャトルが {len(det)} コマしか見つからず、当てはめられません"})
            continue
        out.append({**row, "ball_speed_kmh": res.speed_kmh, "ball_elevation_deg": res.elevation_deg,
                    "ball_hit_height_m": res.hit_height_m, "apex_height_m": res.apex_height_m,
                    "apex_from_hitter_m": res.apex_from_hitter_m, "land_from_hitter_m": res.land_from_hitter_m,
                    "land_depth_m": res.land_depth_m, "ball_flight_s": res.flight_time_s, "ball_n_obs": res.n_obs,
                    "ball_rms_px": res.rms_px, "ball_in_frame_s": res.in_frame_until_s, "ball_note": ""})
    write_rows(d / BALL_FILE, out, BALL_COLUMNS)
    return out


# --- 合成データでの精度 ------------------------------------------------------------------

@dataclass
class SideCase:
    """合成の 1 本の条件。"""

    distance_m: float = 7.0          # 打つ人からカメラまで
    cam_height_m: float = 1.3
    fps: float = 60.0
    noise_px: float = 1.5
    dropout: float = 0.1
    stature_err: float = 0.0         # 身長から決めた縮尺の誤差（割合）
    ground_err_px: float = 0.0       # 床の高さの誤差 [px]
    contact_offset_m: float = 0.25   # 打点が打つ人の体の面よりカメラ側にある距離 [m]
    cross_deg: float | None = None   # 打つ向きのずれ（None なら狙った着地点まかせ）
    t_hit_err_s: float = 0.0
    refine_plane: bool = True        # 面の向き・位置も当てはめる（クロスに打つと面が斜めになるため）
    width: int = 1920                # 画像の大きさ（縦向きに撮るなら 1080 × 1920）
    height: int = 1080
    hfov_deg: float = 69.0           # 横の画角（縦向きなら同じレンズで約 42.6°）
    lead_m: float = 2.0              # 画面の真ん中を、打つ人よりこれだけ打つ向きの前にする [m]


def _truth(seg, phys: PhysicsParams, hitter_xy: np.ndarray, direction: np.ndarray) -> dict[str, float]:
    tr = simulate_until_ground(seg.p0, seg.v0, seg.t_hit, phys, t_max=8.0, n_dense=4000)
    pos = tr["pos"]
    k = int(np.argmax(pos[:, 2]))
    along = (pos[:, :2] - hitter_xy) @ direction
    return {"speed_kmh": seg.speed0 * 3.6, "elevation_deg": seg.elevation_deg, "apex_height_m": float(pos[k, 2]),
            "land_from_hitter_m": float(along[-1]), "land_depth_m": float(pos[-1, 0])}


def side_benchmark(n: int = 40, case: SideCase | None = None, seed: int = 0, phys: PhysicsParams | None = None
                   ) -> dict[str, Any]:
    """合成のクリアを側方のカメラで観測し、``fit_clear`` の誤差を測る。"""
    from smash_speed.synth_rally import make_typed_shot

    case = case or SideCase()
    phys = phys or PhysicsParams()
    rng = np.random.default_rng(seed)
    errs: dict[str, list[float]] = {k: [] for k in ("speed_kmh", "elevation_deg", "apex_height_m",
                                                    "land_from_hitter_m", "land_depth_m")}
    rel_speed, frame_s, fails = [], [], 0
    for _ in range(n):
        seg = make_typed_shot("clear", rng, phys, side="h")
        if case.cross_deg is not None:  # 打つ向きを決めた角度だけずらす（速さ・打ち出し角はそのまま）
            r = np.radians(case.cross_deg * rng.choice([-1.0, 1.0]))
            vh = np.linalg.norm(seg.v0[:2])
            seg.v0 = np.array([vh * np.cos(r), vh * np.sin(r), seg.v0[2]])
        # 打つ人は打点から体の面の側（カメラと反対）へ contact_offset_m、打つ向きは +x（コートの長い向き）
        hitter = seg.p0[:2] + np.array([0.0, case.contact_offset_m])
        u_dir = np.array([1.0, 0.0])
        cam = side_camera(distance_m=case.distance_m, height_m=case.cam_height_m, target_x_m=hitter[0] + case.lead_m,
                          width=case.width, height=case.height, hfov_deg=case.hfov_deg)
        cam.tvec = cam.tvec - cam.R @ np.array([0.0, hitter[1], 0.0])  # 打つ人の真横に置く
        tr = simulate_until_ground(seg.p0, seg.v0, seg.t_hit, phys, t_max=8.0)
        times = np.arange(seg.t_hit + rng.uniform(0, 1 / case.fps), tr["t_land"], 1 / case.fps)
        from smash_speed.physics import simulate
        pos, _ = simulate(seg.p0, seg.v0, seg.t_hit, times, phys)
        uv = cam.project(pos) + rng.normal(0, case.noise_px, (len(times), 2))
        ok = ((uv[:, 0] >= 0) & (uv[:, 0] < cam.width) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.height)
              & (rng.uniform(size=len(times)) >= case.dropout))
        # 解析で使うカメラ：打つ人の体から決めた縮尺（誤差つき）と、打つ人の足元を原点にした座標
        true_scale = cam.f / case.distance_m
        hit_u = float(cam.project(np.r_[hitter, 1.0])[0, 0])
        ground_v = float(cam.project(np.r_[hitter, 0.0])[0, 1])
        setup = SideSetup(px_per_m=true_scale * (1 + case.stature_err), ground_v_px=ground_v + case.ground_err_px,
                          hitter_u_px=hit_u, width=cam.width, height=cam.height, f_px=cam.f)
        t_hit = seg.t_hit + case.t_hit_err_s
        res = fit_clear(times[ok], uv[ok], t_hit, setup.camera(), phys, hitter_from_net_m=-float(hitter[0]),
                        refine_plane=case.refine_plane)
        if res is None or not np.isfinite(res.speed_kmh):
            fails += 1
            continue
        truth = _truth(seg, phys, hitter, u_dir)
        for k in errs:
            est = getattr(res, k)
            errs[k].append(np.nan if est is None else est - truth[k])
        rel_speed.append(res.speed_kmh / truth["speed_kmh"] - 1)
        frame_s.append(res.in_frame_until_s)

    def summ(x: list[float]) -> dict[str, float]:
        a = np.array(x, float)
        a = a[np.isfinite(a)]
        return {"bias": float(np.mean(a)), "mae": float(np.mean(np.abs(a))), "p90": float(np.percentile(np.abs(a), 90))} \
            if len(a) else {"bias": np.nan, "mae": np.nan, "p90": np.nan}

    out = {k: summ(v) for k, v in errs.items()}
    out["speed_mape"] = float(np.mean(np.abs(rel_speed)) * 100) if rel_speed else np.nan
    out["in_frame_s"] = float(np.median(frame_s)) if frame_s else np.nan
    out["n"] = n - fails
    out["fails"] = fails
    out["case"] = asdict(case)
    return out


BENCH_CASES: dict[str, SideCase] = {
    "基準（7 m・高さ 1.3 m・60fps・打点が体より 0.25 m 手前）": SideCase(),
    "打点が体の面の上（0 m）": SideCase(contact_offset_m=0.0),
    "6 m": SideCase(distance_m=6.0),
    "8 m・高さ 1.5 m": SideCase(distance_m=8.0, cam_height_m=1.5),
    "120fps": SideCase(fps=120.0),
    "身長の縮尺 +2%": SideCase(stature_err=0.02),
    "床の高さ +10 px": SideCase(ground_err_px=10.0),
    "面の向きを当てはめない（打つ向きは狙った着地点まかせ）": SideCase(refine_plane=False),
    "まっすぐ打つ（クロス 0°）": SideCase(cross_deg=0.0),
    "クロスに 10°": SideCase(cross_deg=10.0),
    "クロスに 10°（面の向きを当てはめない）": SideCase(cross_deg=10.0, refine_plane=False),
    "打点の時刻 +5 ms": SideCase(t_hit_err_s=0.005),
    "縦向きに撮る（1080×1920）": SideCase(width=1080, height=1920, hfov_deg=42.6, lead_m=1.0),
    "縦向き・120fps": SideCase(width=1080, height=1920, hfov_deg=42.6, lead_m=1.0, fps=120.0),
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="側方から撮ったクリアの打球の結果の、合成データでの精度")
    p.add_argument("--bench", action="store_true")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if not args.bench:
        p.print_help()
        return 0
    print("条件 | 初速 MAPE % | 初速 km/h 平均の誤差 | 打ち出し角 ° | 最高点 m | 落ちた所 m | 画面の中 s")
    for name, case in BENCH_CASES.items():
        r = side_benchmark(args.n, case, args.seed)
        print(f"{name} | {r['speed_mape']:.1f} | {r['speed_kmh']['bias']:+.1f} | {r['elevation_deg']['mae']:.1f} | "
              f"{r['apex_height_m']['mae']:.2f} | {r['land_depth_m']['mae']:.2f} | {r['in_frame_s']:.2f}"
              + (f"（失敗 {r['fails']}）" if r["fails"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
