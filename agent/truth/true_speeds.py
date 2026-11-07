"""コマを見て決めたシャトルの位置から、実動画のスマッシュの「正しい初速」を出す。

``smash_points.json``（コマを 1 枚ずつ見て確かめたシャトルの画像上の位置・打った時刻・打つ人と受けた所の床の点）から、
空気抵抗つきの運動方程式を 3 次元で当てはめ（平面に縛らない 6 個のパラメータ）、打った時刻の速さを初速とする。
結果は数字だけを ``agent/checks/real_videos_truth.json`` に書く（品質チェックの実動画の基準の隣）。

当てはめに使うもの：

- シャトルの画像上の位置（1 コマずつ目で確かめたもの。動きのぶれの帯の真ん中）
- 打った時刻（240 fps はラケットが当たったコマ、30 fps は打球音から音の遅れを引いた時刻）
- 打つ人の床の点（跳ぶ前・着いた後の足の間）と、受けた人の床の点・または着地したコマ（床の上、z = 0）。
  どちらも「この辺り」という目安（標準偏差つき）として入れる。カメラ 1 台では奥行きが決まりにくいため

不確かさ（1σ）は、当てはめの誤差（ヤコビアン）・終端速度（6.3〜7.1 m/s）・打った時刻の誤差（その時刻の減速 × σ）・
打つ人の点の誤差（目安の 2 倍・半分で当て直した差）を 2 乗和で足したもの。

使い方::

    .venv\\Scripts\\python.exe agent\\truth\\true_speeds.py            # 当てはめて real_videos_truth.json を書く
    .venv\\Scripts\\python.exe agent\\truth\\true_speeds.py --check    # 書かずに表だけ出す
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from smash_speed.camera import Camera  # noqa: E402
from smash_speed.physics import MPS_TO_KMH, PhysicsParams, simulate  # noqa: E402

POINTS = Path(__file__).with_name("smash_points.json")
BASELINE = ROOT / "agent" / "checks" / "real_videos_baseline.json"
TRUTH = ROOT / "agent" / "checks" / "real_videos_truth.json"
VT_RANGE = (6.3, 7.1)          # 終端速度の範囲 [m/s]（羽根のシャトルの文献値）
FLOOR_SIGMA_M = 0.03           # 着地したコマの高さの目安 [m]


def fit_smash(camera: Camera, t: np.ndarray, uv: np.ndarray, t_hit: float, hitter_xy: Any, hitter_sigma: float,
              end: dict[str, Any] | None = None, vt: float = 6.7, sigma_px: float = 1.5,
              init: np.ndarray | None = None) -> dict[str, Any]:
    """打った時刻 ``t_hit`` の位置・速度（6 個）を、画像上の位置と床の点の目安に当てはめる。

    Args:
        camera: キャリブレーションしたカメラ
        t, uv: シャトルを確かめたコマの時刻 (N,) [s] と画像上の位置 (N, 2) [px]
        hitter_xy: 打点の真下の床の点の目安 [m]
        hitter_sigma: その標準偏差 [m]
        end: 飛行の終わり。``{"t": 時刻, "xy": [x, y], "sigma_m": σ}``（受けた人の床の点）か
            ``{"t": 時刻, "floor": true}``（着地。そのコマのシャトルは ``uv`` に入れる）
        vt: 終端速度 [m/s]
        sigma_px: 画像上の位置の誤差 [px]（残差の重み）
    """
    phys = PhysicsParams(terminal_velocity=vt)
    t = np.asarray(t, float)
    uv = np.asarray(uv, float).reshape(-1, 2)
    hxy = np.asarray(hitter_xy, float)
    t_eval = np.r_[t, end["t"]] if end else t

    def resid(th: np.ndarray) -> np.ndarray:
        pos, _ = simulate(th[:3], th[3:], t_hit, t_eval, phys)
        if not np.isfinite(pos).all():
            return np.full(2 * len(t) + 4, 1e3)
        r = [((camera.project(pos[:len(t)]) - uv) / sigma_px).ravel(), (th[:2] - hxy) / hitter_sigma]
        if end and end.get("floor"):
            r.append(np.array([pos[-1, 2] / end.get("sigma_m", FLOOR_SIGMA_M), 0.0]))
        elif end:
            r.append((pos[-1, :2] - np.asarray(end["xy"], float)) / end["sigma_m"])
        else:
            r.append(np.zeros(2))
        return np.concatenate(r)

    if init is None:
        goal = np.r_[end["xy"], 0.5] if end and "xy" in end else np.r_[-hxy[0], hxy[1], 0.0]
        p0 = np.r_[hxy, 2.6]
        d = goal - p0
        init = np.r_[p0, 70.0 * d / np.linalg.norm(d)]
    sol = least_squares(resid, init, x_scale=np.r_[1, 1, 1, 30, 30, 30], method="trf")
    th = sol.x
    J = sol.jac
    n_img = 2 * len(t)
    r_img = sol.fun[:n_img] * sigma_px
    rms = float(np.sqrt(np.mean(r_img ** 2)))
    scale = max(1.0, rms / sigma_px)          # 画像の誤差が思ったより大きければ、その分だけ不確かさを広げる
    try:
        cov = np.linalg.inv(J.T @ J) * scale ** 2
    except np.linalg.LinAlgError:
        cov = np.full((6, 6), np.nan)
    v0 = th[3:]
    speed = float(np.linalg.norm(v0))
    g = np.r_[0, 0, 0, v0 / speed]
    sd_fit = float(np.sqrt(max(g @ cov @ g, 0.0)))
    # 打った時刻の減速の大きさ |d|v|/dt| = k |v|^2 + g vz / |v|
    decel = float(phys.k * speed ** 2 + phys.g * v0[2] / speed)
    return {"theta": th, "speed_mps": speed, "sd_fit_mps": sd_fit, "decel_mps2": decel, "rms_px": rms,
            "n": int(len(t)), "p0": th[:3], "v0": v0, "success": bool(sol.success)}


def reference_speed(s: dict[str, Any], camera: Camera) -> dict[str, Any]:
    """1 本のスマッシュの正しい初速と不確かさ（1σ）。"""
    pts = np.asarray(s["points"], float)
    t, uv = pts[:, 0], pts[:, 1:3]
    h = s["hitter"]
    kw = dict(end=s.get("end"), sigma_px=s.get("sigma_px", 1.5))
    base = fit_smash(camera, t, uv, s["t_hit"], h["xy"], h["sigma_m"], **kw)
    sp = base["speed_mps"]
    vts = [fit_smash(camera, t, uv, s["t_hit"], h["xy"], h["sigma_m"], vt=vt, init=base["theta"], **kw)["speed_mps"]
           for vt in VT_RANGE]
    sd_vt = abs(vts[1] - vts[0]) / 2
    sd_t = abs(base["decel_mps2"]) * s["t_hit_sigma_s"]
    hs = [fit_smash(camera, t, uv, s["t_hit"], h["xy"], h["sigma_m"] * f, init=base["theta"], **kw)["speed_mps"]
          for f in (0.5, 2.0)]
    sd_h = max(abs(x - sp) for x in hs)
    sd_same = float(np.sqrt(base["sd_fit_mps"] ** 2 + sd_vt ** 2 + sd_h ** 2))
    sd = float(np.sqrt(sd_same ** 2 + sd_t ** 2))
    return {
        "video": s["video"], "id": s["id"], "t_hit_s": round(s["t_hit"], 4), "t_hit_sigma_s": s["t_hit_sigma_s"],
        "kmh": round(sp * MPS_TO_KMH, 1), "sigma_kmh": round(sd * MPS_TO_KMH, 1),
        "sigma_parts_kmh": {k: round(v * MPS_TO_KMH, 1) for k, v in
                            (("fit", base["sd_fit_mps"]), ("terminal_velocity", sd_vt), ("hit_time", sd_t),
                             ("hitter_point", sd_h))},
        # 打った時刻を決めずに同じ時刻の速さで比べるとき用（打った時刻の誤差を除いた不確かさ）
        "sigma_same_time_kmh": round(sd_same * MPS_TO_KMH, 1),
        "hit_point_m": [round(float(x), 2) for x in base["p0"]],
        # 当てはめた軌道（打った時刻の位置・速度、終端速度 6.7 m/s）。``speed_at`` でほかの時刻の速さを出せる
        "state": {"t_s": s["t_hit"], "p0_m": [round(float(x), 4) for x in base["p0"]],
                  "v0_mps": [round(float(x), 4) for x in base["v0"]], "terminal_velocity_mps": 6.7},
        "n_points": base["n"], "rms_px": round(base["rms_px"], 2), "fps": s["fps"],
        "method": s["method"], "note": s.get("note", ""),
    }


def speed_at(ref: dict[str, Any], t: float) -> float:
    """正しい初速の軌道（``ref["state"]``）の、時刻 ``t`` [s] の速さ [km/h]。"""
    st = ref["state"]
    _, v = simulate(np.asarray(st["p0_m"]), np.asarray(st["v0_mps"]), st["t_s"], np.array([t]),
                    PhysicsParams(terminal_velocity=st["terminal_velocity_mps"]))
    return float(np.linalg.norm(v[0]) * MPS_TO_KMH)


def build(points: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    cams = {e["name"]: Camera.from_dict(e["calibration"]["camera"]) for e in baseline["videos"]}
    rows = []
    for s in points["smashes"]:
        r = reference_speed(s, cams[s["video"]])
        rows.append(r)
        print(f"  {r['video']} #{r['id']} {r['t_hit_s']:.3f}s：{r['kmh']} ± {r['sigma_kmh']} km/h"
              f"（{r['n_points']} コマ、{r['rms_px']} px、打点 {r['hit_point_m']}、{r['sigma_parts_kmh']}）")
    return {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "source": "agent/truth/smash_points.json",
            "tool": "agent/truth/true_speeds.py", "sigma": "1σ（当てはめ・終端速度・打った時刻・打つ人の点の 2 乗和）",
            "smashes": rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="書かずに表だけ出す")
    a = ap.parse_args(argv)
    if (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    out = build(json.loads(POINTS.read_text(encoding="utf-8")), json.loads(BASELINE.read_text(encoding="utf-8")))
    if not a.check:
        TRUTH.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"書きました：{TRUTH}（{len(out['smashes'])} 本）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
