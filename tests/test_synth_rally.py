"""すべての種類のショットと、合成ラリーのテスト。"""

import cv2
import numpy as np
import pytest
from scipy.io import wavfile

from smash_speed.config import load_config
from smash_speed.events import audio_onsets
from smash_speed.physics import PhysicsParams, simulate
from smash_speed.shots import SHOT_TYPES, read_rallies, read_shots, write_rallies, write_shots
from smash_speed.synth_rally import (NET_HEIGHT_M, NEXT_TYPES, SHOT_PROFILES, make_rally, make_typed_shot,
                                     observe_rally, rally_camera, render_rally_video)

PHYS = PhysicsParams()


def _net_height(seg):
    """ネットの真上を通るときの高さ [m]。"""
    t = np.linspace(seg.t_hit, seg.t_land, 3000)
    pos, _ = simulate(seg.p0, seg.v0, seg.t_hit, t, PHYS)
    i = np.nonzero(np.sign(pos[:-1, 0]) != np.sign(pos[1:, 0]))[0][0]
    return pos[i, 2]


@pytest.mark.parametrize("profile", list(SHOT_PROFILES))
def test_typed_shot_stays_in_its_ranges(profile):
    pr = SHOT_PROFILES[profile]
    rng = np.random.default_rng(3)
    for side in ("h", "o"):
        s = make_typed_shot(profile, rng, PHYS, side=side)
        assert s.shot_type == pr.shot_type and s.shot_type in SHOT_TYPES
        assert s.side == side
        assert pr.speed_mps[0] - 1e-6 <= s.speed0 <= pr.speed_mps[1] + 1e-6
        assert pr.elevation_deg[0] - 1e-6 <= s.elevation_deg <= pr.elevation_deg[1] + 1e-6
        assert pr.hit_z_m[0] <= s.p0[2] <= pr.hit_z_m[1]
        assert pr.hit_depth_m[0] <= abs(s.p0[0]) <= pr.hit_depth_m[1]
        # ネットを越えて、相手側の狙った深さに落ちる
        assert _net_height(s) >= NET_HEIGHT_M
        assert np.sign(s.landing_xy[0]) == -np.sign(s.p0[0])
        assert pr.land_depth_m[0] - 0.05 <= abs(s.landing_xy[0]) <= pr.land_depth_m[1] + 0.05
        assert abs(s.landing_xy[1]) <= 2.45
        assert not s.intercepted and s.t_end == s.t_land


def test_shot_types_differ_in_speed_and_angle():
    """種類ごとの初速・打ち出し角が、それらしい大小になる（分類の正解として使えるか）。"""
    rng = np.random.default_rng(0)
    med = {}
    for profile in ("smash", "clear", "drop", "net", "lob"):
        segs = [make_typed_shot(profile, rng, PHYS) for _ in range(6)]
        med[profile] = (np.median([s.speed0 for s in segs]), np.median([s.elevation_deg for s in segs]))
    assert med["smash"][0] > med["clear"][0] > med["drop"][0] > med["net"][0]
    assert med["smash"][1] < 0 < med["clear"][1]
    assert med["lob"][1] > 30 and med["net"][0] < 15


def test_rally_shots_connect_and_alternate():
    rally = make_rally(np.random.default_rng(5), PHYS, n_shots=8, t_start=1.0)
    segs = rally.segments
    assert len(segs) >= 4
    assert segs[0].shot_type == "serve" and segs[0].side == "h" and segs[0].t_hit == 1.0
    for a, b in zip(segs[:-1], segs[1:]):
        # 相手が、前のショットの軌道の上で打ち返す
        assert a.intercepted and a.t_end == b.t_hit and a.t_hit + 0.25 <= b.t_hit < a.t_land
        p, _ = simulate(a.p0, a.v0, a.t_hit, np.array([b.t_hit]), PHYS)
        assert np.allclose(p[0], b.p0, atol=1e-6)
        assert a.side != b.side
        assert b.profile in NEXT_TYPES[a.profile]
    assert not segs[-1].intercepted and segs[-1].t_end == segs[-1].t_land
    # 同じ乱数の種なら同じラリー
    again = make_rally(np.random.default_rng(5), PHYS, n_shots=8, t_start=1.0)
    assert [s.profile for s in again.segments] == [s.profile for s in segs]
    assert np.allclose(again.segments[-1].v0, segs[-1].v0)


def test_rallies_cover_every_shot_type():
    seen = set()
    for seed in range(8):
        seen |= {s.shot_type for s in make_rally(np.random.default_rng(seed), PHYS, n_shots=10).segments}
    assert seen == set(SHOT_TYPES) - {"unknown"}


def test_rally_truth_as_shots_csv(tmp_path):
    rally = make_rally(np.random.default_rng(2), PHYS, n_shots=5)
    shots = rally.to_shots(time_offset=0.3)
    back = read_shots(write_shots(tmp_path / "shots.csv", shots))
    assert [s.shot_type for s in back] == [s.shot_type for s in rally.segments]
    assert [s.index_in_rally for s in back] == list(range(1, len(shots) + 1))
    for s, seg in zip(back, rally.segments):
        assert s.t_hit_s == pytest.approx(seg.t_hit + 0.3)
        assert s.speed_kmh == pytest.approx(seg.speed0 * 3.6)
        # 軌道を描き直すと、同じ所に落ちる
        tr = s.trajectory(PHYS)
        assert np.allclose(tr["pos"][-1, :2], seg.landing_xy, atol=1e-3)
    r, = read_rallies(write_rallies(tmp_path / "rallies.csv", [rally.to_rally(0.3)]))
    assert r.n_shots == len(shots) and r.winner_side == rally.segments[-1].side
    assert r.t_end_s == pytest.approx(rally.t_end + 0.3)


def test_observe_rally_labels_each_frame_with_its_shot():
    cfg = load_config()
    cam = rally_camera(cfg)
    rally = make_rally(np.random.default_rng(1), PHYS, n_shots=6)
    df = observe_rally(rally, cam, 120.0, PHYS, np.random.default_rng(0), noise_px=0.0, dropout=0.0)
    assert set(df["shot_index"]) == set(range(len(rally.segments)))
    for i, seg in enumerate(rally.segments):
        t = df.loc[df["shot_index"] == i, "timestamp"]
        assert t.min() >= seg.t_hit and t.max() <= seg.t_end
    # ラリー用のカメラには、ほとんどのコマでシャトルが写る
    assert df["visible"].mean() > 0.8
    vis = df[df["visible"] == 1]
    uv = cam.project(vis[["x_m", "y_m", "z_m"]].to_numpy())
    assert np.allclose(uv, vis[["x_px", "y_px"]].to_numpy(), atol=1e-6)


def test_render_rally_video_with_a_click_per_hit(tmp_path):
    cfg = load_config()
    cam = rally_camera(cfg)
    rally = make_rally(np.random.default_rng(4), PHYS, n_shots=3)
    truth = render_rally_video(rally, cam, cfg, PHYS, tmp_path / "rally.mp4", fps=60.0, pre_s=0.3, post_s=0.2,
                               audio_path=tmp_path / "rally.wav")
    cap = cv2.VideoCapture(str(tmp_path / "rally.mp4"))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert n == len(truth)
    assert set(truth.loc[truth["visible"] == 1, "shot_index"]) == set(range(len(rally.segments)))
    # 打つたびに打球音が 1 つ（音が届くまでの遅れの分だけ後ろ）
    rate, x = wavfile.read(str(tmp_path / "rally.wav"))
    onsets = [t for t, _ in audio_onsets(x.astype(float) / 32767, rate)]
    shots = rally.to_shots(time_offset=0.3 - rally.t_start)
    assert len(onsets) == len(shots)
    for t, s in zip(onsets, shots):
        delay = np.linalg.norm(cam.center - s.hit_point) / 343.0
        assert t == pytest.approx(s.t_hit_s + delay, abs=0.002)
