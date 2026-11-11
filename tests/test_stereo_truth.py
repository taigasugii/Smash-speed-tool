"""2 台のカメラからの正しい初速（時刻合わせ・一緒の当てはめ・書き出し・1 台の解析との答え合わせ）。"""

import json

import numpy as np
import pytest

from smash_speed import cli
from smash_speed.config import load_config
from smash_speed.physics import MPS_TO_KMH, PhysicsParams
from smash_speed.shots import Shot, read_shots, write_shots
from smash_speed.stereo_study import (CameraClock, StudyCase, hit_sounds, observe_camera, placement_cameras, run_case,
                                      summarize, true_sync)
from smash_speed.stereo_sync import (SPEED_OF_SOUND_MPS, SyncModel, SyncReport, TrackInterpolator, audio_coarse_offset,
                                     audio_pairs, fit_audio_sync, ray_gap, sharp_envelope, video_sync)
from smash_speed.stereo_truth import (StereoRallyOptions, TruthOptions, analyze_stereo_rally, build_truth,
                                      compare_shots, compare_summary, find_hits_3d, solve_session, speed_at,
                                      trajectory_table, truth_to_shots, write_truth)
from smash_speed.synth_rally import make_rally, make_session, render_rally_video

CFG = load_config()
PHYS = PhysicsParams.from_config(CFG)


class Session:
    """合成ラリー 2 つを、fps・時計の違う 2 台で観測したもの（テストで使い回す）。"""

    def __init__(self, seed: int = 0, fps=(60.0, 120.0), vt: float | None = None, outlier: float = 0.03):
        rng = np.random.default_rng(seed)
        self.world = PHYS if vt is None else PhysicsParams(PHYS.g, vt, PHYS.method, PHYS.rtol, PHYS.atol)
        self.rallies = make_session(rng, self.world, n_rallies=2, n_shots=(5, 7))
        self.cams = placement_cameras("back_corners")
        t0 = self.rallies[0].t_start
        self.clocks = [CameraClock(fps[0], t0 - 1.0, jitter_s=0.0005, av_offset_s=0.004),
                       CameraClock(fps[1], t0 - 2.37, drift=40e-6, jitter_s=0.0005, av_offset_s=-0.003)]
        self.dets = [observe_camera(self.rallies, c, k, self.world, rng, 1.5, 0.1, outlier)
                     for c, k in zip(self.cams, self.clocks)]
        self.sync = true_sync(*self.clocks)
        self.segs = [s for r in self.rallies for s in r.segments]
        self.t_hit = np.array([float(self.clocks[0].video_time(s.t_hit)) for s in self.segs])
        self.onsets = [hit_sounds(self.rallies, c, k, rng) for c, k in zip(self.cams, self.clocks)]
        self.rough = list(self.t_hit + rng.uniform(-0.008, 0.008, len(self.t_hit)))


@pytest.fixture(scope="module")
def session():
    return Session()


@pytest.fixture(scope="module")
def solved(session):
    shots, info = solve_session(session.dets, session.cams, session.sync, session.rough, PHYS, TruthOptions(),
                                session.onsets, 0.0003)
    return shots, info


# ----------------------------------------------------------------------------- 時刻合わせ
def test_sync_model_round_trip():
    m = SyncModel(1.37, 40e-6)
    t = np.array([0.0, 12.5, 600.0])
    np.testing.assert_allclose(m.to_cam1(m.to_cam2(t)), t, atol=1e-12)
    assert m.shifted(0.01).offset_s == pytest.approx(1.38)


def _click_track(rate: int, duration: float, clicks: list[float], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 0.003, int(duration * rate))
    shape = np.exp(-np.arange(200) / 30.0)
    for i, t in enumerate(clicks):
        k = int(round(t * rate))
        x[k:k + 200] += 0.5 * shape * np.random.default_rng(100 + i).normal(0.0, 1.0, 200)
    return x


def test_audio_sync_removes_the_sound_travel_time():
    """音は打点から 2 台へ別々の距離を飛ぶ。距離の差を引くと、時計のずれがサンプルほどの細かさで出る。"""
    from smash_speed.events import audio_onsets

    rate, offset = 48000, 2.4321
    rng = np.random.default_rng(1)
    t_hit = 1.0 + np.cumsum(rng.uniform(0.5, 1.4, 9))     # 等間隔だと 1 打ずれた所でも合ってしまう
    d1, d2 = rng.uniform(4, 14, len(t_hit)), rng.uniform(4, 14, len(t_hit))
    x1 = _click_track(rate, 12.0, list(t_hit + d1 / SPEED_OF_SOUND_MPS), 0)
    x2 = _click_track(rate, 15.0, list(t_hit + offset + d2 / SPEED_OF_SOUND_MPS), 1)
    e1, e2 = sharp_envelope(x1, rate), sharp_envelope(x2, rate)
    coarse, sharp = audio_coarse_offset(e1, e2, rate)
    assert coarse == pytest.approx(offset, abs=0.04)   # 距離の差の分（±30 ms）はまだ入っている
    assert sharp > 1.2
    pairs = audio_pairs(e1, e2, rate, audio_onsets(x1, rate), coarse)
    assert len(pairs) == len(t_hit)
    rough, _, _ = fit_audio_sync(pairs)
    dd = (d2 - d1) / SPEED_OF_SOUND_MPS
    model, sd, n = fit_audio_sync(pairs, dd)
    assert n == len(t_hit)
    assert model.offset_s == pytest.approx(offset, abs=1e-4)
    assert sd < 1e-4
    assert abs(rough.offset_s - offset) > 5 * abs(model.offset_s - offset)


def test_track_interpolator_stays_inside_runs():
    t = np.r_[np.arange(0, 0.1, 1 / 120), np.arange(0.3, 0.4, 1 / 120)]
    uv = np.column_stack([1000 * t, 500 * t**2])
    f = TrackInterpolator(t, uv)
    out = f(np.array([0.05, 0.2, 0.35]))
    np.testing.assert_allclose(out[0], [50.0, 500 * 0.05**2], atol=1e-3)
    assert np.isnan(out[1]).all()                      # 見えていない間は補間しない
    np.testing.assert_allclose(out[2], [350.0, 500 * 0.35**2], atol=1e-3)


def test_ray_gap_of_a_real_point_is_zero():
    cams = placement_cameras("back_corners")
    X = np.array([[1.0, 0.5, 2.0], [-3.0, -1.0, 3.5]])
    g, mid = ray_gap(cams[0], cams[0].project(X), cams[1], cams[1].project(X))
    np.testing.assert_allclose(g, 0.0, atol=1e-6)
    np.testing.assert_allclose(mid, X, atol=1e-6)


def test_video_sync_recovers_offset_with_different_fps(session):
    """音だけの大まかなずれ（25 ms の誤差）から、映像で 1 ms より細かく合う（60 fps と 120 fps・コマの時刻のゆらぎあり）。"""
    rep = video_sync(session.dets, session.cams, session.sync.shifted(0.025))
    t = float(np.mean(session.t_hit))
    assert rep.method == "video"
    assert abs(rep.model.to_cam2(t) - session.sync.to_cam2(t)) < 0.001
    assert rep.gap_cm < 5
    assert rep.sharpness > 2
    assert not rep.warnings


def test_video_sync_reports_when_nothing_overlaps(session):
    empty = session.dets[1].assign(visible=0, x_px=np.nan, y_px=np.nan)
    rep = video_sync([session.dets[0], empty], session.cams, session.sync)
    assert rep.method == "none" and rep.warnings


# ----------------------------------------------------------------------------- 一緒の当てはめ
def test_joint_fit_beats_naive_triangulation(session, solved):
    """連続した時間の当てはめは初速を 2% 以内で出す。コマの組の三角測量の最初の速さは、それよりずっと悪い。"""
    shots, _ = solved
    assert len(shots) == len(session.segs)
    err, naive = [], []
    o = TruthOptions()
    for s, g in zip(shots, session.segs):
        assert s.fit is not None and s.ok(o), s.message
        true = g.speed0 * MPS_TO_KMH
        err.append(abs(s.kmh - true) / true)
        naive.append(abs(s.naive_kmh["nearest"] - true) / true)
        assert np.linalg.norm(s.fit.p0 - g.p0) < 0.15
    assert np.mean(err) < 0.02
    assert np.mean(naive) > 3 * np.mean(err)


def test_hit_times_come_from_the_3d_crossing(session, solved):
    shots, info = solved
    dt = np.array([s.t_hit for s in shots]) - session.t_hit
    assert np.median(np.abs(dt)) < 0.002               # 渡した時刻は ±8 ms ずれていた
    # ラリーの最初のショットは前の軌道がないので打球音から。ほかは交わる時刻か、ずれを直した打球音の確かな方
    assert shots[0].hit_method == "audio"
    assert {s.hit_method for s in shots} <= {"crossing", "audio"}
    assert all(s.t_hit_sigma < 0.004 for s in shots)
    # スマホごとの音と映像のずれ（+4 ms・−3 ms）が、交わる時刻と打球音の差から見える
    assert info["av_offset_ms"][0] == pytest.approx(4.0, abs=1.5)
    assert info["av_offset_ms"][1] == pytest.approx(-3.0, abs=1.5)


def test_sigma_covers_the_error(session, solved):
    shots, _ = solved
    z = [abs(s.kmh - g.speed0 * MPS_TO_KMH) / s.sigma_kmh for s, g in zip(shots, session.segs)]
    assert np.mean(np.array(z) <= 2.0) >= 0.9
    assert all(set(s.sigma_parts) >= {"fit", "hit_time", "terminal_velocity", "sync"} for s in shots)
    assert all(s.sigma_same_time_kmh <= s.sigma_kmh for s in shots)


def test_false_hit_is_merged_and_outliers_are_dropped(session, solved):
    """飛行の途中に打球ではない区切りを入れても、前後がつながっていれば 1 つの飛行に戻す。"""
    shots, _ = solved
    i = int(np.argmax(np.diff(session.t_hit)[:len(session.rallies[0].segments) - 1]))   # 長い飛行の真ん中に区切りを足す
    extra = (session.t_hit[i] + session.t_hit[i + 1]) / 2
    got, info = solve_session(session.dets, session.cams, session.sync, session.rough + [extra], PHYS,
                              TruthOptions(uncertainty=False), session.onsets)
    assert info["n_merged"] == 1 and len(got) == len(session.segs)
    assert got[i].kmh == pytest.approx(session.segs[i].speed0 * MPS_TO_KMH, rel=0.03)
    assert sum(s.n_rejected for s in shots) > 0         # でたらめな検出（3%）は外されている


def test_session_terminal_velocity_is_estimated():
    """シャトルの終端速度が設定（6.7 m/s）と違っても、長い飛行から決め直して初速が合う。"""
    s = Session(seed=3, fps=(60.0, 60.0), vt=6.2, outlier=0.0)
    o = TruthOptions(uncertainty=False)
    shots, info = solve_session(s.dets, s.cams, s.sync, s.rough, PHYS, o, s.onsets)
    assert info["terminal_velocity"]["source"] == "session"
    assert info["terminal_velocity"]["terminal_velocity_mps"] == pytest.approx(6.2, abs=0.15)
    fixed, _ = solve_session(s.dets, s.cams, s.sync, s.rough, PHYS,
                             TruthOptions(uncertainty=False, terminal_velocity="fixed"), s.onsets)

    def mape(ss):
        return np.mean([abs(x.kmh - g.speed0 * MPS_TO_KMH) / (g.speed0 * MPS_TO_KMH)
                        for x, g in zip(ss, s.segs) if x.fit is not None and x.ok(o)])

    assert mape(shots) < 0.02
    assert mape(fixed) > 2 * mape(shots)


def test_shot_seen_by_one_camera_gets_no_truth(session):
    blind = session.dets[1].copy()
    m = blind.timestamp < float(session.sync.to_cam2(session.t_hit[1]))
    blind.loc[m, ["visible"]] = 0
    blind.loc[m, ["x_px", "y_px"]] = np.nan
    shots, _ = solve_session([session.dets[0], blind], session.cams, session.sync, session.rough[:3], PHYS,
                             TruthOptions(uncertainty=False))
    assert shots[0].fit is None and "カメラ 1 にしか" in shots[0].message
    assert shots[1].fit is not None


def test_hits_are_found_from_3d_points(session):
    """打球は、三角測量した 3 次元の点の速度が急に変わる所として見つかる（1〜2 コマの粗さ）。"""
    from smash_speed.stereo_sync import RayPairs

    t, X = RayPairs(session.dets, session.cams).points(session.sync)
    found = np.array(find_hits_3d(t, X, PHYS))
    hit = [np.abs(found - th).min() < 0.05 for th in session.t_hit]
    assert np.mean(hit) >= 0.9
    # 見つけ方は多めに拾う。打球ではなかった区切りは、当てはめた前後の軌道がつながっていることから消える
    o = TruthOptions(uncertainty=False)
    shots, info = solve_session(session.dets, session.cams, session.sync, list(found), PHYS, o, session.onsets)
    good = [s for s in shots if s.fit is not None and s.ok(o)]
    assert sum(np.abs(session.t_hit - s.t_hit).min() >= 0.05 for s in good) <= 1
    assert sum(np.abs(np.array([s.t_hit for s in good]) - th).min() < 0.01 for th in session.t_hit) >= len(session.segs) - 1


# ----------------------------------------------------------------------------- 書き出しと答え合わせ
@pytest.fixture(scope="module")
def truth(session, solved):
    rep = SyncReport(session.sync, "video", sigma_s=0.0003, n_pairs=100, gap_cm=1.5)
    return build_truth(solved[0], rep, TruthOptions(), ["a.mp4", "b.mp4"], [60.0, 120.0])


def test_truth_file_has_speed_sigma_and_state(session, truth, tmp_path):
    write_truth(tmp_path, truth, PHYS)
    data = json.loads((tmp_path / "stereo_truth.json").read_text(encoding="utf-8"))
    e = data["shots"][1]
    assert {"id", "t_hit_s", "t_hit_cam2_s", "kmh", "sigma_kmh", "sigma_parts_kmh", "sigma_same_time_kmh",
            "hit_point_m", "launch_angle_deg", "state", "n_points", "rms_px"} <= set(e)
    g = session.segs[1]
    assert e["launch_angle_deg"] == pytest.approx(g.elevation_deg, abs=1.5)
    assert e["t_hit_cam2_s"] == pytest.approx(float(session.sync.to_cam2(e["t_hit_s"])), abs=1e-3)
    assert speed_at(e, e["state"]["t_s"]) == pytest.approx(e["kmh"], abs=0.2)
    assert speed_at(e, e["state"]["t_s"] + 0.05) < e["kmh"]
    shots = read_shots(tmp_path / "shots.csv")
    assert len(shots) == len(data["shots"]) and shots[1].source == "stereo"
    assert shots[1].speed_kmh == pytest.approx(e["kmh"])
    traj = trajectory_table(data)
    assert set(traj.id) == {x["id"] for x in data["shots"] if x["ok"]}
    assert (tmp_path / "trajectories.csv").exists()


def test_compare_single_camera_shots_in_either_clock(session, truth):
    """1 台の解析（初速を 10% 高く・時刻を 20 ms 遅くしたもの）との差が、どちらの動画の時刻でも同じに出る。"""
    ref = truth_to_shots(truth, PHYS, classify=False)
    for cam in (1, 2):
        single = []
        for s in ref[:-1]:                               # 最後の 1 本は 1 台の解析で見つからなかったことにする
            t = s.t_hit_s + 0.02
            single.append(Shot(t_hit_s=t if cam == 1 else float(session.sync.to_cam2(t)), speed_kmh=s.speed_kmh * 1.1,
                               hit_x_m=s.hit_x_m + 1.0, hit_y_m=s.hit_y_m, hit_z_m=s.hit_z_m, shot_type="smash"))
        df = compare_shots(truth, single, camera=cam)
        s = compare_summary(df, len(single))
        assert s["n_truth"] == len(ref) and s["n_compared"] == len(ref) - 1
        assert s["mape_pct"] == pytest.approx(10.0, abs=0.1)
        assert s["median_abs_dt_ms"] == pytest.approx(20.0, abs=0.5)
        assert s["median_hit_point_err_m"] == pytest.approx(1.0, abs=0.01)
        # 20 ms 後の正解の軌道は遅くなっているので、同じ時刻で比べると差はもっと大きい
        assert (df.err_same_t_pct.dropna() > 10.0).all()
        assert np.isnan(df.kmh.iloc[-1])


def test_stereo_compare_command(session, truth, tmp_path, capsys):
    write_truth(tmp_path / "truth", truth, PHYS)
    single = tmp_path / "single"
    single.mkdir()
    write_shots(single / "shots.csv", truth_to_shots(truth, PHYS, classify=False))
    assert cli.main(["stereo-compare", "--truth", str(tmp_path / "truth"), "--shots", str(single)]) == 0
    out = capsys.readouterr().out
    assert "中央値 0.0%" in out
    assert (tmp_path / "truth" / "stereo_compare.csv").exists()
    assert json.loads((tmp_path / "truth" / "stereo_compare.json").read_text(encoding="utf-8"))["n_compared"] > 0


# ----------------------------------------------------------------------------- 調べ・動画からの通し
def test_study_case_finds_hits_by_itself():
    """打球を自動で見つける流れ（``find_hits_3d``）でも、ほとんどのショットに初速が出て、大きく外れたものがない。"""
    case = StudyCase("t", fps=(120.0, 120.0), hits="found", n_rallies=2)
    s = summarize(run_case(case, CFG, seed=1))
    assert s["got_pct"] >= 85
    assert s["mape_pct"] < 2.0


def test_bad_placement_is_worse_than_good():
    good = summarize(run_case(StudyCase("g", n_rallies=2), CFG, seed=2))
    bad = summarize(run_case(StudyCase("b", placement="narrow", n_rallies=2), CFG, seed=2))
    assert good["mape_pct"] < 2.0
    assert bad["mape_pct"] > 2 * good["mape_pct"]


def test_two_videos_end_to_end(tmp_path):
    """合成ラリーを 2 台の動画に描き、音で大まかに・映像で細かく時刻を合わせて、初速を出すところまで。"""
    from smash_speed.camera import look_at

    rally = make_rally(np.random.default_rng(4), PHYS, n_shots=5)
    cams = [look_at(np.array([-10.0, -3.6, 2.0]), np.array([1.0, 0.0, 2.2]), 475.0, 960, 540),
            look_at(np.array([-10.0, 3.6, 2.0]), np.array([1.0, 0.0, 2.2]), 475.0, 960, 540)]
    pre = (0.4, 1.15)
    for k, cam in enumerate(cams):
        render_rally_video(rally, cam, CFG, PHYS, tmp_path / f"v{k}.mp4", fps=60.0, pre_s=pre[k], radius_px=3,
                           audio_path=tmp_path / f"a{k}.wav", seed=k)
        (tmp_path / f"c{k}.json").write_text(json.dumps({"camera": cam.to_dict()}), encoding="utf-8")
    o = StereoRallyOptions(videos=[tmp_path / "v0.mp4", tmp_path / "v1.mp4"],
                           calibs=[tmp_path / "c0.json", tmp_path / "c1.json"], out_dir=tmp_path / "out",
                           audio_paths=[tmp_path / "a0.wav", tmp_path / "a1.wav"])
    truth = analyze_stereo_rally(o, CFG, progress=lambda m: None)
    assert truth["sync"]["method"] == "video"
    assert truth["sync"]["model"]["offset_s"] == pytest.approx(pre[1] - pre[0], abs=0.003)
    assert abs(truth["sync"]["audio"]["audio_minus_video_ms"]) < 3
    ok = [e for e in truth["shots"] if e["ok"]]
    assert len(ok) >= len(rally.segments) - 1
    t_true = np.array([s.t_hit - rally.t_start + pre[0] for s in rally.segments])
    for e in ok:
        i = int(np.argmin(np.abs(t_true - e["t_hit_s"])))
        assert abs(t_true[i] - e["t_hit_s"]) < 0.02
        assert e["kmh"] == pytest.approx(rally.segments[i].speed0 * MPS_TO_KMH, rel=0.06)
    assert (tmp_path / "out" / "stereo_truth.json").exists() and (tmp_path / "out" / "shots.csv").exists()
