"""姿勢推定（pose.py）と合成の骨格（synth_pose.py）のテスト。実際のモデルの代わりに偽物のモデルを使う。"""

import numpy as np
import pytest

from smash_speed.pose import (
    FLIP_INDEX, KEYPOINTS, KP, NUM_KEYPOINTS, Labels, PoseSequence, clean, estimate_frames, labels_from_sequence,
    load_labels, main, pck, pick_person, torso_length, write_labels,
)
from smash_speed.synth_pose import (
    SwingParams, angles, joints_3d, project_swing, side_camera, swing_times, wrist_speed_peak_time,
)


class FakeModel:
    """コマの番号（画像の左上の画素に書いた値）から、合成の骨格に揺れを足して返す偽物のモデル。"""

    name = "fake"

    def __init__(self, truth: PoseSequence, noise_px: float = 0.0, others: int = 0, seed: int = 0):
        self.truth = truth
        self.noise = noise_px
        self.others = others
        self.rng = np.random.default_rng(seed)

    def __call__(self, image):
        k = int(image[0, 0, 0]) + 256 * int(image[0, 0, 1])
        xy = self.truth.xy[k] + self.rng.normal(0, self.noise, (NUM_KEYPOINTS, 2))
        score = np.full(NUM_KEYPOINTS, 0.9)
        score[KP["left_wrist"]] = 0.1 if k % 10 == 0 else 0.9  # ときどき隠れる非利き腕
        people = [(xy, score)]
        for i in range(self.others):  # 奥の小さい人（相手・審判など）
            people.insert(0, (self.truth.xy[k] * 0.3 + [100 + 300 * i, 50], np.full(NUM_KEYPOINTS, 0.9)))
        return people


def _frames(n):
    for k in range(n):
        img = np.zeros((4, 4, 3), np.uint8)
        img[0, 0, 0], img[0, 0, 1] = k % 256, k // 256
        yield k / 60.0, img


def _truth(fps=60.0):
    return project_swing(SwingParams(), side_camera(), swing_times(fps))


def test_synthetic_swing_is_consistent():
    """骨の長さはいつも同じ、肘・上腕の角度はインパクトでちょうど決めた値、画像の中に収まる。"""
    p = SwingParams()
    t = swing_times(120)
    X = joints_3d(p, t)
    up = np.linalg.norm(X[:, KP["right_elbow"]] - X[:, KP["right_shoulder"]], axis=1)
    fore = np.linalg.norm(X[:, KP["right_wrist"]] - X[:, KP["right_elbow"]], axis=1)
    assert np.ptp(up) < 1e-9 and np.ptp(fore) < 1e-9
    a = angles(p, np.array([0.0]))
    assert a["elbow"][0] == pytest.approx(p.elbow_impact_deg)
    assert a["arm"][0] == pytest.approx(p.arm_impact_deg)
    seq = project_swing(p, side_camera(), t)
    assert np.isfinite(seq.xy).all()
    assert (seq.xy[..., 0] > 0).all() and (seq.xy[..., 0] < 1920).all()
    assert (seq.xy[..., 1] > 0).all() and (seq.xy[..., 1] < 1080).all()
    # 打つ向き（+x）は画像の右：踏み込むと腰が右へ動く
    hip = (seq.joint("left_hip") + seq.joint("right_hip")) / 2
    assert hip[-1, 0] > hip[0, 0] + 50
    # 手首の速さの最大は、インパクトの少し前
    assert -0.1 < wrist_speed_peak_time(p) < 0.0


def test_flip_index_swaps_left_and_right():
    for i, j in enumerate(FLIP_INDEX):
        a, b = KEYPOINTS[i], KEYPOINTS[j]
        assert a.replace("left", "right") == b or a.replace("right", "left") == b or a == b == "nose"


def test_save_and_load(tmp_path):
    seq = _truth()
    seq.meta["video"] = "a.mp4"
    seq.score[3, 4] = 0.25
    back = PoseSequence.load(seq.save(tmp_path / "p.npz"))
    assert np.allclose(back.xy, seq.xy) and np.allclose(back.score, seq.score)
    assert back.width == 1920 and back.meta["video"] == "a.mp4" and back.fps == pytest.approx(60.0)


def test_pick_person_prefers_large_then_nearest():
    big = (np.array([[0, 0], [100, 300], [50, 150]] + [[50, 150]] * 14, float), np.ones(17))
    small = (np.array([[500, 0], [520, 60], [510, 30]] + [[510, 30]] * 14, float), np.ones(17))
    assert pick_person([small, big]) == 1
    assert pick_person([small, big], prev_center=np.array([505.0, 30.0])) == 0
    assert pick_person([small, big], prev_center=np.array([505.0, 30.0]), max_jump_px=10.0) == 0
    assert pick_person([small], prev_center=np.array([0.0, 0.0]), max_jump_px=10.0) is None
    assert pick_person([]) is None
    hidden = (big[0], np.zeros(17))
    assert pick_person([hidden]) is None


def test_estimate_frames_follows_the_hitter():
    truth = _truth()
    seq = estimate_frames(_frames(len(truth)), FakeModel(truth, others=2), width=1920, height=1080, fps=60.0)
    assert len(seq) == len(truth) and seq.model == "fake"
    assert np.allclose(seq.xy, truth.xy)  # 小さい人は選ばない


def test_clean_drops_low_scores_and_fills_short_gaps():
    truth = _truth()
    seq = truth.copy()
    seq.score[:] = 0.9
    seq.score[10, KP["left_wrist"]] = 0.1          # 1 コマだけ → 埋める
    seq.score[30:40, KP["left_elbow"]] = 0.1       # 10 コマ → 欠けのまま
    seq.score[0, KP["nose"]] = 0.1                 # 端 → 欠けのまま
    out = clean(seq)
    assert np.isfinite(out.xy[10, KP["left_wrist"]]).all()
    assert np.allclose(out.xy[10, KP["left_wrist"]], truth.xy[10, KP["left_wrist"]], atol=3.0)
    assert out.score[10, KP["left_wrist"]] == pytest.approx(0.9)
    assert np.isnan(out.xy[30:40, KP["left_elbow"]]).all() and (out.score[30:40, KP["left_elbow"]] == 0).all()
    assert np.isnan(out.xy[0, KP["nose"]]).all()
    assert np.isfinite(seq.xy).all()  # 元は変えない


def test_pck_matches_the_noise(tmp_path):
    """関節に正規分布の揺れ σ を足すと、誤差 r は Rayleigh 分布：PCK = 1 − exp(−(αL)² / 2σ²)。"""
    truth = _truth()
    labels = labels_from_sequence(truth, frames=np.arange(0, len(truth), 3), video="v.mp4")
    path = write_labels(tmp_path / "labels.csv", labels)
    back = load_labels(path)
    assert back.video == "v.mp4" and np.allclose(back.xy, labels.xy, atol=0.01)
    perfect = pck(truth, back)
    assert perfect["pck"] == 1.0 and perfect["mean_error_px"] < 0.01 and perfect["missing"] == 0
    L = float(np.median(torso_length(back.xy)))
    sigma = 0.2 * L / np.sqrt(2 * np.log(2))   # 半分が当たる揺れ
    noisy = truth.copy(xy=truth.xy + np.random.default_rng(1).normal(0, sigma, truth.xy.shape))
    r = pck(noisy, back)
    assert r["n"] == len(back.t) * NUM_KEYPOINTS
    assert r["pck"] == pytest.approx(0.5, abs=0.06)
    assert r["mean_error_px"] == pytest.approx(sigma * np.sqrt(np.pi / 2), rel=0.1)
    # 信頼度の低い関節は外れに数える、隠れた関節は数えない
    low = truth.copy()
    low.score[:, KP["left_wrist"]] = 0.0
    assert pck(low, back)["per_joint"]["left_wrist"] == 0.0
    back.visible[:, KP["left_wrist"]] = 1
    r2 = pck(low, back)
    assert r2["per_joint"]["left_wrist"] is None and r2["pck"] == 1.0
    assert pck(low, back, include_hidden=True)["per_joint"]["left_wrist"] == 0.0


def test_pck_missing_frames_and_bad_names(tmp_path):
    truth = _truth()
    labels = labels_from_sequence(truth, frames=np.array([0, 5]))
    labels.t = labels.t + np.array([0.0, 10.0])   # 2 つ目のコマは推定の列にない
    assert pck(truth, labels)["missing"] == 1
    path = tmp_path / "bad.csv"
    path.write_text("video,frame,t_s,joint,x,y,visible\nv,0,0,elbow,1,2,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="知らない関節"):
        load_labels(path)


def test_cli_pck(tmp_path, capsys):
    truth = _truth()
    write_labels(tmp_path / "l.csv", labels_from_sequence(truth, frames=np.arange(0, 60, 6)))
    truth.save(tmp_path / "p.npz")
    assert main(["pck", str(tmp_path / "l.csv"), str(tmp_path / "p.npz")]) == 0
    assert "PCK@0.2 1.000" in capsys.readouterr().out


def test_mediapipe_runs_if_installed():
    """MediaPipe と重みがあれば、実際に動いて 17 点の形で返す（人がいない画像なので空）。"""
    pytest.importorskip("mediapipe")
    from smash_speed.pose import MediaPipeModel, mediapipe_model_path
    try:
        path = mediapipe_model_path("lite", download=False)
    except FileNotFoundError:
        pytest.skip("MediaPipe の重みがない")
    try:
        model = MediaPipeModel(model_path=path)
    except OSError as e:  # 画面のない環境で、描画のライブラリがない
        pytest.skip(str(e))
    try:
        assert model(np.zeros((240, 320, 3), np.uint8)) == []
    finally:
        model.close()
