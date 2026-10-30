"""フォームの解析の流れ（form.py）のテスト。動画と姿勢推定のモデルの代わりに、合成の骨格を返す偽物を使う。"""

import numpy as np
import pytest

from smash_speed.form import FEATURES_FILE, FORM_DIR, FormOptions, analyze, read_rows, shot_id
from smash_speed.pose import NUM_KEYPOINTS, PoseSequence
from smash_speed.pose_prep import SPEED_OF_SOUND_MPS, NormalizedPose, mirror
from smash_speed.synth_pose import SwingParams, joints_3d, side_camera, vary

CAM = side_camera()
FPS = 60.0


class SessionModel:
    """「動画」の時刻から、その時刻の骨格を返す偽物のモデル。スイングの間は、前のスイングの終わりの姿勢から
    次のスイングの構えへゆっくり戻る。"""

    name = "fake"

    def __init__(self, swings: dict[float, SwingParams], left_handed: bool = False):
        self.swings = swings
        self.left = left_handed

    def __call__(self, image):
        t = float(image[0, 0])
        cs = sorted(self.swings)
        nxt = next((c for c in cs if t <= c + 1.0), None)
        prev = max((c for c in cs if c + 1.0 < t), default=None)
        if nxt is not None and t >= nxt - 1.5:
            X = joints_3d(self.swings[nxt], np.array([t - nxt]))
        elif prev is None or nxt is None:
            c = nxt if nxt is not None else prev
            X = joints_3d(self.swings[c], np.array([-1.5 if nxt is not None else 1.0]))
        else:
            w = (t - prev - 1.0) / (nxt - 1.5 - prev - 1.0)
            X = (1 - w) * joints_3d(self.swings[prev], np.array([1.0])) + w * joints_3d(self.swings[nxt], np.array([-1.5]))
        xy = CAM.project(X.reshape(-1, 3)).reshape(1, NUM_KEYPOINTS, 2)
        if self.left:
            xy = mirror(PoseSequence(t=[t], xy=xy, score=np.ones((1, NUM_KEYPOINTS)), width=CAM.width)).xy
        return [(xy[0], np.full(NUM_KEYPOINTS, 0.9))]


def _frames(t0, t1):
    for k in range(int(np.ceil(max(t0, 0) * FPS)), int(np.floor(t1 * FPS)) + 1):
        yield k / FPS, np.array([[k / FPS]])


INFO = {"width": CAM.width, "height": CAM.height, "fps": FPS, "duration_s": 20.0}
SWINGS = {5.0: SwingParams(), 10.0: vary(SwingParams(), elbow_impact_deg=140.0), 15.0: SwingParams()}


def test_analyze_with_audio(tmp_path):
    delay = 7.0 / SPEED_OF_SOUND_MPS
    onsets = [(c + delay, 100.0) for c in SWINGS] + [(5.8, 60.0), (18.0, 80.0), (12.0, 5.0)]
    rows = analyze("clear_01.mp4", tmp_path, FormOptions(), frames=_frames, info=INFO, onsets=onsets,
                   model=SessionModel(SWINGS), progress=lambda s: None)
    # 18 秒の音（スイングがない）と弱い音は捨て、5.8 秒の音（相手の打球）は近すぎるのでまとめる
    assert [r["shot_id"] for r in rows] == [shot_id("clear_01.mp4", c) for c in SWINGS]
    assert all(r["impact_method"] == "audio" for r in rows)
    assert [r["t_impact_s"] for r in rows] == pytest.approx(list(SWINGS), abs=1e-6)
    assert rows[1]["elbow_angle_deg"] == pytest.approx(140.0, abs=3.0)
    assert rows[0]["elbow_angle_deg"] == pytest.approx(160.0, abs=3.0)
    back = read_rows(tmp_path / FORM_DIR / FEATURES_FILE)
    assert len(back) == 3 and back[0]["shot_id"] == "clear_01_000500" and back[0]["left_handed"] is False
    assert back[1]["elbow_angle_deg"] == pytest.approx(rows[1]["elbow_angle_deg"], rel=1e-5)
    n = NormalizedPose.load(tmp_path / FORM_DIR / "poses" / "clear_01_001000.npz")
    assert n.meta["shot_id"] == "clear_01_001000" and abs(n.t[0] + 1.0) < 0.02
    raw = PoseSequence.load(tmp_path / FORM_DIR / "raw" / "clear_01_001000.npz")
    assert raw.model == "fake" and raw.t[0] < 9.0
    assert (tmp_path / FORM_DIR / "form.json").exists()


def test_analyze_without_audio_finds_swings_from_the_wrist(tmp_path):
    rows = analyze("v.mp4", tmp_path, FormOptions(), frames=_frames, info=INFO, onsets=[],
                   model=SessionModel(SWINGS), progress=lambda s: None)
    assert len(rows) == 3 and all(r["impact_method"] == "wrist" for r in rows)
    # 手首の速さが最大の時刻（打点の少し前）を打点にしている
    assert [r["t_impact_s"] for r in rows] == pytest.approx([c - 0.03 for c in SWINGS], abs=0.02)


def test_analyze_given_times_and_left_handed(tmp_path):
    onsets = [(10.0 + 7.0 / SPEED_OF_SOUND_MPS, 100.0)]
    rows = analyze("v.mp4", tmp_path, FormOptions(left_handed=True), hit_times=[10.05], frames=_frames, info=INFO,
                   onsets=onsets, model=SessionModel(SWINGS, left_handed=True), progress=lambda s: None)
    assert len(rows) == 1 and rows[0]["left_handed"] is True
    assert rows[0]["elbow_angle_deg"] == pytest.approx(140.0, abs=3.0)
    assert rows[0]["chain_in_order"] == 1.0


def test_wrist_only_impact_is_early_unless_calibrated(tmp_path):
    """音がないと打点は手首の速さの最大（打点の約 0.03 秒前）になり、肘はまだ伸び切っていない。
    手で付けた打点から決めた差（wrist_offset_s）を足すと、打点の角度に戻る。"""
    kw = dict(hit_times=[10.0], frames=_frames, info=INFO, onsets=[], model=SessionModel(SWINGS),
              progress=lambda s: None)
    early = analyze("v.mp4", tmp_path, FormOptions(), **kw)[0]
    fixed = analyze("v.mp4", tmp_path, FormOptions(wrist_offset_s=0.032), **kw)[0]
    assert early["elbow_angle_deg"] < 135.0
    assert fixed["elbow_angle_deg"] == pytest.approx(140.0, abs=3.0)
