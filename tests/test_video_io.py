"""Phase 2：動画読み込み（ffprobe・タイムスタンプ・撮影 fps の上書き・音声）のテスト。"""

import shutil

import numpy as np
import pytest

from smash_speed import synth
from smash_speed.config import load_config
from smash_speed.physics import PhysicsParams
from smash_speed.video_io import VideoReader, cut_clip, extract_audio, load_wav, probe

pytestmark = pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffmpeg がありません")


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    cfg = load_config()
    cfg["synthetic"]["camera"].update(width=640, height=360, f_px=500.0)
    phys = PhysicsParams.from_config(cfg)
    cam = synth.default_camera(cfg)
    shot = synth.make_shot(70.0, cfg, phys)
    d = tmp_path_factory.mktemp("vid")
    path = d / "synthetic.mp4"
    truth = synth.render_video(shot, cam, cfg, phys, path, fps=60.0, audio_path=d / "a.wav")
    delay = float(np.linalg.norm(cam.center - shot.p0)) / 343.0  # 打球音の伝搬遅れ
    return path, truth, d, delay


def test_probe_reads_fps_and_timestamps(video):
    path, truth, _, _ = video
    meta = probe(path)
    assert meta.width == 640 and meta.height == 360
    assert meta.fps_meta == pytest.approx(60.0, rel=1e-3)
    assert meta.n_frames == len(truth)
    np.testing.assert_allclose(np.diff(meta.timestamps), 1 / 60, atol=1e-3)
    assert not meta.is_vfr


def test_capture_fps_override_rescales_time(video):
    path, _, _, _ = video
    meta = probe(path, capture_fps=240.0)  # 60fps 記録のスロー動画とみなす
    assert meta.time_scale == pytest.approx(0.25, rel=1e-3)
    np.testing.assert_allclose(np.diff(meta.timestamps), 1 / 240, atol=3e-4)


def test_reader_iterates_range(video):
    path, _, _, _ = video
    meta = probe(path)
    with VideoReader(meta, scale=0.5) as r:
        frames = list(r.iter_range(0.1, 0.2))
    assert frames[0][1] >= 0.1 - 1e-9 and frames[-1][1] <= 0.2 + 1e-9
    assert frames[0][2].shape == (180, 320, 3)


def test_wav_roundtrip(video):
    _, _, d, delay = video
    x, rate = load_wav(d / "a.wav")
    assert rate == 48000 and np.abs(x).max() <= 1.0
    peak_t = np.argmax(np.abs(x)) / rate
    assert peak_t == pytest.approx(0.2 + delay, abs=0.003)  # 打点（pre_s = 0.2 s）+ 伝搬遅れ


def test_cut_clip_and_extract_audio(video, tmp_path):
    path, _, d, _ = video
    clip = cut_clip(path, tmp_path / "clip.mp4", 0.1, 0.4)
    meta = probe(clip)
    assert 0.25 < meta.duration_s < 0.35
    assert not meta.has_audio  # 合成動画には音声トラックがない
    with pytest.raises(Exception):
        extract_audio(path, tmp_path / "x.wav")
