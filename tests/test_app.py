"""GUI（tkinter）の自動テスト：動画の読み込み、画面クリックによる床の座標、ROI、裏でのコマンド実行。"""

import time

import numpy as np
import pytest

tk = pytest.importorskip("tkinter")


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    from smash_speed.config import load_config
    from smash_speed.validate import make_synthetic_session

    cfg = load_config()
    cfg["synthetic"]["camera"].update(width=960, height=540, f_px=750.0)
    d = tmp_path_factory.mktemp("gui")
    info = make_synthetic_session(cfg, d, speed=75.0, fps=120.0)
    return d, info


@pytest.fixture()
def app(tmp_path, monkeypatch, session):
    from smash_speed.app import main as m

    monkeypatch.setattr(m, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(m.messagebox, "showinfo", lambda *a, **k: None)
    monkeypatch.setattr(m.messagebox, "showwarning", lambda *a, **k: None)
    try:
        a = m.App()
    except tk.TclError:
        pytest.skip("画面がない環境")
    a.withdraw()
    yield a
    a.destroy()


def pump(app, cond, limit=60.0):
    t = time.time()
    while not cond() and time.time() - t < limit:
        app.update()
        time.sleep(0.02)
    return cond()


def test_load_video_and_pick_floor_point(app, session):
    d, info = session
    app.var("video").set(str(d / "synthetic.mp4"))
    app.var("calib").set(str(d / "calibration.json"))
    app.load_video()
    assert pump(app, lambda: app.meta is not None and app.camera is not None)
    assert app.viewer.meta.n_frames > 10

    # 打者の足元を、カメラで投影した画像座標でクリックしたことにする
    X = np.array([[info["hitter_xy"][0], info["hitter_xy"][1], 0.0]])
    u, v = app.camera.project(X)[0]
    app._pick("hitter")
    assert app.viewer.mode == "point"
    app._on_viewer_point(u, v)
    assert float(app.get("hitter_x")) == pytest.approx(info["hitter_xy"][0], abs=0.05)
    assert float(app.get("hitter_y")) == pytest.approx(info["hitter_xy"][1], abs=0.05)
    assert "打者" in app.viewer.markers

    # ROI をドラッグで描いたことにする
    app.viewer.roi = [100.0, 50.0, 800.0, 500.0]
    app._on_roi(app.viewer.roi)
    assert "100" in app.lbl_roi.cget("text")


def test_runner_streams_output_and_reports_failure(app):
    from smash_speed.app.runner import cli_command

    results = []
    app.runner.run([cli_command(["--help"])], results.append, "help")
    assert pump(app, lambda: results)
    assert results == [True]
    assert "analyze" in app.txt_log.get("1.0", "end")

    results.clear()
    app.runner.run([cli_command(["analyze"])], results.append)  # 必須の引数がない
    assert pump(app, lambda: results)
    assert results == [False]
