"""重ねた動画と見どころの書き出し（``smash_speed.overlay``）のテスト。合成のラリー動画を使う。"""

import json
import shutil

import cv2
import numpy as np
import pytest

from smash_speed.cli import main as cli_main
from smash_speed.config import load_config
from smash_speed.overlay import (DrawOptions, HighlightOptions, OverlayData, OverlayOptions, pick_fastest,
                                 pick_longest_rallies, render_overlay, shot_label, write_highlights)
from smash_speed.physics import PhysicsParams
from smash_speed.shot_list import DETECTIONS_FILE, SHOT_LIST_FILE
from smash_speed.shots import Rally, Shot, write_rallies, write_shots
from smash_speed.synth_rally import make_rally, rally_camera, render_rally_video

CFG = load_config()
PHYS = PhysicsParams.from_config(CFG)
CAM = rally_camera(CFG).scaled(0.5)     # 960x540（テストを速くする）
FPS = 30.0
HAS_FFMPEG = shutil.which("ffmpeg") is not None or bool(
    list((__import__("pathlib").Path(__import__("os").environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet"
          / "Packages").glob("Gyan.FFmpeg*/*/bin/ffmpeg.exe")))


@pytest.fixture(scope="module")
def shots_dir(tmp_path_factory):
    """合成のラリー動画と、その正解で作ったショットの一覧の出力先（解析は動かさない）。"""
    d = tmp_path_factory.mktemp("overlay")
    rally = make_rally(np.random.default_rng(6), PHYS, n_shots=5)
    truth = render_rally_video(rally, CAM, CFG, PHYS, d / "rally.mp4", fps=FPS, pre_s=0.3)
    (d / "calibration.json").write_text(json.dumps({"camera": CAM.to_dict()}), encoding="utf-8")
    out = d / "list"
    out.mkdir()
    shots = rally.to_shots(0.3 - rally.t_start, phys=PHYS)
    write_shots(out / "shots.csv", shots)
    write_rallies(out / "rallies.csv", [Rally(1, shots[0].t_hit_s, shots[-1].t_land_s, len(shots))])
    truth.to_csv(out / DETECTIONS_FILE, index=False)
    (out / SHOT_LIST_FILE).write_text(json.dumps({"video": str(d / "rally.mp4"), "calib": str(d / "calibration.json"),
                                                  "capture_fps": None}), encoding="utf-8")
    return out, shots, truth


def _frames(path):
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def _yellow(img):
    """軌跡の色（黄色）に近い画素の数。"""
    b, g, r = (img[..., k].astype(int) for k in range(3))
    return int(((b < 90) & (g > 180) & (r > 180)).sum())


@pytest.mark.parametrize("ffmpeg", [True, False])
def test_overlay_draws_trail_trajectory_and_label(shots_dir, tmp_path, ffmpeg):
    if ffmpeg and not HAS_FFMPEG:
        pytest.skip("ffmpeg がない")
    d, shots, truth = shots_dir
    out = render_overlay(OverlayOptions(d, tmp_path / "o.mp4"), CFG, progress=lambda m: None, use_ffmpeg=ffmpeg)
    plain = render_overlay(OverlayOptions(d, tmp_path / "plain.mp4", draw=DrawOptions(False, 0.4, False, False)), CFG,
                           progress=lambda m: None, use_ffmpeg=ffmpeg)
    fo, fp = _frames(out), _frames(plain)
    assert len(fo) == len(fp) == len(truth)
    assert fo[0].shape[:2] == (CAM.height, CAM.width)
    # 何も描かなければ元の画像と同じ（書き方による違いだけ）。描くと軌跡の色の画素が増える
    k = int(truth.index[truth.visible > 0][10])
    assert _yellow(fp[k]) == 0 and _yellow(fo[k]) > 20
    # 打った直後は、打点の近くに文字の箱（半透明の黒）が出る
    data = OverlayData(d, cfg=CFG)
    sd = data.draws[2]
    assert sd.anchor is not None and len(sd.uv) > 5
    kh = int(np.searchsorted(truth.timestamp, sd.shot.t_hit_s)) + 3
    x, y = sd.anchor
    box_o = fo[kh][max(0, y - 80):y, max(0, x - 60):x + 60].astype(int)
    box_p = fp[kh][max(0, y - 80):y, max(0, x - 60):x + 60].astype(int)
    assert box_o.mean() < box_p.mean() - 20
    # 2 秒以上たつと文字は消える（文字の箱の暗さがなくなる）
    k_late = int(np.searchsorted(truth.timestamp, shots[-1].t_land_s or shots[-1].t_hit_s)) - 1
    assert abs(fo[k_late].astype(int).mean() - fp[k_late].astype(int).mean()) < 8


def test_overlay_time_range_and_height(shots_dir, tmp_path):
    d, shots, truth = shots_dir
    msgs = []
    t0, t1 = float(shots[1].t_hit_s), float(shots[1].t_hit_s) + 1.0
    out = render_overlay(OverlayOptions(d, tmp_path / "part.mp4", start_s=t0, end_s=t1, height=270), CFG,
                         progress=msgs.append, use_ffmpeg=HAS_FFMPEG)
    f = _frames(out)
    assert abs(len(f) - FPS * (t1 - t0)) <= 2
    assert f[0].shape[:2] == (270, 480)
    assert any("完了" in m for m in msgs) and any("%" in m for m in msgs)


def test_shot_label_and_pickers():
    s = [Shot(1.0, "h", "smash", 300.0, rally_id=1), Shot(2.0, "o", "clear", 200.0, rally_id=1),
         Shot(5.0, "h", "smash", 350.4, rally_id=2), Shot(6.0, "o", "drive", None, rally_id=2),
         Shot(7.0, "h", "smash", 280.0, rally_id=2)]
    assert shot_label(s[2]) == "スマッシュ 350 km/h" and shot_label(s[3]) == "ドライブ"
    hl = pick_fastest(s, 2)
    assert [h.t_hit_s for h in hl] == [5.0, 1.0] and hl[0].rank == 1 and "350" in hl[0].title
    assert hl[0].start_s == pytest.approx(4.0) and hl[0].end_s == pytest.approx(7.0)
    assert [h.shot_type for h in pick_fastest(s, 5, None)] == ["smash", "smash", "smash", "clear"]
    ra = [Rally(1, 1.0, 3.0, 2), Rally(2, 5.0, 8.0, 3)]
    lr = pick_longest_rallies(ra, s, 2)
    assert [h.rally_id for h in lr] == [2, 1] and lr[0].n_shots == 3
    assert lr[0].start_s == pytest.approx(4.0) and lr[0].end_s == pytest.approx(9.5)


def test_highlights_write_clips_and_joined_video(shots_dir, tmp_path):
    d, shots, truth = shots_dir
    info = write_highlights(HighlightOptions(d, tmp_path / "hl", n_fastest=2, shot_type=None, n_rallies=1,
                                             after_s=1.0), CFG, progress=lambda m: None, use_ffmpeg=HAS_FFMPEG)
    assert len(info["clips"]) == 3
    speeds = [c["speed_kmh"] for c in info["clips"] if c["kind"] == "fastest"]
    assert speeds == sorted([s.speed_kmh for s in shots], reverse=True)[:2]
    n = 0
    for c in info["clips"]:
        f = _frames(tmp_path / "hl" / c["file"])
        assert abs(len(f) - FPS * (c["end_s"] - c["start_s"])) <= 2
        n += len(f)
    assert abs(len(_frames(tmp_path / "hl" / "highlights.mp4")) - n) <= 3
    saved = json.loads((tmp_path / "hl" / "highlights.json").read_text(encoding="utf-8"))
    assert saved["joined"] == "highlights.mp4" and len(saved["clips"]) == 3


def test_highlights_fall_back_to_all_types_without_smashes(shots_dir, tmp_path):
    d, shots, truth = shots_dir
    msgs = []
    info = write_highlights(HighlightOptions(d, tmp_path / "hl", n_fastest=1, shot_type="net", after_s=0.5,
                                             join=False), CFG, progress=msgs.append, use_ffmpeg=HAS_FFMPEG)
    if any(s.shot_type == "net" for s in shots):
        assert info["clips"][0]["shot_type"] == "net"
    else:
        assert any("すべてのショット" in m for m in msgs) and len(info["clips"]) == 1
    assert info["joined"] is None


def test_cli_overlay(shots_dir, tmp_path):
    d, shots, truth = shots_dir
    out = tmp_path / "cli.mp4"
    assert cli_main(["overlay", str(d), "--out", str(out), "--start", "0", "--end", "0.5", "--no-audio"]) == 0
    assert out.exists() and len(_frames(out)) > 10
    # 一覧がないフォルダは、原因を出して 2 で終わる
    assert cli_main(["overlay", str(tmp_path)]) == 2


def test_gui_starts_overlay_and_highlights(shots_dir, tmp_path, monkeypatch):
    """ショットの一覧の画面の「動画に書き出す」：裏で ``overlay`` を動かすコマンドを組み、できた動画を一覧に出す。"""
    from smash_speed.webapp import server

    d, shots, truth = shots_dir
    monkeypatch.setattr(server, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(server, "S", server.Session())
    server.S.state = {"out_root": str(tmp_path / "out")}
    started = []
    monkeypatch.setattr(server, "start_task", lambda title, cmds: started.append((title, cmds[0])) or server.Task(title, []))
    client = server.app.test_client()
    assert not client.post("/api/run/overlay", json={}).get_json()["ok"]   # 一覧がない
    server.S.state["shots_dir"] = str(d)
    assert client.post("/api/run/overlay", json={"mode": "video", "start": 1, "end": 3}).get_json()["ok"]
    assert not client.post("/api/run/overlay", json={"mode": "video", "start": 3, "end": 1}).get_json()["ok"]
    assert client.post("/api/run/overlay", json={"mode": "highlights", "top": 3, "rallies": 1, "type": "all"}).get_json()["ok"]
    (_, a), (_, b) = started
    assert a[a.index("overlay") + 1] == str(d) and a[a.index("--start") + 1] == "1.000" and a[-1] == "3.000"
    assert a[a.index("--out") + 1].endswith("overlay.mp4")
    assert "--highlights" in b and b[b.index("--top") + 1] == "3" and b[b.index("--type") + 1] == "all"
    assert b[b.index("--rallies") + 1] == "1"
    # 書き出した動画は一覧の files に出る
    (d / "overlay.mp4").write_bytes(b"x")
    try:
        files = client.get("/api/shots").get_json()["files"]
        assert files["overlay"].endswith("overlay.mp4") and "highlights" not in files
    finally:
        (d / "overlay.mp4").unlink()
