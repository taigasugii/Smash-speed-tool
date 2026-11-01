"""角・低いカメラでの短いショットの当てはめ（shot_fit.py の plausible_start・hit_point_options・choose_hit_points）のテスト。"""

from types import SimpleNamespace

import numpy as np
import pytest

from smash_speed.config import load_config
from smash_speed.physics import MPS_TO_KMH, FitOptions, PhysicsParams
from smash_speed.shot_fit import (ShotFit, choose_hit_points, hit_point_options, initial_fit, joint_hit_points,
                                  plausible_start, split_observations)
from smash_speed.synth_rally import (RALLY_CAMERAS, SyntheticRally, _segment, corner_rally_camera,
                                     named_rally_camera, observe_rally)

PHYS = PhysicsParams()


def test_named_rally_cameras_see_the_court():
    """名前で選べる合成のカメラは、どれも奥のコートの角とネットの上を画面の中に写す（低いカメラは手前の角が切れる）。"""
    cfg = load_config()
    pts = np.array([[-6.7, 3.05, 0.0], [6.7, 3.05, 0.0], [0.0, 0.0, 1.55], [0.0, 3.05, 1.55]])
    for name in RALLY_CAMERAS:
        cam = named_rally_camera(name, cfg)
        uv = cam.project(pts)
        assert ((uv[:, 0] >= 0) & (uv[:, 0] < cam.width) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.height)).all(), name
    with pytest.raises(ValueError):
        named_rally_camera("top", cfg)


def test_plausible_start():
    """打点が自分の側にあり、相手側へ向かう打ち出しだけを、ありえるとする。"""
    v = np.array([10.0, 2.0, 3.0])
    assert plausible_start(np.array([-3.0, 0.5, 1.0]), v, "h")
    assert not plausible_start(np.array([-3.0, 0.5, 1.0]), -v, "h")          # 自分の側へ飛ぶ
    assert not plausible_start(np.array([2.5, 0.5, 1.0]), v, "h")            # 相手の側から打つ
    assert plausible_start(np.array([0.5, 0.5, 1.0]), v, "h")                # ネットの少し向こう（はみ出し）
    assert not plausible_start(np.array([-3.0, 0.5, 1.0]), np.array([0.5, 9.0, 3.0]), "h")  # ほぼ横へ飛ぶ
    assert plausible_start(np.array([3.0, 0.5, 1.0]), np.array([-10.0, 1.0, 3.0]), "o")
    assert plausible_start(np.array([3.0, 0.5, 1.0]), v, "")                 # 打った側が分からない
    assert not plausible_start(np.array([12.0, 0.5, 1.0]), v, "")            # コートから遠い
    assert not plausible_start(np.array([np.nan, 0.5, 1.0]), v, "h")


def _corner_serve():
    """角のカメラで、ネットの反対側から逆向きに飛ぶ軌道と見分けにくい、斜めのショートサーブ（30 fps）。"""
    seg = _segment("serve_short", np.array([2.29, 2.11, 0.97]), np.array([-9.9, -8.0, 4.8]), 0.0, PHYS)
    cam = corner_rally_camera()
    det = observe_rally(SyntheticRally([seg]), cam, 30.0, PHYS, np.random.default_rng(1), dropout=0.1)
    return seg, cam, det


def test_initial_fit_rejects_a_flight_from_the_wrong_side():
    """いちばん合う格子から大きく動かすと、打ったのと反対の側から逆向きに飛ぶ軌道になるサーブでも、
    打った側から見てありえる軌道を返す。"""
    seg, cam, det = _corner_serve()
    obs = split_observations(det, cam, [seg.t_hit])[0]
    f = initial_fit(obs, seg.t_hit, seg.side, PHYS, FitOptions())
    assert f is not None
    assert plausible_start(f.p0, f.v0, seg.side)
    assert f.p0[0] > 0.5                                       # 打った側（o は x > 0）
    assert abs(f.speed0 / seg.speed0 - 1) < 0.6                # 正しい側の軌道（逆向きの軌道は 14 km/h ほど）


def _fake_fit(p0, v0, rms, n, dpsi=0.0, dn=0.0):
    return SimpleNamespace(p0=np.asarray(p0, float), v0=np.asarray(v0, float), rms_px=rms, n_obs=n, mode="plane",
                           theta=np.array([0.0, 0.0, 0.0, 0.0, dpsi, dn]))


def test_choose_hit_points_uses_both_shots():
    """打点の候補が 2 つあるときは、前のショットの終わりと次のショットの始まりの両方のずれで選ぶ。

    次のショットだけなら候補 1 がわずかに合うが、前のショットは候補 1 を終わりにすると平面を大きく回すので、
    候補 0 にする。前のショットの罰がなければ候補 1。
    """
    fits = [ShotFit(0.0, "h", 1.0), ShotFit(1.0, "o", 1.3)]
    pts = [[np.array([-3.0, 0.0, 1.0])], [np.array([4.0, 0.0, 2.5]), np.array([2.0, -2.5, 2.8])],
           [np.array([-2.0, 0.0, 0.0])]]
    v_h, v_o = [10.0, 0.0, 3.0], [-30.0, 0.0, -5.0]
    res = {(0, 0, 0): _fake_fit(pts[0][0], v_h, 1.4, 40),
           (0, 0, 1): _fake_fit(pts[0][0], v_h, 1.4, 40, dpsi=0.7),
           (1, 0, 0): _fake_fit(pts[1][0], v_o, 1.71, 11),
           (1, 1, 0): _fake_fit(pts[1][1], v_o, 1.69, 11)}
    opts = FitOptions(refine_plane=True)
    assert choose_hit_points(fits, pts, res, opts) == [0, 0, 0]
    res[(0, 0, 1)] = _fake_fit(pts[0][0], v_h, 1.4, 40)
    assert choose_hit_points(fits, pts, res, opts) == [0, 1, 0]
    # 打った側から見てありえない当てはめの候補は選ばない
    res[(1, 1, 0)] = _fake_fit(pts[1][1], [30.0, 0.0, -5.0], 1.0, 11)
    assert choose_hit_points(fits, pts, res, opts) == [0, 0, 0]


def test_hit_point_options_drop_points_on_the_wrong_side():
    """打点の候補は、打った側から見てありえない所（相手側）を外し、離れた 2 つは両方残す。"""
    def fake(p0, t_end):
        sf = ShotFit(0.0, "o", t_end)
        # x の負の向きへ飛ぶ軌道（打点 p0、終わりは t_end の位置）
        sf.fit = SimpleNamespace(p0=np.asarray(p0, float), v0=np.array([-5.0, 0.0, 3.0]), t_ref=0.0,
                                 phys=lambda phys: phys)
        return sf

    a, b = fake([3.0, 0.0, 1.5], 0.6), fake([-1.0, 0.0, 1.5], 1.2)
    b.t_hit, b.side, b.fit.t_ref = 0.6, "h", 0.6
    opts = hit_point_options([a, b], PHYS)
    assert len(opts) == 3 and len(opts[0]) == 1
    # 打点 1 の候補：a の終わり（x ≈ 0、h の側として ありえる）と b の始まり（x = −1）。近いので平均の 1 つ
    assert len(opts[1]) == 1 and opts[1][0][0] < 0.5
    assert joint_hit_points([a, b], PHYS)[1] is not None
    # b の始まりを相手側（x = 3）に置くと、その候補は外れて a の終わりだけになる
    b.fit.p0 = np.array([3.0, 0.0, 1.5])
    opts = hit_point_options([a, b], PHYS)
    assert len(opts[1]) == 1 and opts[1][0][0] < 1.0
