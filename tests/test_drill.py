"""基礎打ちの合成データ（synth_drill）のテスト。"""

import numpy as np
import pytest

from smash_speed.config import load_config
from smash_speed.physics import PhysicsParams
from smash_speed.synth_clutter import ClutterOptions
from smash_speed.synth_drill import (DRILLS, MAIN_DRILLS, DrillOptions, make_drill_scene, make_streak,
                                     observe_drill)
from smash_speed.synth_rally import DRILL_CAMERAS, named_rally_camera

CFG = load_config()
PHYS = PhysicsParams.from_config(CFG)
SMALL = DrillOptions(n_streaks=3, streak_hits=(4, 9), pause_s=(3.0, 10.0))


@pytest.fixture(scope="module")
def drive_scene():
    return make_drill_scene(np.random.default_rng(3), PHYS, "drive", SMALL)


def test_scene_is_reproducible():
    a = make_drill_scene(np.random.default_rng(5), PHYS, "clear", DrillOptions(n_streaks=2, streak_hits=(3, 5)))
    b = make_drill_scene(np.random.default_rng(5), PHYS, "clear", DrillOptions(n_streaks=2, streak_hits=(3, 5)))
    assert a.hits() and [(t, s) for t, _, s, _ in a.hits()] == [(t, s) for t, _, s, _ in b.hits()]


def test_unknown_drill_is_rejected():
    with pytest.raises(ValueError, match="知らない練習"):
        make_drill_scene(np.random.default_rng(0), PHYS, "badminton")


def test_truth_counts_match_the_scene(drive_scene):
    cam = named_rally_camera("side", CFG)
    ob = observe_drill(drive_scene, cam, 30.0, np.random.default_rng(1), DrillOptions(clutter=ClutterOptions.clean()))
    tr = ob.truth
    # 正解の打球は続いた打ち合いの本数の合計と同じで、1 本ごとに打球音が 1 つ
    assert len(tr.hits) == int(tr.streaks.n_hits.sum()) == len(ob.onsets)
    assert (tr.streaks.n_hits >= 3).all() and (tr.streaks.n_hits <= SMALL.streak_hits[1]).all()
    # 続いた打ち合いの中では打った側が交互で、1 本目は手前（h）
    for _, s in tr.streaks.iterrows():
        sides = tr.hits.side[(tr.hits.t >= s.t_start) & (tr.hits.t <= s.t_end)].tolist()
        assert sides[0] == "h" and all(a != b for a, b in zip(sides, sides[1:]))
        assert s.n_h + s.n_o == s.n_hits == len(sides)
    # とぎれの間は 3 秒以上（拾って球を出し直す）
    r = drive_scene.rallies
    assert all(b.t_start - a.t_end >= SMALL.pause_s[0] - 1e-9 for a, b in zip(r, r[1:]))
    assert tr.kind == "drive" and set(tr.streaks.end) <= {"in", "out", "net"}


def test_points_outside_the_oblique_camera_are_not_detected(drive_scene):
    assert "oblique" in DRILL_CAMERAS
    cam = named_rally_camera("oblique", CFG)
    ob = observe_drill(drive_scene, cam, 30.0, np.random.default_rng(2), DrillOptions(clutter=ClutterOptions.clean()))
    vis = ob.det[ob.det.visible > 0]
    assert len(vis) > 20
    assert vis.x_px.between(0, cam.width).all() and vis.y_px.between(0, cam.height).all()
    # シャトルが飛んでいても画面の外にあるコマは、検出に出ない
    f = ob.frames
    flying = np.isfinite(f.x_m.to_numpy())
    uv = cam.project(np.nan_to_num(f[["x_m", "y_m", "z_m"]].to_numpy()))
    outside = flying & ~((uv[:, 0] >= 0) & (uv[:, 0] < cam.width) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.height))
    assert outside.sum() > 0
    seen = set(vis.frame)
    assert not any(k in seen for k in f.frame[outside])


def test_near_player_is_cut_by_the_oblique_camera(drive_scene):
    cam = named_rally_camera("oblique", CFG)
    ob = observe_drill(drive_scene, cam, 30.0, np.random.default_rng(4), DrillOptions(clutter=ClutterOptions.clean()))
    p = ob.players
    assert (p.x0 >= 0).all() and (p.x1 <= cam.width).all() and (p.y0 >= 0).all() and (p.y1 <= cam.height).all()
    n = len(ob.frames)
    near, far = (p.side == "h").sum(), (p.side == "o").sum()
    assert far > 0.95 * n and near < 0.9 * far   # 奥の人は全身が写り、手前の人は画面の端で切れて外れるコマが出る


@pytest.mark.parametrize("kind,lo,hi", [("drive", 0.5, 0.8), ("smash_receive", 1.0, 1.6)])
def test_tempo_is_in_the_expected_range(kind, lo, hi):
    o = DrillOptions(n_streaks=4, streak_hits=(6, 10))
    tempo = [make_drill_scene(np.random.default_rng(s), PHYS, kind, o) for s in (11, 12)]
    from smash_speed.synth_drill import drill_truth
    iv = np.concatenate([drill_truth(sc).intervals() for sc in tempo])
    assert len(iv) > 20
    assert lo <= float(iv.mean()) <= hi


def test_fatigue_slows_the_tempo():
    from smash_speed.synth_drill import drill_truth

    def last_over_first(fatigue):
        o = DrillOptions(n_streaks=3, streak_hits=(8, 10), fatigue=fatigue, pause_s=(3.0, 3.5))
        tr = drill_truth(make_drill_scene(np.random.default_rng(21), PHYS, "drive", o))
        per = [np.diff(tr.hits.t[(tr.hits.t >= s.t_start) & (tr.hits.t <= s.t_end)]).mean()
               for _, s in tr.streaks.iterrows()]
        return per[-1] / per[0]

    assert last_over_first(0.8) > 1.2        # 疲れると、最後の打ち合いは 2 割以上遅い
    assert last_over_first(0.0) < 1.2        # 疲れなしでもばらつきで少しは違うが、そこまでは変わらない


def test_every_main_drill_has_a_pattern():
    assert set(MAIN_DRILLS) <= set(DRILLS)
    for kind in MAIN_DRILLS:
        s = make_streak(np.random.default_rng(0), PHYS, kind, 3)
        assert len(s.segments) >= 3 and s.segments[0].side == "h"
