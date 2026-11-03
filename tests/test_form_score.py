"""お手本との類似度スコア（form_score.py、E5）のテスト。合成のスイングの角度・タイミングを変えて比べる。"""

import numpy as np
import pytest

from smash_speed.form_score import (
    Reference, dtw, main, make_reference, reference_from, reference_sensitivity, score, score_dir,
)
from smash_speed.pose_prep import normalize
from smash_speed.synth_pose import SwingParams, project_swing, side_camera, swing_times, vary

P = SwingParams()


def _pose(p=P, fps=60.0, time_scale=1.0, noise_px=0.0, seed=0, shot_id=""):
    seq = project_swing(p, side_camera(), swing_times(fps, 1.5, 0.8), noise_px=noise_px,
                        rng=np.random.default_rng(seed))
    seq.t = seq.t * time_scale
    n = normalize(seq.slice(-1.3, 0.8), 0.0)
    n.meta["shot_id"] = shot_id
    return n


REF = reference_from(_pose(), "base")


def test_dtw_aligns_a_shifted_series():
    a = np.sin(np.linspace(0, 6, 80))[:, None]
    b = np.sin(np.linspace(0, 6, 80) - 0.3)[:, None]
    d, path = dtw(a, b)
    assert path[0] == (0, 0) and path[-1] == (79, 79)
    assert d < np.sqrt(np.mean((a - b) ** 2)) * 0.5
    assert dtw(a, a)[0] == 0.0


def test_same_swing_scores_100_regardless_of_fps_and_size():
    for n in (_pose(), _pose(fps=120.0), _pose(vary(P, height_m=1.85))):
        s = score(n, REF)
        assert s.score_dtw > 98.5 and s.score_cos > 99.5


def test_score_falls_with_the_size_of_the_change():
    scores = [score(_pose(vary(P, elbow_impact_deg=e)), REF).score_dtw for e in (160, 150, 140, 120)]
    assert scores[0] > scores[1] > scores[2] > scores[3]
    assert scores[3] < 80
    cos = [score(_pose(vary(P, elbow_impact_deg=e)), REF).score_cos for e in (160, 140, 120)]
    assert cos[0] > cos[1] > cos[2]


def test_dtw_forgives_timing_but_not_shape():
    """10% ゆっくりのスイング（形は同じ）は、時間をずらして合わせると高い点。時間をずらさない比べ方では差が出る。"""
    slow = score(_pose(time_scale=1.1), REF)
    assert slow.score_dtw > 97 and slow.lockstep_deg > 3 * slow.dtw_deg
    bent = score(_pose(vary(P, elbow_impact_deg=130)), REF)
    assert bent.score_dtw < slow.score_dtw - 10


def test_small_noise_keeps_a_high_score():
    assert score(_pose(noise_px=1.0, seed=2), REF).score_dtw > 85


def test_average_reference_and_sensitivity(tmp_path):
    """お手本を差し替えると点は変わる。平均のお手本は、元のどのショットにも近い。"""
    group = [_pose(vary(P, elbow_impact_deg=e), shot_id=f"s{e}") for e in (150, 160, 170)]
    avg = make_reference(group, "avg")
    assert avg.sources == ["s150", "s160", "s170"]
    assert all(score(n, avg).score_dtw > 90 for n in group)
    path = avg.save(tmp_path / "avg.npz")
    back = Reference.load(path)
    assert back.name == "avg" and np.allclose(back.angles, avg.angles, equal_nan=True)
    with pytest.raises(ValueError):
        make_reference([])
    # お手本によって点の付き方が変わる（伸ばした肘のお手本では、伸ばしたショットほど点が高い）
    straight = reference_from(_pose(vary(P, elbow_impact_deg=175)), "straight")
    bent = reference_from(_pose(vary(P, elbow_impact_deg=125)), "bent")
    shots = {f"e{e}": _pose(vary(P, elbow_impact_deg=e)) for e in (125, 140, 155, 170)}
    table = {r.name: {k: score(n, r).score_dtw for k, n in shots.items()} for r in (straight, bent, REF)}
    sens = reference_sensitivity(table)
    assert sens["n_shots"] == 4
    assert sens["rank_correlation"]["straight / bent"] == pytest.approx(-1.0)
    assert set(sens["per_reference"]) == {"straight", "bent", "base"}


def test_score_dir_and_cli(tmp_path):
    form = tmp_path / "out" / "form"
    for e in (140, 160):
        _pose(vary(P, elbow_impact_deg=e), shot_id=f"v_{e}").save(form / "poses" / f"v_{e}.npz")
    rows, sens = score_dir(form, [REF])
    assert [r["shot_id"] for r in rows] == ["v_140", "v_160"] and rows[1]["score_dtw"] > rows[0]["score_dtw"]
    ref = tmp_path / "ref.npz"
    assert main(["make-ref", str(ref), str(form / "poses" / "v_160.npz")]) == 0
    assert main(["score", str(tmp_path / "out"), "--ref", str(ref)]) == 0
    text = (form / "scores.csv").read_text(encoding="utf-8-sig")
    assert text.splitlines()[0].startswith("shot_id,reference,score_dtw,score_cos")
    assert (form / "scores_sensitivity.json").exists()
