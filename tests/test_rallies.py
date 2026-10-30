"""ラリーの区切り（rallies.py）と、続けて打った合成ラリー（synth_rally.make_session）のテスト。"""

import numpy as np
import pandas as pd
import pytest

from smash_speed.config import load_config
from smash_speed.physics import PhysicsParams
from smash_speed.rallies import (REASON_END, REASON_GAP, REASON_LANDED, RallyOptions, rally_outcome, rally_scores,
                                 split_rallies, synthetic_benchmark, visible_fraction)
from smash_speed.shots import Rally, Shot
from smash_speed.synth_rally import make_session, observe_session, rally_camera

CFG = load_config()
PHYS = PhysicsParams.from_config(CFG)


def _shot(t, side="h", t_land=None, land=(None, None), ok=True):
    sign = -1.0 if side == "h" else 1.0
    return Shot(t_hit_s=t, side=side, speed_kmh=100.0 if ok else None, hit_x_m=sign * 3.0, t_land_s=t_land,
                land_x_m=land[0], land_y_m=land[1])


def _det(t0, t1, fps=30.0, visible=lambda t: False):
    t = np.arange(t0, t1, 1 / fps)
    return pd.DataFrame({"timestamp": t, "visible": [int(visible(x)) for x in t]})


def test_landing_splits_a_short_gap_and_interception_does_not():
    # 1 本目は 2 本目より前に床に着く（打ち返されていない）→ 2.2 秒しか空いていなくても区切る
    # 2 本目は 3 本目の後に床に着く（打ち返された）→ 同じラリー
    shots = [_shot(0.0, "h", t_land=1.0, land=(4.0, 1.0)), _shot(2.2, "o", t_land=3.6), _shot(3.5, "h", t_land=4.4,
                                                                                         land=(3.0, -1.0))]
    rallies = split_rallies(shots)
    assert [(r.rally_id, r.n_shots) for r in rallies] == [(1, 1), (2, 2)]
    assert [s.rally_id for s in shots] == [1, 2, 2] and [s.index_in_rally for s in shots] == [1, 1, 2]
    assert rallies[0].t_end_s == 1.0 and rallies[0].note == f"終わり：{REASON_LANDED}"
    assert rallies[1].t_end_s == 4.4 and rallies[1].note == f"終わり：{REASON_END}"
    # どちらも相手のコートの中に落ちた → 最後に打った側の得点
    assert (rallies[0].end_reason, rallies[0].winner_side) == ("in", "h")
    assert (rallies[1].end_reason, rallies[1].winner_side) == ("in", "h")


def test_failed_fit_landing_is_not_used():
    # 当てはめに失敗したショット（初速なし）の着地の時刻は、打球を見落とした軌道かもしれないので使わない
    shots = [_shot(0.0, t_land=1.0, ok=False), _shot(2.2, "o")]
    assert len(split_rallies(shots)) == 1


def test_long_gap_splits_only_when_the_shuttle_is_not_flying():
    shots = [_shot(0.0), _shot(4.0, "o")]
    # 間ずっとシャトルが見えている → 打球を見落としただけとみなして同じラリー
    flying = _det(-0.5, 5.0, visible=lambda t: True)
    assert len(split_rallies([Shot(**vars(s)) for s in shots], flying)) == 1
    # 見えていない（検出がない）→ 区切る
    still = _det(-0.5, 5.0, visible=lambda t: t < 0.5)
    rallies = split_rallies([Shot(**vars(s)) for s in shots], still)
    assert len(rallies) == 2 and rallies[0].note == f"終わり：{REASON_GAP}"
    # 最後に見えた時刻をラリーの終わりにする（着地が分からないとき）
    assert rallies[0].t_end_s == pytest.approx(0.5, abs=0.04)
    assert rallies[0].end_reason == "" and rallies[0].winner_side == ""
    # 検出がなければ、今までどおり間隔だけで区切る
    assert len(split_rallies([Shot(**vars(s)) for s in shots])) == 2
    # とても長い間は、シャトルが見えていても区切る
    far = [_shot(0.0), _shot(9.0, "o")]
    assert len(split_rallies(far, _det(-0.5, 10.0, visible=lambda t: True))) == 2


def test_visible_fraction():
    vis = np.arange(0.0, 10.0, 1 / 30)
    assert visible_fraction(vis, 1 / 30, 1.0, 3.0) == pytest.approx(1.0, abs=0.05)
    assert visible_fraction(vis[vis < 2.0], 1 / 30, 1.0, 3.0) == pytest.approx(0.5, abs=0.05)
    assert visible_fraction(vis, None, 1.0, 3.0) is None


@pytest.mark.parametrize("land, singles, expected", [
    ((3.0, 1.0), False, ("in", "h", "")),
    ((7.2, 0.0), False, ("out", "o", "")),
    ((3.0, 2.8), False, ("in", "h", "")),       # ダブルスなら中
    ((3.0, 2.8), True, ("out", "o", "")),       # シングルスなら外
    ((6.65, 0.0), False, ("in", "h", "線の近く")),
    ((-0.5, 0.0), False, ("net", "o", "")),     # ネットを越えずに自分の側に落ちた
])
def test_rally_outcome(land, singles, expected):
    s = _shot(0.0, "h", t_land=1.0, land=land)
    assert rally_outcome(s, RallyOptions(singles=singles), CFG["court"]) == expected


def test_rally_outcome_unknown_without_landing():
    assert rally_outcome(_shot(0.0, "h"), RallyOptions()) == ("", "", "")
    assert rally_outcome(_shot(0.0, "h", t_land=1.0, land=(3.0, 0.0), ok=False), RallyOptions()) == ("", "", "")


def test_rally_scores_counts_boundaries_and_exact_rallies():
    truth = [Shot(t_hit_s=t, rally_id=r) for t, r in ((0.0, 1), (1.0, 1), (2.0, 2), (3.0, 2), (4.0, 3))]
    true_r = [Rally(1, 0.0, 1.5, winner_side="h"), Rally(2, 2.0, 3.5, winner_side="o"), Rally(3, 4.0, 4.5)]
    # 2 と 3 のラリーをまとめてしまった推定（区切りを 1 つ見落とした）
    pred = [Shot(t_hit_s=t + 0.01, rally_id=r) for t, r in ((0.0, 1), (1.0, 1), (2.0, 2), (3.0, 2), (4.0, 2))]
    pred_r = [Rally(1, 0.0, 1.6, winner_side="h"), Rally(2, 2.0, 4.5)]
    s = rally_scores(pred, pred_r, truth, true_r)
    assert (s["boundary_tp"], s["boundary_fp"], s["boundary_fn"]) == (1, 0, 1)
    assert s["exact"] == pytest.approx(1 / 3)
    assert s["end_err_s"] == pytest.approx(0.1)
    assert s["outcome_acc"] == 1.0 and s["n_outcome"] == 1


def test_session_rallies_follow_each_other():
    rng = np.random.default_rng(0)
    session = make_session(rng, PHYS, n_rallies=4, rest_s=(1.0, 2.0))
    assert len(session) == 4
    for a, b in zip(session[:-1], session[1:]):
        assert 1.0 <= b.t_start - a.t_end <= 2.0
        assert b.segments[0].side == a.segments[-1].side  # 得点した側がサーブ
        assert b.segments[0].shot_type == "serve"
    det = observe_session(session, rally_camera(CFG), 30.0, PHYS, np.random.default_rng(1))
    assert set(det.rally_index.unique()) == {-1, 0, 1, 2, 3}
    # 休みの間は見えない
    rest = (det.timestamp > session[0].t_end + 0.05) & (det.timestamp < session[1].t_start - 0.05)
    assert rest.any() and not det.visible[rest].any()
    assert np.allclose(np.diff(det.timestamp), 1 / 30)


def test_truth_benchmark_beats_gap_only_with_short_rests():
    # 休みが短い（1〜2 秒）と、打球の間隔だけでは多くのラリーがつながってしまう
    r = synthetic_benchmark(n_sessions=2, n_rallies=5, fps=30.0, seed=0, mode="truth", rest_s=(1.0, 2.0))
    assert r["new"]["boundary_recall"] >= 0.95 and r["new"]["boundary_precision"] >= 0.95
    assert r["new"]["exact"] >= 0.9
    assert r["gap"]["boundary_recall"] < r["new"]["boundary_recall"]
    assert r["new"]["outcome_acc"] >= 0.9


def test_fit_benchmark_on_a_short_session():
    # 動画全体の検出 → 打球 → 当てはめ → 区切り（実際の流れと同じ）
    r = synthetic_benchmark(n_sessions=1, n_rallies=3, fps=30.0, seed=1, mode="fit", rest_s=(1.0, 2.0), workers=2)
    assert r["new"]["boundary_recall"] == 1.0 and r["new"]["boundary_precision"] == 1.0
    assert r["new"]["exact"] == 1.0 and r["new"]["outcome_acc"] >= 0.6
