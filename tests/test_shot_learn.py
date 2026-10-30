"""ショットの種類の判定（学習、ROADMAP B7）のテスト。"""

import json

import numpy as np
import pandas as pd
import pytest

from smash_speed.shot_labels import ShotLabel, read_labels, write_labels
from smash_speed.shot_learn import (
    FEATURE_NAMES, ShotTypeModel, compare_with_rules, feature_vector, fit_model, label_table, load_model, main,
    save_model, synthetic_labels,
)
from smash_speed.shot_types import RuleThresholds, ShotFeatures, classify_features


def _random_features(rng, n):
    """手がかりの範囲から一様に引いた ShotFeatures（軌道の計算をしないので速い）。"""
    out = []
    for _ in range(n):
        hit_z = rng.uniform(0.2, 3.1)
        out.append(ShotFeatures(speed_mps=rng.uniform(1, 110), elevation_deg=rng.uniform(-35, 65), hit_z_m=hit_z,
                                hit_depth_m=rng.uniform(0.3, 6.6), apex_z_m=hit_z + rng.uniform(0, 4),
                                net_z_m=rng.uniform(1.6, 5) if rng.uniform() < 0.9 else np.nan,
                                land_depth_m=rng.uniform(-1, 7), first_in_rally=bool(rng.uniform() < 0.1)))
    return out


# 実動画で打点が低く・初速が遅く出るときに、正解に合うしきい値（決まりの既定値とずれている）
SHIFTED = RuleThresholds(overhead_z_m=1.6, smash_speed_mps=30.0, clear_land_m=3.2, net_land_m=2.2,
                         drive_speed_mps=16.0)


def _table(feats, labels, group_size=8):
    X = np.array([feature_vector(f) for f in feats])
    return pd.DataFrame({"video_name": "v.mp4", "t_hit_s": np.arange(len(feats), dtype=float),
                         "group": [f"v#{i // group_size}" for i in range(len(feats))], "true": labels,
                         "rule": [classify_features(f) for f in feats],
                         **{k: X[:, j] for j, k in enumerate(FEATURE_NAMES)}})


def test_model_learns_shifted_thresholds():
    """正解がずれたしきい値で決まっているとき、学習した分類器は既定の決まりより正しく分ける。"""
    rng = np.random.default_rng(0)
    train, test = _random_features(rng, 800), _random_features(rng, 400)
    y_train = [classify_features(f, SHIFTED) for f in train]
    y_test = [classify_features(f, SHIFTED) for f in test]
    model = fit_model(np.array([feature_vector(f) for f in train]), y_train)
    pred = model.predict(np.array([feature_vector(f) for f in test]))
    acc_model = np.mean([p == t for p, t in zip(pred, y_test)])
    acc_rule = np.mean([classify_features(f) == t for f, t in zip(test, y_test)])
    assert acc_model >= 0.9
    assert acc_model > acc_rule + 0.1
    # 1 本ずつの判定も同じ
    assert model.classify(test[0]) == pred[0]
    p = model.predict_proba(np.array([feature_vector(f) for f in test[:5]]))
    assert p.shape == (5, len(model.classes)) and np.allclose(p.sum(axis=1), 1.0)


def test_model_only_predicts_seen_classes():
    rng = np.random.default_rng(1)
    feats = _random_features(rng, 60)
    y = ["smash" if f.speed_mps > 50 else "drop" for f in feats]
    model = fit_model(np.array([feature_vector(f) for f in feats]), y)
    assert model.classes == ["smash", "drop"]
    assert set(model.predict(np.array([feature_vector(f) for f in feats]))) <= {"smash", "drop"}
    # 1 種類だけなら、いつもその種類
    one = fit_model(np.array([feature_vector(f) for f in feats]), ["lob"] * len(feats))
    assert set(one.predict(np.array([feature_vector(f) for f in feats]))) == {"lob"}
    with pytest.raises(ValueError):
        fit_model(np.zeros((0, len(FEATURE_NAMES))), [])


def test_save_and_load_model(tmp_path):
    rng = np.random.default_rng(2)
    feats = _random_features(rng, 200)
    X = np.array([feature_vector(f) for f in feats])
    model = fit_model(X, [classify_features(f, SHIFTED) for f in feats])
    path = save_model(tmp_path / "m" / "shot_model.json", model)
    again = load_model(path)
    assert again.predict(X) == model.predict(X)
    assert again.n_train == 200
    d = json.loads(path.read_text(encoding="utf-8"))
    d["feature_names"] = ["speed_mps"]
    with pytest.raises(ValueError):
        ShotTypeModel.from_json(d)


def test_compare_with_rules_keeps_rallies_together():
    """ラリーの組ごとに外して判定する。正解がずれたしきい値なら、学習の方が正解率が高い。"""
    rng = np.random.default_rng(3)
    feats = _random_features(rng, 600)
    table = _table(feats, [classify_features(f, SHIFTED) for f in feats])
    r = compare_with_rules(table, k=5, seed=0)
    assert r["n"] == 600 and r["k"] == 5
    assert r["model"]["scores"]["accuracy"] > r["rule"]["scores"]["accuracy"] + 0.1
    assert int(r["model"]["matrix"].values.sum()) == 600
    assert set(r["rows"]["model"]) <= set(table["true"])
    # 正解が決まりそのものなら、決まりは全部合う
    same = compare_with_rules(_table(feats, [classify_features(f) for f in feats]), k=3)
    assert same["rule"]["scores"]["accuracy"] == 1.0
    with pytest.raises(ValueError):
        compare_with_rules(table.iloc[:0])


def test_synthetic_labels_round_trip(tmp_path, capsys):
    """合成ラベルを shot_labels.csv に書いて読み直し、表にする（打点・速度のないラベルと知らない種類は外す）。"""
    labels = synthetic_labels(n_rallies=3, seed=0)
    assert len(labels) >= 12
    assert {r.rally_id for r in labels} == {0, 1, 2}
    assert all(r.index_in_rally >= 1 and r.v0_x_mps is not None for r in labels)
    assert [r.label for r in labels if r.index_in_rally == 1] == ["serve"] * 3
    extra = [ShotLabel(video_name="x.mp4", t_hit_s=1.0, label="smash"),
             ShotLabel(video_name="x.mp4", t_hit_s=2.0, label="unknown", hit_x_m=-3.0, hit_y_m=0.0, hit_z_m=2.5,
                       v0_x_mps=30.0, v0_y_mps=0.0, v0_z_mps=-5.0)]
    path = write_labels(tmp_path / "shot_labels.csv", labels + extra)
    table = label_table(read_labels(path))
    assert len(table) == len(labels)
    # ラベルに残した決まりの判定と、表で出し直した判定は同じ
    by_t = {(r.video_name, r.t_hit_s): r.predicted for r in labels}
    assert all(by_t[(v, t)] == rule for v, t, rule in zip(table.video_name, table.t_hit_s, table.rule))
    assert table["group"].nunique() == 3
    # コマンド：比べて、モデルを書く
    main([str(path), "-k", "2", "--out", str(tmp_path / "shot_model.json")])
    out = capsys.readouterr().out
    assert "ルール：正解率" in out and "学習：正解率" in out and "200 本より少ない" in out
    assert load_model(tmp_path / "shot_model.json").n_train == len(labels)
