"""ショットの種類の正解ラベル（ROADMAP B6）の保存と読み出しのテスト。"""

import math

import pytest

from smash_speed.shot_labels import (LABEL_COLUMNS, label_summary, match_labels, read_labels, set_label,
                                     video_key)
from smash_speed.shots import Shot


def _shots():
    return [Shot(t_hit_s=1.00, side="h", shot_type="clear", speed_kmh=150.0, hit_x_m=-5.0, hit_y_m=0.5, hit_z_m=2.6),
            Shot(t_hit_s=2.30, side="o", shot_type="smash", speed_kmh=280.0),
            Shot(t_hit_s=2.42, side="h", shot_type="unknown")]


def test_video_key_ignores_folder_and_case():
    assert video_key(r"C:\Users\x\Videos\IMG_4397.MOV") == "img_4397.mov"
    assert video_key("/home/u/img_4397.mov") == "img_4397.mov"


def test_set_replace_and_remove_label(tmp_path):
    f = tmp_path / "shot_labels.csv"
    assert read_labels(f) == []
    s = _shots()
    set_label(f, r"D:\v\IMG_1.mp4", s[0], "drop")
    labels = read_labels(f)
    assert len(labels) == 1
    r = labels[0]
    # 付けたときの値（特徴量）と決まりの判定もいっしょに残る
    assert (r.video_name, r.label, r.predicted, r.side) == ("img_1.mp4", "drop", "clear", "h")
    assert r.speed_kmh == 150.0 and r.hit_z_m == 2.6 and r.labeled_at
    assert f.read_text(encoding="utf-8-sig").splitlines()[0].split(",") == LABEL_COLUMNS
    # 同じショット（時刻が少しずれていても）に付け直すと置き換わる。別の動画は別のラベル
    set_label(f, "/other/IMG_1.mp4", Shot(t_hit_s=1.03, shot_type="clear"), "clear")
    set_label(f, "IMG_2.mp4", s[0], "lob")
    labels = read_labels(f)
    assert sorted((r.video_name, r.label) for r in labels) == [("img_1.mp4", "clear"), ("img_2.mp4", "lob")]
    # 空のラベルは消す
    set_label(f, "IMG_1.mp4", s[0], "")
    assert [r.video_name for r in read_labels(f)] == ["img_2.mp4"]
    with pytest.raises(ValueError):
        set_label(f, "IMG_1.mp4", s[0], "badminton")


def test_match_labels_after_reanalysis(tmp_path):
    """解析し直して打った時刻が少し変わっても、近いショットにラベルが付き直す。1 つのラベルは 1 本だけ。"""
    f = tmp_path / "shot_labels.csv"
    s = _shots()
    set_label(f, "IMG_1.mp4", s[1], "drive")
    set_label(f, "IMG_1.mp4", s[0], "drop")
    again = [Shot(t_hit_s=1.05), Shot(t_hit_s=2.34), Shot(t_hit_s=2.40), Shot(t_hit_s=5.0)]
    assert match_labels(read_labels(f), "C:/x/img_1.MP4", again) == {0: "drop", 1: "drive"}
    assert match_labels(read_labels(f), "IMG_9.mp4", again) == {}
    assert match_labels(read_labels(f), "IMG_1.mp4", [Shot(t_hit_s=1.5)]) == {}  # 離れすぎ


def test_label_summary(tmp_path):
    f = tmp_path / "shot_labels.csv"
    s = _shots()
    set_label(f, "IMG_1.mp4", s[0], "clear")
    set_label(f, "IMG_1.mp4", s[1], "drive")
    set_label(f, "IMG_2.mp4", s[1], "smash")
    m = label_summary(read_labels(f))
    assert m["n"] == 3 and m["n_videos"] == 2 and m["by_type"] == {"clear": 1, "drive": 1, "smash": 1}
    assert m["rule_accuracy"] == pytest.approx(2 / 3)
    assert math.isnan(label_summary([])["rule_accuracy"])
