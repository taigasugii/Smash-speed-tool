"""Phase 0：設定・CLI・再現性まわりの骨組みのテスト。"""

import numpy as np
import pytest

from smash_speed import cli
from smash_speed.config import load_config, save_config
from smash_speed.repro import run_metadata, set_seed


@pytest.fixture(scope="module")
def config():
    return load_config()


def test_landmarks_match_court_dimensions(config):
    """ランドマーク座標が court の寸法だけから成り、名前が重複しないこと。"""
    court = config["court"]
    allowed_x = {0.0, court["short_service_m"], court["long_service_doubles_m"], court["half_length_m"]}
    allowed_y = {0.0, court["half_width_singles_m"], court["half_width_doubles_m"]}
    allowed_z = {0.0, court["net_height_center_m"], court["net_height_post_m"]}

    names = [lm["name"] for lm in config["landmarks"]]
    assert len(names) == len(set(names))

    for lm in config["landmarks"]:
        x, y, z = lm["xyz"]
        assert abs(x) in allowed_x, lm
        assert abs(y) in allowed_y, lm
        assert z in allowed_z, lm
        # 命名規則（h=打者側 x<0 / o=相手側 x>0、n=y<0 / p=y>0）と符号が一致すること
        parts = lm["name"].split("_")
        if "h" in parts:
            assert x < 0, lm
        if "o" in parts:
            assert x > 0, lm
        if parts[-1] == "n":
            assert y < 0, lm
        if parts[-1] == "p":
            assert y > 0, lm


def test_user_config_overrides_default(tmp_path, config):
    override = tmp_path / "my.yaml"
    override.write_text("physics:\n  terminal_velocity_mps: 7.0\n", encoding="utf-8")
    merged = load_config(override)
    assert merged["physics"]["terminal_velocity_mps"] == 7.0
    assert merged["physics"]["g_mps2"] == config["physics"]["g_mps2"]  # 他の値は残る

    save_config(merged, tmp_path / "out" / "used.yaml")
    assert load_config(tmp_path / "out" / "used.yaml") == merged


def test_set_seed_is_reproducible():
    a = set_seed(0).normal(size=5)
    b = set_seed(0).normal(size=5)
    np.testing.assert_array_equal(a, b)


def test_run_metadata_has_version():
    meta = run_metadata({"mode": "plane"})
    assert meta["smash_speed_version"]
    assert meta["mode"] == "plane"


def test_cli_lists_all_subcommands(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for name in cli.SUBCOMMANDS:
        assert name in out


def test_cli_missing_required_argument_exits_2():
    with pytest.raises(SystemExit) as exc:
        cli.main(["analyze"])
    assert exc.value.code == 2
