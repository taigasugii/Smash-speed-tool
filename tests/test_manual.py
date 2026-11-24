"""使い方の手引き（docs/manual.md。ROADMAP v2 の L1）のテスト。"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_manual_mentions_every_menu_item_and_its_images_exist():
    """GUI の左のメニュー（static/core.js の NAV）の項目が、すべて手引きに「」つきで出てくる。写真のファイルもある。"""
    core = (ROOT / "smash_speed" / "webapp" / "static" / "core.js").read_text(encoding="utf-8")
    nav = core[core.index("const NAV = ["):core.index("];", core.index("const NAV = ["))]
    labels = re.findall(r'label: "([^"]+)"', nav)
    assert len(labels) >= 15
    manual = (ROOT / "docs" / "manual.md").read_text(encoding="utf-8")
    assert [x for x in labels if f"「{x}」" not in manual] == []
    for img in re.findall(r"\]\((manual_img/[^)]+)\)", manual):
        p = ROOT / "docs" / img
        assert p.exists(), img
        assert p.stat().st_size < 300_000, img      # 小さい PNG だけ
    assert "docs/manual.md" in (ROOT / "README.md").read_text(encoding="utf-8")
