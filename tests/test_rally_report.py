"""1 本の動画のレポート（rally_report.py）のテスト。"""

import json
from html.parser import HTMLParser

from smash_speed.config import load_config
from smash_speed.rally_report import REPORT_FILE, bars_html, build_report, main, write_report
from smash_speed.shots import Rally, Shot, write_rallies, write_shots

COURT = load_config()["court"]


def _shots():
    return [
        Shot(t_hit_s=0.0, side="h", shot_type="serve", speed_kmh=60.0, land_x_m=5.0, land_y_m=0.5, rally_id=1,
             index_in_rally=1),
        Shot(t_hit_s=1.5, side="o", shot_type="clear", speed_kmh=150.0, land_x_m=-6.0, land_y_m=-2.5, rally_id=1,
             index_in_rally=2),
        Shot(t_hit_s=3.0, side="h", shot_type="smash", speed_kmh=312.4, land_x_m=3.0, land_y_m=2.5, rally_id=1,
             index_in_rally=3),
        Shot(t_hit_s=10.0, side="h", shot_type="serve", speed_kmh=40.0, rally_id=2, index_in_rally=1),
        Shot(t_hit_s=11.0, side="o", shot_type="unknown", rally_id=2, index_in_rally=2),
    ]


def _rallies():
    return [Rally(1, 0.0, 4.0, 3, winner_side="h", end_reason="in"), Rally(2, 10.0, 12.0, 2)]


class _Tags(HTMLParser):
    """開いたタグと閉じたタグが釣り合っているか、外のファイルを読んでいないかを調べる。"""

    VOID = {"meta", "br", "img", "input", "link", "hr", "rect", "line", "circle", "path"}

    def __init__(self):
        super().__init__()
        self.stack, self.bad, self.external = [], [], []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        for k in ("src", "href"):
            if a.get(k):
                self.external.append(a[k])
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.bad.append(tag)
        else:
            self.stack.pop()


def test_report_is_a_self_contained_page():
    html = build_report(_shots(), _rallies(), {"video": "C:/videos/IMG_1.MOV", "recorded_at": "2026-10-20T09:30:00Z",
                                               "start_s": 20.0, "end_s": 80.0}, COURT)
    p = _Tags()
    p.feed(html)
    assert not p.bad and not p.stack and not p.external
    assert '<meta name="viewport"' in html and "prefers-color-scheme: dark" in html and "@media print" in html
    # 見出し・大きな数字・区間・撮った日時
    assert "IMG_1.MOV" in html and "2026-10-20 09:30" in html and "20.0〜80.0 秒" in html
    assert "312 km/h" in html and "スマッシュ・3.0 秒" in html
    # 割合は種類の分かった 4 本の中で（サーブ 2 本 = 50%）、分からないものは注記
    assert "2 本（50%）" in html and "種類が分からないショット 1 本" in html
    for section in ("ショットの種類の割合", "種類ごとの初速", "ラリーの長さ", "着地点", "速いショット", "ラリーの一覧"):
        assert f"<h2>{section}</h2>" in html
    assert html.count('class="lm"') == 3  # 着地点のある種類ごとに 1 面


def test_report_without_rallies_or_landings():
    html = build_report([Shot(t_hit_s=1.0, shot_type="unknown")], [], {}, COURT)
    assert "<h2>着地点</h2>" not in html and "<h2>ラリーの長さ</h2>" not in html
    assert "種類の分かったショットがありません" in html


def test_bars_scale_to_the_largest_value():
    html = bars_html([("a", 2.0, "2 本"), ("b", 1.0, "1 本")])
    assert "width:100.0%" in html and "width:50.0%" in html and 'title="b：1 本"' in html
    assert bars_html([]) == ""


def test_write_report_and_cli(tmp_path, capsys):
    write_shots(tmp_path / "shots.csv", _shots())
    write_rallies(tmp_path / "rallies.csv", _rallies())
    (tmp_path / "shot_list.json").write_text(json.dumps({"video": "/v/IMG_2.MOV"}), encoding="utf-8")
    assert write_report(tmp_path, COURT) == tmp_path / REPORT_FILE
    assert "IMG_2.MOV" in (tmp_path / REPORT_FILE).read_text(encoding="utf-8")
    (tmp_path / REPORT_FILE).unlink()
    main([str(tmp_path)])
    assert "書き出しました" in capsys.readouterr().out and (tmp_path / REPORT_FILE).exists()
