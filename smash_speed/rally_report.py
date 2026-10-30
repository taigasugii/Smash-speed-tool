"""1 本の動画の統計を、スマホでも見やすい 1 枚のページ（HTML）にまとめる（ROADMAP C3）。

ショットの一覧の出力先（``shot_list.analyze_video`` の ``shots.csv``・``rallies.csv``・``shot_list.json``）から、
``rally_stats`` の統計を出し、``report.html`` に書き出す。ページは 1 つのファイルで完結する（外のファイル・
ネットを使わない。メールで送る・スマホに入れるだけで見られる）。PDF は、ブラウザの「印刷」→「PDF に保存」で作る
（印刷用の見た目にしてある）。

ページの中身（上から）：

1. 動画の名前・撮った日時・解析した区間
2. 大きな数字：ラリーの数・ショットの数・いちばん速い初速（種類）・ラリーの平均の長さ
3. ショットの種類の割合（横の棒）と、種類ごとの本数・平均と最高の初速の表（打った側ごとにも）
4. ラリーの長さ（本数）の分布（棒）
5. 着地点の分布図（``rally_stats.landing_svg``）
6. 速いショット 10 本と、ラリーの一覧の表

``python -m smash_speed.rally_report <ショットの一覧の出力先>`` で書き出す（``shot_list.analyze_video`` も最後に書き出す）。
"""

from __future__ import annotations

import json
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

from smash_speed.rally_stats import END_REASON_LABELS, landing_svg, landing_types, rally_stats
from smash_speed.shots import RALLIES_FILE, SHOT_TYPE_LABELS, SHOTS_FILE, Rally, Shot, read_rallies, read_shots

REPORT_FILE = "report.html"
SHOT_LIST_FILE = "shot_list.json"   # shot_list.SHOT_LIST_FILE と同じ（読み込みの循環を避けて、ここにも書く）
SIDE_NAMES = {"h": "h の側", "o": "o の側", "": "分からない"}  # 手前・奥はカメラの位置で変わるので書かない

CSS = """
:root { color-scheme: light; --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --bar: #2a78d6; --border: rgba(11,11,11,0.10); }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { color-scheme: dark; --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec;
    --ink2: #c3c2b7; --grid: #2c2c2a; --axis: #383835; --bar: #3987e5; --border: rgba(255,255,255,0.10); }
}
:root[data-theme="dark"] { color-scheme: dark; --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec;
  --ink2: #c3c2b7; --grid: #2c2c2a; --axis: #383835; --bar: #3987e5; --border: rgba(255,255,255,0.10); }
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", "Yu Gothic UI", "Hiragino Sans", sans-serif; line-height: 1.5; }
main { max-width: 760px; margin: 0 auto; padding: 16px; }
h1 { font-size: 1.35rem; margin: 0.2rem 0 0.1rem; overflow-wrap: anywhere; }
h2 { font-size: 1.05rem; margin: 0 0 0.6rem; }
.meta, .note { color: var(--ink2); font-size: 0.85rem; margin: 0; }
section { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px;
  margin: 14px 0; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 14px 0; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; }
.tile .k { color: var(--ink2); font-size: 0.8rem; }
.tile .v { font-size: 1.6rem; font-weight: 650; }
.tile .s { color: var(--ink2); font-size: 0.8rem; }
.bars { margin: 4px 0 10px; }
.bars .row { display: grid; grid-template-columns: 6.5em 1fr; align-items: center; gap: 8px; min-height: 28px; }
.bars .lab { text-align: right; font-size: 0.85rem; }
.bars .track { display: flex; align-items: center; gap: 6px; border-left: 1px solid var(--axis); min-height: 28px; }
.bars .bar { display: block; height: 16px; background: var(--bar); border-radius: 0 4px 4px 0; min-width: 0; }
.bars .val { color: var(--ink2); font-size: 0.8rem; white-space: nowrap; font-variant-numeric: tabular-nums; }
.bars .row:hover .bar { opacity: 0.8; }
.courts { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 10px; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 0.85rem; }
th, td { text-align: left; padding: 5px 8px; border-bottom: 1px solid var(--grid); white-space: nowrap; }
th { color: var(--ink2); font-weight: 600; }
td.n { text-align: right; font-variant-numeric: tabular-nums; }
details summary { cursor: pointer; color: var(--ink2); font-size: 0.85rem; margin-top: 8px; }
@media print {
  body { background: #fff; }
  section, .tile { break-inside: avoid; border-color: #ccc; }
  details { display: block; }
  details > summary { display: none; }
}
"""


def _f(v: float | None, fmt: str = "{:.0f}", none: str = "—") -> str:
    return none if v is None else fmt.format(v)


def bars_html(items: list[tuple[str, float, str]], max_value: float | None = None) -> str:
    """横の棒グラフ（1 色。HTML なので、スマホでも文字が小さくならない）。items は（名前、値、値の文字）。

    棒に指を乗せると名前と値が出る。値は棒の右に書く。
    """
    if not items:
        return ""
    top = max_value or max(v for _, v, _ in items) or 1.0
    rows = "".join(
        f'<div class="row" title="{escape(name)}：{escape(text)}"><span class="lab">{escape(name)}</span>'
        f'<span class="track"><span class="bar" style="width:{100 * max(v, 0) / top:.1f}%"></span>'
        f'<span class="val">{escape(text)}</span></span></div>' for name, v, text in items)
    return f'<div class="bars" role="list">{rows}</div>'


def _table(head: list[str], rows: list[list[str]], numeric: set[int] | None = None) -> str:
    numeric = numeric or set()
    th = "".join(f"<th>{escape(h)}</th>" for h in head)
    body = "".join("<tr>" + "".join(f'<td{" class=n" if j in numeric else ""}>{escape(c)}</td>'
                                    for j, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="scroll"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def _recorded_at(info: dict[str, Any]) -> str:
    """撮った日時（``shot_list.json`` の ``recorded_at``。なければ空文字）を読みやすい形にする。"""
    v = info.get("recorded_at") or ""
    try:
        return datetime.fromisoformat(v.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return v


def _span(info: dict[str, Any]) -> str:
    a, b = info.get("start_s"), info.get("end_s")
    if a is None and b is None:
        return "動画全体"
    return f"{_f(a, '{:.1f}', '0')}〜{_f(b, '{:.1f}', '終わり')} 秒"


def build_report(shots: list[Shot], rallies: list[Rally], info: dict[str, Any] | None = None,
                 court: dict[str, Any] | None = None, title: str | None = None) -> str:
    """レポートの HTML（1 つのファイルで完結する）。"""
    info = info or {}
    st = rally_stats(shots, rallies, court)
    name = title or (Path(info["video"]).name if info.get("video") else "ショットの統計")
    meta = [x for x in (_recorded_at(info) and f"撮影 {_recorded_at(info)}", f"解析した区間：{_span(info)}",
                        f"作成 {datetime.now().strftime('%Y-%m-%d %H:%M')}") if x]

    fastest = max((s for s in shots if s.speed_kmh is not None), key=lambda s: s.speed_kmh, default=None)
    rs, sec = st["rally_shots"], st["rally_seconds"]
    tiles = [("ラリー", f"{st['n_rallies']}", f"最長 {_f(rs['max'])} 本" if rs["n"] else ""),
             ("ショット", f"{st['n_shots']}", f"初速が出たもの {st['n_shots_with_speed']} 本"),
             ("いちばん速い初速", f"{_f(fastest.speed_kmh if fastest else None)} km/h",
              f"{SHOT_TYPE_LABELS[fastest.shot_type]}・{fastest.t_hit_s:.1f} 秒" if fastest else ""),
             ("ラリーの平均", f"{_f(rs['mean'], '{:.1f}')} 本", f"{_f(sec['mean'], '{:.1f}')} 秒" if sec["n"] else "")]
    tiles_html = "".join(f'<div class="tile"><div class="k">{escape(k)}</div><div class="v">{escape(v)}</div>'
                         f'<div class="s">{escape(s)}</div></div>' for k, v, s in tiles)

    types = st["types"]
    known = [(k, v) for k, v in types.items() if k != "unknown"]
    known.sort(key=lambda kv: -kv[1]["n"])
    share_bars = bars_html([(v["label"], v["share"] or 0.0, f"{v['n']} 本（{100 * (v['share'] or 0):.0f}%）")
                          for _, v in known])
    unknown = types.get("unknown", {}).get("n", 0)
    speed_rows = [[v["label"], str(v["n"]), _f(v["mean_kmh"]), _f(v["max_kmh"])] for _, v in known]
    side_rows = []
    for side in ("h", "o"):
        for k, v in st["types_by_side"][side].items():
            if k != "unknown":
                side_rows.append([SIDE_NAMES[side], v["label"], str(v["n"]), _f(v["mean_kmh"]), _f(v["max_kmh"])])
    speed_bars = bars_html([(v["label"], v["max_kmh"], f"最高 {v['max_kmh']:.0f} km/h") for _, v in
                          sorted(known, key=lambda kv: -(kv[1]["max_kmh"] or 0)) if v["max_kmh"] is not None])

    hist = st["rally_length_hist"]
    hist_bars = bars_html([(h["label"], float(h["n"]), f"{h['n']} ラリー") for h in hist]) if rallies else ""
    ends = "、".join(f"{v['label']} {v['n']}" for v in st["end_reasons"].values())
    wins = "、".join(f"{SIDE_NAMES[k]} {n}" for k, n in sorted(st["points_won"].items()))

    land = "".join(f"<div>{landing_svg(shots, court, columns=1, types=[k])}</div>" for k in landing_types(shots))
    top10 = sorted((s for s in shots if s.speed_kmh is not None), key=lambda s: -s.speed_kmh)[:10]
    top_rows = [[f"{s.t_hit_s:.2f}", SHOT_TYPE_LABELS[s.shot_type], _f(s.speed_kmh),
                 f"{s.rally_id or '—'}・{s.index_in_rally or '—'}", SIDE_NAMES.get(s.side, s.side)] for s in top10]
    rally_rows = [[str(r.rally_id), f"{r.t_start_s:.1f}", str(r.n_shots),
                   _f(None if r.t_end_s is None else r.t_end_s - r.t_start_s, "{:.1f}"),
                   END_REASON_LABELS.get(r.end_reason, r.end_reason), SIDE_NAMES.get(r.winner_side, r.winner_side)]
                  for r in sorted(rallies, key=lambda r: r.t_start_s)]

    parts = [
        f'<header><p class="meta">ショットの統計</p><h1>{escape(name)}</h1>'
        f'<p class="meta">{escape("・".join(meta))}</p></header>',
        f'<div class="tiles">{tiles_html}</div>',
        '<section><h2>ショットの種類の割合</h2>' + (share_bars or '<p class="note">種類の分かったショットがありません。</p>')
        + (f'<p class="note">種類が分からないショット {unknown} 本は割合に入れていません。</p>' if unknown else "")
        + '<p class="note">種類は軌道から決まりで決めた目安です（手で直した正解との比べはまだ）。</p></section>',
        '<section><h2>種類ごとの初速</h2>' + speed_bars
        + _table(["種類", "本数", "平均 km/h", "最高 km/h"], speed_rows, {1, 2, 3})
        + (f'<details><summary>打った側ごと</summary>'
           + _table(["打った側", "種類", "本数", "平均 km/h", "最高 km/h"], side_rows, {2, 3, 4}) + '</details>'
           if side_rows else "")
        + '</section>',
    ]
    if hist_bars:
        parts.append('<section><h2>ラリーの長さ</h2>' + hist_bars
                     + f'<p class="note">平均 {_f(rs["mean"], "{:.1f}")} 本・中央値 {_f(rs["median"], "{:.1f}")} 本'
                     + (f'、平均 {_f(sec["mean"], "{:.1f}")} 秒' if sec["n"] else "") + "。"
                     + (f"終わり方：{escape(ends)}。" if ends else "")
                     + (f"得点：{escape(wins)}（着地点からの目安）。" if wins else "") + '</p></section>')
    if land:
        parts.append('<section><h2>着地点</h2><div class="courts">' + land + '</div>'
                     + '<p class="note">コートを上から見た図（左が h の側、右が o の側）。点は 1 本ずつの着地点'
                       '（打ち返されたショットは、そのまま飛んだら落ちた所）。色が濃い区画ほど多く落ちています。</p>'
                       '</section>')
    if top_rows:
        parts.append('<section><h2>速いショット</h2>'
                     + _table(["時刻 秒", "種類", "初速 km/h", "ラリー・何本目", "打った側"], top_rows, {0, 2})
                     + '</section>')
    if rally_rows:
        parts.append('<section><h2>ラリーの一覧</h2>'
                     + _table(["#", "始まり 秒", "本数", "長さ 秒", "終わり方", "得点した側"], rally_rows, {0, 1, 2, 3})
                     + '</section>')
    parts.append('<p class="note">smash-speed で作成。数字は動画の解析の結果で、検出の抜け・誤りがあると変わります。</p>')
    return ('<!doctype html><html lang="ja"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{escape(name)} の統計</title><style>{CSS}</style></head>'
            f'<body><main>{"".join(parts)}</main></body></html>')


def write_report(out_dir: str | Path, court: dict[str, Any] | None = None, shots: list[Shot] | None = None,
                 rallies: list[Rally] | None = None) -> Path:
    """ショットの一覧の出力先に ``report.html`` を書く（ショット・ラリーを省略すると CSV から読む）。"""
    out_dir = Path(out_dir)
    shots = read_shots(out_dir / SHOTS_FILE) if shots is None else shots
    if rallies is None:
        rallies = read_rallies(out_dir / RALLIES_FILE) if (out_dir / RALLIES_FILE).exists() else []
    info_path = out_dir / SHOT_LIST_FILE
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    path = out_dir / REPORT_FILE
    path.write_text(build_report(shots, rallies, info, court), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    import argparse

    from smash_speed.config import load_config

    ap = argparse.ArgumentParser(description="ショットの一覧の結果を、1 枚のレポート（HTML）にまとめる")
    ap.add_argument("out_dir", type=Path, help="ショットの一覧の出力先（shots.csv がある所）")
    ap.add_argument("--config", type=Path, default=None)
    a = ap.parse_args(argv)
    path = write_report(a.out_dir, load_config(a.config)["court"])
    print(f"{path} に書き出しました（PDF はブラウザの印刷から保存できます）")


if __name__ == "__main__":
    import sys

    if (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    main()
