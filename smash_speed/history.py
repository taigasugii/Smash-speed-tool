"""セッション（動画）をまたいだ記録（ROADMAP C4）：日付ごとの最高・平均速度の推移と、ショットの種類の割合の変化。

出力の場所の下にある、ショットの一覧の出力先（``shot_list.json`` と ``shots.csv`` がある所）をすべて集め、
撮った日（``shot_list.json`` の ``recorded_at``。なければ動画のファイルの更新日時、それもなければ
``shot_list.json`` の更新日時）ごとにまとめる。同じ日の動画は 1 つにまとめる。スマッシュの一括解析の出力先
（スマッシュだけの ``shots.csv``）は、種類の割合が偏るので入れない。

- ``history.csv``：1 行 1 日（日付・動画の数・ショットとラリーの数・スマッシュの本数と平均・最高の初速・
  全体の最高の初速とその種類・ラリーの平均の長さ・種類ごとの割合）
- ``history.html``：スマッシュの初速の推移（最高・平均の折れ線）、種類の割合の変化（日ごとの積み上げの棒）、表

``python -m smash_speed.history <出力の場所>`` で書き出す（出力の場所の直下に書く）。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import numpy as np

from smash_speed.shots import RALLIES_FILE, SHOT_TYPE_LABELS, SHOTS_FILE, Rally, Shot, read_rallies, read_shots

HISTORY_CSV = "history.csv"
HISTORY_HTML = "history.html"
SHOT_LIST_FILE = "shot_list.json"   # shot_list.SHOT_LIST_FILE と同じ
SPEED_TYPE = "smash"                # 推移を出す種類
# 割合の棒の種類の順（色は ``TYPE_COLORS`` の順に決まっていて、日によって変わらない）
SHARE_TYPES = ("smash", "clear", "drop", "drive", "lob", "net", "push", "serve")


@dataclass
class Session:
    """ショットの一覧の出力先 1 つ（動画 1 本、またはその区間）。"""

    path: Path
    date: str                            # 撮った日 YYYY-MM-DD
    date_source: str                     # recorded_at | video_mtime | file_mtime
    video: str
    shots: list[Shot] = field(default_factory=list)
    rallies: list[Rally] = field(default_factory=list)


def _date_of(info: dict[str, Any], info_path: Path) -> tuple[str, str]:
    v = info.get("recorded_at")
    if v:
        try:
            return datetime.fromisoformat(str(v).replace("Z", "+00:00")).astimezone().strftime("%Y-%m-%d"), \
                "recorded_at"
        except ValueError:
            pass
    video = Path(info.get("video") or "")
    if info.get("video") and video.exists():
        return datetime.fromtimestamp(video.stat().st_mtime).strftime("%Y-%m-%d"), "video_mtime"
    return datetime.fromtimestamp(info_path.stat().st_mtime).strftime("%Y-%m-%d"), "file_mtime"


def find_sessions(root: str | Path) -> list[Session]:
    """``root`` の下のショットの一覧の出力先を、日付の順に読む。"""
    out = []
    for info_path in sorted(Path(root).rglob(SHOT_LIST_FILE)):
        d = info_path.parent
        if not (d / SHOTS_FILE).exists():
            continue
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        date, src = _date_of(info, info_path)
        rallies = read_rallies(d / RALLIES_FILE) if (d / RALLIES_FILE).exists() else []
        out.append(Session(d, date, src, Path(info.get("video") or d.name).name, read_shots(d / SHOTS_FILE), rallies))
    return sorted(out, key=lambda s: (s.date, s.video))


def _round(v: float | None, n: int = 1) -> float | None:
    return None if v is None else round(float(v), n)


def day_row(date: str, sessions: list[Session]) -> dict[str, Any]:
    """1 日分（同じ日の動画をまとめたもの）の数字。"""
    shots = [s for x in sessions for s in x.shots]
    rallies = [r for x in sessions for r in x.rallies]
    sp = [s.speed_kmh for s in shots if s.shot_type == SPEED_TYPE and s.speed_kmh is not None]
    with_speed = [s for s in shots if s.speed_kmh is not None]
    fastest = max(with_speed, key=lambda s: s.speed_kmh, default=None)
    known = [s for s in shots if s.shot_type != "unknown"]
    row: dict[str, Any] = {
        "date": date, "n_videos": len(sessions), "videos": " ".join(x.video for x in sessions),
        "n_shots": len(shots), "n_rallies": len(rallies),
        f"{SPEED_TYPE}_n": len(sp), f"{SPEED_TYPE}_mean_kmh": _round(np.mean(sp) if sp else None),
        f"{SPEED_TYPE}_max_kmh": _round(max(sp) if sp else None),
        "max_kmh": _round(fastest.speed_kmh if fastest else None), "max_type": fastest.shot_type if fastest else "",
        "rally_mean_shots": _round(np.mean([r.n_shots for r in rallies]) if rallies else None, 2),
    }
    for k in SHARE_TYPES:
        row[f"share_{k}"] = _round(sum(s.shot_type == k for s in known) / len(known), 4) if known else None
    return row


def history_rows(sessions: list[Session]) -> list[dict[str, Any]]:
    by_date: dict[str, list[Session]] = {}
    for s in sessions:
        by_date.setdefault(s.date, []).append(s)
    return [day_row(d, by_date[d]) for d in sorted(by_date)]


def write_history_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    path = Path(path)
    cols = list(rows[0]) if rows else ["date"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow(["" if r[c] is None else r[c] for c in cols])
    return path


# =============================================================================
# ページ（HTML）
# =============================================================================
# 種類の色（分類の色。いつも同じ順。明るい画面・暗い画面）
TYPE_COLORS = {
    "smash": ("#2a78d6", "#3987e5"), "clear": ("#eb6834", "#d95926"), "drop": ("#1baf7a", "#199e70"),
    "drive": ("#eda100", "#c98500"), "lob": ("#e87ba4", "#d55181"), "net": ("#008300", "#008300"),
    "push": ("#6250d6", "#9085e9"), "serve": ("#e34948", "#e66767"),
}

CSS = """
:root { color-scheme: light; --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10); --s1: #2a78d6; --s2: #eb6834; %LIGHT% }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { color-scheme: dark; --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec;
    --ink2: #c3c2b7; --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --s1: #3987e5;
    --s2: #d95926; %DARK% }
}
:root[data-theme="dark"] { color-scheme: dark; --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec;
  --ink2: #c3c2b7; --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --s1: #3987e5;
  --s2: #d95926; %DARK% }
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", "Yu Gothic UI", "Hiragino Sans", sans-serif; line-height: 1.5; }
main { max-width: 860px; margin: 0 auto; padding: 16px; }
h1 { font-size: 1.35rem; margin: 0.2rem 0; }
h2 { font-size: 1.05rem; margin: 0 0 0.6rem; }
.note { color: var(--ink2); font-size: 0.85rem; margin: 6px 0 0; }
section { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px; margin: 14px 0; }
.chart { width: 100%; height: auto; display: block; }
.chart text { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.chart .grid { stroke: var(--grid); stroke-width: 1; }
.chart .axis { stroke: var(--axis); stroke-width: 1; }
.chart .l1 { stroke: var(--s1); } .chart .l2 { stroke: var(--s2); }
.chart .ln { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.chart .p1 { fill: var(--s1); } .chart .p2 { fill: var(--s2); }
.chart .pt { stroke: var(--surface); stroke-width: 2; }
.chart .lab { fill: var(--ink); font-size: 11px; }
.chart .hit { fill: transparent; }
.chart g.day:hover .pt { r: 6; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 0.85rem; margin: 0 0 8px; color: var(--ink2); }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }
.stack .row { display: grid; grid-template-columns: 6.5em 1fr; gap: 8px; align-items: center; min-height: 26px; }
.stack .d { font-size: 0.8rem; color: var(--ink2); text-align: right; font-variant-numeric: tabular-nums; }
.stack .bar { display: flex; height: 16px; gap: 2px; }
.stack .seg { display: block; height: 100%; min-width: 2px; }
.stack .seg:first-child { border-radius: 0; } .stack .seg:last-child { border-radius: 0 4px 4px 0; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 0.85rem; }
th, td { text-align: left; padding: 5px 8px; border-bottom: 1px solid var(--grid); white-space: nowrap; }
th { color: var(--ink2); font-weight: 600; }
td.n { text-align: right; font-variant-numeric: tabular-nums; }
.narrow { display: none; }
@media (max-width: 560px) { .wide { display: none; } .narrow { display: block; } }
@media print { body { background: #fff; } section { break-inside: avoid; } .wide { display: block; } .narrow { display: none; } }
"""


def _css() -> str:
    light = " ".join(f"--t-{k}: {c[0]};" for k, c in TYPE_COLORS.items())
    dark = " ".join(f"--t-{k}: {c[1]};" for k, c in TYPE_COLORS.items())
    return CSS.replace("%LIGHT%", light).replace("%DARK%", dark)


def speed_chart(rows: list[dict[str, Any]], width: int = 820, height: int = 280) -> str:
    """スマッシュの初速の推移（最高・平均の 2 本の折れ線。日付は等間隔）。点に指を乗せると値が出る。"""
    pts = [(i, r) for i, r in enumerate(rows) if r.get(f"{SPEED_TYPE}_max_kmh") is not None]
    if not pts:
        return ""
    vals = [r[f"{SPEED_TYPE}_max_kmh"] for _, r in pts] + [r[f"{SPEED_TYPE}_mean_kmh"] for _, r in pts]
    lo = max(0.0, 20 * np.floor((min(vals) - 20) / 20))
    hi = 20 * np.ceil((max(vals) + 10) / 20)
    left, right, top, bottom = 44, 56, 12, 30
    pw, ph = width - left - right, height - top - bottom
    n = len(rows)
    X = lambda i: left + (pw / 2 if n == 1 else pw * i / (n - 1))  # noqa: E731
    Y = lambda v: top + ph * (1 - (v - lo) / (hi - lo))  # noqa: E731
    out = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="スマッシュの初速の推移">']
    step = 20 if hi - lo <= 120 else 50
    for v in np.arange(lo, hi + 0.1, step):
        out.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>'
                   f'<text x="{left - 6}" y="{Y(v) + 4:.1f}" text-anchor="end">{v:.0f}</text>')
    out.append(f'<line class="axis" x1="{left}" x2="{width - right}" y1="{top + ph}" y2="{top + ph}"/>')
    every = max(1, int(np.ceil(n / max(2, width // 100))))  # 日付の目盛りは 100 px に 1 つまで
    for i, r in enumerate(rows):
        if i % every == 0 or i == n - 1:
            out.append(f'<text x="{X(i):.1f}" y="{height - 10}" text-anchor="middle">{escape(r["date"][5:])}</text>')
    for key, cls in ((f"{SPEED_TYPE}_max_kmh", "1"), (f"{SPEED_TYPE}_mean_kmh", "2")):
        d = " ".join(f'{"M" if j == 0 else "L"}{X(i):.1f},{Y(r[key]):.1f}' for j, (i, r) in enumerate(pts))
        out.append(f'<path class="ln l{cls}" d="{d}"/>')
    for i, r in pts:
        tip = (f'{r["date"]}：最高 {r[f"{SPEED_TYPE}_max_kmh"]:.0f} km/h・平均 {r[f"{SPEED_TYPE}_mean_kmh"]:.0f} km/h'
               f'（{r[f"{SPEED_TYPE}_n"]} 本）')
        out.append(f'<g class="day"><title>{escape(tip)}</title>'
                   f'<rect class="hit" x="{X(i) - 14:.1f}" y="{top}" width="28" height="{ph}"/>'
                   f'<circle class="pt p1" cx="{X(i):.1f}" cy="{Y(r[f"{SPEED_TYPE}_max_kmh"]):.1f}" r="4"/>'
                   f'<circle class="pt p2" cx="{X(i):.1f}" cy="{Y(r[f"{SPEED_TYPE}_mean_kmh"]):.1f}" r="4"/></g>')
    # 線の名前は右の端に直接書く（凡例もある）
    i, r = pts[-1]
    out.append(f'<text class="lab" x="{X(i) + 8:.1f}" y="{Y(r[f"{SPEED_TYPE}_max_kmh"]) + 4:.1f}">最高</text>'
               f'<text class="lab" x="{X(i) + 8:.1f}" y="{Y(r[f"{SPEED_TYPE}_mean_kmh"]) + 4:.1f}">平均</text>')
    out.append("</svg>")
    return "".join(out)


def share_bars(rows: list[dict[str, Any]]) -> str:
    """日ごとの種類の割合（積み上げの横の棒。種類の順と色はいつも同じ）。"""
    lines = []
    for r in rows:
        segs = []
        for k in SHARE_TYPES:
            v = r.get(f"share_{k}")
            if v:
                segs.append(f'<span class="seg" style="flex:{v:.4f};background:var(--t-{k})" '
                            f'title="{r["date"]}：{SHOT_TYPE_LABELS[k]} {100 * v:.0f}%"></span>')
        if segs:
            lines.append(f'<div class="row"><span class="d">{escape(r["date"])}</span>'
                         f'<span class="bar">{"".join(segs)}</span></div>')
    if not lines:
        return ""
    legend = "".join(f'<span><i style="background:var(--t-{k})"></i>{SHOT_TYPE_LABELS[k]}</span>' for k in SHARE_TYPES)
    return f'<div class="legend">{legend}</div><div class="stack">{"".join(lines)}</div>'


def _f(v: Any, fmt: str = "{:.0f}") -> str:
    return "—" if v is None else fmt.format(v)


def build_history_page(rows: list[dict[str, Any]], title: str = "記録の推移") -> str:
    # 横に広い画面用と、スマホ用（文字が小さくなりすぎないように、狭い図を別に作る）
    chart = speed_chart(rows)
    if chart:
        chart = f'<div class="wide">{chart}</div><div class="narrow">{speed_chart(rows, 400, 260)}</div>'
    shares = share_bars(rows)
    head = ["日付", "動画", "ショット", "ラリー", f"{SHOT_TYPE_LABELS[SPEED_TYPE]}", "平均 km/h", "最高 km/h",
            "全体の最高", "ラリーの平均"]
    body = "".join(
        "<tr>" + "".join(f"<td{' class=n' if j >= 1 else ''}>{escape(c)}</td>" for j, c in enumerate([
            r["date"], str(r["n_videos"]), str(r["n_shots"]), str(r["n_rallies"]), str(r[f"{SPEED_TYPE}_n"]),
            _f(r[f"{SPEED_TYPE}_mean_kmh"]), _f(r[f"{SPEED_TYPE}_max_kmh"]),
            (_f(r["max_kmh"]) + (f"（{SHOT_TYPE_LABELS[r['max_type']]}）" if r["max_type"] else "")),
            _f(r["rally_mean_shots"], "{:.1f} 本")])) + "</tr>" for r in reversed(rows))
    table = (f'<div class="scroll"><table><thead><tr>{"".join(f"<th>{h}</th>" for h in head)}</tr></thead>'
             f'<tbody>{body}</tbody></table></div>')
    legend = ('<div class="legend"><span><i style="background:var(--s1)"></i>最高</span>'
              '<span><i style="background:var(--s2)"></i>平均</span></div>')
    parts = [f'<header><h1>{escape(title)}</h1><p class="note">{len(rows)} 日分・作成 '
             f'{datetime.now().strftime("%Y-%m-%d %H:%M")}</p></header>']
    if chart:
        parts.append(f'<section><h2>{SHOT_TYPE_LABELS[SPEED_TYPE]}の初速の推移（km/h）</h2>{legend}{chart}'
                     '<p class="note">日付は等間隔に並べています（間の空いた日は詰めています）。</p></section>')
    if shares:
        parts.append(f'<section><h2>ショットの種類の割合</h2>{shares}'
                     '<p class="note">種類の分かったショットの中の割合。種類は軌道から決まりで決めた目安です。</p></section>')
    parts.append(f'<section><h2>日ごとの数字</h2>{table}</section>')
    return ('<!doctype html><html lang="ja"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{escape(title)}</title><style>{_css()}</style></head><body><main>{"".join(parts)}</main>'
            '</body></html>')


def write_history(root: str | Path, out_dir: str | Path | None = None) -> tuple[list[dict[str, Any]], Path, Path]:
    """``root`` の下のセッションを集めて、``history.csv``・``history.html`` を ``out_dir``（省略時は root）に書く。"""
    out = Path(out_dir or root)
    out.mkdir(parents=True, exist_ok=True)
    rows = history_rows(find_sessions(root))
    csv_path = write_history_csv(out / HISTORY_CSV, rows)
    html_path = out / HISTORY_HTML
    html_path.write_text(build_history_page(rows), encoding="utf-8")
    return rows, csv_path, html_path


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="ショットの一覧の結果を日付ごとにまとめ、記録の推移を書き出す")
    ap.add_argument("root", type=Path, help="出力の場所（この下のショットの一覧の出力先をすべて集める）")
    ap.add_argument("--out", type=Path, default=None, help="書き出す場所（省略時は root）")
    a = ap.parse_args(argv)
    rows, csv_path, html_path = write_history(a.root, a.out)
    print(f"{len(rows)} 日分を {csv_path}・{html_path} に書き出しました")
    for r in rows[-5:]:
        print(f"  {r['date']}：ショット {r['n_shots']} 本、{SHOT_TYPE_LABELS[SPEED_TYPE]} 最高 {_f(r[f'{SPEED_TYPE}_max_kmh'])}"
              f"・平均 {_f(r[f'{SPEED_TYPE}_mean_kmh'])} km/h")


if __name__ == "__main__":
    import sys

    if (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    main()
