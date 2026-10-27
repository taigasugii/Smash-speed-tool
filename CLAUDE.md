# smash-speed → Badminton Analytics

バドミントンの動画から、シャトルの速度・ショットの種類・ラリー・選手の動き・フォームを解析するツール。
いまはスマッシュの初速の解析が中心で、`agent/ROADMAP.md` に沿って Badminton Analytics へ広げている。

## 動かし方

- 環境：`.venv\Scripts\python.exe`（`pip install -e .[dev]`）。Python 3.11 以上。
- テスト：`.venv\Scripts\python.exe -m pytest -q`（約 3 分）
- 品質チェック（テスト + 合成データでの速度精度）：`.venv\Scripts\python.exe agent\eval_gate.py`（約 5 分）
- GUI：`.venv\Scripts\python.exe -m smash_speed.webapp`（`smash-speed-gui.bat`）

## 決まりごと

- 文章（README・コメント・画面の文言・コミット以外の記録）は日本語。画面の文言は専門用語を避け、操作する人の言葉で書く。
- コメントの量・名前の付け方は周りのコードに合わせる。
- 新しい機能には `tests/` にテストを付ける。重い処理は合成データ（`smash_speed/synth.py`）で確かめる。
- 既存のスマッシュ初速の精度を落とさない。`agent/eval_gate.py` が通らない変更は入れない。
- `configs/default.yaml` の既定値を変えるときは、理由と前後の精度を記録に残す。
- 動画・出力（`data/`、`outputs/`）は Git に入れない。`outputs/remote_access.json`（合言葉）は読まない・書かない。
- 大きな依存（数百 MB 以上のモデル・ライブラリ）や、ライセンスが不明なモデル・データは、使う前に `agent/QUESTIONS.md` で確認する。

## 自律エージェント

デスクトップで `agent/run.ps1` が Claude Code を繰り返し起動し、`agent/ROADMAP.md` の課題を 1 つずつ進める。
仕組みと使い方は `agent/README.md`、1 回分の手順は `agent/prompt.md`。
