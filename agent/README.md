# 自律開発エージェント

デスクトップで Claude Code を無人で繰り返し動かし、`agent/ROADMAP.md` の課題（すべてのショットの速度・種類、ラリー統計、選手の動き、フォーム解析…）を 1 つずつ実装していく仕組み。

## 動き方

- **毎晩 0:00〜12:00**：タスク スケジューラが `run.ps1 -Until 12:00` を起動し、課題を 1 つずつ続けて進める。12:00 を過ぎたら新しい回は始めず、そのとき作業中の回には区切りのいい所（テストが通る状態）まで終えてコミットしてもらう。それでも終わらないときは 13:00 に打ち切る（午後の利用枠を使いすぎないため。`-GraceMinutes` で変えられる）。
- **一時停止**：夜間の作業は、次のあいだ一時停止する。作業中の回には区切りのいい所まで終えてコミットしてもらい、再開したら次の回から続ける。止まっていても 12:00 の締めは変わらない。
  - GUI の「一時停止」を押したとき（スマホからも可）。「再開」を押すまで待つ。動いていないときに押しておくと、その夜は始めずに待つ。
  - この PC をキーボードやマウスで使っているとき（3 分ほど続けて操作したら）。操作が 10 分なければ再開する。GUI のチェックで切り替えられる（`.agent/config.json` の `auto_pause`。再開までの分数は `auto_pause_idle_minutes`）。
  - 「課題を 1 つ進める」の回は、人が頼んだものなので一時停止しない。
- **それ以外の時間**：GUI の「開発エージェント」で「課題を 1 つ進める」を押すと、5 分以内に 1 回分（3 時間が目安、区切りまでの延長を含めて最長 4 時間）だけ動く。スマホからも押せる。
- **1 回分の流れ**
  1. 最新の main を取り込む
  2. Claude が課題を 1 つ選んで実装し、テストを書く
  3. 品質チェック（`eval_gate.py`）を通す
  4. `agent/dev` にコミットする
  5. GitHub に push し、main への PR を更新する
- **品質チェック**：テストがすべて通り、合成データでのスマッシュの初速の誤差が `baseline.json` より悪くなっていないこと。チェックは `agent/checks/` に足していける。通らない変更は取り消し、`.agent/failed/` にパッチとして残す。
- **確認のしかた**：PR「Badminton Analytics：エージェントの作業」を GitHub（スマホのアプリでも可）で見る。よければマージする。直してほしい所を PR にコメントすると、次の回の最優先の指示になる。
- **Issue での頼みごと**：GitHub に Issue を立てると（スマホのアプリでも可）、次の回で対応する。PR があってもなくても使える。対応したら Issue に「🤖 エージェント（対応しました／途中／質問）」で始まる返事を書き、済んだものは閉じる。質問の返事が付いた Issue は、あなたがコメントで答えるまで待つ。優先順位は PR へのコメント → Issue（古い順）→ ROADMAP。
- **読むのは自分の書き込みだけ**：PR のコメントも Issue も、`gh` でログインしているアカウントが書いたものだけを読む（公開リポジトリで他人の書き込みを指示として扱わないため）。
- **質問**：エージェントが決められないことは `agent/QUESTIONS.md` に書かれ、PR と GUI に出る。答えは PR にコメントする。
- **利用枠の上限**：上限に達したら回復するまで待つ。ただし回復が 12:00 を過ぎるときは、その日は止まる。

## 安全のための決まり

- エージェントは本体（`C:\Users\Public\Claude\smash-speed`）とは別のコピー `C:\Users\Public\Claude\smash-speed-agent` で作業する。本体のファイルには触らない。
- push と PR の作成は `run.ps1` だけが行う。Claude には `git push`・`gh`・ブランチの切り替え・`git reset` を許していない（`claude-settings.json`）。
- `run.ps1`・`claude-settings.json`・`eval_gate.py`・`baseline.json` は Claude が書き換えられない。
- main には直接入らない。必ず PR を人がマージする。
- GUI のボタンは合図のファイル（`.agent/REQUEST`・`.agent/STOP`・`.agent/PAUSE`）を置くだけで、プログラムは起動しない。

## デスクトップでの準備（1 回だけ）

1. 次の道具を入れ、ログインする。
   - `winget install --id GitHub.cli` を実行してから `gh auth login`
   - Claude Code のコマンド版：`irm https://claude.ai/install.ps1 | iex` を実行してから `claude` でログイン
2. 作業用のコピーを作り、Python の環境を入れる。
   ```
   cd C:\Users\Public\Claude
   git clone https://github.com/taigasugii/Smash-speed-tool.git smash-speed-agent
   cd smash-speed-agent
   py -3 -m venv .venv
   .venv\Scripts\pip install -e .[dev]
   ```
3. `smash-speed-agent\.agent\config.json` を作る。実動画のチェックで使う。
   ```json
   {"main_repo": "C:\\Users\\Public\\Claude\\smash-speed", "videos_dir": "C:\\Users\\<ユーザー名>\\Downloads\\Badminton解析用", "notify_url": ""}
   ```
   `notify_url` に ntfy.sh などの URL を入れると、1 回終わるごとにスマホへ通知する。その場合、課題名と要約がそのサービスに送られる。
4. 品質チェックが動くことを確かめる（5 分ほどかかる）。
   ```
   .venv\Scripts\python.exe agent\eval_gate.py
   ```
5. タスク スケジューラに 2 つ登録する。どちらも共通の設定にする。
   - 実行するユーザー：今のユーザー。サインインしているときだけ動かす。
   - 「スリープを解除して実行」と「開始時刻を過ぎていたらすぐ実行」をオンにする。
   - すでに動いているときは、新しく起動しない。

   それぞれの起動の時刻と操作は次のとおり。
   - 毎日 0:00（実行時間の上限 14 時間）
     `powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "C:\Users\Public\Claude\smash-speed-agent\agent\run.ps1" -Until 12:00`
   - 5 分おき（上限 5 時間）
     `powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "C:\Users\Public\Claude\smash-speed-agent\agent\run.ps1" -OnRequest`
6. PC は夜もサインインしたままにする（画面のロックは構わない）。スリープさせる場合は、電源オプションで「スリープ解除タイマーの許可」を有効にする。動いている間は `run.ps1` がスリープを止める。
7. GUI（`smash-speed-remote.bat`）を起動し直すと、メニューに「開発エージェント」が出る。

## よく使う操作

| やりたいこと | 方法 |
|---|---|
| 様子を見る | GUI の「開発エージェント」（作業中は 10 秒ごとに更新） |
| 今すぐ 1 つ進める | GUI の「課題を 1 つ進める」、または `run.ps1 -Once` |
| しばらく待ってもらう | GUI の「一時停止」→「再開」、または `.agent\PAUSE` という空のファイルを置く・消す |
| すぐ止める | GUI の「止める」、または `.agent\STOP` という空のファイルを置く（区切りを待たずに止め、コミットしていない作業はパッチとして残す） |
| 頼みごとをする | GitHub に Issue を立てる（タイトルと本文にふつうの言葉で書く）。午後なら、そのあと「課題を 1 つ進める」を押す |
| 課題の順番を変える・足す | Issue で頼むか、`agent/ROADMAP.md` を main で直す（次の回から反映） |
| 結果を取り込む | PR をマージする。デスクトップの本体は `git pull` で最新になる |
| 品質の基準を更新する | 人が決めたときだけ `eval_gate.py --update-baseline` を実行し、`baseline.json` をコミットする |

## ファイル

- `prompt.md`：1 回分の指示。`run.ps1` が「今回の状況」（残り時間・PR へのコメント・前回の失敗）を付け足して渡す。
- `ROADMAP.md`・`JOURNAL.md`・`QUESTIONS.md`：課題・作業の記録・質問。エージェントが更新する。
- `.agent/`（Git に入らない）：状態（`status.json`）、履歴（`history.jsonl`）、ログ（`logs/`）、取り消した変更（`failed/`）。
