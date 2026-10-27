# 作業の記録

エージェントが 1 回の作業ごとに、新しいものを上に書く。

書き方：

```
## 2026-11-01 01:23 B2 すべての打球を見つける（途中）
- やったこと：
- 結果（数字）：
- 品質チェック：通った（テスト 105 件、合成 MAPE 30fps 3.7% / 240fps 4.2%）
- 次にやること：
```

---

## 2026-10-28 01:08 A3 ショットとラリーのデータ形式（済み）
- やったこと：
  - `smash_speed/shots.py` を作った。`Shot`（打った時刻・打った側・種類・初速と 95% 信頼区間・打点・着地点と着地時刻・
    打点での速度と終端速度・信頼度・ラリーの番号と順番・出どころ）と `Rally`（始まり・終わり・本数・得点した側・終わり方・得点）。
  - 軌道は点の列ではなく物理モデルの初期値（打点・速度・終端速度）で持ち、`Shot.trajectory` で描き直す（CSV の 1 行に収まる）。
  - 打った側は座標系の x の符号で `h`（x < 0）／`o`（x > 0）。ランドマークの名前と同じ。手前・奥はカメラ次第なので使わない。
  - 種類は smash / clear / drop / drive / lob / net / push / serve / unknown（B4 の 8 種類 + 不明）。
  - `write_shots`・`read_shots`・`write_rallies`・`read_rallies`（BOM 付き UTF-8、空欄 = None、知らない列は読み飛ばす）、
    種類・側・信頼度の値の確かめ、打球の間隔で仮に分ける `make_rallies`（本当の区切りは C1）、`shots_of`。
  - 今のスマッシュの結果も出せるようにした：`analyze_selected` が `ranking.csv` と一緒に `shots.csv` を書く。
    初速は順位表と同じ値、打点・速度・着地点は各候補の `fit_result.json` から。
    前の出力先からは `python -m smash_speed.shots <出力先>` で書き出せる。信頼度はいまは「1 −（信頼区間の幅 ÷ 初速）」。
  - README に列の説明。テスト `tests/test_shots.py`（8 件）と、`test_batch.py` の合成動画の一括解析で shots.csv を確かめる行。
- 結果（数字）：解析の中身は変えていないので初速は同じ（実動画 15 本のずれ 0.00%）。
- 品質チェック：通った（pytest 通過 162 秒、合成 MAPE physics/audio 30fps 3.688% / 240fps 4.193%、
  hitter_position 30fps 6.955% / 240fps 6.059%（基準と同じ）、実動画 15 本の最大のずれ 0.00%）。
- 次にやること：B1（合成データをすべてのショットに広げる）。出力は `Shot` の形で正解を持たせる。

## 2026-10-28 00:30 A2 GPU の使い方を調べる（済み。A2b の PyTorch は確認待ち）
- やったこと：
  - `onnxruntime-directml` 1.24.4（約 25 MB、MIT）と `onnx`（約 8 MB、Apache-2.0）を入れた。pyproject に `gpu` の追加の依存として足した。
    DirectML で RX 7900 XT（メモリ 20 GB）が見えることを確かめた。
  - `smash_speed/accel.py` を作った：`choose_providers`（auto / gpu / cpu、環境変数 `SMASH_SPEED_DEVICE` が優先）、
    `make_session`（GPU で読み込めなければ警告を出して CPU で作り直す）、`device_info`（装置の名前とメモリ）、
    `python -m smash_speed.accel --bench`（TrackNet に似た U-Net 形のモデルで GPU と CPU を比べる）。
  - `docs/gpu.md` に結果と候補ごとのまとめ。README のセットアップに入れ方を足した。テスト `tests/test_accel.py`（7 件）。
  - 学習用の PyTorch（torch-directml / ROCm 版）は大きいので入れず、ROADMAP に A2b（`[?]`）として分け、QUESTIONS.md で確認した。
- 結果（数字）：288×512・9 チャンネル 1 枚で CPU 553 ms → GPU 6.2 ms（約 89 倍）、8 枚まとめて 3980 → 55 ms（約 72 倍）、
  576×1024 で 2136 → 24 ms（約 90 倍）。CPU と GPU の出力の差は最大 6×10⁻⁷。セッション作成 約 0.2 秒、初回推論 約 50 ms。
- 品質チェック：通った（pytest 通過 305 秒、合成 MAPE physics/audio 30fps 3.688% / 240fps 4.193%、
  hitter_position 30fps 6.955% / 240fps 6.059%（基準と同じ）、実動画 15 本の最大のずれ 0.00%）。
- 次にやること：A3（ショットとラリーのデータ形式）。A2b は答え待ち。

## 2026-10-27 22:16 A1 実動画の回帰チェック（済み）
- やったこと：
  - `agent/checks/check_real_videos.py` を作った。本体の `outputs/*_smashes/ranking.csv` に載ったスマッシュ（4 本の動画・15 本）の
    区間・打つ人・着地点・候補探しの検出結果・キャリブレーションの中身を `agent/checks/real_videos_baseline.json` に固め、
    GUI の「詳しく解析する」（`batch.analyze_selected`）と同じ手順で解析し直して、初速が 3% を超えてずれたら失敗にする。
    動画がない PC・基準がないときは「飛ばした」で通す。動画ごとに別のプロセスで並べて動かす（12 分 → 約 4 分）。
  - `--make-baseline` で本体の outputs から基準を作り直せる（人が決めたときだけ）。
  - テスト `tests/test_real_video_check.py`（基準の作り方・3% の判定・動画や基準がないときに飛ばす・動画の探し方）。README に使い方と所要時間。
- 結果（数字）：
  - `ranking.csv` のうち IMG_3391（3 本）・IMG_4521（4 本）・IMG_4397 #53 は、今のコードでも完全に同じ初速になった。
  - IMG_4397 の 4 本と IMG_5096 の 2 本は、今のコードでは違う値になる：
    4397 #50 305.8→339.7、#22 297.3→228.2、#60 249.7→236.0、#55 430.0→351.4、5096 #3 310.4→295.1、#4 82.7→37.0 km/h（#1 は +1.2%）。
    これらの `ranking.csv` は古いコード（7bd7b0c・a3b56b1、10/19〜20）で作られており、その後の 5f3de77
    「Make the fastest-smash batch match single-shot analysis」で手順が変わったため。今回のコードの退行ではない。
  - そこで基準の初速は「今のコードで解析し直した値」にし、`ranking.csv` の値は `ranking_kmh` として残した。
    2 回解析し直して（1 本ずつ・並べて）同じ値になることを確かめた（決定的）。
- 品質チェック：通った（テストは pytest が通過。合成 MAPE physics/audio 30fps 3.688% / 240fps 4.193%、
  hitter_position 30fps 6.955% / 240fps 6.059%（基準と同じ）、実動画 15 本の最大のずれ 0.00%）。
  品質チェック全体は約 12 分になった（テスト 294 秒・合成 169 秒・実動画 236 秒）。
  なお eval_gate の表示が「テスト：0 件通過」になっている（pytest の最後の行から件数を読み取れていない様子。判定は終了コードなので影響なし）。
- 次にやること：A2（GPU の使い方を調べる）。重いものを入れる前に QUESTIONS.md で確認する。
