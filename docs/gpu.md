# GPU（Radeon RX 7900 XT）の使い方

ROADMAP の A2 の調べた結果。シャトル検出（TrackNet）や姿勢推定・人の検出のモデルを、
デスクトップの GPU で速く動かすための方法をまとめる。

## 結論

- **推論（モデルを動かすだけ）は ONNX ＋ onnxruntime-directml を使う。**
  Windows の DirectML（DirectX 12）経由なので、AMD・NVIDIA・Intel のどの GPU でも同じ書き方で動く。
  入れるもの：`onnxruntime-directml`（約 25 MB、MIT License）と、モデルを作る・調べるための `onnx`（約 8 MB、Apache-2.0）。
  `pip install -e ".[gpu]"` で入る。
- コードからは `smash_speed/accel.py` の `make_session(モデル)` を使う。GPU があれば GPU、なければ CPU で動く。
  GPU で読み込めないとき（ドライバの不具合など）も、警告を出して CPU で動かし直す。
- **学習（TrackNetV3 の追加学習など、F1）は PyTorch ＋ torch-directml を使う**（A2b、2026-11-05）。
  大きい（約 1.4 GB）ので本体の `.venv` には入れず、別の仮想環境 `.venv-train` に入れる（下の「学習（PyTorch）」）。
  TrackNetV3 と同じ形のモデルの学習で、CPU の約 50 倍速い。ROCm 版 PyTorch は試していない。

## 測った結果（2026-10-28）

PC：Intel Core i7-14700F／AMD Radeon RX 7900 XT（メモリ 20 GB）、Windows 11、Python 3.12、onnxruntime-directml 1.24.4。

モデル：TrackNet に似た形（畳み込み → 縮小 3 段 → 拡大 3 段、幅 64〜512 チャンネル、重みは乱数）を
`accel.build_bench_model()` で作ったもの。入力は TrackNetV3 と同じ 288×512・9 チャンネル（3 コマ × RGB）。
`python -m smash_speed.accel --bench` で測れる（20 回の中央値）。

| 入力 | CPU | GPU（DirectML） | 何倍速いか |
|---|---:|---:|---:|
| 1 枚（1×9×288×512） | 553 ms | 6.2 ms | 約 89 倍 |
| 8 枚まとめて（8×9×288×512） | 3980 ms | 55 ms | 約 72 倍 |
| 4 倍の大きさ（1×9×576×1024） | 2136 ms | 24 ms | 約 90 倍 |

- CPU と GPU の出力の差は最大 6×10⁻⁷ で、結果は同じと見てよい。
- 1 枚 6 ms なので、240fps で 10 秒の動画（2400 コマ、3 コマずつ）でも推論は 5 秒ほど。
  CPU だと同じ動画で 7 分ほどかかる計算になる。
- 実際の TrackNetV3 の重さは、このモデルと同じくらい（同じ規模の U-Net）。ONNX に変えたときの実測は F1 で行う。

## 候補ごとのまとめ

| 候補 | 大きさ | 試した結果 | 使い道 |
|---|---|---|---|
| onnxruntime-directml | 約 25 MB | 入れて動いた。上の表のとおり速い | 推論すべて（採用） |
| torch-directml | torch 2.4.1 と合わせて約 1.4 GB | 入れて動いた。学習が CPU の約 50 倍速い（下の表） | 学習（F1、採用）。torch は 2.4.1 に固定される |
| ROCm for Windows 版 PyTorch | 数 GB | 試していない（人が「試さない」と決めた） | — |
| OpenCV の OpenCL（`cv2.UMat`） | 追加なし | OpenCL は使える（`cv2.ocl.haveOpenCL()` が True） | 画像の前処理を速くしたいとき（F3）に試す |

## 使い方（コード）

```python
from smash_speed.accel import make_session, device_info

print(device_info())                 # 使われる装置（例：AMD Radeon RX 7900 XT）
sess = make_session("model.onnx")    # GPU があれば GPU、なければ CPU
y = sess.run(None, {"x": x})[0]
```

GPU を使いたくないとき（比べたいとき・不具合を疑うとき）は、環境変数 `SMASH_SPEED_DEVICE=cpu` にする
（`gpu` にすると GPU を使う、`auto` が既定）。

## 気を付けること

- DirectML ではメモリの並べ方の最適化（`enable_mem_pattern`）と並列実行が使えないため、`make_session` で切っている。
- `onnxruntime`（CPU 版）と `onnxruntime-directml` を同じ環境に両方入れると壊れる。入れ替えるときは片方を消してから入れる。
- 準備に時間がかかる：セッションを作るのに約 0.2 秒、最初の 1 回の推論は約 50 ms（2 回目からは 5 ms ほど）。
  セッションは動画ごとに作り直さず使い回し、速さを測るときは最初に 1 回空で動かしておく。

## 学習（PyTorch）（A2b、2026-11-05）

### 入れ方

本体の `.venv` とは別の仮想環境に入れる（torch は大きく、本体の解析には要らないため）。

```
py -3.12 -m venv .venv-train
.venv-train\Scripts\python -m pip install torch-directml
```

入るもの：torch-directml 0.2.5.dev240914（MIT License）、torch 2.4.1（CPU 版。BSD 系）、torchvision 0.19.1、numpy など。
合わせて約 1.4 GB。torch-directml が torch 2.4.1 を指定するので、torch の版はこれに固定される。
smash_speed を `.venv-train` から使うときは、`PYTHONPATH` にこのリポジトリを入れる（`pip install -e .` はしていない）。

### 使い方（コード）

```python
from smash_speed.train.torch_device import train_device, make_adam

device, name = train_device()        # GPU があれば DirectML の装置（例：AMD Radeon RX 7900 XT）、なければ CPU
model = model.to(device)
opt = make_adam(model.parameters(), lr=1e-4, device=device)
```

装置の選び方は推論と同じ（環境変数 `SMASH_SPEED_DEVICE`＝`auto` / `gpu` / `cpu`）。

### 測った結果

モデル：TrackNetV3 の TrackNet と同じ層の並び（`torch_device.build_tracknet`。入力 27 チャンネル＝8 コマ＋背景の RGB、
出力 8 枚の熱の地図、288×512、重みは乱数）。損失は TrackNetV3 の重み付き交差エントロピー、Adam（lr 1e-4）。
1 歩＝順伝播・逆伝播・重みの更新。GPU は 12 歩の中央値（先に 4 歩空で動かす）、CPU は 2 歩の中央値。
`.venv-train\Scripts\python -m smash_speed.train.torch_device --bench`（1・2 組）で測れる。4 組以上は `--batch 4` のように 1 つずつ別に動かす。

| 一度に学習する組の数 | CPU | GPU（DirectML） | 何倍速いか | GPU で 1 秒に学習できる組 |
|---:|---:|---:|---:|---:|
| 1 | 2062〜2183 ms | 41〜49 ms | 約 50 倍 | 約 22 |
| 2 | 3931〜4047 ms | 84〜87 ms | 約 47 倍 | 約 23 |
| 4 | 8499 ms | 172〜224 ms | 約 40 倍 | 約 20 |
| 6 | — | 1500〜1650 ms | — | 約 4（メモリがあふれる） |
| 10 | — | メモリ不足で止まる | — | — |

- **CPU と同じ結果になる**：同じ重み・同じ入力で、損失の差は相対 1×10⁻⁷、勾配の差は相対 2〜4×10⁻³。
  幅 32・144×256 で 60 歩学習すると、損失の下がり方は小数 4 桁まで同じ（0.2119 → 0.1283）。
- **Adam は `make_adam` を使う**：`torch.optim.Adam` は更新の一部（`aten::lerp`）が DirectML になく、CPU に回されて遅い
  （1 組で 104〜136 ms、`make_adam` の 2.5 倍）。`make_adam` は同じ式を GPU で動く演算だけで書いたもので、CPU では
  `torch.optim.Adam` と同じ重みになる（テスト `test_plain_adam_matches_torch_adam`）。
- **一度に学習する組は 2 まで（多くても 4）にする**：6 組で GPU のメモリ（20 GB）があふれて急に遅くなり、10 組（`configs/train.yaml` の
  `batch_size` の値）は「DML allocator out of memory」で止まる。4 組は、同じプロセスで先に別のモデルを動かしていると、
  エラーの表示なしにプロセスごと落ちた（終了コード 0xC0000005）。DirectML は使い終わったメモリをすぐには返さないようなので、
  学習は 1 つのプロセスで 1 つのモデルにする。1 秒に学習できる組の数は 1〜4 組でほぼ同じ（約 20〜23 組）なので、小さくしても遅くはならない。
  ただし BatchNorm は組が少ないと不安定になりやすいので、F1 で組の数と精度の関係を見る。
- 目安：1 万組の学習データなら、1 周（エポック）が GPU で約 8 分、CPU で約 6 時間。

## エージェントの正解での学習（F1b、2026-11-08）

道具：`smash_speed/train/tracknet_train.py`。TrackNetV3 の `train.py` は CUDA と「ラリーの全コマに正解がある」ことを前提にしているので使わず、
`torch_device` の装置・Adam で自前に回す。F1a の正解はとびとびのコマで、「印はシャトルでない」のようにコマのほかの所が分からない答えも
あるので、損失は正解のある所だけで数える（見える・写っていない：コマ全体、シャトルでない印：候補の周り半径 8 px（288×512 で）、
分からない・隣のコート・正解のないコマ：数えない）。入力は TrackNetV3（`bg_mode: concat`）と同じ、背景（区間の 40 コマの中央値）＋ 8 コマの RGB。

```
# .venv：正解のコマを含む 8 コマの区間を 288×512 にして書き出す（学習用・答え合わせ用）
.venv\Scripts\python -m smash_speed.train.tracknet_train build outputs\f1b\train outputs\train_labels\IMG_3391 ...
.venv\Scripts\python -m smash_speed.train.tracknet_train build --contiguous outputs\f1b\eval outputs\eval_labels\IMG_4397_80_100 outputs\eval_labels\IMG_5109_2
# .venv-train（PYTHONPATH にこのリポジトリ）：学習と、答え合わせ用の熱の地図
.venv-train\Scripts\python -m smash_speed.train.tracknet_train train outputs\f1b\train outputs\f1b\run --epochs 30 --init ckpts\tracknetv3\TrackNet_best.pt
.venv-train\Scripts\python -m smash_speed.train.tracknet_train predict outputs\f1b\eval outputs\f1b\run\model.pt
# .venv：熱の地図を点にして（閾値 0.5 の最も大きな塊の中心）、今の検出（eval_labels の eval.json）と比べる
.venv\Scripts\python -m smash_speed.train.tracknet_train eval outputs\f1b\eval outputs\f1b\run\model_pred.npy
# 閾値を並べて比べる（<熱の地図>_sweep.json に書く）
.venv\Scripts\python -m smash_speed.train.tracknet_train eval outputs\f1b\eval outputs\f1b\run\model_pred.npy --threshold 0.4 0.45 0.5 0.55
```

- 重み：学習したものは `<出力>/model.pt`（検証の損失がいちばん小さいとき）と `model_last.pt`。`outputs/` の中なので Git に入らない。
  TrackNetV3 の学習済みの重みは `ckpts/tracknetv3/TrackNet_best.pt` に置く（`ckpts/` も Git に入らない）。`--init` は層の並びが
  同じなので名前ではなく順番で対応させて読む（形が違えば止まる）。
- 書き出しの量：学習用 8 フォルダー（9 区間）・652 区間（正解 902 コマ：見える 509・シャトルでない印 393）で 2.3 GB、約 6.5 分。
- 乱数からの学習（2026-11-08、学習済みの重みがないため）：検証 65 区間、1 周 約 60 秒（RX 7900 XT、2 組）、30 周で損失 0.166 → 0.0003。
  答え合わせ用では、閾値 0.5 でシャトルを 1 つも出さず、正解の所が熱の地図のいちばん高い所になるのは見える 527 コマの 12% だけ
  （閾値を 0.1〜0.2 に下げても適合率 0.02〜0.36）。1,000 コマほどのとびとびの正解では、乱数からはシャトルを覚えない。
  追加学習には TrackNetV3 の学習済みの重みが要る（`agent/QUESTIONS.md` の F1b）。

### 学習済みの重みからの追加学習（2026-11-09）

学習用 12 フォルダー（13 区間）・1,075 区間（正解 1,405 コマ）、3.6 GB・約 10 分で書き出し（`outputs/f1b/train2`）。
`--init ckpts\tracknetv3\TrackNet_best.pt --epochs 15 --lr 1e-4`（2 組）で 1 周 約 115 秒、15 周 28 分（`outputs/f1b/ft1/model.pt`、
検証の損失 0.00065 → 0.00007。周ごとの検証の損失は 0.00007〜0.00018 で上下する）。

答え合わせ用（コマごとの熱の地図の点。軌道の絞り込みはしていない。距離の許し：幅 512 px で 4 px）：

| | IMG_4397 80〜100 s（4K） | IMG_5109_2 |
|---|---|---|
| 今の検出（選んだ軌道） | 再現率 0.936・適合率 0.953・F1 0.945 | 0.877・0.902・0.889 |
| 学習済みのまま（閾値 0.5） | 0.962・0.813・0.881（ずれの中央値 13 px） | 0.870・0.855・0.862（8 px） |
| 追加学習（閾値 0.5） | 0.789・1.000・0.882 | 0.932・0.982・0.957 |
| 追加学習（閾値 0.25） | 0.929・0.977・0.952（4.5 px） | 0.983・0.983・0.983（3.1 px） |
| 追加学習（閾値 0.2） | 0.943・0.981・0.962 | 0.987・0.975・0.981 |

- 学習済みのままは熱の地図の高さが 0.5〜0.7 にとどまり、シャトルでない所（遠くの点）も 0.4〜0.6 になる。閾値で適合率を上げると再現率が急に落ちる。
- 追加学習で「写っていない」コマの最大は中央値 0.05〜0.07 に下がり、シャトルでない所を出さなくなった。代わりにシャトルの所も
  0.3〜0.8 と低めに出るので、閾値は 0.5 より下（0.2〜0.3）が合う。閾値は答え合わせ用を見て選んだので、数字は少し良く出ている。
- 位置のずれは 4K で 13 → 4.5 px。正解は候補の点に寄せてあるので、今の検出のずれ（1 px）は小さく出る。

### 動画の全コマの検出（2026-11-09）

学習したモデルで動画の区間の全コマを検出し、今の検出と同じ形の CSV（`frame, timestamp, x_px, y_px, visible, confidence, track_id, heat`）
にする。`.venv` から動かし、熱の地図だけ `.venv-train` を呼ぶ。

```
.venv\Scripts\python.exe -m smash_speed.train.tracknet_train detect IMG_4397.MOV outputs\f1b\ft1\model.pt outputs\f1b\det\IMG_4397_80_100 --start 80 --end 100 --threshold 0.25
```

- 中では `frames`（区間の全コマを 8 コマずつ重ならずに 288×512 で書き出す。割り切れない分は最後のコマで埋める）→
  `predict`（`.venv-train`）→ `detections`（熱の地図の点）の順。
- `confidence` は `detection.tracknet` と同じく 1/0（見つかったか）。0〜1 の間の値を入れると、`pipeline` の軌道の選び方が
  今の検出の軌道のスコアとして読んでしまう。熱の地図の最大は `heat` の列に入れる。
- 点は 1 コマ 1 つなので、軌道は「4 コマ以上空いた」「1 コマで画像の幅の 0.15 より大きく跳んだ」「1 コマの動きが大きいまま向きが
  90° より大きく変わった（打球）」所で分ける（`split_tracks`）。打球で分けないと、打つ前の球と返球が 1 本の軌道になる。
- 速さ：4K の 600 コマ（20 秒）で 24 秒（読み込み 15 秒・推論 7 秒）。1080p の 287 コマで 10 秒。
- 答え合わせ用での数字は、区間ごとに測った前の表とほぼ同じ（閾値 0.25。IMG_4397 80〜100 s 再現率 0.924・適合率 0.973、
  IMG_5109_2 0.983・0.983）。ショットの一覧と同じ軌道の絞り込みの後は 0.907・0.981（F1 0.942）、0.983・0.987（F1 0.985）。

打球の見つけ方（`check_real_hits` の 4 区間で、今の検出の代わりにこの検出を使った。正解は `agent/truth/`）：

| | 再現率 | 適合率 | 打った側 |
|---|---|---|---|
| 今の検出 | 0.929（65/70） | 0.823 | 0.969 |
| 学習したモデル（閾値 0.25） | 0.971（68/70） | 0.944 | 1.000 |
| 学習したモデル（閾値 0.5） | 0.929 | 0.942 | 1.000 |

- 区間ごと（閾値 0.25、今の検出 → モデル）：IMG_5109 29 → 32/32、IMG_5109_1 13 → 14/15、IMG_5109_2 9 → 9/9、IMG_4397 80〜100 s 14 → 13/14。
  IMG_5109・IMG_5109_1 は学習に使った動画（の別のコマ）なので、良く出ているかもしれない。
- 答え合わせ用の正解のうち 2 本は時刻が誤っていた（IMG_5109_2 0.84 s → 0.34 s、IMG_4397 92.98 s → 92.52 s。コマを拡大して確かめた。
  `agent/QUESTIONS.md`）。モデルは IMG_4397 で正しい 92.54 s を出している。正解を直すと今の検出 0.900（63/70）、モデル 0.971（68/70）。
- 今の検出が出していた IMG_4397 97 s 台の余計な 2 本（ラリーの後）は、モデルでは出ない。

スマッシュの初速（`check_real_videos` の 15 本）：

- モデルの検出を `--detections` で直接渡すと 15 本とも崩れる（例：IMG_4397 #50 は 317 → 18 km/h）。外から渡した検出では、
  打点から打ち出した軌道の探索（`guided`）が働かず、古い選び方（打球音の直後に始まるか・打点の目安に近いか）だけで選ぶため、
  打った直後の 1〜2 コマを逃したモデルの軌道より、打つ前に落ちてくる球の軌道が選ばれる。
- そこで、モデルの点を軌道の探索の「候補」として渡す口を作った（`AnalyzeOptions.candidates`、`smash-speed analyze --candidates`）。
  打点からの探索・漏れたコマの探し直しは自前で行い、コマごとに予測位置にいちばん近い候補を使う。
  既定は自前の候補に足す。`candidates_only`（`--candidates-only`）では自前の検出をせず、モデルの点だけを候補にする。
- GUI の「詳しく解析する」と同じ手順（自前の探索で解析し、ずれが 5 px を超えたら①の検出でも解析してずれの小さい方）で
  15 本を解析した（閾値 0.25。`.agent/tmp/f1b/cmp_speeds3.py`。モデルを使わないと基準と 15 本とも一致する）：

| 候補 | 基準からの変化 | 正しい初速との差 \|差\|/σ（11 本の中央値・合計） |
|---|---|---|
| 今の検出（基準） | — | 0.45・6.6 |
| 今の検出 + モデルの点 | 最大 2.5%（6 本が 0.03〜2.5% 変わり、9 本は同じ） | 0.45・6.7 |
| モデルの点だけ | 最大 47%（10 本が変わる） | 0.78・7.8 |

- モデルの点を足しても初速はほとんど変わらない。今の検出の候補と探し直しで、打った後の軌道はもう拾えているため。
  変わった 6 本も、正しい初速に近づいたもの（IMG_4397 #50 0.18 → 0.08σ）と離れたもの（#60 0.25 → 0.37σ）があり、良し悪しはつかない。
- モデルの点だけにすると悪くなる。モデルは打った直後のぶれの長い 1〜2 コマを逃し（IMG_4397 #50 は打点の次のコマに画面の別の所の点を出す）、
  点の位置が熱の地図の格子（4K で 7.5 px）に丸められる。30fps では初速が最初の数コマで決まるので、これが効く。
- まとめ：学習したモデルは打球の見つけ方（上の表）を良くするが、スマッシュの初速は良くしない。スマッシュの解析の既定の検出は
  替えず、切り替えるならショットの一覧（打球の見つけ方）から考える（F1c）。
