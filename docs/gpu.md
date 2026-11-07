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
```

- 重み：学習したものは `<出力>/model.pt`（検証の損失がいちばん小さいとき）と `model_last.pt`。`outputs/` の中なので Git に入らない。
  TrackNetV3 の学習済みの重みは `ckpts/tracknetv3/TrackNet_best.pt` に置く（`ckpts/` も Git に入らない）。`--init` は層の並びが
  同じなので名前ではなく順番で対応させて読む（形が違えば止まる）。
- 書き出しの量：学習用 8 フォルダー（9 区間）・652 区間（正解 902 コマ：見える 509・シャトルでない印 393）で 2.3 GB、約 6.5 分。
- 乱数からの学習（2026-11-08、学習済みの重みがないため）：検証 65 区間、1 周 約 60 秒（RX 7900 XT、2 組）、30 周で損失 0.166 → 0.0003。
  答え合わせ用では、閾値 0.5 でシャトルを 1 つも出さず、正解の所が熱の地図のいちばん高い所になるのは見える 527 コマの 12% だけ
  （閾値を 0.1〜0.2 に下げても適合率 0.02〜0.36）。1,000 コマほどのとびとびの正解では、乱数からはシャトルを覚えない。
  追加学習には TrackNetV3 の学習済みの重みが要る（`agent/QUESTIONS.md` の F1b）。
