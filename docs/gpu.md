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
- **学習（TrackNetV3 の追加学習など、F1）は PyTorch が要る。** 候補は torch-directml と AMD の ROCm 版 PyTorch。
  どちらも数百 MB〜数 GB あるので、入れる前に `agent/QUESTIONS.md` で確認している（A2b）。

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
| torch-directml | torch 本体と合わせて数百 MB 以上 | 未（確認待ち） | 学習（F1）。対応する torch の版が古めに固定される |
| ROCm for Windows 版 PyTorch | 数 GB | 未（確認待ち） | 学習（F1）。RX 7900 XT は対応の GPU。版の組み合わせの制約が強い |
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
