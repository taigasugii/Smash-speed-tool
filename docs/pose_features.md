# フォームの数値（姿勢推定と特徴量）

ROADMAP の E 章（フォームと姿勢）で作った処理の、式と理由をまとめる。
数値はすべて側方の撮影（ラケットを持つ腕の側から、打つ向きに直交、距離 6〜8 m、高さ 1.2〜1.5 m、水平）を前提にしている。
画像に写った 2 次元の位置から出すので、カメラに向かう向き（奥行き）の動きは見えない。どの値がその影響を受けるかを、各項に書いた。

## 1. 姿勢推定（E1、`smash_speed/pose.py`）

### 関節の決まり

関節は COCO の 17 点（鼻・両目・両耳・両肩・両肘・両手首・両腰・両膝・両足首）にそろえる。
RTMPose はもともと 17 点、MediaPipe Pose は 33 点のうち同じ所の 17 点を使う（`MEDIAPIPE_TO_COCO`）。
信頼度は 0〜1（RTMPose は 1 を超える値を 1 に収める、MediaPipe は visibility と presence の小さい方）。

### モデル

| モデル | ライセンス | 重み | GPU | 入れ方 |
|---|---|---|---|---|
| RTMPose（rtmlib） | Apache 2.0（rtmlib・RTMPose のコード） | `balanced` で人の検出（YOLOX-m）と関節（RTMPose-m）で約 100 MB。初めて使うときに OpenMMLab の配布元から取ってくる | ONNX なので `accel.make_session` に載せ替え、デスクトップでは DirectML | `pip install --no-deps rtmlib tqdm`（下の注意） |
| MediaPipe Pose Landmarker | Apache 2.0（コードと重み） | lite 約 6 MB・full 約 9 MB・heavy 約 30 MB。`ckpts/pose/` に取ってくる | Python 版は CPU だけ | `pip install -e .[pose]` |

- rtmlib は `onnxruntime`（CPU 版）を一緒に入れようとする。Windows で `onnxruntime-directml` と両方入ると壊れる（`docs/gpu.md`）ので、
  `--no-deps` で入れる。
- RTMPose の重みは body7（COCO・AI Challenger・CrowdPose・MPII・Halpe など 7 つのデータの集まり）で学習されている。
  コードは Apache 2.0 だが、学習に使ったデータには研究用に限るものが含まれる。個人の解析に使う分には問題になりにくいが、
  解析の結果やモデルを配るときは確かめ直す。
- どちらのモデルを使うかは、手で付けた関節（約 200 コマ）との PCK で決める（下）。まだ人が付けていないので決めていない。

### 打つ人の選び方

1 コマに複数の人（相手・審判・通りかかった人）が写ることがある。最初のコマは関節の範囲（外接の四角の面積）がいちばん大きい人
（側方の近くから撮るので、打つ人がいちばん大きく写る）、次のコマからは前のコマの打つ人の中心にいちばん近い人を選ぶ。
画像の幅の 15% より離れた人は選ばず、10 コマ続けて見失ったら選び直す。

### 信頼度の低い関節

信頼度が 0.3 より低い関節は欠け（NaN）にする。体の陰になった非利き腕・奥の脚は信頼度が下がるので、
推定の位置をそのまま使うと角度が大きく狂う。3 コマ以下の欠けは前後から直線で埋め（60fps で 0.05 秒。手首の速い動きでも
この長さなら形が大きく変わらない）、それより長い欠けは残して、その関節を使う特徴量を「出せない」にする。

### 精度（PCK）

PCK（Percentage of Correct Keypoints）：推定と手で付けた位置の距離が「胴体の長さ × α」以内の関節の割合。α = 0.2 を使う。

- 胴体の長さ = 手で付けた肩の中点と腰の中点の距離。人の大きさ（カメラからの距離）によらず比べられる。
  側方の撮影では肩幅が向きによって 0 近くまで縮むので、COCO の頭の大きさや肩幅ではなく胴体の長さにした。
- 推定で欠け・信頼度が低い関節は外れに数える（「見つけられなかった」も間違いのうち）。
- 隠れている（visible = 1）関節は、既定では数えない（人が付けた位置も推測なので）。

手で付けた関節の CSV（`pose.LABEL_COLUMNS`）は 1 行 1 関節：

```
video,frame,t_s,joint,x,y,visible
clear_01.mp4,612,10.200000,right_wrist,1043.5,402.0,2
```

`frame` は動画のコマの番号、`t_s` はその実時間 [s]（比べるときは時刻で推定のコマと合わせる）、`joint` は `pose.KEYPOINTS` の名前、
`visible` は COCO と同じ（0 付けていない／1 隠れているが位置は推測／2 見える）。

付け方の目安：200 コマは、ショット 20 本 × 10 コマ（構え・テイクバック・振り出し・インパクトの前後 3 コマ・フォロースルー）にすると、
速く動いてぶれたコマも含まれる。関節は骨の中心（肩は肩峰ではなく腕の付け根の中心）に付ける。

```
python -m smash_speed.pose estimate clear_01.mp4 --model rtmpose --start 10 --end 12 --out rtmpose.npz
python -m smash_speed.pose estimate clear_01.mp4 --model mediapipe --start 10 --end 12 --out mediapipe.npz
python -m smash_speed.pose pck labels.csv rtmpose.npz mediapipe.npz
```

合成の骨格（`synth_pose.py`）に正規分布の揺れ σ を足すと、誤差は Rayleigh 分布になり、PCK = 1 − exp(−(αL)²/2σ²) になることを
テスト（`tests/test_pose.py`）で確かめている。
