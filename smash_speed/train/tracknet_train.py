"""エージェントが付けたシャトルの正解（F1a）で、TrackNetV3 と同じ形のモデルを学習し、答え合わせ用の正解で比べる（F1b）。

TrackNetV3 の ``train.py`` は CUDA を前提にし、正解が続いたコマ（ラリーの全コマ）にあることを前提にしている。
F1a の正解はとびとびのコマ（怪しいコマを選んで付けた）で、「印はシャトルでない」のように、そのコマのほかの所は
分からない答えもある。そこで、学習は自前の繰り返し（``torch_device`` の装置・Adam）で行い、損失を正解のある所だけで数える。

- 見える（``visible`` 1）：そのコマの全体で数える（シャトルの所に半径 2.5 px の円。TrackNetV3 と同じ）。
- 写っていない（0）：そのコマの全体で数える（すべて 0）。
- 印はシャトルでない（``agent_x``）：候補の点の周り（半径 ``NEG_RADIUS``）だけを 0 として数える。
- 分からない・隣のコートのシャトル・正解のないコマ：数えない。

入力は TrackNetV3（``bg_mode: concat``）と同じで、背景（区間のコマの中央値）＋ 8 コマの RGB を 288×512 にして 0〜1 に
したもの（27 チャンネル。背景が先）。出力は 8 コマ分の熱の地図。

流れ（torch は学習用の ``.venv-train`` にしかないので、動画を読む所と学習の所を分ける）::

    # .venv：正解のコマとその前後を 288×512 にして書き出す（学習用・答え合わせ用）
    python -m smash_speed.train.tracknet_train build outputs/f1b/train outputs/train_labels/IMG_5109 ...
    python -m smash_speed.train.tracknet_train build --contiguous outputs/f1b/eval outputs/eval_labels/IMG_5109_2 ...
    # .venv-train：学習（--init で TrackNetV3 の学習済みの重みから始める）と、答え合わせ用の熱の地図
    python -m smash_speed.train.tracknet_train train outputs/f1b/train outputs/f1b/run1 --epochs 20
    python -m smash_speed.train.tracknet_train predict outputs/f1b/eval outputs/f1b/run1/model.pt
    # .venv：熱の地図を点にして、今の検出（eval.json）と比べる
    python -m smash_speed.train.tracknet_train eval outputs/f1b/eval outputs/f1b/run1/model_pred.npy

動画の区間の全コマを検出して、今の検出と同じ形の CSV（``--detections`` に渡せる）にするには::

    # .venv：区間の全コマを書き出す → 熱の地図（.venv-train を呼ぶ）→ 検出の CSV。3 つをまとめて行う
    python -m smash_speed.train.tracknet_train detect IMG_4397.MOV outputs/f1b/run1/model.pt outputs/f1b/det/IMG_4397 --start 80 --end 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HEIGHT, WIDTH = 288, 512   # TrackNetV3 の入力の大きさ
SEQ_LEN = 8                # TrackNetV3 の学習済みの重みの seq_len
POS_RADIUS = 2.5           # シャトルの円の半径 [px]（288×512 で。TrackNetV3 と同じ）
NEG_RADIUS = 8.0           # 「シャトルでない」候補の周りで 0 として数える半径 [px]（288×512 で）
N_BG = 40                  # 背景（中央値）に使うコマの数

# 正解の種類（targets の 0 列目）
IGNORE, ABSENT, VISIBLE, NOT_SHUTTLE = -1, 0, 1, 2


# =============================================================================
# 正解 → 熱の地図・数える所（numpy だけ。.venv と .venv-train の両方で使う）
# =============================================================================
def target_and_mask(kind: int, x: float, y: float, h: int = HEIGHT, w: int = WIDTH) -> tuple[np.ndarray, np.ndarray]:
    """1 コマの正解の熱の地図（0/1）と、損失を数える所（0/1）。``x``・``y`` は ``h``×``w`` の px。"""
    target = np.zeros((h, w), np.float32)
    mask = np.zeros((h, w), np.float32)
    if kind in (VISIBLE, NOT_SHUTTLE):
        yy, xx = np.ogrid[:h, :w]
        d2 = (xx - x) ** 2 + (yy - y) ** 2
    if kind == VISIBLE:
        target[d2 <= POS_RADIUS ** 2] = 1.0
        mask[:] = 1.0
    elif kind == ABSENT:
        mask[:] = 1.0
    elif kind == NOT_SHUTTLE:
        mask[d2 <= NEG_RADIUS ** 2] = 1.0
    return target, mask


def clip_starts(frames: np.ndarray, n_total: int, seq_len: int = SEQ_LEN, rng: np.random.Generator | None = None,
                contiguous: bool = False) -> list[int]:
    """正解のあるコマ（番号）がどれかの区間（``seq_len`` コマ）に入るように、区間の先頭を選ぶ。

    とびとびの正解では、まだ入っていないコマごとに、それが区間のどこかに来るよう（乱数で）先頭を決める。
    ``contiguous`` なら、最初のコマから重ならずに並べる（答え合わせ用。続いたコマの正解）。
    """
    fs = np.sort(np.unique(np.asarray(frames, int)))
    if len(fs) == 0 or n_total < seq_len:
        return []
    last = n_total - seq_len
    if contiguous:
        return [min(s, last) for s in range(int(fs[0]), int(fs[-1]) + 1, seq_len)]
    rng = rng or np.random.default_rng(0)
    starts: list[int] = []
    for f in fs:
        if starts and starts[-1] <= f < starts[-1] + seq_len:
            continue
        starts.append(int(np.clip(f - rng.integers(0, seq_len), 0, last)))
    return starts


def masked_wbce(y_pred, y, mask):
    """TrackNetV3 の重み付き 2 値交差エントロピーを、``mask`` の所だけで平均する（torch）。"""
    import torch

    p = torch.clamp(y_pred, 1e-7, 1 - 1e-7)
    loss = -(torch.square(1 - p) * y * torch.log(p) + torch.square(p) * (1 - y) * torch.log(1 - p))
    return (loss * mask).sum() / torch.clamp(mask.sum(), min=1.0)


# =============================================================================
# 書き出し（.venv。動画を読む）
# =============================================================================
def load_label_dir(d: Path):
    """``labels.csv`` を (frame, kind, x_px, y_px) にする。「印はシャトルでない」は ``candidates.csv`` の候補の点を使う。"""
    import pandas as pd

    labels = pd.read_csv(d / "labels.csv")
    kind = np.full(len(labels), IGNORE)
    kind[labels.visible.to_numpy() == 1] = VISIBLE
    kind[labels.visible.to_numpy() == 0] = ABSENT
    src = labels.source.astype(str).to_numpy()
    kind[src == "agent_x"] = NOT_SHUTTLE
    out = pd.DataFrame({"frame": labels.frame.astype(int), "kind": kind,
                        "x_px": labels.x_px.astype(float), "y_px": labels.y_px.astype(float)})
    cands_csv = d / "candidates.csv"
    if (out.kind == NOT_SHUTTLE).any() and cands_csv.exists():
        c = pd.read_csv(cands_csv).drop_duplicates("frame", keep="last").set_index("frame")
        m = (out.kind == NOT_SHUTTLE) & out.frame.isin(c.index)
        out.loc[m, "x_px"] = c.loc[out.frame[m], "cand_x"].to_numpy()
        out.loc[m, "y_px"] = c.loc[out.frame[m], "cand_y"].to_numpy()
        out.loc[(out.kind == NOT_SHUTTLE) & out.x_px.isna(), "kind"] = IGNORE
    return out[out.kind != IGNORE].reset_index(drop=True)


def _to_input(frame_bgr: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(cv2.resize(frame_bgr, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def build_dataset(label_dirs: list[Path], out: Path, seq_len: int = SEQ_LEN, contiguous: bool = False, seed: int = 0,
                  progress=print) -> dict[str, Any]:
    """正解のフォルダー（``labels.csv`` と ``candidates.csv`` の動画）から、学習・答え合わせ用のデータを書き出す。

    書くもの：``frames.npy``（区間ごとの ``seq_len`` コマ、288×512 の RGB、uint8）、``bgs.npy``（フォルダーごとの背景）、
    ``clips.npy``（区間ごとに [背景の番号, 先頭のコマ番号]）、``targets.npy``（区間 × コマ × [種類, x, y]。288×512 の px）、
    ``meta.json``（フォルダー・動画・縮尺）。
    """
    import pandas as pd

    from smash_speed.config import load_config
    from smash_speed.train.agent_labels import _video_path
    from smash_speed.video_io import VideoReader, probe

    cfg = load_config(None)
    rng = np.random.default_rng(seed)
    plan = []
    for d in label_dirs:
        lab = load_label_dir(d)
        video = str(pd.read_csv(d / "candidates.csv").video.iloc[0])
        meta = probe(_video_path(video), None, cfg)
        starts = clip_starts(lab.frame.to_numpy(), len(meta.timestamps), seq_len, rng, contiguous)
        plan.append((d, video, meta, lab, starts))
    n_clips = sum(len(p[4]) for p in plan)
    out.mkdir(parents=True, exist_ok=True)
    frames = np.lib.format.open_memmap(out / "frames.npy", mode="w+", dtype=np.uint8,
                                       shape=(n_clips * seq_len, HEIGHT, WIDTH, 3))
    bgs = np.zeros((len(plan), HEIGHT, WIDTH, 3), np.uint8)
    clips = np.zeros((n_clips, 2), np.int64)
    targets = np.full((n_clips, seq_len, 3), np.nan, np.float32)
    targets[:, :, 0] = IGNORE
    dirs_meta = []
    k = 0
    t0 = time.time()
    for di, (d, video, meta, lab, starts) in enumerate(plan):
        sx, sy = WIDTH / meta.width, HEIGHT / meta.height
        reader = VideoReader(meta)
        lo, hi = min(starts), max(starts) + seq_len - 1
        bg_idx = np.unique(np.linspace(lo, hi, N_BG).round().astype(int))
        bgs[di] = np.median(np.stack([_to_input(reader.read_frame(int(i))) for i in bg_idx]), axis=0).astype(np.uint8)
        by_frame = lab.set_index("frame")
        prev_end = -10
        for s in starts:
            for j in range(seq_len):
                f = s + j
                # 前の区間の続きなら順に読む（4K の動画は飛ぶのが遅い）
                if f == prev_end + 1:
                    ok, img = reader.cap.read()
                    img = img if ok else reader.read_frame(f)
                else:
                    img = reader.read_frame(f)
                prev_end = f
                frames[k * seq_len + j] = _to_input(img)
                if f in by_frame.index:
                    r = by_frame.loc[f]
                    targets[k, j] = (r.kind, r.x_px * sx, r.y_px * sy)
            clips[k] = (di, s)
            k += 1
        dirs_meta.append({"dir": str(d), "video": video, "width": int(meta.width), "height": int(meta.height),
                          "n_clips": len(starts), "n_labels": int(len(lab)),
                          "counts": {name: int((lab.kind == v).sum()) for name, v in
                                     (("visible", VISIBLE), ("absent", ABSENT), ("not_shuttle", NOT_SHUTTLE))}})
        progress(f"{d.name}: 区間 {len(starts)}・正解 {len(lab)}（{time.time() - t0:.0f} 秒）")
    frames.flush()
    np.save(out / "bgs.npy", bgs)
    np.save(out / "clips.npy", clips)
    np.save(out / "targets.npy", targets)
    info = {"seq_len": seq_len, "size": [HEIGHT, WIDTH], "contiguous": contiguous, "seed": seed,
            "n_clips": n_clips, "dirs": dirs_meta}
    (out / "meta.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return info


def build_video(video: Path, out: Path, start_s: float | None = None, end_s: float | None = None,
                seq_len: int = SEQ_LEN, progress=print) -> dict[str, Any]:
    """動画の区間の全コマを、重ならない ``seq_len`` コマずつの区間として書き出す（正解なし。``predict`` に渡す）。

    ``build_dataset`` と同じ形に、区間のコマの番号と実時間（``frame_times.npy``：[番号, 時刻]）を足す。
    コマの数が ``seq_len`` で割り切れなければ、最後のコマを繰り返して埋める（埋めたコマの予測は使わない）。
    背景は区間から等間隔に選んだ ``N_BG`` コマの中央値。
    """
    from smash_speed.config import load_config
    from smash_speed.train.agent_labels import _video_path
    from smash_speed.video_io import VideoReader, probe

    path = _video_path(str(video))
    meta = probe(path, None, load_config(None))
    ts = meta.timestamps
    i0 = 0 if start_s is None else int(np.searchsorted(ts, start_s))
    i1 = len(ts) - 1 if end_s is None else int(np.searchsorted(ts, end_s, side="right")) - 1
    n_frames = i1 - i0 + 1
    if n_frames < 1:
        raise ValueError("区間にコマがありません")
    n_clips = -(-n_frames // seq_len)
    out.mkdir(parents=True, exist_ok=True)
    frames = np.lib.format.open_memmap(out / "frames.npy", mode="w+", dtype=np.uint8,
                                       shape=(n_clips * seq_len, HEIGHT, WIDTH, 3))
    t0 = time.time()
    k = 0
    with VideoReader(meta) as reader:
        for i, _, img in reader.iter_range(float(ts[i0]) - 1e-6, float(ts[i1]) + 1e-6):
            frames[k] = _to_input(img)
            k += 1
            if k % 600 == 0:
                progress(f"{k}/{n_frames} コマ（{time.time() - t0:.0f} 秒）")
    if k == 0:
        raise OSError(f"{path} を読めません")
    frames[k:] = frames[k - 1]   # 読めなかった分・割り切れない分は最後のコマで埋める
    frames.flush()
    bg_idx = np.unique(np.linspace(0, k - 1, N_BG).round().astype(int))
    np.save(out / "bgs.npy", np.median(frames[bg_idx], axis=0).astype(np.uint8)[None])
    np.save(out / "clips.npy", np.array([[0, i0 + seq_len * c] for c in range(n_clips)], np.int64))
    targets = np.full((n_clips, seq_len, 3), np.nan, np.float32)
    targets[:, :, 0] = IGNORE
    np.save(out / "targets.npy", targets)
    np.save(out / "frame_times.npy", np.stack([np.arange(i0, i0 + k), ts[i0:i0 + k]], axis=1))
    info = {"seq_len": seq_len, "size": [HEIGHT, WIDTH], "contiguous": True, "video": str(path),
            "start_s": start_s, "end_s": end_s, "n_clips": n_clips, "n_frames": k,
            "dirs": [{"dir": str(out), "video": path.name, "width": int(meta.width), "height": int(meta.height),
                      "n_clips": n_clips, "n_labels": 0}]}
    (out / "meta.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    progress(f"{path.name}：{k} コマ・{n_clips} 区間（{time.time() - t0:.0f} 秒）")
    return info


# =============================================================================
# 学習・推論（.venv-train。torch）
# =============================================================================
class ClipData:
    """書き出したデータを読む（``frames.npy`` はメモリに載せずに読む）。"""

    def __init__(self, data_dir: Path):
        self.dir = Path(data_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        self.seq_len = int(self.meta["seq_len"])
        self.frames = np.load(self.dir / "frames.npy", mmap_mode="r")
        self.bgs = np.load(self.dir / "bgs.npy")
        self.clips = np.load(self.dir / "clips.npy")
        self.targets = np.load(self.dir / "targets.npy")

    def __len__(self) -> int:
        return len(self.clips)

    def item(self, i: int, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """区間 ``i`` の入力（27×288×512、0〜1）・熱の地図・数える所。``rng`` があれば左右反転・明るさを変える。"""
        n = self.seq_len
        imgs = np.concatenate([self.bgs[self.clips[i, 0]][None], self.frames[i * n:(i + 1) * n]]).astype(np.float32)
        tg = self.targets[i].copy()
        if rng is not None:
            if rng.random() < 0.5:
                imgs = imgs[:, :, ::-1]
                tg[:, 1] = WIDTH - 1 - tg[:, 1]
            imgs = imgs * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20)
        x = np.clip(imgs / 255.0, 0, 1).transpose(0, 3, 1, 2).reshape(-1, HEIGHT, WIDTH)
        ys, ms = zip(*(target_and_mask(int(k), float(px), float(py)) for k, px, py in tg))
        return np.ascontiguousarray(x, np.float32), np.stack(ys), np.stack(ms)

    def batch(self, idx, rng=None):
        xs, ys, ms = zip(*(self.item(int(i), rng) for i in idx))
        return np.stack(xs), np.stack(ys), np.stack(ms)


def load_tracknetv3_weights(model, ckpt_path: Path) -> int:
    """TrackNetV3 の学習済みの重み（``TrackNet_best.pt``）を ``build_tracknet`` のモデルに入れ、入れた数を返す。

    層の並びが同じなので、名前ではなく順番で対応させる（形が違えば止める）。TrackNetV3 の畳み込みは後ろに BatchNorm が
    続くので偏り（bias）を持たない。このモデルの畳み込みの偏りで学習済みに対応するものがなければ 0 にする
    （BatchNorm の前の偏りは 0 と同じ働きになるので、学習済みのモデルと同じ出力になる）。
    """
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)  # 重みの数字だけを読む（中のプログラムを動かさない）
    src = list((ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt).items())
    dst = model.state_dict()
    new, j, prev_dim = {}, 0, 0
    for kd, vd in dst.items():
        conv_bias = kd.endswith(".bias") and prev_dim == 4  # 畳み込みの重みのすぐ後の偏り
        prev_dim = vd.dim()
        if conv_bias and j < len(src) and not src[j][0].endswith(".bias"):
            new[kd] = torch.zeros_like(vd)  # 学習済みにない畳み込みの偏り
        elif j < len(src) and tuple(src[j][1].shape) == tuple(vd.shape):
            new[kd] = src[j][1]
            j += 1
        else:
            ks, vs = src[j] if j < len(src) else ("（なし）", torch.empty(0))
            raise ValueError(f"重みの形が違います：{kd} {tuple(vd.shape)} ← {ks} {tuple(vs.shape)}")
    if j != len(src):
        raise ValueError(f"学習済みの重みが余りました（{len(src) - j} 個。モデルの形が違います）")
    model.load_state_dict(new)
    return j


def _loss_on(model, data: ClipData, idx, device, batch: int) -> float:
    import torch

    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for b in range(0, len(idx), batch):
            x, y, m = (torch.from_numpy(a).to(device) for a in data.batch(idx[b:b + batch]))
            tot += float(masked_wbce(model(x), y, m)) * len(x)
            n += len(x)
    model.train()
    return tot / max(n, 1)


def train(data_dir: Path, out_dir: Path, epochs: int = 20, batch: int = 2, lr: float = 1e-4, init: Path | None = None,
          width: int = 64, val_frac: float = 0.1, seed: int = 0, prefer: str | None = None,
          max_minutes: float | None = None, progress=print) -> dict[str, Any]:
    """学習し、検証の損失がいちばん小さいときの重みを ``model.pt``、記録を ``history.json`` に書く。

    検証には学習用の区間から ``val_frac`` を分けて使う（答え合わせ用は使わない）。
    """
    import torch

    from smash_speed.train.torch_device import build_tracknet, make_adam, train_device

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = ClipData(data_dir)
    device, dev_name = train_device(prefer)
    model = build_tracknet(in_dim=3 * (data.seq_len + 1), out_dim=data.seq_len, width=width)
    n_init = load_tracknetv3_weights(model, init) if init else 0
    model = model.to(device)
    opt = make_adam(model.parameters(), lr=lr, device=device)
    order = rng.permutation(len(data))
    n_val = max(1, int(round(len(data) * val_frac)))
    val_idx, tr_idx = order[:n_val], order[n_val:]
    out_dir.mkdir(parents=True, exist_ok=True)
    hist: dict[str, Any] = {"device": dev_name, "data": str(data_dir), "init": str(init) if init else None,
                            "n_init_weights": n_init, "epochs": epochs, "batch": batch, "lr": lr, "width": width,
                            "n_train": int(len(tr_idx)), "n_val": int(n_val), "train_loss": [], "val_loss": []}
    best = _loss_on(model, data, val_idx, device, batch)
    hist["val_loss_init"] = best
    progress(f"{dev_name}：学習 {len(tr_idx)}・検証 {n_val} 区間、始めの検証の損失 {best:.5f}")
    t0 = time.time()

    def save(path: Path, epoch: int) -> None:
        torch.save({"epoch": epoch, "model": {k: v.cpu() for k, v in model.state_dict().items()},
                    "param_dict": {"seq_len": data.seq_len, "bg_mode": "concat", "width": width}}, path)

    save(out_dir / "model.pt", 0)
    hist["best_epoch"] = 0
    for ep in range(1, epochs + 1):
        perm = rng.permutation(tr_idx)
        tot = 0.0
        for b in range(0, len(perm), batch):
            x, y, m = (torch.from_numpy(a).to(device) for a in data.batch(perm[b:b + batch], rng))
            loss = masked_wbce(model(x), y, m)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(x)
        hist["train_loss"].append(tot / len(perm))
        vl = _loss_on(model, data, val_idx, device, batch)
        hist["val_loss"].append(vl)
        if vl < best:
            best, hist["best_epoch"] = vl, ep
            save(out_dir / "model.pt", ep)
        save(out_dir / "model_last.pt", ep)
        progress(f"{ep}/{epochs}：学習の損失 {hist['train_loss'][-1]:.5f}・検証 {vl:.5f}（{time.time() - t0:.0f} 秒）")
        (out_dir / "history.json").write_text(json.dumps(hist, ensure_ascii=False, indent=1), encoding="utf-8")
        if max_minutes and time.time() - t0 > 60 * max_minutes:
            progress("時間の上限で止めます")
            break
    hist["seconds"] = round(time.time() - t0, 1)
    (out_dir / "history.json").write_text(json.dumps(hist, ensure_ascii=False, indent=1), encoding="utf-8")
    return hist


def predict(data_dir: Path, ckpt: Path, out: Path | None = None, batch: int = 2, prefer: str | None = None) -> Path:
    """書き出したデータのすべての区間の熱の地図（区間 × コマ × 288 × 512、0〜255 の uint8）を ``.npy`` に書く。"""
    import torch

    from smash_speed.train.torch_device import build_tracknet, train_device

    data = ClipData(data_dir)
    device, _ = train_device(prefer)
    c = torch.load(ckpt, map_location="cpu", weights_only=True)
    model = build_tracknet(in_dim=3 * (data.seq_len + 1), out_dim=data.seq_len, width=int(c["param_dict"]["width"]))
    model.load_state_dict(c["model"])
    model = model.to(device).eval()
    out = out or ckpt.with_name(ckpt.stem + "_pred.npy")
    heat = np.lib.format.open_memmap(out, mode="w+", dtype=np.uint8, shape=(len(data), data.seq_len, HEIGHT, WIDTH))
    with torch.no_grad():
        for b in range(0, len(data), batch):
            idx = list(range(b, min(b + batch, len(data))))
            x = torch.from_numpy(data.batch(idx)[0]).to(device)
            heat[idx] = (model(x).cpu().numpy() * 255).round().astype(np.uint8)
    heat.flush()
    return out


# =============================================================================
# 答え合わせ（.venv）
# =============================================================================
def heatmap_point(hm: np.ndarray, threshold: float = 0.5) -> tuple[float, float] | None:
    """熱の地図（0〜1 か 0〜255）から、閾値を超えたいちばん大きな塊の中心（288×512 の px）。なければ None。

    TrackNetV3 の ``predict.py`` と同じく、塊の外枠の中心を使う。
    """
    import cv2

    scale = 255.0 if hm.dtype == np.uint8 else 1.0
    binary = (hm >= threshold * scale).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = stats[i, :4]
    return x + w / 2.0, y + h / 2.0


def predictions_table(data_dir: Path, heat: np.ndarray, threshold: float = 0.5):
    """熱の地図を、フォルダーごとの元の解像度の予測（frame, x_px, y_px, visible）にする。"""
    import pandas as pd

    meta = json.loads((Path(data_dir) / "meta.json").read_text(encoding="utf-8"))
    clips = np.load(Path(data_dir) / "clips.npy")
    n = int(meta["seq_len"])
    rows = []
    for k, (di, s) in enumerate(clips):
        dm = meta["dirs"][int(di)]
        sx, sy = dm["width"] / WIDTH, dm["height"] / HEIGHT
        for j in range(n):
            hm = np.asarray(heat[k, j])
            pt = heatmap_point(hm, threshold)
            rows.append({"dir": int(di), "frame": int(s) + j, "visible": int(pt is not None),
                         "x_px": pt[0] * sx if pt else np.nan, "y_px": pt[1] * sy if pt else np.nan,
                         "peak": float(hm.max()) / (255.0 if hm.dtype == np.uint8 else 1.0)})
    # 答え合わせ用の最後の区間は前と重なることがあるので、後の方を使う
    return pd.DataFrame(rows).drop_duplicates(["dir", "frame"], keep="last")


def evaluate(data_dir: Path, heat_path: Path, threshold: float = 0.5, px_at_512: float = 4.0) -> dict[str, Any]:
    """フォルダーごとに、正解（``labels.csv``）と学習したモデルの予測を ``detection_metrics`` で比べ、
    今の検出（そのフォルダーの ``eval.json`` があれば）と並べる。``<熱の地図>_eval.json`` に書く。"""
    import pandas as pd

    from smash_speed.train.agent_labels import tolerance_for_width
    from smash_speed.train.metrics import detection_metrics

    meta = json.loads((Path(data_dir) / "meta.json").read_text(encoding="utf-8"))
    heat = np.load(heat_path, mmap_mode="r")
    pred = predictions_table(data_dir, heat, threshold)
    out: dict[str, Any] = {"threshold": threshold, "dirs": {}}
    for di, dm in enumerate(meta["dirs"]):
        d = Path(dm["dir"])
        gt = pd.read_csv(d / "labels.csv")[["frame", "x_px", "y_px", "visible"]]
        tol = tolerance_for_width(dm["width"], px_at_512)
        p = pred[pred.dir == di][["frame", "x_px", "y_px", "visible"]]
        r: dict[str, Any] = {"model": detection_metrics(p, gt, tol), "model_tol2x": detection_metrics(p, gt, 2 * tol)}
        if (d / "eval.json").exists():
            base = json.loads((d / "eval.json").read_text(encoding="utf-8"))
            r["current_selected"] = base.get("selected")
            r["current_all"] = base.get("all")
        out["dirs"][d.name] = r
    path = Path(heat_path).with_name(Path(heat_path).stem + "_eval.json")
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return out


def sweep_thresholds(data_dir: Path, heat_path: Path, thresholds: list[float], px_at_512: float = 4.0) -> dict[str, Any]:
    """閾値を変えて ``evaluate`` し、フォルダーごとに閾値 → 学習したモデルの数字を並べる。

    ``<熱の地図>_eval.json`` は最後の閾値のものになるので、並べたものを ``<熱の地図>_sweep.json`` に書く。
    """
    out: dict[str, Any] = {"thresholds": list(thresholds), "dirs": {}}
    for th in thresholds:
        r = evaluate(data_dir, heat_path, th, px_at_512)
        for name, m in r["dirs"].items():
            dm = out["dirs"].setdefault(name, {"current_selected": m.get("current_selected"), "model": {}})
            dm["model"][f"{th:g}"] = m["model"]
    path = Path(heat_path).with_name(Path(heat_path).stem + "_sweep.json")
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return out


# =============================================================================
# 動画の全コマの検出（.venv。今の検出と同じ形の CSV にする）
# =============================================================================
MAX_JUMP = 0.15   # 同じ軌道とみなす 1 コマあたりの動きの上限 [画像の幅]（30 fps のスマッシュで 0.1 ほど）
MAX_GAP = 3       # 同じ軌道とみなす、見つからないコマの数の上限
MIN_TURN_STEP = (0.004, 0.0015)  # 向きの折り返しを見る 1 コマあたりの動きの下限 [画像の幅]（前後の大きい方・小さい方。
                                 # 頂点の近くのゆっくりした動きでは見ない。打球の前の球は 4K で 1 コマ 12 px まで遅くなる）


def split_tracks(frames: np.ndarray, x: np.ndarray, y: np.ndarray, width: float,
                 max_jump: float = MAX_JUMP, max_gap: int = MAX_GAP, min_turn_step: tuple[float, float] = MIN_TURN_STEP
                 ) -> np.ndarray:
    """1 コマ 1 点の検出（見つかったコマだけ、番号の順）を軌道に分け、軌道の番号を返す。

    次のときに新しい軌道にする。``shot_list.shuttle_tracks``・``pipeline`` の打球後の軌道の選び方が
    軌道ごとに働くようにするため（今の検出は打球をはさんだ軌道を別々に出す）。

    - 前の点から ``max_gap`` コマより多く空いた
    - 1 コマあたり ``max_jump`` × 画像の幅より大きく跳んだ（別の物に移った）
    - 前後の 1 コマあたりの動きの大きい方が ``min_turn_step[0]``、小さい方が ``min_turn_step[1]`` × 画像の幅より大きく、
      向きが 90° より大きく変わった（打球）
    """
    frames = np.asarray(frames, int)
    x, y = np.asarray(x, float), np.asarray(y, float)
    tid = np.zeros(len(frames), int)
    prev_step = None   # 同じ軌道の前の 1 コマあたりの動き
    for i in range(1, len(frames)):
        gap = frames[i] - frames[i - 1]
        step = np.array([x[i] - x[i - 1], y[i] - y[i - 1]]) / max(gap, 1)
        new = gap > max_gap + 1 or float(np.hypot(*step)) > max_jump * width
        if not new and prev_step is not None:
            n1, n2 = sorted((float(np.hypot(*prev_step)), float(np.hypot(*step))))
            big = n2 > min_turn_step[0] * width and n1 > min_turn_step[1] * width
            new = big and float(prev_step @ step) < 0.0
        tid[i] = tid[i - 1] + int(new)
        prev_step = None if new else step
    return tid


def detections_from_heat(data_dir: Path, heat_path: Path, threshold: float = 0.5, out: Path | None = None,
                         max_jump: float = MAX_JUMP, max_gap: int = MAX_GAP):
    """``build_video`` で書き出した区間の熱の地図を、今の検出と同じ形の表（``frame, timestamp, x_px, y_px,
    visible, confidence, track_id, heat``。座標は元の解像度）にする。``out`` があれば CSV に書く。

    ``confidence`` は ``detection.tracknet`` と同じく見つかったかどうか（1/0。0〜1 の間の値は ``pipeline`` が今の検出の
    軌道のスコアとして読むので入れない）、``heat`` は熱の地図のそのコマの最大（0〜1）、``track_id`` は ``split_tracks`` の軌道
    （見つからないコマは -1）。
    """
    import pandas as pd

    data_dir = Path(data_dir)
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    ft = np.load(data_dir / "frame_times.npy")
    pred = predictions_table(data_dir, np.load(heat_path, mmap_mode="r"), threshold).set_index("frame")
    frames = ft[:, 0].astype(int)
    pred = pred.reindex(frames)   # 埋めたコマ（区間の外）を落とす
    det = pd.DataFrame({"frame": frames, "timestamp": ft[:, 1], "x_px": pred.x_px.to_numpy(float),
                        "y_px": pred.y_px.to_numpy(float), "visible": pred.visible.fillna(0).astype(int).to_numpy(),
                        "track_id": -1, "heat": pred.peak.fillna(0.0).round(4).to_numpy()})
    det.insert(5, "confidence", det.visible.astype(float))
    vis = det.visible.to_numpy() == 1
    det.loc[vis, "track_id"] = split_tracks(frames[vis], det.x_px[vis].to_numpy(), det.y_px[vis].to_numpy(),
                                            float(meta["dirs"][0]["width"]), max_jump, max_gap)
    if out is not None:
        det.to_csv(out, index=False)
    return det


def predict_in_train_env(data_dir: Path, ckpt: Path, out: Path | None = None, prefer: str | None = None) -> Path:
    """``predict`` を、torch のある Python（この Python か、なければ ``.venv-train``）で動かす。"""
    import importlib.util
    import os
    import subprocess

    if importlib.util.find_spec("torch") is not None:
        return predict(data_dir, ckpt, out, prefer=prefer)
    root = Path(__file__).resolve().parents[2]
    py = root / ".venv-train" / "Scripts" / "python.exe"
    if not py.exists():
        py = root / ".venv-train" / "bin" / "python"
    if not py.exists():
        raise FileNotFoundError("torch の入った Python（.venv-train）が見つかりません（docs/gpu.md）")
    out = out or Path(ckpt).with_name(Path(ckpt).stem + "_pred.npy")
    args = [str(py), "-W", "ignore", "-m", "smash_speed.train.tracknet_train", "predict", str(data_dir), str(ckpt),
            "--out", str(out)] + (["--device", prefer] if prefer else [])
    env = dict(os.environ, PYTHONPATH=str(root), PYTHONIOENCODING="utf-8")
    r = subprocess.run(args, env=env, cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"熱の地図を作れませんでした：{r.stderr.strip()[-2000:]}")
    return out


def detect_video(video: Path, ckpt: Path, out_dir: Path, start_s: float | None = None, end_s: float | None = None,
                 threshold: float = 0.5, prefer: str | None = None, keep_frames: bool = False, progress=print):
    """動画の区間の全コマを学習したモデルで検出し、``out_dir/detections.csv`` に書いて表を返す。

    コマ（288×512）と熱の地図は大きい（1 コマ 0.6 MB）ので、``keep_frames`` でなければ最後に消す。
    """
    out_dir = Path(out_dir)
    build_video(video, out_dir, start_s, end_s, progress=progress)
    t0 = time.time()
    heat = predict_in_train_env(out_dir, Path(ckpt), out_dir / "heat.npy", prefer)
    progress(f"熱の地図（{time.time() - t0:.0f} 秒）")
    det = detections_from_heat(out_dir, heat, threshold, out_dir / "detections.csv")
    if not keep_frames:
        for name in ("frames.npy", "heat.npy"):
            (out_dir / name).unlink(missing_ok=True)
    return det


def _fmt(m: dict[str, Any] | None) -> str:
    if not m:
        return "-"
    return f"再現率 {m['recall']:.3f}・適合率 {m['precision']:.3f}・ずれの中央値 {m['median_error_px']:.1f} px"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="エージェントの正解で TrackNetV3 と同じ形のモデルを学習し、今の検出と比べる（F1b）")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="正解のフォルダーから学習・答え合わせ用のデータを書き出す（.venv）")
    b.add_argument("out", type=Path)
    b.add_argument("dirs", type=Path, nargs="+", help="labels.csv・candidates.csv のあるフォルダー")
    b.add_argument("--contiguous", action="store_true", help="続いたコマを重ならずに並べる（答え合わせ用）")
    b.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("train", help="学習する（.venv-train）")
    t.add_argument("data", type=Path)
    t.add_argument("out", type=Path)
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--batch", type=int, default=2)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--init", type=Path, help="TrackNetV3 の学習済みの重み（TrackNet_best.pt）。なければ乱数から")
    t.add_argument("--width", type=int, default=64)
    t.add_argument("--val-frac", type=float, default=0.1)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default=None, help="auto / gpu / cpu")
    t.add_argument("--max-minutes", type=float, default=None)
    pr = sub.add_parser("predict", help="熱の地図を書き出す（.venv-train）")
    pr.add_argument("data", type=Path)
    pr.add_argument("ckpt", type=Path)
    pr.add_argument("--out", type=Path)
    pr.add_argument("--device", default=None)
    e = sub.add_parser("eval", help="熱の地図を正解と比べる（.venv）")
    e.add_argument("data", type=Path)
    e.add_argument("heat", type=Path)
    e.add_argument("--threshold", type=float, nargs="+", default=[0.5], help="2 つ以上なら並べて _sweep.json に書く")
    fr = sub.add_parser("frames", help="動画の区間の全コマを書き出す（.venv。predict に渡す）")
    fr.add_argument("video", type=Path)
    fr.add_argument("out", type=Path)
    fr.add_argument("--start", type=float)
    fr.add_argument("--end", type=float)
    dt = sub.add_parser("detections", help="frames の熱の地図を検出の CSV にする（.venv）")
    dt.add_argument("data", type=Path)
    dt.add_argument("heat", type=Path)
    dt.add_argument("--threshold", type=float, default=0.5)
    dt.add_argument("--out", type=Path, help="既定は <data>/detections.csv")
    dv = sub.add_parser("detect", help="frames → predict（.venv-train）→ detections をまとめて行う（.venv）")
    dv.add_argument("video", type=Path)
    dv.add_argument("ckpt", type=Path)
    dv.add_argument("out", type=Path)
    dv.add_argument("--start", type=float)
    dv.add_argument("--end", type=float)
    dv.add_argument("--threshold", type=float, default=0.5)
    dv.add_argument("--device", default=None)
    dv.add_argument("--keep-frames", action="store_true", help="書き出したコマと熱の地図を消さない")
    a = p.parse_args(argv)
    if a.cmd == "frames":
        info = build_video(a.video, a.out, a.start, a.end)
        print(json.dumps({"n_frames": info["n_frames"], "n_clips": info["n_clips"]}, ensure_ascii=False))
    elif a.cmd == "detections":
        det = detections_from_heat(a.data, a.heat, a.threshold, a.out or a.data / "detections.csv")
        print(json.dumps({"n_frames": len(det), "n_visible": int(det.visible.sum()),
                          "n_tracks": int(det.track_id.max() + 1)}, ensure_ascii=False))
    elif a.cmd == "detect":
        det = detect_video(a.video, a.ckpt, a.out, a.start, a.end, a.threshold, a.device, a.keep_frames)
        print(json.dumps({"n_frames": len(det), "n_visible": int(det.visible.sum()),
                          "n_tracks": int(det.track_id.max() + 1), "csv": str(a.out / "detections.csv")},
                         ensure_ascii=False))
    elif a.cmd == "build":
        info = build_dataset(a.dirs, a.out, contiguous=a.contiguous, seed=a.seed)
        print(json.dumps({"n_clips": info["n_clips"]}, ensure_ascii=False))
    elif a.cmd == "train":
        h = train(a.data, a.out, a.epochs, a.batch, a.lr, a.init, a.width, a.val_frac, a.seed, a.device, a.max_minutes)
        print(json.dumps({"best_epoch": h["best_epoch"], "val_loss": min(h["val_loss"], default=h["val_loss_init"])}))
    elif a.cmd == "predict":
        print(predict(a.data, a.ckpt, a.out, prefer=a.device))
    elif len(a.threshold) > 1:
        r = sweep_thresholds(a.data, a.heat, a.threshold)
        for name, m in r["dirs"].items():
            print(f"{name}\n  今の検出（選んだ軌道）：{_fmt(m.get('current_selected'))}")
            for th, mm in m["model"].items():
                print(f"  学習したモデル（閾値 {th}）：{_fmt(mm)}")
    else:
        r = evaluate(a.data, a.heat, a.threshold[0])
        for name, m in r["dirs"].items():
            print(f"{name}\n  学習したモデル：{_fmt(m['model'])}\n  今の検出（選んだ軌道）：{_fmt(m.get('current_selected'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
