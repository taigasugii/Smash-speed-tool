"""smash-speed の GUI。動画を開いてから、打球音の検索・キャリブレーション・解析・結果の確認・研究用ツールまでを1つの画面で行う。

起動: リポジトリ直下の ``smash-speed-gui.bat`` をダブルクリック（または ``python -m smash_speed.app``）。
各処理は ``smash-speed`` コマンドを裏で実行しており、実行したコマンドは下のログに表示される。
"""

from __future__ import annotations

import json
import os
import tkinter as tk
import tkinter.font  # noqa: F401  （tk.font を使う）
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import numpy as np
import pandas as pd
from PIL import Image, ImageTk

from smash_speed.app.runner import TaskRunner, cli_command
from smash_speed.app.viewer import VideoViewer
from smash_speed.config import PROJECT_ROOT

STATE_FILE = PROJECT_ROOT / "outputs" / "gui_state.json"
FONT = ("Yu Gothic UI", 10)
IMPACT_METHODS = {"自動（打球音 → 軌道の交点 → 打者の位置）": "auto", "打球音": "audio", "軌道の交点": "crossing",
                  "打者の位置": "hitter_position", "最初の検出": "first_frame", "手動で指定": "manual"}
CI_METHODS = {"ヤコビアン（速い）": "jacobian", "ブートストラップ（遅い）": "bootstrap", "出さない": "none"}
RESULT_IMAGES = {"画像上の当てはめ": "image_fit.png", "速さと時間": "speed_time.png", "3次元軌道": "trajectory.png"}


def open_path(path: Path) -> None:
    """ファイル・フォルダを Windows の既定のアプリで開く。"""
    if not Path(path).exists():
        messagebox.showwarning("smash-speed", f"見つかりません:\n{path}")
        return
    os.startfile(str(path))  # noqa: S606  Windows 専用の GUI


class ImagePanel(ttk.Frame):
    """画像ファイルを枠の大きさに合わせて表示する。枠の大きさは周りのレイアウトで決まり、画像では広がらない。"""

    def __init__(self, master, width: int = 560, height: int = 330):
        super().__init__(master, width=width, height=height)
        self.pack_propagate(False)
        self.lbl = ttk.Label(self, anchor="center", text="（まだありません）")
        self.lbl.pack(fill="both", expand=True)
        self.w, self.h = width, height
        self._photo = None
        self._path: Path | None = None
        self._job = None
        self.bind("<Configure>", self._on_resize)

    def _on_resize(self, e) -> None:
        if abs(e.width - self.w) < 8 and abs(e.height - self.h) < 8:
            return
        self.w, self.h = max(e.width - 4, 50), max(e.height - 4, 50)
        if self._job:
            self.after_cancel(self._job)
        self._job = self.after(150, lambda: self.show(self._path))

    def show(self, path: Path | None) -> None:
        self._path = Path(path) if path else None
        if path is None or not Path(path).exists():
            self.lbl.configure(image="", text="（まだありません）")
            self._photo = None
            return
        img = Image.open(path)
        img.thumbnail((self.w, self.h))
        self._photo = ImageTk.PhotoImage(img)
        self.lbl.configure(image=self._photo, text="")


def auto_wrap(label: ttk.Label) -> ttk.Label:
    """ラベルの折り返し幅を、親の枠の幅に合わせる（ラベル自身の幅を使うと際限なく広がるため）。"""
    parent = label.master
    parent.bind("<Configure>", lambda e: label.configure(wraplength=max(e.width - 24, 100)), add="+")
    return label


def _enable_dpi_awareness() -> None:
    """高解像度ディスプレイ（拡大表示 150% など）で文字がぼやけたり、画面からはみ出したりしないようにする。"""
    if os.name != "nt":
        return
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass


class App(tk.Tk):
    """メインウィンドウ。"""

    def __init__(self):
        _enable_dpi_awareness()
        super().__init__()
        self.title("smash-speed  スマッシュ速度解析")
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = int(sw * 0.94), int(sh * 0.88)
        self.geometry(f"{w}x{h}+{(sw - w) // 2}+10")
        self.minsize(min(1200, w), min(760, h))
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        for name in ("TLabel", "TButton", "TCheckbutton", "TRadiobutton", "TLabelframe.Label", "TNotebook.Tab", "TEntry"):
            style.configure(name, font=FONT)
        style.configure("Big.TButton", font=("Yu Gothic UI", 11, "bold"))
        # 拡大表示の画面でも表の行が欠けないよう、行の高さを文字の大きさに合わせる
        line = tk.font.Font(font=FONT).metrics("linespace")
        style.configure("Treeview", font=FONT, rowheight=int(line * 1.5))
        style.configure("Treeview.Heading", font=FONT)
        style.configure("Title.TLabel", font=("Yu Gothic UI", 11, "bold"))

        self.v: dict[str, tk.Variable] = {}
        self.meta = None
        self.camera = None
        self.point_target: str | None = None
        self._build()
        self.runner = TaskRunner(self, self.log, self._set_busy)
        self._load_state()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ================================================================ 変数
    def var(self, key: str, default="", kind=tk.StringVar) -> tk.Variable:
        if key not in self.v:
            self.v[key] = kind(value=default)
        return self.v[key]

    def get(self, key: str) -> str:
        return str(self.var(key).get()).strip()

    def getf(self, key: str) -> float | None:
        s = self.get(key)
        try:
            return float(s) if s else None
        except ValueError:
            return None

    # ================================================================ 画面
    def _build(self) -> None:
        top = ttk.Frame(self, padding=(8, 6))
        top.pack(fill="x")
        ttk.Button(top, text="動画を開く…", style="Big.TButton", command=self.choose_video).pack(side="left")
        ttk.Entry(top, textvariable=self.var("video"), width=70).pack(side="left", padx=6)
        ttk.Label(top, text="撮影 fps（スロー動画のみ）").pack(side="left", padx=(12, 2))
        ttk.Entry(top, textvariable=self.var("capture_fps"), width=6).pack(side="left")
        ttk.Button(top, text="再読み込み", command=self.load_video).pack(side="left", padx=4)
        ttk.Label(top, text="出力フォルダ").pack(side="left", padx=(16, 2))
        ttk.Entry(top, textvariable=self.var("out_root", str(PROJECT_ROOT / "outputs")), width=34).pack(side="left")
        ttk.Button(top, text="…", width=3, command=lambda: self._ask_dir("out_root")).pack(side="left")
        ttk.Button(top, text="開く", command=lambda: open_path(Path(self.get("out_root")))).pack(side="left", padx=2)

        body = ttk.PanedWindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=8)
        left = ttk.Frame(body)
        self.viewer = VideoViewer(left, on_status=self.status)
        self.viewer.pack(fill="both", expand=True)
        self.viewer.on_point = self._on_viewer_point
        self.viewer.on_roi = self._on_roi
        self.lbl_meta = ttk.Label(left, text="", foreground="#555555")
        self.lbl_meta.pack(anchor="w", pady=(4, 0))
        body.add(left, weight=3)

        self.nb = ttk.Notebook(body)
        body.add(self.nb, weight=2)
        self._body = body
        self.after(300, self._place_sash)
        self._tab_find()
        self._tab_calib()
        self._tab_analyze()
        self._tab_result()
        self._tab_list()
        self._tab_research()
        self._tab_settings()

        bottom = ttk.Frame(self, padding=(8, 4))
        bottom.pack(fill="both")
        bar = ttk.Frame(bottom)
        bar.pack(fill="x")
        self.lbl_status = ttk.Label(bar, text="準備ができました。上の「動画を開く…」から始めてください。")
        self.lbl_status.pack(side="left")
        self.btn_cancel = ttk.Button(bar, text="中止", command=lambda: self.runner.cancel(), state="disabled")
        self.btn_cancel.pack(side="right")
        self.prog = ttk.Progressbar(bar, mode="indeterminate", length=180)
        self.prog.pack(side="right", padx=8)
        self.txt_log = ScrolledText(bottom, height=9, font=("Consolas", 9), wrap="word")
        self.txt_log.pack(fill="both", expand=True, pady=(4, 0))

    def _place_sash(self, tries: int = 0) -> None:
        """左（動画）と右（操作）の境目を、ウィンドウ幅の 56% の位置に置く（ウィンドウの大きさが決まってから）。"""
        w = self._body.winfo_width()
        if w < 400 and tries < 20:
            self.after(200, lambda: self._place_sash(tries + 1))
            return
        self._body.sashpos(0, int(w * 0.56))

    def _row(self, parent, r: int, label: str, key: str, default="", width: int = 12, browse: str | None = None,
             hint: str = "") -> ttk.Entry:
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", pady=2)
        e = ttk.Entry(parent, textvariable=self.var(key, default), width=width)
        e.grid(row=r, column=1, sticky="we", pady=2)
        if browse == "file":
            ttk.Button(parent, text="…", width=3, command=lambda: self._ask_file(key)).grid(row=r, column=2, padx=2)
        elif browse == "dir":
            ttk.Button(parent, text="…", width=3, command=lambda: self._ask_dir(key)).grid(row=r, column=2, padx=2)
        if hint:
            ttk.Label(parent, text=hint, foreground="#666666").grid(row=r, column=3, sticky="w", padx=4)
        return e

    def _help(self, parent, text: str) -> None:
        auto_wrap(ttk.Label(parent, text=text, foreground="#444444", justify="left")).pack(
            anchor="w", fill="x", pady=(0, 6))

    # ---------------------------------------------------------------- ① スマッシュを探す
    def _tab_find(self) -> None:
        f = ttk.Frame(self.nb, padding=10)
        self.nb.add(f, text="① スマッシュを探す")
        self._help(f, "打球音の候補を一覧します。ダブルクリックするとその時刻へ移動します。左の動画で、解析したい"
                      "スマッシュの「打つ少し前」と「着地の少し後」を、それぞれ開始・終了に設定してください（1 秒前後）。")
        bar = ttk.Frame(f)
        bar.pack(fill="x")
        ttk.Button(bar, text="打球音を探す", style="Big.TButton", command=self.find_onsets).pack(side="left")
        ttk.Label(bar, text="  強さ").pack(side="left")
        ttk.Spinbox(bar, from_=0, to=200, increment=5, textvariable=self.var("min_strength", "20"), width=5,
                    command=self._fill_onsets).pack(side="left")
        ttk.Label(bar, text="以上を表示").pack(side="left")
        lf = ttk.Frame(f)
        lf.pack(fill="both", expand=True, pady=6)
        self.lst_onsets = ttk.Treeview(lf, columns=("t", "s"), show="headings", height=14)
        self.lst_onsets.heading("t", text="時刻 [s]")
        self.lst_onsets.heading("s", text="強さ")
        self.lst_onsets.column("t", width=120, anchor="e")
        self.lst_onsets.column("s", width=80, anchor="e")
        self.lst_onsets.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(lf, command=self.lst_onsets.yview)
        sb.pack(side="left", fill="y")
        self.lst_onsets.configure(yscrollcommand=sb.set)
        self.lst_onsets.bind("<Double-1>", self._jump_onset)
        self._onsets: list[tuple[float, float]] = []

        g = ttk.LabelFrame(f, text="解析する区間", padding=8)
        g.pack(fill="x")
        ttk.Label(g, text="開始 [s]").grid(row=0, column=0)
        ttk.Entry(g, textvariable=self.var("start"), width=10).grid(row=0, column=1, padx=4)
        ttk.Button(g, text="今の位置を開始に", command=lambda: self.var("start").set(f"{self.viewer.time:.3f}")).grid(row=0, column=2)
        ttk.Label(g, text="終了 [s]").grid(row=1, column=0)
        ttk.Entry(g, textvariable=self.var("end"), width=10).grid(row=1, column=1, padx=4)
        ttk.Button(g, text="今の位置を終了に", command=lambda: self.var("end").set(f"{self.viewer.time:.3f}")).grid(row=1, column=2)
        ttk.Button(g, text="この区間を再生", command=self._play_range).grid(row=0, column=3, rowspan=2, padx=12)
        ttk.Label(g, text="解析の名前").grid(row=2, column=0, pady=(6, 0))
        ttk.Entry(g, textvariable=self.var("shot"), width=24).grid(row=2, column=1, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Label(g, text="（出力フォルダ名。空なら自動）", foreground="#666666").grid(row=2, column=3, sticky="w")

    # ---------------------------------------------------------------- ② キャリブレーション
    def _tab_calib(self) -> None:
        f = ttk.Frame(self.nb, padding=10)
        self.nb.add(f, text="② キャリブレーション")
        self._help(f, "コートとネットがよく映っているフレームを左に表示してから「キャリブレーションを始める」を押すと、"
                      "別のウィンドウが開きます。右上の上面図で赤く光る点（ラインの外側の角）を順にクリックし、"
                      "映っていない点は s でスキップ、c で確認、Enter で保存します。"
                      "カメラを動かしていなければ、同じファイルを別の動画にも使えます。")
        g = ttk.Frame(f)
        g.pack(fill="x")
        self._row(g, 0, "保存先", "calib", width=50, browse="file")
        g.columnconfigure(1, weight=1)
        bar = ttk.Frame(f)
        bar.pack(fill="x", pady=6)
        ttk.Button(bar, text="キャリブレーションを始める（今のフレーム）", style="Big.TButton",
                   command=lambda: self.calibrate(False)).pack(side="left")
        ttk.Button(bar, text="保存済みの点を直す", command=lambda: self.calibrate(True)).pack(side="left", padx=6)
        ttk.Button(bar, text="今のフレームに重ねて確認", command=self.check_calib).pack(side="left")
        ttk.Checkbutton(f, text="レンズの歪み（k1）も推定する（超広角のとき）", variable=self.var("k1", False, tk.BooleanVar)).pack(anchor="w")
        self.lbl_calib = auto_wrap(ttk.Label(f, text="", foreground="#333333", justify="left"))
        self.lbl_calib.pack(anchor="w", fill="x", pady=4)
        self.img_calib = ImagePanel(f, 580, 340)
        self.img_calib.pack(fill="both", expand=True)

    # ---------------------------------------------------------------- ③ 解析
    def _tab_analyze(self) -> None:
        f = ttk.Frame(self.nb, padding=10)
        self.nb.add(f, text="③ 解析")
        self._help(f, "飛行平面（シャトルが飛ぶ鉛直な面）を、打者の足元と着地点で決めます。ボタンを押してから"
                      "左の動画の床の上をクリックしてください（キャリブレーションが必要）。座標を直接入力しても構いません。")
        g = ttk.LabelFrame(f, text="飛行平面", padding=8)
        g.pack(fill="x")
        ttk.Button(g, text="打者の足元を画面で指定", command=lambda: self._pick("hitter")).grid(row=0, column=0, sticky="w")
        ttk.Label(g, text="x, y [m]").grid(row=0, column=1, padx=(8, 2))
        ttk.Entry(g, textvariable=self.var("hitter_x"), width=8).grid(row=0, column=2)
        ttk.Entry(g, textvariable=self.var("hitter_y"), width=8).grid(row=0, column=3, padx=2)
        ttk.Button(g, text="着地点を画面で指定", command=lambda: self._pick("landing")).grid(row=1, column=0, sticky="w", pady=2)
        ttk.Label(g, text="x, y [m]").grid(row=1, column=1, padx=(8, 2))
        ttk.Entry(g, textvariable=self.var("landing_x"), width=8).grid(row=1, column=2)
        ttk.Entry(g, textvariable=self.var("landing_y"), width=8).grid(row=1, column=3, padx=2)
        ttk.Label(g, text="座標：ネット直下の中央が原点、打者側が x < 0。空欄なら自動推定（試験的）",
                  foreground="#666666").grid(row=2, column=0, columnspan=5, sticky="w")

        g = ttk.LabelFrame(f, text="打点時刻", padding=8)
        g.pack(fill="x", pady=6)
        ttk.Combobox(g, textvariable=self.var("impact_method", list(IMPACT_METHODS)[0]), values=list(IMPACT_METHODS),
                     state="readonly", width=36).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(g, text="手動の時刻 [s]").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(g, textvariable=self.var("impact_time"), width=10).grid(row=1, column=1, sticky="w")
        ttk.Button(g, text="今の位置を打点に", command=self._impact_here).grid(row=1, column=2, padx=4)
        ttk.Label(g, text="別の音声ファイル（任意）").grid(row=2, column=0, sticky="w")
        ttk.Entry(g, textvariable=self.var("audio"), width=30).grid(row=2, column=1, columnspan=2, sticky="we")
        ttk.Button(g, text="…", width=3, command=lambda: self._ask_file("audio")).grid(row=2, column=3)

        g = ttk.LabelFrame(f, text="シャトルの検出", padding=8)
        g.pack(fill="x")
        ttk.Radiobutton(g, text="ベースライン（背景差分）", value="baseline", variable=self.var("detector", "baseline")).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(g, text="TrackNetV3（⑦ 設定で準備が必要）", value="tracknet", variable=self.var("detector")).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(g, text="検出する範囲（ROI）を左の動画でドラッグして指定", variable=self.var("roi_mode", False, tk.BooleanVar),
                        command=self._toggle_roi).grid(row=1, column=0, columnspan=2, sticky="w")
        self.lbl_roi = ttk.Label(g, text="ROI：なし（画面全体）", foreground="#666666")
        self.lbl_roi.grid(row=2, column=0, sticky="w")
        ttk.Button(g, text="ROI を消す", command=lambda: self._on_roi(None, clear=True)).grid(row=2, column=1, sticky="w")

        g = ttk.LabelFrame(f, text="オプション", padding=8)
        g.pack(fill="x", pady=6)
        ttk.Label(g, text="信頼区間").grid(row=0, column=0, sticky="w")
        ttk.Combobox(g, textvariable=self.var("ci", list(CI_METHODS)[0]), values=list(CI_METHODS), state="readonly",
                     width=24).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(g, text="オーバーレイ動画を作る", variable=self.var("overlay", True, tk.BooleanVar)).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(g, text="保存済みの途中結果を使わず作り直す", variable=self.var("force", True, tk.BooleanVar)).grid(row=1, column=1, sticky="w")

        ttk.Button(f, text="解析を実行する", style="Big.TButton", command=self.analyze).pack(fill="x", pady=8, ipady=6)

    # ---------------------------------------------------------------- ④ 結果
    def _tab_result(self) -> None:
        f = ttk.Frame(self.nb, padding=10)
        self.nb.add(f, text="④ 結果")
        self.lbl_result = auto_wrap(ttk.Label(f, text="解析を実行すると、ここに結果が表示されます。", style="Title.TLabel",
                                              justify="left"))
        self.lbl_result.pack(anchor="w", fill="x")
        self.lbl_result2 = auto_wrap(ttk.Label(f, text="", justify="left", foreground="#333333"))
        self.lbl_result2.pack(anchor="w", fill="x", pady=(2, 6))
        bar = ttk.Frame(f)
        bar.pack(fill="x")
        for name in RESULT_IMAGES:
            ttk.Radiobutton(bar, text=name, value=name, variable=self.var("result_img", list(RESULT_IMAGES)[0]),
                            command=self._show_result_image).pack(side="left")
        self.img_result = ImagePanel(f, 600, 380)
        self.img_result.pack(fill="both", expand=True, pady=6)
        bar = ttk.Frame(f)
        bar.pack(fill="x")
        ttk.Button(bar, text="オーバーレイ動画を再生", command=lambda: self._open_in_result("overlay.mp4")).pack(side="left")
        ttk.Button(bar, text="結果の CSV を開く", command=lambda: self._open_in_result("result.csv")).pack(side="left", padx=4)
        ttk.Button(bar, text="フォルダを開く", command=lambda: self._open_in_result("")).pack(side="left")
        ttk.Button(bar, text="別の結果を開く…", command=self._choose_result).pack(side="right")
        self.result_dir: Path | None = None

    # ---------------------------------------------------------------- ⑤ 一覧
    def _tab_list(self) -> None:
        f = ttk.Frame(self.nb, padding=10)
        self.nb.add(f, text="⑤ 結果一覧")
        self._help(f, "出力フォルダの中の解析結果をすべて集めます。ダブルクリックで④に表示します。"
                      "「集計表を保存」で平均・標準偏差つきの CSV（summary.csv）を作ります。")
        bar = ttk.Frame(f)
        bar.pack(fill="x")
        ttk.Button(bar, text="一覧を更新", command=self.refresh_list).pack(side="left")
        ttk.Button(bar, text="集計表を保存", command=self.save_summary).pack(side="left", padx=4)
        cols = ("shot", "v0", "ci", "mean", "fps", "imp", "rms")
        names = ("名前", "初速 km/h", "95%CI", "平均 km/h", "fps", "打点", "誤差 px")
        self.tree = ttk.Treeview(f, columns=cols, show="headings", height=20)
        for c, n, w in zip(cols, names, (150, 80, 100, 80, 50, 110, 60)):
            self.tree.heading(c, text=n)
            self.tree.column(c, width=w, anchor="w" if c in ("shot", "imp") else "e")
        self.tree.pack(fill="both", expand=True, pady=6)
        self.tree.bind("<Double-1>", self._open_from_list)
        self._list_dirs: dict[str, Path] = {}

    # ---------------------------------------------------------------- ⑥ 研究ツール
    def _tab_research(self) -> None:
        outer = ttk.Frame(self.nb, padding=6)
        self.nb.add(outer, text="⑥ 研究ツール")
        f = outer

        g = ttk.LabelFrame(f, text="合成データで試す（正解つきの動画を作って、そのまま読み込む）", padding=8)
        g.pack(fill="x")
        ttk.Label(g, text="初速 [m/s]").grid(row=0, column=0)
        ttk.Entry(g, textvariable=self.var("syn_speed", "80"), width=6).grid(row=0, column=1)
        ttk.Label(g, text="fps").grid(row=0, column=2, padx=(8, 0))
        ttk.Combobox(g, textvariable=self.var("syn_fps", "240"), values=["30", "60", "120", "240"], width=5).grid(row=0, column=3)
        ttk.Label(g, text="露光 [s]").grid(row=0, column=4, padx=(8, 0))
        ttk.Entry(g, textvariable=self.var("syn_exposure", "0"), width=7).grid(row=0, column=5)
        ttk.Button(g, text="作成して読み込む", command=self.make_synth).grid(row=0, column=6, padx=8)

        g = ttk.LabelFrame(f, text="正解ラベルを付ける（アノテーション）", padding=8)
        g.pack(fill="x", pady=4)
        ttk.Label(g, text="①の区間で別ウィンドウが開きます。クリックで位置、x 見えない、h 打点、g 着地、i 補間、"
                          "a/d 前後のコマ、Esc 保存して終了。", foreground="#444444").grid(row=0, column=0, columnspan=4, sticky="w")
        self._row(g, 1, "保存先フォルダ", "ann_out", width=40, browse="dir")
        ttk.Checkbutton(g, text="直前の解析の検出結果を初期値にする", variable=self.var("ann_init", True, tk.BooleanVar)).grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Button(g, text="アノテーションを始める", command=self.annotate).grid(row=2, column=2, columnspan=2, sticky="e")
        g.columnconfigure(1, weight=1)

        g = ttk.LabelFrame(f, text="疑似低 fps 動画を作る（240fps の動画とラベルから）", padding=8)
        g.pack(fill="x", pady=4)
        self._row(g, 0, "ラベル（labels.csv）", "low_labels", width=40, browse="file")
        self._row(g, 1, "保存先フォルダ", "low_out", width=40, browse="dir")
        fr = ttk.Frame(g)
        fr.grid(row=2, column=0, columnspan=4, sticky="w")
        for fps in ("30", "60", "120"):
            ttk.Checkbutton(fr, text=f"{fps}fps", variable=self.var(f"low_{fps}", True, tk.BooleanVar)).pack(side="left")
        ttk.Checkbutton(fr, text="ブレを再現（連続フレームの平均）", variable=self.var("low_blur", True, tk.BooleanVar)).pack(side="left", padx=8)
        ttk.Button(fr, text="作成", command=self.make_lowfps).pack(side="left", padx=8)
        g.columnconfigure(1, weight=1)

        g = ttk.LabelFrame(f, text="精度検証", padding=8)
        g.pack(fill="x", pady=4)
        ttk.Label(g, text="合成データの本数").grid(row=0, column=0, sticky="w")
        ttk.Entry(g, textvariable=self.var("val_n", "100"), width=6).grid(row=0, column=1, sticky="w")
        ttk.Button(g, text="合成データで検証（数分）", command=self.validate_synthetic).grid(row=0, column=2, padx=6)
        self._row(g, 1, "実データの一覧 CSV", "val_index", width=34, browse="file")
        self._row(g, 2, "外部計測値 CSV（任意）", "val_ref", width=34, browse="file")
        ttk.Button(g, text="実データで検証", command=self.validate_dataset).grid(row=3, column=2, padx=6, pady=2)
        g.columnconfigure(1, weight=1)

        g = ttk.LabelFrame(f, text="TrackNetV3 の学習データ", padding=8)
        g.pack(fill="x", pady=4)
        self._row(g, 0, "データ一覧 CSV", "exp_index", width=34, browse="file")
        ttk.Button(g, text="TrackNetV3 形式で書き出す", command=self.export_tracknet).grid(row=1, column=1, sticky="e", pady=2)
        ttk.Button(g, text="Colab 用ノートブックの場所を開く",
                   command=lambda: open_path(PROJECT_ROOT / "notebooks")).grid(row=1, column=2, columnspan=2, padx=4)
        g.columnconfigure(1, weight=1)

    # ---------------------------------------------------------------- ⑦ 設定
    def _tab_settings(self) -> None:
        f = ttk.Frame(self.nb, padding=10)
        self.nb.add(f, text="⑦ 設定")
        self._help(f, "細かい設定（終端速度・検出の閾値など）は YAML ファイルで変えます。既定の設定は直接書き換えず、"
                      "変えたい項目だけを書いたファイルを作って、ここで指定してください。")
        g = ttk.Frame(f)
        g.pack(fill="x")
        self._row(g, 0, "上書きする設定 YAML", "config", width=46, browse="file")
        g.columnconfigure(1, weight=1)
        bar = ttk.Frame(f)
        bar.pack(fill="x", pady=6)
        ttk.Button(bar, text="既定の設定を開く（参照用）", command=lambda: open_path(PROJECT_ROOT / "configs" / "default.yaml")).pack(side="left")
        ttk.Button(bar, text="上書き用のファイルを作る", command=self._new_override).pack(side="left", padx=6)
        ttk.Button(bar, text="README を開く", command=lambda: open_path(PROJECT_ROOT / "README.md")).pack(side="left")
        lf = ttk.LabelFrame(f, text="TrackNetV3（任意）", padding=8)
        lf.pack(fill="x", pady=8)
        auto_wrap(ttk.Label(lf, text="https://github.com/qaz812345/TrackNetV3 を clone し、README の Google Drive から重みを "
                           "ckpts\\ に置いたうえで、上書き設定 YAML の detection.tracknet に repo_dir / tracknet_ckpt を書きます。"
                           "「上書き用のファイルを作る」で書き方の例が入ります。", justify="left",
                  foreground="#444444")).pack(anchor="w", fill="x")

    # ================================================================ 動画
    def choose_video(self) -> None:
        p = filedialog.askopenfilename(title="動画を選ぶ", filetypes=[("動画", "*.mp4 *.mov *.MOV *.MP4 *.avi *.m4v"), ("すべて", "*.*")])
        if p:
            self.var("video").set(p)
            self.load_video()

    def load_video(self, then=None) -> None:
        path = self.get("video")
        if not path:
            return
        self.viewer.load(path, self.getf("capture_fps"), lambda meta, err: self._video_loaded(meta, err, then))

    def _video_loaded(self, meta, err, then=None) -> None:
        if meta is None:
            messagebox.showerror("smash-speed", f"動画を読み込めませんでした。\n{err}")
            return
        self.meta = meta
        vfr = "（可変フレームレート）" if meta.is_vfr else ""
        slow = f"  ※撮影 {meta.capture_fps:g}fps として時刻を換算" if meta.time_scale != 1.0 else ""
        self.lbl_meta.configure(text=f"{Path(meta.path).name}   {meta.width}×{meta.height}   {meta.fps_meta:.2f}fps{vfr}   "
                                     f"{meta.timestamps[-1]:.1f} 秒   音声{'あり' if meta.has_audio else 'なし'}{slow}")
        stem = Path(meta.path).stem
        if not self.get("calib") or not Path(self.get("calib")).exists():
            self.var("calib").set(str(Path(self.get("out_root")) / f"{stem}_calibration.json"))
        self._load_camera()
        self.status("動画を読み込みました。①でスマッシュを探してください。")
        self._save_state()
        if then:
            then()

    def _load_camera(self) -> None:
        """キャリブレーションがあれば読み込み（画面クリックで床の座標を出すため）、確認画像を表示する。"""
        self.camera = None
        p = Path(self.get("calib")) if self.get("calib") else None
        if p and p.exists() and self.meta is not None:
            from smash_speed.calibration import load_calibration
            from smash_speed.pipeline import _camera_for_video
            try:
                cam, data = load_calibration(p)
                self.camera = _camera_for_video(cam, self.meta)
                warn = "\n".join("・" + w for w in data.get("warnings", []))
                self.lbl_calib.configure(text=f"読み込み済み：再投影誤差 RMSE {data['rmse_px']:.2f} px、"
                                              f"使った点 {len(data['points'])} 個、カメラ位置 {np.round(self.camera.center, 2)} m"
                                              + (f"\n{warn}" if warn else ""))
            except (ValueError, KeyError) as e:
                self.lbl_calib.configure(text=f"このキャリブレーションはこの動画に使えません：{e}")
            self.img_calib.show(p.with_suffix(".png"))
        else:
            self.lbl_calib.configure(text="まだキャリブレーションしていません。")
            self.img_calib.show(None)

    # ================================================================ ① 打球音
    def find_onsets(self) -> None:
        if not self._need_video():
            return
        out = Path(self.get("out_root")) / f"{Path(self.get('video')).stem}_onsets.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        args = ["onsets", "--video", self.get("video")] + self._fps_args()
        if self.get("audio"):
            args += ["--audio", self.get("audio")]
        elif not self.meta.has_audio:
            messagebox.showinfo("smash-speed", "この動画には音声がありません。③の「別の音声ファイル」で WAV を指定するか、"
                                               "左の動画を見て区間を決めてください。")
            return
        self._onset_file = out

        def done(ok):
            if ok:
                self._fill_onsets()
        # onsets は標準出力に CSV を書くので、ログから拾う
        self._capture_onsets = []
        self.runner.run([cli_command(args, self._cfg())], done, "打球音を探す")

    def _fill_onsets(self) -> None:
        if getattr(self, "_capture_onsets", None) is not None and self._capture_onsets:
            self._onsets = list(self._capture_onsets)
            pd.DataFrame(self._onsets, columns=["time_s", "strength"]).to_csv(self._onset_file, index=False)
        self.lst_onsets.delete(*self.lst_onsets.get_children())
        th = float(self.get("min_strength") or 0)
        for t, s in self._onsets:
            if s >= th:
                self.lst_onsets.insert("", "end", values=(f"{t:.3f}", f"{s:.0f}"))
        self.status(f"打球音の候補 {sum(s >= th for _, s in self._onsets)} 個（全 {len(self._onsets)} 個）")

    def _jump_onset(self, _e) -> None:
        sel = self.lst_onsets.selection()
        if sel:
            t = float(self.lst_onsets.item(sel[0], "values")[0])
            self.viewer.seek(t)

    def _play_range(self) -> None:
        t0, t1 = self.getf("start"), self.getf("end")
        if t0 is None or t1 is None:
            messagebox.showinfo("smash-speed", "先に開始と終了を設定してください。")
            return
        self.viewer.play_range(t0, t1)

    # ================================================================ ② キャリブレーション
    def calibrate(self, use_points: bool) -> None:
        if not self._need_video():
            return
        calib = self.get("calib")
        if not calib:
            messagebox.showinfo("smash-speed", "保存先を入力してください。")
            return
        Path(calib).parent.mkdir(parents=True, exist_ok=True)
        args = ["calibrate", "--video", self.get("video"), "--time", f"{self.viewer.time:.3f}", "--out", calib] + self._fps_args()
        if use_points:
            if not Path(calib).exists():
                messagebox.showinfo("smash-speed", "保存済みのキャリブレーションがありません。")
                return
            args += ["--points", calib]
        if self.var("k1").get():
            args.append("--k1")
        self.status("キャリブレーションのウィンドウでクリックしてください（Enter で保存、Esc で中止）。")
        self.runner.run([cli_command(args, self._cfg())], lambda ok: self._load_camera(), "キャリブレーション")

    def check_calib(self) -> None:
        if not self._need_video() or not self._need_calib():
            return
        out = Path(self.get("out_root")) / "calib_check.png"
        args = ["check-calib", "--video", self.get("video"), "--calib", self.get("calib"), "--time",
                f"{self.viewer.time:.3f}", "--out", str(out)] + self._fps_args()
        self.runner.run([cli_command(args, self._cfg())], lambda ok: ok and self.img_calib.show(out), "キャリブレーションの確認")

    # ================================================================ ③ 解析
    def _pick(self, target: str) -> None:
        if not self._need_video() or not self._need_calib():
            return
        self.point_target = target
        self.var("roi_mode").set(False)
        self.viewer.mode = "point"
        self.status(("打者の足元" if target == "hitter" else "着地点") + "を、左の動画の床の上でクリックしてください。")

    def _on_viewer_point(self, u: float, v: float) -> None:
        from smash_speed.geometry import floor_point
        target = self.point_target
        self.viewer.mode = "roi" if self.var("roi_mode").get() else "none"
        self.point_target = None
        if target is None or self.camera is None:
            return
        X = floor_point(self.camera, np.array([[u, v]]))[0]
        if not np.isfinite(X).all():
            messagebox.showwarning("smash-speed", "床の上の点ではないようです（地平線より上をクリックしました）。")
            return
        self.var(f"{target}_x").set(f"{X[0]:.2f}")
        self.var(f"{target}_y").set(f"{X[1]:.2f}")
        name = "打者" if target == "hitter" else "着地"
        self.viewer.set_marker(name, (u, v), "#ffcc00" if target == "hitter" else "#ff5050")
        self.status(f"{name}：x = {X[0]:.2f} m、y = {X[1]:.2f} m")

    def _toggle_roi(self) -> None:
        on = self.var("roi_mode").get()
        self.viewer.mode = "roi" if on else "none"
        if on:
            self.status("左の動画の上をドラッグして、シャトルが飛ぶ範囲を囲んでください。")

    def _on_roi(self, roi, clear: bool = False) -> None:
        if clear:
            self.viewer.roi = None
            self.viewer._redraw()
            roi = None
        self.lbl_roi.configure(text="ROI：なし（画面全体）" if roi is None else
                               f"ROI：x {roi[0]:.0f}〜{roi[2]:.0f}、y {roi[1]:.0f}〜{roi[3]:.0f} px")

    def _impact_here(self) -> None:
        self.var("impact_time").set(f"{self.viewer.time:.4f}")
        self.var("impact_method").set("手動で指定")

    def _shot_name(self) -> str:
        name = self.get("shot")
        if not name:
            t0 = self.getf("start") or 0.0
            name = f"{Path(self.get('video')).stem}_{t0:.1f}s"
        return name

    def analyze(self) -> None:
        if not self._need_video() or not self._need_calib():
            return
        t0, t1 = self.getf("start"), self.getf("end")
        if t0 is None or t1 is None or t1 <= t0:
            messagebox.showinfo("smash-speed", "①で解析する区間（開始・終了）を設定してください。")
            return
        name = self._shot_name()
        out = Path(self.get("out_root")) / name
        common = ["--video", self.get("video"), "--start", f"{t0:.3f}", "--end", f"{t1:.3f}"] + self._fps_args()
        args = ["analyze", *common, "--calib", self.get("calib"), "--out", str(out), "--name", name]
        hx, hy, lx, ly = (self.getf(k) for k in ("hitter_x", "hitter_y", "landing_x", "landing_y"))
        if None not in (hx, hy, lx, ly):
            args += [f"--hitter={hx},{hy}", f"--landing={lx},{ly}"]
        elif any(v is not None for v in (hx, hy, lx, ly)):
            messagebox.showinfo("smash-speed", "打者と着地点の x, y を4つとも入れるか、すべて空にしてください。")
            return
        method = IMPACT_METHODS[self.get("impact_method")]
        args += ["--impact-method", method]
        if method == "manual":
            if self.getf("impact_time") is None:
                messagebox.showinfo("smash-speed", "手動の打点時刻を入れてください。")
                return
            args += ["--impact-time", self.get("impact_time")]
        if self.get("audio"):
            args += ["--audio", self.get("audio")]
        args += ["--ci", CI_METHODS[self.get("ci")]]
        if not self.var("overlay").get():
            args.append("--no-overlay")
        if self.var("force").get():
            args.append("--force")
        roi = self.viewer.roi
        roi_s = ",".join(f"{v:.0f}" for v in roi) if roi else None
        commands = []
        if self.get("detector") == "tracknet":
            det = out / "detections_tracknet.csv"
            out.mkdir(parents=True, exist_ok=True)
            commands.append(cli_command(["detect", *common, "--detector", "tracknet", "--out", str(det)], self._cfg()))
            args += ["--detections", str(det)]
        elif roi_s:
            args += ["--roi", roi_s]
        commands.append(cli_command(args, self._cfg()))
        self._save_state()

        def done(ok):
            if ok:
                self.show_result(out)
                self.nb.select(3)
        self.runner.run(commands, done, f"解析：{name}")

    # ================================================================ ④ 結果
    def show_result(self, d: Path) -> None:
        self.result_dir = Path(d)
        csv = self.result_dir / "result.csv"
        if not csv.exists():
            self.lbl_result.configure(text=f"結果がありません：{d}")
            return
        r = pd.read_csv(csv).iloc[0]
        lo, hi = r.get("speed_initial_ci_low_kmh"), r.get("speed_initial_ci_high_kmh")
        ci = f"（95%信頼区間 {lo:.0f}〜{hi:.0f} km/h）" if pd.notna(lo) else "（信頼区間は求められませんでした）"
        self.lbl_result.configure(text=f"{r['shot']}\n初速  {r['speed_initial_kmh']:.1f} km/h = {r['speed_initial_mps']:.1f} m/s  {ci}")
        notes = []
        if r["rms_px"] > 10:
            notes.append("⚠ 再投影誤差が大きいので、検出かキャリブレーションを確認してください。")
        if r["n_frames_used"] < 6:
            notes.append("⚠ 使えたフレームが少ないです。区間・ROI を見直してください。")
        log_file = self.result_dir / "fit_result.json"
        if log_file.exists():
            notes += ["・" + s for s in json.loads(log_file.read_text(encoding="utf-8")).get("log", [])]
        self.lbl_result2.configure(text=(
            f"平均速度  移動距離ベース {r['speed_mean_path_kmh']:.1f} km/h ／ 直線距離ベース {r['speed_mean_straight_kmh']:.1f} km/h\n"
            f"到達時の速さ {r['speed_arrival_kmh']:.1f} km/h　減速率 {r['deceleration_ratio']:.2f}　飛行時間 {r['flight_time_s']:.3f} s\n"
            f"参考：フレーム間差分（最初の区間）{r['naive_first_interval_kmh']:.1f} km/h\n"
            f"打点時刻 {r['t_impact_s']:.4f} s（{r['impact_method']}, ±{r['impact_sigma_s'] * 1000:.1f} ms）"
            f"　使ったフレーム {int(r['n_frames_used'])}　再投影誤差 {r['rms_px']:.2f} px"
            + ("\n" + "\n".join(notes) if notes else "")))
        self._show_result_image()

    def _show_result_image(self) -> None:
        if self.result_dir:
            self.img_result.show(self.result_dir / RESULT_IMAGES[self.get("result_img")])

    def _open_in_result(self, name: str) -> None:
        if self.result_dir is None:
            messagebox.showinfo("smash-speed", "まだ結果がありません。")
            return
        open_path(self.result_dir / name if name else self.result_dir)

    def _choose_result(self) -> None:
        d = filedialog.askdirectory(title="結果のフォルダ（result.csv があるところ）", initialdir=self.get("out_root"))
        if d:
            self.show_result(Path(d))

    # ================================================================ ⑤ 一覧
    def refresh_list(self) -> None:
        self.tree.delete(*self.tree.get_children())
        self._list_dirs.clear()
        root = Path(self.get("out_root"))
        files = sorted(root.glob("**/result.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in files:
            try:
                r = pd.read_csv(p).iloc[0]
            except (OSError, pd.errors.ParserError, IndexError):
                continue
            ci = (f"{r['speed_initial_ci_low_kmh']:.0f}–{r['speed_initial_ci_high_kmh']:.0f}"
                  if pd.notna(r.get("speed_initial_ci_low_kmh")) else "-")
            iid = self.tree.insert("", "end", values=(r["shot"], f"{r['speed_initial_kmh']:.1f}", ci,
                                                      f"{r['speed_mean_path_kmh']:.1f}", f"{r.get('capture_fps', float('nan')):.0f}",
                                                      r.get("impact_method", ""), f"{r['rms_px']:.1f}"))
            self._list_dirs[iid] = p.parent
        self.status(f"{len(self._list_dirs)} 件の結果があります。")

    def _open_from_list(self, _e) -> None:
        sel = self.tree.selection()
        if sel:
            self.show_result(self._list_dirs[sel[0]])
            self.nb.select(3)

    def save_summary(self) -> None:
        out = Path(self.get("out_root")) / "summary.csv"
        self.runner.run([cli_command(["summarize", "--results", self.get("out_root"), "--out", str(out)], self._cfg())],
                        lambda ok: ok and open_path(out), "集計表")

    # ================================================================ ⑥ 研究ツール
    def make_synth(self) -> None:
        speed = self.getf("syn_speed") or 80.0
        fps = self.getf("syn_fps") or 240.0
        out = Path(self.get("out_root")) / f"synthetic_{speed:g}mps_{fps:g}fps"
        args = ["synth", "--out", str(out), "--speed", f"{speed}", "--fps", f"{fps}", "--exposure", self.get("syn_exposure") or "0"]

        def done(ok):
            if not ok:
                return
            info = json.loads((out / "truth.json").read_text(encoding="utf-8"))
            for k, v in (("video", out / "synthetic.mp4"), ("calib", out / "calibration.json"), ("audio", out / "synthetic.wav"),
                         ("capture_fps", ""), ("hitter_x", f"{info['hitter_xy'][0]:.2f}"), ("hitter_y", f"{info['hitter_xy'][1]:.2f}"),
                         ("landing_x", f"{info['landing_xy'][0]:.2f}"), ("landing_y", f"{info['landing_xy'][1]:.2f}"),
                         ("start", "0"), ("end", f"{info['t_land_video_s'] + 0.15:.2f}"), ("shot", out.name + "_analysis"),
                         ("impact_method", list(IMPACT_METHODS)[0])):
                self.var(k).set(str(v))
            self.load_video()
            messagebox.showinfo("smash-speed", f"合成動画を読み込みました（正解の初速 {info['speed0_kmh']:.0f} km/h）。\n"
                                               "キャリブレーション・飛行平面・区間も設定済みなので、③の「解析を実行する」を押してください。")
            self.nb.select(2)
        self.runner.run([cli_command(args, self._cfg())], done, "合成データの作成")

    def annotate(self) -> None:
        if not self._need_video():
            return
        t0, t1 = self.getf("start"), self.getf("end")
        if t0 is None or t1 is None:
            messagebox.showinfo("smash-speed", "①で区間を設定してください。")
            return
        out = self.get("ann_out") or str(Path(self.get("out_root")) / "labels" / self._shot_name())
        self.var("ann_out").set(out)
        args = ["annotate", "--video", self.get("video"), "--start", f"{t0:.3f}", "--end", f"{t1:.3f}", "--out", out] + self._fps_args()
        det = Path(self.get("out_root")) / self._shot_name() / "detections.csv"
        if self.var("ann_init").get() and det.exists():
            args += ["--init", str(det)]
        self.var("low_labels").set(str(Path(out) / "labels.csv"))
        self.status("アノテーションのウィンドウで作業してください（Esc で保存して終了）。")
        self.runner.run([cli_command(args, self._cfg())], None, "アノテーション")

    def make_lowfps(self) -> None:
        if not self._need_video():
            return
        fps = [f for f in ("30", "60", "120") if self.var(f"low_{f}").get()]
        if not fps:
            return
        out = self.get("low_out") or str(Path(self.get("out_root")) / "lowfps")
        self.var("low_out").set(out)
        args = ["make-lowfps", "--video", self.get("video"), "--out", out, "--target-fps", *fps] + self._fps_args()
        if self.get("low_labels"):
            args += ["--labels", self.get("low_labels")]
        if self.var("low_blur").get():
            args.append("--blur")
        self.runner.run([cli_command(args, self._cfg())], lambda ok: ok and open_path(Path(out)), "疑似低 fps 動画")

    def validate_synthetic(self) -> None:
        out = Path(self.get("out_root")) / "validation_synthetic"
        args = ["validate", "--synthetic", "--n", self.get("val_n") or "100", "--out", str(out)]
        self.runner.run([cli_command(args, self._cfg())], lambda ok: ok and open_path(out / "mape_vs_fps.png"), "精度検証（合成）")

    def validate_dataset(self) -> None:
        if not self.get("val_index"):
            messagebox.showinfo("smash-speed", "実データの一覧 CSV を選んでください（列は README 参照）。")
            return
        out = Path(self.get("out_root")) / "validation"
        args = ["validate", "--dataset", self.get("val_index"), "--out", str(out)]
        if self.get("val_ref"):
            args += ["--reference", self.get("val_ref")]
        self.runner.run([cli_command(args, self._cfg())], lambda ok: ok and open_path(out), "精度検証（実データ）")

    def export_tracknet(self) -> None:
        if not self.get("exp_index"):
            messagebox.showinfo("smash-speed", "データ一覧 CSV を選んでください（列は README 参照）。")
            return
        out = Path(self.get("out_root")) / "tracknet_dataset"
        self.runner.run([cli_command(["export-tracknet", "--index", self.get("exp_index"), "--out", str(out)], self._cfg())],
                        lambda ok: ok and open_path(out), "TrackNetV3 形式で書き出す")

    def _new_override(self) -> None:
        p = filedialog.asksaveasfilename(title="上書き用の設定ファイル", defaultextension=".yaml",
                                         initialdir=str(PROJECT_ROOT / "configs"), initialfile="my_settings.yaml",
                                         filetypes=[("YAML", "*.yaml")])
        if not p:
            return
        Path(p).write_text(
            "# 既定の設定（configs/default.yaml）のうち、変えたい項目だけを書く。行頭の # を外すと有効になる。\n"
            "physics:\n"
            "  terminal_velocity_mps: 6.7      # シャトルの終端速度 [m/s]\n"
            "  # refine_plane: false           # 落下中のシャトルを解析するときは false\n"
            "# detection:\n"
            "#   baseline:\n"
            "#     diff_threshold: 15          # 小さくすると薄いシャトルも拾う（誤検出も増える）\n"
            "#   tracknet:\n"
            "#     repo_dir: C:/Users/Public/Claude/TrackNetV3\n"
            "#     tracknet_ckpt: C:/Users/Public/Claude/TrackNetV3/ckpts/TrackNet_best.pt\n"
            "#     inpaintnet_ckpt: C:/Users/Public/Claude/TrackNetV3/ckpts/InpaintNet_best.pt\n",
            encoding="utf-8")
        self.var("config").set(p)
        open_path(Path(p))

    # ================================================================ 共通
    def _cfg(self) -> str | None:
        return self.get("config") or None

    def _fps_args(self) -> list[str]:
        return ["--capture-fps", self.get("capture_fps")] if self.getf("capture_fps") else []

    def _need_video(self) -> bool:
        if self.meta is None:
            messagebox.showinfo("smash-speed", "先に「動画を開く…」で動画を読み込んでください。")
            return False
        return True

    def _need_calib(self) -> bool:
        if self.camera is None:
            messagebox.showinfo("smash-speed", "先に②でキャリブレーションをしてください。")
            self.nb.select(1)
            return False
        return True

    def _ask_file(self, key: str) -> None:
        p = filedialog.askopenfilename(initialdir=self.get("out_root"))
        if p:
            self.var(key).set(p)
            if key == "calib":
                self._load_camera()

    def _ask_dir(self, key: str) -> None:
        p = filedialog.askdirectory(initialdir=self.get(key) or self.get("out_root"))
        if p:
            self.var(key).set(p)

    def log(self, text: str) -> None:
        # onsets の出力（"時刻,強さ" の行）を拾う
        if getattr(self, "_capture_onsets", None) is not None:
            parts = text.strip().split(",")
            if len(parts) == 2:
                try:
                    self._capture_onsets.append((float(parts[0]), float(parts[1])))
                    return
                except ValueError:
                    pass
        self.txt_log.insert("end", text)
        self.txt_log.see("end")

    def status(self, text: str) -> None:
        self.lbl_status.configure(text=text)

    def _set_busy(self, busy: bool) -> None:
        self.btn_cancel.configure(state="normal" if busy else "disabled")
        if busy:
            self.prog.start(12)
        else:
            self.prog.stop()
            if getattr(self, "_capture_onsets", None) is not None:
                self.after(50, self._end_capture)

    def _end_capture(self) -> None:
        self._capture_onsets = None

    _SAVED = ("video", "capture_fps", "out_root", "calib", "start", "end", "shot", "hitter_x", "hitter_y", "landing_x",
              "landing_y", "config", "audio", "ann_out", "low_labels", "low_out", "val_index", "val_ref", "exp_index")

    def _save_state(self) -> None:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {k: self.get(k) for k in self._SAVED}
        STATE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    def _load_state(self) -> None:
        if not STATE_FILE.exists():
            return
        try:
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for k, v in data.items():
            if k in self._SAVED:
                self.var(k).set(v)
        if self.get("video") and Path(self.get("video")).exists():
            self.after(200, self.load_video)

    def _on_close(self) -> None:
        if self.runner.busy and not messagebox.askyesno("smash-speed", "実行中の処理を中止して終了しますか？"):
            return
        self.runner.cancel()
        self._save_state()
        self.destroy()


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
