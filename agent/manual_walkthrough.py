"""使い方の手引き（docs/manual.md）の手順を、合成の動画でヘッドレスのブラウザに通して確かめる（ROADMAP v2 の L1）。

自分のリポジトリの GUI（``python -m smash_speed.webapp``）を空いているポートで起こし、Edge（なければ Chrome）を
ヘッドレスで開いて、DevTools のプロトコル（外のライブラリを使わない小さな WebSocket）で画面のボタンを押していく。
手引きの写真（``docs/manual_img/*.png``）もここで撮る（合成の動画だけ。実動画のコマは使わない）。

    .venv\\Scripts\\python.exe agent\\manual_walkthrough.py [--port 8771] [--shots-dir docs\\manual_img] [--no-shots]

動画を選ぶ所（ファイルの選び方の窓）はヘッドレスでは開けないので、選んだ後に画面がするのと同じ ``openVideo`` を呼ぶ。
GUI の状態（``outputs/web_state.json``）は空にして始め、終わったら元に戻す。結果の保存先は ``outputs/manual_walk``。
結果は標準出力と ``outputs/manual_walkthrough.json``。夜のエージェントの品質チェックには入れない（ブラウザが要るため）。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
# 左のメニューの項目（``static/core.js`` の NAV）。手引きにすべて出てくることも確かめる
MENU = ["ホーム", "スマッシュを選ぶ", "コートを合わせる", "打つ人と着地点", "解析する", "結果", "いちばん速いスマッシュ",
        "ショットの一覧", "まとめて解析", "フォームを比べる", "関節・打点を付ける", "結果の一覧", "記録の推移", "研究ツール",
        "開発エージェント", "設定"]


# =============================================================================
# DevTools のプロトコル（WebSocket のテキストの枠だけ）
# =============================================================================
class Cdp:
    def __init__(self, ws_url: str) -> None:
        rest = ws_url.split("://", 1)[1]
        hostport, path = rest.split("/", 1)
        host, port = hostport.split(":")
        self.sock = socket.create_connection((host, int(port)), timeout=120)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((f"GET /{path} HTTP/1.1\r\nHost: {hostport}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                           f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            buf += self.sock.recv(4096)
        if b" 101 " not in buf.split(b"\r\n", 1)[0]:
            raise RuntimeError("DevTools に接続できませんでした")
        self.extra = buf.split(b"\r\n\r\n", 1)[1]
        self.next_id = 0
        self.errors: list[str] = []

    def _recv_exact(self, n: int) -> bytes:
        while len(self.extra) < n:
            chunk = self.sock.recv(1 << 16)
            if not chunk:
                raise ConnectionError("DevTools の接続が切れました")
            self.extra += chunk
        out, self.extra = self.extra[:n], self.extra[n:]
        return out

    def _recv(self) -> dict[str, Any]:
        data = b""
        while True:
            b0, b1 = self._recv_exact(2)
            n = b1 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._recv_exact(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._recv_exact(8))[0]
            data += self._recv_exact(n)
            if b0 & 0x80:
                return json.loads(data.decode("utf-8"))

    def _send(self, msg: dict[str, Any]) -> None:
        payload = json.dumps(msg).encode("utf-8")
        head = bytearray([0x81])
        n = len(payload)
        if n < 126:
            head.append(0x80 | n)
        elif n < 1 << 16:
            head += bytes([0x80 | 126]) + struct.pack(">H", n)
        else:
            head += bytes([0x80 | 127]) + struct.pack(">Q", n)
        mask = os.urandom(4)
        head += mask
        self.sock.sendall(bytes(head) + bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))

    def call(self, method: str, **params: Any) -> dict[str, Any]:
        self.next_id += 1
        mid = self.next_id
        self._send({"id": mid, "method": method, "params": params})
        while True:
            m = self._recv()
            if m.get("method") == "Runtime.exceptionThrown":
                d = m["params"]["exceptionDetails"]
                self.errors.append(str(d.get("exception", {}).get("description") or d.get("text")))
            if m.get("id") == mid:
                if "error" in m:
                    raise RuntimeError(m["error"])
                return m.get("result", {})

    def js(self, expr: str) -> Any:
        r = self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        if "exceptionDetails" in r:
            raise RuntimeError(f"{expr[:80]}…: {r['exceptionDetails'].get('exception', {}).get('description')}")
        return r.get("result", {}).get("value")


# =============================================================================
# 手順
# =============================================================================
class Walk:
    def __init__(self, cdp: Cdp, base: str, shots_dir: Path | None) -> None:
        self.c, self.base, self.shots_dir = cdp, base, shots_dir
        self.log: list[dict[str, Any]] = []

    def step(self, name: str, ok: bool, note: str = "") -> None:
        self.log.append({"step": name, "ok": bool(ok), "note": note})
        print(("OK " if ok else "NG ") + name + (f"：{note}" if note else ""), flush=True)

    def wait(self, cond: str, timeout: float = 60.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                if self.c.js(f"!!({cond})"):
                    return True
            except RuntimeError:
                pass
            time.sleep(0.5)
        return False

    def click(self, text: str, within: str = "document") -> bool:
        """``text`` を含むボタン（いちばん短い文字のもの）を押す。"""
        ok = bool(self.c.js(f"""(() => {{
          const bs = [...({within}).querySelectorAll('button')].filter(b => b.offsetParent !== null &&
            b.textContent.includes({json.dumps(text)}));
          bs.sort((a, b) => a.textContent.length - b.textContent.length);
          if (!bs.length) return false; bs[0].click(); return true; }})()"""))
        if not ok:
            print("  （押せませんでした。画面：" + (self.c.js("document.querySelector('#content').innerText.slice(0, 300)") or "")
                  .replace(chr(10), " / ") + "）", flush=True)
        return ok

    def task_id(self) -> str | None:
        return self.c.js("Task.id")

    def wait_task(self, prev: str | None, timeout: float = 900.0) -> str:
        """``prev`` の後に始まった処理（画面の下）が終わるまで待ち、終わり方（done / failed / cancelled）を返す。
        ``prev`` が今の処理なら、その処理を待つ。"""
        if not self.wait(f"Task.id && Task.id !== {json.dumps(prev)}", 30):
            if prev is None:
                return "not started"
        tid = self.task_id() or prev
        t0 = time.time()
        while time.time() - t0 < timeout:
            r = json.loads(urllib.request.urlopen(f"{self.base}/api/task/{tid}?since=999999", timeout=10).read())
            if r.get("status") != "running":
                time.sleep(1.5)       # 画面が終わりを受け取って次の画面へ移るまで
                return r.get("status", "")
            time.sleep(1.0)
        return "timeout"

    def menu(self, label: str) -> bool:
        ok = self.click(label, "document.querySelector('#nav')")
        time.sleep(0.8)
        return ok

    def shot(self, name: str, width: int = 900) -> None:
        if self.shots_dir is None:
            return
        import cv2
        import numpy as np

        r = self.c.call("Page.captureScreenshot", format="png")
        img = cv2.imdecode(np.frombuffer(base64.b64decode(r["data"]), np.uint8), cv2.IMREAD_COLOR)
        if img.shape[1] > width:
            img = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
        self.shots_dir.mkdir(parents=True, exist_ok=True)
        try:   # 画面の写真は色が少ないので、256 色にして小さくする
            from PIL import Image

            Image.fromarray(img[..., ::-1]).quantize(256).save(self.shots_dir / f"{name}.png", optimize=True)
        except ImportError:
            cv2.imwrite(str(self.shots_dir / f"{name}.png"), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])


def _wait_file(p: Path, timeout: float) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if p.exists():
            return True
        time.sleep(1.0)
    return False


STATE_FILE = ROOT / "outputs" / "web_state.json"   # GUI の状態（``webapp.server.STATE_FILE``）


def run(port: int, shots_dir: Path | None, work: Path) -> list[dict[str, Any]]:
    """GUI の状態を空にし（終わったら元に戻す）、結果の保存先を ``work`` にして手順を通す。"""
    saved = STATE_FILE.read_bytes() if STATE_FILE.exists() else None
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps({"out_root": str(work)}), encoding="utf-8")
    try:
        return _run(port, shots_dir, work)
    finally:
        if saved is None:
            STATE_FILE.unlink(missing_ok=True)
        else:
            STATE_FILE.write_bytes(saved)


def _run(port: int, shots_dir: Path | None, work: Path) -> list[dict[str, Any]]:
    base = f"http://127.0.0.1:{port}"
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    server = subprocess.Popen([sys.executable, "-m", "smash_speed.webapp", "--no-browser", "--port", str(port)],
                              cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    exe = next((b for b in BROWSERS if Path(b).exists()), None)
    if exe is None:
        raise SystemExit("Edge・Chrome が見つかりません")
    prof = Path(tempfile.mkdtemp(prefix="manual_walk_"))
    dport = port + 1000
    browser = subprocess.Popen([exe, "--headless=new", f"--remote-debugging-port={dport}", f"--user-data-dir={prof}",
                                "--window-size=1400,900", "--no-first-run", "about:blank"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(base + "/api/ping", timeout=2)
                pages = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{dport}/json", timeout=2).read())
                ws = next(p["webSocketDebuggerUrl"] for p in pages if p.get("type") == "page")
                break
            except (OSError, StopIteration, ValueError):
                time.sleep(1.0)
        else:
            raise SystemExit("GUI かブラウザが起きませんでした")
        c = Cdp(ws)
        c.call("Runtime.enable")
        c.call("Page.enable")
        c.call("Emulation.setDeviceMetricsOverride", width=1400, height=900, deviceScaleFactor=1, mobile=False)
        w = Walk(c, base, shots_dir)
        c.call("Page.navigate", url=base + "/")
        w.step("GUI を開く", w.wait("typeof App !== 'undefined' && App.current === 'home' && "
                                    "document.querySelectorAll('#nav button').length > 0", 30))
        time.sleep(1.0)
        labels = c.js("[...document.querySelectorAll('#nav button')].map(b => b.querySelector('span:not(.num):not(.sub)')"
                      ".firstChild.textContent.trim())")
        w.step("左のメニューの項目", labels == MENU, "、".join(labels))
        w.shot("home")

        # ---- スマッシュの初速：お試し用の動画で手順 1〜5
        w.step("「お試し用の動画を作って開く」", w.click("お試し用の動画を作って開く"))
        w.wait_task(None, 300)
        w.step("手順 4「解析する」の画面になる", w.wait("App.current === 'analyze' && App.meta", 60))
        for v in ("clip", "calib", "plane"):
            c.js(f"go('{v}')")
            time.sleep(1.5)
            w.step(f"{v} の画面", c.js("document.querySelector('#content').children.length > 0"))
            if v == "calib":
                w.shot("calib_done")
        c.js("go('analyze')")
        time.sleep(1.0)
        w.shot("analyze")
        prev = w.task_id()
        w.step("「▶ 解析する」", w.click("▶ 解析する"))
        st = w.wait_task(prev, 600)
        w.step("解析が終わる", w.wait("App.current === 'result'", 60), st)
        time.sleep(2.0)
        speed = c.js("(document.querySelector('#content').innerText.match(/([0-9.]+)\\s*km\\/h/) || [])[1] || ''")
        w.step("手順 5「結果」に初速が出る", bool(speed), f"{speed} km/h（正解 288 km/h）")
        w.shot("result")

        # ---- ホームの「この動画の進み具合」から：動画 → 基礎打ちの解析 → コート合わせ → ショットの一覧 → レポート（L2）
        #      合成のラリーの動画（打球音の WAV・仮想カメラのコート合わせつき）
        rally = work / "manual_rally"
        if not (rally / "rally.mp4").exists():
            subprocess.run([sys.executable, "-m", "smash_speed.synth_rally", str(rally), "--shots", "8", "--fps", "60"],
                           cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL)
        video, wav, cal = (str(rally / n) for n in ("rally.mp4", "rally.wav", "calibration.json"))
        w.step("ラリーの動画を開く（openVideo。「動画を選ぶ…」の後と同じ）",
               c.js(f"openVideo({json.dumps(video)}, null).then(r => !!r)"))
        c.js(f"App.save({{audio: {json.dumps(wav)}}})")
        w.menu("ホーム")

        def home_next() -> str:
            w.wait("document.querySelector('.home-status')", 15)
            return c.js("(document.querySelector('.hs-row.next') || {dataset: {}}).dataset.k || ''")

        w.step("ホームに進み具合が出て、次は「コートを合わせる」", home_next() == "calib")
        prev = w.task_id()
        w.step("「基礎打ちを解析する」（コート合わせの前にできる）", w.click("基礎打ちを解析する"))
        st = w.wait_task(prev, 900)
        w.wait("App.current === 'home' && document.querySelector('.home-status')", 30)
        time.sleep(1.5)
        txt = c.js("document.querySelector('.hs-row[data-k=drill]').innerText")
        w.step("基礎打ちのまとめ（打った数）がホームに出る", "打った数" in (txt or "") and "✓" in (txt or ""),
               f"{st}：{(txt or '').splitlines()[-2:] if txt else ''}")
        w.step("次も「コートを合わせる」", home_next() == "calib")
        w.shot("home_status")
        w.step("「コートを合わせる →」", w.click("コートを合わせる →", "document.querySelector('.home-status')"))
        w.wait("App.current === 'calib'", 10)
        time.sleep(1.5)
        r = c.js(f"api.post('/api/calib/load', {{path: {json.dumps(cal)}}}).then(r => {{ App.calib = r.calib; return r.ok; }})")
        w.step("コート合わせを読み込む（「別のファイルを読み込む…」と同じ）", r)
        w.menu("ホーム")
        w.step("次は「ショットの一覧を作る」", home_next() == "shots")
        w.step("「ショットの一覧へ →」", w.click("ショットの一覧へ →", "document.querySelector('.home-status')"))
        w.step("ショットの一覧の画面", w.wait("App.current === 'shots' && [...document.querySelectorAll('#content button')]"
                                          ".some(b => b.textContent.includes('解析を始める'))", 30))
        w.shot("shots_start")
        prev = w.task_id()
        w.step("「解析を始める」", w.click("解析を始める"))
        st = w.wait_task(prev, 900)
        w.step("ショットの一覧ができる", w.wait("document.querySelector('#content table')", 60), st)
        time.sleep(1.5)
        n = c.js("document.querySelectorAll('#content table tbody tr').length")
        w.step("表にショットが並ぶ", (n or 0) > 0, f"{n} 本（正解 8 本）")
        w.shot("shots")
        w.menu("ホーム")
        w.step("次は「結果を見る」", home_next() == "view")
        w.shot("home_status_done")
        c.js("window.__opened = null; window.open = (u) => { window.__opened = u; return null; }")
        w.click("レポートを開く", "document.querySelector('.home-status')")
        opened = c.js("window.__opened")
        rep = urllib.request.urlopen(base + (opened or "/report/shots"), timeout=120).status
        w.step("「レポートを開く」でショットの一覧のレポートが開く", opened == "/report/shots" and rep == 200, str(opened))

        # ---- まとめて解析：合成の動画 3 本（1 本はコート合わせつき）を順に解析・途中で止める・続きから（L3）
        vdir = work / "videos"
        vdir.mkdir(exist_ok=True)
        now = time.time()
        for i in range(3):
            d = work / f"q{i}"
            if not (d / "rally.mp4").exists():
                subprocess.run([sys.executable, "-m", "smash_speed.synth_rally", str(d), "--shots", "6", "--fps", "60",
                                "--seed", str(10 + i)], cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL)
            v = vdir / f"drill_{i + 1}.mp4"
            shutil.copy(d / "rally.mp4", v)
            os.utime(v, (now - 100 + i, now - 100 + i))
        shutil.copy(work / "q0" / "calibration.json", work / "drill_1_calibration.json")
        c.js(f"App.save({{videos_dir: {json.dumps(str(vdir))}}})")
        w.menu("まとめて解析")
        w.step("まとめて解析の画面に 3 本", w.wait("App.current === 'queue' && "
                                                 "document.querySelectorAll('#content tbody tr').length === 3", 15))
        w.shot("queue")
        t0 = time.time()
        prev = w.task_id()
        w.step("「選んだ動画を順に解析する」", w.click("選んだ動画を順に解析する"))
        w.wait(f"Task.id && Task.id !== {json.dumps(prev)}", 30)
        qid = w.task_id()
        w.step("1 本目の基礎打ちの解析が終わる", _wait_file(work / "drills" / "drill_1" / "drill_detect.json", 600))
        w.click("中止", "document.querySelector('#taskbar')")
        st = w.wait_task(qid, 180)
        w.step("「中止」で止まる", st == "cancelled", st)
        time.sleep(2.0)
        c.js("go('queue')")
        time.sleep(2.0)
        todo = c.js("[...document.querySelectorAll('#content tbody input')].map(x => !x.disabled)")
        drills = [(work / "drills" / f"drill_{i + 1}" / "drill_detect.json").exists() for i in range(3)]
        w.step("止めた所までが済み・残りはまだ", drills[0] and not drills[2] and bool(todo) and todo[2],
               f"基礎打ちの解析 済み {drills}、選べる（まだ）{todo}")
        w.step("もう一度押して続きから", w.click("選んだ動画を順に解析する"))
        st = w.wait_task(qid, 1800)
        time.sleep(2.0)
        c.js("go('queue')")
        time.sleep(2.0)
        todo = c.js("[...document.querySelectorAll('#content tbody input')].filter(x => !x.disabled).length")
        w.step("3 本とも済み（ショットの一覧はコート合わせのある 1 本だけ）",
               todo == 0 and (work / "drill_1_shots" / "shot_list.json").exists() and not (work / "drill_2_shots").exists(),
               f"{st}、合わせて {time.time() - t0:.0f} 秒")
        w.shot("queue_done")

        # ---- 記録の推移・結果の一覧・そのほかの画面
        w.menu("記録の推移")
        w.click("作り直す")
        time.sleep(4.0)
        w.step("記録の推移", w.wait("App.current === 'history'", 10))
        w.shot("history")
        for lab in MENU:
            w.menu(lab)
            time.sleep(1.0)
            w.step(f"「{lab}」を開く", c.js("document.querySelector('#content').innerText.length > 0"))
            if lab in ("結果の一覧", "いちばん速いスマッシュ", "フォームを比べる", "設定"):
                w.shot({"結果の一覧": "list", "いちばん速いスマッシュ": "fastest", "フォームを比べる": "form",
                        "設定": "settings"}[lab])
        w.step("画面のスクリプトのエラー", not c.errors, " / ".join(c.errors[:3]))
        return w.log
    finally:
        browser.terminate()
        try:
            urllib.request.urlopen(urllib.request.Request(base + "/api/shutdown", method="POST"), timeout=5)
        except OSError:
            pass
        time.sleep(1.0)
        server.terminate()
        shutil.rmtree(prof, ignore_errors=True)


def main() -> int:
    if (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8772)
    ap.add_argument("--shots-dir", type=Path, default=ROOT / "docs" / "manual_img")
    ap.add_argument("--no-shots", action="store_true", help="写真を撮らない")
    a = ap.parse_args()
    work = ROOT / "outputs" / "manual_walk"
    t0 = time.time()
    log = run(a.port, None if a.no_shots else a.shots_dir, work)
    manual = (ROOT / "docs" / "manual.md").read_text(encoding="utf-8") if (ROOT / "docs" / "manual.md").exists() else ""
    missing = [m for m in MENU if f"「{m}」" not in manual]
    log.append({"step": "手引きに左のメニューの項目がすべて出てくる", "ok": not missing, "note": "、".join(missing)})
    print(("OK " if not missing else "NG ") + "手引きに左のメニューの項目がすべて出てくる" + (f"：足りない {missing}" if missing else ""))
    bad = [s for s in log if not s["ok"]]
    print(f"{len(log) - len(bad)}/{len(log)} 通過（{time.time() - t0:.0f} 秒）")
    (ROOT / "outputs" / "manual_walkthrough.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
