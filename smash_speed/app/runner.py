"""GUI からコマンド（``python -m smash_speed.cli ...``）を裏で実行し、出力をログに流す。"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path

from smash_speed.config import PROJECT_ROOT


def python_executable() -> str:
    """子プロセス用の python.exe（GUI を pythonw.exe で起動していても、標準出力を読める方を使う）。"""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and (exe.parent / "python.exe").exists():
        return str(exe.parent / "python.exe")
    return str(exe)


def cli_command(args: list[str], config: str | None = None) -> list[str]:
    """``smash-speed`` のサブコマンドを実行するコマンド列。"""
    cmd = [python_executable(), "-m", "smash_speed.cli"]
    if config:
        cmd += ["--config", config]
    return cmd + [str(a) for a in args]


class TaskRunner:
    """コマンドを1つずつ順番に実行する。同時に実行できるのは1組だけ。"""

    def __init__(self, root, log: Callable[[str], None], on_busy: Callable[[bool], None]):
        self.root = root
        self.log = log
        self.on_busy = on_busy
        self.q: queue.Queue = queue.Queue()
        self.proc: subprocess.Popen | None = None
        self.busy = False
        self._cancel = False
        self.root.after(100, self._poll)

    def run(self, commands: list[list[str]], on_done: Callable[[bool], None] | None = None,
            title: str = "") -> bool:
        """``commands`` を順に実行する。途中で失敗したら止める。``on_done(成功したか)`` を GUI のスレッドで呼ぶ。"""
        if self.busy:
            self.log("[実行中の処理があります。終わるまで待つか「中止」を押してください]\n")
            return False
        self.busy = True
        self._cancel = False
        self.on_busy(True)
        if title:
            self.log(f"\n===== {title} =====\n")
        threading.Thread(target=self._worker, args=(commands, on_done), daemon=True).start()
        return True

    def cancel(self) -> None:
        self._cancel = True
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()

    def _worker(self, commands: list[list[str]], on_done) -> None:
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        ok = True
        for cmd in commands:
            if self._cancel:
                ok = False
                break
            self.q.put(("log", "$ " + " ".join(_short(c) for c in cmd[2:]) + "\n"))
            try:
                self.proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT, env=env, creationflags=flags,
                                             text=True, encoding="utf-8", errors="replace")
            except OSError as e:
                self.q.put(("log", f"[起動できませんでした] {e}\n"))
                ok = False
                break
            for line in self.proc.stdout:
                self.q.put(("log", line))
            rc = self.proc.wait()
            if rc != 0:
                self.q.put(("log", "[中止しました]\n" if self._cancel else f"[失敗しました（終了コード {rc}）]\n"))
                ok = False
                break
        self.q.put(("done", (ok, on_done)))

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self.log(payload)
                else:
                    ok, cb = payload
                    self.busy = False
                    self.proc = None
                    self.on_busy(False)
                    self.log("[完了]\n" if ok else "")
                    if cb is not None:
                        cb(ok)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)


def _short(s: str) -> str:
    return f'"{s}"' if " " in s else s
