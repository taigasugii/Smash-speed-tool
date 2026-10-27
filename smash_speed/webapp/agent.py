"""開発エージェント（``agent/run.ps1``）の様子を見る画面の API。

エージェントは本体とは別の作業用のコピー（既定は ``smash-speed`` の隣の ``smash-speed-agent``、
環境変数 ``SMASH_AGENT_DIR`` で変えられる）で動き、状態を ``.agent/`` に書く。ここではそれを読むだけで、
プロセスは起動しない。「1 つ進める」は ``.agent/REQUEST`` を置き、タスク スケジューラ（5 分おきの
``run.ps1 -OnRequest``）がそれを見て動かす。「止める」は ``.agent/STOP`` を置く。
「一時停止」は ``.agent/PAUSE`` を置き、「再開」はそれを消す。PC を使っているあいだの自動の一時停止は
``.agent/config.json`` の ``auto_pause`` で切り替える。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request

from smash_speed.config import PROJECT_ROOT

TAIL_BYTES = 400_000      # 作業の様子を出すときに読む、ログの末尾の大きさ
RESUME_MINUTES = 10       # 自動の一時停止から再開するまでの、操作のない時間（run.ps1 の既定と同じ）


def agent_dir() -> Path:
    d = os.environ.get("SMASH_AGENT_DIR")
    return Path(d) if d else PROJECT_ROOT.parent / "smash-speed-agent"


def _read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def alive(pid: Any) -> bool:
    """そのプロセスが動いているか（Windows の os.kill はプロセスを終了させてしまうため、API で調べる）。"""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, pid)        # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return bool(ok) and code.value == 259         # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def activity(log: Path, n: int = 25) -> list[dict[str, str]]:
    """Claude のログ（stream-json）から、最近の発言と使った道具を新しい順に取り出す。"""
    try:
        with open(log, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - TAIL_BYTES))
            raw = f.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    out: list[dict[str, str]] = []
    for line in raw.splitlines():
        if not line.startswith("{"):
            continue
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if m.get("type") != "assistant":
            continue
        for c in (m.get("message") or {}).get("content") or []:
            if c.get("type") == "text" and c.get("text", "").strip():
                out.append({"kind": "text", "text": c["text"].strip()[:400]})
            elif c.get("type") == "tool_use":
                inp = c.get("input") or {}
                what = inp.get("command") or inp.get("file_path") or inp.get("pattern") or inp.get("url") or ""
                out.append({"kind": "tool", "text": f"{c.get('name')}: {str(what)[:200]}"})
    return out[-n:][::-1]


def _section(text: str, start: str, end: str | None = None, max_chars: int = 6000) -> str:
    i = text.find(start)
    if i < 0:
        return ""
    i += len(start)
    j = text.find(end, i) if end else -1
    return text[i: j if j >= 0 else None].strip()[:max_chars]


def status() -> dict[str, Any]:
    d = agent_dir()
    st = d / ".agent"
    s = _read_json(st / "status.json") or {}
    running = alive((_read_json(st / "lock.json") or {}).get("pid"))
    if s.get("state") in ("running", "gate", "waiting_limit", "starting", "wrapping", "paused") and not running:
        s["state"] = "stopped"                      # 途中で PC が止まった・プロセスが落ちた
    hist = []
    try:
        for line in (st / "history.jsonl").read_text(encoding="utf-8-sig").splitlines()[-15:]:
            try:
                hist.append(json.loads(line))
            except ValueError:
                pass
    except OSError:
        pass
    journal = questions = ""
    try:
        journal = _section((d / "agent" / "JOURNAL.md").read_text(encoding="utf-8"), "\n---\n")
        questions = _section((d / "agent" / "QUESTIONS.md").read_text(encoding="utf-8"), "## 未回答", "## 済み")
    except OSError:
        pass
    log = Path(s["log"]) if s.get("log") else None
    cfg = _read_json(st / "config.json") or {}
    return {
        "dir": str(d), "installed": (d / "agent" / "run.ps1").exists(), "running": running, "status": s,
        "requested": (st / "REQUEST").exists(), "stop_requested": (st / "STOP").exists(),
        "paused": (st / "PAUSE").exists(), "auto_pause": cfg.get("auto_pause", True) is not False,
        "auto_pause_idle_minutes": cfg.get("auto_pause_idle_minutes") or RESUME_MINUTES,
        "history": hist[::-1], "journal": journal, "questions": questions,
        "activity": activity(log) if log and running else [],
    }


def _flag(name: str) -> None:
    st = agent_dir() / ".agent"
    st.mkdir(parents=True, exist_ok=True)
    (st / name).write_text(name.lower() + "\n", encoding="utf-8")


def install(app: Flask) -> None:
    @app.get("/api/agent")
    def agent_get():
        return jsonify(ok=True, **status())

    @app.post("/api/agent/request")
    def agent_request():
        s = status()
        if not s["installed"]:
            return jsonify(ok=False, error=f"エージェントの作業フォルダが見つかりません（{s['dir']}）。agent/README.md の準備をしてください。")
        if s["running"]:
            return jsonify(ok=False, error="もう動いています。")
        _flag("REQUEST")
        return jsonify(ok=True)

    @app.post("/api/agent/stop")
    def agent_stop():
        if not status()["running"]:
            return jsonify(ok=False, error="いまは動いていません。")
        _flag("STOP")
        return jsonify(ok=True)

    @app.post("/api/agent/pause")
    def agent_pause():
        s = status()
        if not s["installed"]:
            return jsonify(ok=False, error=f"エージェントの作業フォルダが見つかりません（{s['dir']}）。")
        if s["running"] and s["status"].get("mode") == "once":
            return jsonify(ok=False, error="「課題を 1 つ進める」の回は一時停止できません。やめるときは「止める」を押してください。")
        _flag("PAUSE")
        return jsonify(ok=True)

    @app.post("/api/agent/resume")
    def agent_resume():
        (agent_dir() / ".agent" / "PAUSE").unlink(missing_ok=True)
        return jsonify(ok=True)

    @app.post("/api/agent/auto_pause")
    def agent_auto_pause():
        # config.json のほかの項目（main_repo など）はそのまま残す
        p = agent_dir() / ".agent" / "config.json"
        if not p.parent.exists():
            return jsonify(ok=False, error="エージェントの作業フォルダが見つかりません。")
        cfg = _read_json(p) or {}
        cfg["auto_pause"] = bool((request.get_json(silent=True) or {}).get("enabled"))
        p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        return jsonify(ok=True, auto_pause=cfg["auto_pause"])
