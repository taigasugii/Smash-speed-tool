"""開発エージェントの画面の API と、品質チェックの追加チェックの読み込みのテスト。"""

import importlib.util
import json
import os
from pathlib import Path

import pytest

from smash_speed.webapp import agent
from smash_speed.webapp.server import app

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def adir(tmp_path, monkeypatch):
    d = tmp_path / "smash-speed-agent"
    (d / "agent").mkdir(parents=True)
    (d / ".agent").mkdir()
    (d / "agent" / "run.ps1").write_text("# dummy\n", encoding="utf-8")
    monkeypatch.setenv("SMASH_AGENT_DIR", str(d))
    return d


def _w(p: Path, obj) -> None:
    p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def test_not_installed(tmp_path, monkeypatch):
    monkeypatch.setenv("SMASH_AGENT_DIR", str(tmp_path / "none"))
    c = app.test_client()
    assert c.get("/api/agent").get_json()["installed"] is False
    r = c.post("/api/agent/request").get_json()
    assert r["ok"] is False and "見つかりません" in r["error"]


def test_status_history_journal_and_activity(adir):
    st = adir / ".agent"
    log = st / "claude.jsonl"
    log.write_text("\n".join([
        json.dumps({"type": "system"}),
        json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "B2 を始めます"}]}}),
        json.dumps({"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Bash",
                                                                   "input": {"command": "pytest -q"}}]}}),
    ]), encoding="utf-8")
    _w(st / "status.json", {"state": "running", "task": "B2", "log": str(log)})
    _w(st / "lock.json", {"pid": os.getpid()})
    (st / "history.jsonl").write_text(json.dumps({"start": "20261101-0100", "task": "B1", "result": "done"}) + "\n",
                                      encoding="utf-8")
    (adir / "agent" / "JOURNAL.md").write_text("# 記録\n\n---\n## B1 済み\n", encoding="utf-8")
    (adir / "agent" / "QUESTIONS.md").write_text("## 未回答\n\n- GPU のドライバは？\n\n## 済み\n", encoding="utf-8")
    r = app.test_client().get("/api/agent").get_json()
    assert r["running"] is True and r["status"]["task"] == "B2"
    assert r["history"][0]["task"] == "B1"
    assert r["journal"].startswith("## B1")
    assert "GPU" in r["questions"] and "済み" not in r["questions"]
    assert [a["kind"] for a in r["activity"]] == ["tool", "text"]       # 新しい順
    assert "pytest" in r["activity"][0]["text"]


def test_dead_process_shows_stopped(adir):
    _w(adir / ".agent" / "status.json", {"state": "running"})
    _w(adir / ".agent" / "lock.json", {"pid": 999999})
    r = app.test_client().get("/api/agent").get_json()
    assert r["running"] is False and r["status"]["state"] == "stopped"


def test_request_and_stop_only_write_flag_files(adir):
    c = app.test_client()
    assert c.post("/api/agent/stop").get_json()["ok"] is False           # 動いていないときは止められない
    assert c.post("/api/agent/request").get_json()["ok"] is True
    assert (adir / ".agent" / "REQUEST").exists()
    _w(adir / ".agent" / "lock.json", {"pid": os.getpid()})
    assert c.post("/api/agent/request").get_json()["ok"] is False        # もう動いている
    assert c.post("/api/agent/stop").get_json()["ok"] is True
    assert (adir / ".agent" / "STOP").exists()


def test_alive():
    assert agent.alive(os.getpid())
    assert not agent.alive(None) and not agent.alive(0) and not agent.alive("x")


def _gate():
    spec = importlib.util.spec_from_file_location("eval_gate", ROOT / "agent" / "eval_gate.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_extra_checks_are_loaded_and_broken_ones_fail(tmp_path, monkeypatch):
    g = _gate()
    (tmp_path / "check_ok.py").write_text("def run(ctx):\n    return {'ok': True, 'summary': ctx['config']['x']}\n",
                                          encoding="utf-8")
    (tmp_path / "check_broken.py").write_text("def run(ctx):\n    raise RuntimeError('boom')\n", encoding="utf-8")
    (tmp_path / "helper.py").write_text("raise SystemExit\n", encoding="utf-8")   # check_ で始まらないものは読まない
    monkeypatch.setattr(g, "CHECKS", tmp_path)
    res = {r["name"]: r for r in g.run_extra_checks({"x": "よし"})}
    assert set(res) == {"check_ok", "check_broken"}
    assert res["check_ok"]["ok"] and res["check_ok"]["summary"] == "よし"
    assert not res["check_broken"]["ok"] and "boom" in res["check_broken"]["detail"]
