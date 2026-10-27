// 開発エージェント：自動で機能を足していく Claude の様子を見る・1 つ進めてもらう・止める
"use strict";

const AGENT_STATE = {
  running: ["作業中", "ok"], gate: ["品質チェック中", "ok"], starting: ["準備中", "ok"], wrapping: ["区切りまで作業中", "warn"],
  waiting_limit: ["利用枠の回復待ち", "warn"], idle: ["止まっています", ""], stopped: ["途中で止まりました", "warn"],
  error: ["エラーで止まりました", "bad"],
};
const AGENT_RESULT = {
  done: ["済み", "ok"], partial: ["途中まで", "warn"], failed: ["できなかった", "bad"], no_task: ["課題なし", ""],
  gate_failed: ["品質チェック NG・取り消し", "bad"], limit: ["利用枠の上限", "warn"], error: ["エラー", "bad"],
};

App.views.agent = {
  title: "開発エージェント",
  timer: null,

  async render(el) {
    this.el = el;
    await this.refresh();
    this.timer = setInterval(() => { if (App.current === "agent") this.refresh(); }, 10000);
  },
  leave() { clearInterval(this.timer); this.timer = null; },

  async refresh() {
    const r = await api.get("/api/agent");
    const el = this.el;
    el.innerHTML = "";
    if (!r.installed) {
      el.appendChild(h(`<div class="card"><h2>エージェントはまだ準備されていません</h2>
        <p class="desc">デスクトップで作業用のフォルダ（<code>${esc(r.dir)}</code>）を作り、タスク スケジューラに登録すると使えるようになります。
        手順は <code>agent/README.md</code> にあります。</p></div>`));
      return;
    }
    this.renderHead(el, r);
    this.renderBody(el, r);
  },

  // ---- いまの状態と、1 つ進める・止めるボタン
  renderHead(el, r) {
    const s = r.status || {};
    const [label, kind] = AGENT_STATE[s.state] || ["まだ動いたことがありません", ""];
    const until = s.deadline ? new Date(s.deadline).toLocaleTimeString("ja-JP", { hour: "2-digit", minute: "2-digit" }) : "";
    const head = h(`<div class="card"><div class="row" style="justify-content:space-between;align-items:flex-start">
        <div><h2 style="margin:0 0 6px">いまの状態 <span class="badge ${kind}">${label}</span></h2>
          <div class="muted small">${esc(s.message || "")}${r.running && until ? `（${until} に止まります）` : ""}</div>
          ${s.task ? `<div style="margin-top:6px">課題：<b>${esc(s.task)}</b></div>` : ""}</div>
        <div class="row"><button class="btn primary" id="req" ${r.running || r.requested ? "disabled" : ""}>${r.requested ? "まもなく始まります…" : "課題を 1 つ進める"}</button>
          <button class="btn danger" id="stop" ${r.running && !r.stop_requested ? "" : "disabled"}>${r.stop_requested ? "止めています…" : "止める"}</button></div></div>
      <p class="muted small" style="margin:10px 0 0">毎晩 0:00〜12:00 は自動で動き続けます。それ以外の時間は「課題を 1 つ進める」を押すと、5 分以内に始まります（1 回は 3 時間が目安）。止める時刻を過ぎた作業は、区切りのいい所まで終えてから止まります。
        作業の結果は GitHub の PR にまとまります。${s.pr_url ? `<a href="${esc(s.pr_url)}" target="_blank" rel="noopener">PR を開く</a>（直してほしい所はコメントすると、次の回に反映します）` : ""}</p></div>`);
    el.appendChild(head);
    $("#req", head).onclick = async () => {
      const x = await api.post("/api/agent/request");
      if (!x.ok) return toast(x.error, "bad");
      toast("お願いしました。5 分以内に始まります");
      this.refresh();
    };
    $("#stop", head).onclick = async () => {
      if (!confirm("止めますか？ コミットしていない作業は取り消されます（パッチとして残ります）。")) return;
      const x = await api.post("/api/agent/stop");
      if (!x.ok) return toast(x.error, "bad");
      toast("止めています（数十秒かかります）");
      this.refresh();
    };
  },

  // ---- いまの作業・質問・これまでの回・記録
  renderBody(el, r) {
    if (r.activity.length) {
      const a = h(`<div class="card"><h2>いまの作業（新しい順）</h2><div class="stack small" style="margin-top:8px"></div></div>`);
      for (const x of r.activity) {
        $(".stack", a).appendChild(h(`<div class="${x.kind === "tool" ? "muted mono" : ""}" style="white-space:pre-wrap;word-break:break-all">${x.kind === "tool" ? "🔧 " : "💬 "}${esc(x.text)}</div>`));
      }
      el.appendChild(a);
    }
    if (r.questions && r.questions.trim() !== "（なし）") {
      el.appendChild(h(`<div class="card" style="border-color:#f59e0b"><h2>あなたへの質問</h2>
        <p class="desc">答えは PR にコメントしてください。次の回に反映します。</p>
        <pre class="mono" style="white-space:pre-wrap;margin:0">${esc(r.questions)}</pre></div>`));
    }
    const hist = h(`<div class="card"><h2>これまでの回</h2><div style="overflow:auto"><table class="tbl"><thead><tr>
      <th>始まり</th><th>課題</th><th>結果</th><th>要約</th></tr></thead><tbody></tbody></table></div></div>`);
    const tb = $("tbody", hist);
    if (!r.history.length) tb.appendChild(h(`<tr><td colspan="4" class="muted">まだありません</td></tr>`));
    for (const x of r.history) {
      const [rl, rk] = AGENT_RESULT[x.result] || [x.result || "—", ""];
      const t = (x.start || "").replace(/^(\d{4})(\d\d)(\d\d)-(\d\d)(\d\d).*/, "$2/$3 $4:$5");
      tb.appendChild(h(`<tr><td class="small">${esc(t)}</td><td>${esc(x.task || "—")}</td><td><span class="badge ${rk}">${rl}</span>${x.pushed ? " ↑" : ""}</td><td class="small">${esc(x.summary || "")}</td></tr>`));
    }
    el.appendChild(hist);
    if (r.journal) {
      el.appendChild(h(`<div class="card"><h2>作業の記録</h2><pre class="mono" style="white-space:pre-wrap;margin:0;max-height:60vh;overflow:auto">${esc(r.journal)}</pre></div>`));
    }
  },
};
