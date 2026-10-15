// 共通：API・状態・画面の切り替え・通知・処理パネル
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const h = (html) => { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstElementChild; };
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(+v)) ? "—" : (+v).toFixed(d);
const fileUrl = (p) => "/file?path=" + encodeURIComponent(p) + "&t=" + Date.now();

const api = {
  async get(url) { const r = await fetch(url); return r.json(); },
  async post(url, data = {}) {
    const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
    return r.json();
  },
};

// アプリ全体の状態（サーバーにも保存される）
const App = {
  state: {},
  meta: null,        // 開いている動画
  calib: null,       // 読み込んだキャリブレーションの情報
  landmarks: null,   // ランドマークとコートの線分
  current: "home",
  views: {},
  async save(patch) { Object.assign(this.state, patch); await api.post("/api/state", patch); refreshNav(); },
};

function toast(msg, kind = "") {
  const t = h(`<div class="toast ${kind}">${esc(msg)}</div>`);
  document.body.appendChild(t);
  setTimeout(() => t.remove(), kind === "bad" ? 6000 : 3500);
}

// ---------------- 画面（左のメニュー） ----------------
const NAV = [
  { group: "はじめに" },
  { id: "home", num: "⌂", label: "ホーム", sub: "使い方と動画の選択" },
  { group: "解析の手順" },
  { id: "clip", num: 1, label: "スマッシュを選ぶ", sub: "動画の中の区間" },
  { id: "calib", num: 2, label: "コートを合わせる", sub: "キャリブレーション" },
  { id: "plane", num: 3, label: "打つ人と着地点", sub: "シャトルが飛ぶ面" },
  { id: "analyze", num: 4, label: "解析する", sub: "設定を確認して実行" },
  { id: "result", num: 5, label: "結果", sub: "初速・平均速度" },
  { group: "そのほか" },
  { id: "list", num: "≡", label: "結果の一覧", sub: "これまでの解析" },
  { id: "tools", num: "⚗", label: "研究ツール", sub: "ラベル付け・検証など" },
  { id: "settings", num: "⚙", label: "設定", sub: "出力先・詳細設定" },
];

function stepDone(id) {
  const s = App.state;
  switch (id) {
    case "clip": return s.start != null && s.end != null && +s.end > +s.start;
    case "calib": return !!(App.calib && !App.calib.error);
    case "plane": return !!(s.hitter && s.landing);
    case "analyze": case "result": return !!s.last_result_ok;
    default: return false;
  }
}

function refreshNav() {
  const nav = $("#nav");
  nav.innerHTML = "";
  for (const n of NAV) {
    if (n.group) { nav.appendChild(h(`<div class="group">${n.group}</div>`)); continue; }
    const done = stepDone(n.id);
    const b = h(`<button class="${App.current === n.id ? "active" : ""} ${done ? "done" : ""}">
      <span class="num">${done ? "✓" : n.num}</span><span>${n.label}<span class="sub">${n.sub}</span></span></button>`);
    b.onclick = () => go(n.id);
    nav.appendChild(b);
  }
  $("#foot-video").textContent = App.meta ? `動画：${App.meta.name}` : "動画：未選択";
}

async function go(id, opts = {}) {
  const prev = App.views[App.current];
  if (prev && prev.leave) prev.leave();
  App.current = id;
  const v = App.views[id];
  const content = $("#content");
  content.innerHTML = "";
  $("#top-actions").innerHTML = "";
  $("#page-title").textContent = v.title;
  $("#page-crumb").textContent = v.crumb ? v.crumb() : "";
  refreshNav();
  await v.render(content, opts);
}

function requireVideo(container) {
  if (App.meta) return true;
  container.appendChild(h(`<div class="card"><h2>まず動画を開いてください</h2>
    <p class="desc">ホームで解析したい動画を選ぶと、この画面が使えるようになります。</p>
    <button class="btn primary" onclick="go('home')">ホームへ</button></div>`));
  return false;
}

function requireCalib(container) {
  if (App.calib && !App.calib.error) return true;
  container.appendChild(h(`<div class="card"><h2>先にコートを合わせてください</h2>
    <p class="desc">画面上の位置を実際の距離（メートル）に直すために、手順 2 の「コートを合わせる」が必要です。</p>
    <button class="btn primary" onclick="go('calib')">手順 2 へ</button></div>`));
  return false;
}

// ---------------- 処理パネル（下） ----------------
const Task = {
  id: null, n: 0, timer: null, onDone: null,
  async start(kind, data = {}, onDone = null, title = "") {
    const r = await api.post(`/api/run/${kind}`, data);
    if (!r.ok) { toast(r.error, "bad"); return false; }
    this.id = r.task; this.n = 0; this.onDone = onDone;
    const bar = $("#taskbar");
    bar.classList.remove("hidden", "done", "failed");
    $("#task-title").textContent = title || "処理中…";
    $("#task-status").textContent = "実行中";
    $("#task-log").textContent = "";
    $("#task-cancel").classList.remove("hidden");
    clearInterval(this.timer);
    this.timer = setInterval(() => this.poll(), 500);
    return true;
  },
  async poll() {
    const r = await api.get(`/api/task/${this.id}?since=${this.n}`);
    if (!r.ok) return;
    this.n = r.n;
    if (r.lines.length) {
      const pre = $("#task-log");
      pre.textContent += r.lines.join("\n") + "\n";
      pre.scrollTop = pre.scrollHeight;
      const last = r.lines.filter((l) => l.trim() && !l.startsWith("$")).pop();
      if (last) $("#task-status").textContent = last.slice(0, 90);
    }
    if (r.status !== "running") {
      clearInterval(this.timer);
      const bar = $("#taskbar");
      bar.classList.add(r.status === "done" ? "done" : "failed");
      $("#task-title").textContent = (r.status === "done" ? "✓ " : "✕ ") + r.title;
      $("#task-status").textContent = { done: "完了しました", failed: "失敗しました（ログを確認してください）", cancelled: "中止しました" }[r.status];
      $("#task-cancel").classList.add("hidden");
      if (r.status !== "done") $("#task-log").classList.remove("hidden");
      const cb = this.onDone; this.onDone = null;
      if (cb) cb(r.status === "done", this.id);
    }
  },
};

$("#task-cancel").onclick = () => Task.id && api.post(`/api/task/${Task.id}/cancel`);
$("#task-close").onclick = () => $("#taskbar").classList.add("hidden");
$("#task-log-toggle").onclick = () => {
  const pre = $("#task-log");
  pre.classList.toggle("hidden");
  $("#task-log-toggle").textContent = pre.classList.contains("hidden") ? "ログを表示" : "ログを隠す";
};
$("#btn-quit").onclick = async () => {
  if (!confirm("アプリを終了しますか？（実行中の処理も止まります）")) return;
  await api.post("/api/shutdown");
  document.body.innerHTML = '<div style="padding:40px;font-family:sans-serif">終了しました。このタブは閉じて構いません。</div>';
};

// ファイル選択（この PC のダイアログを開く）
async function pickPath(kind, opts = {}) {
  const r = await api.post("/api/pick", { kind, ...opts });
  return r.ok ? r.path : "";
}

function pathField(label, key, kind = "file", opts = {}) {
  const el = h(`<div class="field"><label>${label}</label><div class="pathbox">
    <input class="input" value="${esc(App.state[key] || "")}" placeholder="${esc(opts.placeholder || "")}"><button class="btn sm">選ぶ…</button></div></div>`);
  const input = $("input", el);
  input.onchange = () => App.save({ [key]: input.value.trim() });
  $("button", el).onclick = async () => {
    const p = await pickPath(kind, opts);
    if (p) { input.value = p; await App.save({ [key]: p }); if (opts.onPick) opts.onPick(p); }
  };
  return el;
}

// 生存確認（ブラウザを閉じるとサーバーが自動で止まる）
setInterval(() => fetch("/api/ping").catch(() => {}), 10000);
