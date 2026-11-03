// 関節・打点を付ける（フォームの答え合わせ用の正解づくり。ROADMAP E1・E1b）
// 動画は「動画のフォルダ」の中から選ぶ（いま開いている動画とは別に読む）。付けた結果は出力の場所の pose_labels/ に保存する。
"use strict";

const PL_JOINTS = [["nose", "鼻"], ["left_eye", "左目"], ["right_eye", "右目"], ["left_ear", "左耳"], ["right_ear", "右耳"],
  ["left_shoulder", "左肩"], ["right_shoulder", "右肩"], ["left_elbow", "左肘"], ["right_elbow", "右肘"],
  ["left_wrist", "左手首"], ["right_wrist", "右手首"], ["left_hip", "左腰"], ["right_hip", "右腰"],
  ["left_knee", "左膝"], ["right_knee", "右膝"], ["left_ankle", "左足首"], ["right_ankle", "右足首"]];
const PL_INDEX = Object.fromEntries(PL_JOINTS.map(([k], i) => [k, i]));
const PL_BONES = [["left_shoulder", "right_shoulder"], ["left_hip", "right_hip"], ["left_shoulder", "left_hip"],
  ["right_shoulder", "right_hip"], ["left_shoulder", "left_elbow"], ["left_elbow", "left_wrist"],
  ["right_shoulder", "right_elbow"], ["right_elbow", "right_wrist"], ["left_hip", "left_knee"],
  ["left_knee", "left_ankle"], ["right_hip", "right_knee"], ["right_knee", "right_ankle"],
  ["nose", "left_eye"], ["nose", "right_eye"], ["left_eye", "left_ear"], ["right_eye", "right_ear"]].map(([a, b]) => [PL_INDEX[a], PL_INDEX[b]]);
const PL_FLIP = PL_JOINTS.map(([k]) => PL_INDEX[k.replace("left_", "@").replace("right_", "left_").replace("@", "right_")]);
const plColor = (j) => PL_JOINTS[j][0].startsWith("left_") ? "#3b82f6" : PL_JOINTS[j][0].startsWith("right_") ? "#f97316" : "#a855f7";
const PL_PREFILL = { none: "使わない", form: "フォームの解析の結果", mediapipe: "軽いモデル（CPU）", rtmpose: "標準のモデル（GPU）" };

(() => {
  const css = document.createElement("style");
  css.textContent = `
  .pl-joints { display: grid; grid-template-columns: 1fr 1fr; gap: 2px 8px; font-size: 13px; }
  .pl-joints div { display: flex; align-items: center; gap: 6px; padding: 3px 6px; border-radius: 6px; cursor: pointer; }
  .pl-joints div:hover { background: #f1f3f6; }
  .pl-joints div.cur { background: #dbeafe; font-weight: 600; }
  .pl-joints i { width: 10px; height: 10px; border-radius: 50%; display: inline-block; flex: none; }
  .pl-joints .st { margin-left: auto; font-size: 11px; color: #6b7280; }
  .pl-bar { height: 8px; border-radius: 4px; background: #eef0f3; overflow: hidden; margin-top: 6px; }
  .pl-bar i { display: block; height: 100%; background: var(--primary); }
  .pl-chips { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px; }
  .pl-chips button { border: 1px solid var(--line-strong); background: #fff; border-radius: 6px; padding: 2px 7px; font-size: 12px; cursor: pointer; }
  .pl-chips button.done { background: #dcfce7; border-color: #86efac; }
  .pl-chips button.cur { outline: 2px solid var(--primary); }
  .pl-swing { padding: 6px 0; border-top: 1px solid var(--line); }
  .pl-swing:first-child { border-top: 0; }
  .pl-nav { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; margin-top: 8px; }
  .pl-nav input[type=range] { flex: 1; min-width: 160px; }`;
  document.head.appendChild(css);
})();

/** 動画のフォルダの動画を読むフレーム表示（FrameView の読み先だけを変えたもの）。 */
class LabelFrameView extends FrameView {
  setVideo(rel, meta) { this.rel = rel; this.vmeta = meta; this.img = null; this.index = -1; this.want = -1; this.fit(); }
  get W() { return this.vmeta ? this.vmeta.width : 1; }
  get H() { return this.vmeta ? this.vmeta.height : 1; }
  show(i) {
    if (!this.vmeta) return;
    this.want = Math.max(0, Math.min(this.vmeta.n_frames - 1, Math.round(i)));
    if (!this.loading) this._load();
  }
  _load() {
    const i = this.want, rel = this.rel;
    this.loading = true;
    const img = new Image();
    img.onload = () => {
      this.loading = false;
      if (rel !== this.rel) { if (this.want >= 0) this._load(); return; }
      this.img = img; this.index = i;
      this.redraw();
      if (this.opt.onFrame) this.opt.onFrame(i);
      if (this.want !== i) this._load();
    };
    img.onerror = () => { this.loading = false; };
    img.src = `/api/poselab/frame?video=${encodeURIComponent(rel)}&i=${i}&w=1920`;
  }
}

// 2 つの画面で共通：動画のフォルダ・動画の選択・進み具合
const PoseLab = {
  rel: null, meta: null, swings: [], done: {}, impacts: [], list: null, hasForm: false,

  async header(el, mode) {
    this.list = await api.get("/api/poselab/videos");
    const L = this.list;
    const pr = L.progress || { frames: 0, usable: 0, target: 200 };
    const from = { setting: "", agent: "（開発エージェントの設定の動画フォルダ）", video: "（開いている動画のフォルダ）" }[L.root_from] || "";
    const card = h(`<div class="card"><div class="row" style="flex-wrap:wrap;gap:10px;align-items:center">
        <div class="seg" id="mode"><button data-v="joints">関節を付ける</button><button data-v="impact">打った瞬間のコマを付ける</button></div>
        <select class="input" id="vid" style="height:32px;min-width:220px;flex:1"></select>
        <span class="small muted">${mode === "joints"
          ? `付けたコマ <b id="pl-n">${pr.frames}</b> / 目安 ${pr.target}（肩と腰がそろったコマ <span id="pl-u">${pr.usable}</span>）`
          : `付けた打点 <b id="pl-imp">${L.impacts || 0}</b> 本`}</span></div>
      ${mode === "joints" ? `<div class="pl-bar"><i id="pl-bar" style="width:${Math.min(100, 100 * pr.frames / pr.target)}%"></i></div>` : ""}
      <details style="margin-top:8px"><summary class="small muted">動画のフォルダ：${esc(L.root || "未設定")} ${from}・保存先：${esc(L.labels_dir)}</summary>
        <div id="folder" style="margin-top:6px"></div>
        <p class="small muted" style="margin:4px 0 0">ここで選んだフォルダ（と、その 1 つ下のフォルダ）の動画だけを読みます。動画はこの PC から出ません。</p></details></div>`);
    $$("#mode button", card).forEach((b) => {
      b.classList.toggle("on", b.dataset.v === mode);
      b.onclick = () => { if (b.dataset.v !== mode) { App.views.posejoints.mode = b.dataset.v; go("posejoints"); } };
    });
    const f = pathField("動画のフォルダ", "label_videos_dir", "dir", { onPick: () => go("posejoints") });
    $("input", f).addEventListener("change", () => setTimeout(() => go("posejoints"), 50));
    $("#folder", card).appendChild(f);
    const sel = $("#vid", card);
    sel.innerHTML = `<option value="">— 動画を選ぶ（${L.videos.length} 本）—</option>` + L.videos.map((v) =>
      `<option value="${esc(v.rel)}">${esc(v.rel)}${v.frames ? `　関節 ${v.frames} コマ` : ""}${v.impacts ? `　打点 ${v.impacts}` : ""}</option>`).join("");
    const want = this.rel || App.state.label_video;
    if (want && L.videos.some((v) => v.rel === want)) sel.value = want;
    sel.onchange = async () => { this.rel = sel.value || null; await App.save({ label_video: sel.value }); go("posejoints"); };
    el.appendChild(card);
    await this.open(sel.value);  // 毎回読み直す（もう一方の画面で付けた打点・コマを反映する）
    return !!this.meta;
  },

  setProgress(pr) {
    if (!pr) return;
    const n = $("#pl-n"), u = $("#pl-u"), b = $("#pl-bar");
    if (n) n.textContent = pr.frames;
    if (u) u.textContent = pr.usable;
    if (b) b.style.width = Math.min(100, 100 * pr.frames / pr.target) + "%";
  },

  async open(rel) {
    if (!rel) { this.rel = null; this.meta = null; return; }
    const r = await api.get(`/api/poselab/video?video=${encodeURIComponent(rel)}`);
    if (!r.ok) { toast(r.error, "bad"); this.rel = null; this.meta = null; return; }
    Object.assign(this, { rel, meta: r.meta, swings: r.swings, done: r.done, impacts: r.impacts, hasForm: r.has_form });
    if (App.state.label_video !== rel) await App.save({ label_video: rel });
  },

  async findOnsets() {
    toast("打った音を探しています…");
    const r = await api.post("/api/poselab/onsets", { video: this.rel });
    if (!r.ok) return toast(r.error, "bad");
    if (!r.has_audio) toast("この動画には音がありません", "bad");
    else toast(`打った音を ${r.n} 個見つけました`, "ok");
    this.swings = r.swings;
  },

  nearest(t) {
    const ts = this.meta.timestamps;
    let lo = 0, hi = ts.length - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (ts[m] < t) lo = m; else hi = m; }
    return Math.abs(ts[lo] - t) <= Math.abs(ts[hi] - t) ? lo : hi;
  },

  /** コマの移動（ボタン・つまみ・時刻の表示）。 */
  navBar(host, onSeek) {
    const n = this.meta.n_frames;
    const bar = h(`<div class="pl-nav">
      <button class="btn sm" data-d="-10">« 10</button><button class="btn sm" data-d="-1">‹ 1コマ</button>
      <input type="range" min="0" max="${n - 1}" value="0">
      <button class="btn sm" data-d="1">1コマ ›</button><button class="btn sm" data-d="10">10 »</button>
      <span class="small mono" id="pos" style="min-width:150px;text-align:right"></span></div>`);
    host.appendChild(bar);
    const range = $("input", bar);
    let cur = 0;
    const api_ = {
      get index() { return cur; },
      set(i) {
        cur = Math.max(0, Math.min(n - 1, i));
        range.value = cur;
        $("#pos", bar).textContent = `${PoseLab.meta.timestamps[cur].toFixed(3)} 秒（${cur} コマ目）`;
      },
    };
    $$("[data-d]", bar).forEach((b) => { b.onclick = () => onSeek(cur + +b.dataset.d); });
    range.oninput = () => onSeek(+range.value);
    return api_;
  },
};

// 左のメニューは 1 つ（「関節・打点を付ける」）で、上の切り替えで 2 つの画面を行き来する
App.views.posejoints = {
  mode: "joints",
  get title() { return this.mode === "impact" ? "打った瞬間のコマを付ける" : "関節を付ける"; },
  crumb: () => PoseLab.meta ? PoseLab.meta.name : "",
  view() { return this.mode === "impact" ? PoseImpactView : PoseJointsView; },
  render(el) { this.shown = this.view(); return this.shown.render(el); },
  leave() { if (this.shown) this.shown.leave(); },
};

// ---------------- 関節を付ける ----------------
const PoseJointsView = {

  async render(el) {
    if (!(await PoseLab.header(el, "joints"))) {
      el.appendChild(h(`<div class="card"><h2>体の関節の正しい位置を、手で付けます</h2>
        <p class="desc">姿勢を見つける 2 つのやり方（標準・軽い）のどちらが正確かを、手で付けた位置と比べて決めます。
          横から撮ったスイングの動画を上で選んでください。1 本のスイングから 10 コマ、20 本ほどで約 200 コマが目安です。</p>
        <p class="small muted">動画が出てこないときは、上の「動画のフォルダ」を開いてフォルダを選んでください。</p></div>`));
      return;
    }
    const M = PoseLab.meta;
    let frame = -1, J = Array(17).fill(null), cur = 0, undo = [], dirty = false, prefill = "", people = [], personIdx = 0, nextHidden = false;
    let dragging = null;
    const model = () => App.state.label_prefill || (PoseLab.hasForm ? "form" : "none");

    const ws = h(`<div class="workspace"><div class="left"></div><div class="side stack"></div></div>`);
    el.appendChild(ws);
    const left = $(".left", ws), side = $(".side", ws);

    const fv = new LabelFrameView(left, {
      width: 1920, loupe: true,
      draw: (ctx, f) => {
        for (const [a, b] of PL_BONES) {
          if (J[a] && J[a].v && J[b] && J[b].v) {
            drawPolyline(ctx, f, [[J[a].x, J[a].y], [J[b].x, J[b].y]], "rgba(255,255,255,.75)", 2, (J[a].v === 1 || J[b].v === 1) ? [5, 4] : null);
          }
        }
        J.forEach((p, j) => {
          if (!p || !p.v) return;
          const [x, y] = f.toScreen(p.x, p.y);
          ctx.globalAlpha = p.draft ? 0.55 : 1;
          ctx.fillStyle = plColor(j); ctx.strokeStyle = "#fff"; ctx.lineWidth = 2;
          ctx.beginPath(); ctx.arc(x, y, j === cur ? 7 : 5, 0, Math.PI * 2);
          if (p.v === 2) { ctx.fill(); ctx.stroke(); } else { ctx.setLineDash([3, 3]); ctx.strokeStyle = plColor(j); ctx.stroke(); ctx.setLineDash([]); }
          ctx.globalAlpha = 1;
          if (j === cur) { ctx.strokeStyle = "#facc15"; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x, y, 12, 0, Math.PI * 2); ctx.stroke(); }
        });
      },
      drawLoupe: (lc, u, v, mag, L) => {
        J.forEach((p, j) => {
          if (!p || !p.v) return;
          const x = L / 2 + (p.x - u) * mag, y = L / 2 + (p.y - v) * mag;
          if (x < 0 || y < 0 || x > L || y > L) return;
          lc.fillStyle = plColor(j); lc.beginPath(); lc.arc(x, y, 5, 0, Math.PI * 2); lc.fill();
        });
      },
      onClick: (u, v) => place(u, v),
    });
    function place(u, v) {
      snapshot();
      J[cur] = { x: +u.toFixed(1), y: +v.toFixed(1), v: nextHidden ? 1 : 2 };
      nextHidden = false; dirty = true;
      advance();
    }
    fv.setVideo(PoseLab.rel, M);
    const nav = PoseLab.navBar(left, (i) => gotoFrame(i));

    // 置いた点をつかんで動かす（FrameView のクリックより先に受け取る）。動かさずに離したら、ふつうのクリック（今の関節を置く）
    let press = null;
    const at = (e) => { const r = fv.canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    fv.el.addEventListener("mousedown", (e) => {
      if (e.button !== 0 || fv.spaceDown || e.target !== fv.canvas) return;
      const [mx, my] = at(e);
      let best = -1, bd = 12;
      J.forEach((p, j) => {
        if (!p || !p.v) return;
        const [x, y] = fv.toScreen(p.x, p.y);
        const d = Math.hypot(x - mx, y - my);
        if (d < bd) { bd = d; best = j; }
      });
      if (best < 0) return;
      e.stopPropagation(); e.preventDefault();
      press = { j: best, x: mx, y: my };
    }, true);
    const onMove = (e) => {
      if (!press || !fv.el.isConnected) return;
      const [mx, my] = at(e);
      if (dragging === null) {
        if (Math.hypot(mx - press.x, my - press.y) < 4) return;
        snapshot();
        dragging = press.j; cur = press.j;
        refresh();
      }
      const [u, v] = fv.toImage(mx, my);
      Object.assign(J[dragging], { x: +Math.max(0, Math.min(M.width, u)).toFixed(1), y: +Math.max(0, Math.min(M.height, v)).toFixed(1), draft: false });
      dirty = true;
      fv.redraw();
    };
    const onUp = (e) => {
      if (!press) return;
      if (dragging === null && fv.el.isConnected) {
        // 今の関節がまだなら置く（顔の点のように近い所に続けて付けるとき）。もう付いていれば、押した点を選んで確かめ済みにする
        if (!J[cur] || !J[cur].v) place(...fv.toImage(...at(e)));
        else { cur = press.j; J[cur].draft = false; dirty = true; advance(); }
      }
      press = null; dragging = null;
      refresh();
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);

    // ---- 横の欄
    const jc = h(`<div class="card"><div class="row" style="justify-content:space-between"><h2 style="margin:0">関節</h2>
        <span class="small muted" id="fr"></span></div>
      <p class="small muted" style="margin:4px 0 8px"><span style="color:#3b82f6">●</span> 左　<span style="color:#f97316">●</span> 右（打つ人から見た左右）。
        薄い点は下書き、点線は隠れている所（位置は推測）。</p>
      <div class="pl-joints" id="jl"></div>
      <div class="row" style="flex-wrap:wrap;gap:6px;margin-top:10px">
        <button class="btn sm" id="hid">隠れている</button><button class="btn sm" id="skip">写っていない</button>
        <button class="btn sm" id="flip">左右を入れ替え</button><button class="btn sm" id="und">元に戻す</button></div>
      <div class="row" style="flex-wrap:wrap;gap:6px;margin-top:10px">
        <button class="btn primary" id="save">保存して次のコマへ</button><button class="btn sm danger" id="del">このコマを消す</button></div>
      <div class="field" style="margin-top:10px"><label>下書き（関節の位置の提案）</label>
        <div class="row" style="gap:6px"><select class="input" id="pre" style="flex:1"></select><button class="btn sm" id="redo">出し直す</button>
        <button class="btn sm" id="other" title="別の人の関節にする">ほかの人</button></div>
        <p class="small muted" style="margin:4px 0 0" id="pre-note"></p></div></div>`);
    side.appendChild(jc);
    const preSel = $("#pre", jc);
    preSel.innerHTML = Object.entries(PL_PREFILL).filter(([k]) => k === "none" || k === "form" || (PoseLab.list.models || {})[k])
      .map(([k, v]) => `<option value="${k}">${v}</option>`).join("");
    preSel.value = [...preSel.options].some((o) => o.value === model()) ? model() : "none";
    $("#pre-note", jc).textContent = "下書きは直すための出発点です。下書きのモデルに甘く（PCK が高めに）出やすいので、比べるときは下書きのモデルを替えたコマも混ぜてください。";
    preSel.onchange = async () => { await App.save({ label_prefill: preSel.value }); if (!J.some((p) => p && !p.draft && p.v)) loadPrefill(); };
    $("#redo", jc).onclick = () => { snapshot(); loadPrefill(true); };
    $("#other", jc).onclick = () => { if (people.length > 1) { snapshot(); personIdx = (personIdx + 1) % people.length; usePerson(); } else toast("ほかの人は見つかっていません"); };
    $("#hid", jc).onclick = () => toggleHidden();
    $("#skip", jc).onclick = () => skip();
    $("#flip", jc).onclick = () => flip();
    $("#und", jc).onclick = () => doUndo();
    $("#save", jc).onclick = () => save(true);
    $("#del", jc).onclick = async () => {
      if (!PoseLab.done[frame]) { J = Array(17).fill(null); dirty = false; refresh(); return; }
      if (!confirm("このコマの関節を消しますか？")) return;
      const r = await api.post("/api/poselab/joints/delete", { video: PoseLab.rel, frame });
      if (!r.ok) return toast(r.error, "bad");
      delete PoseLab.done[frame]; PoseLab.setProgress(r.progress);
      J = Array(17).fill(null); dirty = false; cur = 0; refresh(); renderPlan();
    };

    const plan = h(`<div class="card"><div class="row" style="justify-content:space-between"><h2 style="margin:0">付けるコマ</h2>
      <button class="btn sm" id="ons">打った音から探す</button></div>
      <p class="small muted" style="margin:4px 0 6px">スイングごとに 10 コマ（構え〜戻り）。緑は付け終わったコマです。どのコマでも、そのまま付けて保存できます。</p>
      <div id="sw" style="max-height:34vh;overflow:auto"></div></div>`);
    side.appendChild(plan);
    $("#ons", plan).onclick = async () => { await PoseLab.findOnsets(); renderPlan(); };

    side.appendChild(h(`<div class="card"><h2>操作</h2><div class="keys" style="margin-top:8px">
      <span>クリック</span><span>黄色い丸の関節を置く（次の関節へ進む）</span>
      <span>ドラッグ</span><span>置いた点を動かす（点をクリックすると、その点を選んで「確かめた」にする）</span>
      <span><span class="kbd">H</span></span><span>隠れている（位置は推測）に切り替え</span>
      <span><span class="kbd">X</span></span><span>写っていない（付けない）</span>
      <span><span class="kbd">Tab</span></span><span>次の関節（Shift で前）</span>
      <span><span class="kbd">F</span></span><span>左右を入れ替える</span>
      <span><span class="kbd">Ctrl</span>+<span class="kbd">Z</span></span><span>元に戻す</span>
      <span><span class="kbd">Enter</span></span><span>保存して次のコマへ</span>
      <span><span class="kbd">←</span><span class="kbd">→</span></span><span>前 / 次のコマ（Shift で 10 コマ。付けた所は自動で保存）</span>
      <span><span class="kbd">A</span> / <span class="kbd">D</span></span><span>前 / 次の「付けるコマ」</span>
      <span>ホイール・右ドラッグ</span><span>拡大・移動</span></div></div>`));

    // ---- 中身
    const planList = () => PoseLab.swings.flatMap((s) => s.plan.map((p) => p.frame));
    function renderPlan() {
      const box = $("#sw", plan);
      box.innerHTML = PoseLab.swings.length ? "" : `<p class="small muted">スイングの候補がありません。「打った音から探す」か、
        「打った瞬間のコマを付ける」で打点を付けると出てきます。つまみで動かして、好きなコマに付けることもできます。</p>`;
      for (const s of PoseLab.swings) {
        const n = s.plan.filter((p) => PoseLab.done[p.frame]).length;
        const row = h(`<div class="pl-swing"><div class="small"><b>${s.t.toFixed(2)} 秒</b>　<span class="muted">${esc(s.source)}・${n}/${s.plan.length}</span></div>
          <div class="pl-chips"></div></div>`);
        for (const p of s.plan) {
          const b = h(`<button class="${PoseLab.done[p.frame] ? "done" : ""} ${p.frame === frame ? "cur" : ""}" title="${p.frame} コマ目（${p.t.toFixed(3)} 秒）">${esc(p.phase)}</button>`);
          b.onclick = () => gotoFrame(p.frame);
          $(".pl-chips", row).appendChild(b);
        }
        box.appendChild(row);
      }
    }
    function snapshot() { undo.push(JSON.stringify(J)); if (undo.length > 100) undo.shift(); }
    function doUndo() { if (!undo.length) return; J = JSON.parse(undo.pop()); dirty = true; refresh(); }
    function advance() {
      for (let k = cur + 1; k < 17; k++) if (!J[k] || J[k].draft) { cur = k; refresh(); return; }
      cur = Math.min(16, cur + 1);
      refresh();
    }
    function toggleHidden() {
      const p = J[cur];
      if (p && p.v) { snapshot(); p.v = p.v === 2 ? 1 : 2; p.draft = false; dirty = true; }
      else nextHidden = !nextHidden;
      refresh();
    }
    function skip() { snapshot(); J[cur] = { x: null, y: null, v: 0 }; dirty = true; nextHidden = false; advance(); }
    function flip() { snapshot(); J = PL_FLIP.map((k) => J[k]); dirty = true; refresh(); }
    function refresh() {
      const jl = $("#jl", jc);
      jl.innerHTML = "";
      PL_JOINTS.forEach(([, name], j) => {
        const p = J[j];
        const st = !p ? "" : p.v === 0 ? "写っていない" : p.draft ? "下書き" : p.v === 1 ? "隠れている" : "✓";
        const d = h(`<div class="${j === cur ? "cur" : ""}"><i style="background:${plColor(j)}"></i>${name}<span class="st">${st}</span></div>`);
        d.onclick = () => { cur = j; refresh(); };
        jl.appendChild(d);
      });
      const placed = J.filter((p) => p && p.v).length, drafts = J.filter((p) => p && p.v && p.draft).length;
      $("#fr", jc).textContent = frame >= 0 ? `${frame} コマ目・${placed} 点${drafts ? `（下書きのまま ${drafts}）` : ""}${PoseLab.done[frame] ? "・保存済み" : ""}` : "";
      $("#hid", jc).classList.toggle("primary", nextHidden || !!(J[cur] && J[cur].v === 1));
      fv.setTip(`${nextHidden ? "隠れている " : ""}${PL_JOINTS[cur][1]} をクリック`);
      fv.setHud([`${PoseLab.meta.timestamps[Math.max(0, frame)].toFixed(3)} 秒`, prefill ? `下書き：${esc(PL_PREFILL[prefill] || prefill)}` : "手で付けている"]);
      fv.redraw();
    }
    function usePerson() {
      const P = people[personIdx];
      J = P ? P.xy.map(([x, y], j) => (x === null || P.score[j] < 0.3) ? null : { x, y, v: 2, draft: true }) : Array(17).fill(null);
      cur = Math.max(0, J.findIndex((p) => !p || p.draft));
      dirty = false;
      refresh();
    }
    async function loadPrefill(force = false) {
      const m = preSel.value;
      prefill = ""; people = []; personIdx = 0;
      if (m === "none") { if (force) usePerson(); return; }
      const want = frame;
      if (m !== "form") fv.setTip("下書きを作っています…（初めてのときはモデルを取ってくるので時間がかかります）");
      const r = await api.get(`/api/poselab/prefill?video=${encodeURIComponent(PoseLab.rel)}&frame=${want}&model=${m}`);
      if (want !== frame) return;
      if (!r.ok) { toast(r.error, "bad"); refresh(); return; }
      people = r.people || [];
      if (!people.length) { if (force) toast(m === "form" ? "このコマはフォームの解析の結果にありません" : "人が見つかりませんでした"); refresh(); return; }
      prefill = m;
      usePerson();
    }
    async function save(next = false) {
      if (frame < 0) return false;
      if (!J.some((p) => p && p.v)) { if (next) toast("関節が 1 つも付いていません", "bad"); return false; }
      const joints = J.map((p) => (p && p.v ? [p.x, p.y, p.v] : null));
      const r = await api.post("/api/poselab/joints", { video: PoseLab.rel, frame, joints, prefill });
      if (!r.ok) { toast(r.error, "bad"); return false; }
      PoseLab.done[frame] = { n: joints.filter(Boolean).length, prefill };
      PoseLab.setProgress(r.progress);
      dirty = false;
      if (next) {
        toast(`保存しました（${r.progress.frames} / ${r.progress.target} コマ）`, "ok");
        const pl = planList();
        const k = pl.findIndex((f) => f > frame && !PoseLab.done[f]);
        const nf = k >= 0 ? pl[k] : pl.find((f) => !PoseLab.done[f]);
        await gotoFrame(nf !== undefined ? nf : frame + 1);
      } else renderPlan();
      return true;
    }
    async function gotoFrame(i) {
      i = Math.max(0, Math.min(M.n_frames - 1, i));
      if (dirty && frame >= 0 && i !== frame) await save(false);
      frame = i; nav.set(i); fv.show(i);
      undo = []; dirty = false; nextHidden = false; prefill = ""; people = []; personIdx = 0;
      const r = await api.get(`/api/poselab/joints?video=${encodeURIComponent(PoseLab.rel)}&frame=${i}`);
      if (frame !== i) return;
      if (r.ok && r.joints) {
        J = r.joints.map((p) => (p ? { x: p[0], y: p[1], v: p[2] } : null));
        prefill = r.prefill || "";
        cur = Math.max(0, J.findIndex((p) => !p));
        refresh();
      } else {
        J = Array(17).fill(null); cur = 0;
        refresh();
        await loadPrefill();
      }
      renderPlan();
    }

    this.keyh = (e) => {
      if (!fv.el.isConnected || e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
      const k = e.key.toLowerCase();
      if ((e.ctrlKey || e.metaKey) && k === "z") { doUndo(); e.preventDefault(); return; }
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (k === "h") toggleHidden();
      else if (k === "x") skip();
      else if (k === "f") flip();
      else if (e.key === "Tab") { cur = (cur + (e.shiftKey ? 16 : 1)) % 17; refresh(); }
      else if (e.key === "Enter") save(true);
      else if (e.key === "Delete" || e.key === "Backspace") { snapshot(); J[cur] = null; dirty = true; refresh(); }
      else if (e.key === "ArrowRight") gotoFrame(frame + (e.shiftKey ? 10 : 1));
      else if (e.key === "ArrowLeft") gotoFrame(frame - (e.shiftKey ? 10 : 1));
      else if (k === "d" || k === "a") {
        const pl = planList();
        const nf = k === "d" ? pl.find((f) => f > frame) : [...pl].reverse().find((f) => f < frame);
        if (nf !== undefined) gotoFrame(nf);
      } else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", this.keyh);
    this.cleanup = () => { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); if (dirty) save(false); };

    const pl = planList();
    await gotoFrame(pl.find((f) => !PoseLab.done[f]) ?? pl[0] ?? 0);
  },

  leave() {
    if (this.keyh) window.removeEventListener("keydown", this.keyh);
    if (this.cleanup) this.cleanup();
    this.keyh = this.cleanup = null;
  },
};

// ---------------- 打った瞬間のコマを付ける ----------------
const PoseImpactView = {

  async render(el) {
    if (!(await PoseLab.header(el, "impact"))) {
      el.appendChild(h(`<div class="card"><h2>シャトルを打った瞬間のコマを、手で付けます</h2>
        <p class="desc">フォームの解析が「打った瞬間」をどれだけ正しく見つけているかを確かめるのに使います。
          上で動画を選び、スイングの前後を 1 コマずつ見て、ラケットがシャトルに当たったコマで「ここで打った」を押してください。</p></div>`));
      return;
    }
    const M = PoseLab.meta;
    let frame = 0, playing = false;
    const ws = h(`<div class="workspace"><div class="left"></div><div class="side stack"></div></div>`);
    el.appendChild(ws);
    const left = $(".left", ws), side = $(".side", ws);
    const fv = new LabelFrameView(left, { width: 1920, loupe: false });
    fv.setVideo(PoseLab.rel, M);
    const nav = PoseLab.navBar(left, (i) => seek(i));
    const bar = h(`<div class="row" style="gap:8px;margin-top:8px;flex-wrap:wrap">
      <button class="btn primary" id="mark">ここで打った（Enter）</button>
      <button class="btn sm" id="play">▶ ゆっくり再生</button>
      <span class="small muted" id="near"></span></div>`);
    left.appendChild(bar);

    const sw = h(`<div class="card"><div class="row" style="justify-content:space-between"><h2 style="margin:0">スイング</h2>
      <button class="btn sm" id="ons">打った音から探す</button></div>
      <p class="small muted" style="margin:4px 0 6px">押すと、そのスイングの少し前に移ります。候補がなくても、つまみで動かして付けられます。</p>
      <div class="pointlist" id="list" style="max-height:44vh;overflow:auto"></div></div>`);
    side.appendChild(sw);
    side.appendChild(h(`<div class="card"><h2>操作</h2><div class="keys" style="margin-top:8px">
      <span><span class="kbd">←</span><span class="kbd">→</span></span><span>前 / 次のコマ（Shift で 10 コマ）</span>
      <span><span class="kbd">Enter</span></span><span>このコマで打った</span>
      <span><span class="kbd">↑</span><span class="kbd">↓</span></span><span>前 / 次のスイング</span>
      <span><span class="kbd">P</span></span><span>ゆっくり再生（1 秒に 8 コマ）</span>
      <span>ホイール・右ドラッグ</span><span>拡大・移動（ラケットの所を大きくすると見やすい）</span></div>
      <p class="small muted" style="margin:8px 0 0">当たったコマがはっきりしないときは、シャトルの向きが変わる直前のコマにします。
        同じスイングの近く（0.3 秒以内）で付け直すと置き換わります。</p></div>`));
    $("#ons", sw).onclick = async () => { await PoseLab.findOnsets(); renderList(); };

    const marked = (s) => PoseLab.impacts.find((r) => Math.abs(r.t_s - s.t) <= 0.3);
    function renderList() {
      const list = $("#list", sw);
      list.innerHTML = PoseLab.swings.length ? "" : `<p class="small muted">候補がありません。「打った音から探す」を押すか、つまみで動かしてください。</p>`;
      for (const s of PoseLab.swings) {
        const m = marked(s);
        const row = h(`<div class="p"><span>${m ? "✓" : "・"}</span><span style="flex:1">${s.t.toFixed(2)} 秒
          <span class="muted small">${esc(m ? `打点 ${m.frame} コマ目` : s.source)}</span></span>
          ${m ? `<button class="btn sm ghost" title="この打点を消す">消す</button>` : ""}</div>`);
        row.onclick = (e) => { if (e.target.tagName !== "BUTTON") seek(PoseLab.nearest(s.t - 0.1)); };
        if (m) $("button", row).onclick = async () => {
          const r = await api.post("/api/poselab/impacts/delete", { video: PoseLab.rel, frame: m.frame });
          if (!r.ok) return toast(r.error, "bad");
          setImpacts(r); renderList(); seek(frame);
        };
        list.appendChild(row);
      }
    }
    function seek(i) {
      frame = Math.max(0, Math.min(M.n_frames - 1, i));
      nav.set(frame); fv.show(frame);
      const t = M.timestamps[frame];
      const m = PoseLab.impacts.find((r) => Math.abs(r.t_s - t) <= 0.3);
      const d = m ? frame - m.frame : null;
      $("#near", bar).textContent = m ? (d === 0 ? "このコマが付けた打点です" : `付けた打点（${m.frame} コマ目）から ${d > 0 ? "+" : ""}${d} コマ`) : "";
      fv.setHud([`${t.toFixed(3)} 秒`, `${frame} コマ目`]);
      fv.setTip(m && d === 0 ? "✓ 打った瞬間" : "");
    }
    async function mark() {
      const r = await api.post("/api/poselab/impacts", { video: PoseLab.rel, frame });
      if (!r.ok) return toast(r.error, "bad");
      setImpacts(r);
      toast(`${frame} コマ目を打った瞬間にしました`, "ok");
      renderList(); seek(frame);
    }
    function setImpacts(r) {
      const n = $("#pl-imp");
      if (n) n.textContent = +n.textContent + r.impacts.length - PoseLab.impacts.length;
      PoseLab.impacts = r.impacts; PoseLab.swings = r.swings;
    }
    function moveSwing(dir) {
      const t = M.timestamps[frame];
      const s = dir > 0 ? PoseLab.swings.find((s) => s.t > t + 0.35) : [...PoseLab.swings].reverse().find((s) => s.t < t - 0.35);
      if (s) seek(PoseLab.nearest(s.t - 0.1));
    }
    function togglePlay() {
      playing = !playing;
      $("#play", bar).textContent = playing ? "■ 止める" : "▶ ゆっくり再生";
      const tick = () => {
        if (!playing || !fv.el.isConnected) return;
        if (frame >= M.n_frames - 1) { togglePlay(); return; }
        seek(frame + 1);
        setTimeout(tick, 125);
      };
      if (playing) tick();
    }
    $("#mark", bar).onclick = mark;
    $("#play", bar).onclick = togglePlay;

    this.keyh = (e) => {
      if (!fv.el.isConnected || e.target.tagName === "INPUT" || e.target.tagName === "SELECT" || e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.key === "ArrowRight") seek(frame + (e.shiftKey ? 10 : 1));
      else if (e.key === "ArrowLeft") seek(frame - (e.shiftKey ? 10 : 1));
      else if (e.key === "ArrowDown") moveSwing(1);
      else if (e.key === "ArrowUp") moveSwing(-1);
      else if (e.key === "Enter") mark();
      else if (e.key.toLowerCase() === "p") togglePlay();
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", this.keyh);
    renderList();
    const first = PoseLab.swings.find((s) => !marked(s)) || PoseLab.swings[0];
    seek(first ? PoseLab.nearest(first.t - 0.1) : 0);
  },

  leave() {
    if (this.keyh) window.removeEventListener("keydown", this.keyh);
    this.keyh = null;
  },
};
