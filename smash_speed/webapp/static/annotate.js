// 研究ツール：正解ラベルを付ける（1 コマずつシャトルの位置をクリック）
"use strict";

App.views.annotate = {
  title: "正解ラベルを付ける",
  crumb: () => App.meta ? App.meta.name : "",
  async render(el) {
    if (!requireVideo(el)) return;
    const st = App.state;
    if (st.start == null || st.end == null) {
      el.appendChild(h(`<div class="card"><h2>先に区間を選んでください</h2><p class="desc">ラベルを付ける範囲は、手順 1 で選んだ区間です。</p><button class="btn primary" onclick="go('clip')">手順 1 へ</button></div>`));
      return;
    }
    const ts = App.meta.timestamps;
    const i0 = nearestFrame(+st.start), i1 = nearestFrame(+st.end);
    const shot = st.shot || `${App.meta.name.replace(/\.[^.]+$/, "")}_${(+st.start).toFixed(1)}s`;
    const dir = st.ann_dir && st.ann_dir_shot === shot ? st.ann_dir : `${st.out_root}/labels/${shot}`;
    const init = `${st.out_root}/${shot}/detections.csv`;
    const N = i1 - i0 + 1;
    const L = Array.from({ length: N }, (_, k) => ({ frame: i0 + k, x: null, y: null, visible: -1, event: "", source: "" }));
    const got = await api.get(`/api/labels?dir=${encodeURIComponent(dir)}&init=${encodeURIComponent(init)}`);
    for (const r of got.rows || []) { const k = r.frame - i0; if (k >= 0 && k < N) Object.assign(L[k], r); }
    let cur = 0, auto = true, dirty = false;

    const ws = h(`<div class="workspace"><div class="left"></div><div class="side stack"></div></div>`);
    el.appendChild(ws);
    const left = $(".left", ws), side = $(".side", ws);
    const suggestion = (k) => {
      if (L[k].visible === 1) return [L[k].x, L[k].y];
      let a = k - 1, b = k + 1;
      while (a >= 0 && !(L[a].visible === 1 && L[a].source === "manual")) { if (L[a].visible === 0) return null; a--; }
      while (b < N && !(L[b].visible === 1 && L[b].source === "manual")) { if (L[b].visible === 0) return null; b++; }
      if (a < 0 || b >= N) return null;
      const w = (ts[i0 + k] - ts[i0 + a]) / (ts[i0 + b] - ts[i0 + a]);
      return [L[a].x + w * (L[b].x - L[a].x), L[a].y + w * (L[b].y - L[a].y)];
    };
    const fv = new FrameView(left, {
      width: 1920, loupe: true,
      draw: (ctx, f) => {
        for (let k = Math.max(0, cur - 12); k < cur; k++) if (L[k].visible === 1) { const [x, y] = f.toScreen(L[k].x, L[k].y); ctx.fillStyle = "rgba(34,197,94,.8)"; ctx.beginPath(); ctx.arc(x, y, 3, 0, 7); ctx.fill(); }
        const r = L[cur];
        if (r.visible === 1) drawMarker(ctx, f, r.x, r.y, r.source === "manual" ? "#ef4444" : "#38bdf8", r.source === "manual" ? "" : "提案");
        else { const s = suggestion(cur); if (s) drawMarker(ctx, f, s[0], s[1], "#38bdf8", "補間の提案（Space で採用）", "ghost"); }
      },
      onClick: (u, v) => { Object.assign(L[cur], { x: +u.toFixed(1), y: +v.toFixed(1), visible: 1, source: "manual" }); dirty = true; if (auto) move(1); else refresh(); },
    });
    const strip = h(`<div class="player"><div class="row" style="justify-content:space-between"><b id="fi"></b><span class="muted small" id="cnt"></span></div><div class="labelstrip" id="strip"></div></div>`);
    left.appendChild(strip);
    const move = (d) => { cur = Math.max(0, Math.min(N - 1, cur + d)); refresh(); };
    const info = h(`<div class="card"><h2>操作</h2><div class="keys" style="margin-top:8px">
      <span>クリック</span><span>シャトルの中心を置く（自動で次のコマへ）</span>
      <span><span class="kbd">X</span></span><span>このコマではシャトルが見えない</span>
      <span><span class="kbd">Space</span></span><span>水色の提案位置を採用</span>
      <span><span class="kbd">←</span><span class="kbd">→</span> / <span class="kbd">A</span><span class="kbd">D</span></span><span>前 / 次のコマ（Shift で 10 コマ）</span>
      <span><span class="kbd">H</span> / <span class="kbd">G</span></span><span>打った瞬間 / 着地 のコマにする</span>
      <span><span class="kbd">I</span></span><span>手で付けた点の間を、まとめて補間</span>
      <span><span class="kbd">Del</span></span><span>このコマのラベルを消す</span>
      <span><span class="kbd">Ctrl</span>+<span class="kbd">S</span></span><span>保存</span></div>
      <label class="row small" style="margin-top:10px"><input type="checkbox" id="auto" checked> クリックしたら次のコマへ進む</label></div>`);
    side.appendChild(info);
    const act = h(`<div class="card"><div class="row"><button class="btn" data-a="x">見えない</button><button class="btn" data-a="hit">打った瞬間</button><button class="btn" data-a="land">着地</button><button class="btn" data-a="interp">まとめて補間</button></div>
      <div class="field" style="margin-top:10px"><label>保存先フォルダ</label><input class="input" id="dir" value="${esc(dir)}"></div>
      <button class="btn primary block" id="save" style="margin-top:10px">保存する</button>
      <p class="muted small" id="saved" style="margin:6px 0 0">${got.source ? "読み込み元：" + esc(got.source) : "最初から付けます"}</p></div>`);
    side.appendChild(act);
    const save = async () => {
      const d = $("#dir", act).value.trim();
      const r = await api.post("/api/labels", { dir: d, rows: L, start: +st.start, end: +st.end });
      if (!r.ok) return toast(r.error, "bad");
      dirty = false; await App.save({ ann_dir: d, ann_dir_shot: shot, low_labels: r.path });
      $("#saved", act).textContent = "保存しました：" + r.path; toast("保存しました", "ok");
    };
    const doAct = (a) => {
      const r = L[cur];
      if (a === "x") { Object.assign(r, { x: null, y: null, visible: 0, source: "manual" }); dirty = true; if (auto) return move(1); }
      if (a === "hit" || a === "land") { for (const q of L) if (q.event === a) q.event = ""; r.event = r.event === a ? "" : a; dirty = true; }
      if (a === "interp") { let n = 0; for (let k = 0; k < N; k++) if (L[k].visible === -1) { const s = suggestion(k); if (s) { Object.assign(L[k], { x: s[0], y: s[1], visible: 1, source: "interp" }); n++; } } toast(`${n} コマを補間しました`); dirty = true; }
      if (a === "del") { Object.assign(r, { x: null, y: null, visible: -1, source: "" }); dirty = true; }
      if (a === "accept") { const s = suggestion(cur); if (s) { Object.assign(r, { x: s[0], y: s[1], visible: 1, source: r.source === "init" ? "init" : "manual" }); dirty = true; if (auto) return move(1); } }
      refresh();
    };
    $$("[data-a]", act).forEach((b) => b.onclick = () => doAct(b.dataset.a));
    $("#save", act).onclick = save;
    $("#auto", info).onchange = (e) => auto = e.target.checked;
    const refresh = () => {
      fv.show(i0 + cur);
      const r = L[cur];
      const lab = { 1: r.source === "manual" ? "位置あり（手で付けた）" : r.source === "interp" ? "位置あり（補間）" : "位置あり（検出の結果）", 0: "見えない", "-1": "未ラベル" }[r.visible];
      $("#fi", strip).textContent = `${cur + 1} / ${N} コマ目　${ts[i0 + cur].toFixed(3)} s　${lab}${r.event ? `　【${r.event === "hit" ? "打った瞬間" : "着地"}】` : ""}`;
      $("#cnt", strip).textContent = `ラベル済み ${L.filter((q) => q.visible >= 0).length} / ${N}`;
      const sEl = $("#strip", strip); sEl.innerHTML = "";
      L.forEach((q, k) => {
        const c = q.event ? "#a855f7" : q.visible === 0 ? "#94a3b8" : q.visible === 1 ? (q.source === "manual" ? "#22c55e" : "#38bdf8") : "#eef0f3";
        const i = h(`<i style="background:${c};${k === cur ? "outline:2px solid #1a1d23;outline-offset:-2px" : ""}"></i>`);
        i.onclick = () => { cur = k; refresh(); };
        sEl.appendChild(i);
      });
      fv.setTip(r.visible === -1 ? "シャトルの中心をクリック（見えなければ X）" : "");
    };
    this.keyh = (e) => {
      if (!fv.el.isConnected || e.target.tagName === "INPUT") return;
      const k = e.key.toLowerCase();
      if (e.ctrlKey && k === "s") { e.preventDefault(); save(); return; }
      if (k === "arrowright" || k === "d") move(e.shiftKey ? 10 : 1);
      else if (k === "arrowleft" || k === "a") move(e.shiftKey ? -10 : -1);
      else if (k === "x") doAct("x"); else if (k === "h") doAct("hit"); else if (k === "g") doAct("land");
      else if (k === "i") doAct("interp"); else if (k === "delete") doAct("del"); else if (e.key === " ") doAct("accept");
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", this.keyh);
    this.beforeLeave = () => dirty;
    refresh();
  },
  leave() {
    if (this.keyh) window.removeEventListener("keydown", this.keyh);
    if (this.beforeLeave && this.beforeLeave()) toast("保存していないラベルがあります（研究ツール → ラベル付けで続きを開けます）", "bad");
  },
};
