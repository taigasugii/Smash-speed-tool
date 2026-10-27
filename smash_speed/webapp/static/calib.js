// 手順 2：コートを合わせる（キャリブレーションの案内つきウィザード）
"use strict";

// ずれの評価の目安は横幅 1920 px 基準。4K なら 2 倍の px まで同じ評価にする
const rmseScale = () => Math.max(1, ((App.meta && App.meta.width) || 1920) / 1920);

App.views.calib = {
  title: "手順 2　コートを合わせる",
  crumb: () => App.meta ? App.meta.name : "",

  async render(el, opts = {}) {
    if (!requireVideo(el)) return;
    if (!App.landmarks) App.landmarks = await api.get("/api/landmarks");
    this.el = el;
    const st = App.state;
    this.view = st.calib_view || (App.calib && App.calib.view) || null;
    this.points = { ...(st.draft_points || {}) };
    this.skipped = new Set(st.draft_skipped || []);
    this.auto = { ...(st.draft_auto || {}) };   // 自動で置いた点（名前 → "snapped"）
    this.autoSkipped = new Set();
    this.solved = null;
    this.extra = false;
    if (opts.step) return this.step(opts.step);
    if (App.calib && !App.calib.error && !opts.fresh) return this.summary();
    return this.step(this.view ? (Object.keys(this.points).length ? "points" : "points") : "view");
  },

  leave() { this.unbindKeys(); this.saveDraft(); },
  unbindKeys() { if (this.keyh) { window.removeEventListener("keydown", this.keyh); this.keyh = null; } },
  saveDraft() { if (this.points) App.save({ draft_points: this.points, draft_skipped: [...this.skipped], draft_auto: this.auto, calib_view: this.view }); },

  header(active) {
    const steps = [["view", "カメラの位置"], ["points", "点をクリック"], ["check", "確認して保存"]];
    const idx = steps.findIndex((s) => s[0] === active);
    const w = h(`<div class="wizard-steps"></div>`);
    steps.forEach(([k, l], i) => {
      const s = h(`<div class="ws ${i === idx ? "on" : i < idx ? "done" : ""}"><span class="n">${i < idx ? "✓" : i + 1}</span>${l}</div>`);
      s.onclick = () => { if (k === "view" || this.view) this.step(k); };
      w.appendChild(s);
    });
    return w;
  },

  step(name) {
    this.unbindKeys();
    this.el.innerHTML = "";
    if (name === "view") return this.viewStep();
    if (name === "points") return this.pointsStep();
    if (name === "check") return this.checkStep();
  },

  // ---------------- 保存済みのときの画面 ----------------
  summary() {
    const c = App.calib;
    const k = rmseScale();
    const bad = c.rmse_px >= 8 * k || c.n_points < 6;
    const q = bad ? ["bad", "要確認"] : c.rmse_px < 3 * k ? ["ok", "良好"] : ["warn", "まずまず"];
    const reasons = (c.warnings || []).map((w) => `<li>${esc(w)}</li>`).join("");
    this.el.appendChild(h(`<div class="card"><h2>${bad ? "保存済みのコート合わせに問題がありそうです" : "このカメラ位置のコート合わせは済んでいます"}</h2>
      <p class="desc">${bad ? "点が少ないか、クリックした位置とコートの形が合っていません。このまま解析すると速度が大きくずれるので、<b>最初からやり直す</b>のがおすすめです。" : "カメラを動かしていなければ、このまま使えます。"}</p>
      <div class="quality"><span class="big">${fmt(c.rmse_px, 1)} px</span><span class="badge ${q[0]}">${q[1]}</span><span class="muted">${c.n_points} 点を使用${(c.excluded || []).length ? `（${c.excluded.length} 点は合わないため除外）` : ""}・ファイル：${esc(App.state.calib)}</span></div>
      ${reasons ? `<ul class="muted small" style="margin:10px 0 0">${reasons}</ul>` : ""}
      <div class="row" style="margin-top:14px">
        <button class="btn ${bad ? "" : "primary"}" id="use">このまま使う（次へ）</button>
        <button class="btn" id="check">重ねて確認する</button>
        <button class="btn" id="edit">点を直す</button>
        <button class="btn ${bad ? "primary" : ""}" id="redo">最初からやり直す</button>
        <button class="btn ghost" id="load">別のファイルを読み込む…</button></div></div>`));
    $("#use", this.el).onclick = () => go("plane");
    $("#check", this.el).onclick = () => { this.points = { ...c.points }; this.step("check"); };
    $("#edit", this.el).onclick = () => { this.points = { ...c.points }; this.step(this.view ? "points" : "view"); };
    $("#redo", this.el).onclick = () => { this.points = {}; this.skipped = new Set(); this.auto = {}; this.step("view"); };
    $("#load", this.el).onclick = async () => {
      const p = await pickPath("file", { label: "キャリブレーション", pattern: "*.json" });
      if (!p) return;
      const r = await api.post("/api/calib/load", { path: p });
      if (!r.ok || !r.calib) return toast("読み込めませんでした", "bad");
      App.calib = r.calib; App.state.calib = p; toast("読み込みました", "ok"); go("calib");
    };
  },

  // ---------------- ① カメラの位置 ----------------
  viewStep() {
    this.el.appendChild(this.header("view"));
    const v = this.view || {};
    let edge = v.edge || null, hitter = v.hitter || null, camXY = v.camXY || null;
    const L = 6.7, Wd = 3.05, S = 34, M = 3.2;           // 図の縮尺 [px/m] と余白 [m]
    const Wpx = (2 * L + 2 * M) * S, Hpx = (2 * Wd + 2 * M) * S;
    const X = (x) => (x + L + M) * S, Y = (y) => (Wd + M - y) * S;
    const card = h(`<div class="grid2" style="grid-template-columns: 1.25fr 1fr">
      <div class="card"><h2>カメラと打つ人の位置を教えてください</h2>
        <p class="desc">上から見たコートの図です。<b>① カメラを置いた場所</b>をコートの外側でクリックし、<b>② スマッシュを打つ人がいる半面</b>をクリックしてください。</p>
        <div id="pick"></div>
        <div class="row" style="margin-top:8px" id="ans"></div></div>
      <div class="card"><h2>動画の様子（参考）</h2><p class="desc">この映像と見比べて選んでください。</p><img id="thumb" style="width:100%;border-radius:8px"></div></div>`);
    this.el.appendChild(card);
    $("#thumb", card).src = `/api/frame?i=${nearestFrame(App.state.start ?? 0)}&w=960`;
    const draw = () => {
      let s = `<svg class="court-pick" viewBox="0 0 ${Wpx} ${Hpx}" xmlns="http://www.w3.org/2000/svg">
        <rect class="zone" x="0" y="0" width="${Wpx}" height="${Hpx}" data-z="out" fill="#f8fafc"/>
        <rect x="${X(-L)}" y="${Y(Wd)}" width="${2 * L * S}" height="${2 * Wd * S}" fill="#e6f2ef" stroke="#0f766e" stroke-width="3"/>`;
      for (const [half, x0] of [["left", -L], ["right", 0]]) {
        s += `<rect class="half ${hitter === half ? "sel" : ""}" data-z="${half}" x="${X(x0)}" y="${Y(Wd)}" width="${L * S}" height="${2 * Wd * S}"/>`;
      }
      for (const seg of App.landmarks.segments) if (seg.kind === "line") s += `<line x1="${X(seg.a[0])}" y1="${Y(seg.a[1])}" x2="${X(seg.b[0])}" y2="${Y(seg.b[1])}" stroke="#0f766e" stroke-width="2" pointer-events="none"/>`;
      s += `<line x1="${X(0)}" y1="${Y(-Wd - 0.2)}" x2="${X(0)}" y2="${Y(Wd + 0.2)}" stroke="#db2777" stroke-width="5" pointer-events="none"/>`;
      if (hitter) s += `<text x="${X(hitter === "left" ? -L / 2 : L / 2)}" y="${Y(0) + 6}" text-anchor="middle" font-size="30" font-weight="700" fill="#c2410c" pointer-events="none">🏸 打つ人</text>`;
      else s += `<text x="${X(0)}" y="${Y(0) + 6}" text-anchor="middle" font-size="24" font-weight="700" fill="#2563eb" pointer-events="none">② 打つ人の半面をクリック</text>`;
      if (camXY) {
        const [cx, cy] = [X(camXY[0]), Y(camXY[1])];
        s += `<line x1="${cx}" y1="${cy}" x2="${X(0)}" y2="${Y(0)}" stroke="#1a1d23" stroke-width="2" stroke-dasharray="6 5" pointer-events="none"/>
          <g transform="translate(${cx} ${cy})" pointer-events="none"><circle r="26" fill="#1a1d23"/><text y="9" text-anchor="middle" font-size="26">📷</text><text y="52" text-anchor="middle" font-size="22" font-weight="700" fill="#1a1d23">カメラ</text></g>`;
      } else s += `<text x="${Wpx / 2}" y="34" text-anchor="middle" font-size="24" font-weight="700" fill="#2563eb" pointer-events="none">① カメラの位置（コートの外側）をクリック</text>`;
      $("#pick", card).innerHTML = s + "</svg>";
      const svg = $("svg", card);
      svg.onclick = (e) => {
        const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
        const p = pt.matrixTransform(svg.getScreenCTM().inverse());
        const x = p.x / S - L - M, y = Wd + M - p.y / S;
        if (Math.abs(x) <= L && Math.abs(y) <= Wd) hitter = x < 0 ? "left" : "right";
        else { camXY = [x, y]; edge = Math.abs(y) / Wd > Math.abs(x) / L ? (y < 0 ? "bottom" : "top") : (x < 0 ? "left" : "right"); }
        draw();
      };
      const ready = edge && hitter;
      const ans = $("#ans", card);
      ans.innerHTML = "";
      const edgeName = { bottom: "コートの横（手前のサイドライン側）", top: "コートの横（奥のサイドライン側）", left: "コートの後ろ（左のエンド側）", right: "コートの後ろ（右のエンド側）" };
      ans.appendChild(h(`<span class="badge ${edge ? "ok" : ""}">カメラ：${edge ? edgeName[edge] : "未選択"}</span>`));
      ans.appendChild(h(`<span class="badge ${hitter ? "ok" : ""}">打つ人：${hitter ? (hitter === "left" ? "左の半面" : "右の半面") : "未選択"}</span>`));
      const nb = h(`<button class="btn primary" ${ready ? "" : "disabled"} style="margin-left:auto">次へ：点をクリック →</button>`);
      nb.onclick = () => {
        const theta = Court.thetaFrom(edge, hitter);
        if (this.view && this.view.theta !== theta && Object.keys(this.points).length) {
          if (!confirm("カメラの位置を変えると、クリック済みの点の名前がずれるため消去します。よろしいですか？")) return;
          this.points = {}; this.skipped = new Set(); this.auto = {};
        }
        this.view = { edge, hitter, theta, camXY };
        this.saveDraft();
        this.step("points");
      };
      ans.appendChild(nb);
    };
    draw();
  },

  // ---------------- ② 点をクリック ----------------
  order() { return Court.order(App.landmarks.landmarks, this.view.theta, this.extra); },
  nextTarget(from = null) {
    const ord = this.order();
    const start = from ? ord.findIndex((l) => l.name === from) + 1 : 0;
    for (let k = 0; k < ord.length; k++) {
      const lm = ord[(start + k) % ord.length];
      if (!this.points[lm.name] && !this.skipped.has(lm.name) && !this.autoSkipped.has(lm.name)) return lm.name;
    }
    return null;
  },

  async solve() {
    const n = Object.keys(this.points).length;
    if (n < 5) { this.solved = null; return; }
    const r = await api.post("/api/calib/solve", { points: this.points });
    this.solved = r.ok ? r : null;
    this.autoSkipped = new Set();
    if (this.solved && n >= 6) {
      // 予想位置が画面の外になる点は、自動でとばす
      const W = App.meta.width, H = App.meta.height, m = 0.02 * W;
      for (const lm of App.landmarks.landmarks) {
        if (this.points[lm.name]) continue;
        const p = this.solved.predicted[lm.name];
        if (!p || p[0] < -m || p[1] < -m || p[0] > W + m || p[1] > H + m) this.autoSkipped.add(lm.name);
      }
    }
  },

  pointsStep() {
    this.el.appendChild(this.header("points"));
    const ws = h(`<div class="workspace"><div class="left"></div><div class="side stack"></div></div>`);
    this.el.appendChild(ws);
    const left = $(".left", ws), side = $(".side", ws);
    this.cur = this.cur && !this.points[this.cur] ? this.cur : this.nextTarget();
    if (!Object.keys(this.points).length) {
      // まだ何もクリックしていなければ、1 点で自動で合わせられるコートの角から始める
      const corner = this.order().find((l) => l.xyz[2] === 0 && !l.name.startsWith("post_base") && !this.skipped.has(l.name));
      if (corner) this.cur = corner.name;
    }
    this.sel = null;   // 微調整する点
    const lmByName = Object.fromEntries(App.landmarks.landmarks.map((l) => [l.name, l]));
    const fv = new FrameView(left, {
      width: 2560, loupe: true,
      draw: (ctx, f) => {
        if (this.solved && Object.keys(this.points).length >= 6) drawCourtOverlay(ctx, f, this.solved.overlay, 0.35);
        if (this.solved) for (const [n, p] of Object.entries(this.solved.predicted)) {
          if (this.points[n] || this.skipped.has(n) || this.autoSkipped.has(n)) continue;
          if (n === this.cur) drawMarker(ctx, f, p[0], p[1], "#60a5fa", "このあたり？", "ghost");
        }
        if (this.suspect && this.suspect.pred && this.points[this.suspect.name]) {
          const q = this.points[this.suspect.name], pr = this.suspect.pred;
          drawPolyline(ctx, f, [q, pr], "#f97316", 2, [6, 5]);
          drawMarker(ctx, f, pr[0], pr[1], "#f97316", "予想位置", "ghost");
        }
        const ord = this.order().map((l) => l.name);
        for (const [n, p] of Object.entries(this.points)) {
          const e = this.solved ? this.solved.errors[n] : undefined;
          const bad = e !== undefined && e > this.solved.warn_px;
          drawMarker(ctx, f, p[0], p[1], n === this.sel ? "#facc15" : bad ? "#ef4444" : this.auto[n] ? "#06b6d4" : "#22c55e", this.auto[n] ? "自" : String(ord.indexOf(n) + 1 || "+"));
        }
      },
      drawLoupe: (lc, u, v, mag, Lsz) => {
        for (const p of Object.values(this.points)) {
          const x = Lsz / 2 + (p[0] - u) * mag, y = Lsz / 2 + (p[1] - v) * mag;
          if (x > -10 && y > -10 && x < Lsz + 10 && y < Lsz + 10) { lc.strokeStyle = "#22c55e"; lc.lineWidth = 2; lc.beginPath(); lc.arc(x, y, 6, 0, Math.PI * 2); lc.stroke(); }
        }
      },
      onClick: async (u, v) => {
        if (!this.cur) return toast("全部の点が終わりました。「確認へ」を押してください。");
        const placed = this.cur;
        this.suspect = this.checkClick(placed, u, v);
        this.points[placed] = [+u.toFixed(1), +v.toFixed(1)];
        delete this.auto[placed];
        this.skipped.delete(placed);
        this.sel = placed;
        this.history.push(["place", placed]);
        await this.solve();
        this.cur = this.nextTarget(placed);
        this.saveDraft();
        refresh();
      },
    });
    this.history = this.history || [];
    this.fv = fv;
    const tl = new Timeline(left, { onSeek: (t, i) => { this.frameIdx = i; fv.show(i); }, keys: false });
    left.appendChild(h(`<p class="muted small" style="margin:8px 2px">選手でラインが隠れているときは、下の時間軸で見やすいコマに動かしてください。
      拡大：マウスホイール　移動：右ドラッグ（またはスペース＋ドラッグ）　ルーペで角をよく見てクリック</p>`));

    const warn = h(`<div></div>`);
    side.appendChild(warn);
    // 自動で合わせるカードは、いちばん上（スクロールしなくても見える所）に出す
    const snapCard = h(`<div class="card" style="padding:12px;border:2px solid #06b6d4"></div>`);
    side.appendChild(snapCard);
    const target = h(`<div class="target"></div>`);
    side.appendChild(target);
    const ctl = h(`<div class="row"><button class="btn" id="skip">映っていない → とばす <span class="kbd">S</span></button><button class="btn ghost" id="undo">1 つ戻す <span class="kbd">⌫</span></button></div>`);
    side.appendChild(ctl);
    const dia = h(`<div class="card" style="padding:12px"><div class="row" style="justify-content:space-between"><b>コートの図</b><span class="muted small">点をクリックすると、その点を選べます</span></div><div id="dia"></div>
      <div class="row small muted" style="margin-top:6px"><span>● 済み</span><span style="color:#ef4444">● 今の点</span><span>○ これから</span><span style="color:#94a3b8">● とばした</span><span>◇ 高さのある点</span></div>
      <label class="row small" style="margin-top:6px"><input type="checkbox" id="extra"> 内側の交点も使う（より正確に）</label></div>`);
    side.appendChild(dia);
    const status = h(`<div class="card" style="padding:12px"></div>`);
    side.appendChild(status);

    const refresh = () => {
      warn.innerHTML = "";
      if (this.suspect && this.points[this.suspect.name]) {
        const sd = Court.describe(lmByName[this.suspect.name], this.view.theta);
        const box = h(`<div class="callout bad"><b>「${esc(sd.title)}」は、ほかの点から予想される位置と合いません</b>
          <p style="margin:6px 0">${esc(this.suspect.msg)}名前と違う場所（例：ネットポストの「上端」と「根元」、手前と奥）をクリックしていないか確認してください。オレンジの丸が予想位置です。</p>
          <div class="row"><button class="btn sm primary" id="redo1">置き直す</button><button class="btn sm ghost" id="keep1">このままでよい</button></div></div>`);
        $("#redo1", box).onclick = async () => { const n = this.suspect.name; delete this.points[n]; this.cur = n; this.sel = null; this.suspect = null; await this.solve(); this.saveDraft(); refresh(); };
        $("#keep1", box).onclick = () => { this.suspect = null; refresh(); };
        warn.appendChild(box);
      } else this.suspect = null;
      const ord = this.order();
      const nDone = Object.keys(this.points).length;
      const lm = this.cur && lmByName[this.cur];
      if (lm) {
        const d = Court.describe(lm, this.view.theta);
        const idx = ord.findIndex((l) => l.name === lm.name) + 1;
        target.innerHTML = `<div class="step">次にクリックする点（${idx} / ${ord.length}）</div>
          <div class="name">${esc(d.title)}</div>
          <div class="what">${esc(d.what)}</div>
          <div class="muted small" style="margin-top:4px">映っていない・隠れているときは「とばす」。右下の図の点を押せば、どの点からでも始められます。</div>
          <div class="hint">${Court.hintSvg(d.hint)}<div class="small"><b>${esc(d.where)}・${esc(d.side)}</b><br>${{ posttop: "支柱の<b>いちばん上の先端</b>を狙ってください。<b>床に立っている根元ではありません</b>（ネットの白帯と同じくらいの高さ）。", postbase: "支柱の<b>中心が床に接する所</b>を狙ってください。", net: "ネットの白帯の<b>上の縁</b>の、左右の真ん中です。", teec: "センターラインの<b>中心</b>と、横線の<b>外側の縁</b>が交わる所です。", cross: "2 本の線が交わる所の、<b>外側の縁</b>どうしの角です。" }[d.hint] || "線の太さの中ではなく、<b>線の外側の縁</b>（コートの外向き）が交わる所を狙ってください。"}<br>${this.solved && this.solved.predicted[lm.name] ? "動画の上の<b>青い点線の丸</b>が予想位置です。" : ""}</div></div>`;
        fv.setTip(`「${d.title}」をクリック`);
      } else {
        target.innerHTML = `<div class="step">すべての点が終わりました</div><div class="name">確認へ進んでください</div><div class="what">クリックした点は動画の上で直せます（一覧から選んで矢印キーで微調整）。</div>`;
        fv.setTip("");
      }
      const states = {};
      for (const l of App.landmarks.landmarks) {
        if (this.points[l.name]) states[l.name] = this.solved && this.solved.errors[l.name] > this.solved.warn_px ? "err" : "done";
        else if (this.skipped.has(l.name) || this.autoSkipped.has(l.name)) states[l.name] = "skip";
      }
      $("#dia", dia).innerHTML = Court.svg(App.landmarks, { theta: this.view.theta, landmarks: ord.concat(this.extra ? [] : App.landmarks.landmarks.filter((l) => this.points[l.name] && !ord.includes(l))), states, current: this.cur, camera: true });
      $$("#dia .lm", dia).forEach((g) => g.onclick = () => { this.cur = g.dataset.name; this.sel = this.points[this.cur] ? this.cur : null; refresh(); fv.redraw(); });
      this.renderSnap(snapCard, lmByName, refresh);
      const enough = nDone >= 6;
      const hasHeight = Object.keys(this.points).some((n) => lmByName[n].xyz[2] > 0 || n.startsWith("post_base"));
      status.innerHTML = `<div class="row" style="justify-content:space-between"><b>${nDone} 点クリック済み</b>
        ${this.solved ? `<span class="badge ${this.solved.rmse < 3 * rmseScale() ? "ok" : this.solved.rmse < 8 * rmseScale() ? "warn" : "bad"}">ずれ ${fmt(this.solved.rmse, 1)} px</span>` : `<span class="badge">あと ${Math.max(0, 5 - nDone)} 点で計算開始</span>`}</div>
        <p class="muted small" style="margin:6px 0">${enough ? "6 点以上あるので確認に進めます。映っている点が多いほど正確になります。" : "最低 6 点（できれば 8 点以上）クリックしてください。"}${!hasHeight && nDone >= 4 ? "<br><b>ネットポストの上端など、高さのある点も 1 つ以上入れる</b>と安定します。" : ""}${this.autoSkipped.size ? `<br>画面の外にあると予想される ${this.autoSkipped.size} 点は自動でとばしました。` : ""}</p>
        <button class="btn primary block" id="tocheck" ${enough ? "" : "disabled"}>確認へ →</button>`;
      $("#tocheck", status).onclick = () => this.step("check");
      fv.redraw();
    };
    $("#skip", ctl).onclick = () => this.skipCur(refresh);
    $("#undo", ctl).onclick = () => this.undo(refresh);
    $("#extra", dia).onchange = (e) => { this.extra = e.target.checked; this.cur = this.nextTarget(); refresh(); };
    this.keyh = (e) => {
      if (!fv.el.isConnected || e.target.tagName === "INPUT") return;
      if (e.key === "s" || e.key === "S") { this.skipCur(refresh); e.preventDefault(); }
      else if (e.key === "Backspace") { this.undo(refresh); e.preventDefault(); }
      else if (this.sel && this.points[this.sel] && ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(e.key)) {
        const d = e.shiftKey ? 5 : 0.5, p = this.points[this.sel];
        if (e.key === "ArrowLeft") p[0] -= d; if (e.key === "ArrowRight") p[0] += d; if (e.key === "ArrowUp") p[1] -= d; if (e.key === "ArrowDown") p[1] += d;
        e.preventDefault(); fv.redraw();
        clearTimeout(this.nudgeT); this.nudgeT = setTimeout(async () => { await this.solve(); this.saveDraft(); refresh(); }, 400);
      }
    };
    window.addEventListener("keydown", this.keyh);
    this.solve().then(refresh);
    tl.seek(App.state.start != null ? +App.state.start : 0);
  },

  /** 自動で合わせる：コートの角を 1 点（または床の点を 4 点）クリックしたら、残りの点をコートのラインに吸い付けて自動で置く。 */
  renderSnap(card, lmByName, refresh) {
    const floor = Object.keys(this.points).filter((n) => lmByName[n].xyz[2] === 0);
    const corners = floor.filter((n) => !n.startsWith("post_base"));   // ラインの交点（支柱の根元はラインの角ではない）
    const nAuto = Object.keys(this.auto).length;
    const title = (n) => Court.describe(lmByName[n], this.view.theta).title;
    const unconf = (this.unconfirmed || []).filter((n) => this.points[n] && !this.auto[n]);
    let body, btn = null;
    if (nAuto) {
      body = `<b>自動で置いた点を確認してください</b>
        <p class="muted small" style="margin:6px 0"><span style="color:#06b6d4">● 水色の「自」</span>が自動で置いた ${nAuto} 点です。
        下の「確認へ」で黄色の線が実際のラインに重なっているかを見てください。ずれている点があれば、右下の図でその点を選んでクリックし直します。${
        unconf.length ? `<br><b>クリックした点のうち「${unconf.map((n) => esc(title(n))).join("」「")}」は、ラインで確かめられませんでした。</b>ルーペで角をよく見て、正確に置き直してください。` : ""}</p>`;
      btn = ["snap4", "もう一度自動で合わせる", ""];
    } else if (floor.length >= 4) {
      body = `<b>残りの点を自動で合わせられます</b>
        <p class="muted small" style="margin:6px 0">クリックした ${floor.length} 点をもとに、映っているコートのラインを探して、ほかの点を自動で置きます（10 秒ほど）。選手でラインが隠れていないコマで押してください。</p>`;
      btn = ["snap4", "残りの点を自動で合わせる", "primary"];
    } else if (corners.length >= 1) {
      const n = corners[corners.length - 1];
      body = `<b>この 1 点から自動で合わせられます</b>
        <p class="muted small" style="margin:6px 0">「${esc(title(n))}」と、手順 ① のカメラの位置から、コートのラインを探して全部の点を自動で置きます（20〜40 秒）。
        <b>角はルーペで正確に</b>クリックしてください。うまく合わないときは、床の点をあと ${4 - floor.length} 点クリックして 4 点で合わせます。</p>`;
      btn = ["snap1", "この 1 点から自動で合わせる", "primary"];
    } else {
      body = `<b>自動で合わせる：コートの角を 1 点クリック</b>
        <p class="muted small" style="margin:6px 0 0">右下の図で、はっきり見えるコートの角（ラインの交点）を選んで <b>1 点</b>クリックすると、残りの点はコートのラインに吸い付けて自動で置けます。
        1 点で合わないときは、床の点を 4 点（だいたいの位置でよい・なるべく離れた 4 点）クリックして合わせます。</p>`;
    }
    card.innerHTML = body + (btn ? `<button class="btn ${btn[2]} block" id="snap">${btn[1]}</button>` : "");
    if (!btn) return;
    $("#snap", card).onclick = async (e) => {
      e.target.disabled = true; e.target.textContent = "ラインを探しています…";
      const frame = this.frameIdx ?? nearestFrame(App.state.start ?? 0);
      let clicks, r;
      if (btn[0] === "snap1") {
        const n = corners[corners.length - 1];
        clicks = { [n]: this.points[n] };
        r = await api.post("/api/calib/oneclick", { name: n, point: this.points[n], frame, view: this.view });
      } else {
        clicks = { ...this.points };   // 2 回目からは、前に吸い付けた点も手がかりに使う（クリックより正確）
        r = await api.post("/api/calib/snap", { points: clicks, frame });
      }
      if (!r.ok) { toast(r.error, "bad"); refresh(); return; }
      // 前に自動で置いた点は消して置き直す。測れた点（2 本のラインとも見つかった交点）だけを使い、
      // 自分でクリックした点も、吸い付けた位置で置き換える（クリックより正確）
      for (const n of Object.keys(this.auto)) delete this.points[n];
      this.auto = {};
      let nNew = 0;
      for (const [n, st] of Object.entries(r.status)) {
        if (st !== "snapped" || this.skipped.has(n)) continue;
        if (!this.points[n]) nNew++;
        this.points[n] = r.points[n].map((v) => +(+v).toFixed(1));
        this.auto[n] = st;
      }
      // 自分でクリックしたが、ラインで確かめられなかった点（だいたいの位置のまま）
      this.unconfirmed = Object.keys(clicks).filter((n) => !this.auto[n] && lmByName[n].xyz[2] === 0);
      this.sel = null;
      await this.solve();
      this.cur = this.unconfirmed[0] || this.nextTarget();
      this.saveDraft();
      toast(nNew ? `${nNew} 点を自動で置きました。${r.message}` : `新しく置ける点が見つかりませんでした。${r.message}`, nNew ? "ok" : "warn");
      refresh();
    };
  },

  /** クリックした位置が、ほかの点から予想される位置と大きく違えば理由を返す（違わなければ null）。 */
  checkClick(name, u, v) {
    const k = rmseScale();
    // 高さのある点：ネットポストの上端が根元とほぼ同じ高さ（またはそれより下）にある
    if (name.startsWith("post_top_")) {
      const base = this.points[name.replace("post_top_", "post_base_")];
      const pt = this.solved && this.solved.predicted[name], pb = this.solved && this.solved.predicted[name.replace("post_top_", "post_base_")];
      const expect = pt && pb ? Math.hypot(pt[0] - pb[0], pt[1] - pb[1]) : 0;
      if (base && base[1] - v < Math.max(20 * k, 0.4 * expect))
        return { name, pred: pt || null, msg: `クリックした位置が、ネットポストの根元とほぼ同じ高さです（${fmt(base[1] - v, 0)} px 上${expect ? `。予想では約 ${fmt(expect, 0)} px 上` : ""}）。上端は支柱のいちばん上の先です。` };
    }
    if (!this.solved || Object.keys(this.points).filter((n) => n !== name).length < 6) return null;
    const pred = this.solved.predicted[name];
    if (!pred) return null;
    const d = Math.hypot(u - pred[0], v - pred[1]);
    const tol = Math.max(40 * k, 6 * (this.solved.rmse || 0));
    return d > tol ? { name, pred, msg: `予想位置から ${fmt(d, 0)} px 離れています（ほかの点のずれは平均 ${fmt(this.solved.rmse, 1)} px）。` } : null;
  },

  skipCur(refresh) {
    if (!this.cur) return;
    this.skipped.add(this.cur);
    this.history.push(["skip", this.cur]);
    this.cur = this.nextTarget(this.cur);
    this.saveDraft(); refresh();
  },
  async undo(refresh) {
    const last = this.history.pop();
    if (!last) return;
    const [kind, name] = last;
    if (kind === "place") { delete this.points[name]; delete this.auto[name]; } else this.skipped.delete(name);
    this.cur = name; this.sel = null;
    await this.solve(); this.saveDraft(); refresh();
  },

  // ---------------- ③ 確認して保存 ----------------
  async checkStep() {
    this.el.appendChild(this.header("check"));
    const r = await api.post("/api/calib/solve", { points: this.points, k1: !!App.state.calib_k1 });
    if (!r.ok) { this.el.appendChild(h(`<div class="callout bad">${esc(r.error)}</div>`)); return; }
    const ws = h(`<div class="workspace"><div class="left"></div><div class="side stack"></div></div>`);
    this.el.appendChild(ws);
    const left = $(".left", ws), side = $(".side", ws);
    const ex = new Set(r.excluded || []);
    const fv = new FrameView(left, {
      width: 2560, loupe: false,
      draw: (ctx, f) => {
        drawCourtOverlay(ctx, f, r.overlay, 0.95);
        for (const [n, p] of Object.entries(this.points)) drawMarker(ctx, f, p[0], p[1], ex.has(n) ? "#a855f7" : r.errors[n] > r.warn_px ? "#ef4444" : "#22c55e");
      },
    });
    const tl = new Timeline(left, { onSeek: (t, i) => fv.show(i), keys: false });
    const k = rmseScale();
    const q = r.rmse < 3 * k ? ["ok", "良好です"] : r.rmse < 8 * k ? ["warn", "使えますが、赤い点を直すとよくなります"] : ["bad", "ずれが大きいです。赤い点を直してください"];
    if (r.rmse > 20) side.appendChild(h(`<div class="callout bad"><b>このままでは使えません。</b>いくつかの点が、名前と違う場所をクリックしているようです（例：ネットポストの「上端」と「根元」、手前と奥の取り違え）。下の一覧の上から順に、ずれの大きい点を「直す」で確認してください。</div>`));
    side.appendChild(h(`<div class="card"><h2>黄色い線が、実際のコートのラインに重なっていますか？</h2>
      <p class="desc">ピンクはネットです。大きくずれている所があれば、その近くの点を直してください。</p>
      <div class="quality"><span class="big">${fmt(r.rmse, 1)} px</span><span class="badge ${q[0]}">${q[1]}</span></div>
      <p class="muted small" style="margin:6px 0 0">クリックした点と、推定したカメラから計算した位置との平均のずれ（目安：${fmt(3 * k, 0)} px 未満なら良好、${fmt(8 * k, 0)} px 未満なら使えます。画像が大きいほど px の値は大きくなります）。</p></div>`));
    const lmByName = Object.fromEntries(App.landmarks.landmarks.map((l) => [l.name, l]));
    const title = (n) => Court.describe(lmByName[n], this.view ? this.view.theta : 0).title;
    if (ex.size) side.appendChild(h(`<div class="callout warn"><b>${ex.size} 点は、ほかの点と合わないため計算から外しました</b>（紫の点）：${[...ex].map((n) => esc(title(n))).join("、")}。
      名前と違う場所をクリックしている可能性があります。<b>このままでも解析できます</b>が、直すと少し正確になります。</div>`));
    const list = h(`<div class="card" style="padding:12px"><b>点ごとのずれ</b><div class="pointlist"></div></div>`);
    for (const [n, e] of Object.entries(r.errors).sort((a, b) => b[1] - a[1])) {
      const d = Court.describe(lmByName[n], this.view ? this.view.theta : 0);
      const row = h(`<div class="p"><span class="badge ${ex.has(n) ? "" : e > r.warn_px ? "bad" : e > r.warn_px / 2 ? "warn" : "ok"}">${ex.has(n) ? "除外" : `${fmt(e, 1)} px`}</span><span style="flex:1">${esc(d.title)}${ex.has(n) ? `<span class="muted small">（${fmt(e, 0)} px ずれ）</span>` : ""}</span><span class="muted small">直す ›</span></div>`);
      row.onclick = () => { this.cur = n; this.sel = n; this.step("points"); };
      $(".pointlist", list).appendChild(row);
    }
    side.appendChild(list);
    for (const w of r.warnings.filter((w) => !w.includes("再投影誤差") && !w.includes("推定から外しました"))) side.appendChild(h(`<div class="callout warn">${esc(w)}</div>`));
    const save = h(`<div class="card"><div class="stack"></div>
      <label class="row small" style="margin-top:8px"><input type="checkbox" id="k1"> レンズのゆがみも補正する（超広角で撮った場合）</label>
      <div class="row" style="margin-top:12px"><button class="btn" id="back">← 点を直す</button><button class="btn primary" id="save" style="flex:1">保存して次へ →</button></div></div>`);
    $(".stack", save).appendChild(pathField("保存先", "calib", "save", { name: "calibration.json", ext: ".json" }));
    side.appendChild(save);
    const k1 = $("#k1", save); k1.checked = !!App.state.calib_k1;
    k1.onchange = async () => { await App.save({ calib_k1: k1.checked }); this.step("check"); };
    $("#back", save).onclick = () => this.step("points");
    $("#save", save).onclick = async () => {
      if (r.rmse > 20 && !confirm(`ずれが ${r.rmse.toFixed(0)} px と大きく、このまま使うと速度が大きく狂います。
「点ごとのずれ」の上の方（赤）の点を直すことをおすすめします。

それでも保存しますか？`)) return;
      const res = await api.post("/api/calib/save", { points: this.points, k1: k1.checked, time: tl.t, view: this.view, path: App.state.calib });
      if (!res.ok) return toast(res.error, "bad");
      App.calib = res.calib; App.state.calib = res.path; App.state.draft_points = {}; this.points = {}; this.auto = {}; App.save({ draft_auto: {} });
      toast("コートの合わせ込みを保存しました", "ok");
      go("plane");
    };
    tl.seek(App.state.start != null ? +App.state.start : 0);
  },
};
