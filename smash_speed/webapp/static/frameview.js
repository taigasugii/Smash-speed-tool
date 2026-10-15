// 動画のフレーム表示（拡大・移動・ルーペ・重ね描き・クリック）とタイムライン
"use strict";

class FrameView {
  /**
   * @param {HTMLElement} host  表示先
   * @param {object} opt  { width: 取得する画像の横幅, loupe: ルーペを出すか, onClick(u,v), onDragRect(rect), draw(ctx, view) }
   * 座標はすべて元の動画のピクセル（u, v）で扱う。
   */
  constructor(host, opt = {}) {
    this.opt = Object.assign({ width: 1280, loupe: false }, opt);
    this.el = h(`<div class="stage">
      <canvas class="frame"></canvas>
      <div class="hud"></div>
      <div class="zoomctl"><button title="拡大">＋</button><button title="縮小">－</button><button title="全体を表示">⤢</button></div>
      <div class="tip hidden"></div>
      <div class="loupe"><canvas width="360" height="360"></canvas></div></div>`);
    host.appendChild(this.el);
    this.canvas = $("canvas.frame", this.el);
    this.ctx = this.canvas.getContext("2d");
    this.loupe = $(".loupe", this.el);
    this.lctx = $(".loupe canvas", this.el).getContext("2d");
    this.hud = $(".hud", this.el);
    this.tipEl = $(".tip", this.el);
    this.img = null; this.index = -1; this.want = -1; this.loading = false;
    this.zoom = 1; this.cx = null; this.cy = null;
    this.mouse = null; this.dragStart = null; this.panStart = null; this.mode = "click";
    const [zin, zout, zfit] = $$(".zoomctl button", this.el);
    zin.onclick = () => this.zoomAt(1.5); zout.onclick = () => this.zoomAt(1 / 1.5); zfit.onclick = () => this.fit();
    this._bind();
    this.ro = new ResizeObserver(() => this.redraw());
    this.ro.observe(this.canvas);
  }

  get W() { return App.meta ? App.meta.width : 1; }
  get H() { return App.meta ? App.meta.height : 1; }

  show(i) {
    if (!App.meta) return;
    i = Math.max(0, Math.min(App.meta.n_frames - 1, Math.round(i)));
    this.want = i;
    if (!this.loading) this._load();
  }
  _load() {
    const i = this.want;
    this.loading = true;
    const img = new Image();
    const w = this.opt.width === "full" ? 0 : this.opt.width;
    img.onload = () => {
      this.img = img; this.index = i; this.loading = false;
      this.redraw();
      if (this.opt.onFrame) this.opt.onFrame(i);
      if (this.want !== i) this._load();
    };
    img.onerror = () => { this.loading = false; };
    img.src = `/api/frame?i=${i}&w=${w}`;
  }

  // ---- 座標変換 ----
  get base() { const r = this.canvas.getBoundingClientRect(); return Math.min(r.width / this.W, r.height / this.H); }
  get scale() { return this.base * this.zoom; }
  get origin() {
    const r = this.canvas.getBoundingClientRect();
    const cx = this.cx ?? this.W / 2, cy = this.cy ?? this.H / 2;
    return [cx - r.width / (2 * this.scale), cy - r.height / (2 * this.scale)];
  }
  toScreen(u, v) { const [ox, oy] = this.origin; return [(u - ox) * this.scale, (v - oy) * this.scale]; }
  toImage(x, y) { const [ox, oy] = this.origin; return [ox + x / this.scale, oy + y / this.scale]; }

  zoomAt(f, x = null, y = null) {
    const r = this.canvas.getBoundingClientRect();
    if (x === null) { x = r.width / 2; y = r.height / 2; }
    const [u, v] = this.toImage(x, y);
    this.zoom = Math.max(1, Math.min(24, this.zoom * f));
    const s = this.scale;
    this.cx = u - (x - r.width / 2) / s; this.cy = v - (y - r.height / 2) / s;
    this._clamp(); this.redraw();
  }
  fit() { this.zoom = 1; this.cx = this.cy = null; this.redraw(); }
  centerOn(u, v, zoom = null) { if (zoom) this.zoom = zoom; this.cx = u; this.cy = v; this._clamp(); this.redraw(); }
  _clamp() {
    const r = this.canvas.getBoundingClientRect();
    const hw = r.width / (2 * this.scale), hh = r.height / (2 * this.scale);
    if (this.zoom <= 1.0001) { this.cx = this.cy = null; return; }
    this.cx = Math.max(Math.min(this.cx, this.W - hw), hw);
    this.cy = Math.max(Math.min(this.cy, this.H - hh), hh);
  }

  // ---- 描画 ----
  redraw() {
    const c = this.canvas, dpr = window.devicePixelRatio || 1;
    const r = c.getBoundingClientRect();
    if (c.width !== Math.round(r.width * dpr) || c.height !== Math.round(r.height * dpr)) {
      c.width = Math.round(r.width * dpr); c.height = Math.round(r.height * dpr);
    }
    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#111318"; ctx.fillRect(0, 0, r.width, r.height);
    if (!this.img) {
      ctx.fillStyle = "#9aa1ad"; ctx.font = "14px sans-serif"; ctx.textAlign = "center";
      ctx.fillText(App.meta ? "読み込み中…" : "動画が開かれていません", r.width / 2, r.height / 2);
      return;
    }
    const k = this.img.naturalWidth / this.W;  // 元の動画 px → 取得した画像 px
    const [ox, oy] = this.origin, s = this.scale;
    ctx.imageSmoothingEnabled = this.zoom < 4;
    ctx.drawImage(this.img, 0, 0, this.img.naturalWidth, this.img.naturalHeight, -ox * s, -oy * s, this.W * s, this.H * s);
    if (this.opt.draw) this.opt.draw(ctx, this);
    if (this.dragStart && this.mouse && this.mode === "rect") {
      ctx.strokeStyle = "#22d3ee"; ctx.lineWidth = 2; ctx.setLineDash([6, 4]);
      const [x0, y0] = this.dragStart, [x1, y1] = this.mouse;
      ctx.strokeRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
      ctx.setLineDash([]);
    }
    this._drawLoupe(k);
  }

  _drawLoupe(k) {
    if (!this.opt.loupe || !this.mouse || !this.img || this.panStart) { this.loupe.style.display = "none"; return; }
    const [x, y] = this.mouse;
    const [u, v] = this.toImage(x, y);
    if (u < 0 || v < 0 || u > this.W || v > this.H) { this.loupe.style.display = "none"; return; }
    const mag = Math.max(4, this.scale * 6);   // 画面の約 6 倍
    const L = 360, half = L / 2 / mag;          // ルーペに入る元の動画の範囲（片側）[px]
    const lc = this.lctx;
    lc.imageSmoothingEnabled = false;
    lc.fillStyle = "#000"; lc.fillRect(0, 0, L, L);
    lc.drawImage(this.img, (u - half) * k, (v - half) * k, 2 * half * k, 2 * half * k, 0, 0, L, L);
    if (this.opt.drawLoupe) this.opt.drawLoupe(lc, u, v, mag, L);
    lc.strokeStyle = "#ef4444"; lc.lineWidth = 2;
    lc.beginPath(); lc.moveTo(L / 2, L / 2 - 26); lc.lineTo(L / 2, L / 2 - 6); lc.moveTo(L / 2, L / 2 + 6); lc.lineTo(L / 2, L / 2 + 26);
    lc.moveTo(L / 2 - 26, L / 2); lc.lineTo(L / 2 - 6, L / 2); lc.moveTo(L / 2 + 6, L / 2); lc.lineTo(L / 2 + 26, L / 2); lc.stroke();
    const r = this.canvas.getBoundingClientRect();
    const lx = x + 200 > r.width ? x - 200 : x + 20, ly = y - 200 < 0 ? y + 20 : y - 200;
    Object.assign(this.loupe.style, { display: "block", left: lx + "px", top: ly + "px" });
  }

  setTip(text) { this.tipEl.textContent = text || ""; this.tipEl.classList.toggle("hidden", !text); }
  setHud(chips) { this.hud.innerHTML = chips.map((c) => `<span class="chip">${c}</span>`).join(""); }

  // ---- 入力 ----
  _bind() {
    const c = this.canvas;
    const pos = (e) => { const r = c.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    c.addEventListener("contextmenu", (e) => e.preventDefault());
    c.addEventListener("wheel", (e) => { e.preventDefault(); const [x, y] = pos(e); this.zoomAt(e.deltaY < 0 ? 1.25 : 1 / 1.25, x, y); }, { passive: false });
    c.addEventListener("mousedown", (e) => {
      const p = pos(e);
      if (e.button === 1 || e.button === 2 || this.spaceDown) { this.panStart = [p, this.cx ?? this.W / 2, this.cy ?? this.H / 2]; this.el.classList.add("pan"); return; }
      if (e.button === 0) this.dragStart = p;
    });
    window.addEventListener("mousemove", (e) => {
      if (!this.el.isConnected) return;
      const p = pos(e);
      const r = c.getBoundingClientRect();
      const inside = p[0] >= 0 && p[1] >= 0 && p[0] <= r.width && p[1] <= r.height;
      this.mouse = inside || this.dragStart || this.panStart ? p : null;
      if (this.panStart && this.zoom > 1) {
        const [[x0, y0], cx0, cy0] = this.panStart;
        this.cx = cx0 - (p[0] - x0) / this.scale; this.cy = cy0 - (p[1] - y0) / this.scale; this._clamp();
      }
      if (this.opt.onMove && inside) this.opt.onMove(...this.toImage(...p));
      this.redraw();
    });
    window.addEventListener("mouseup", (e) => {
      if (!this.el.isConnected) return;
      const p = pos(e);
      if (this.panStart) { this.panStart = null; this.el.classList.remove("pan"); this.redraw(); return; }
      if (!this.dragStart) return;
      const [x0, y0] = this.dragStart; this.dragStart = null;
      const moved = Math.hypot(p[0] - x0, p[1] - y0) > 6;
      if (this.mode === "rect" && moved) {
        const a = this.toImage(Math.min(x0, p[0]), Math.min(y0, p[1])), b = this.toImage(Math.max(x0, p[0]), Math.max(y0, p[1]));
        const clamp = (v, m) => Math.max(0, Math.min(m, v));
        if (this.opt.onRect) this.opt.onRect([clamp(a[0], this.W), clamp(a[1], this.H), clamp(b[0], this.W), clamp(b[1], this.H)]);
      } else if (!moved && this.opt.onClick) {
        const [u, v] = this.toImage(...p);
        if (u >= 0 && v >= 0 && u <= this.W && v <= this.H) this.opt.onClick(u, v);
      }
      this.redraw();
    });
    window.addEventListener("keydown", (e) => { if (e.code === "Space" && this.el.isConnected && e.target.tagName !== "INPUT") { this.spaceDown = true; e.preventDefault(); } });
    window.addEventListener("keyup", (e) => { if (e.code === "Space") this.spaceDown = false; });
  }
}

// 描画の補助
function drawPolyline(ctx, fv, pts, color, width = 2, dash = null) {
  if (pts.length < 2) return;
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash || []);
  ctx.beginPath();
  pts.forEach(([u, v], k) => { const [x, y] = fv.toScreen(u, v); k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.stroke(); ctx.setLineDash([]);
}
function drawMarker(ctx, fv, u, v, color, label = "", style = "cross") {
  const [x, y] = fv.toScreen(u, v);
  ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 2;
  if (style === "cross") {
    ctx.beginPath(); ctx.moveTo(x - 9, y); ctx.lineTo(x + 9, y); ctx.moveTo(x, y - 9); ctx.lineTo(x, y + 9); ctx.stroke();
    ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.stroke();
  } else if (style === "ghost") {
    ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.arc(x, y, 8, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]);
  } else {
    ctx.beginPath(); ctx.arc(x, y, 7, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "#fff"; ctx.stroke();
  }
  if (label) {
    ctx.font = "bold 12px 'Yu Gothic UI', sans-serif";
    const w = ctx.measureText(label).width + 10;
    ctx.fillStyle = "rgba(17,19,24,.8)"; ctx.fillRect(x + 10, y - 24, w, 18);
    ctx.fillStyle = "#fff"; ctx.fillText(label, x + 15, y - 11);
  }
}
function drawCourtOverlay(ctx, fv, overlay, alpha = 0.9) {
  for (const seg of overlay || []) drawPolyline(ctx, fv, seg.pts, seg.kind === "net" ? `rgba(244,114,182,${alpha})` : `rgba(250,204,21,${alpha})`, 2);
}

// ---------------- タイムライン ----------------
class Timeline {
  /** opt: { onSeek(t), marks: [[t, strength]], range: [t0, t1] } */
  constructor(host, opt = {}) {
    this.opt = opt;
    this.el = h(`<div class="player"><div class="timeline"><canvas></canvas></div>
      <div class="ctl">
        <span class="time">0.000 s</span>
        <button class="btn sm" data-d="-1s">« 1秒</button><button class="btn sm" data-d="-1">‹ 1コマ</button>
        <button class="btn sm primary" data-play>▶ 再生</button>
        <button class="btn sm" data-d="1">1コマ ›</button><button class="btn sm" data-d="1s">1秒 »</button>
        <span class="spacer" style="flex:1"></span>
        <span class="muted small keyhint"><span class="kbd">←</span><span class="kbd">→</span> 1コマ　<span class="kbd">Shift</span>+<span class="kbd">←</span><span class="kbd">→</span> 1秒</span>
      </div></div>`);
    host.appendChild(this.el);
    this.canvas = $("canvas", this.el);
    if (opt.keys === false) $(".keyhint", this.el).remove();
    this.t = 0; this.playing = false;
    $$("[data-d]", this.el).forEach((b) => b.onclick = () => this.step(b.dataset.d));
    $("[data-play]", this.el).onclick = () => this.togglePlay();
    const seekAt = (e) => { const r = this.canvas.getBoundingClientRect(); this.seek(((e.clientX - r.left) / r.width) * this.dur); };
    let drag = false;
    this.canvas.addEventListener("mousedown", (e) => { drag = true; seekAt(e); });
    window.addEventListener("mousemove", (e) => drag && seekAt(e));
    window.addEventListener("mouseup", () => drag = false);
    this.keyh = (e) => {
      if (!this.el.isConnected || e.target.tagName === "INPUT" || this.opt.keys === false) return;
      if (e.key === "ArrowRight") { this.step(e.shiftKey ? "1s" : "1"); e.preventDefault(); }
      if (e.key === "ArrowLeft") { this.step(e.shiftKey ? "-1s" : "-1"); e.preventDefault(); }
    };
    window.addEventListener("keydown", this.keyh);
    new ResizeObserver(() => this.draw()).observe(this.canvas);
  }
  get dur() { return App.meta ? App.meta.duration : 1; }
  get index() { return App.meta ? nearestFrame(this.t) : 0; }
  seek(t) {
    if (!App.meta) return;
    this.t = Math.max(0, Math.min(this.dur, t));
    const i = nearestFrame(this.t);
    this.t = App.meta.timestamps[i];
    $(".time", this.el).textContent = `${this.t.toFixed(3)} s　（${i} / ${App.meta.n_frames - 1} コマ）`;
    this.draw();
    if (this.opt.onSeek) this.opt.onSeek(this.t, i);
  }
  step(d) {
    if (!App.meta) return;
    if (d.endsWith("s")) this.seek(this.t + parseFloat(d));
    else { const i = Math.max(0, Math.min(App.meta.n_frames - 1, this.index + parseInt(d))); this.seek(App.meta.timestamps[i]); }
  }
  togglePlay(until = null) {
    this.playing = !this.playing;
    $("[data-play]", this.el).textContent = this.playing ? "■ 停止" : "▶ 再生";
    if (!this.playing) return;
    this.until = until;
    const tick = () => {
      if (!this.playing || !this.el.isConnected) return;
      if (this.index >= App.meta.n_frames - 1 || (this.until !== null && this.t >= this.until)) { this.togglePlay(); return; }
      this.step("1");
      setTimeout(tick, 1000 / Math.min(App.meta.fps, 30));
    };
    tick();
  }
  playRange(t0, t1) { if (this.playing) this.togglePlay(); this.seek(t0); this.togglePlay(t1); }
  draw() {
    const c = this.canvas, dpr = window.devicePixelRatio || 1, r = c.getBoundingClientRect();
    c.width = Math.round(r.width * dpr); c.height = Math.round(r.height * dpr);
    const ctx = c.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const W = r.width, H = r.height, X = (t) => (t / this.dur) * W;
    ctx.fillStyle = "#f1f3f6"; ctx.fillRect(0, 18, W, H - 26);
    const rg = this.opt.range ? this.opt.range() : null;
    if (rg && rg[0] != null && rg[1] != null) {
      ctx.fillStyle = "rgba(37,99,235,.22)"; ctx.fillRect(X(rg[0]), 18, Math.max(2, X(rg[1]) - X(rg[0])), H - 26);
      ctx.fillStyle = "#2563eb"; ctx.fillRect(X(rg[0]) - 1, 14, 2, H - 18); ctx.fillRect(X(rg[1]) - 1, 14, 2, H - 18);
    }
    const marks = this.opt.marks ? this.opt.marks() : [];
    const maxS = Math.max(1, ...marks.map((m) => m[1]));
    ctx.fillStyle = "#f97316";
    for (const [t, s] of marks) { const hgt = 6 + 20 * Math.sqrt(s / maxS); ctx.fillRect(X(t) - 1, H - 8 - hgt, 2, hgt); }
    // 目盛り
    ctx.fillStyle = "#9aa1ad"; ctx.font = "10px sans-serif"; ctx.textAlign = "center";
    const stepS = [0.1, 0.5, 1, 5, 10, 30, 60].find((s) => this.dur / s < 16) || 120;
    for (let t = 0; t <= this.dur; t += stepS) { ctx.fillRect(X(t), 18, 1, 4); ctx.fillText(t.toFixed(stepS < 1 ? 1 : 0), X(t), 12); }
    ctx.fillStyle = "#1a1d23"; ctx.fillRect(X(this.t) - 1, 14, 3, H - 16);
  }
}

function nearestFrame(t) {
  const ts = App.meta.timestamps;
  let lo = 0, hi = ts.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (ts[m] < t) lo = m; else hi = m; }
  return Math.abs(ts[lo] - t) <= Math.abs(ts[hi] - t) ? lo : hi;
}
