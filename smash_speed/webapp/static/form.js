// フォームを比べる（側方から撮ったスイングを 2 本選び、棒人間と数字を重ねて比べる。ROADMAP E4）
"use strict";

const FORM_COLORS = { a: "#2563eb", b: "#ea580c" };
const FORM_SERIES = { elbow: "肘の角度", shoulder: "肩の角度", trunk: "胸の回り", wrist_speed: "手首の速さ" };
const FORM_UNITS = { elbow: "°", shoulder: "°", trunk: "°", wrist_speed: "身長/秒" };

App.views.form = {
  title: "フォームを比べる",
  crumb: () => "側方から撮ったスイング",
  date: "all",
  a: null, b: null,       // 選んだショット（dir + id）
  series: "elbow",
  t: 0,                   // 見ている時刻（打点が 0）[s]
  mode: "overlay",        // overlay（重ねる）| side（並べる）

  async render(el) {
    const r = await api.get("/api/form");
    this.info = r.info || [];
    this.shots = r.shots || [];
    this.renderStart(el);
    if (!this.shots.length) return;
    this.renderTable(el);
    this.compare = h(`<div></div>`);
    el.appendChild(this.compare);
    await this.renderCompare();
  },

  key: (s) => s ? `${s.dir}|${s.id}` : "",
  find(k) { return this.shots.find((s) => this.key(s) === k) || null; },

  // ---- 解析する
  renderStart(el) {
    const st = App.state;
    const card = h(`<div class="card"><h2>スイングの形を数字にします</h2>
      <p class="desc">ラケットを持つ腕の側から、打つ向きに<b>まっすぐ横</b>から撮った動画を使います（6〜8 m 離れて、腰くらいの高さ、カメラは水平）。
        打った音のする所ごとに体の関節を見つけ、打った瞬間の肘・肩の角度、手首の速さ、体の回る順番などを出します。</p>
      ${App.meta ? `<div class="row" style="flex-wrap:wrap;gap:12px">
        <div class="seg" id="hand"><button data-v="r">右利き</button><button data-v="l">左利き</button></div>
        <label class="row small">カメラまで <input class="input" id="dist" type="number" step="0.5" min="2" max="15" style="width:70px"> m</label>
        <div class="seg" id="model"><button data-v="rtmpose">標準（速い・GPU）</button><button data-v="mediapipe">軽い（CPU）</button></div>
        <button class="btn primary" id="go">「${esc(App.meta.name)}」を解析する</button></div>
        <p class="small muted" style="margin:8px 0 0">初めて使うときは、関節を見つけるためのデータ（数 MB〜100 MB ほど）を取ってくるので、少し時間がかかります。</p>`
        : `<p class="small">解析するには、先に<a href="#home">ホーム</a>で動画を開いてください。下の一覧は、これまでに解析したスイングです。</p>`}</div>`);
    if (App.meta) {
      let left = !!st.form_left_handed, model = st.form_model || "rtmpose";
      $("#dist", card).value = st.form_distance || 7;
      const upd = () => {
        $$("#hand button", card).forEach((b) => b.classList.toggle("on", (b.dataset.v === "l") === left));
        $$("#model button", card).forEach((b) => b.classList.toggle("on", b.dataset.v === model));
      };
      $$("#hand button", card).forEach((b) => { b.onclick = () => { left = b.dataset.v === "l"; upd(); }; });
      $$("#model button", card).forEach((b) => { b.onclick = () => { model = b.dataset.v; upd(); }; });
      upd();
      $("#go", card).onclick = async () => {
        const distance = +$("#dist", card).value || 7;
        await App.save({ form_left_handed: left, form_distance: distance, form_model: model });
        Task.start("form", { left_handed: left, distance, model },
          (okk) => { toast(okk ? "解析が終わりました" : "解析できませんでした（ログを見てください）", okk ? "ok" : "bad"); go("form"); },
          "スイングを解析しています");
      };
    }
    el.appendChild(card);
  },

  // ---- ショットの一覧（A・B を選ぶ）
  renderTable(el) {
    const dates = [...new Set(this.shots.map((s) => s.date))];
    if (this.date !== "all" && !dates.includes(this.date)) this.date = "all";
    const shown = this.shots.filter((s) => this.date === "all" || s.date === this.date);
    if (!this.find(this.a) && shown.length) this.a = this.key(shown[0]);
    if (!this.find(this.b) && shown.length > 1) this.b = this.key(shown[1]);
    const F = (s, k, d = 1) => fmt(s.features[k], d);
    const card = h(`<div class="card"><div class="row" style="flex-wrap:wrap;gap:8px;margin-bottom:8px">
        <h2 style="margin:0;flex:1">スイング（${shown.length} 本）</h2>
        <select class="input" id="date" style="height:32px"><option value="all">すべての日</option>
          ${dates.map((d) => `<option value="${esc(d)}">${esc(d)}</option>`).join("")}</select>
        <button class="btn sm" id="fastslow">速い球と遅い球を選ぶ</button></div>
      <p class="small" style="margin:0 0 8px"><b style="color:${FORM_COLORS.a}">A</b> と <b style="color:${FORM_COLORS.b}">B</b> の欄で、比べる 2 本を選んでください。
        ほかの日のスイングと比べるときは「すべての日」にします。</p>
      <div style="overflow:auto;max-height:45vh"><table class="tbl"><thead><tr><th>A</th><th>B</th><th>撮った日</th><th>動画</th><th class="r">時刻</th>
        <th class="r">打点の高さ</th><th class="r">肘</th><th class="r">手首の速さ</th><th>体の回る順</th><th class="r">球の速さ</th><th class="r">お手本との近さ</th></tr></thead>
        <tbody></tbody></table></div></div>`);
    $("#date", card).value = this.date;
    $("#date", card).onchange = (e) => { this.date = e.target.value; go("form"); };
    $("#fastslow", card).onclick = () => {
      // 球の速さが分かればそれで、なければ手首の速さで、いちばん速い球と遅い球
      const v = (s) => s.ball_speed_kmh ?? s.features.wrist_speed_max;
      const ok = shown.filter((s) => v(s) != null).sort((x, y) => v(y) - v(x));
      if (ok.length < 2) return toast("速さの分かるスイングが 2 本以上ありません", "bad");
      this.a = this.key(ok[0]); this.b = this.key(ok[ok.length - 1]);
      go("form");
    };
    const tb = $("tbody", card);
    for (const s of shown) {
      const k = this.key(s);
      const order = s.features.chain_in_order == null ? "—" : s.features.chain_in_order ? '<span class="badge ok">腰→肩→肘→手首</span>' : '<span class="badge warn">順番が違う</span>';
      const tr = h(`<tr><td><input type="radio" name="fa" ${k === this.a ? "checked" : ""}></td><td><input type="radio" name="fb" ${k === this.b ? "checked" : ""}></td>
        <td>${esc(s.date)}</td><td class="small">${esc(s.video)}</td><td class="r">${fmt(s.t, 2)} s</td>
        <td class="r">${F(s, "wrist_height_ratio", 2)}</td><td class="r">${F(s, "elbow_angle_deg", 0)}°</td><td class="r">${F(s, "wrist_speed_max", 1)}</td>
        <td>${order}</td><td class="r">${s.ball_speed_kmh != null ? fmt(s.ball_speed_kmh, 0) + " km/h" : "—"}</td>
        <td class="r">${s.score != null ? fmt(s.score, 0) + " 点" : "—"}</td></tr>`);
      const [ra, rb] = $$("input", tr);
      ra.onchange = () => { this.a = k; this.renderCompare(); };
      rb.onchange = () => { this.b = k; this.renderCompare(); };
      tb.appendChild(tr);
    }
    el.appendChild(card);
  },

  // ---- 比べる
  async renderCompare() {
    const box = this.compare;
    box.innerHTML = "";
    const A = this.find(this.a), B = this.find(this.b);
    if (!A) return;
    const get = (s) => s ? api.get(`/api/form/pose?dir=${encodeURIComponent(s.dir)}&id=${encodeURIComponent(s.id)}`) : null;
    const [pa, pb] = await Promise.all([get(A), get(B)]);
    if (!pa || !pa.ok) return box.appendChild(h(`<div class="card"><p class="desc">${esc(pa ? pa.error : "姿勢を読めません")}</p></div>`));
    this.pa = pa; this.pb = pb && pb.ok ? pb : null;
    const label = (s, c, n) => s ? `<b style="color:${c}">${n}</b> ${esc(s.date)} ${esc(s.video)} ${fmt(s.t, 2)} s` : "";
    const card = h(`<div class="card"><div class="row" style="flex-wrap:wrap;gap:8px">
        <h2 style="margin:0;flex:1">形を比べる</h2>
        <div class="seg" id="mode"><button data-v="overlay">重ねる</button><button data-v="side">並べる</button></div></div>
      <p class="small" style="margin:4px 0 8px">${label(A, FORM_COLORS.a, "A")}<br>${label(B, FORM_COLORS.b, "B")}</p>
      <p class="small muted" style="margin:0 0 8px">打った瞬間を 0 秒にそろえ、腰の高さと体の大きさ（胴体の長さ）をそろえて描いています。右が打つ向きです。</p>
      <canvas id="fig" style="width:100%;max-width:720px;aspect-ratio:3/2;display:block;background:var(--bg);border-radius:8px"></canvas>
      <div class="row" style="gap:10px;margin-top:8px;flex-wrap:wrap">
        <button class="btn sm" id="play">▶ 動かす</button>
        <input type="range" id="tt" min="-1" max="0.5" step="0.005" style="flex:1;min-width:160px">
        <span class="small" id="tlabel" style="min-width:110px"></span></div>
      <h3>時間の変化</h3>
      <div class="seg" id="ser">${Object.entries(FORM_SERIES).map(([k, v]) => `<button data-v="${k}">${v}</button>`).join("")}</div>
      <canvas id="chart" style="width:100%;max-width:720px;height:200px;display:block;margin-top:8px"></canvas>
      <h3>打った瞬間の数字</h3>
      <div style="overflow:auto"><table class="tbl"><thead><tr><th></th><th class="r" style="color:${FORM_COLORS.a}">A</th>
        <th class="r" style="color:${FORM_COLORS.b}">B</th><th class="r">B − A</th></tr></thead><tbody id="ftab"></tbody></table></div>
      <p class="small muted" style="margin:8px 0 0">数字の意味と出し方は docs/pose_features.md にまとめています。横から撮った画面の上の値なので、カメラに向かう向きの動きは入りません。</p></div>`);
    box.appendChild(card);
    $$("#mode button", card).forEach((b) => { b.classList.toggle("on", b.dataset.v === this.mode); b.onclick = () => { this.mode = b.dataset.v; $$("#mode button", card).forEach((x) => x.classList.toggle("on", x === b)); this.draw(); }; });
    $$("#ser button", card).forEach((b) => { b.classList.toggle("on", b.dataset.v === this.series); b.onclick = () => { this.series = b.dataset.v; $$("#ser button", card).forEach((x) => x.classList.toggle("on", x === b)); this.drawChart(); }; });
    const tt = $("#tt", card);
    tt.value = this.t;
    tt.oninput = () => { this.t = +tt.value; this.draw(); this.drawChart(); };
    $("#play", card).onclick = () => this.play(tt, $("#play", card));
    const tb = $("#ftab", card);
    for (const [k, name, unit] of this.info) {
      const a = A.features[k], b = B ? B.features[k] : null;
      const d = k === "chain_in_order" ? 2 : Math.abs(a ?? b ?? 0) < 10 ? 2 : 0;
      const val = (v) => k === "chain_in_order" ? (v == null ? "—" : v ? "そう" : "違う") : (v == null ? "—" : fmt(v, d) + (unit && unit !== "秒" ? unit : unit ? " 秒" : ""));
      const diff = a != null && b != null && k !== "chain_in_order" ? (b - a > 0 ? "+" : "") + fmt(b - a, d) : "";
      tb.appendChild(h(`<tr><td>${esc(name)}</td><td class="r">${val(a)}</td><td class="r">${val(b)}</td><td class="r">${diff}</td></tr>`));
    }
    this.fig = $("#fig", card); this.chart = $("#chart", card); this.tlabel = $("#tlabel", card);
    this.draw(); this.drawChart();
  },

  play(tt, btn) {
    if (this.timer) { clearInterval(this.timer); this.timer = null; btn.textContent = "▶ 動かす"; return; }
    btn.textContent = "■ 止める";
    if (this.t >= 0.49) this.t = -1;
    this.timer = setInterval(() => {
      if (!document.body.contains(tt)) { clearInterval(this.timer); this.timer = null; return; }
      this.t = Math.min(0.5, this.t + 0.01 / 4);  // 4 倍ゆっくり
      tt.value = this.t; this.draw(); this.drawChart();
      if (this.t >= 0.5) { clearInterval(this.timer); this.timer = null; btn.textContent = "▶ 動かす"; }
    }, 10);
  },

  // 時刻 t のコマ（いちばん近いもの）
  frameAt(p, t) {
    let best = 0;
    for (let i = 0; i < p.t.length; i++) if (Math.abs(p.t[i] - t) < Math.abs(p.t[best] - t)) best = i;
    return best;
  },

  canvas(c, hgt) {
    const dpr = window.devicePixelRatio || 1, w = c.clientWidth, hh = hgt || c.clientHeight;
    c.width = w * dpr; c.height = hh * dpr;
    const g = c.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, hh);
    return [g, w, hh];
  },

  draw() {
    if (!this.fig) return;
    const [g, w, hh] = this.canvas(this.fig);
    this.tlabel.textContent = `${this.t >= 0 ? "+" : ""}${this.t.toFixed(2)} 秒${Math.abs(this.t) < 0.003 ? "（打った瞬間）" : ""}`;
    const figs = [[this.pa, FORM_COLORS.a], [this.pb, FORM_COLORS.b]].filter(([p]) => p);
    // 胴体の長さを単位に、横 −2.2〜2.8、縦 −2.0〜2.6 を描く（並べるときは横に 2 つ）
    const n = this.mode === "side" ? figs.length : 1;
    const span = [5 * n, 4.6], s = Math.min(w / span[0], hh / span[1]);
    const x0 = (w / s - span[0]) / 2, oy = (hh / s - span[1]) / 2 + 2.6;
    let floor = null;
    figs.forEach(([p, color], k) => {
      const i = this.frameAt(p, this.t), xy = p.xy[i], hip = p.hip[i] || [0, 0];
      const ox = x0 + (this.mode === "side" ? k * 5 : 0) + 2.2;
      const P = (j) => { const q = xy[j]; return q[0] == null ? null : [(ox + q[0] + (hip[0] || 0)) * s, (oy - q[1] - (hip[1] || 0)) * s]; };
      g.strokeStyle = color; g.fillStyle = color; g.lineWidth = 3; g.lineCap = "round";
      g.globalAlpha = 0.9;
      for (const [a, b] of p.skeleton) {
        const pa = P(p.keypoints.indexOf(a)), pb = P(p.keypoints.indexOf(b));
        if (!pa || !pb) continue;
        g.beginPath(); g.moveTo(...pa); g.lineTo(...pb); g.stroke();
      }
      const wr = P(p.keypoints.indexOf("right_wrist"));
      if (wr) { g.beginPath(); g.arc(wr[0], wr[1], 5, 0, 2 * Math.PI); g.fill(); }
      // 頭（鼻・耳の真ん中に、胴体の 0.3 倍の円）
      const face = ["nose", "left_ear", "right_ear"].map((j) => P(p.keypoints.indexOf(j))).filter((q) => q);
      if (face.length) {
        g.beginPath(); g.arc(face.reduce((a, q) => a + q[0], 0) / face.length, face.reduce((a, q) => a + q[1], 0) / face.length, 0.3 * s, 0, 2 * Math.PI); g.stroke();
      }
      // 床：区間の中でいちばん低い足首の少し下
      p.xy.forEach((f, j) => ["left_ankle", "right_ankle"].forEach((a) => {
        const q = f[p.keypoints.indexOf(a)], hy = (p.hip[j] || [0, 0])[1] || 0;
        if (q[1] != null) floor = Math.min(floor ?? Infinity, q[1] + hy);
      }));
    });
    g.globalAlpha = 1;
    if (floor != null) {
      const y = (oy - floor + 0.14) * s;
      g.strokeStyle = "rgba(100,107,120,.5)"; g.lineWidth = 1;
      g.beginPath(); g.moveTo(0, y); g.lineTo(w, y); g.stroke();
    }
  },

  drawChart() {
    if (!this.chart) return;
    const [g, w, hh] = this.canvas(this.chart, 200);
    const pad = { l: 44, r: 10, t: 10, b: 24 };
    const lines = [[this.pa, FORM_COLORS.a], [this.pb, FORM_COLORS.b]].filter(([p]) => p);
    const vals = lines.flatMap(([p]) => p.series[this.series].filter((v) => v != null));
    if (!vals.length) return;
    let lo = Math.min(...vals), hi = Math.max(...vals);
    if (hi - lo < 1e-6) { lo -= 1; hi += 1; }
    const X = (t) => pad.l + (t + 1) / 1.5 * (w - pad.l - pad.r);
    const Y = (v) => pad.t + (hi - v) / (hi - lo) * (hh - pad.t - pad.b);
    g.font = "11px system-ui, sans-serif"; g.fillStyle = "#646b78"; g.strokeStyle = "#e3e6eb"; g.lineWidth = 1;
    for (const t of [-1, -0.5, 0, 0.5]) {
      g.beginPath(); g.moveTo(X(t), pad.t); g.lineTo(X(t), hh - pad.b); g.stroke();
      g.fillText(`${t > 0 ? "+" : ""}${t} 秒`, X(t) - 14, hh - 6);
    }
    g.fillText(FORM_UNITS[this.series], pad.l + 4, pad.t + 10);
    for (const v of [lo, (lo + hi) / 2, hi]) g.fillText(fmt(v, hi - lo < 10 ? 1 : 0), 4, Y(v) + 4);
    for (const [p, color] of lines) {
      g.strokeStyle = color; g.lineWidth = 2; g.beginPath();
      let pen = false;
      p.t.forEach((t, i) => { const v = p.series[this.series][i]; if (v == null) { pen = false; return; } pen ? g.lineTo(X(t), Y(v)) : g.moveTo(X(t), Y(v)); pen = true; });
      g.stroke();
    }
    g.strokeStyle = "#1a1d23"; g.setLineDash([4, 3]);
    g.beginPath(); g.moveTo(X(this.t), pad.t); g.lineTo(X(this.t), hh - pad.b); g.stroke(); g.setLineDash([]);
  },
};
