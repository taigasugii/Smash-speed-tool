// ショットの一覧（動画 1 本のすべてのショットを、種類・初速つきで並べる。ROADMAP B5b）
"use strict";

App.views.shots = {
  title: "ショットの一覧",
  crumb: () => App.meta ? App.meta.name : "",
  filter: "all",   // all か種類（smash など）
  sort: "time",    // time（時刻の順）| speed（速い順）

  async render(el) {
    if (!requireVideo(el) || !requireCalib(el)) return;
    const r = await api.get("/api/shots");
    const same = (a, b) => (a || "").replace(/\\/g, "/").toLowerCase() === (b || "").replace(/\\/g, "/").toLowerCase();
    const base = (p) => (p || "").replace(/\\/g, "/").split("/").pop().toLowerCase();
    const mine = r.info && (same(r.info.video, App.meta.path) || base(r.info.video) === base(App.meta.path));
    const shots = mine ? r.shots : [];
    const info = mine ? r.info : {};
    this.labels = r.labels || {};
    this.renderStart(el, info, shots.length > 0);
    if (!mine) return;
    this.renderSummary(el, info, shots);
    this.renderTable(el, info, shots, r.files || {});
  },

  // ---- 解析を始める
  renderStart(el, info, has) {
    const dur = App.meta.duration;
    // 実動画（4K 30fps）の 20 秒で約 70 秒かかった（JOURNAL B5a3）
    const minutes = (s) => Math.max(1, Math.round(s * 3.5 / 60));
    const card = h(`<div class="card"><h2>動画の中のすべてのショットを、種類と初速つきで並べます</h2>
      <p class="desc">打った音とシャトルの向きが変わる所から打った瞬間を見つけ、1 本ずつ初速と種類（スマッシュ・クリア・ドロップなど）を出します。
        手順 2 の<b>コートの合わせ込みをそのまま使う</b>ので、動画の途中でカメラを動かしていないことが前提です。</p>
      <div class="row" style="flex-wrap:wrap;gap:12px;margin-top:8px">
        <div class="seg" id="rng"><button data-v="all">動画全体</button><button data-v="part">区間を決める</button></div>
        <span id="part" class="row small hidden">
          <input class="input" id="s0" type="number" step="0.1" min="0" style="width:90px"> 秒から
          <input class="input" id="s1" type="number" step="0.1" min="0" style="width:90px"> 秒まで</span>
        ${App.state.roi ? `<label class="row small"><input type="checkbox" id="useroi" ${info.roi ? "checked" : ""}> 手順 3 の「検出する範囲」の中だけで探す</label>` : ""}
      </div>
      <div class="row" style="margin-top:12px"><button class="btn primary" id="go">${has ? "解析し直す" : "解析を始める"}</button>
        <span class="muted small" id="eta"></span></div></div>`);
    let part = info.start_s != null || info.end_s != null;
    $("#s0", card).value = fmt(info.start_s ?? 0, 1);
    $("#s1", card).value = fmt(info.end_s ?? dur, 1);
    const upd = () => {
      $$("#rng button", card).forEach((b) => b.classList.toggle("on", (b.dataset.v === "part") === part));
      $("#part", card).classList.toggle("hidden", !part);
      const len = part ? Math.max(0, +$("#s1", card).value - +$("#s0", card).value) : dur;
      $("#eta", card).textContent = `目安 約 ${minutes(len)} 分（動画 ${fmt(len, 0)} 秒）。解析中もほかの画面は使えます。`;
    };
    $$("#rng button", card).forEach((b) => { b.onclick = () => { part = b.dataset.v === "part"; upd(); }; });
    $("#s0", card).oninput = upd;
    $("#s1", card).oninput = upd;
    upd();
    $("#go", card).onclick = () => {
      if (has && !confirm("解析し直すと、前回の一覧は置き換わります。よろしいですか？")) return;
      const data = { use_roi: !!($("#useroi", card) && $("#useroi", card).checked) };
      if (part) {
        data.start = +$("#s0", card).value;
        data.end = +$("#s1", card).value;
        if (!(data.end > data.start)) return toast("終わりの時刻を始まりより後にしてください", "bad");
      }
      Task.start("shots", data, (okk) => { toast(okk ? "解析が終わりました" : "解析できませんでした（ログを見てください）", okk ? "ok" : "bad"); go("shots"); },
        "すべてのショットを解析しています");
    };
    el.appendChild(card);
  },

  // ---- 種類ごとの本数と速さ
  renderSummary(el, info, shots) {
    const by = info.by_type || {};
    const n = shots.length;
    const nSpeed = shots.filter((s) => s.speed_kmh != null).length;
    const range = info.start_s != null ? `${fmt(info.start_s, 1)}〜${fmt(info.end_s, 1)} 秒` : "動画全体";
    const card = h(`<div class="card"><h2>種類ごとの本数と速さ</h2>
      <p class="desc">${range}：ショット ${n} 本（ラリー ${info.n_rallies ?? "—"} 本）、そのうち初速が出たもの ${nSpeed} 本。
        ${info.used_audio ? "打った音も使いました。" : "音がないため、シャトルの動きだけで打った瞬間を見つけました。"}
        カードを押すと、その種類だけを下の表に出します。</p>
      <div class="row" id="tiles" style="flex-wrap:wrap;gap:10px"></div>
      <div class="callout" style="margin-top:12px">種類は、初速・打ち出しの角度・打った高さ・落ちた所から<b>決まりに沿って</b>分けています。
        実際の試合の動画ではまだ確かめきれていないので、<b>目安</b>として見てください。</div>
      <div class="row small" style="margin-top:10px;gap:8px;flex-wrap:wrap"><span class="muted">スマッシュの初速をもっと正確に測るなら、打つ人の足元と落ちた所を確かめてから解析する</span>
        <button class="btn sm" id="tofast">★ いちばん速いスマッシュ へ</button></div></div>`);
    $("#tofast", card).onclick = () => go("fastest");
    const tiles = $("#tiles", card);
    for (const [k, v] of Object.entries(by)) {
      const t = h(`<button class="card" style="padding:10px 14px;min-width:130px;text-align:left;cursor:pointer;margin:0;${this.filter === k ? "outline:2px solid var(--primary)" : ""}">
        <div class="muted small">${esc(v.label)}</div><div style="font-size:22px;font-weight:800">${v.n} <span style="font-size:13px">本</span></div>
        <div class="small">${v.max_kmh != null ? `平均 ${fmt(v.mean_kmh, 0)} / 最高 <b>${fmt(v.max_kmh, 0)}</b> km/h` : '<span class="muted">初速なし</span>'}</div></button>`);
      t.onclick = () => { this.filter = this.filter === k ? "all" : k; go("shots"); };
      tiles.appendChild(t);
    }
    el.appendChild(card);
  },

  // ---- 表
  renderTable(el, info, shots, files) {
    const types = Object.keys(info.by_type || {});
    if (this.filter !== "all" && !types.includes(this.filter)) this.filter = "all";
    const shown = shots.filter((s) => this.filter === "all" || s.shot_type === this.filter);
    if (this.sort === "speed") shown.sort((a, b) => (b.speed_kmh ?? -1) - (a.speed_kmh ?? -1));
    // 打った側（h は x < 0、o は x > 0）を、カメラから見て手前か奥かで言い直す
    const cx = App.calib && App.calib.camera_center ? App.calib.camera_center[0] : null;
    const sideName = (s) => !s ? "—" : cx == null ? (s === "h" ? "片側" : "反対側") : ((s === "h") === (cx < 0) ? "手前" : "奥");
    const card = h(`<div class="card"><div class="row" style="margin-bottom:10px;flex-wrap:wrap;gap:8px">
        <h2 style="margin:0;flex:1">ショット（${shown.length} 本）</h2>
        <div class="seg" id="flt"><button data-v="all">すべて</button>${types.map((k) => `<button data-v="${k}">${esc(this.labels[k] || k)}</button>`).join("")}</div>
        <div class="seg" id="srt"><button data-v="time">時刻の順</button><button data-v="speed">速い順</button></div></div>
      <div style="overflow:auto;max-height:70vh"><table class="tbl"><thead><tr><th>時刻</th><th>ラリー</th><th>打った人</th><th>種類</th>
        <th class="r">初速 km/h</th><th class="r">95% の範囲</th><th>確かさ</th></tr></thead><tbody></tbody></table></div>
      ${files.shots ? `<p class="small" style="margin:10px 0 0"><a href="${fileUrl(files.shots)}" download="shots.csv">一覧を CSV で保存（Excel で開けます）</a></p>` : ""}</div>`);
    $$("#flt button", card).forEach((b) => { b.classList.toggle("on", b.dataset.v === this.filter); b.onclick = () => { this.filter = b.dataset.v; go("shots"); }; });
    $$("#srt button", card).forEach((b) => { b.classList.toggle("on", b.dataset.v === this.sort); b.onclick = () => { this.sort = b.dataset.v; go("shots"); }; });
    const tb = $("tbody", card);
    if (!shown.length) tb.appendChild(h(`<tr><td colspan="7" class="muted">ショットがありません。</td></tr>`));
    for (const s of shown) {
      const ci = s.speed_ci_low_kmh != null && s.speed_ci_high_kmh != null ? `${fmt(s.speed_ci_low_kmh, 0)}–${fmt(s.speed_ci_high_kmh, 0)}` : "—";
      const c = s.speed_kmh == null ? null : s.confidence;  // 初速が出なかったショットの確かさは出さない
      const conf = c == null ? '<span class="badge">—</span>' : c >= 0.7 ? '<span class="badge ok">高い</span>' : c >= 0.4 ? '<span class="badge warn">ふつう</span>' : '<span class="badge bad">低い</span>';
      const rally = s.rally_id != null ? `${s.rally_id}${s.index_in_rally != null ? `<span class="muted small">（${s.index_in_rally} 本目）</span>` : ""}` : "—";
      const tr = tb.appendChild(h(`<tr class="click"${s.id === this.sel ? ' style="outline:2px solid var(--primary)"' : ""}><td>${fmt(s.t_hit_s, 2)} s</td><td>${rally}</td><td>${sideName(s.side)}</td>
        <td>${s.shot_type === "unknown" ? `<span class="muted">${esc(s.label)}</span>` : `<span class="badge info">${esc(s.label)}</span>`}</td>
        <td class="r">${s.speed_kmh != null ? `<b>${fmt(s.speed_kmh, 0)}</b>` : `<span class="muted" title="${esc(s.note)}">出せず</span>`}</td><td class="r">${ci}</td><td>${conf}</td></tr>`));
      tr.dataset.id = s.id;
      // 行を選んだら、画面を作り直さずに詳しい所だけ入れ替えて、そこへ移る
      tr.onclick = () => {
        this.sel = s.id;
        $$("tr.click", tb).forEach((x) => { x.style.outline = +x.dataset.id === s.id ? "2px solid var(--primary)" : ""; });
        this.renderDetail(host, s, true);
      };
    }
    el.appendChild(card);
    const host = el.appendChild(h(`<div></div>`));
    if (this.sel != null && shots.some((s) => s.id === this.sel)) this.renderDetail(host, shots.find((s) => s.id === this.sel), false);
  },

  leave() { this.unbindKeys(); },
  unbindKeys() { if (this.keyh) { window.removeEventListener("keydown", this.keyh); this.keyh = null; } },

  // ---- 選んだ 1 本の場面と、コートを上から見た軌道（ROADMAP B5c）
  async renderDetail(host, s, scroll) {
    const v = await api.get(`/api/shot_view?i=${s.id}`);
    if (!v.ok || this.sel !== s.id) return;  // 待っている間に別の行が選ばれたら描かない
    this.unbindKeys();
    host.innerHTML = "";
    const card = h(`<div class="card"><h2>${fmt(s.t_hit_s, 2)} 秒の${esc(s.label)}${s.speed_kmh != null ? `（${fmt(s.speed_kmh, 0)} km/h）` : ""}</h2>
      <p class="desc">左は打った少し後の場面です。<span style="color:#ea580c">○</span> は見つけたシャトル（<span style="color:#ea580c">●</span> がこのコマ）、<span style="color:#2563eb">線</span> は計算した飛び方
        （<span style="color:#2563eb">●</span> がこのコマの時刻の位置）です。下のボタンか、キーボードの ← → でコマを前後に送れます。
        右はコートを上から見た図で、<span style="color:#16a34a">●</span> が打った所、<span style="color:#dc2626">×</span> が落ちた所です。
        ${s.speed_kmh == null ? "このショットは初速が出せなかったので、飛び方の線はありません。" : ""}</p>
      <div class="row" style="gap:12px;align-items:flex-start;flex-wrap:wrap">
        <div style="flex:2;min-width:320px"><div id="scene" style="position:relative"></div><div id="step"></div></div>
        <div id="top" style="flex:1;min-width:240px"></div></div></div>`);
    if (v.frame != null && v.image_size) this.renderScene(card, v, s);
    // 上から見た図：コートの座標 [m]（x が長い方向）。少し余白をとる
    const xs = v.court.flat().map((p) => Math.abs(p[0])), ys = v.court.flat().map((p) => Math.abs(p[1]));
    const X = Math.max(...xs) + 0.6, Y = Math.max(...ys) + 0.6, sw = 0.05;
    const pathTop = v.path.map((p) => `${p[0]},${-p[1]}`).join(" ");
    $("#top", card).innerHTML = `<svg viewBox="${-X} ${-Y} ${2 * X} ${2 * Y}" style="width:100%;background:#e8f3ea;border-radius:6px">
      ${v.court.map(([a, b]) => `<line x1="${a[0]}" y1="${-a[1]}" x2="${b[0]}" y2="${-b[1]}" stroke="#fff" stroke-width="${sw}"/>`).join("")}
      <line x1="0" y1="${-Y + 0.3}" x2="0" y2="${Y - 0.3}" stroke="#334155" stroke-width="${sw * 1.5}" stroke-dasharray="0.15 0.1"/>
      ${pathTop ? `<polyline points="${pathTop}" fill="none" stroke="#2563eb" stroke-width="${sw * 1.6}"/>` : ""}
      ${v.hit ? `<circle cx="${v.hit[0]}" cy="${-v.hit[1]}" r="0.18" fill="#16a34a"/>` : ""}
      ${v.land ? `<path d="M${v.land[0] - 0.18},${-v.land[1] - 0.18}l0.36,0.36m0,-0.36l-0.36,0.36" stroke="#dc2626" stroke-width="${sw * 1.6}"/>` : ""}</svg>
      ${v.hit ? `<p class="small muted" style="margin:6px 0 0">打った高さ ${fmt(v.hit[2], 2)} m${v.land ? `・ネットから落ちた所まで ${fmt(Math.abs(v.land[0]), 1)} m` : ""}</p>` : ""}`;
    host.appendChild(card);
    if (scroll) card.scrollIntoView({ behavior: "smooth", block: "start" });
  },

  // 場面：コマの画像に、元の画像の大きさの座標で SVG を重ねる。コマは前後に送れる
  renderScene(card, v, s) {
    const [W, H] = v.image_size;
    const frames = v.frames && v.frames.length ? v.frames : [[v.frame, null, null, null]];
    let k = Math.max(0, frames.findIndex((f) => f[0] === v.frame));
    const line = v.path_px.filter((p) => p).map((p) => p.join(",")).join(" ");
    const scene = $("#scene", card);
    scene.innerHTML = `<img style="width:100%;display:block;border-radius:6px;background:#0f172a;aspect-ratio:${W}/${H}">
      <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" style="position:absolute;inset:0;width:100%;height:100%">
        ${line ? `<polyline points="${line}" fill="none" stroke="#2563eb" stroke-width="${W / 300}" opacity=".9"/>` : ""}
        ${v.detections.map(([, x, y]) => `<circle cx="${x}" cy="${y}" r="${W / 250}" fill="none" stroke="#ea580c" stroke-width="${W / 600}"/>`).join("")}
        <circle id="now-det" r="${W / 200}" fill="#ea580c" stroke="#fff" stroke-width="${W / 800}" class="hidden"/>
        <circle id="now-fit" r="${W / 260}" fill="#2563eb" stroke="#fff" stroke-width="${W / 800}" class="hidden"/></svg>`;
    const step = $("#step", card);
    step.innerHTML = `<div class="row" style="gap:8px;margin-top:8px;flex-wrap:wrap">
        <button class="btn sm" id="prev">◀ 前のコマ</button><button class="btn sm" id="next">次のコマ ▶</button>
        <button class="btn sm ghost" id="hit">打った瞬間へ</button>
        <input type="range" id="sl" min="0" max="${frames.length - 1}" value="${k}" style="flex:1;min-width:120px">
        <span class="small muted" id="when" style="min-width:150px"></span></div>`;
    const img = $("img", scene), sl = $("#sl", step);
    // 計算した飛び方の、その時刻の画像の上の位置（path_t の間を直線でつなぐ）
    const fitAt = (t) => {
      const T = v.path_t || [];
      if (t == null || T.length < 2 || t < T[0] || t > T[T.length - 1]) return null;
      let j = 1;
      while (j < T.length - 1 && T[j] < t) j++;
      const a = v.path_px[j - 1], b = v.path_px[j];
      if (!a || !b) return null;
      const r = T[j] > T[j - 1] ? (t - T[j - 1]) / (T[j] - T[j - 1]) : 0;
      return [a[0] + r * (b[0] - a[0]), a[1] + r * (b[1] - a[1])];
    };
    const put = (el, p) => { el.classList.toggle("hidden", !p); if (p) { el.setAttribute("cx", p[0]); el.setAttribute("cy", p[1]); } };
    const show = (i) => {
      k = Math.min(frames.length - 1, Math.max(0, i));
      const [f, t, x, y] = frames[k];
      img.src = `/api/frame?i=${f}&w=1280`;
      sl.value = k;
      put($("#now-det", scene), x != null ? [x, y] : null);
      put($("#now-fit", scene), fitAt(t));
      const dt = t != null ? t - s.t_hit_s : null;
      $("#when", step).textContent = t == null ? `コマ ${f}`
        : `${fmt(t, 2)} 秒（${dt < -0.0005 ? `打つ ${fmt(-dt, 2)} 秒前` : `打ってから ${fmt(dt, 2)} 秒`}）${x == null ? "・シャトルは見つからず" : ""}`;
      $("#prev", step).disabled = k === 0;
      $("#next", step).disabled = k === frames.length - 1;
    };
    $("#prev", step).onclick = () => show(k - 1);
    $("#next", step).onclick = () => show(k + 1);
    sl.oninput = () => show(+sl.value);
    $("#hit", step).onclick = () => {
      let best = 0;
      frames.forEach((f, i) => { if (f[1] != null && Math.abs(f[1] - s.t_hit_s) < Math.abs((frames[best][1] ?? 1e9) - s.t_hit_s)) best = i; });
      show(best);
    };
    this.keyh = (e) => {
      // 入力欄（スライダーも）の上では、その欄の動きに任せる
      if (!scene.isConnected || ["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) return;
      if (e.key === "ArrowLeft") { show(k - 1); e.preventDefault(); }
      if (e.key === "ArrowRight") { show(k + 1); e.preventDefault(); }
    };
    window.addEventListener("keydown", this.keyh);
    show(k);
  },
};
