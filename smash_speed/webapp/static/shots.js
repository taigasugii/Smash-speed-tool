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
        実際の試合の動画ではまだ確かめきれていないので、<b>目安</b>として見てください。</div></div>`);
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
      tb.appendChild(h(`<tr><td>${fmt(s.t_hit_s, 2)} s</td><td>${rally}</td><td>${sideName(s.side)}</td>
        <td>${s.shot_type === "unknown" ? `<span class="muted">${esc(s.label)}</span>` : `<span class="badge info">${esc(s.label)}</span>`}</td>
        <td class="r">${s.speed_kmh != null ? `<b>${fmt(s.speed_kmh, 0)}</b>` : `<span class="muted" title="${esc(s.note)}">出せず</span>`}</td><td class="r">${ci}</td><td>${conf}</td></tr>`));
    }
    el.appendChild(card);
  },
};
