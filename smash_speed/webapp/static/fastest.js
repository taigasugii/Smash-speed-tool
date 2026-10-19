// いちばん速いスマッシュを探す（① 候補を自動で探す → ② 上位の打つ人・着地点を確認 → ③ 詳しく解析して速い順）
"use strict";

App.views.fastest = {
  title: "いちばん速いスマッシュを探す",
  crumb: () => App.meta ? App.meta.name : "",
  filter: "smash",
  editing: null,

  async render(el) {
    if (!requireVideo(el) || !requireCalib(el)) return;
    this.el = el;
    if (!App.landmarks) App.landmarks = await api.get("/api/landmarks");
    const r = await api.get("/api/batch");
    const same = (a, b) => (a || "").replace(/\\/g, "/").toLowerCase() === (b || "").replace(/\\/g, "/").toLowerCase();
    const base = (p) => (p || "").replace(/\\/g, "/").split("/").pop().toLowerCase();
    const cands = r.info && (same(r.info.video, App.meta.path) || base(r.info.video) === base(App.meta.path)) ? r.candidates : [];
    this.cands = cands;
    const minutes = Math.max(1, Math.round(App.meta.duration * 0.9 / 60));

    // ---- 説明と ① 候補探し
    const intro = h(`<div class="card"><h2>動画の中のスマッシュをまとめて解析し、いちばん速いものを探します</h2>
      <div class="steps-intro" style="margin-top:10px">
        <div class="s"><span class="n">①</span><b>候補を自動で探す</b><span class="muted small">打球音の直後に飛んだシャトルを探し、だいたいの初速を出します（目安 約 ${minutes} 分）</span></div>
        <div class="s"><span class="n">②</span><b>上位の候補を確認</b><span class="muted small">速そうな候補の「打つ人の足元」「着地点」を確認します（推定値が入っています）</span></div>
        <div class="s"><span class="n">③</span><b>詳しく解析して並べる</b><span class="muted small">確認した値で解析し、速い順に並べます（1 本 5〜30 秒）</span></div>
      </div>
      <div class="callout" style="margin-top:12px">手順 2 の<b>コートの合わせ込みをそのまま使います</b>。動画の途中でカメラを動かしていないことが前提です。
        ①のだいたいの初速は、打つ人と着地点を推定して出しているため <b>±30% ほどずれる</b>ことがあります（速そうな 8 個を最初から選んでおきます）。順位は ③ の結果で決めてください。</div>
      <div class="row" style="margin-top:12px"><button class="btn primary" id="find">${cands.length ? "候補を探し直す" : "① 候補を探す"}</button>
        ${App.state.roi ? `<label class="row small"><input type="checkbox" id="useroi" ${r.info && r.info.roi ? "checked" : ""}> 手順 3 の「検出する範囲」の中だけで探す（選手の動きと間違えやすい動画で）</label>` : ""}
        ${cands.length ? `<span class="muted small">前回の結果：打球音 ${cands.length} 個のうち、シャトルの軌道が見つかったもの ${cands.filter((c) => c.status === "ok").length} 個</span>` : ""}</div></div>`);
    el.appendChild(intro);
    $("#find", intro).onclick = () => {
      if (cands.some((c) => c.confirmed || c.result) && !confirm("探し直すと、確認した打つ人・着地点と詳しい解析の結果が消えます。よろしいですか？")) return;
      const useRoi = !!($("#useroi", intro) && $("#useroi", intro).checked);
      Task.start("batch_find", { select: 8, use_roi: useRoi }, (okk) => { if (okk) toast("候補を探し終わりました", "ok"); go("fastest"); }, "スマッシュの候補を探しています");
    };
    if (!cands.length) return;

    this.renderRanking(el);
    this.editorHost = h(`<div></div>`);
    el.appendChild(this.editorHost);
    this.renderList(el);
    if (this.editing) this.openEditor(this.editing);
  },

  // ---- ③ の結果（速い順）
  renderRanking(el) {
    const overlap = (a, b) => Number.isFinite(a.ci_low_kmh) && Number.isFinite(b.ci_high_kmh) ? a.ci_low_kmh <= b.ci_high_kmh : true;
    const MAX_KMH = 420;  // これより速い結果はありえないので、順位から外す（batch.MAX_PLAUSIBLE_KMH と同じ）
    const all = this.cands.filter((c) => c.result && c.result.kmh != null);
    const done = all.filter((c) => c.result.kmh <= MAX_KMH).sort((a, b) => b.result.kmh - a.result.kmh);
    const odd = all.filter((c) => c.result.kmh > MAX_KMH).sort((a, b) => b.result.kmh - a.result.kmh);
    const failed = this.cands.filter((c) => c.result && c.result.error);
    if (!done.length && !failed.length && !odd.length) return;
    const top = done[0];
    const card = h(`<div class="card"><h2>③ 詳しく解析した結果（速い順）</h2>
      ${top ? `<div class="hero-num" style="margin:8px 0 12px"><span class="muted small">いちばん速いスマッシュ</span>
        <div style="font-size:30px;font-weight:800">${fmt(top.result.kmh, 1)} <span style="font-size:16px">km/h</span>
        <span class="muted" style="font-size:14px;font-weight:600">${fmt(top.t_onset, 2)} 秒のスマッシュ（95% の範囲 ${fmt(top.result.ci_low_kmh, 0)}〜${fmt(top.result.ci_high_kmh, 0)} km/h）</span></div></div>` : ""}
      <div style="overflow:auto"><table class="tbl"><thead><tr><th>順位</th><th></th><th>時刻</th><th class="r">初速 km/h</th><th class="r">95% の範囲</th><th class="r">コマ</th><th class="r">ずれ px</th><th>打つ人・着地点</th><th></th></tr></thead><tbody></tbody></table></div>
      ${top && done[1] && overlap(top.result, done[1].result) ? `<div class="callout warn" style="margin-top:10px">1 位と 2 位の 95% の範囲が重なっています。<b>差は誤差の範囲内</b>で、どちらが速いかははっきりしません（${fmt(top.t_onset, 2)} 秒と ${fmt(done[1].t_onset, 2)} 秒）。</div>` : ""}
      ${odd.length ? `<div class="callout bad" style="margin-top:10px"><b>ありえない速さになった候補（順位から外しました）</b>：${odd.map((c) => `${fmt(c.t_onset, 2)} 秒（${fmt(c.result.kmh, 0)} km/h）`).join("、")}。
        打つ人の足元・着地点がずれているか、使えたコマが少ない可能性があります。下の一覧の「確認する」で見直してから、もう一度 ③ を押してください。</div>` : ""}
      ${failed.length ? `<p class="muted small" style="margin:8px 0 0">解析できなかった候補 ${failed.length} 個：${failed.map((c) => `${fmt(c.t_onset, 2)} 秒（${esc(c.result.error)}）`).join("、")}</p>` : ""}</div>`);
    const tb = $("tbody", card);
    done.forEach((c, k) => {
      const R = c.result, q = R.rms_px > 10 ? "bad" : R.rms_px > 4 ? "warn" : "ok";
      const ci = R.ci_low_kmh != null && Number.isFinite(R.ci_low_kmh) ? `${fmt(R.ci_low_kmh, 0)}–${fmt(R.ci_high_kmh, 0)}` : "—";
      const wide = !(R.ci_low_kmh != null && Number.isFinite(R.ci_low_kmh)) || (R.ci_high_kmh - R.ci_low_kmh) > 0.4 * R.kmh;
      const tr = h(`<tr><td><b>${k + 1}</b></td><td>${c.thumb_path ? `<img src="${fileUrl(c.thumb_path)}" style="width:120px;border-radius:6px;display:block">` : ""}</td>
        <td>${fmt(c.t_onset, 2)} s</td><td class="r"><b>${fmt(R.kmh, 1)}</b></td><td class="r">${ci}${wide ? '<br><span class="badge warn">幅が広い</span>' : ""}</td><td class="r">${R.n_frames ?? "—"}</td>
        <td class="r"><span class="badge ${q}">${fmt(R.rms_px, 1)}</span></td><td>${R.confirmed ? '<span class="badge ok">確認済み</span>' : '<span class="badge warn">推定のまま</span>'}</td>
        <td><button class="btn sm">詳しく見る</button></td></tr>`);
      $("button", tr).onclick = () => go("result", { dir: R.dir });
      tb.appendChild(tr);
    });
    el.appendChild(card);
  },

  // ---- ①の候補の一覧と ②
  renderList(el) {
    const all = this.cands.slice().sort((a, b) => (b.rough_kmh ?? -1) - (a.rough_kmh ?? -1));
    const shown = all.filter((c) => this.filter === "all" ? true : this.filter === "ok" ? c.status === "ok" : c.status === "ok" && c.smash_like);
    const nSel = this.cands.filter((c) => c.selected && c.status === "ok").length;
    const card = h(`<div class="card"><div class="row" style="margin-bottom:10px;flex-wrap:wrap;gap:8px">
        <h2 style="margin:0;flex:1">① 見つかった候補（だいたいの初速が速い順）</h2>
        <div class="seg" id="flt"><button data-v="smash">スマッシュらしいもの</button><button data-v="ok">軌道が見つかったもの</button><button data-v="all">すべての打球音</button></div></div>
      <p class="desc">「詳しく解析する」に印をつけた候補だけを ③ で解析します。<b>確認する</b>を押すと、打つ人の足元と着地点を動画の上で確かめて直せます。</p>
      <div style="overflow:auto"><table class="tbl"><thead><tr><th>詳しく<br>解析する</th><th></th><th>時刻</th><th class="r">だいたいの<br>初速 km/h</th><th>状態</th><th></th></tr></thead><tbody></tbody></table></div>
      <div class="row" style="margin-top:14px"><button class="btn primary big" id="run" ${nSel ? "" : "disabled"}>③ 選んだ ${nSel} 個を詳しく解析する</button>
        <span class="muted small">目安 約 ${Math.max(1, Math.round(nSel * 20 / 60))} 分。打つ人・着地点を確認していない候補は、推定値のまま解析します。</span></div></div>`);
    $$("#flt button", card).forEach((b) => { b.classList.toggle("on", b.dataset.v === this.filter); b.onclick = () => { this.filter = b.dataset.v; go("fastest"); }; });
    const tb = $("tbody", card);
    if (!shown.length) tb.appendChild(h(`<tr><td colspan="6" class="muted">この絞り込みでは候補がありません。上の切り替えで「軌道が見つかったもの」を選んでください。</td></tr>`));
    for (const c of shown) {
      const st = c.status !== "ok" ? `<span class="badge">${esc(c.note || "シャトルが見つからない")}</span>`
        : [c.confirmed ? '<span class="badge ok">確認済み</span>' : "", c.doubtful ? `<span class="badge bad" title="${esc(c.note)}">推定が怪しい</span>` : "",
          c.smash_like ? '<span class="badge info">スマッシュらしい</span>' : '<span class="badge">スマッシュ以外かも</span>',
          c.result && c.result.kmh != null ? `<span class="badge ok">解析済み ${fmt(c.result.kmh, 0)} km/h</span>` : "",
          c.doubtful ? `<div class="muted small">${esc(c.note)}</div>` : ""].join(" ");
      const tr = h(`<tr><td><input type="checkbox" ${c.selected ? "checked" : ""} ${c.status === "ok" ? "" : "disabled"}></td>
        <td>${c.thumb_path ? `<img src="${fileUrl(c.thumb_path)}" style="width:160px;border-radius:6px;display:block">` : ""}</td>
        <td>${fmt(c.t_onset, 2)} s</td><td class="r"><b>${c.rough_kmh != null ? fmt(c.rough_kmh, 0) : "—"}</b></td><td>${st}</td>
        <td>${c.status === "ok" ? `<button class="btn sm ${this.editing === c.id ? "primary" : ""}">確認する</button>` : ""}</td></tr>`);
      $("input", tr).onchange = async (e) => { await api.post("/api/batch/update", { id: c.id, selected: e.target.checked }); c.selected = e.target.checked; go("fastest"); };
      const btn = $("button", tr);
      if (btn) btn.onclick = () => { this.editing = c.id; go("fastest"); };
      tb.appendChild(tr);
    }
    $("#run", card).onclick = () => Task.start("batch_analyze", {}, (okk) => { toast(okk ? "詳しい解析が終わりました" : "一部の候補を解析できませんでした", okk ? "ok" : "bad"); go("fastest"); }, "選んだ候補を詳しく解析しています");
    el.appendChild(card);
  },

  // ---- ② 打つ人・着地点の確認
  async openEditor(id) {
    const c = this.cands.find((x) => x.id === id);
    if (!c) return;
    const host = this.editorHost;
    host.innerHTML = "";
    const pt = { hitter: c.hitter ? { ...c.hitter } : null, landing: c.landing ? { ...c.landing } : null };
    let tool = "hitter";
    const ov = (await api.get("/api/calib/overlay")).overlay;
    const card = h(`<div class="card"><div class="row" style="margin-bottom:8px"><h2 style="margin:0;flex:1">② ${fmt(c.t_onset, 2)} 秒の候補の「打つ人の足元」と「着地点」</h2>
      <button class="btn sm ghost" id="close">閉じる</button></div>
      <div class="workspace"><div class="left"></div><div class="side stack"></div></div></div>`);
    host.appendChild(card);
    const left = $(".left", card), side = $(".side", card);
    const proj = {};
    const fv = new FrameView(left, {
      width: 1920, loupe: true,
      draw: (ctx, f) => {
        drawCourtOverlay(ctx, f, ov, 0.5);
        if (proj.hitter && proj.landing) drawPolyline(ctx, f, [proj.hitter, proj.landing], "#60a5fa", 3, [10, 6]);
        if (proj.hitter) drawMarker(ctx, f, ...proj.hitter, "#f97316", "打つ人の足元", "dot");
        if (proj.landing) drawMarker(ctx, f, ...proj.landing, "#ef4444", "着地点", "dot");
      },
      onClick: async (u, v) => {
        const r = await api.post("/api/floor", { u, v });
        if (!r.ok) return toast(r.error, "bad");
        pt[tool] = { x: +r.x.toFixed(3), y: +r.y.toFixed(3) };
        if (tool === "hitter") { tool = "landing"; fv.setTip("シャトルが落ちた所（床）をクリック。画面の外なら、そのままで決定"); }
        await upd();
      },
    });
    const tl = new Timeline(left, { onSeek: (t, i) => fv.show(i), range: () => [c.start_s, c.end_s], keys: false });
    side.appendChild(h(`<div class="callout">推定した位置が入っています。<b>オレンジ</b>が打つ人の足元、<b>赤</b>が着地点です。違っていれば、ボタンで選んでから動画の<b>床の上</b>をクリックしてください。動画は下の時間軸で前後に動かせます。</div>`));
    const ctl = h(`<div class="card" style="padding:12px"><div class="stack">
      <button class="btn block" data-t="hitter"></button><button class="btn block" data-t="landing"></button></div>
      <div id="dia" style="margin-top:10px"></div>
      <div class="row" style="margin-top:12px"><button class="btn primary" id="ok" style="flex:1">この位置で決定</button></div>
      <p class="muted small" style="margin:6px 0 0">決定すると「詳しく解析する」にも印がつきます。</p></div>`);
    side.appendChild(ctl);
    const refreshBtns = () => {
      const lab = { hitter: "① 打つ人の足元", landing: "② シャトルの着地点" };
      for (const b of $$("[data-t]", ctl)) {
        const k = b.dataset.t, p = pt[k];
        b.className = "btn block" + (tool === k ? " primary" : "");
        b.innerHTML = `${lab[k]}　${p ? `<span class="badge ${tool === k ? "" : "ok"}">x ${fmt(p.x, 2)} / y ${fmt(p.y, 2)} m</span>` : '<span class="badge warn">未設定</span>'}`;
        b.onclick = () => { tool = k; refreshBtns(); fv.setTip(k === "hitter" ? "打つ人の足元（床）をクリック" : "シャトルが落ちた所（床）をクリック"); };
      }
      $("#dia", ctl).innerHTML = Court.svg(App.landmarks, {
        theta: (App.calib && App.calib.view && App.calib.view.theta) || 0, camera: true,
        points: [pt.hitter && { ...pt.hitter, color: "#f97316", label: "打つ人" }, pt.landing && { ...pt.landing, color: "#ef4444", label: "着地" }].filter(Boolean),
        line: pt.hitter && pt.landing ? [[pt.hitter.x, pt.hitter.y], [pt.landing.x, pt.landing.y]] : null,
      });
    };
    const upd = async () => {
      const pts = [pt.hitter, pt.landing].map((p) => p ? [p.x, p.y, 0] : [0, 0, 0]);
      const r = await api.post("/api/project", { points: pts });
      proj.hitter = pt.hitter ? r.points[0] : null; proj.landing = pt.landing ? r.points[1] : null;
      refreshBtns(); fv.redraw();
    };
    $("#close", card).onclick = () => { this.editing = null; go("fastest"); };
    $("#ok", ctl).onclick = async () => {
      if (!pt.hitter || !pt.landing) return toast("2 か所とも決めてください", "bad");
      await api.post("/api/batch/update", { id: c.id, hitter: pt.hitter, landing: pt.landing, confirmed: true, selected: true });
      toast(`${fmt(c.t_onset, 2)} 秒の候補を確認しました`, "ok");
      // 次の「選んだ・未確認」の候補へ（速い順）
      const next = this.cands.filter((x) => x.id !== c.id && x.selected && !x.confirmed && x.status === "ok").sort((a, b) => (b.rough_kmh ?? 0) - (a.rough_kmh ?? 0))[0];
      this.editing = next ? next.id : null;
      go("fastest");
    };
    fv.setTip("打つ人の足元（床）をクリック");
    await upd();
    // 打った少し後のコマ（打つ人の足元とシャトルの両方が見える）
    tl.seek(Math.min(c.end_s, (c.t_impact ?? c.t_onset) + 0.05));
    host.scrollIntoView({ behavior: "smooth", block: "start" });
  },
};
