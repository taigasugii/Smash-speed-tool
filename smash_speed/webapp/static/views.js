// 各画面（キャリブレーションとアノテーション以外）
"use strict";

async function openVideo(path, captureFps) {
  const r = await api.post("/api/video", { path, capture_fps: captureFps });
  if (!r.ok) { toast(r.error, "bad"); return false; }
  App.meta = r.meta; App.calib = r.calib; App.state = r.state;
  refreshNav();
  return true;
}

// =====================================================================
// ホーム
// =====================================================================
App.views.home = {
  title: "ホーム",
  async render(el) {
    const m = App.meta;
    el.appendChild(h(`<div class="hero">
      <div class="card">
        <h2>スマッシュの初速を、動画から測ります</h2>
        <p class="desc">左のメニューの 1〜5 を順に進めるだけです。各画面に、何をすればよいかを書いてあります。</p>
        <div class="steps-intro">
          <div class="s"><span class="n">手順 1</span><b>スマッシュを選ぶ</b><span class="muted small">動画の中の 1 本を、時間の範囲で選びます</span></div>
          <div class="s"><span class="n">手順 2</span><b>コートを合わせる</b><span class="muted small">コートの角などをクリックして、画面と実際の距離を対応させます（カメラを動かさなければ 1 回だけ）</span></div>
          <div class="s"><span class="n">手順 3</span><b>打つ人と着地点</b><span class="muted small">床の上を 2 回クリックします</span></div>
          <div class="s"><span class="n">手順 4・5</span><b>解析して結果を見る</b><span class="muted small">ボタンを押すと初速・平均速度が出ます</span></div>
        </div>
      </div>
      <div class="card" id="home-open"></div></div>`));
    const box = $("#home-open", el);
    if (m) {
      box.appendChild(h(`<div><h2>開いている動画</h2>
        <p style="margin:6px 0 2px;font-weight:700;word-break:break-all">${esc(m.name)}</p>
        <div class="row small muted" style="margin-bottom:12px">
          <span class="badge">${m.width}×${m.height}</span><span class="badge">${fmt(m.fps, 2)} fps${m.vfr ? "（可変）" : ""}</span>
          <span class="badge">${fmt(m.duration, 1)} 秒</span><span class="badge ${m.has_audio ? "ok" : "warn"}">音声${m.has_audio ? "あり" : "なし"}</span>
          ${m.slow ? `<span class="badge info">撮影 ${m.capture_fps} fps として換算</span>` : ""}</div>
        <div class="row"><button class="btn primary" id="go1">手順 1 へ進む →</button><button class="btn" id="other">別の動画を開く</button></div></div>`));
      $("#go1", box).onclick = () => go("clip");
      $("#other", box).onclick = () => chooseVideo();
    } else {
      box.appendChild(h(`<div class="dropzone"><div class="icon">🎬</div>
        <p style="font-weight:700;margin:6px 0">解析する動画を選んでください</p>
        <p class="muted small" style="margin:0 0 14px">スマホ・カメラで撮った mp4 / mov。動画はこの PC から外には送られません。</p>
        <button class="btn primary big" id="pick">動画を選ぶ…</button></div>`));
      $("#pick", box).onclick = () => chooseVideo();
    }
    const slow = h(`<div class="card"><h2>スロー動画の場合</h2>
      <p class="desc">240fps などで撮ったスロー動画は、書き出すと 30fps として記録されることがあります。その場合は実際の撮影 fps を入れてください（普通の動画は空欄のまま）。</p>
      <div class="row"><div class="field"><label>実際の撮影 fps</label><input class="input num" id="cfps" placeholder="例 240" value="${esc(App.state.capture_fps || "")}"></div>
      <button class="btn" id="apply" style="margin-top:20px">反映する</button></div></div>`);
    el.appendChild(slow);
    $("#apply", slow).onclick = async () => {
      const v = $("#cfps", slow).value.trim();
      if (!App.state.video) return toast("先に動画を選んでください", "bad");
      if (await openVideo(App.state.video, v ? +v : null)) { toast("反映しました", "ok"); go("home"); }
    };
    const demo = h(`<div class="card"><h2>はじめての方へ：お試し用の動画で一通り試す</h2>
      <p class="desc">正解（初速 288 km/h）がわかっている合成動画を作り、コート合わせ・打つ人と着地点・区間を自動で設定します。あとは手順 4 の「解析する」を押すだけです。</p>
      <button class="btn" id="demo">お試し用の動画を作って開く</button></div>`);
    el.appendChild(demo);
    $("#demo", demo).onclick = () => makeDemo();
  },
};

async function chooseVideo() {
  const p = await pickPath("video", { initial: App.state.video ? App.state.video.replace(/[\\/][^\\/]*$/, "") : "" });
  if (!p) return;
  if (await openVideo(p, null)) { toast("動画を開きました"); go("clip"); }
}

async function makeDemo(speed = 80, fps = 240, exposure = 0) {
  await Task.start("synth", { speed, fps, exposure }, async (okk) => {
    if (!okk) return;
    const r = await api.get("/api/synth_info");
    if (!r.ok) return toast(r.error, "bad");
    App.state = (await api.get("/api/state")).state;
    await openVideo(App.state.video, null);
    App.state = (await api.get("/api/state")).state;
    refreshNav();
    toast(`お試し用の動画を開きました（正解の初速 ${r.info.speed0_kmh.toFixed(0)} km/h）`, "ok");
    go("analyze");
  }, "お試し用の動画を作成中");
}

// =====================================================================
// 手順 1：スマッシュを選ぶ
// =====================================================================
App.views.clip = {
  title: "手順 1　スマッシュを選ぶ",
  crumb: () => App.meta ? App.meta.name : "",
  async render(el) {
    if (!requireVideo(el)) return;
    const st = App.state;
    const ws = h(`<div class="workspace"><div class="left"></div><div class="side stack"></div></div>`);
    el.appendChild(ws);
    const left = $(".left", ws), side = $(".side", ws);
    const onsetRows = () => (st.onsets && st.onsets.video === App.meta.path ? st.onsets.rows : []);
    let ons = null;  // 「打球音から探す」カード（下で作る）。時間軸の描画はこれより先に走る
    const minS = () => +((ons && $("#mins", ons)?.value) || 20);
    const fv = new FrameView(left, { width: 1280 });
    const tl = new Timeline(left, { onSeek: (t, i) => fv.show(i), marks: () => onsetRows().filter((r) => r[1] >= minS()), range: () => [st.start, st.end] });
    this.tl = tl;
    side.appendChild(h(`<div class="callout">解析したいスマッシュを <b>1 本</b> 選びます。<br>動画を動かして、<b>打つ少し前</b>を「開始」、<b>シャトルが床に落ちた少し後</b>を「終了」にしてください（1 秒前後）。</div>`));
    const range = h(`<div class="card"><h2>解析する区間</h2>
      <div class="rangebox" style="margin-top:8px">
        <div><div class="muted small">開始</div><div class="v" id="v0">—</div><button class="btn sm block" id="set0">今の位置を開始に</button></div>
        <div><div class="muted small">終了</div><div class="v" id="v1">—</div><button class="btn sm block" id="set1">今の位置を終了に</button></div>
      </div>
      <div class="row" style="margin-top:10px"><button class="btn sm" id="play">▶ この区間を再生</button><span class="muted small" id="len"></span></div>
      <div class="field" style="margin-top:10px"><label>この解析の名前（結果のフォルダ名。空欄なら自動）</label><input class="input" id="shot" value="${esc(st.shot || "")}" placeholder="例 sugii_smash_01"></div>
      <button class="btn primary block" id="next" style="margin-top:12px">次へ：コートを合わせる →</button></div>`);
    side.appendChild(range);
    const upd = () => {
      $("#v0", range).textContent = st.start != null ? `${(+st.start).toFixed(3)} s` : "—";
      $("#v1", range).textContent = st.end != null ? `${(+st.end).toFixed(3)} s` : "—";
      $("#len", range).textContent = st.start != null && st.end != null ? `長さ ${(st.end - st.start).toFixed(2)} 秒` : "";
      tl.draw();
    };
    $("#set0", range).onclick = async () => { await App.save({ start: +tl.t.toFixed(4) }); upd(); };
    $("#set1", range).onclick = async () => { await App.save({ end: +tl.t.toFixed(4) }); upd(); };
    $("#play", range).onclick = () => (st.start != null && st.end != null) ? tl.playRange(+st.start, +st.end) : toast("開始と終了を決めてください");
    $("#shot", range).onchange = (e) => App.save({ shot: e.target.value.trim() });
    $("#next", range).onclick = () => {
      if (!(st.start != null && st.end != null && +st.end > +st.start)) return toast("開始と終了を決めてください", "bad");
      go("calib");
    };
    upd();

    ons = h(`<div class="card"><h2>打球音から探す</h2>
      <p class="desc">ラケットで打った音の候補を探し、上の時間軸にオレンジの線で表示します。クリックするとその少し前へ移動します。</p>
      <div class="row"><button class="btn" id="find">打球音を探す</button>
      <label class="small muted">強さ <input class="input num" id="mins" value="20" style="width:64px;height:30px"> 以上</label></div>
      <div class="onsets" id="list" style="margin-top:10px"></div></div>`);
    side.appendChild(ons);
    const fill = () => {
      const list = $("#list", ons); list.innerHTML = "";
      const rows = onsetRows().filter((r) => r[1] >= minS());
      if (!rows.length) { list.innerHTML = `<div class="muted small" style="padding:10px">まだありません</div>`; return; }
      const mx = Math.max(...rows.map((r) => r[1]));
      for (const [t, s] of rows) {
        const o = h(`<div class="o"><span>${t.toFixed(2)} s</span><span style="width:110px"><div class="bar" style="width:${(100 * s / mx).toFixed(0)}%"></div></span></div>`);
        o.onclick = () => tl.seek(t - 0.3);
        list.appendChild(o);
      }
      tl.draw();
    };
    $("#mins", ons).oninput = fill;
    $("#find", ons).onclick = () => {
      if (!App.meta.has_audio && !st.audio) return toast("この動画には音声がありません。動画を見て区間を決めてください。", "bad");
      Task.start("onsets", {}, async (okk, id) => {
        if (!okk) return;
        const r = await api.get(`/api/onsets?task=${id}`);
        st.onsets = { video: App.meta.path, rows: r.rows };
        fill();
        toast(`打球音の候補が ${r.rows.length} 個見つかりました`, "ok");
      }, "打球音を探しています");
    };
    fill();
    tl.seek(st.start != null ? +st.start : 0);
  },
};

// =====================================================================
// 手順 3：打つ人と着地点（＋検出範囲）
// =====================================================================
App.views.plane = {
  title: "手順 3　打つ人と着地点",
  crumb: () => App.meta ? App.meta.name : "",
  async render(el) {
    if (!requireVideo(el) || !requireCalib(el)) return;
    const st = App.state;
    if (!App.landmarks) App.landmarks = await api.get("/api/landmarks");
    const ov = (await api.get("/api/calib/overlay")).overlay;
    let tool = st.hitter ? (st.landing ? "none" : "landing") : "hitter";
    const ws = h(`<div class="workspace"><div class="left"></div><div class="side stack"></div></div>`);
    el.appendChild(ws);
    const left = $(".left", ws), side = $(".side", ws);
    const proj = {};
    const updateProj = async () => {
      const pts = [st.hitter, st.landing].map((p) => p ? [p.x, p.y, 0] : [0, 0, 0]);
      const r = await api.post("/api/project", { points: pts });
      proj.hitter = st.hitter ? r.points[0] : null; proj.landing = st.landing ? r.points[1] : null;
      fv.redraw();
    };
    const fv = new FrameView(left, {
      width: 1920, loupe: true,
      draw: (ctx, f) => {
        drawCourtOverlay(ctx, f, ov, 0.55);
        if (proj.hitter && proj.landing) drawPolyline(ctx, f, [proj.hitter, proj.landing], "#60a5fa", 3, [10, 6]);
        if (proj.hitter) drawMarker(ctx, f, ...proj.hitter, "#f97316", "打つ人の足元", "dot");
        if (proj.landing) drawMarker(ctx, f, ...proj.landing, "#ef4444", "着地点", "dot");
        if (st.roi) {
          const [a, b] = [f.toScreen(st.roi[0], st.roi[1]), f.toScreen(st.roi[2], st.roi[3])];
          ctx.strokeStyle = "#22d3ee"; ctx.lineWidth = 2; ctx.setLineDash([8, 5]); ctx.strokeRect(a[0], a[1], b[0] - a[0], b[1] - a[1]); ctx.setLineDash([]);
          ctx.fillStyle = "#22d3ee"; ctx.font = "bold 12px sans-serif"; ctx.fillText("検出する範囲", a[0] + 6, a[1] + 16);
        }
      },
      onClick: async (u, v) => {
        if (tool !== "hitter" && tool !== "landing") return;
        const r = await api.post("/api/floor", { u, v });
        if (!r.ok) return toast(r.error, "bad");
        const key = tool;
        await App.save({ [key]: { x: r.x, y: r.y } });
        tool = key === "hitter" && !st.landing ? "landing" : "none";
        await updateProj(); refresh();
      },
      onRect: async (rect) => { if (tool === "roi") { await App.save({ roi: rect.map((v) => Math.round(v)) }); tool = "none"; fv.mode = "click"; refresh(); fv.redraw(); } },
    });
    const tl = new Timeline(left, { onSeek: (t, i) => fv.show(i), range: () => [st.start, st.end] });

    side.appendChild(h(`<div class="callout">シャトルは「打つ人の足元」と「着地点」を通る<b>縦の面</b>の中を飛ぶとして計算します。<br>動画を動かして、それぞれの位置を<b>床の上で</b>クリックしてください。</div>`));
    const card = h(`<div class="card"><h2>位置を決める</h2>
      <div class="stack" style="margin-top:10px">
        <button class="btn block" data-t="hitter"></button>
        <button class="btn block" data-t="landing"></button>
      </div>
      <div id="dia" style="margin-top:12px"></div>
      <details class="adv"><summary>数値で入力する（コート座標 m）</summary>
        <p class="muted small">原点はネット直下のコート中央、打つ人の側が x &lt; 0。</p>
        <div class="grid2">
          <div class="field"><label>打つ人 x, y</label><div class="row"><input class="input num" data-k="hitter.x"><input class="input num" data-k="hitter.y"></div></div>
          <div class="field"><label>着地点 x, y</label><div class="row"><input class="input num" data-k="landing.x"><input class="input num" data-k="landing.y"></div></div>
        </div></details>
      <button class="btn primary block" id="next" style="margin-top:14px">次へ：解析する →</button></div>`);
    side.appendChild(card);
    const roiCard = h(`<div class="card"><h2>検出する範囲（任意）</h2>
      <p class="desc">ふつうは不要です。①②を指定すると、打つ人から着地点までの空間に自動で絞り込み、打球音の時刻に打点から飛び出したシャトルを探します。それでも選手の動きなどと間違えるときだけ、シャトルが飛ぶ範囲を四角で囲みます。</p>
      <div class="row"><button class="btn" id="roi"></button><button class="btn ghost" id="roiclr">消す</button></div></div>`);
    side.appendChild(roiCard);

    const refresh = () => {
      const lab = { hitter: "① 打つ人の足元", landing: "② シャトルの着地点" };
      for (const b of $$("[data-t]", card)) {
        const k = b.dataset.t, p = st[k];
        b.className = "btn block" + (tool === k ? " primary" : "");
        b.innerHTML = `${lab[k]}　${p ? `<span class="badge ${tool === k ? "" : "ok"}">x ${fmt(p.x, 2)} / y ${fmt(p.y, 2)} m</span>` : '<span class="badge warn">未設定</span>'}`;
        b.onclick = () => { tool = k; fv.mode = "click"; refresh(); };
      }
      for (const inp of $$("[data-k]", card)) { const [k, c] = inp.dataset.k.split("."); inp.value = st[k] ? st[k][c] : ""; }
      fv.setTip(tool === "hitter" ? "打つ人の足元（床）をクリック" : tool === "landing" ? "シャトルが落ちた所（床）をクリック。画面の外なら数値で入力" : tool === "roi" ? "シャトルが飛ぶ範囲をドラッグで囲む" : "");
      $("#dia", card).innerHTML = Court.svg(App.landmarks, {
        theta: (App.calib && App.calib.view && App.calib.view.theta) || 0, camera: true,
        points: [st.hitter && { ...st.hitter, color: "#f97316", label: "打つ人" }, st.landing && { ...st.landing, color: "#ef4444", label: "着地" }].filter(Boolean),
        line: st.hitter && st.landing ? [[st.hitter.x, st.hitter.y], [st.landing.x, st.landing.y]] : null,
      });
      $("#roi", roiCard).textContent = tool === "roi" ? "動画の上をドラッグしてください" : st.roi ? "範囲を描き直す" : "範囲を囲む";
      $("#roi", roiCard).className = "btn" + (tool === "roi" ? " primary" : "");
    };
    for (const inp of $$("[data-k]", card)) inp.onchange = async () => {
      const [k, c] = inp.dataset.k.split(".");
      const cur = { ...(st[k] || { x: 0, y: 0 }) }; cur[c] = +inp.value;
      await App.save({ [k]: cur }); await updateProj(); refresh();
    };
    $("#roi", roiCard).onclick = () => { tool = "roi"; fv.mode = "rect"; refresh(); };
    $("#roiclr", roiCard).onclick = async () => { await App.save({ roi: null }); refresh(); fv.redraw(); };
    $("#next", card).onclick = () => (st.hitter && st.landing) ? go("analyze") : toast("2 か所とも決めてください", "bad");
    refresh(); await updateProj();
    tl.seek(st.start != null ? +st.start : 0);
  },
};

// =====================================================================
// 手順 4：解析する
// =====================================================================
const IMPACT = [["auto", "自動（おすすめ）", "打球音 → 打点前の軌道との交点 → 打つ人の位置、の順に使えるものを使います"],
  ["audio", "打球音", "動画の音声から打った瞬間を探します"], ["crossing", "軌道の交点", "飛んできたシャトルと打ち返したシャトルの軌道が交わる時刻"],
  ["hitter_position", "打つ人の位置", "当てはめた軌道が打つ人の位置を通る時刻"], ["first_frame", "最初に写ったコマ", "低 fps では誤差が大きくなります"],
  ["manual", "手動で指定", "下に時刻を入れます"]];

App.views.analyze = {
  title: "手順 4　解析する",
  crumb: () => App.meta ? App.meta.name : "",
  async render(el) {
    if (!requireVideo(el)) return;
    const st = App.state;
    const items = [
      ["clip", "解析する区間", st.start != null && st.end != null ? `${(+st.start).toFixed(2)} 〜 ${(+st.end).toFixed(2)} s（${(st.end - st.start).toFixed(2)} 秒）` : "まだ決めていません"],
      ["calib", "コートの合わせ込み", App.calib && !App.calib.error ? `ずれ ${fmt(App.calib.rmse_px, 1)} px・${App.calib.n_points} 点${App.calib.rmse_px > 20 || App.calib.n_points < 6 ? "（大きすぎます）" : ""}` : "まだです"],
      ["plane", "打つ人と着地点", st.hitter && st.landing ? `打つ人 (${fmt(st.hitter.x, 2)}, ${fmt(st.hitter.y, 2)}) → 着地 (${fmt(st.landing.x, 2)}, ${fmt(st.landing.y, 2)}) m` : "まだ決めていません（空のままでも自動で推定しますが、精度は下がります）"],
    ];
    const g = h(`<div class="grid2"><div class="card"><h2>準備の確認</h2><div class="checklist"></div></div>
      <div class="card"><h2>設定</h2>
        <div class="field"><label>打った瞬間（打点時刻）の求め方</label><select class="input" id="imp">${IMPACT.map(([v, l]) => `<option value="${v}">${l}</option>`).join("")}</select>
        <span class="muted small" id="impd"></span></div>
        <div class="field hidden" id="manual" style="margin-top:8px"><label>打点時刻 [s]</label><input class="input num" id="impt" value="${esc(st.impact_time ?? "")}"></div>
        <details class="adv"><summary>詳細設定</summary>
          <div class="stack" style="margin-top:10px">
            <div class="field"><label>シャトルの見つけ方</label><div class="seg" id="det"><button data-v="baseline">ふつう（背景との差）</button><button data-v="labels">手でつけた位置</button><button data-v="tracknet">TrackNetV3（要準備）</button></div>
              <span class="muted small">「手でつけた位置」は、研究ツールの「正解ラベルを付ける」で保存した位置を使います（打った瞬間の印があれば、その後のコマを使います）。</span></div>
            <div class="field"><label>信頼区間（誤差の幅）の計算</label><div class="seg" id="ci"><button data-v="jacobian">速い</button><button data-v="bootstrap">丁寧（遅い）</button><button data-v="none">出さない</button></div></div>
            <label class="row small"><input type="checkbox" id="ovl"> 結果を描き込んだ動画も作る</label>
            <div id="audio"></div>
          </div></details>
      </div></div>`);
    el.appendChild(g);
    const cl = $(".checklist", g);
    const calibBad = App.calib && !App.calib.error && (App.calib.rmse_px > 20 || App.calib.n_points < 6);
    for (const [id, name, val] of items) {
      const okk = stepDone(id) && !(id === "calib" && calibBad);
      const row = h(`<div class="ci"><span class="ic ${okk ? "ok" : "ng"}">${okk ? "✓" : "!"}</span><span class="t"><b>${name}</b><span class="muted small">${esc(val)}</span></span><button class="btn sm ghost">${okk ? "変更" : "設定する"}</button></div>`);
      $("button", row).onclick = () => go(id);
      cl.appendChild(row);
    }
    const sel = $("#imp", g); sel.value = st.impact_method || "auto";
    const showImp = () => { $("#impd", g).textContent = IMPACT.find((x) => x[0] === sel.value)[2]; $("#manual", g).classList.toggle("hidden", sel.value !== "manual"); };
    sel.onchange = () => { App.save({ impact_method: sel.value }); showImp(); }; showImp();
    $("#impt", g).onchange = (e) => App.save({ impact_time: e.target.value ? +e.target.value : null });
    const seg = (id, key, def) => { const s = $(id, g); const set = (v) => { $$("button", s).forEach((b) => b.classList.toggle("on", b.dataset.v === v)); }; set(st[key] || def); $$("button", s).forEach((b) => b.onclick = () => { App.save({ [key]: b.dataset.v }); set(b.dataset.v); }); };
    seg("#det", "detector", "baseline"); seg("#ci", "ci", "jacobian");
    const ovl = $("#ovl", g); ovl.checked = st.overlay !== false; ovl.onchange = () => App.save({ overlay: ovl.checked });
    $("#audio", g).appendChild(pathField("別の音声ファイル（任意・WAV）", "audio", "file", { label: "WAV", pattern: "*.wav" }));

    const ready = stepDone("clip") && stepDone("calib") && !calibBad;
    const runCard = h(`<div class="card" style="margin-top:16px;text-align:center">
      ${calibBad ? `<div class="callout bad" style="text-align:left;margin-bottom:12px"><b>コートの合わせ込みのずれが ${fmt(App.calib.rmse_px, 0)} px（${App.calib.n_points} 点）と大きいため、解析できません。</b>
        このまま計算すると速度が大きく狂います。手順 2 で、ずれの大きい点を直してください（目安：5 px 以下、6 点以上）。
        <div style="margin-top:8px"><button class="btn sm" onclick="go('calib')">手順 2 へ</button></div></div>` : ""}
      <button class="btn primary big" id="run" ${ready ? "" : "disabled"} style="min-width:280px">▶ 解析する</button>
      <p class="muted small" style="margin:8px 0 0">${ready ? "4K の動画では 1〜3 分ほどかかります。進み具合は画面の下に出ます。" : "区間とコートの合わせ込みが済むと押せます。"}</p></div>`);
    el.appendChild(runCard);
    const errBox = h(`<div></div>`);
    el.appendChild(errBox);
    $("#run", runCard).onclick = () => { errBox.innerHTML = ""; Task.start("analyze", {}, async (okk, id) => {
      await App.save({ last_result_ok: okk });
      if (okk) { toast("解析が終わりました", "ok"); go("result"); return; }
      // 失敗の理由（[エラー]）と対処（[対処]）をログから取り出して、画面に出す
      const lines = ((await api.get(`/api/task/${id}?since=0`)).lines || []);
      const msg = lines.filter((l) => l.startsWith("[エラー]")).map((l) => l.slice(5).trim());
      const hints = lines.filter((l) => l.startsWith("[対処]")).map((l) => l.slice(4).trim());
      errBox.appendChild(h(`<div class="callout bad" style="margin-top:16px"><b>解析できませんでした。</b>
        ${msg.length ? `<p style="margin:6px 0">${esc(msg.join(" "))}</p>` : `<p style="margin:6px 0">画面の下の「ログを表示」で詳細を確認してください。</p>`}
        ${hints.length ? `<b>次のことを試してください</b><ol style="margin:6px 0 0;padding-left:20px">${hints.map((x) => `<li>${esc(x)}</li>`).join("")}</ol>` : ""}</div>`));
      errBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
      toast("解析できませんでした。理由を画面に表示しました。", "bad");
    }, "解析しています"); };
  },
};

// =====================================================================
// 手順 5：結果
// =====================================================================
App.views.result = {
  title: "手順 5　結果",
  crumb: () => "",
  async render(el, opts = {}) {
    const r = await api.get("/api/result" + (opts.dir ? `?dir=${encodeURIComponent(opts.dir)}` : ""));
    if (!r.ok) { el.appendChild(h(`<div class="card"><h2>まだ結果がありません</h2><p class="desc">手順 4 で解析すると、ここに表示されます。</p><button class="btn primary" onclick="go('analyze')">手順 4 へ</button></div>`)); return; }
    $("#page-crumb").textContent = r.row.shot;
    const R = r.row;
    const ciOk = R.speed_initial_ci_low_kmh != null;
    const quality = [];
    if (R.rms_px > 10) quality.push(["bad", "画面上のずれが大きい（" + fmt(R.rms_px, 1) + " px）。シャトルの検出かコートの合わせ込みを確認してください"]);
    else if (R.rms_px > 4) quality.push(["warn", "画面上のずれがやや大きい（" + fmt(R.rms_px, 1) + " px）"]);
    else quality.push(["ok", "画面上のずれは小さい（" + fmt(R.rms_px, 1) + " px）"]);
    if (R.n_frames_used < 6) quality.push(["bad", `使えたコマが ${R.n_frames_used} 個と少ない。手順 3 の「打つ人の足元」「着地点」を正確に指定し直すか、「検出する範囲」で打った後にシャトルが飛ぶ範囲を囲むと拾いやすくなります`]);
    else quality.push(["ok", `${R.n_frames_used} コマを使って計算`]);
    const impName = { audio: "打球音", crossing: "軌道の交点", hitter_position: "打つ人の位置", first_frame: "最初に写ったコマ", manual: "手動" }[R.impact_method] || R.impact_method;
    quality.push([R.impact_method === "audio" || R.impact_method === "manual" ? "ok" : "warn", `打った瞬間は「${impName}」から推定（±${fmt(R.impact_sigma_s * 1000, 1)} ms）`]);

    el.appendChild(h(`<div class="kpis">
      <div class="kpi main"><div class="l">初速（打った直後の速さ）</div><div class="v">${fmt(R.speed_initial_kmh, 1)}<small>km/h</small></div>
        <div class="s">${fmt(R.speed_initial_mps, 1)} m/s　${ciOk ? `95% の範囲 ${fmt(R.speed_initial_ci_low_kmh, 0)}〜${fmt(R.speed_initial_ci_high_kmh, 0)} km/h` : "範囲は求められませんでした"}</div></div>
      <div class="kpi"><div class="l">平均速度（飛んだ道のり）</div><div class="v">${fmt(R.speed_mean_path_kmh, 1)}<small>km/h</small></div><div class="s">直線距離ベース ${fmt(R.speed_mean_straight_kmh, 1)} km/h</div></div>
      <div class="kpi"><div class="l">着地直前の速さ</div><div class="v">${fmt(R.speed_arrival_kmh, 1)}<small>km/h</small></div><div class="s">初速の ${fmt(R.deceleration_ratio * 100, 0)}%（減速率）</div></div>
      <div class="kpi"><div class="l">飛行時間・距離</div><div class="v">${fmt(R.flight_time_s, 3)}<small>秒</small></div><div class="s">${fmt(R.path_length_m, 2)} m を飛行</div></div></div>`));
    const q = h(`<div class="card" style="margin-top:16px"><h2>結果の信頼性</h2><div class="stack" style="margin-top:8px"></div>
      <p class="muted small" style="margin:10px 0 0">参考：単純に 2 コマの間の距離から計算すると ${fmt(R.naive_first_interval_kmh, 1)} km/h（低い fps ほど実際より小さく出ます）。</p></div>`);
    for (const [k, t] of quality) $(".stack", q).appendChild(h(`<div class="row"><span class="badge ${k}">${k === "ok" ? "✓ 良好" : k === "warn" ? "注意" : "要確認"}</span><span>${esc(t)}</span></div>`));
    for (const l of r.log) $(".stack", q).appendChild(h(`<div class="row"><span class="badge">メモ</span><span class="muted">${esc(l)}</span></div>`));
    el.appendChild(q);
    const figs = [["image_fit", "画面上の当てはめ"], ["speed_time", "速さの変化"], ["trajectory", "飛んだ軌道（横・上から）"], ["overlay", "結果を描き込んだ動画"]].filter(([k]) => r.files[k]);
    const fc = h(`<div class="card" style="margin-top:16px"><div class="tabs"></div><div id="fig"></div>
      <div class="row" style="margin-top:10px"><button class="btn sm" id="folder">結果のフォルダを開く</button><button class="btn sm" id="csv">CSV を開く</button><span class="spacer" style="flex:1"></span><button class="btn sm" onclick="go('list')">ほかの結果を見る</button></div></div>`);
    el.appendChild(fc);
    const show = (k) => {
      $$(".tabs button", fc).forEach((b) => b.classList.toggle("on", b.dataset.k === k));
      $("#fig", fc).innerHTML = k === "overlay" ? `<video class="figure" src="${fileUrl(r.files[k])}" controls autoplay muted loop></video>` : `<img class="figure" src="${fileUrl(r.files[k])}">`;
    };
    for (const [k, l] of figs) { const b = h(`<button data-k="${k}">${l}</button>`); b.onclick = () => show(k); $(".tabs", fc).appendChild(b); }
    if (figs.length) show(figs[0][0]);
    $("#folder", fc).onclick = () => api.post("/api/open", { path: r.dir });
    $("#csv", fc).onclick = () => api.post("/api/open", { path: r.files.csv });
  },
};

// =====================================================================
// 結果の一覧
// =====================================================================
App.views.list = {
  title: "結果の一覧",
  async render(el) {
    const r = await api.get("/api/results");
    const c = h(`<div class="card"><div class="row" style="margin-bottom:10px"><h2 style="margin:0;flex:1">これまでの解析（${r.rows.length} 件）</h2>
      <button class="btn sm" id="sum">集計表（CSV）を作る</button><button class="btn sm" id="open">出力フォルダを開く</button></div>
      <div style="overflow:auto"><table class="tbl"><thead><tr><th>名前</th><th class="r">初速 km/h</th><th class="r">95% の範囲</th><th class="r">平均 km/h</th><th class="r">fps</th><th>打点の求め方</th><th class="r">ずれ px</th><th>日時</th></tr></thead><tbody></tbody></table></div></div>`);
    el.appendChild(c);
    const tb = $("tbody", c);
    if (!r.rows.length) tb.appendChild(h(`<tr><td colspan="8" class="muted">まだありません</td></tr>`));
    for (const R of r.rows) {
      const ci = R.speed_initial_ci_low_kmh != null ? `${fmt(R.speed_initial_ci_low_kmh, 0)}–${fmt(R.speed_initial_ci_high_kmh, 0)}` : "—";
      const q = R.rms_px > 10 ? "bad" : R.rms_px > 4 ? "warn" : "ok";
      const tr = h(`<tr class="click"><td><b>${esc(R.shot)}</b></td><td class="r"><b>${fmt(R.speed_initial_kmh, 1)}</b></td><td class="r">${ci}</td><td class="r">${fmt(R.speed_mean_path_kmh, 1)}</td>
        <td class="r">${fmt(R.capture_fps, 0)}</td><td>${esc(R.impact_method || "")}</td><td class="r"><span class="badge ${q}">${fmt(R.rms_px, 1)}</span></td><td class="muted small">${new Date(R.mtime * 1000).toLocaleString("ja-JP")}</td></tr>`);
      tr.onclick = () => go("result", { dir: R.dir });
      tb.appendChild(tr);
    }
    $("#sum", c).onclick = () => Task.start("summarize", {}, (okk) => okk && api.post("/api/open", { path: App.state.out_root + "/summary.csv" }), "集計表を作成");
    $("#open", c).onclick = () => api.post("/api/open", { path: App.state.out_root });
  },
};

// =====================================================================
// 研究ツール
// =====================================================================
App.views.tools = {
  title: "研究ツール",
  async render(el) {
    const grid = h(`<div class="grid2"></div>`);
    el.appendChild(grid);
    const card = (title, desc, body) => { const c = h(`<div class="card"><h2>${title}</h2><p class="desc">${desc}</p></div>`); c.appendChild(body); grid.appendChild(c); return c; };

    const ann = h(`<div class="stack"><button class="btn primary" id="go">ラベル付けを始める</button></div>`);
    card("正解ラベルを付ける", "手順 1 で選んだ区間のシャトルの位置を 1 コマずつクリックします。TrackNetV3 の学習や精度検証の正解になります。", ann);
    $("#go", ann).onclick = () => App.meta ? go("annotate") : toast("先に動画を開いてください", "bad");

    const syn = h(`<div class="stack"><div class="row">
      <div class="field"><label>初速 [m/s]</label><input class="input num" id="sp" value="80"></div>
      <div class="field"><label>fps</label><select class="input" id="fp"><option>30</option><option>60</option><option>120</option><option selected>240</option></select></div>
      <div class="field"><label>露光 [s]</label><input class="input num" id="ex" value="0"></div></div>
      <button class="btn" id="go">作って開く</button></div>`);
    card("合成データ（正解つき）", "既知の初速のシャトルが飛ぶ動画を作ります。fps や露光（ブレ）を変えて、解析がどこまで正確かを確かめられます。", syn);
    $("#go", syn).onclick = () => makeDemo(+$("#sp", syn).value, +$("#fp", syn).value, +$("#ex", syn).value);

    const low = h(`<div class="stack"></div>`);
    low.appendChild(pathField("正解ラベル（labels.csv・任意）", "low_labels", "file", { label: "CSV", pattern: "*.csv" }));
    const lowOpt = h(`<div class="row"><label><input type="checkbox" data-f="30" checked> 30fps</label><label><input type="checkbox" data-f="60" checked> 60fps</label><label><input type="checkbox" data-f="120" checked> 120fps</label>
      <label><input type="checkbox" id="blur" checked> ブレを再現</label><button class="btn" id="go">作る</button></div>`);
    low.appendChild(lowOpt);
    card("疑似低 fps 動画", "開いている高 fps の動画（例 240fps）から 30 / 60 / 120fps 版を作ります。ラベルがあれば正解も自動で付きます。", low);
    $("#go", lowOpt).onclick = () => {
      if (!App.meta) return toast("先に動画を開いてください", "bad");
      const fps = $$("[data-f]", lowOpt).filter((c) => c.checked).map((c) => +c.dataset.f);
      Task.start("lowfps", { fps, blur: $("#blur", lowOpt).checked, labels: App.state.low_labels }, (okk) => okk && api.post("/api/open", { path: App.state.out_root + "/lowfps" }), "疑似低 fps 動画を作成");
    };

    const val = h(`<div class="stack"><div class="row"><div class="field"><label>合成データの本数</label><input class="input num" id="n" value="100"></div><button class="btn" id="vs" style="margin-top:20px">合成データで検証（数分）</button></div></div>`);
    val.appendChild(pathField("実データの一覧 CSV（README 参照）", "val_index", "file", { label: "CSV", pattern: "*.csv" }));
    val.appendChild(pathField("スピードガンなどの値（任意）", "val_ref", "file", { label: "CSV", pattern: "*.csv" }));
    const vd = h(`<button class="btn">実データで検証</button>`); val.appendChild(vd);
    card("精度検証", "fps ごとに初速の誤差を比べ、表・Bland–Altman 図を作ります（研究の中核）。", val);
    $("#vs", val).onclick = () => Task.start("validate_synthetic", { n: +$("#n", val).value }, (okk) => okk && api.post("/api/open", { path: App.state.out_root + "/validation_synthetic" }), "精度検証（合成データ）");
    vd.onclick = () => App.state.val_index ? Task.start("validate_dataset", { index: App.state.val_index, reference: App.state.val_ref }, (okk) => okk && api.post("/api/open", { path: App.state.out_root + "/validation" }), "精度検証（実データ）") : toast("一覧 CSV を選んでください", "bad");

    const tn = h(`<div class="stack"></div>`);
    tn.appendChild(pathField("データ一覧 CSV", "exp_index", "file", { label: "CSV", pattern: "*.csv" }));
    const tnb = h(`<div class="row"><button class="btn">TrackNetV3 の形式で書き出す</button><button class="btn ghost">Colab 用ノートブックを開く</button></div>`); tn.appendChild(tnb);
    card("TrackNetV3 の学習データ", "ラベル付きの動画を撮影セッション単位で学習 / テストに分けて書き出します。学習は Google Colab で行います。", tn);
    const [b1, b2] = $$("button", tnb);
    b1.onclick = () => App.state.exp_index ? Task.start("export_tracknet", { index: App.state.exp_index }, (okk) => okk && api.post("/api/open", { path: App.state.out_root + "/tracknet_dataset" }), "学習データを書き出し") : toast("一覧 CSV を選んでください", "bad");
    b2.onclick = async () => { const s = await api.get("/api/state"); api.post("/api/open", { path: s.project_root + "/notebooks" }); };
  },
};

// =====================================================================
// 設定
// =====================================================================
App.views.settings = {
  title: "設定",
  async render(el) {
    const c = h(`<div class="card"><h2>保存先と詳細設定</h2><div class="stack" style="margin-top:10px"></div></div>`);
    $(".stack", c).appendChild(pathField("結果の保存先フォルダ", "out_root", "dir"));
    $(".stack", c).appendChild(pathField("上書き用の設定ファイル（YAML・任意）", "config", "file", { label: "YAML", pattern: "*.yaml" }));
    el.appendChild(c);
    el.appendChild(h(`<div class="card"><h2>詳細設定について</h2>
      <p class="desc">終端速度・シャトル検出の感度などは YAML ファイルで変えます。<code>configs/default.yaml</code> は直接書き換えず、変えたい項目だけを書いたファイルを作って上で指定してください。</p>
      <pre class="mono" style="background:#f6f8fa;padding:10px;border-radius:8px;margin:0">physics:
  terminal_velocity_mps: 6.9     # シャトルの終端速度 [m/s]
detection:
  baseline:
    diff_threshold: 12           # 小さくすると薄いシャトルも拾う
  tracknet:                      # TrackNetV3 を使う場合
    repo_dir: C:/Users/Public/Claude/TrackNetV3
    tracknet_ckpt: C:/Users/Public/Claude/TrackNetV3/ckpts/TrackNet_best.pt</pre></div>`));
  },
};
