// まとめて解析：動画のフォルダの、まだ解析していない動画を順に解析する（ROADMAP v2 の L3）
"use strict";

App.views.queue = {
  title: "まとめて解析",
  async render(el) {
    const r = await api.get("/api/queue");
    const top = h(`<div class="card"><h2>新しい動画を、まとめて順に解析します</h2>
      <p class="desc">動画のフォルダの中で、まだ解析していない動画を選んで「選んだ動画を順に解析する」を押すと、1 本ずつ続けて解析します。
        基礎打ちの解析（打った数・時刻。コート合わせはいりません）と、その動画のコート合わせを保存してあればショットの一覧も作ります。
        止めるときは画面の下の「中止」。もう一度押すと、終わっていない動画から続けます。押したときだけ動きます。</p>
      <div id="dir"></div>
      <div class="row" style="flex-wrap:wrap;gap:12px;margin-top:8px">
        <button class="btn sm" id="reload">一覧を新しくする</button>
        <label class="row small"><input type="checkbox" id="model"> 練習させた見つけ方でシャトルを探す（モデルがある PC だけ）</label>
      </div></div>`);
    el.appendChild(top);
    $("#dir", top).appendChild(pathField("解析する動画のフォルダ", "videos_dir", "dir", { onPick: () => go("queue") }));
    $("#dir input", top).value = r.videos_dir || "";
    $("#reload", top).onclick = () => go("queue");
    const items = r.items || [];
    const nTodo = items.filter((x) => !x.done).length;
    const c = h(`<div class="card"><div class="row" style="margin-bottom:10px">
        <h2 style="margin:0;flex:1">動画（${items.length} 本・まだのもの ${nTodo} 本）</h2>
        <button class="btn sm ghost" id="all">まだのものを全部選ぶ</button>
        <button class="btn primary" id="go">選んだ動画を順に解析する</button></div>
      <div style="overflow:auto"><table class="tbl"><thead><tr><th></th><th>動画</th><th class="r">大きさ</th><th>日時</th>
        <th>基礎打ちの解析</th><th>コート合わせ</th><th>ショットの一覧</th></tr></thead><tbody></tbody></table></div>
      <p class="muted small" style="margin-top:8px">${r.error ? esc(r.error) : "解析した結果は「設定」の保存先（" + esc(r.out_root || "") + "）に入ります。"}</p></div>`);
    el.appendChild(c);
    const tb = $("tbody", c);
    if (!items.length) tb.appendChild(h(`<tr><td colspan="7" class="muted">動画がありません</td></tr>`));
    const badge = (ok, txt) => `<span class="badge ${ok ? "ok" : "warn"}">${txt}</span>`;
    for (const it of items) {
      const tr = h(`<tr><td><input type="checkbox" ${it.done ? "disabled" : "checked"}></td><td><b>${esc(it.name)}</b></td>
        <td class="r">${fmt(it.size_mb, 0)} MB</td><td class="muted small">${new Date(it.mtime * 1000).toLocaleString("ja-JP")}</td>
        <td>${badge(it.drill_done, it.drill_done ? "済み" : "まだ")}</td>
        <td>${it.has_calib ? badge(true, "あり") : '<span class="muted small">なし</span>'}</td>
        <td>${it.has_calib ? badge(it.shots_done, it.shots_done ? "済み" : "まだ") : '<span class="muted small">—</span>'}</td></tr>`);
      tr.dataset.path = it.path;
      tb.appendChild(tr);
    }
    $("#all", c).onclick = () => $$("tbody input:not(:disabled)", c).forEach((x) => { x.checked = true; });
    $("#go", c).onclick = () => {
      const paths = $$("tbody tr", c).filter((tr) => { const x = $("input", tr); return x && x.checked && !x.disabled; })
        .map((tr) => tr.dataset.path);
      if (!paths.length) return toast("解析する動画を選んでください", "bad");
      Task.start("queue", { paths, model: $("#model", top).checked }, (okk) => {
        toast(okk ? "まとめて解析が終わりました" : "止めたか、うまくいかなかった動画があります（ログを見てください）", okk ? "ok" : "bad");
        if (App.current === "queue") go("queue");
      }, `まとめて解析（${paths.length} 本）`);
    };
  },
};
