// コートの図（上から見た図）：カメラ側を下にして描き、ランドマークを分かりやすい言葉で説明する
"use strict";

const Court = {
  // 世界座標 (x, y) → 図の座標（カメラが下に来るように θ 度回転）
  rot(x, y, theta) {
    switch (((theta % 360) + 360) % 360) {
      case 90: return [-y, x];
      case 180: return [-x, -y];
      case 270: return [y, -x];
      default: return [x, y];
    }
  },

  // カメラの位置と打つ人の側から、図の回転角を決める（world は x<0 が打つ人の側）
  // edge: カメラがある辺（図の上で 'bottom' | 'top' | 'left' | 'right'）、hitter: 打つ人のコート 'left' | 'right'
  thetaFrom(edge, hitter) {
    // 打つ人が右なら 180° 回して左に来るようにする（このとき辺も入れ替わる）
    const flip = { bottom: "top", top: "bottom", left: "right", right: "left" };
    const e = hitter === "right" ? flip[edge] : edge;
    // e は「打つ人が左」の図でのカメラの辺。世界座標（x 右・y 上）でカメラが下に来る回転角
    return { bottom: 0, top: 180, left: 90, right: 270 }[e];
  },

  // 線の呼び方：エンドライン＝バックバウンダリーライン（コートの端の短い辺）、サイドライン＝長い辺。
  // ダブルスのサイドラインがいちばん外、シングルスのサイドラインはその 46 cm 内側。
  TYPES: {
    back_corner: { title: "コートの角", what: "エンドラインと、ダブルスのサイドライン（いちばん外の長い線）が交わる角", hint: "corner" },
    back_singles: { title: "エンドライン × シングルスのサイドライン", what: "エンドラインと、シングルスのサイドライン（外から 2 本目の長い線）の交点", hint: "tee" },
    back_center: { title: "エンドラインの中央", what: "エンドラインとセンターラインの T 字の交点", hint: "teec" },
    long_service: { title: "ロングサービスラインの端", what: "エンドラインの 76 cm 内側の線（ダブルスのロングサービスライン）と、ダブルスのサイドラインの交点", hint: "tee" },
    long_service_singles: { title: "ロングサービスライン × シングルス", what: "ダブルスのロングサービスラインと、シングルスのサイドラインの交点", hint: "cross" },
    long_service_center: { title: "ロングサービスライン × センター", what: "ダブルスのロングサービスラインとセンターラインの交点", hint: "cross" },
    short_service_side: { title: "ショートサービスラインの端", what: "ネットから 1.98 m の線（ショートサービスライン）と、ダブルスのサイドラインの交点", hint: "tee" },
    short_service_singles: { title: "ショートサービスライン × シングルス", what: "ショートサービスラインと、シングルスのサイドラインの交点", hint: "cross" },
    short_service_center: { title: "ショートサービスラインの中央", what: "ショートサービスラインとセンターラインの T 字の交点", hint: "teec" },
    post_base: { title: "ネットポストの根元", what: "ネットを張る支柱が床に立っている所", hint: "postbase" },
    post_top: { title: "ネットポストの上端", what: "ネットを張る支柱のいちばん上（高さ 1.55 m）", hint: "posttop" },
    net_top_center: { title: "ネットの中央の上端", what: "ネットの白帯の上の縁の、左右の真ん中（高さ 1.524 m）", hint: "net" },
  },
  EXTRA: ["long_service_singles", "long_service_center", "short_service_singles", "back_center"],

  typeOf(name) {
    const keys = Object.keys(this.TYPES).sort((a, b) => b.length - a.length);
    return keys.find((k) => name.startsWith(k)) || name;
  },

  where(x, y, theta) {
    const [rx, ry] = this.rot(x, y, theta);
    const hz = rx < -0.3 ? "左" : rx > 0.3 ? "右" : "";
    const dp = ry < -0.3 ? "手前" : ry > 0.3 ? "奥" : "";
    return [dp, hz].filter(Boolean).join("・") || "中央";
  },

  describe(lm, theta) {
    const type = this.typeOf(lm.name);
    const T = this.TYPES[type] || { title: lm.name, what: "", hint: "cross" };
    const [x, y, z] = lm.xyz;
    const side = Math.abs(x) < 1e-6 ? "ネットの位置" : x < 0 ? "打つ人の側のコート" : "相手側のコート";
    return { type, title: `${this.where(x, y, theta)}の${T.title}`, what: T.what, side, hint: T.hint, height: z > 0,
             where: `カメラから見て ${this.where(x, y, theta)}` };
  },

  // 案内する順番：どのカメラ位置でも映りやすく見分けやすい「ネットの近く」から始め、奥へ向かう。
  // 同じ種類の中では、手前から奥・左から右の順。
  GROUP: { post_base: 0, post_top: 0, net_top_center: 0, short_service_side: 1, short_service_center: 1, short_service_singles: 1,
           long_service: 2, long_service_center: 2, long_service_singles: 2, back_corner: 3, back_singles: 3, back_center: 3 },
  order(landmarks, theta, includeExtra = false) {
    const list = landmarks.filter((lm) => includeExtra || !this.EXTRA.includes(this.typeOf(lm.name)));
    const key = (lm) => {
      const [rx, ry] = this.rot(lm.xyz[0], lm.xyz[1], theta);
      const t = this.typeOf(lm.name);
      return [this.GROUP[t] ?? 4, (t === "post_top" || t === "net_top_center") ? 1 : 0, Math.round(ry * 2) / 2, rx];
    };
    return list.sort((a, b) => { const ka = key(a), kb = key(b); return ka[0] - kb[0] || ka[1] - kb[1] || ka[2] - kb[2] || ka[3] - kb[3]; });
  },

  /**
   * 上から見たコートの図（SVG 文字列）。
   * opt: { theta, landmarks, states: {name: 'done'|'skip'|'err'}, current, points: [{x,y,color,label}], line: [[x,y],[x,y]], camera: true }
   */
  svg(data, opt = {}) {
    const theta = opt.theta || 0;
    const c = data.court, L = c.half_length_m, Wd = c.half_width_doubles_m;
    // 回転後の外接範囲
    const vert = [[-L, -Wd], [L, -Wd], [L, Wd], [-L, Wd]].map(([x, y]) => this.rot(x, y, theta));
    const xs = vert.map((p) => p[0]), ys = vert.map((p) => p[1]);
    const pad = 1.4, minx = Math.min(...xs) - pad, maxx = Math.max(...xs) + pad, miny = Math.min(...ys) - pad - (opt.camera ? 1.6 : 0), maxy = Math.max(...ys) + pad;
    const S = 34, Wpx = (maxx - minx) * S, Hpx = (maxy - miny) * S;
    const P = (x, y) => { const [rx, ry] = this.rot(x, y, theta); return [((rx - minx) * S).toFixed(1), ((maxy - ry) * S).toFixed(1)]; };
    let s = `<svg class="diagram" viewBox="0 0 ${Wpx.toFixed(0)} ${Hpx.toFixed(0)}" xmlns="http://www.w3.org/2000/svg">`;
    // 床（打つ人の側を薄いオレンジ）
    const poly = (pts, fill) => `<polygon points="${pts.map(([x, y]) => P(x, y).join(",")).join(" ")}" fill="${fill}"/>`;
    s += poly([[-L, -Wd], [L, -Wd], [L, Wd], [-L, Wd]], "#e6f2ef");
    s += poly([[-L, -Wd], [0, -Wd], [0, Wd], [-L, Wd]], "#fdebd8");
    for (const seg of data.segments) {
      if (seg.kind !== "line") continue;
      const [x1, y1] = P(seg.a[0], seg.a[1]), [x2, y2] = P(seg.b[0], seg.b[1]);
      s += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#0f766e" stroke-width="2"/>`;
    }
    const [n1x, n1y] = P(0, -Wd - 0.15), [n2x, n2y] = P(0, Wd + 0.15);
    s += `<line x1="${n1x}" y1="${n1y}" x2="${n2x}" y2="${n2y}" stroke="#db2777" stroke-width="4" stroke-linecap="round"/>`;
    // 打つ人 / 相手 のラベル
    const [hx, hy] = P(-L / 2, 0), [ox, oy] = P(L / 2, 0);
    s += `<text x="${hx}" y="${hy}" text-anchor="middle" font-size="15" fill="#c2410c" font-weight="700" opacity=".75">打つ人の側</text>`;
    s += `<text x="${ox}" y="${oy}" text-anchor="middle" font-size="15" fill="#0f766e" font-weight="700" opacity=".6">相手側</text>`;
    if (opt.line) {
      const [a, b] = opt.line.map(([x, y]) => P(x, y));
      s += `<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="#2563eb" stroke-width="3" stroke-dasharray="8 5"/>`;
    }
    for (const lm of opt.landmarks || []) {
      const [x, y] = P(lm.xyz[0], lm.xyz[1]);
      const st = (opt.states || {})[lm.name];
      const cur = opt.current === lm.name;
      const extra = this.EXTRA.includes(this.typeOf(lm.name));
      const fill = st === "done" ? "#16a34a" : st === "err" ? "#dc2626" : st === "skip" ? "#cbd5e1" : cur ? "#ef4444" : "#ffffff";
      const stroke = cur ? "#ef4444" : st === "done" ? "#15803d" : "#64748b";
      const r = cur ? 7 : extra ? 4 : 5.5;
      const high = lm.xyz[2] > 0;
      const dy = high ? (lm.name.startsWith("net_top") ? -14 : -14) : 0;
      s += `<g class="lm" data-name="${lm.name}">`;
      if (high) s += `<line x1="${x}" y1="${y}" x2="${x}" y2="${(+y + dy).toFixed(1)}" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="2 2"/>`;
      if (cur) s += `<circle class="pulse" cx="${x}" cy="${(+y + dy).toFixed(1)}" r="7"/>`;
      s += `<circle class="dot" cx="${x}" cy="${(+y + dy).toFixed(1)}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="2"/>`;
      if (high) s += `<rect x="${(+x - 4).toFixed(1)}" y="${(+y + dy - 4).toFixed(1)}" width="8" height="8" fill="none" stroke="${stroke}" stroke-width="1.2" transform="rotate(45 ${x} ${(+y + dy).toFixed(1)})"/>`;
      s += `<circle cx="${x}" cy="${(+y + dy).toFixed(1)}" r="12" fill="transparent"/></g>`;
    }
    for (const p of opt.points || []) {
      const [x, y] = P(p.x, p.y);
      s += `<circle cx="${x}" cy="${y}" r="8" fill="${p.color}" stroke="#fff" stroke-width="2"/>`;
      if (p.label) s += `<text x="${(+x + 12).toFixed(1)}" y="${(+y + 5).toFixed(1)}" font-size="14" font-weight="700" fill="${p.color}">${p.label}</text>`;
    }
    if (opt.camera) {
      const cx = Wpx / 2, cy = Hpx - 18;
      s += `<g transform="translate(${cx.toFixed(0)} ${cy.toFixed(0)})"><rect x="-17" y="-11" width="28" height="20" rx="4" fill="#1a1d23"/><polygon points="11,-5 20,-10 20,8 11,3" fill="#1a1d23"/><circle cx="-3" cy="-1" r="5" fill="#94a3b8"/>
        <text x="30" y="5" font-size="14" fill="#1a1d23" font-weight="700">カメラ</text></g>`;
    }
    return s + "</svg>";
  },

  // ラインの交点のどこをクリックするかの図
  hintSvg(kind) {
    const g = "#0f766e", lw = 9;
    const base = (inner) => `<svg width="96" height="78" viewBox="0 0 96 78" xmlns="http://www.w3.org/2000/svg"><rect width="96" height="78" rx="8" fill="#f0f7f5"/>${inner}</svg>`;
    const dot = (x, y) => `<circle cx="${x}" cy="${y}" r="5" fill="#ef4444" stroke="#fff" stroke-width="2"/>`;
    const out = `<text x="6" y="74" font-size="9" fill="#64748b">コートの外</text>`;
    switch (kind) {
      case "corner": return base(`<rect x="34" y="18" width="${lw}" height="60" fill="${g}"/><rect x="34" y="18" width="62" height="${lw}" fill="${g}"/>${dot(34, 18)}<text x="6" y="12" font-size="9" fill="#64748b">外側の角</text>`);
      case "tee": return base(`<rect x="30" y="0" width="${lw}" height="78" fill="${g}"/><rect x="30" y="34" width="66" height="${lw}" fill="${g}"/>${dot(30, 34)}${out}`);
      case "teec": return base(`<rect x="0" y="22" width="96" height="${lw}" fill="${g}"/><rect x="44" y="22" width="${lw}" height="56" fill="${g}"/>${dot(48.5, 22)}`);
      case "cross": return base(`<rect x="40" y="0" width="${lw}" height="78" fill="${g}"/><rect x="0" y="34" width="96" height="${lw}" fill="${g}"/>${dot(44.5, 38.5)}`);
      case "postbase": return base(`<rect x="0" y="60" width="96" height="18" fill="#e7d3b0"/><rect x="42" y="6" width="10" height="56" fill="#7c8a6a"/>${dot(47, 61)}`);
      case "posttop": return base(`<rect x="0" y="66" width="96" height="12" fill="#e7d3b0"/><rect x="42" y="12" width="10" height="56" fill="#7c8a6a"/>${dot(47, 12)}`);
      case "net": return base(`<rect x="4" y="22" width="88" height="30" fill="none" stroke="#475569" stroke-dasharray="3 3"/><rect x="4" y="20" width="88" height="5" fill="#f8fafc" stroke="#475569"/>${dot(48, 20)}`);
      default: return base(dot(48, 39));
    }
  },
};
