// 起動
"use strict";

(async () => {
  const s = await api.get("/api/state");
  App.state = s.state;
  App.landmarks = await api.get("/api/landmarks");
  if (App.state.video) {
    const r = await api.post("/api/video", { path: App.state.video, capture_fps: App.state.capture_fps });
    if (r.ok) { App.meta = r.meta; App.calib = r.calib; App.state = r.state; }
  }
  const start = location.hash.replace("#", "");
  go(App.views[start] ? start : "home");
})();
window.addEventListener("hashchange", () => { const v = location.hash.replace("#", ""); if (App.views[v] && v !== App.current) go(v); });
