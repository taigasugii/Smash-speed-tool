"""離れた端末（このPC以外のPC・スマホ）からアプリを使うための入口の制限と合言葉。

``--remote`` で起動すると、サーバーは全部のネットワークで待ち受けるが、次の端末しか通さない。

- この PC 自身（127.0.0.1）：合言葉なしで使える
- Tailscale（自分の端末どうしをつなぐ VPN）の端末（100.64.0.0/10・fd7a:115c:a1e0::/48）：
  合言葉でログインしたブラウザだけ
- それ以外（家の Wi-Fi の他人の端末・インターネット）：すべて断る

合言葉はそのまま保存せず、PBKDF2 で変換した値だけを ``outputs/remote_access.json`` に保存する。
ログインしたブラウザには、合言葉から作った値ではなく、保存した乱数の秘密から作った値を Cookie で渡す。
合言葉を変えると秘密も作り直すので、ログイン済みの全端末がログアウトされる。
通信の暗号化は Tailscale（WireGuard）が行う。
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import secrets
import time
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, redirect, request

from smash_speed.config import PROJECT_ROOT

CONFIG_FILE = PROJECT_ROOT / "outputs" / "remote_access.json"
COOKIE = "smash_speed_auth"
COOKIE_DAYS = 30
PBKDF2_ITER = 200_000
MIN_PASSPHRASE = 8
MAX_FAILS, FAIL_WINDOW_S = 5, 600         # この時間内にこの回数まちがえたら、しばらく受け付けない

TAILNET = (ipaddress.ip_network("100.64.0.0/10"), ipaddress.ip_network("fd7a:115c:a1e0::/48"))
# ログインしなくても返すもの（ログイン画面の見た目）
PUBLIC_PATHS = ("/login", "/static/style.css")


def _hash(passphrase: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", passphrase.encode("utf-8"), salt, PBKDF2_ITER).hex()


def set_passphrase(passphrase: str, path: Path = CONFIG_FILE) -> None:
    """合言葉を決める（変える）。ログイン済みの端末はすべてログアウトされる。"""
    if len(passphrase) < MIN_PASSPHRASE:
        raise ValueError(f"合言葉は {MIN_PASSPHRASE} 文字以上にしてください")
    salt = secrets.token_bytes(16)
    data = {"salt": salt.hex(), "hash": _hash(passphrase, salt), "secret": secrets.token_hex(32),
            "iterations": PBKDF2_ITER}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")


def load(path: Path = CONFIG_FILE) -> dict[str, Any] | None:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return d if {"salt", "hash", "secret"} <= set(d) else None


def is_local(addr: str | None) -> bool:
    try:
        return ipaddress.ip_address(addr or "").is_loopback
    except ValueError:
        return False


def is_tailnet(addr: str | None) -> bool:
    try:
        ip = ipaddress.ip_address(addr or "")
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return any(ip in net for net in TAILNET if ip.version == net.version)


class RemoteAccess:
    """``--remote`` のときの入口の制限。``install`` でアプリに組み込む。"""

    def __init__(self, path: Path = CONFIG_FILE) -> None:
        self.path = path
        self.enabled = False
        self.fails: dict[str, list[float]] = {}

    # ------------------------------------------------------------------ 合言葉
    def _token(self, cfg: dict[str, Any]) -> str:
        return hmac.new(bytes.fromhex(cfg["secret"]), b"smash-speed-session", hashlib.sha256).hexdigest()

    def check_passphrase(self, passphrase: str) -> bool:
        cfg = load(self.path)
        if cfg is None:
            return False
        got = hashlib.pbkdf2_hmac("sha256", passphrase.encode("utf-8"), bytes.fromhex(cfg["salt"]),
                                  int(cfg.get("iterations", PBKDF2_ITER))).hex()
        return hmac.compare_digest(got, cfg["hash"])

    def logged_in(self) -> bool:
        cfg = load(self.path)
        v = request.cookies.get(COOKIE, "")
        return cfg is not None and bool(v) and hmac.compare_digest(v, self._token(cfg))

    def _locked(self, addr: str) -> bool:
        now = time.time()
        self.fails[addr] = [t for t in self.fails.get(addr, []) if now - t < FAIL_WINDOW_S]
        return len(self.fails[addr]) >= MAX_FAILS

    # ------------------------------------------------------------------ 組み込み
    def is_remote_client(self) -> bool:
        """今のリクエストが、この PC 以外の端末からか。"""
        return self.enabled and not is_local(request.remote_addr)

    def install(self, app: Flask) -> None:
        app.before_request(self._guard)
        app.add_url_rule("/login", "login", self._login, methods=["GET", "POST"])
        app.add_url_rule("/logout", "logout", self._logout, methods=["POST"])

    def _guard(self):
        if not self.enabled or is_local(request.remote_addr):
            return None
        if not is_tailnet(request.remote_addr):
            return Response("この端末からは使えません（Tailscale でつないだ端末だけが使えます）", 403,
                            mimetype="text/plain; charset=utf-8")
        if request.path in PUBLIC_PATHS or self.logged_in():
            return None
        if request.path.startswith("/api/") or request.path == "/file":
            return jsonify({"ok": False, "error": "ログインしてください（ページを読み込み直してください）"}), 401
        return redirect("/login")

    def _login(self):
        if request.method == "GET":
            return _page()
        addr = request.remote_addr or ""
        if self._locked(addr):
            return _page("何度もまちがえたため、10 分ほど受け付けません。"), 429
        if not self.check_passphrase(request.form.get("passphrase", "")):
            self.fails.setdefault(addr, []).append(time.time())
            return _page("合言葉がちがいます。"), 401
        self.fails.pop(addr, None)
        resp = redirect("/")
        cfg = load(self.path)
        resp.set_cookie(COOKIE, self._token(cfg), max_age=COOKIE_DAYS * 86400, httponly=True, samesite="Strict")
        return resp

    def _logout(self):
        resp = redirect("/login")
        resp.delete_cookie(COOKIE)
        return resp


def _page(message: str = "") -> Response:
    msg = f'<p class="callout bad" style="margin:0 0 12px">{message}</p>' if message else ""
    html = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>smash-speed ログイン</title>
<link rel="stylesheet" href="/static/style.css"></head>
<body style="display:grid;place-items:center;min-height:100vh;padding:16px">
<form method="post" action="/login" class="card" style="width:100%;max-width:360px">
<h2 style="margin-top:0">smash-speed</h2>
<p class="desc">自宅のデスクトップで動いているアプリです。決めた合言葉を入れてください。</p>{msg}
<input class="input" type="password" name="passphrase" autocomplete="current-password" autofocus
 style="width:100%;margin-bottom:12px" placeholder="合言葉">
<button class="btn primary block" type="submit">ログイン</button></form></body></html>"""
    return Response(html, mimetype="text/html; charset=utf-8")


REMOTE = RemoteAccess()
