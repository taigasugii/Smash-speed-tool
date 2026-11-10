"""smash-speed のローカル Web アプリ（ブラウザで操作する GUI）。"""

from __future__ import annotations

import argparse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """``python -m smash_speed.webapp`` の引数。知らない引数は、起動せずに使い方を出して終わる。"""
    p = argparse.ArgumentParser(prog="python -m smash_speed.webapp",
                                description="smash-speed の GUI（この PC の中で動く Web アプリ）を起動する")
    p.add_argument("--remote", action="store_true",
                   help="離れた端末（Tailscale でつないだスマホ・PC）からも使えるようにする（合言葉が必要）")
    p.add_argument("--set-passphrase", action="store_true", help="離れた端末からログインするときの合言葉を決める（変える）")
    p.add_argument("--port", type=int, default=None, help="待ち受けるポート（既定 8765）")
    p.add_argument("--no-browser", action="store_true", help="起動してもブラウザを開かない")
    return p.parse_args(argv)
