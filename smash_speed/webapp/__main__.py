"""``python -m smash_speed.webapp`` で起動する。起動に失敗したら outputs/gui_error.log に書く。

- ``--remote``：離れた端末（Tailscale でつないだスマホ・PC）からも使えるようにする（合言葉が必要）
- ``--set-passphrase``：その合言葉を決める（変える）
- ``--port 8765`` ・ ``--no-browser``

知らない引数（``--help`` も）では起動しない（以前は読み飛ばして既定のポートで起動していた）。
"""

import sys
import traceback

from smash_speed.config import PROJECT_ROOT
from smash_speed.webapp import parse_args


def _set_passphrase() -> None:
    from getpass import getpass

    from smash_speed.webapp.remote import MIN_PASSPHRASE, set_passphrase

    print(f"離れた端末からログインするときの合言葉を決めます（{MIN_PASSPHRASE} 文字以上。入力しても画面には出ません）")
    while True:
        a = getpass("合言葉：")
        if len(a) < MIN_PASSPHRASE:
            print(f"{MIN_PASSPHRASE} 文字以上にしてください")
            continue
        if getpass("もう一度：") != a:
            print("一致しません。もう一度入れてください")
            continue
        set_passphrase(a)
        print("保存しました（ログイン済みの端末はログアウトされます）")
        return


args = parse_args()
if args.set_passphrase:
    _set_passphrase()
    sys.exit(0)

try:
    from smash_speed.webapp.server import DEFAULT_PORT, main

    main(open_browser=not args.no_browser, port=args.port or DEFAULT_PORT, remote=args.remote)
except Exception:  # noqa: BLE001  pythonw ではエラーが見えないので必ず残す
    log = PROJECT_ROOT / "outputs" / "gui_error.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(traceback.format_exc(), encoding="utf-8")
    raise
