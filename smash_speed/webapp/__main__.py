"""``python -m smash_speed.webapp`` で起動する。起動に失敗したら outputs/gui_error.log に書く。

- ``--remote``：離れた端末（Tailscale でつないだスマホ・PC）からも使えるようにする（合言葉が必要）
- ``--set-passphrase``：その合言葉を決める（変える）
- ``--port=8765`` ・ ``--no-browser``
"""

import sys
import traceback

from smash_speed.config import PROJECT_ROOT


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


if "--set-passphrase" in sys.argv:
    _set_passphrase()
    sys.exit(0)

try:
    from smash_speed.webapp.server import DEFAULT_PORT, main

    port = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--port=")), DEFAULT_PORT))
    main(open_browser="--no-browser" not in sys.argv, port=port, remote="--remote" in sys.argv)
except Exception:  # noqa: BLE001  pythonw ではエラーが見えないので必ず残す
    log = PROJECT_ROOT / "outputs" / "gui_error.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(traceback.format_exc(), encoding="utf-8")
    raise
