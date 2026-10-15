"""``python -m smash_speed.webapp`` で起動する。起動に失敗したら outputs/gui_error.log に書く。"""

import sys
import traceback

from smash_speed.config import PROJECT_ROOT

try:
    from smash_speed.webapp.server import DEFAULT_PORT, main

    port = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--port=")), DEFAULT_PORT))
    main(open_browser="--no-browser" not in sys.argv, port=port)
except Exception:  # noqa: BLE001  pythonw ではエラーが見えないので必ず残す
    log = PROJECT_ROOT / "outputs" / "gui_error.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(traceback.format_exc(), encoding="utf-8")
    raise
