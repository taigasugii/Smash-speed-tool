"""``python -m smash_speed.app`` で GUI を起動する。起動に失敗したらダイアログとログファイルで知らせる。"""

import traceback

from smash_speed.config import PROJECT_ROOT

try:
    from smash_speed.app.main import main

    main()
except Exception:  # noqa: BLE001  pythonw ではエラーが見えないので必ず知らせる
    msg = traceback.format_exc()
    log = PROJECT_ROOT / "outputs" / "gui_error.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(msg, encoding="utf-8")
    try:
        import tkinter as tk
        from tkinter import messagebox

        r = tk.Tk()
        r.withdraw()
        messagebox.showerror("smash-speed", f"GUI の起動中にエラーが起きました。\n詳細: {log}\n\n{msg[-1500:]}")
    except Exception:  # noqa: BLE001
        pass
    raise
