"""テスト共通の設定。ffmpeg が PATH にない場合、winget の既定のインストール先も探す。"""

import os
import shutil
from pathlib import Path

if shutil.which("ffprobe") is None:
    base = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    for exe in base.glob("Gyan.FFmpeg*/*/bin/ffprobe.exe"):
        os.environ["PATH"] = str(exe.parent) + os.pathsep + os.environ["PATH"]
        break
