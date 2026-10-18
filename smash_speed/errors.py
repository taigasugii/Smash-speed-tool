"""利用者に原因と対処を伝えるための例外。"""

from __future__ import annotations


class AnalysisError(RuntimeError):
    """解析を続けられないときの例外。``hints`` は利用者がすること（画面にそのまま出す）。"""

    def __init__(self, message: str, hints: list[str] | None = None):
        super().__init__(message)
        self.message = message
        self.hints = list(hints or [])

    def report_lines(self) -> list[str]:
        """CLI・GUI に出す行（``[エラー]`` と ``[対処]``）。"""
        return [f"[エラー] {self.message}"] + [f"[対処] {h}" for h in self.hints]
