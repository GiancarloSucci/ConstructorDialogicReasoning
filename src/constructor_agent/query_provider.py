from __future__ import annotations

from pathlib import Path
from typing import Protocol


class QueryProvider(Protocol):
    def get_query(self) -> str: ...


class InlineQueryProvider:
    def __init__(self, query: str) -> None:
        self.query = query

    def get_query(self) -> str:
        return self.query.strip()


class FileQueryProvider:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def get_query(self) -> str:
        if not self.path.exists():
            raise FileNotFoundError(f"Query file not found: {self.path}")
        return self.path.read_text(encoding="utf-8").strip()
