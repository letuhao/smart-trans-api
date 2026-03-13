from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

from config import Settings


class TranslationCache:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._data: Dict[str, str] = {}
        self._path = Path(self._settings.cache.persistent_file)
        self._load()

    def _load(self) -> None:
        if not self._path.is_file():
            return
        try:
            with self._path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                self._data.update({str(k): str(v) for k, v in raw.items()})
        except Exception:
            # Corrupted cache; start fresh.
            self._data = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False)

    @staticmethod
    def make_key(source_lang: str, target_lang: str, text: str) -> str:
        return f"{source_lang}|{target_lang}|{text}"

    def get_many(self, keys: Iterable[str]) -> Dict[str, str]:
        return {k: self._data[k] for k in keys if k in self._data}

    def set_many(self, entries: Dict[str, str]) -> None:
        if not entries:
            return
        self._data.update(entries)
        self._save()

