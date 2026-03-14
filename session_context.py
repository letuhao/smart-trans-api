"""
In-memory store of translation context per session for consistency across requests.
Used when client sends session_id; context is injected into the LM prompt and updated after each batch.
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple


class SessionContextStore:
    """
    Store (source, target) pairs per session_id. TTL and size limits apply.
    get_context returns a formatted string for prompt injection; append adds new pairs.
    """

    def __init__(self, max_entries: int = 100, max_chars: int = 4000, ttl_seconds: int = 3600) -> None:
        self._max_entries = max_entries
        self._max_chars = max_chars
        self._ttl_seconds = ttl_seconds
        # session_id -> (last_access_ts, list of (source, target))
        self._data: Dict[str, Tuple[float, List[Tuple[str, str]]]] = {}

    def get_context(self, session_id: str) -> str:
        """Return formatted previous translations for prompt, or empty string if expired/empty."""
        now = time.monotonic()
        if session_id not in self._data:
            return ""
        last_ts, pairs = self._data[session_id]
        if self._ttl_seconds > 0 and (now - last_ts) > self._ttl_seconds:
            del self._data[session_id]
            return ""
        if not pairs:
            return ""
        # Take last max_entries, then trim by max_chars (drop oldest lines).
        recent = pairs[-self._max_entries :] if len(pairs) > self._max_entries else pairs
        lines: List[str] = []
        total = 0
        for src, tgt in reversed(recent):
            line = f"{src} -> {tgt}\n"
            if total + len(line) > self._max_chars and lines:
                break
            lines.append(line)
            total += len(line)
        lines.reverse()
        if not lines:
            return ""
        return "Previous translations in this session (use for consistency):\n" + "".join(lines)

    def append(self, session_id: str, pairs: List[Tuple[str, str]]) -> None:
        """Append new (source, target) pairs and trim to limits."""
        if not pairs:
            return
        now = time.monotonic()
        if session_id not in self._data:
            self._data[session_id] = (now, [])
        last_ts, existing = self._data[session_id]
        self._data[session_id] = (now, existing + pairs)
        # Trim to max_entries (keep last N).
        _, full_list = self._data[session_id]
        if len(full_list) > self._max_entries:
            self._data[session_id] = (now, full_list[-self._max_entries :])
