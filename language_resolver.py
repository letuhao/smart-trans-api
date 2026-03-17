"""
Resolve language code to (name, display_code) using language_codes.txt.
Used by Translategemma v2 to fill SOURCE_LANG, SOURCE_CODE, TARGET_LANG, TARGET_CODE in user prompt.
"""
from __future__ import annotations

import os
from pathlib import Path

_CODE_TO_NAME: dict[str, str] | None = None
_DEFAULT_PATH = Path(__file__).resolve().parent / "language_codes.txt"


def _load_codes(path: Path | None = None) -> dict[str, str]:
    """Load code -> name from tab-separated file. First occurrence wins."""
    global _CODE_TO_NAME
    if _CODE_TO_NAME is not None:
        return _CODE_TO_NAME
    p = path or Path(os.getenv("LANGUAGE_CODES_PATH", str(_DEFAULT_PATH)))
    result: dict[str, str] = {}
    if p.is_file():
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "\t" not in line:
                    continue
                code, name = line.split("\t", 1)
                code = code.strip()
                name = name.strip()
                if code and code not in result:
                    result[code] = name
    _CODE_TO_NAME = result
    return result


def resolve_lang(
    code: str,
    fallback_name: str,
    fallback_code: str,
    *,
    _path: Path | None = None,
) -> tuple[str, str]:
    """
    Resolve language code to (language_name, display_code).
    Tries exact match, then base code (e.g. zh-CN -> zh), then returns (fallback_name, fallback_code).
    """
    if not code or not code.strip():
        return (fallback_name, fallback_code)
    code = code.strip().lower()
    codes = _load_codes(_path)
    # Exact match
    if code in codes:
        return (codes[code], code)
    # Try base (part before first -)
    if "-" in code:
        base = code.split("-", 1)[0]
        if base in codes:
            return (codes[base], code)
    return (fallback_name, fallback_code)
