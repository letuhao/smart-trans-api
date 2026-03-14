"""
Gemma (non-Translategemma) pipeline: slice by chars, call Gemma-3-12b with raw user input,
retry on zh-vi validation failure, post-process newlines, cache.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Awaitable, Callable, List, Optional, Tuple

from config import Settings

if TYPE_CHECKING:
    from cache import TranslationCache
    from session_context import SessionContextStore

from pipeline_general import _contains_chinese, _is_translation_acceptable_zh_vi


# For Gemma system prompt template formatting (avoid circular import with translator).
_LANGUAGE_NAMES: dict[str, str] = {
    "zh": "Chinese",
    "vi": "Vietnamese",
    "en": "English",
}


def _language_name_from_code(code: str) -> str:
    return _LANGUAGE_NAMES.get(code.lower(), code)


def _build_system_prompt_gemma_3_12b(
    settings: Settings, source_lang: str, target_lang: str
) -> str:
    """Build system prompt for Gemma-3-12b using gemma_3_12b_{source}_{target} templates."""
    source_lang = (source_lang or settings.default.source_lang).lower()
    target_lang = (target_lang or settings.default.target_lang).lower()
    templates = settings.prompts or {}
    prompt_key = f"gemma_3_12b_{source_lang}_{target_lang}"
    template = templates.get(prompt_key) or templates.get("gemma_3_12b_default")
    if not template:
        template = (
            "You are a professional translation engine. Translate the user message "
            "from {source_lang_name} to {target_lang_name}. Output only the translation."
        )
    return template.format(
        source_lang_code=source_lang,
        target_lang_code=target_lang,
        source_lang_name=_language_name_from_code(source_lang),
        target_lang_name=_language_name_from_code(target_lang),
    )


def _normalize_excess_newlines(text: str) -> str:
    """Collapse 3+ consecutive newlines to at most 2; strip leading/trailing whitespace."""
    normalized = re.sub(r"\n{3,}", "\n\n", text)
    return normalized.strip()


def _strip_source_arrow_target(content: str) -> str:
    """If model returned 'source -> translation' format, keep only the translation part."""
    if " -> " not in content:
        return content
    parts = content.split(" -> ", 1)
    if len(parts) != 2:
        return content
    before, after = parts[0].strip(), parts[1].strip()
    if not after:
        return content
    if _contains_chinese(before) and len(before) < 200:
        return after
    return content


# Sentence/paragraph end characters for smart slice (avoid breaking context).
_SLICE_PARAGRAPH = "\n\n"
_SLICE_LINE = "\n"
_SLICE_SENTENCE = ".?!。？！…"
_SLICE_CLAUSE = ";；,，"


def _last_break_position(chunk: str) -> int:
    """
    Return the position after the last "good" break in chunk (paragraph > line > sentence > clause).
    Position is 1-based end index (split after this index). Returns 0 if no break found.
    """
    best = 0
    i = chunk.rfind(_SLICE_PARAGRAPH)
    if i != -1:
        best = max(best, i + len(_SLICE_PARAGRAPH))
    i = chunk.rfind(_SLICE_LINE)
    if i != -1:
        best = max(best, i + 1)
    for c in _SLICE_SENTENCE:
        i = chunk.rfind(c)
        if i != -1:
            best = max(best, i + 1)
    for c in _SLICE_CLAUSE:
        i = chunk.rfind(c)
        if i != -1:
            best = max(best, i + 1)
    return best


def _slice_text_by_chars(text: str, max_chars: int) -> List[str]:
    """
    Split text into parts of at most max_chars, at natural boundaries to avoid breaking
    sentences or paragraphs. Prefer: paragraph (\\n\\n) > line (\\n) > sentence (.?!。？！…) > clause (;，；).
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return [text] if text.strip() else []
    parts: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end]
        if end == len(text):
            if chunk.strip():
                parts.append(chunk)
            break
        last_break = _last_break_position(chunk)
        if last_break > 0:
            cut = start + last_break
            part = text[start:cut]
            if part.strip():
                parts.append(part)
            start = cut
        else:
            if chunk.strip():
                parts.append(chunk)
            start = end
    return parts if parts else [text]


async def translate_batch_gemma(
    texts: List[str],
    source_lang: str,
    target_lang: str,
    session_id: Optional[str],
    settings: Settings,
    cache: "TranslationCache",
    session_store: Optional["SessionContextStore"],
    call_gemma_3_12b: Callable[
        [List[str], str, str, str], Awaitable[List[str]]
    ],
) -> List[Tuple[str, Optional[str]]]:
    """
    Gemma-3-12b pipeline: cache hit (no validate), slice large input by chars,
    translate parts with retry on zh-vi validation failure, post-process newlines, cache set.
    Returns list of (translated_text, detected_source_language).
    """
    detected = source_lang if source_lang != "auto" else None
    session_context = ""
    if session_id and session_store and getattr(
        settings.session, "inject_context_into_prompt", False
    ):
        session_context = session_store.get_context(session_id)
    gemma_cfg = getattr(settings, "gemma", None)
    max_slice = getattr(gemma_cfg, "max_slice_chars", 2000) if gemma_cfg else 2000
    max_retry = getattr(gemma_cfg, "max_retry_broken", 3) if gemma_cfg else 3
    is_zh_vi = (
        source_lang
        and target_lang
        and source_lang.lower() == "zh"
        and target_lang.lower() == "vi"
    )
    results: List[Tuple[str, Optional[str]]] = []
    for text in texts:
        key = cache.make_key(source_lang, target_lang, text)
        cached = cache.get_many([key])
        if key in cached:
            results.append((cached[key], detected))
            continue
        if not text.strip():
            results.append((text, detected))
            continue
        parts = _slice_text_by_chars(text, max_slice)
        if not parts:
            parts = [text]
        translated_parts: List[str] = []
        for part in parts:
            last_out = ""
            for _attempt in range(max_retry):
                out_list = await call_gemma_3_12b(
                    [part], source_lang, target_lang, session_context
                )
                last_out = out_list[0] if out_list else ""
                if is_zh_vi and not _is_translation_acceptable_zh_vi(
                    last_out, settings.validation
                ):
                    continue
                break
            translated_parts.append(last_out)
        full = "\n".join(translated_parts)
        full = _normalize_excess_newlines(full)
        cache.set_many({key: full})
        if session_id and session_store:
            session_store.append(session_id, [(text, full)])
        results.append((full, detected))
    return results
