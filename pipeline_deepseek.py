"""
DeepSeek pipeline: slice by chars, call model with zh→vi-specific system/user prompts,
retry on zh-vi validation failure, post-process newlines, cache.
Same flow as Gemma; differs only in prompt format.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Awaitable, Callable, List, Optional, Tuple

from config import Settings

if TYPE_CHECKING:
    from cache import TranslationCache
    from session_context import SessionContextStore

from pipeline_general import _is_translation_acceptable_zh_vi
from pipeline_gemma import _normalize_excess_newlines, _slice_text_by_chars

# Fallback user prefix when not in config (zh-vi).
_DEFAULT_USER_PREFIX_ZH_VI = "Hãy dịch ra tiếng Việt với nội dung bên dưới:\n"

# Strip <think>...</think> blocks (DeepSeek R1 / chain-of-thought style output).
_RE_THINK_BLOCK = re.compile(r"\s*<think>[\s\S]*?</think>\s*", re.IGNORECASE)


def strip_think_block(text: str) -> str:
    """Remove <think>...</think> blocks from model output so only the final answer is returned."""
    if not text:
        return text
    cleaned = _RE_THINK_BLOCK.sub("", text)
    # Unclosed <think>: remove from first <think> to end of string.
    if "<think>" in cleaned:
        idx = cleaned.lower().find("<think>")
        cleaned = cleaned[:idx]
    return cleaned.strip()


# Strip "Dịch nghĩa: ..." explanation blocks (e.g. *(Dịch nghĩa: Tên sách...)*).
_RE_DICH_NGHIA = re.compile(r"\s*\*\(Dịch nghĩa:[\s\S]*?\)\*", re.IGNORECASE)


def strip_translation_artifacts(text: str) -> str:
    """
    Clean zh→vi model output:
    - If output is "Chinese\\n\\nDịch: <translation>", keep only <translation>.
    - Remove *(Dịch nghĩa: ...)* explanation blocks (can be multiline).
    """
    if not text:
        return text
    # Pattern 1: "original\\n\\nDịch: translation" -> keep only "translation"
    if "\n\nDịch: " in text:
        parts = text.split("\n\nDịch: ", 1)
        if len(parts) == 2:
            text = parts[1].strip()
    # Pattern 2: remove *(Dịch nghĩa: ...)* blocks
    text = _RE_DICH_NGHIA.sub("", text)
    return text.strip()


def _language_name_from_code(code: str) -> str:
    """For default prompt placeholders."""
    names = {"zh": "Chinese", "vi": "Vietnamese", "en": "English"}
    return names.get(code.lower(), code)


def build_system_prompt_deepseek(
    settings: Settings, source_lang: str, target_lang: str
) -> str:
    """System prompt for DeepSeek from config: deepseek_{source}_{target} or deepseek_default."""
    source_lang = (source_lang or settings.default.source_lang).lower()
    target_lang = (target_lang or settings.default.target_lang).lower()
    templates = settings.prompts or {}
    prompt_key = f"deepseek_{source_lang}_{target_lang}"
    template = templates.get(prompt_key) or templates.get("deepseek_default")
    if not template:
        return (
            f"Translate the user input from {source_lang} to {target_lang}. "
            "Output only the translation, no questions, no explanations, no notes. Preserve input structure."
        )
    if "{source_lang_name}" in template or "{target_lang_name}" in template:
        return template.format(
            source_lang_code=source_lang,
            target_lang_code=target_lang,
            source_lang_name=_language_name_from_code(source_lang),
            target_lang_name=_language_name_from_code(target_lang),
        )
    return template


def build_user_prompt_deepseek(
    text: str, source_lang: str, target_lang: str, settings: Optional[Settings] = None
) -> str:
    """User message for DeepSeek. Zh→vi: prefix from config + raw text; other pairs: raw text only."""
    source_lang = (source_lang or "").lower()
    target_lang = (target_lang or "").lower()
    if source_lang == "zh" and target_lang == "vi":
        prefix = _DEFAULT_USER_PREFIX_ZH_VI
        if settings and settings.prompts and "deepseek_user_prefix_zh_vi" in settings.prompts:
            prefix = settings.prompts.get("deepseek_user_prefix_zh_vi") or prefix
        return prefix + text
    return text


async def translate_batch_deepseek(
    texts: List[str],
    source_lang: str,
    target_lang: str,
    session_id: Optional[str],
    settings: Settings,
    cache: "TranslationCache",
    session_store: Optional["SessionContextStore"],
    call_deepseek: Callable[
        [List[str], str, str, str], Awaitable[List[str]]
    ],
) -> List[Tuple[str, Optional[str]]]:
    """
    DeepSeek pipeline: same as Gemma (cache, slice by chars, retry zh-vi, post-process);
    differs in prompt format (system + user with Vietnamese instructions for zh→vi).
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
                out_list = await call_deepseek(
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
