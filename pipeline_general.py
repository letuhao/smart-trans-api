"""
General translation pipeline: segment extraction (tags, lines), cache with zh-vi validation,
translate missing, retry broken (with attempt limit), reassemble.
"""
from __future__ import annotations

import json
import re
import sys
from typing import TYPE_CHECKING, Awaitable, Callable, List, Optional, Tuple

from config import Settings, ValidationSettings

if TYPE_CHECKING:
    from cache import TranslationCache
    from session_context import SessionContextStore


def _contains_chinese(text: str) -> bool:
    """True if text contains CJK Unified Ideographs (Chinese characters)."""
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _is_translation_acceptable_zh_vi(text: str, validation: ValidationSettings) -> bool:
    """
    Whether a zh→vi translation result is acceptable (no retry).
    Strict: any Chinese = not acceptable. Smart: accept if Chinese count or ratio below threshold.
    Single O(n) pass; no AI.
    """
    if not text:
        return True
    mode = (validation.mode or "smart").lower()
    if mode == "strict":
        return not _contains_chinese(text)
    # Smart: single pass count CJK codepoints
    chinese_count = 0
    for c in text:
        if "\u4e00" <= c <= "\u9fff":
            chinese_count += 1
    total = len(text)
    if total == 0:
        return True
    return (
        chinese_count <= validation.max_chinese_chars
        or (chinese_count / total) <= validation.max_chinese_ratio
    )


def _extract_content_and_parts(text: str) -> Tuple[List[str], List[int]]:
    """
    Split text by XML-like tags (<...>). Return (parts, segment_indices).
    parts: alternating [content, tag, content, tag, ...]; segment_indices are
    the indices in parts that are non-empty content (to be translated).
    """
    parts = re.split(r"(<[^>]*>)", text)
    segment_indices = [
        i
        for i in range(0, len(parts), 2)
        if i < len(parts) and parts[i].strip()
    ]
    return parts, segment_indices


def _reassemble(
    parts: List[str],
    segment_indices: List[int],
    translations: List[str],
    segment_counts: Optional[List[int]] = None,
) -> str:
    """Replace content at segment_indices with translations; leave tags and empty content as-is.
    If segment_counts is given, each content part was split into that many lines:
    take that many translations and join with '\\n' for that part.
    """
    if segment_counts is not None:
        assert len(segment_counts) == len(
            segment_indices
        ), "segment_counts length must match segment_indices"
    seg_set = set(segment_indices)
    trans_idx = 0
    seg_idx = 0
    out: List[str] = []
    for i, p in enumerate(parts):
        if i in seg_set:
            if segment_counts is not None and seg_idx < len(segment_counts):
                n = segment_counts[seg_idx]
                block = "\n".join(translations[trans_idx : trans_idx + n])
                trans_idx += n
                seg_idx += 1
                out.append(block)
            else:
                out.append(translations[trans_idx])
                trans_idx += 1
                if segment_counts is not None:
                    seg_idx += 1
        else:
            out.append(p)
    return "".join(out)


async def translate_batch_general(
    texts: List[str],
    source_lang: str,
    target_lang: str,
    session_id: Optional[str],
    settings: Settings,
    cache: "TranslationCache",
    session_store: Optional["SessionContextStore"],
    translate_with_lmstudio: Callable[
        [List[str], str, str, str], Awaitable[List[str]]
    ],
) -> List[Tuple[str, Optional[str]]]:
    """
    General pipeline: pre-process to segments (extract tags, split lines), cache with
    zh-vi validation, translate missing, retry broken (up to max_retry_broken attempts,
    then keep last translation), reassemble.
    Returns list of (translated_text, detected_source_language).
    """
    detected = source_lang if source_lang != "auto" else None
    per_text: List[Tuple[List[str], List[int], List[int], int, int]] = []
    all_segments: List[str] = []
    for text in texts:
        parts, segment_indices = _extract_content_and_parts(text)
        segments: List[str] = []
        segment_counts: List[int] = []
        for idx in segment_indices:
            lines = parts[idx].splitlines()
            segment_counts.append(len(lines))
            segments.extend(lines)
        start = len(all_segments)
        all_segments.extend(segments)
        per_text.append((parts, segment_indices, segment_counts, start, len(segments)))
    print(
        "[translate_batch] start",
        json.dumps(
            {
                "source_lang": source_lang,
                "target_lang": target_lang,
                "texts_count": len(texts),
                "segments_count": len(all_segments),
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
    )
    if not all_segments:
        return [(text, detected) for text in texts]
    session_context = ""
    if session_id and session_store and getattr(settings.session, "inject_context_into_prompt", False):
        session_context = session_store.get_context(session_id)
    keys = [cache.make_key(source_lang, target_lang, s) for s in all_segments]
    cached = cache.get_many(keys)
    is_zh_vi = (
        source_lang
        and target_lang
        and source_lang.lower() == "zh"
        and target_lang.lower() == "vi"
    )
    missing_indices: List[int] = []
    broken_keys: List[str] = []
    for i in range(len(all_segments)):
        if keys[i] not in cached:
            missing_indices.append(i)
        elif is_zh_vi and not _is_translation_acceptable_zh_vi(
            cached.get(keys[i], ""), settings.validation
        ):
            broken_keys.append(keys[i])
            missing_indices.append(i)
    if broken_keys:
        cache.delete_many(broken_keys)
    missing_texts = [all_segments[i] for i in missing_indices]
    valid_cache = {i for i in range(len(all_segments)) if i not in missing_indices}
    all_translations: List[str] = [
        cached.get(keys[i], "") if i in valid_cache else ""
        for i in range(len(all_segments))
    ]
    if missing_texts:
        new_translations = await translate_with_lmstudio(
            missing_texts, source_lang, target_lang, session_context
        )
        for j, idx in enumerate(missing_indices):
            all_translations[idx] = new_translations[j]
        max_retry = getattr(settings.batch, "max_retry_broken", 3)
        retry_list: List[Tuple[int, str]] = [
            (idx, all_segments[idx])
            for j, idx in enumerate(missing_indices)
            if is_zh_vi
            and not _is_translation_acceptable_zh_vi(
                new_translations[j], settings.validation
            )
        ]
        for _attempt in range(max_retry):
            if not retry_list:
                break
            retry_texts = [seg for _idx, seg in retry_list]
            retry_trans = await translate_with_lmstudio(
                retry_texts, source_lang, target_lang, session_context
            )
            next_retry: List[Tuple[int, str]] = []
            for k, (idx, _seg) in enumerate(retry_list):
                all_translations[idx] = retry_trans[k]
                if is_zh_vi and not _is_translation_acceptable_zh_vi(
                    retry_trans[k], settings.validation
                ):
                    next_retry.append((idx, all_segments[idx]))
            retry_list = next_retry
        cache_updates = {
            keys[idx]: all_translations[idx]
            for idx in missing_indices
            if not (
                is_zh_vi
                and not _is_translation_acceptable_zh_vi(
                    all_translations[idx], settings.validation
                )
            )
        }
        cache.set_many(cache_updates)
        if session_id and session_store:
            session_store.append(
                session_id,
                [(all_segments[idx], all_translations[idx]) for idx in missing_indices],
            )
    results: List[Tuple[str, Optional[str]]] = []
    for text, (parts, segment_indices, segment_counts, start, count) in zip(
        texts, per_text
    ):
        slice_trans = all_translations[start : start + count]
        reassembled = _reassemble(parts, segment_indices, slice_trans, segment_counts)
        results.append((reassembled, detected))
    return results
