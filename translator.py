from __future__ import annotations

from dataclasses import dataclass
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx

# #region agent log
_DEBUG_LOG_PATH = Path(__file__).resolve().parent / "debug-b5af83.log"

def _debug_log(session_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    try:
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"sessionId": session_id, "hypothesisId": hypothesis_id, "location": location, "message": message, "data": data, "timestamp": int(time.time() * 1000)}, ensure_ascii=False) + "\n")
    except Exception:
        pass
# #endregion

from cache import TranslationCache
from config import Settings, get_settings
from session_context import SessionContextStore


LANGUAGE_NAMES: Dict[str, str] = {
    "zh": "Chinese",
    "vi": "Vietnamese",
    "en": "English",
}


def _language_name_from_code(code: str) -> str:
    return LANGUAGE_NAMES.get(code.lower(), code)


def _contains_chinese(text: str) -> bool:
    """True if text contains CJK Unified Ideographs (Chinese characters)."""
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _build_system_prompt(settings: Settings, source_lang: str, target_lang: str) -> str:
    """
    Build the system prompt using config templates and language codes.
    """
    source_lang = (source_lang or settings.default.source_lang).lower()
    target_lang = (target_lang or settings.default.target_lang).lower()

    key = f"{source_lang}-{target_lang}"
    templates = settings.prompts or {}
    template = templates.get(key) or templates.get("default")
    if not template:
        # Fallback generic template.
        template = (
            "You are a professional translation engine. Translate the user message "
            "from {source_lang_name} (language code: {source_lang_code}) to "
            "{target_lang_name} (language code: {target_lang_code}). "
            "Output only the translation."
        )

    return template.format(
        source_lang_code=source_lang,
        target_lang_code=target_lang,
        source_lang_name=_language_name_from_code(source_lang),
        target_lang_name=_language_name_from_code(target_lang),
    )


def _is_gemma_pipeline_model(settings: Settings) -> bool:
    """True if the model is a Gemma model (use Gemma pipeline) but not Translategemma."""
    name = (settings.lmstudio.model or "").lower()
    return "gemma" in name and "translategemma" not in name


def _build_system_prompt_gemma_3_12b(
    settings: Settings, source_lang: str, target_lang: str
) -> str:
    """Build system prompt for Gemma-3-12b using gemma_3_12b_{source}_{target} templates."""
    source_lang = (source_lang or settings.default.source_lang).lower()
    target_lang = (target_lang or settings.default.target_lang).lower()
    templates = settings.prompts or {}
    # Key: gemma_3_12b + from + to (underscores to match config keys e.g. gemma_3_12b_zh_vi).
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
    # Paragraph: last double newline
    i = chunk.rfind(_SLICE_PARAGRAPH)
    if i != -1:
        best = max(best, i + len(_SLICE_PARAGRAPH))
    # Line: last single newline
    i = chunk.rfind(_SLICE_LINE)
    if i != -1:
        best = max(best, i + 1)
    # Sentence end: last of . ? ! 。 ？ ！ …
    for c in _SLICE_SENTENCE:
        i = chunk.rfind(c)
        if i != -1:
            best = max(best, i + 1)
    # Clause: last of ; ， ；
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


@dataclass
class TranslationResult:
    text: str
    detected_source_language: Optional[str] = None


class TranslatorService:
    def __init__(
        self,
        settings: Settings,
        cache: TranslationCache,
        session_store: Optional[SessionContextStore] = None,
    ) -> None:
        self._settings = settings
        self._cache = cache
        self._session_store = session_store
        self._client = httpx.AsyncClient(base_url=self._settings.lmstudio.base_url, timeout=60.0)

    async def translate_batch(
        self,
        texts: List[str],
        source: str,
        target: str,
        session_id: Optional[str] = None,
    ) -> List[TranslationResult]:
        if not texts:
            return []
        source_lang = source or self._settings.default.source_lang
        target_lang = target or self._settings.default.target_lang
        if _is_gemma_pipeline_model(self._settings):
            return await self._translate_batch_gemma(texts, source_lang, target_lang, session_id)
        return await self._translate_batch_general(texts, source_lang, target_lang, session_id)

    async def _translate_batch_gemma(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        session_id: Optional[str],
    ) -> List[TranslationResult]:
        """Gemma-3-12b pipeline: cache hit (no validate), slice large input, translate parts with retry, post-process, cache set."""
        detected = source_lang if source_lang != "auto" else None
        session_context = ""
        if session_id and self._session_store and getattr(self._settings.session, "inject_context_into_prompt", False):
            session_context = self._session_store.get_context(session_id)
        gemma_cfg = getattr(self._settings, "gemma", None)
        max_slice = getattr(gemma_cfg, "max_slice_chars", 2000) if gemma_cfg else 2000
        max_retry = getattr(gemma_cfg, "max_retry_broken", 3) if gemma_cfg else 3
        is_zh_vi = source_lang and target_lang and source_lang.lower() == "zh" and target_lang.lower() == "vi"
        results: List[TranslationResult] = []
        for text in texts:
            key = self._cache.make_key(source_lang, target_lang, text)
            cached = self._cache.get_many([key])
            if key in cached:
                results.append(TranslationResult(text=cached[key], detected_source_language=detected))
                continue
            if not text.strip():
                results.append(TranslationResult(text=text, detected_source_language=detected))
                continue
            parts = _slice_text_by_chars(text, max_slice)
            if not parts:
                parts = [text]
            translated_parts: List[str] = []
            for part in parts:
                last_out = ""
                for _attempt in range(max_retry):
                    out_list = await self._call_gemma_3_12b([part], source_lang, target_lang, context=session_context)
                    last_out = out_list[0] if out_list else ""
                    if is_zh_vi and _contains_chinese(last_out):
                        continue
                    break
                translated_parts.append(last_out)
            full = "\n".join(translated_parts)
            full = _normalize_excess_newlines(full)
            self._cache.set_many({key: full})
            if session_id and self._session_store:
                self._session_store.append(session_id, [(text, full)])
            results.append(TranslationResult(text=full, detected_source_language=detected))
        return results

    async def _translate_batch_general(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        session_id: Optional[str],
    ) -> List[TranslationResult]:
        """General pipeline: pre-process to segments (extract tags, split lines), cache with validate, translate missing, retry broken, reassemble."""
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
                {"source_lang": source_lang, "target_lang": target_lang, "texts_count": len(texts), "segments_count": len(all_segments)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        if not all_segments:
            return [TranslationResult(text=text, detected_source_language=detected) for text in texts]
        session_context = ""
        if session_id and self._session_store and getattr(self._settings.session, "inject_context_into_prompt", False):
            session_context = self._session_store.get_context(session_id)
        keys = [self._cache.make_key(source_lang, target_lang, s) for s in all_segments]
        cached = self._cache.get_many(keys)
        is_zh_vi = source_lang and target_lang and source_lang.lower() == "zh" and target_lang.lower() == "vi"
        missing_indices: List[int] = []
        broken_keys: List[str] = []
        for i in range(len(all_segments)):
            if keys[i] not in cached:
                missing_indices.append(i)
            elif is_zh_vi and _contains_chinese(cached.get(keys[i], "")):
                broken_keys.append(keys[i])
                missing_indices.append(i)
        if broken_keys:
            self._cache.delete_many(broken_keys)
        missing_texts = [all_segments[i] for i in missing_indices]
        valid_cache = {i for i in range(len(all_segments)) if i not in missing_indices}
        all_translations: List[str] = [cached.get(keys[i], "") if i in valid_cache else "" for i in range(len(all_segments))]
        if missing_texts:
            new_translations = await self._translate_with_lmstudio(
                missing_texts, source_lang, target_lang, context=session_context
            )
            for j, idx in enumerate(missing_indices):
                all_translations[idx] = new_translations[j]
            max_retry = getattr(self._settings.batch, "max_retry_broken", 3)
            retry_list: List[Tuple[int, str]] = [
                (idx, all_segments[idx])
                for j, idx in enumerate(missing_indices)
                if is_zh_vi and _contains_chinese(new_translations[j])
            ]
            for _attempt in range(max_retry):
                if not retry_list:
                    break
                retry_texts = [seg for _idx, seg in retry_list]
                retry_trans = await self._translate_with_lmstudio(
                    retry_texts, source_lang, target_lang, context=session_context
                )
                next_retry: List[Tuple[int, str]] = []
                for k, (idx, _seg) in enumerate(retry_list):
                    all_translations[idx] = retry_trans[k]
                    if is_zh_vi and _contains_chinese(retry_trans[k]):
                        next_retry.append((idx, all_segments[idx]))
                retry_list = next_retry
            cache_updates = {
                keys[idx]: all_translations[idx]
                for idx in missing_indices
                if not (is_zh_vi and _contains_chinese(all_translations[idx]))
            }
            self._cache.set_many(cache_updates)
            if session_id and self._session_store:
                self._session_store.append(
                    session_id,
                    [(all_segments[idx], all_translations[idx]) for idx in missing_indices],
                )
        results = []
        for text, (parts, segment_indices, segment_counts, start, count) in zip(texts, per_text):
            slice_trans = all_translations[start : start + count]
            reassembled = _reassemble(parts, segment_indices, slice_trans, segment_counts)
            results.append(TranslationResult(text=reassembled, detected_source_language=detected))
        return results

    async def _translate_with_lmstudio(
        self,
        texts: List[str],
        source: str,
        target: str,
        context: str = "",
    ) -> List[str]:
        batch_cfg = self._settings.batch
        if source and target and source.lower() == "zh" and target.lower() == "vi" and getattr(batch_cfg, "zh_vi_max_size", None) is not None:
            max_size = batch_cfg.zh_vi_max_size
        else:
            max_size = batch_cfg.max_size
        max_size = max(1, max_size)
        max_chars = batch_cfg.max_chars

        outputs: List[str] = []
        start = 0
        while start < len(texts):
            batch: List[str] = []
            char_count = 0
            while start < len(texts) and len(batch) < max_size:
                next_text = texts[start]
                if char_count + len(next_text) > max_chars and batch:
                    break
                batch.append(next_text)
                char_count += len(next_text)
                start += 1

            batch_outputs = await self._call_lmstudio_batch(batch, source, target, context=context)
            # #region agent log
            _debug_log("b5af83", "H4", "translator.py:_translate_with_lmstudio", "batch in/out counts", {"batch_start": start - len(batch), "batch_size": len(batch), "batch_output_len": len(batch_outputs)})
            # #endregion
            outputs.extend(batch_outputs)

        return outputs

    async def _call_lmstudio_batch(
        self,
        batch: List[str],
        source: str,
        target: str,
        context: str = "",
    ) -> List[str]:
        model_name = self._settings.lmstudio.model.lower()
        endpoint_type = self._settings.lmstudio.endpoint_type

        print(
            "[_call_lmstudio_batch] deciding_backend",
            json.dumps(
                {
                    "model": self._settings.lmstudio.model,
                    "endpoint_type": endpoint_type,
                    "batch_size": len(batch),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )

        # If we are using a Translategemma model, use the custom payload.
        if "translategemma" in model_name:
            return await self._call_translategemma(batch, source, target, context=context)

        # Gemma (non-Translategemma): raw input, dedicated prompt, post-process excess newlines.
        if "gemma" in model_name and "translategemma" not in model_name:
            return await self._call_gemma_3_12b(batch, source, target, context=context)

        if endpoint_type == "chat":
            return await self._call_lmstudio_chat(batch, source, target, context=context)
        if endpoint_type == "completion":
            return await self._call_lmstudio_completion(batch, source, target, context=context)
        # Fallback to chat if misconfigured.
        return await self._call_lmstudio_chat(batch, source, target, context=context)

    async def _call_gemma_3_12b(
        self,
        batch: List[str],
        source: str,
        target: str,
        context: str = "",
    ) -> List[str]:
        """
        Call Gemma-3-12b with raw user input only (no numbering/instructions in user message).
        System prompt carries all instructions; post-process response to normalize excess newlines.
        One request per batch item for 1:1 output mapping.
        """
        system_prompt = _build_system_prompt_gemma_3_12b(
            self._settings, source, target
        )
        if context:
            system_prompt = context.rstrip() + "\n\n" + system_prompt

        gemma_cfg = getattr(self._settings, "gemma", None)
        temperature = getattr(gemma_cfg, "temperature", 1.0) if gemma_cfg else 1.0
        outputs: List[str] = []
        for text in batch:
            body = {
                "model": self._settings.lmstudio.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                "temperature": temperature,
            }
            print(
                "[_call_gemma_3_12b] request",
                json.dumps(
                    {
                        "url": f"{self._settings.lmstudio.base_url}/chat/completions",
                        "model": self._settings.lmstudio.model,
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )
            resp = await self._client.post("/chat/completions", json=body)
            resp.raise_for_status()
            data = resp.json()
            print(
                "[_call_gemma_3_12b] response",
                json.dumps(
                    {"raw_preview": str(data)[:200]},
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )
            content = data["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                content = str(content)
            content = _strip_source_arrow_target(content)
            outputs.append(_normalize_excess_newlines(content))
        return outputs

    async def _call_lmstudio_chat(
        self,
        batch: List[str],
        source: str,
        target: str,
        context: str = "",
    ) -> List[str]:
        system_prompt = _build_system_prompt(self._settings, source, target)
        if context:
            system_prompt = context.rstrip() + "\n\n" + system_prompt

        if len(batch) == 1:
            user_content = batch[0]
        else:
            numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(batch, start=1))
            # zh-vi: Gemma often echoes Chinese; prepend a short imperative so the model sees "translate, no Chinese" right before content.
            if source and target and source.lower() == "zh" and target.lower() == "vi":
                user_content = "Translate each line below to Vietnamese. One translation per line, same order. Do not output Chinese.\n\n" + numbered
            else:
                user_content = numbered

        body = {
            "model": self._settings.lmstudio.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }

        print(
            "[_call_lmstudio_chat] request",
            json.dumps(
                {
                    "url": f"{self._settings.lmstudio.base_url}/chat/completions",
                    "model": self._settings.lmstudio.model,
                    "batch_size": len(batch),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )

        resp = await self._client.post("/chat/completions", json=body)
        resp.raise_for_status()
        data = resp.json()
        print(
            "[_call_lmstudio_chat] response",
            json.dumps(
                {"raw_preview": str(data)[:200]},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        content = data["choices"][0]["message"]["content"]
        # Preserve empty lines so output order matches batch (do not filter by line.strip()).
        lines = content.splitlines()
        cleaned = [line.lstrip("0123456789.:- ").strip() for line in lines]

        # #region agent log
        _debug_log("b5af83", "H1", "translator.py:_call_lmstudio_chat", "response line count", {"len_batch": len(batch), "len_lines": len(lines), "len_cleaned": len(cleaned), "mismatch": len(cleaned) != len(batch)})
        # #endregion

        if len(cleaned) != len(batch):
            # Fallback: if mismatch, pad or truncate.
            if len(cleaned) < len(batch):
                cleaned.extend([""] * (len(batch) - len(cleaned)))
            else:
                cleaned = cleaned[: len(batch)]

        return cleaned

    async def _call_lmstudio_completion(
        self,
        batch: List[str],
        source: str,
        target: str,
        context: str = "",
    ) -> List[str]:
        system_prompt = _build_system_prompt(self._settings, source, target)
        if context:
            system_prompt = context.rstrip() + "\n\n" + system_prompt

        if len(batch) == 1:
            user_content = batch[0]
        else:
            numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(batch, start=1))
            if source and target and source.lower() == "zh" and target.lower() == "vi":
                user_content = "Translate each line below to Vietnamese. One translation per line, same order. Do not output Chinese.\n\n" + numbered
            else:
                user_content = numbered
        prompt = system_prompt + "\n\n" + user_content

        body = {
            "model": self._settings.lmstudio.model,
            "prompt": prompt,
        }

        print(
            "[_call_lmstudio_completion] request",
            json.dumps(
                {
                    "url": f"{self._settings.lmstudio.base_url}/completions",
                    "model": self._settings.lmstudio.model,
                    "batch_size": len(batch),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )

        resp = await self._client.post("/completions", json=body)
        resp.raise_for_status()
        data = resp.json()
        print(
            "[_call_lmstudio_completion] response",
            json.dumps(
                {"raw_preview": str(data)[:200]},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        content = data["choices"][0]["text"]
        # Preserve empty lines so output order matches batch (do not filter by line.strip()).
        lines = content.splitlines()
        cleaned = [line.lstrip("0123456789.:- ").strip() for line in lines]

        if len(cleaned) != len(batch):
            if len(cleaned) < len(batch):
                cleaned.extend([""] * (len(batch) - len(cleaned)))
            else:
                cleaned = cleaned[: len(batch)]

        return cleaned

    async def _call_translategemma(
        self,
        batch: List[str],
        source: str,
        target: str,
        context: str = "",
    ) -> List[str]:
        """
        Call Translategemma via LM Studio's OpenAI-compatible /chat/completions API,
        but use a custom content payload that carries source/target language codes.
        """
        system_prompt = _build_system_prompt(self._settings, source, target)
        if context:
            system_prompt = context.rstrip() + "\n\n" + system_prompt

        outputs: List[str] = []
        for text in batch:
            # Custom payload: include source/target language codes in the content block.
            payload = {
                "model": self._settings.lmstudio.model,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": text,
                                "source_lang_code": source,
                                "target_lang_code": target,
                            }
                        ],
                    },
                ],
                "temperature": 0.1,
                "max_tokens": 500,
            }

            print(
                "[_call_translategemma] request",
                json.dumps(
                    {
                        "url": f"{self._settings.lmstudio.base_url}/chat/completions",
                        "model": self._settings.lmstudio.model,
                        "source_lang": source,
                        "target_lang": target,
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )

            resp = await self._client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            print(
                "[_call_translategemma] response",
                json.dumps(
                    {"raw_preview": str(data)[:200]},
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )

            # If LM Studio returns an error object, surface it instead of caching it as a translation.
            if isinstance(data, dict) and "error" in data:
                raise RuntimeError(f"LM Studio Translategemma error: {data['error']}")

            # Expect OpenAI-style chat completion: choices[0].message.content
            try:
                content = data["choices"][0]["message"]["content"]
                if isinstance(content, str):
                    generated = content
                else:
                    # If content is a list of blocks, concatenate their text.
                    generated = " ".join(
                        block.get("text", "") for block in content if isinstance(block, dict)
                    )
            except (KeyError, IndexError, TypeError):
                # If the shape differs, fall back to stringifying the whole response.
                generated = str(data)

            outputs.append(generated)

        return outputs


_translator_service: Optional[TranslatorService] = None


def get_translator_service() -> TranslatorService:
    global _translator_service
    if _translator_service is None:
        settings = get_settings()
        cache = TranslationCache(settings)
        session_store = SessionContextStore(
            max_entries=settings.session.max_entries,
            max_chars=settings.session.max_chars,
            ttl_seconds=settings.session.ttl_seconds,
        )
        _translator_service = TranslatorService(
            settings=settings, cache=cache, session_store=session_store
        )
    return _translator_service

