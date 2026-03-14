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
    def __init__(self, settings: Settings, cache: TranslationCache) -> None:
        self._settings = settings
        self._cache = cache
        self._client = httpx.AsyncClient(base_url=self._settings.lmstudio.base_url, timeout=60.0)

    async def translate_batch(self, texts: List[str], source: str, target: str) -> List[TranslationResult]:
        if not texts:
            return []

        source_lang = source or self._settings.default.source_lang
        target_lang = target or self._settings.default.target_lang
        detected = source_lang if source_lang != "auto" else None

        # Extract content segments per text (split by tags, then by newline); collect all for batch translation.
        per_text: List[Tuple[List[str], List[int], List[int], int, int]] = []
        all_segments: List[str] = []
        for text in texts:
            parts, segment_indices = _extract_content_and_parts(text)
            segments: List[str] = []
            segment_counts: List[int] = []
            for i in segment_indices:
                lines = parts[i].splitlines()
                segment_counts.append(len(lines))
                segments.extend(lines)
            start = len(all_segments)
            all_segments.extend(segments)
            per_text.append((parts, segment_indices, segment_counts, start, len(segments)))

        # #region agent log
        if per_text:
            p0 = per_text[0]
            start0, count0 = p0[3], p0[4]
            seg_previews = [all_segments[i][:55] for i in range(start0, min(start0 + 12, len(all_segments)))]
            _debug_log("b5af83", "H3", "translator.py:extract", "per_text segment_counts and first segments", {"segments_count": len(all_segments), "text0_start": start0, "text0_count": count0, "text0_segment_counts": p0[2], "first_segment_previews": seg_previews})
        # #endregion

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
            return [TranslationResult(text=text, detected_source_language=detected) for text in texts]

        keys = [self._cache.make_key(source_lang, target_lang, s) for s in all_segments]
        cached = self._cache.get_many(keys)

        # One pass: check cache existence, detect broken (zh-vi + value has Chinese), collect missing + broken_keys.
        is_zh_vi = source_lang and target_lang and source_lang.lower() == "zh" and target_lang.lower() == "vi"
        missing_indices: List[int] = []
        broken_keys: List[str] = []
        for i in range(len(all_segments)):
            if keys[i] not in cached:
                missing_indices.append(i)
            elif is_zh_vi and _contains_chinese(cached.get(keys[i], "")):
                broken_keys.append(keys[i])
                missing_indices.append(i)
            # else: cache valid, use it
        if broken_keys:
            self._cache.delete_many(broken_keys)

        missing_texts = [all_segments[i] for i in missing_indices]
        valid_cache = {i for i in range(len(all_segments)) if i not in missing_indices}

        # #region agent log
        cache_hit_sample = [(i, all_segments[i][:40], (cached.get(keys[i], "") or "")[:40]) for i in range(min(5, len(all_segments))) if i in valid_cache][:3]
        _debug_log("b5af83", "H2", "translator.py:cache_lookup", "cache hit sample", {"cache_hits": len(valid_cache), "missing_count": len(missing_indices), "cache_hit_sample": cache_hit_sample})
        # #endregion

        for idx in valid_cache:
            print(
                "[translate_batch] cache_hit",
                json.dumps(
                    {"index": idx, "key_preview": keys[idx][:80]},
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )

        all_translations: List[str] = [cached.get(keys[i], "") if i in valid_cache else "" for i in range(len(all_segments))]
        if missing_texts:
            print(
                "[translate_batch] cache_miss_batch",
                json.dumps({"missing_count": len(missing_texts)}, ensure_ascii=False),
                file=sys.stderr,
            )
            new_translations = await self._translate_with_lmstudio(
                missing_texts, source_lang, target_lang
            )
            for j, idx in enumerate(missing_indices):
                all_translations[idx] = new_translations[j]

            # After translate: validate (zh-vi no Chinese); retry broken up to max_retry_broken; cache only valid.
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
                retry_trans = await self._translate_with_lmstudio(retry_texts, source_lang, target_lang)
                next_retry: List[Tuple[int, str]] = []
                for k, (idx, _seg) in enumerate(retry_list):
                    all_translations[idx] = retry_trans[k]
                    if is_zh_vi and _contains_chinese(retry_trans[k]):
                        next_retry.append((idx, all_segments[idx]))
                    # else: OK, no retry
                retry_list = next_retry

            cache_updates: Dict[str, str] = {}
            for j, idx in enumerate(missing_indices):
                trans = all_translations[idx]
                if not (is_zh_vi and _contains_chinese(trans)):
                    cache_updates[keys[idx]] = trans
            self._cache.set_many(cache_updates)
        # #region agent log
        if per_text and all_segments and all_translations:
            start0, count0 = per_text[0][3], per_text[0][4]
            alignment = [{"idx": i, "seg": all_segments[i][:50], "trans": all_translations[i][:50]} for i in range(start0, min(start0 + 20, start0 + count0))]
            _debug_log("b5af83", "H2_H4", "translator.py:after_merge", "segment-to-translation alignment", {"start": start0, "count": count0, "alignment": alignment})
        # #endregion

        results: List[TranslationResult] = []
        for text, (parts, segment_indices, segment_counts, start, count) in zip(texts, per_text):
            slice_trans = all_translations[start : start + count]
            reassembled = _reassemble(parts, segment_indices, slice_trans, segment_counts)
            results.append(TranslationResult(text=reassembled, detected_source_language=detected))

        return results

    async def _translate_with_lmstudio(self, texts: List[str], source: str, target: str) -> List[str]:
        batch_cfg = self._settings.batch
        if source and target and source.lower() == "zh" and target.lower() == "vi" and getattr(batch_cfg, "zh_vi_max_size", None) is not None:
            max_size = batch_cfg.zh_vi_max_size
        else:
            max_size = batch_cfg.max_size
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

            batch_outputs = await self._call_lmstudio_batch(batch, source, target)
            # #region agent log
            _debug_log("b5af83", "H4", "translator.py:_translate_with_lmstudio", "batch in/out counts", {"batch_start": start - len(batch), "batch_size": len(batch), "batch_output_len": len(batch_outputs)})
            # #endregion
            outputs.extend(batch_outputs)

        return outputs

    async def _call_lmstudio_batch(self, batch: List[str], source: str, target: str) -> List[str]:
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
            return await self._call_translategemma(batch, source, target)

        if endpoint_type == "chat":
            return await self._call_lmstudio_chat(batch, source, target)
        if endpoint_type == "completion":
            return await self._call_lmstudio_completion(batch, source, target)
        # Fallback to chat if misconfigured.
        return await self._call_lmstudio_chat(batch, source, target)

    async def _call_lmstudio_chat(self, batch: List[str], source: str, target: str) -> List[str]:
        system_prompt = _build_system_prompt(self._settings, source, target)

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

    async def _call_lmstudio_completion(self, batch: List[str], source: str, target: str) -> List[str]:
        system_prompt = _build_system_prompt(self._settings, source, target)

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

    async def _call_translategemma(self, batch: List[str], source: str, target: str) -> List[str]:
        """
        Call Translategemma via LM Studio's OpenAI-compatible /chat/completions API,
        but use a custom content payload that carries source/target language codes.
        """
        system_prompt = _build_system_prompt(self._settings, source, target)

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
        _translator_service = TranslatorService(settings=settings, cache=cache)
    return _translator_service

