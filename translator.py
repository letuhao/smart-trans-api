from __future__ import annotations

from dataclasses import dataclass
import json
import sys
from typing import Dict, List, Optional

import httpx

from cache import TranslationCache
from config import Settings, get_settings


LANGUAGE_NAMES: Dict[str, str] = {
    "zh": "Chinese",
    "vi": "Vietnamese",
    "en": "English",
}


def _language_name_from_code(code: str) -> str:
    return LANGUAGE_NAMES.get(code.lower(), code)


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

        keys = [self._cache.make_key(source_lang, target_lang, t) for t in texts]
        cached = self._cache.get_many(keys)

        results: List[Optional[TranslationResult]] = [None] * len(texts)
        missing_indices: List[int] = []
        missing_texts: List[str] = []

        print(
            "[translate_batch] start",
            json.dumps(
                {
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "texts_count": len(texts),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )

        for idx, (k, t) in enumerate(zip(keys, texts)):
            if k in cached:
                print(
                    "[translate_batch] cache_hit",
                    json.dumps(
                        {"index": idx, "key": k, "cached_value_preview": cached[k][:80]},
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                )
                results[idx] = TranslationResult(text=cached[k], detected_source_language=source_lang if source_lang != "auto" else None)
            else:
                missing_indices.append(idx)
                missing_texts.append(t)

        if missing_texts:
            print(
                "[translate_batch] cache_miss_batch",
                json.dumps(
                    {
                        "missing_count": len(missing_texts),
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )
            new_translations = await self._translate_with_lmstudio(missing_texts, source_lang, target_lang)
            cache_updates = {}
            for idx, text, translated in zip(missing_indices, missing_texts, new_translations):
                results[idx] = TranslationResult(text=translated, detected_source_language=source_lang if source_lang != "auto" else None)
                cache_key = self._cache.make_key(source_lang, target_lang, text)
                cache_updates[cache_key] = translated
            self._cache.set_many(cache_updates)

        # At this point all results should be filled
        return [r for r in results if r is not None]

    async def _translate_with_lmstudio(self, texts: List[str], source: str, target: str) -> List[str]:
        max_size = self._settings.batch.max_size
        max_chars = self._settings.batch.max_chars

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
            user_content = "\n".join(f"{i}. {t}" for i, t in enumerate(batch, start=1))

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
        lines = [line.strip() for line in content.splitlines() if line.strip()]

        # If the model echoed numbering, strip leading digits and punctuation.
        cleaned: List[str] = []
        for line in lines:
            cleaned.append(line.lstrip("0123456789.:- ").strip())

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
            user_content = "\n".join(f"{i}. {t}" for i, t in enumerate(batch, start=1))
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
        lines = [line.strip() for line in content.splitlines() if line.strip()]

        cleaned: List[str] = []
        for line in lines:
            cleaned.append(line.lstrip("0123456789.:- ").strip())

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

