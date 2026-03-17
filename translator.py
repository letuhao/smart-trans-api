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
from config import Settings, ValidationSettings, get_settings
from session_context import SessionContextStore
from pipeline_translategemma import call_translategemma_v2, call_translategemma_with_json
from pipeline_general import (
    _is_translation_acceptable_zh_vi,
    translate_batch_general,
)
from pipeline_gemma import (
    _build_system_prompt_gemma_3_12b,
    _normalize_excess_newlines,
    _slice_text_by_chars,
    _strip_source_arrow_target,
    translate_batch_gemma,
    translate_batch_gemma_v2,
)
from pipeline_deepseek import (
    build_system_prompt_deepseek,
    build_user_prompt_deepseek,
    strip_think_block,
    strip_translation_artifacts,
    translate_batch_deepseek,
)


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


def _is_gemma_pipeline_model(settings: Settings) -> bool:
    """True if the model is a Gemma model (use Gemma pipeline) but not Translategemma."""
    name = (settings.lmstudio.model or "").lower()
    return "gemma" in name and "translategemma" not in name


def _is_deepseek_pipeline_model(settings: Settings) -> bool:
    """True if the model is DeepSeek (use DeepSeek pipeline with zh→vi prompts)."""
    return "deepseek" in (settings.lmstudio.model or "").lower()


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
        model_name = (self._settings.lmstudio.model or "").lower()
        if "translategemma" in model_name:
            return await self._translate_batch_translategemma(texts, source_lang, target_lang, session_id)
        if _is_deepseek_pipeline_model(self._settings):
            return await self._translate_batch_deepseek(texts, source_lang, target_lang, session_id)
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
        """Delegate to Gemma pipeline v1 or v2 (by config); wrap (text, detected) results as TranslationResult."""
        version = getattr(self._settings.gemma, "version", "v2").lower()
        if version == "v2":
            raw = await translate_batch_gemma_v2(
                texts=texts,
                source_lang=source_lang,
                target_lang=target_lang,
                session_id=session_id,
                settings=self._settings,
                cache=self._cache,
                session_store=self._session_store,
                call_gemma_3_12b_v2=self._call_gemma_3_12b_v2,
            )
        else:
            raw = await translate_batch_gemma(
                texts=texts,
                source_lang=source_lang,
                target_lang=target_lang,
                session_id=session_id,
                settings=self._settings,
                cache=self._cache,
                session_store=self._session_store,
                call_gemma_3_12b=self._call_gemma_3_12b,
            )
        return [
            TranslationResult(text=t, detected_source_language=d) for t, d in raw
        ]

    async def _translate_batch_deepseek(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        session_id: Optional[str],
    ) -> List[TranslationResult]:
        """Delegate to DeepSeek pipeline; wrap (text, detected) results as TranslationResult."""
        raw = await translate_batch_deepseek(
            texts=texts,
            source_lang=source_lang,
            target_lang=target_lang,
            session_id=session_id,
            settings=self._settings,
            cache=self._cache,
            session_store=self._session_store,
            call_deepseek=self._call_deepseek,
        )
        return [
            TranslationResult(text=t, detected_source_language=d) for t, d in raw
        ]

    async def _translate_batch_translategemma(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        session_id: Optional[str],
    ) -> List[TranslationResult]:
        """Translategemma pipeline: v1 or v2 by config; cache, slice, retry, post-process (v1 or when v2 post_process=True)."""
        detected = source_lang if source_lang != "auto" else None
        session_context = ""
        if session_id and self._session_store and getattr(self._settings.session, "inject_context_into_prompt", False):
            session_context = self._session_store.get_context(session_id)
        version = getattr(self._settings.translategemma, "version", "v2").lower()
        call_translategemma = call_translategemma_v2 if version == "v2" else call_translategemma_with_json
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
                    out_list = await call_translategemma(
                        self._client,
                        self._settings,
                        [part],
                        source_lang,
                        target_lang,
                        session_context,
                        normalize_newlines_fn=_normalize_excess_newlines,
                        strip_source_arrow_fn=_strip_source_arrow_target,
                    )
                    last_out = out_list[0] if out_list else ""
                    if is_zh_vi and not _is_translation_acceptable_zh_vi(last_out, self._settings.validation):
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
        """Delegate to general pipeline; wrap (text, detected) results as TranslationResult."""
        raw = await translate_batch_general(
            texts=texts,
            source_lang=source_lang,
            target_lang=target_lang,
            session_id=session_id,
            settings=self._settings,
            cache=self._cache,
            session_store=self._session_store,
            translate_with_lmstudio=self._translate_with_lmstudio,
        )
        return [
            TranslationResult(text=t, detected_source_language=d) for t, d in raw
        ]

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

        # Translategemma uses its own pipeline (_translate_batch_translategemma), not _call_lmstudio_batch.

        # Gemma (non-Translategemma): use versioned pipeline callback.
        if "gemma" in model_name and "translategemma" not in model_name:
            version = getattr(self._settings.gemma, "version", "v2").lower()
            if version == "v2":
                return await self._call_gemma_3_12b_v2(batch, source, target, context=context)
            return await self._call_gemma_3_12b(batch, source, target, context=context)

        # DeepSeek: system + user prompt (zh→vi Vietnamese instructions), one request per item.
        if "deepseek" in model_name:
            return await self._call_deepseek(batch, source, target, context=context)

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

    async def _call_gemma_3_12b_v2(
        self,
        batch: List[str],
        source: str,
        target: str,
        context: str = "",
    ) -> List[str]:
        """
        Call Gemma-3-12b for v2 pipeline: user content is already a prompt template
        (<start_of_turn>user ... {user_input}<end_of_turn>). Optional system prompt
        comes from gemma.system_prompt; session context is injected into system when present.
        """
        gemma_cfg = getattr(self._settings, "gemma", None)
        system_prompt = ""
        if gemma_cfg:
            system_prompt = getattr(gemma_cfg, "system_prompt", "") or ""
            system_prompt = system_prompt.strip()
        if system_prompt and context:
            system_prompt = context.rstrip() + "\n\n" + system_prompt

        temperature = getattr(gemma_cfg, "temperature", 1.0) if gemma_cfg else 1.0
        outputs: List[str] = []
        for text in batch:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": text})
            body = {
                "model": self._settings.lmstudio.model,
                "messages": messages,
                "temperature": temperature,
            }
            print(
                "[_call_gemma_3_12b_v2] request",
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
                "[_call_gemma_3_12b_v2] response",
                json.dumps(
                    {"raw_preview": str(data)[:200]},
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )
            content = data["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                content = str(content)
            # Minimal post-process: strip optional source->target arrow and normalize newlines.
            content = _strip_source_arrow_target(content)
            outputs.append(_normalize_excess_newlines(content))
        return outputs

    async def _call_deepseek(
        self,
        batch: List[str],
        source: str,
        target: str,
        context: str = "",
    ) -> List[str]:
        """
        Call DeepSeek with zh→vi system/user prompts (Vietnamese instructions).
        User message: "Hãy dịch ra tiếng Việt với nội dung bên dưới:\n" + raw input.
        One request per batch item; post-process excess newlines.
        """
        system_prompt = build_system_prompt_deepseek(
            self._settings, source, target
        )
        if context:
            system_prompt = context.rstrip() + "\n\n" + system_prompt
        gemma_cfg = getattr(self._settings, "gemma", None)
        temperature = getattr(gemma_cfg, "temperature", 1.0) if gemma_cfg else 1.0
        outputs: List[str] = []
        for text in batch:
            user_content = build_user_prompt_deepseek(text, source, target, self._settings)
            body = {
                "model": self._settings.lmstudio.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "temperature": temperature,
            }
            print(
                "[_call_deepseek] request",
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
                "[_call_deepseek] response",
                json.dumps(
                    {"raw_preview": str(data)[:200]},
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )
            content = data["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                content = str(content)
            content = strip_think_block(content)
            content = strip_translation_artifacts(content)
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

