"""
Translategemma pipeline: prompt builder and API call with JSON user content.
Input format: user message content = {"role": "user", "source_lang_code": "...", "target_lang_code": "...", "text": "..."}.
"""
from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Callable, List

if TYPE_CHECKING:
    from config import Settings

import httpx


def _language_name_from_code(code: str) -> str:
    names = {"zh": "Chinese", "vi": "Vietnamese", "en": "English"}
    return names.get(code.lower(), code)


def build_system_prompt_translategemma(
    settings: "Settings", source_lang: str, target_lang: str
) -> str:
    """Build system prompt for Translategemma from config translategemma_{source}_{target}."""
    source_lang = (source_lang or settings.default.source_lang).lower()
    target_lang = (target_lang or settings.default.target_lang).lower()
    templates = settings.prompts or {}
    key = f"{source_lang}_{target_lang}"
    prompt_key = f"translategemma_{key}"
    template = templates.get(prompt_key) or templates.get("translategemma_default")
    if not template:
        template = (
            "Translate from {source_lang_name} to {target_lang_name}. "
            "Output only the translation, preserve structure."
        )
    return template.format(
        source_lang_code=source_lang,
        target_lang_code=target_lang,
        source_lang_name=_language_name_from_code(source_lang),
        target_lang_name=_language_name_from_code(target_lang),
    )


# Sentinels to strip from model output before parse/use.
_ASSISTANT_MARKER = "<|assistant|>\n"
_START_TOKENS = ("<|assistant|>", "<|im_start|>assistant")
_END_TOKENS = ("<|end|>", "<|im_end|>", "<|endoftext|>", "<|file_separator|>")


def _extract_translation_from_content(content: str) -> str:
    """
    Extract translation from content that may be raw text or JSON-wrapped,
    or wrapped in <|instruction|>...<|end|><|assistant|>...<|file_separator|>.
    Returns only the translation text.
    """
    if not isinstance(content, str):
        content = str(content) if content is not None else ""
    stripped = content.strip()
    # Pattern 1: <|instruction|>...<|end|>\n<|assistant|>\n"translation"\n<|file_separator|> -> take after last <|assistant|>\n
    if _ASSISTANT_MARKER in stripped:
        idx = stripped.rfind(_ASSISTANT_MARKER)
        stripped = stripped[idx + len(_ASSISTANT_MARKER) :].lstrip()
    for token in _START_TOKENS:
        if stripped.startswith(token):
            stripped = stripped[len(token) :].lstrip()
            break
    while True:
        removed = False
        for token in _END_TOKENS:
            if stripped.endswith(token):
                stripped = stripped[: -len(token)].rstrip()
                removed = True
                break
        if not removed:
            break
    if not stripped:
        return ""
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict) and "text" in parsed and isinstance(parsed["text"], str):
            return parsed["text"]
    except (json.JSONDecodeError, TypeError):
        pass
    return stripped


def _normalize_ai_added_punctuation(translation: str, source: str) -> str:
    """
    Remove punctuation from translation only when the source does not have it
    (so we do not remove characters that were in the original).
    - Surrounding double quotes: strip only when source contains no ".
    - Trailing period: strip only when source does not end with . or 。
    """
    if not translation:
        return translation
    source_stripped = source.strip() if source else ""
    # Strip one pair of surrounding " only when output is fully wrapped and source has no ".
    if (
        '"' not in source_stripped
        and len(translation) >= 2
        and translation.startswith('"')
        and translation.endswith('"')
        and translation.count('"') == 2
    ):
        translation = translation[1:-1]
    # Strip trailing period only when source does not end with period/full stop.
    if translation.endswith(".") and not source_stripped.endswith((".", "。")):
        translation = translation[:-1]
    return translation


async def call_translategemma_with_json(
    client: httpx.AsyncClient,
    settings: "Settings",
    batch: List[str],
    source: str,
    target: str,
    context: str = "",
    *,
    normalize_newlines_fn: Callable[[str], str],
    strip_source_arrow_fn: Callable[[str], str],
) -> List[str]:
    """
    Call Translategemma with user content as JSON object:
    {"role": "user", "source_lang_code": source, "target_lang_code": target, "text": text}.
    """
    system_prompt = build_system_prompt_translategemma(settings, source, target)
    if context:
        system_prompt = context.rstrip() + "\n\n" + system_prompt

    outputs: List[str] = []
    for text in batch:
        user_content = {
            "role": "user",
            "source_lang_code": source,
            "target_lang_code": target,
            "text": text,
        }
        # LM Studio requires 'content' to be a string or array of objects; serialize JSON.
        content_str = json.dumps(user_content, ensure_ascii=False)
        payload = {
            "model": settings.lmstudio.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_str},
            ],
            "temperature": 0.1,
            "max_tokens": 500,
        }

        print(
            "[call_translategemma_with_json] request",
            json.dumps(
                {
                    "url": f"{settings.lmstudio.base_url}/chat/completions",
                    "model": settings.lmstudio.model,
                    "source_lang": source,
                    "target_lang": target,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )

        resp = await client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()

        print(
            "[call_translategemma_with_json] response",
            json.dumps({"raw_preview": str(data)[:200]}, ensure_ascii=False),
            file=sys.stderr,
        )

        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(f"LM Studio Translategemma error: {data['error']}")

        try:
            content = data["choices"][0]["message"]["content"]
            if isinstance(content, str):
                raw = content
            else:
                raw = " ".join(
                    block.get("text", "") for block in content if isinstance(block, dict)
                )
            generated = _extract_translation_from_content(raw)
        except (KeyError, IndexError, TypeError):
            generated = _extract_translation_from_content(str(data))

        generated = _normalize_ai_added_punctuation(generated, text)
        generated = strip_source_arrow_fn(generated)
        generated = normalize_newlines_fn(generated)
        outputs.append(generated)

    return outputs
