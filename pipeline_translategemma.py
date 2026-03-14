"""
Translategemma pipeline: prompt builder and API call with JSON user content.
Input format: user message content = {"role": "user", "source_lang_code": "...", "target_lang_code": "...", "text": "..."}.
"""
from __future__ import annotations

import json
import re
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
    # input: config may use {input}; actual text is sent in user message JSON
    return template.format(
        source_lang_code=source_lang,
        target_lang_code=target_lang,
        source_lang_name=_language_name_from_code(source_lang),
        target_lang_name=_language_name_from_code(target_lang),
        input="(provided in user message)",
    )


# Sentinels to strip from model output before parse/use.
_ASSISTANT_MARKER = "<|assistant|>\n"
_ASSISTANT_RESPONSE_MARKER = "<|assistant_response|>\n"
_INSTRUCTION_MARKER = "<|instruction|>\n"
_MODEL_MARKER = "<|model|>\n"
_ANSWER_MARKER = "<|answer|>\n"
_ASSISTANT_REPLY_MARKER = "<|assistant_reply|>\n"
_ASSISTANT_ANS_MARKER = "<|assistant_ans|>\n"
_ASSISTANT_ANSWER_MARKER = "<|assistant_answer|>\n"
_ASSISTANT_LANG_MARKER = "<|assistant_lang|>\n"
_ASSISTANT_LANG_UNDERSCORE_MARKER = "<|assistant_lang_|>\n"
_TAKE_AFTER_MARKERS = (
    _ASSISTANT_MARKER,
    _ASSISTANT_RESPONSE_MARKER,
    _INSTRUCTION_MARKER,
    _MODEL_MARKER,
    _ANSWER_MARKER,
    _ASSISTANT_REPLY_MARKER,
    _ASSISTANT_ANS_MARKER,
    _ASSISTANT_ANSWER_MARKER,
    _ASSISTANT_LANG_MARKER,
    _ASSISTANT_LANG_UNDERSCORE_MARKER,
)
_START_TOKENS = ("<|assistant|>", "<|im_start|>assistant", "<|start|>", "<|model|>")
_END_TOKENS = ("<|end|>", "<|im_end|>", "<|endoftext|>", "<|file_separator|>", "<|output|>", "<|assistant|>", "<|answer|>")
_JSON_TRANSLATION_KEYS = ("text", "translation", "response", "content", "answer")

# Match JSON object containing one of our keys and capture the string value (handles \", \\, etc.).
_EMBEDDED_JSON_PATTERN = re.compile(
    r'\{\s*[^{}]*"(?:' + "|".join(_JSON_TRANSLATION_KEYS) + r')"\s*:\s*"((?:[^"\\]|\\.)*)"\s*[^{}]*\}',
    re.DOTALL,
)


def _unescape_json_string_value(s: str) -> str:
    """Unescape a JSON string value (\\, \", \\n, \\t, \\[, \\])."""
    out = []
    i = 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            n = s[i + 1]
            if n == "n":
                out.append("\n")
            elif n == "t":
                out.append("\t")
            elif n == '"':
                out.append('"')
            elif n == "\\":
                out.append("\\")
            elif n == "[":
                out.append("[")
            elif n == "]":
                out.append("]")
            else:
                out.append(n)
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def _replace_embedded_json(s: str) -> str:
    """Replace embedded JSON objects like {\"answer\": \"...\"} with just the value (for mixed tag+JSON content)."""
    if not s or "{" not in s:
        return s

    def repl(match):
        val = _unescape_json_string_value(match.group(1))
        val = val.replace("<|assistant|>", "").strip()
        return val

    return _EMBEDDED_JSON_PATTERN.sub(repl, s)


def _extract_translation_from_content(content: str) -> str:
    """
    Extract translation from content that may be raw text or JSON-wrapped,
    or wrapped in <|assistant|> / <|assistant_response|> / <|start|>, or multiple options separated by <|file_separator|>.
    Returns only the translation text (first option when multiple).
    """
    if not isinstance(content, str):
        content = str(content) if content is not None else ""
    stripped = content.strip()
    # Take content after last occurrence of <|assistant|>\n or <|assistant_response|>\n
    for marker in _TAKE_AFTER_MARKERS:
        if marker in stripped:
            idx = stripped.rfind(marker)
            stripped = stripped[idx + len(marker) :].lstrip()
            break
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
    # Replace embedded JSON (e.g. <color>{"answer": "..."}</color>) with just the value
    stripped = _replace_embedded_json(stripped)
    # Multiple options separated by <|file_separator|> -> keep first only
    if "<|file_separator|>" in stripped:
        stripped = stripped.split("<|file_separator|>")[0].strip()
    if not stripped:
        return ""
    # Strip markdown code block (e.g. ```json\n{...}\n```)
    if stripped.startswith("```"):
        idx = stripped.find("\n")
        if idx >= 0:
            stripped = stripped[idx + 1 :].strip()
        else:
            stripped = stripped.lstrip("`").strip()
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()
        stripped = stripped.strip()
    if not stripped:
        return ""
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            for key in _JSON_TRANSLATION_KEYS:
                if key not in parsed:
                    continue
                val = parsed[key]
                if isinstance(val, str):
                    return val
                if isinstance(val, dict):
                    if "title" in val and isinstance(val["title"], str):
                        return val["title"].replace("\\n", "\n")
                    parts = [v for v in val.values() if isinstance(v, str)]
                    if parts:
                        return "\n".join(parts)
                    return json.dumps(val, ensure_ascii=False)
            # Plural "translations": dict of term -> translation; join with 、
            if "translations" in parsed and isinstance(parsed["translations"], dict):
                return "、".join(str(v) for v in parsed["translations"].values())
    except (json.JSONDecodeError, TypeError):
        pass
    return stripped


def _take_first_option(translation: str) -> str:
    """When AI returns multiple options (e.g. "Xác nhận" hoặc "Được rồi"), keep first only."""
    if not translation:
        return translation
    for sep in (" hoặc ", " or "):
        if sep in translation:
            translation = translation.split(sep)[0].strip()
            break
    return translation


# Only strip "explanation" when source is short and output is disproportionately long.
_SOURCE_MAX_LEN_FOR_STRIP = 30
_TRANSLATION_MIN_LEN_FOR_STRIP = 50
_LENGTH_RATIO_THRESHOLD = 4.0

# Less-strict patterns: start of explanation (truncate before first match).
_EXPLANATION_PATTERNS = [
    re.compile(r"\s+\(tạm dịch\s*:", re.IGNORECASE),  # "(tạm dịch: ...)"
    re.compile(r"\s+trong\s+tiếng\s+Việt\s+có\s+thể", re.IGNORECASE),
    re.compile(r"\s+có\s+thể\s+được\s+(hiểu|dịch)\s+là", re.IGNORECASE),
    re.compile(r"\n\n\s*[\*\-]\s*", re.MULTILINE),  # bullet list
    re.compile(r"\s+Đây\s+là\s+(một\s+)?(cách\s+)?(bản\s+)?dịch", re.IGNORECASE),
    re.compile(r"\s+Tùy\s+thuộc\s+vào\s+ngữ\s+cảnh", re.IGNORECASE),
    re.compile(r"\s+hãy\s+chọn\s+bản\s+dịch", re.IGNORECASE),
    re.compile(r"\s+\(Simplified\s+Chinese\)", re.IGNORECASE),
    re.compile(r"\s+\(Traditional\s+Chinese\)", re.IGNORECASE),
]


def _strip_explanation_if_length_mismatch(translation: str, source: str) -> str:
    """
    When source is short but translation is very long (likely AI added explanation),
    truncate at the first "explanation" pattern so we keep only the core translation.
    Only runs when length ratio suggests explanation (avoid breaking valid long outputs).
    """
    if not translation or not source:
        return translation
    src_len = len(source.strip())
    out_len = len(translation)
    if src_len > _SOURCE_MAX_LEN_FOR_STRIP or out_len < _TRANSLATION_MIN_LEN_FOR_STRIP:
        return translation
    if out_len < src_len * _LENGTH_RATIO_THRESHOLD:
        return translation
    truncate_at = len(translation)
    for pat in _EXPLANATION_PATTERNS:
        m = pat.search(translation)
        if m and m.start() < truncate_at:
            truncate_at = m.start()
    if truncate_at >= len(translation):
        return translation
    translation = translation[:truncate_at].rstrip()
    if translation.endswith(":"):
        translation = translation[:-1].rstrip()
    return translation


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

    translategemma_cfg = getattr(settings, "translategemma", None)
    user_input_format = (
        getattr(translategemma_cfg, "user_input_format", "json").lower()
        if translategemma_cfg
        else "json"
    )
    use_json_content = user_input_format != "raw"

    outputs: List[str] = []
    for text in batch:
        if use_json_content:
            user_content = {
                "role": "user",
                "source_lang_code": source,
                "target_lang_code": target,
                "text": text,
            }
            # LM Studio requires 'content' to be a string or array of objects; serialize JSON.
            content_str = json.dumps(user_content, ensure_ascii=False)
        else:
            content_str = text
        gemma_cfg = getattr(settings, "gemma", None)
        temperature = getattr(gemma_cfg, "temperature", 0.1) if gemma_cfg else 0.1
        payload = {
            "model": settings.lmstudio.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_str},
            ],
            "temperature": temperature,
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

        generated = _take_first_option(generated)
        generated = _strip_explanation_if_length_mismatch(generated, text)
        generated = _normalize_ai_added_punctuation(generated, text)
        generated = strip_source_arrow_fn(generated)
        generated = normalize_newlines_fn(generated)
        outputs.append(generated)

    return outputs
