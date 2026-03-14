import asyncio
import dataclasses
from unittest.mock import patch

from translator import (
    _build_system_prompt,
    _slice_text_by_chars,
    _strip_source_arrow_target,
    get_translator_service,
)
from pipeline_general import _extract_content_and_parts, _reassemble
from config import get_settings


def test_slice_text_by_chars_short():
    """Text under limit returns one part."""
    out = _slice_text_by_chars("short", 100)
    assert out == ["short"]


def test_slice_text_by_chars_splits_at_sentence():
    """Long text splits at sentence boundary, not mid-word."""
    text = "First sentence. Second sentence. Third sentence."
    out = _slice_text_by_chars(text, 20)
    assert len(out) >= 2
    assert out[0].endswith(".")
    assert "First sentence." == out[0]
    assert out[1].strip().startswith("Second")


def test_slice_text_by_chars_splits_at_newline():
    """Long text splits at newline when present."""
    text = "Line one\nLine two\nLine three"
    out = _slice_text_by_chars(text, 12)
    assert len(out) >= 2
    assert out[0].endswith("\n") or "Line one" in out[0]


def test_slice_text_by_chars_chinese_sentence_end():
    """Chinese sentence end (。) is a break point."""
    text = "这是第一句。这是第二句。"
    out = _slice_text_by_chars(text, 8)
    assert len(out) >= 2
    assert out[0].endswith("。")


def test_strip_source_arrow_target():
    """Gemma 'source -> translation' format is stripped to translation only."""
    assert _strip_source_arrow_target("相机抖动 -> Lắc máy ảnh") == "Lắc máy ảnh"
    assert _strip_source_arrow_target("plain text") == "plain text"
    assert _strip_source_arrow_target("no arrow here") == "no arrow here"


def test_extract_content_and_parts_plain_text():
    """No tags: one part, one segment."""
    parts, segment_indices = _extract_content_and_parts("hello world")
    assert parts == ["hello world"]
    assert segment_indices == [0]


def test_extract_content_and_parts_with_tags():
    """Tags split content; only non-empty content is a segment."""
    text = "【Xiao】<color=#FFF>太</color><color=#DDD>极</color>卦"
    parts, segment_indices = _extract_content_and_parts(text)
    assert parts == [
        "【Xiao】",
        "<color=#FFF>",
        "太",
        "</color>",
        "",
        "<color=#DDD>",
        "极",
        "</color>",
        "卦",
    ]
    # Content at even indices: 0, 2, 4, 6, 8. Non-empty: 0, 2, 6, 8 (4 is empty)
    assert segment_indices == [0, 2, 6, 8]
    assert [parts[i] for i in segment_indices] == ["【Xiao】", "太", "极", "卦"]


def test_extract_content_and_parts_only_tags():
    """Only tags and empty content: no segments."""
    parts, segment_indices = _extract_content_and_parts("<a></a><b></b>")
    assert segment_indices == []
    assert "" in parts


def test_multiline_content_yields_multiple_segments_and_segment_counts():
    """Content with newlines is split into one segment per line; segment_counts match."""
    # Same logic as in translate_batch: split content by newline per part.
    text = "line1\nline2\nline3"
    parts, segment_indices = _extract_content_and_parts(text)
    assert parts == ["line1\nline2\nline3"]
    assert segment_indices == [0]
    segments = []
    segment_counts = []
    for i in segment_indices:
        lines = parts[i].splitlines()
        segment_counts.append(len(lines))
        segments.extend(lines)
    assert segments == ["line1", "line2", "line3"]
    assert segment_counts == [3]


def test_reassemble_plain():
    """One segment replaced."""
    parts = ["hello"]
    segment_indices = [0]
    translations = ["xin chào"]
    assert _reassemble(parts, segment_indices, translations) == "xin chào"


def test_reassemble_with_tags():
    """Content segments replaced; tags unchanged."""
    parts = ["【Xiao】", "<color=#FFF>", "太", "</color>", "卦"]
    segment_indices = [0, 2, 4]
    translations = ["【Tiểu】", "Thái", "Quái"]
    out = _reassemble(parts, segment_indices, translations)
    assert out == "【Tiểu】<color=#FFF>Thái</color>Quái"


def test_reassemble_with_segment_counts():
    """When segment_counts is given, each content part uses that many translations joined by newline."""
    # One content part split into 3 lines -> 3 segments
    parts = ["a\nb\nc"]
    segment_indices = [0]
    translations = ["A", "B", "C"]
    out = _reassemble(parts, segment_indices, translations, segment_counts=[3])
    assert out == "A\nB\nC"

    # Two content parts: first 2 lines, second 1 line
    parts = ["x", "<tag>", "p\nq", "</tag>", "z"]
    segment_indices = [0, 2, 4]
    segment_counts = [1, 2, 1]  # "x" -> 1, "p\nq" -> 2, "z" -> 1
    translations = ["X", "P", "Q", "Z"]
    out = _reassemble(parts, segment_indices, translations, segment_counts=segment_counts)
    assert out == "X<tag>P\nQ</tag>Z"


def test_translate_batch_markup_structure_preserved():
    """Integration: translate_batch with markup preserves tags; segments are translated."""
    async def _run():
        from unittest.mock import AsyncMock

        from cache import TranslationCache
        from translator import TranslatorService

        settings = get_settings()
        # Use a non-Gemma model so we exercise general pipeline (extract/reassemble, tag preservation).
        settings = dataclasses.replace(
            settings,
            lmstudio=dataclasses.replace(settings.lmstudio, model="local-model"),
        )
        cache = TranslationCache(settings)
        service = TranslatorService(settings=settings, cache=cache)

        # "【X】<t>太</t>卦" -> 3 segments (one per content part, no newlines). Use empty cache so all 3 come from mock.
        class FakeResponse:
            status_code = 200
            def raise_for_status(self): pass
            def json(self):
                return {"choices": [{"message": {"content": "L1\nL2\nL3"}}]}

        with patch.object(cache, "get_many", return_value={}):
            with patch.object(service._client, "post", new_callable=AsyncMock, return_value=FakeResponse()):
                results = await service.translate_batch(
                    texts=["【X】<t>太</t>卦"],
                    source="zh",
                    target="vi",
                )
        assert len(results) == 1
        out = results[0].text
        assert "<t>" in out and "</t>" in out
        assert "L1" in out and "L2" in out and "L3" in out

    asyncio.run(_run())


def test_translate_batch_multiline_preserves_newlines():
    """Integration: content with newlines is split into lines, translated, then rejoined."""
    async def _run():
        from unittest.mock import AsyncMock

        from cache import TranslationCache
        from translator import TranslatorService

        settings = get_settings()
        cache = TranslationCache(settings)
        service = TranslatorService(settings=settings, cache=cache)

        # "a\nb" -> 2 segments; mock returns two lines so we get two translations rejoined.
        class FakeResponse:
            status_code = 200
            def raise_for_status(self): pass
            def json(self):
                return {"choices": [{"message": {"content": "A\nB"}}]}

        with patch.object(cache, "get_many", return_value={}):
            with patch.object(service._client, "post", new_callable=AsyncMock, return_value=FakeResponse()):
                results = await service.translate_batch(
                    texts=["a\nb"],
                    source="zh",
                    target="vi",
                )
        assert len(results) == 1
        assert results[0].text == "A\nB"

    asyncio.run(_run())


def test_build_system_prompt_zh_vi_uses_direction_specific_template():
    settings = get_settings()

    prompt = _build_system_prompt(settings, "zh", "vi")

    assert "Chinese" in prompt and "Vietnamese" in prompt
    assert "source_lang_code" not in prompt  # placeholders should be formatted


def test_translate_chinese_to_vietnamese_end_to_end():
    async def _run():
        # Mock LM Studio response so the test does not depend on a running server.
        fake_response = {
            "choices": [
                {
                    "message": {
                        "content": "Hôm nay thời tiết rất đẹp, chúng ta hãy cùng đi dạo trong công viên."
                    }
                }
            ]
        }

        async def fake_post(*args, **kwargs):
            class FakeResponse:
                status_code = 200

                def raise_for_status(self):
                    return None

                def json(self):
                    return fake_response

            return FakeResponse()

        with patch("translator.httpx.AsyncClient.post", side_effect=fake_post):
            service = get_translator_service()

            long_chinese_paragraph = (
                "今天天气很好，我们一起去公园散步吧。"
                "公园里有很多花和树，小朋友们在草地上玩耍，老人们在长椅上聊天。"
                "远处还有人放风筝，颜色非常鲜艳。"
            )

            texts = [long_chinese_paragraph for _ in range(3)]

            result = await service.translate_batch(
                texts=texts,
                source="zh",
                target="vi",
            )

            # We only assert that the call pipeline works without raising and
            # returns the expected number of items; the exact content is owned
            # by the model.
            assert len(result) == len(texts)
            for item in result:
                assert isinstance(item.text, str)

    asyncio.run(_run())

