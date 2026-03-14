"""
Unit tests for Translategemma pipeline response patterns:
extract translation, take first option, normalize punctuation.
"""
import pytest

from pipeline_translategemma import (
    _extract_translation_from_content,
    _normalize_ai_added_punctuation,
    _strip_explanation_if_length_mismatch,
    _take_first_option,
)


# --- _extract_translation_from_content ---


def test_extract_plain_text():
    """Plain translation is returned as-is."""
    assert _extract_translation_from_content("Đây là thông báo") == "Đây là thông báo"
    assert _extract_translation_from_content("  Danh sách  ") == "Danh sách"


def test_extract_strip_assistant_response_prefix():
    """<|assistant_response|>\\n prefix is stripped."""
    raw = "<|assistant_response|>\nĐây là thông báo"
    assert _extract_translation_from_content(raw) == "Đây là thông báo"


def test_extract_strip_assistant_prefix():
    """<|assistant|>\\n prefix is stripped."""
    raw = "<|assistant|>\nGợi ý tiếp theo."
    assert _extract_translation_from_content(raw) == "Gợi ý tiếp theo."


def test_extract_strip_start_prefix():
    """<|start|> prefix is stripped."""
    raw = "<|start|>Danh sách"
    assert _extract_translation_from_content(raw) == "Danh sách"


def test_extract_strip_model_prefix():
    """<|model|>\\n prefix is stripped."""
    raw = "<|model|>\nChất lượng hiệu ứng chiến đấu"
    assert _extract_translation_from_content(raw) == "Chất lượng hiệu ứng chiến đấu"


def test_extract_take_after_answer():
    """Content after <|answer|>\\n is taken; end tokens stripped."""
    raw = "Translate the text to Vietnamese.\n<|end|>\n<|answer|>\n\"Tiếng Trung giản thể\" (Simplified Chinese)"
    got = _extract_translation_from_content(raw)
    assert "<|answer|>" not in got
    assert "Tiếng Trung giản thể" in got


def test_extract_strip_assistant_reply_prefix():
    """<|assistant_reply|>\\n prefix is stripped."""
    raw = "<|assistant_reply|>\nTất cả"
    assert _extract_translation_from_content(raw) == "Tất cả"


def test_extract_strip_assistant_ans_prefix():
    """<|assistant_ans|>\\n prefix; then embedded JSON is inlined."""
    raw = '<|assistant_ans|>\n{"role": "assistant", "text": "\\[Khi nào]"}<color=#1E6DF2>Bão tuyết</color>'
    got = _extract_translation_from_content(raw)
    assert "<|assistant_ans|>" not in got
    assert "[Khi nào]" in got or "\\[Khi nào]" in got
    assert "Bão tuyết" in got


def test_extract_strip_assistant_answer_prefix():
    """<|assistant_answer|>\\n prefix is stripped."""
    raw = "<|assistant_answer|>\nTông phái Xîyáo Wújí"
    assert _extract_translation_from_content(raw) == "Tông phái Xîyáo Wújí"


def test_extract_strip_assistant_lang_prefix():
    """<|assistant_lang|>\\n prefix is stripped."""
    raw = '<|assistant_lang|>\n"Vật liệu tu luyện."'
    got = _extract_translation_from_content(raw)
    assert "Vật liệu tu luyện" in got


def test_extract_strip_assistant_lang_underscore_prefix():
    """<|assistant_lang_|>\\n prefix is stripped."""
    raw = "<|assistant_lang_|>\nViệc tặng đồ cho người khác có thể tăng mức độ thân thiết."
    assert _extract_translation_from_content(raw) == "Việc tặng đồ cho người khác có thể tăng mức độ thân thiết."


def test_extract_embedded_json_replaced():
    """Mixed content: <color>{\"answer\": \"x\"}</color> -> <color>x</color>."""
    raw = '<color=#CC0099>{"answer": "<|assistant|> Không rõ nghĩa."}</color>-<color=#0066FF>Nhân vật</color>'
    got = _extract_translation_from_content(raw)
    assert "Không rõ nghĩa" in got
    assert "<|assistant|>" not in got


def test_extract_strip_end_and_file_separator():
    """Trailing <|end|> and <|file_separator|> are stripped."""
    raw = "Lưu hình nền.\n<|file_separator|>"
    assert _extract_translation_from_content(raw) == "Lưu hình nền."
    raw2 = "Done\n<|end|>"
    assert _extract_translation_from_content(raw2) == "Done"


def test_extract_first_option_when_file_separator():
    """When multiple options separated by <|file_separator|>, first only."""
    raw = '"Lưu hình nền".\n<|file_separator|>\n"Lưu ảnh làm hình nền".\n<|file_separator|>\n"Đặt làm hình nền".'
    assert _extract_translation_from_content(raw) == '"Lưu hình nền".'


def test_extract_json_text_key():
    """JSON with \"text\" key is unwrapped."""
    raw = '{"role": "assistant", "text": "Chất lượng hiệu ứng chiến đấu."}'
    assert _extract_translation_from_content(raw) == "Chất lượng hiệu ứng chiến đấu."


def test_extract_json_translation_key():
    """JSON with \"translation\" key is unwrapped."""
    raw = '{"translation": "Tạm biệt."}'
    assert _extract_translation_from_content(raw) == "Tạm biệt."
    raw2 = '{"role": "assistant", "translation": "Chất lượng hiệu ứng chiến đấu."}'
    assert _extract_translation_from_content(raw2) == "Chất lượng hiệu ứng chiến đấu."


def test_extract_json_response_key():
    """JSON with \"response\" key is unwrapped."""
    raw = '{"response": "Tối giản"}'
    assert _extract_translation_from_content(raw) == "Tối giản"


def test_extract_json_content_key():
    """JSON with \"content\" key is unwrapped."""
    raw = '{"role": "assistant", "content": "Cài đặt âm thanh"}'
    assert _extract_translation_from_content(raw) == "Cài đặt âm thanh"


def test_extract_json_answer_key():
    """JSON with \"answer\" key is unwrapped."""
    raw = '{"answer": "Module địa phương."}'
    assert _extract_translation_from_content(raw) == "Module địa phương."


def test_extract_json_inside_markdown_code_block():
    """JSON wrapped in ```json ... ``` is unwrapped and parsed."""
    raw = '```json\n{\n  "translation": "-YuYi- Hủy bỏ mối quan hệ với NPC."\n}\n```'
    assert _extract_translation_from_content(raw) == "-YuYi- Hủy bỏ mối quan hệ với NPC."


def test_extract_json_text_as_dict_with_title():
    """JSON with \"text\" as object {\"title\": \"...\"} returns title string (\\\\n -> newline)."""
    raw = '{"role": "assistant", "text": {"title": "Tác giả: Vô Danh Cáo\\nNhãn: Chức năng"}}'
    got = _extract_translation_from_content(raw)
    assert "Tác giả: Vô Danh Cáo" in got
    assert "Nhãn: Chức năng" in got
    assert "\\n" not in got or "\n" in got


def test_extract_json_translations_plural():
    """JSON with \"translations\" (dict of term -> translation) is joined with 、."""
    raw = '''```json
{
  "translations": {
    "战斗": "Chiến đấu",
    "气运": "May mắn / Dịch vận",
    "英语": "Tiếng Anh"
  }
}
```'''
    got = _extract_translation_from_content(raw)
    assert "Chiến đấu" in got
    assert "May mắn" in got
    assert "Tiếng Anh" in got
    assert "、" in got
    assert got.count("、") == 2  # 3 items -> 2 separators


def test_extract_json_prefers_text_then_translation_then_response():
    """When multiple keys exist, order is text > translation > response."""
    raw = '{"response": "R", "translation": "T", "text": "X"}'
    assert _extract_translation_from_content(raw) == "X"
    raw2 = '{"response": "R", "translation": "T"}'
    assert _extract_translation_from_content(raw2) == "T"
    raw3 = '{"response": "R"}'
    assert _extract_translation_from_content(raw3) == "R"


def test_extract_invalid_json_returns_stripped_raw():
    """Invalid JSON returns the stripped string."""
    raw = "Not JSON at all"
    assert _extract_translation_from_content(raw) == "Not JSON at all"


def test_extract_empty_after_strip_returns_empty():
    """Whitespace-only or empty after strip returns \"\"."""
    assert _extract_translation_from_content("   ") == ""
    assert _extract_translation_from_content("<|assistant|>\n") == ""


# --- _take_first_option ---


def test_take_first_option_vietnamese():
    """'X hoặc Y' -> keep X only."""
    assert _take_first_option('"Xác nhận" hoặc "Được rồi"') == '"Xác nhận"'
    assert _take_first_option("A hoặc B") == "A"


def test_take_first_option_english():
    """'X or Y' -> keep X only."""
    assert _take_first_option('"Confirm" or "OK"') == '"Confirm"'


def test_take_first_option_no_separator_unchanged():
    """No ' hoặc ' or ' or ' leaves string unchanged."""
    assert _take_first_option("Single option") == "Single option"
    assert _take_first_option("") == ""


# --- _normalize_ai_added_punctuation ---


def test_normalize_strip_quotes_when_source_has_no_quotes():
    """Surrounding \" stripped when source contains no \"."""
    assert _normalize_ai_added_punctuation('"Bách Hoang Bảo Thư Chí"', "八荒博物志") == "Bách Hoang Bảo Thư Chí"
    assert _normalize_ai_added_punctuation('"Lưu hình nền"', "保存壁纸") == "Lưu hình nền"


def test_normalize_keep_quotes_when_source_has_quotes():
    """Quotes kept when source contains \" (so we do not strip)."""
    # Source has "; output would otherwise be stripped as single pair
    out = _normalize_ai_added_punctuation('"X"', 'source " quote')
    assert out == '"X"'


def test_normalize_strip_trailing_period_when_source_has_no_period():
    """Trailing . stripped when source does not end with . or 。."""
    assert _normalize_ai_added_punctuation("Đây là thông báo.", "这是公告") == "Đây là thông báo"
    assert _normalize_ai_added_punctuation("Done.", "确定") == "Done"


def test_normalize_keep_trailing_period_when_source_ends_with_period():
    """Trailing . kept when source ends with . or 。."""
    assert _normalize_ai_added_punctuation("Đây là câu.", "这是句子。") == "Đây là câu."
    assert _normalize_ai_added_punctuation("End.", "End.") == "End."


def test_normalize_empty_unchanged():
    """Empty translation is returned unchanged."""
    assert _normalize_ai_added_punctuation("", "source") == ""
    assert _normalize_ai_added_punctuation("  ", "source") == "  "


# --- _strip_explanation_if_length_mismatch ---


def test_strip_explanation_module():
    """Short source 模组 + long output with 'trong tiếng Việt có thể' -> keep only first part."""
    source = "模组"
    long_out = '"Module" trong tiếng Việt có thể được hiểu là:\n\n*   **Mô-đun:** Đây là cách dịch phổ biến nhất.'
    got = _strip_explanation_if_length_mismatch(long_out, source)
    assert "trong tiếng Việt" not in got
    assert got.strip().startswith('"Module"')


def test_strip_explanation_dong_tian():
    """Short source 洞天福地 + long output with 'có thể được dịch là' -> truncate before explanation."""
    source = "洞天福地"
    long_out = '"Động thiên phúc địa" có thể được dịch là:\n\n* **Vùng đất tươi đẹp**'
    got = _strip_explanation_if_length_mismatch(long_out, source)
    assert "có thể được dịch" not in got
    assert "Động thiên phúc địa" in got


def test_strip_explanation_no_op_when_short_output():
    """Long source + short translation -> no truncation."""
    source = "A" * 50
    short_out = "Short translation."
    assert _strip_explanation_if_length_mismatch(short_out, source) == short_out


def test_strip_explanation_no_op_when_long_source():
    """Long source (e.g. paragraph) -> do not treat as mismatch."""
    source = "A" * 100
    long_out = "B" * 500
    assert _strip_explanation_if_length_mismatch(long_out, source) == long_out


def test_strip_explanation_tam_dich():
    """Short source + output with '(tạm dịch: ...)' -> keep only part before it."""
    source = "八荒博物志"
    long_out = '"Bách Hoang Bảo Thư Chí" (tạm dịch: Khảo luận về các vùng đất hoang vu). Đây là một cuốn sách'
    got = _strip_explanation_if_length_mismatch(long_out, source)
    assert "(tạm dịch" not in got
    assert "Đây là" not in got
    assert "Bách Hoang Bảo Thư Chí" in got


def test_strip_explanation_simplified_chinese():
    """Short source + '(Simplified Chinese)' parenthetical -> truncate before it."""
    source = "简体中文"
    long_out = '"Tiếng Trung giản thể" (Simplified Chinese) and some extra explanation text so length triggers strip.'
    got = _strip_explanation_if_length_mismatch(long_out, source)
    assert "(Simplified Chinese)" not in got
    assert "Tiếng Trung giản thể" in got
