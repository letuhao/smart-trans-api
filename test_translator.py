import asyncio
from unittest.mock import patch

from translator import _build_system_prompt, get_translator_service
from config import get_settings


def test_build_system_prompt_zh_vi_uses_direction_specific_template():
    settings = get_settings()

    prompt = _build_system_prompt(settings, "zh", "vi")

    assert "Chinese-to-Vietnamese literary translator" in prompt
    assert "source_lang_code" not in prompt  # placeholders should be formatted
    assert "zh" in prompt
    assert "vi" in prompt


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

