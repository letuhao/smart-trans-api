from unittest.mock import patch

from fastapi.testclient import TestClient

from main import app
from translator import TranslatorService


client = TestClient(app)


class FakeTranslator(TranslatorService):
    async def translate_batch(self, texts, source, target):
        # Echo back a simple deterministic "translation" for testing.
        from translator import TranslationResult

        return [TranslationResult(text=f"{t} -> {target}") for t in texts]


def test_get_translate_single_q():
    with patch("api.get_translator_service", return_value=FakeTranslator(None)):  # type: ignore[arg-type]
        resp = client.get(
            "/language/translate/v2",
            params={"q": "hello", "source": "en", "target": "vi"},
        )
    assert resp.status_code == 200
    data = resp.json()
    translations = data["data"]["translations"]
    assert len(translations) == 1


def test_get_translate_multiple_q():
    with patch("api.get_translator_service", return_value=FakeTranslator(None)):  # type: ignore[arg-type]
        resp = client.get(
            "/language/translate/v2",
            params=[("q", "foo"), ("q", "bar"), ("source", "en"), ("target", "vi")],
        )
    assert resp.status_code == 200
    data = resp.json()
    translations = data["data"]["translations"]
    assert len(translations) == 2


def test_get_translate_google_style_text_and_lang_codes():
    # Simulate a Google Translate–style URL:
    # ?sl=zh-CN&tl=vi&text=...
    with patch("api.get_translator_service", return_value=FakeTranslator(None)):  # type: ignore[arg-type]
        resp = client.get(
            "/language/translate/v2",
            params={
                "sl": "zh-CN",
                "tl": "vi",
                "text": "今天天气很好，我们一起去公园散步吧。",
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    translations = data["data"]["translations"]
    assert len(translations) == 1


