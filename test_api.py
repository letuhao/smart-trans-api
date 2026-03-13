from fastapi.testclient import TestClient

from main import app
from translator import get_translator_service


client = TestClient(app)


class FakeTranslator:
    async def translate_batch(self, texts, source, target):
        # Echo back a simple deterministic "translation" for testing.
        from translator import TranslationResult

        return [TranslationResult(text=f"{t} -> {target}") for t in texts]


# Override the real dependency with our fake translator for all tests.
app.dependency_overrides[get_translator_service] = lambda: FakeTranslator()


def test_get_translate_single_q():
    resp = client.get(
        "/language/translate/v2",
        params={"q": "hello", "source": "en", "target": "vi"},
    )
    assert resp.status_code == 200
    assert resp.text == "hello -> vi"


def test_get_translate_multiple_q():
    resp = client.get(
        "/language/translate/v2",
        params=[("q", "foo"), ("q", "bar"), ("source", "en"), ("target", "vi")],
    )
    assert resp.status_code == 200
    assert resp.text == "foo -> vi\nbar -> vi"


def test_get_translate_google_style_text_and_lang_codes():
    # Simulate a Google Translate–style URL:
    # ?sl=zh-CN&tl=vi&text=...
    resp = client.get(
        "/language/translate/v2",
        params={
            "sl": "zh-CN",
            "tl": "vi",
            "text": "今天天气很好，我们一起去公园散步吧。",
        },
    )
    assert resp.status_code == 200
    assert resp.text == "今天天气很好，我们一起去公园散步吧。 -> vi"


def test_post_translate_single_q():
    resp = client.post(
        "/language/translate/v2",
        json={"q": "hello", "source": "en", "target": "vi"},
    )
    assert resp.status_code == 200
    assert resp.text == "hello -> vi"


