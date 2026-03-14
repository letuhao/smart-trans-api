from config import get_settings
from translator import _build_system_prompt


def test_zh_vi_prompt_includes_game_instructions() -> None:
    settings = get_settings()
    prompt = _build_system_prompt(settings, "zh", "vi")

    assert "Preserve all structural markers" in prompt
    assert "Sino-Vietnamese (Hán-Việt)" in prompt or "Hán-Việt" in prompt
    assert "【】" in prompt or "bracketed segments" in prompt

