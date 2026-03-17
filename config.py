from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import os

import yaml


@dataclass
class LMStudioSettings:
    base_url: str
    model: str
    endpoint_type: str = "chat"


@dataclass
class BatchSettings:
    max_size: int = 8
    max_chars: int = 8000
    zh_vi_max_size: Optional[int] = None
    max_retry_broken: int = 3


@dataclass
class CacheSettings:
    persistent_file: str = "cache.json"


@dataclass
class GemmaSettings:
    max_slice_chars: int = 2000
    max_retry_broken: int = 3
    temperature: float = 1.0


@dataclass
class TranslategemmaSettings:
    version: str = "v2"  # "v1" | "v2"
    system_prompt: str = ""  # v2: if empty, do not send system message
    post_process: bool = False  # v2: when True, run full post-process chain
    user_input_format: str = "json"  # "json" | "raw" (v1)


@dataclass
class SessionSettings:
    # "persistent" | "request" | "none". request = one random session per request; none = no session.
    mode: str = "request"
    max_entries: int = 100
    max_chars: int = 4000
    ttl_seconds: int = 3600
    inject_context_into_prompt: bool = False


@dataclass
class DefaultSettings:
    source_lang: str = "auto"
    target_lang: str = "en"


@dataclass
class ValidationSettings:
    mode: str = "smart"  # "strict" | "smart"
    max_chinese_ratio: float = 0.05
    max_chinese_chars: int = 5


@dataclass
class Settings:
    lmstudio: LMStudioSettings
    batch: BatchSettings
    cache: CacheSettings
    gemma: GemmaSettings
    translategemma: TranslategemmaSettings
    session: SessionSettings
    default: DefaultSettings
    validation: ValidationSettings
    prompts: dict | None = None


def _load_raw_config() -> dict:
    config_path = os.getenv("TRANSLATOR_CONFIG_PATH", "config.yaml")
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_settings() -> Settings:
    data = _load_raw_config()

    lm = data.get("lmstudio", {})
    batch = data.get("batch", {})
    cache = data.get("cache", {})
    gemma = data.get("gemma", {})
    translategemma = data.get("translategemma", {})
    session = data.get("session", {})
    default = data.get("default", {})
    validation = data.get("validation", {})
    prompts = data.get("prompts", {})

    lm_settings = LMStudioSettings(
        base_url=str(lm.get("base_url", "http://127.0.0.1:1234/v1")),
        model=str(lm.get("model", "local-model-name")),
        endpoint_type=str(lm.get("endpoint_type", "chat")),
    )
    batch_settings = BatchSettings(
        max_size=int(batch.get("max_size", 8)),
        max_chars=int(batch.get("max_chars", 8000)),
        zh_vi_max_size=int(batch["zh_vi_max_size"]) if "zh_vi_max_size" in batch else None,
        max_retry_broken=int(batch.get("max_retry_broken", 3)),
    )
    cache_settings = CacheSettings(
        persistent_file=str(cache.get("persistent_file", "cache.json")),
    )
    gemma_settings = GemmaSettings(
        max_slice_chars=int(gemma.get("max_slice_chars", 2000)),
        max_retry_broken=int(gemma.get("max_retry_broken", 3)),
        temperature=float(gemma.get("temperature", 1.0)),
    )
    translategemma_settings = TranslategemmaSettings(
        version=str(translategemma.get("version", "v2")).lower(),
        system_prompt=str(translategemma.get("system_prompt", "")).strip(),
        post_process=bool(translategemma.get("post_process", False)),
        user_input_format=str(translategemma.get("user_input_format", "json")).lower(),
    )
    session_settings = SessionSettings(
        mode=str(session.get("mode", "request")).lower(),
        max_entries=int(session.get("max_entries", 100)),
        max_chars=int(session.get("max_chars", 4000)),
        ttl_seconds=int(session.get("ttl_seconds", 3600)),
        inject_context_into_prompt=bool(session.get("inject_context_into_prompt", False)),
    )
    default_settings = DefaultSettings(
        source_lang=str(default.get("source_lang", "auto")),
        target_lang=str(default.get("target_lang", "en")),
    )
    validation_settings = ValidationSettings(
        mode=str(validation.get("mode", "smart")),
        max_chinese_ratio=float(validation.get("max_chinese_ratio", 0.05)),
        max_chinese_chars=int(validation.get("max_chinese_chars", 5)),
    )

    return Settings(
        lmstudio=lm_settings,
        batch=batch_settings,
        cache=cache_settings,
        gemma=gemma_settings,
        translategemma=translategemma_settings,
        session=session_settings,
        default=default_settings,
        validation=validation_settings,
        prompts=prompts or {},
    )

