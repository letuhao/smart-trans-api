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


@dataclass
class CacheSettings:
    persistent_file: str = "cache.json"


@dataclass
class DefaultSettings:
    source_lang: str = "auto"
    target_lang: str = "en"


@dataclass
class Settings:
    lmstudio: LMStudioSettings
    batch: BatchSettings
    cache: CacheSettings
    default: DefaultSettings
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
    default = data.get("default", {})
    prompts = data.get("prompts", {})

    lm_settings = LMStudioSettings(
        base_url=str(lm.get("base_url", "http://127.0.0.1:1234/v1")),
        model=str(lm.get("model", "local-model-name")),
        endpoint_type=str(lm.get("endpoint_type", "chat")),
    )
    batch_settings = BatchSettings(
        max_size=int(batch.get("max_size", 8)),
        max_chars=int(batch.get("max_chars", 8000)),
    )
    cache_settings = CacheSettings(
        persistent_file=str(cache.get("persistent_file", "cache.json")),
    )
    default_settings = DefaultSettings(
        source_lang=str(default.get("source_lang", "auto")),
        target_lang=str(default.get("target_lang", "en")),
    )

    return Settings(
        lmstudio=lm_settings,
        batch=batch_settings,
        cache=cache_settings,
        default=default_settings,
        prompts=prompts or {},
    )

