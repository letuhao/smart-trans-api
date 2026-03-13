## Simple LM Studio Translator API

A small Python FastAPI service that exposes a **Google Translate–compatible HTTP API** and uses **LM Studio** (Gemma / Translategemma or other models) to perform translations.  
It supports **batching**, **in-memory + JSON persistent caching**, and **direction-aware prompts** configured via a YAML file.

---

## Features

- **Google Translate–style interface**
  - `POST /language/translate/v2` with JSON body (Google-like).
  - `GET /language/translate/v2` with:
    - Native style: `q`, `source`, `target`.
    - Google style: `sl`, `tl`, `text`, `op` (compatible with Google Translate URLs).
- **LM Studio backend**
  - Connects to LM Studio via OpenAI-compatible HTTP API (e.g. `/v1/chat/completions`).
  - Supports Gemma / Translategemma and other chat/completion models.
- **Batching**
  - Configurable `max_size` and `max_chars` per batch to avoid token limits.
- **Caching**
  - In-memory cache for fast repeated lookups.
  - JSON file persistent cache to avoid re-calling the model between runs.
- **Prompt control**
  - English system prompt template.
  - Direction-specific overrides (e.g. `zh → vi` with Hán–Việt style rules).

---

## Project structure

- `main.py` – FastAPI app entry point.
- `api.py` – HTTP API routes (`/language/translate/v2` GET & POST).
- `translator.py` – `TranslatorService`, batching, LM Studio integration, prompt logic.
- `cache.py` – In-memory + JSON persistent translation cache.
- `config.py` – Loads `config.yaml` into strongly-typed settings.
- `config.yaml` – Configuration for LM Studio, batching, cache, and prompts.
- `schemas.py` – Pydantic models for Google Translate–style request/response.
- `test_translator.py` – Unit tests for translator logic.
- `test_api.py` – API-level tests using FastAPI `TestClient`.

---

## Requirements

- Python 3.10+

Install dependencies (adjust as needed):

```bash
pip install fastapi uvicorn httpx pyyaml pytest
# For FastAPI TestClient and extras (optional but recommended)
pip install "fastapi[all]"
```

---

## Configuration

Edit `config.yaml`:

```yaml
lmstudio:
  base_url: "http://localhost:1234/v1"      # LM Studio OpenAI-compatible base URL
  model: "google/gemma-3-1b-it"             # or your Translategemma model
  endpoint_type: "chat"                     # "chat" or "completion"

batch:
  max_size: 4
  max_chars: 4000

cache:
  persistent_file: "cache.json"

default:
  source_lang: "auto"
  target_lang: "vi"

prompts:
  default: >
    You are a professional translation engine. Translate the user message from
    {source_lang_name} (language code: {source_lang_code}) to {target_lang_name}
    (language code: {target_lang_code}). Rules: 1) Output only the translation,
    no explanations. 2) Preserve meaning and tone. 3) Use vocabulary that is
    natural to native speakers of {target_lang_name}.

  zh-vi: >
    You are a professional Chinese-to-Vietnamese literary translator. Translate
    the user message from Chinese (language code: {source_lang_code}) to
    Vietnamese (language code: {target_lang_code}). Rules: 1) Output only the
    translation, no explanations or comments. 2) Prefer Sino-Vietnamese
    (Hán-Việt) vocabulary to preserve a classical or formal tone when
    appropriate. 3) Do not add or omit information compared to the source. 4)
    Use a refined, literary register that still sounds natural in Vietnamese.
```

Key points:

- `lmstudio.base_url` must point to LM Studio’s OpenAI-compatible endpoint root (e.g. `http://localhost:1234/v1`).
- `lmstudio.model` must match a model you have loaded in LM Studio.
- `prompts` can define:
  - `default` template.
  - Direction-specific templates under keys like `zh-vi`.
- You can override the config path with the `TRANSLATOR_CONFIG_PATH` environment variable.

---

## Running the API

1. Start LM Studio with the desired model and enable the OpenAI-compatible API (e.g. listening on `http://localhost:1234/v1`).

2. From the project directory, run:

```bash
uvicorn main:app --reload
```

The API is now available at `http://127.0.0.1:8000`.

---

## Using the API

### 1. POST – Google-style JSON

Endpoint: `POST /language/translate/v2`

Body (Google Translate–compatible):

```json
{
  "q": [
    "今天天气很好，我们一起去公园散步吧。",
    "修为越深，因果越重。"
  ],
  "source": "zh",
  "target": "vi"
}
```

Example with `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/language/translate/v2" \
  -H "Content-Type: application/json" \
  -d '{
    "q": ["今天天气很好，我们一起去公园散步吧。"],
    "source": "zh",
    "target": "vi"
  }'
```

Response shape:

```json
{
  "data": {
    "translations": [
      {
        "translatedText": "...",
        "detectedSourceLanguage": "zh"
      }
    ]
  }
}
```

### 2. GET – Native style (`q/source/target`)

Single text:

```bash
curl "http://127.0.0.1:8000/language/translate/v2?q=Hello%20world&source=en&target=vi"
```

Multiple texts:

```bash
curl "http://127.0.0.1:8000/language/translate/v2?q=Hello&q=How%20are%20you%3F&source=en&target=vi"
```

### 3. GET – Google Translate style (`sl/tl/text`)

This mirrors URLs like Google Translate’s:

```bash
curl "http://127.0.0.1:8000/language/translate/v2?sl=zh-CN&tl=vi&text=今天天气很好，我们一起去公园散步吧。"
```

- `sl` – source language (e.g. `zh-CN`).
- `tl` – target language (e.g. `vi`).
- `text` – the text to translate. Multi-line text is supported; `%0A` is decoded as `\n` and split into separate items.

If both `q` and `text` are present, `q` takes precedence.

---

## Batching & Caching

- **Batching**:
  - Inputs are chunked according to `batch.max_size` and `batch.max_chars`.
  - Each chunk is sent to LM Studio in one or more API calls, depending on the backend type.

- **Caching**:
  - Keys are `"{source_lang}|{target_lang}|{text}"`.
  - In-memory cache speeds up repeated translations during a single run.
  - `cache.json` stores translations across restarts. To clear bad entries, you can delete or edit this file.

---

## Testing

To run tests (translator logic + API):

```bash
pytest -q
```

- `test_translator.py`:
  - Tests prompt building and translation pipeline (with LM Studio mocked).
- `test_api.py`:
  - Tests GET `/language/translate/v2` for:
    - Single and multiple `q`.
    - Google-style `sl/tl/text`.

---

## Notes / Tips

- If you change LM Studio’s port or base path, update `lmstudio.base_url` in `config.yaml`.
- For Translategemma models, ensure the model name in config matches the model you load in LM Studio and that the OpenAI-compatible endpoint is enabled.
- If you see error strings being returned as `translatedText`, clear `cache.json` to remove previously cached error responses.

