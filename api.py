from typing import List, Optional, Sequence

from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import PlainTextResponse

from schemas import TranslateRequest, TranslateResponse, TranslateData, TranslationItem
from translator import TranslatorService, get_translator_service


router = APIRouter(prefix="/language/translate", tags=["translate"])


def _build_response(translations) -> TranslateResponse:
    items = [
        TranslationItem(
            translatedText=t.text,
            detectedSourceLanguage=t.detected_source_language,
        )
        for t in translations
    ]
    return TranslateResponse(data=TranslateData(translations=items))


def _build_plain_response(translations: Sequence) -> str:
    texts = [t.text for t in translations]
    if not texts:
        return ""
    if len(texts) == 1:
        return texts[0]
    return "\n".join(texts)


@router.post("/v2", response_class=PlainTextResponse)
async def translate_v2(
    body: TranslateRequest,
    translator: TranslatorService = Depends(get_translator_service),
) -> str:
    texts: List[str]
    if isinstance(body.q, str):
        texts = [body.q]
    else:
        texts = body.q

    translations = await translator.translate_batch(
        texts=texts,
        source=body.source or "auto",
        target=body.target,
    )

    return _build_plain_response(translations)


@router.get("/v2", response_class=PlainTextResponse)
async def translate_v2_get(
    q: Optional[List[str]] = Query(default=None),
    source: Optional[str] = Query(default="auto"),
    target: Optional[str] = Query(default=None),
    # Google Translate–style aliases:
    sl: Optional[str] = Query(default=None),
    tl: Optional[str] = Query(default=None),
    # Game-style aliases:
    from_: Optional[str] = Query(default=None, alias="from"),
    to: Optional[str] = Query(default=None),
    text: Optional[str] = Query(default=None),
    op: Optional[str] = Query(default=None),  # not used, but accepted for compatibility
    translator: TranslatorService = Depends(get_translator_service),
) -> str:
    # Resolve source language: sl (Google) has priority, then from_ (game), then source, then "auto".
    src = sl or from_ or source or "auto"
    # Normalize locale codes like zh-CN -> zh
    if "-" in src:
        src = src.split("-", 1)[0]

    # Resolve target language: tl (Google) has priority, then to (game), then target (required in our API).
    tgt = tl or to or target
    if not tgt:
        raise HTTPException(status_code=400, detail="target (or tl/to) is required")
    if "-" in tgt:
        tgt = tgt.split("-", 1)[0]

    texts: List[str]
    if q:
        # Our native query parameter; supports multiple values (?q=a&q=b).
        texts = q
    elif text:
        # Google-style ?text=... (possibly multi-line, %0A -> '\n').
        # Split on lines and drop empty ones.
        texts = [line for line in text.splitlines() if line.strip()]
    else:
        texts = []

    translations = await translator.translate_batch(
        texts=texts,
        source=src or "auto",
        target=tgt,
    )

    return _build_plain_response(translations)

