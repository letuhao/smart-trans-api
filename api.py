import re
import uuid
from typing import List, Optional, Sequence

from fastapi import APIRouter, Depends, Query, HTTPException, Request
from fastapi.responses import PlainTextResponse

from config import get_settings
from schemas import TranslateRequest, TranslateResponse, TranslateData, TranslationItem
from translator import TranslatorService, get_translator_service


router = APIRouter(prefix="/language/translate", tags=["translate"])

# Prefix for server-derived session IDs so they don't collide with client-provided ones.
SESSION_PREFIX_IP = "ip:"
SESSION_PREFIX_REQUEST = "req:"


def _resolve_session_id(
    session_id: Optional[str], request: Request
) -> Optional[str]:
    """
    Resolve session_id according to session.mode:
    - none: always None.
    - request: one random session per request (client session_id ignored).
    - persistent: client session_id if provided, else ip:{client_host}.
    """
    settings = get_settings()
    mode = (getattr(settings.session, "mode", "request") or "request").lower()
    if mode == "none":
        return None
    if mode == "request":
        return f"{SESSION_PREFIX_REQUEST}{uuid.uuid4().hex}"
    # persistent
    if session_id and session_id.strip():
        return session_id.strip()
    client_host = request.client.host if request.client else None
    if client_host:
        forwarded = request.headers.get("x-forwarded-for") or request.headers.get("x-real-ip")
        if forwarded:
            first = forwarded.split(",")[0].strip()
            if first and re.match(r"^[\d.a-f:]+$", first, re.IGNORECASE):
                client_host = first
        return f"{SESSION_PREFIX_IP}{client_host}"
    return None


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
    request: Request,
    body: TranslateRequest,
    translator: TranslatorService = Depends(get_translator_service),
) -> str:
    texts: List[str]
    if isinstance(body.q, str):
        texts = [body.q]
    else:
        texts = body.q

    session_id = _resolve_session_id(body.session_id, request)
    translations = await translator.translate_batch(
        texts=texts,
        source=body.source or "auto",
        target=body.target,
        session_id=session_id,
    )

    return _build_plain_response(translations)


@router.get("/v2", response_class=PlainTextResponse)
async def translate_v2_get(
    request: Request,
    q: Optional[List[str]] = Query(default=None),
    source: Optional[str] = Query(default="auto"),
    target: Optional[str] = Query(default=None),
    session_id: Optional[str] = Query(default=None, description="Optional session for context consistency"),
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
        # Gemma-3-12b: pass raw text as one item (no line split) so model gets full context.
        settings = get_settings()
        # Gemma and Translategemma: pass raw text as one item (no line split).
        if settings.lmstudio.model and ("gemma" in settings.lmstudio.model.lower() or "translategemma" in settings.lmstudio.model.lower()):
            texts = [text]
        else:
            texts = [line for line in text.splitlines() if line.strip()]
    else:
        texts = []

    resolved_session_id = _resolve_session_id(session_id, request)
    translations = await translator.translate_batch(
        texts=texts,
        source=src or "auto",
        target=tgt,
        session_id=resolved_session_id,
    )

    return _build_plain_response(translations)

