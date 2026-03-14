from typing import List, Optional, Union

from pydantic import BaseModel, Field


class TranslateRequest(BaseModel):
    q: Union[str, List[str]]
    source: Optional[str] = Field(default="auto")
    target: str
    session_id: Optional[str] = Field(default=None, description="Optional session for context consistency across requests")
    format: Optional[str] = None
    model: Optional[str] = None
    mimeType: Optional[str] = None


class TranslationItem(BaseModel):
    translatedText: str
    detectedSourceLanguage: Optional[str] = None


class TranslateData(BaseModel):
    translations: List[TranslationItem]


class TranslateResponse(BaseModel):
    data: TranslateData

