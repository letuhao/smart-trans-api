from typing import List, Optional, Union

from pydantic import BaseModel, Field


class TranslateRequest(BaseModel):
    q: Union[str, List[str]]
    source: Optional[str] = Field(default="auto")
    target: str
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

