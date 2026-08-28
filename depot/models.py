from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, field_validator


class ClassificationResult(BaseModel):
    """Structured response expected from the local LLM's classification call."""

    folder: str = Field(min_length=1)
    is_new_folder: bool
    title: str = Field(min_length=1)
    issue_date: date | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""

    @field_validator("folder")
    @classmethod
    def _strip_folder(cls, v: str) -> str:
        return v.strip().strip("/")

    @field_validator("title")
    @classmethod
    def _strip_title(cls, v: str) -> str:
        return v.strip()


class OcrResult(BaseModel):
    """Result of running OCR on one input file."""

    text: str
    page_count: int
    ocr_pdf_path: str
    ocr_failed: bool

    model_config = {"arbitrary_types_allowed": True}
