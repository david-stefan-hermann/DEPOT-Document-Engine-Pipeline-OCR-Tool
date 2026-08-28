from __future__ import annotations

import logging
from datetime import date, timedelta

from pydantic import BaseModel, Field, field_validator

log = logging.getLogger(__name__)

# A real document's issue date is essentially always in the past; allow a small
# buffer for documents dated slightly ahead (e.g. subscription renewals) and
# reject anything clearly implausible (OCR/model garbage like year 3107).
_MIN_PLAUSIBLE_DATE = date(1900, 1, 1)
_MAX_FUTURE_BUFFER = timedelta(days=60)


class ClassificationResult(BaseModel):
    """Structured response expected from the local LLM's classification call."""

    folder: str = Field(min_length=1)
    is_new_folder: bool
    title: str = Field(min_length=1)
    issue_date: date | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, v: float | int) -> float:
        # Small models occasionally answer with a 0-100 percentage instead of
        # the requested 0.0-1.0 scale (e.g. 95 meaning "95%"). Rescale rather
        # than hard-failing the whole classification over a formatting slip.
        if isinstance(v, (int, float)) and v > 1:
            log.warning("Model returned confidence=%r outside 0-1; treating as a percentage.", v)
            v = v / 100
        return max(0.0, min(1.0, float(v)))

    @field_validator("folder")
    @classmethod
    def _strip_folder(cls, v: str) -> str:
        return v.strip().strip("/")

    @field_validator("title")
    @classmethod
    def _strip_title(cls, v: str) -> str:
        return v.strip()

    @field_validator("issue_date")
    @classmethod
    def _reject_implausible_date(cls, v: date | None) -> date | None:
        if v is None:
            return None
        if v < _MIN_PLAUSIBLE_DATE or v > date.today() + _MAX_FUTURE_BUFFER:
            log.warning("Model returned an implausible issue_date %s; discarding it.", v)
            return None
        return v


class OcrResult(BaseModel):
    """Result of running OCR on one input file."""

    text: str
    page_count: int
    ocr_pdf_path: str
    ocr_failed: bool

    model_config = {"arbitrary_types_allowed": True}
