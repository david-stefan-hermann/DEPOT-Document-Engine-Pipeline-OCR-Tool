from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Literal

from pydantic import BaseModel, Field, field_validator

log = logging.getLogger(__name__)

# A real document's issue date is essentially always in the past; allow a small
# buffer for documents dated slightly ahead (e.g. subscription renewals) and
# reject anything clearly implausible (OCR/model garbage like year 3107).
_MIN_PLAUSIBLE_DATE = date(1900, 1, 1)
_MAX_FUTURE_BUFFER = timedelta(days=60)


def _normalize_confidence_value(v: float | int) -> float:
    # Small models occasionally answer with a 0-100 percentage instead of
    # the requested 0.0-1.0 scale (e.g. 95 meaning "95%"). Rescale rather
    # than hard-failing the whole classification over a formatting slip.
    if isinstance(v, (int, float)) and v > 1:
        log.warning("Model returned confidence=%r outside 0-1; treating as a percentage.", v)
        v = v / 100
    return max(0.0, min(1.0, float(v)))


class ContentExtraction(BaseModel):
    """What the document IS, independent of where it should be filed:
    title, issue date and correspondent, extracted from OCR text alone."""

    title: str = Field(min_length=1)
    # Deliberately REQUIRED (no default), not Optional: a live test against
    # the real model showed it reliably returns null/omits this field when
    # it's merely optional in the JSON schema, even with an explicit prompt
    # instruction saying otherwise - but reliably fills it in correctly once
    # the schema itself marks it required. Empty string ("") is still a
    # legitimate value, meaning "genuinely no sender found".
    correspondent: str
    issue_date: date | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, v: float | int) -> float:
        return _normalize_confidence_value(v)

    @field_validator("title")
    @classmethod
    def _strip_title(cls, v: str) -> str:
        return v.strip()

    @field_validator("correspondent")
    @classmethod
    def _strip_correspondent(cls, v: str) -> str:
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


class FolderStepDecision(BaseModel):
    """One step of the level-by-level descent through the Dokumente/ tree:
    given the current folder's direct children, either go into one of them,
    stay at the current level, or propose a new child folder here."""

    action: Literal["descend", "stay", "new_folder"]
    folder_name: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, v: float | int) -> float:
        return _normalize_confidence_value(v)

    @field_validator("folder_name")
    @classmethod
    def _strip_folder_name(cls, v: str | None) -> str | None:
        return v.strip().strip("/") if v else None


class OcrResult(BaseModel):
    """Result of running OCR on one input file."""

    text: str
    page_count: int
    ocr_pdf_path: str
    ocr_failed: bool

    model_config = {"arbitrary_types_allowed": True}
