from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from depot.signals import MAX_FUTURE_DAYS

log = logging.getLogger(__name__)

# A real document's issue date is never meaningfully in the future (see
# signals.MAX_FUTURE_DAYS); reject that and anything clearly implausible
# (OCR/model garbage like year 3107).
_MIN_PLAUSIBLE_DATE = date(1900, 1, 1)
_MAX_FUTURE_BUFFER = timedelta(days=MAX_FUTURE_DAYS)

# Hard limits on the locally generated keywords - they may be sent to the
# cloud classifier, so anything that isn't a short plain topic word is
# dropped rather than trusted to the prompt alone.
_MAX_KEYWORDS = 6
_MAX_KEYWORD_LENGTH = 40
_MAX_SUMMARY_LENGTH = 400
_SUMMARY_EMAIL = re.compile(r"\S+@\S+")
# Numbers that could identify someone or something: any run from a first to
# a last digit (letters, dots, slashes and dashes in between, spaces only
# between two digits, as in an IBAN), with the letters attached to it, and
# four or more digits in total - account, customer, phone and policy
# numbers, amounts, full dates. A lone year ("2026") stays.
_SUMMARY_NUMBER = re.compile(r"[^\W\d_]*\d(?:[\w./-]|(?<=\d) (?=\d))*\d[^\W\d_]*")
_SUMMARY_YEAR = re.compile(r"(19|20)\d{2}")


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
    # General topic/document-type words ("Rechnung", "Strom"), no personal
    # data. Optional here so a filename-only extraction can be built
    # without them, but forced to "required" in the schema handed to the
    # model (see extraction_json_schema) for the same reason as
    # `correspondent` above.
    keywords: list[str] = Field(default_factory=list)
    # Two or three sentences on what the document is about, for the filing
    # decision - the only description of the content the cloud classifier
    # gets. No names, addresses or numbers (the prompt asks for that; what
    # still looks like one is removed below).
    summary: str = ""

    @field_validator("summary", mode="before")
    @classmethod
    def _clean_summary(cls, v: object) -> str:
        if not isinstance(v, str):
            return ""
        text = _SUMMARY_EMAIL.sub("", v)
        text = _SUMMARY_NUMBER.sub(
            lambda m: m[0] if _SUMMARY_YEAR.fullmatch(m[0]) or sum(c.isdigit() for c in m[0]) < 4 else "", text
        )
        return " ".join(text.split())[:_MAX_SUMMARY_LENGTH]

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, v: float | int) -> float:
        return _normalize_confidence_value(v)

    @field_validator("keywords", mode="before")
    @classmethod
    def _clean_keywords(cls, v: object) -> list[str]:
        if not isinstance(v, list):
            return []
        cleaned: list[str] = []
        for item in v:
            word = " ".join(str(item).split())
            if not word or len(word) > _MAX_KEYWORD_LENGTH or any(ch.isdigit() for ch in word):
                continue
            if word not in cleaned:
                cleaned.append(word)
        return cleaned[:_MAX_KEYWORDS]

    @field_validator("title")
    @classmethod
    def _strip_title(cls, v: str) -> str:
        return v.strip()

    @field_validator("correspondent")
    @classmethod
    def _strip_correspondent(cls, v: str) -> str:
        return v.strip()

    @field_validator("issue_date", mode="before")
    @classmethod
    def _unparseable_date_is_no_date(cls, v: object) -> object:
        # The model sometimes fills in "0000-00-00" when it finds no date.
        # pydantic lets that escape as a bare ValueError ("year 0 is out of
        # range"), which failed the whole document - three times, i.e. into
        # quarantine, since the model answers the same every time.
        if isinstance(v, str):
            try:
                return date.fromisoformat(v.strip())
            except ValueError:
                log.warning("Model returned an unparseable issue_date %r; discarding it.", v)
                return None
        return v

    @field_validator("issue_date")
    @classmethod
    def _reject_implausible_date(cls, v: date | None) -> date | None:
        if v is None:
            return None
        if v < _MIN_PLAUSIBLE_DATE or v > date.today() + _MAX_FUTURE_BUFFER:
            log.warning("Model returned an implausible issue_date %s; discarding it.", v)
            return None
        return v


def extraction_json_schema(summary: bool = False) -> dict:
    """ContentExtraction's JSON schema as given to the model, with
    `keywords` marked required so the model actually fills it in. The
    `summary` only the cloud classifier reads is required with `summary`
    and left out otherwise, so a fully local run does not generate it."""
    schema = ContentExtraction.model_json_schema()
    required = schema.setdefault("required", [])
    if not summary:
        schema["properties"].pop("summary")
    for name in ("keywords", "summary") if summary else ("keywords",):
        if name not in required:
            required.append(name)
    return schema


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


class FolderPick(BaseModel):
    """The one decision among a shortlist of folders. `folder` is restricted
    to the offered paths in the schema handed to the model.

    Deliberately the path itself, not its number in the list: asked for a
    number, the model's answer depended on the order of the list (with the
    list reversed the same document went to a different folder in 6 of 6
    real cases); writing out the path, it chose the same folder either way
    in 5 of 6."""

    folder: str = Field(min_length=1)
    # Set only to create a new subfolder under the chosen folder.
    new_folder_name: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, v: float | int) -> float:
        return _normalize_confidence_value(v)

    @field_validator("folder")
    @classmethod
    def _strip_folder(cls, v: str) -> str:
        return v.strip().strip("/")

    @field_validator("new_folder_name")
    @classmethod
    def _strip_new_folder_name(cls, v: str | None) -> str | None:
        return v.strip().strip("/") or None if v else None


class AnthropicFolderDecision(BaseModel):
    """A single-shot filing decision from the cloud (Anthropic) classifier,
    given the whole existing folder tree at once rather than one level at a
    time - a strong model doesn't need the small-model-oriented hierarchical
    walk that FolderStepDecision/_walk_folder_tree exists for."""

    action: Literal["existing", "new_folder"]
    # For "existing": must be one of the offered existing folder paths
    # exactly. For "new_folder": the EXISTING parent path the new folder
    # should be created under (new_folder_name holds the new leaf name).
    folder: str = Field(min_length=1)
    new_folder_name: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, v: float | int) -> float:
        return _normalize_confidence_value(v)

    @field_validator("folder")
    @classmethod
    def _strip_folder(cls, v: str) -> str:
        return v.strip().strip("/")

    @field_validator("new_folder_name")
    @classmethod
    def _strip_new_folder_name(cls, v: str | None) -> str | None:
        return v.strip().strip("/") if v else None


class OcrResult(BaseModel):
    """Result of running OCR on one input file."""

    text: str
    page_count: int
    # The file to archive: a new searchable PDF when OCR ran, otherwise the
    # untouched input (born-digital PDF, or nothing recognizable at all).
    ocr_pdf_path: str
    ocr_failed: bool
    # True when the text came straight from the PDF's own text layer on
    # every page, i.e. a born-digital document rather than a scan.
    born_digital: bool = False

    model_config = {"arbitrary_types_allowed": True}
