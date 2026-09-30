"""Deterministic (no LLM) signals about a document: what its original
filename and PDF metadata already say about title and date, and which dates
literally occur in its text. Cheap, reproducible, and independent of OCR
quality - real production logs showed these being thrown away: photos with
a perfectly descriptive filename went to Unsortiert unclassified, and files
with the date right in their name were still filed as "Datum unsicher"."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pymupdf as fitz

from depot.naming import normalize

log = logging.getLogger(__name__)

# A received document can't have been issued after it was scanned/processed;
# the small buffer only absorbs timezone/clock slack. Deliberately tight: a
# real case filed an election notice under the (future) election day instead
# of the letter's own date, because a date up to 60 days ahead used to count
# as plausible.
MAX_FUTURE_DAYS = 3
_MIN_PLAUSIBLE_DATE = date(1900, 1, 1)

# Words a scanner/phone/OS puts into a filename on its own. A stem made up of
# nothing but these (plus digits and separators) says nothing about the
# document, so it is not treated as a user-given title.
_GENERIC_WORDS = frozenset({
    "scn", "scan", "scans", "scanned", "img", "image", "images", "pxl", "dsc", "dscn", "dscf",
    "doc", "document", "dokument", "photo", "foto", "bild", "pic", "picture", "camscanner",
    "adobe", "whatsapp", "wa", "screenshot", "bildschirmfoto", "unbenannt", "untitled",
    "neu", "new", "file", "datei", "copy", "kopie", "von", "of", "at", "um",
})

_MONTHS = {
    "januar": 1, "jan": 1, "februar": 2, "feb": 2, "märz": 3, "maerz": 3, "mär": 3, "mrz": 3,
    "april": 4, "apr": 4, "mai": 5, "juni": 6, "jun": 6, "juli": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "oktober": 10, "okt": 10,
    "november": 11, "nov": 11, "dezember": 12, "dez": 12,
}

_ISO_DATE = re.compile(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)")
_GERMAN_DATE = re.compile(r"(?<!\d)(\d{1,2})\s?[./-]\s?(\d{1,2})\s?[./-]\s?(\d{4}|\d{2})(?!\d)")
_MONTH_NAME_DATE = re.compile(
    r"(?<!\d)(\d{1,2})\.?\s+(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?\s+(\d{4})(?!\d)",
    re.IGNORECASE,
)
# Filenames only count a date with a four-digit year - "Seite 1-2-3" or a
# version number must not turn into a date.
_GERMAN_DATE_FULL_YEAR = re.compile(r"(?<!\d)(\d{1,2})[./-](\d{1,2})[./-](\d{4})(?!\d)")
_COMPACT_DATE = re.compile(r"(?<!\d)((?:19|20)\d{2})(\d{2})(\d{2})(?!\d)")
# A time of day directly following a date in a filename ("... 28-09-2026
# 13-52-56", "SCN_20260828_220109").
_TRAILING_TIME = re.compile(r"^[\s_T-]*\d{2}[-.:_]?\d{2}(?:[-.:_]?\d{2})?(?!\d)")
_COPY_COUNTER = re.compile(r"(?:-\d{1,2}|\s*\(\d{1,2}\))$")


@dataclass(frozen=True)
class FilenameSignals:
    # What the user called the file, cleaned of date/time - None for names a
    # scanner or phone generated on its own.
    title: str | None = None
    # A date the user put into the name.
    date: date | None = None
    # The timestamp of a machine-generated name: when it was scanned, NOT
    # when the document was issued - only usable as an upper bound.
    scan_date: date | None = None


def _make_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _expand_year(year: int) -> int:
    if year >= 100:
        return year
    return 2000 + year if year <= (date.today().year % 100) + 1 else 1900 + year


def find_dates(text: str) -> set[date]:
    """Every calendar date that literally occurs in `text`, in the formats
    German documents use (TT.MM.JJJJ, TT.MM.JJ, "14. August 2026") plus ISO.
    Tolerates the stray spaces OCR inserts around the separators."""
    found: set[date] = set()
    for m in _ISO_DATE.finditer(text):
        d = _make_date(int(m[1]), int(m[2]), int(m[3]))
        if d:
            found.add(d)
    for m in _GERMAN_DATE.finditer(text):
        d = _make_date(_expand_year(int(m[3])), int(m[2]), int(m[1]))
        if d:
            found.add(d)
    for m in _MONTH_NAME_DATE.finditer(text):
        d = _make_date(int(m[3]), _MONTHS[m[2].lower()], int(m[1]))
        if d:
            found.add(d)
    return {d for d in found if d >= _MIN_PLAUSIBLE_DATE}


def _first_filename_date(stem: str) -> tuple[date, int, int] | None:
    """The first date in a filename stem as (date, start, end), where `end`
    also swallows a time of day directly following it."""
    def numeric(order: tuple[int, int, int]):
        return lambda m: _make_date(int(m[order[0]]), int(m[order[1]]), int(m[order[2]]))

    parsers = (
        (_ISO_DATE, numeric((1, 2, 3))),
        (_GERMAN_DATE_FULL_YEAR, numeric((3, 2, 1))),
        (_COMPACT_DATE, numeric((1, 2, 3))),
        (_MONTH_NAME_DATE, lambda m: _make_date(int(m[3]), _MONTHS[m[2].lower()], int(m[1]))),
    )
    best: tuple[date, int, int] | None = None
    for pattern, parse in parsers:
        for m in pattern.finditer(stem):
            d = parse(m)
            if d is None or d < _MIN_PLAUSIBLE_DATE:
                continue
            if best is None or m.start() < best[1]:
                best = (d, m.start(), m.end())
            break
    if best is None:
        return None
    time_match = _TRAILING_TIME.match(stem[best[2]:])
    end = best[2] + (time_match.end() if time_match else 0)
    return best[0], best[1], end


def _is_generic(words: list[str]) -> bool:
    letters = [w for w in words if any(ch.isalpha() for ch in w)]
    return not letters or all(w.casefold() in _GENERIC_WORDS for w in letters)


def analyze_filename(filename: str) -> FilenameSignals:
    stem = normalize(Path(filename).stem)
    found = _first_filename_date(stem)
    rest = stem if found is None else f"{stem[:found[1]]} {stem[found[2]:]}"
    rest = _COPY_COUNTER.sub("", rest.strip())
    words = re.findall(r"[^\W\d_]+", rest)

    if _is_generic(words):
        return FilenameSignals(scan_date=found[0] if found else None)

    title = " ".join(rest.replace("_", " ").split()).strip(" ,;-.")
    if sum(ch.isalpha() for ch in title) < 3:
        return FilenameSignals(scan_date=found[0] if found else None)
    return FilenameSignals(title=title, date=found[0] if found else None)


def usable_pdf_title(raw_title: str | None) -> str | None:
    """A PDF's metadata title, if it actually describes the document (many
    are empty, or just the authoring tool's default like "Microsoft Word -
    Dokument1" / the scanner's own filename)."""
    if not raw_title:
        return None
    title = " ".join(normalize(raw_title).split())
    title = re.sub(r"^Microsoft Word - ", "", title)
    title = re.sub(r"\.(pdf|docx?|odt)$", "", title, flags=re.IGNORECASE)
    if sum(ch.isalpha() for ch in title) < 3 or _is_generic(re.findall(r"[^\W\d_]+", title)):
        return None
    return title[:150]


def read_pdf_metadata(path: Path) -> tuple[str | None, date | None]:
    """(usable title, creation date) from a PDF's document info. Never
    raises: a non-PDF or unreadable file just has no metadata."""
    if path.suffix.lower() != ".pdf":
        return None, None
    try:
        with fitz.open(path) as doc:
            meta = doc.metadata or {}
    except Exception:
        return None, None
    created: date | None = None
    m = re.match(r"D:(\d{4})(\d{2})(\d{2})", meta.get("creationDate") or "")
    if m:
        created = _make_date(int(m[1]), int(m[2]), int(m[3]))
    return usable_pdf_title(meta.get("title")), created


def resolve_issue_date(
    model_date: date | None,
    text: str,
    filename: FilenameSignals,
    pdf_creation_date: date | None,
    today: date,
) -> tuple[date | None, str | None]:
    """Decides the document's issue date from all available evidence.
    Returns (date, source) with source one of "model"/"filename"/"metadata",
    or (None, None) if nothing trustworthy is available.

    The model's date only counts if it is a date that really occurs in the
    document (its text or filename): a small model given several dates on
    one form has been seen blending digits from two of them into a date
    that appears nowhere. If the text contains no recognizable date at all
    there is nothing to check against, and the model's answer is kept.

    `pdf_creation_date` must only be passed for born-digital PDFs - for a
    scan it is the scan time, not the issue date."""
    latest = today
    if filename.scan_date is not None and filename.scan_date < latest:
        latest = filename.scan_date
    latest += timedelta(days=MAX_FUTURE_DAYS)

    def plausible(d: date | None) -> bool:
        return d is not None and _MIN_PLAUSIBLE_DATE <= d <= latest

    candidates = find_dates(text)
    if filename.date is not None:
        candidates.add(filename.date)

    if model_date is not None:
        if not plausible(model_date):
            log.warning("Discarding issue_date %s: later than the document can have been issued.", model_date)
        elif candidates and model_date not in candidates:
            log.warning("Discarding issue_date %s: does not occur anywhere in the document.", model_date)
        else:
            return model_date, "model"

    if plausible(filename.date):
        return filename.date, "filename"
    if plausible(pdf_creation_date):
        return pdf_creation_date, "metadata"
    return None, None
