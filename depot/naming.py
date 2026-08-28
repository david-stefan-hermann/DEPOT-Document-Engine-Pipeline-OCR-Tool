from __future__ import annotations

import difflib
import unicodedata
from datetime import date

from pathvalidate import sanitize_filename

# Characters beyond pathvalidate's default set that we also want gone from
# titles (quotes/asterisks etc. sneak in from OCR'd document text).
_INVALID_CHARS = '/\\:*?"<>|'


def normalize(text: str) -> str:
    """NFC-normalize text before any comparison or WebDAV path use, so
    filenames from different sources (e.g. macOS NFD) compare equal."""
    return unicodedata.normalize("NFC", text)


def sanitize_title(title: str) -> str:
    title = normalize(title).strip()
    for ch in _INVALID_CHARS:
        title = title.replace(ch, "")
    title = " ".join(title.split())  # collapse whitespace
    sanitized = str(sanitize_filename(title, replacement_text=""))
    return sanitized.strip() or "Dokument"


def build_filename(title: str, issue_date: date | None, processed_on: date, ext: str = "pdf") -> str:
    """Build the canonical `YYYY-MM-DD Titel.pdf` filename. Falls back to the
    processing date (with a visible marker) when no issue date could be
    determined, so the sortable date-prefix convention never breaks. `ext`
    defaults to pdf (OCR always produces a searchable PDF) but callers may
    pass the original extension for the rare case where OCR produced nothing
    usable at all and the raw scan bytes are being filed as-is."""
    clean_title = sanitize_title(title)
    ext = ext.lstrip(".")
    if issue_date is not None:
        return f"{issue_date.isoformat()} {clean_title}.{ext}"
    return f"{processed_on.isoformat()} {clean_title} (Datum unsicher).{ext}"


def resolve_collision(desired_name: str, existing_names: set[str]) -> str:
    """Append ' (2)', ' (3)', ... if desired_name already exists in the
    target folder."""
    if desired_name not in existing_names:
        return desired_name

    stem, _, ext = desired_name.rpartition(".")
    if not stem:
        stem, ext = desired_name, ""

    counter = 2
    while True:
        candidate = f"{stem} ({counter}).{ext}" if ext else f"{stem} ({counter})"
        if candidate not in existing_names:
            return candidate
        counter += 1


def folder_similarity(a: str, b: str) -> float:
    """Similarity ratio (0-1) between two folder leaf names, used to flag
    likely near-duplicate new-folder proposals (e.g. 'Rechnung' vs
    'Rechnungen')."""
    a_norm = normalize(a).strip().casefold()
    b_norm = normalize(b).strip().casefold()
    return difflib.SequenceMatcher(None, a_norm, b_norm).ratio()


def closest_existing_leaf(proposed_folder: str, existing_folders: list[str]) -> tuple[str, float] | None:
    """For a proposed new folder path, find the most similar leaf name among
    existing folders' last path segments. Returns (existing_full_path, ratio)
    or None if there are no existing folders to compare against."""
    proposed_leaf = proposed_folder.rstrip("/").rsplit("/", 1)[-1]
    best: tuple[str, float] | None = None
    for existing in existing_folders:
        leaf = existing.rstrip("/").rsplit("/", 1)[-1]
        ratio = folder_similarity(proposed_leaf, leaf)
        if best is None or ratio > best[1]:
            best = (existing, ratio)
    return best
