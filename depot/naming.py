from __future__ import annotations

import difflib
import unicodedata
from datetime import date

from pathvalidate import sanitize_filename

# Characters beyond pathvalidate's default set that we also want gone from
# titles (quotes/asterisks etc. sneak in from OCR'd document text).
_INVALID_CHARS = '/\\:*?"<>|'

# Conservative cap on the generated filename's length (date prefix + name +
# extension), leaving headroom for Nextcloud's own path-length limit given
# that files also sit several folder levels deep under Dokumente/.
MAX_FILENAME_LENGTH = 150


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


def sanitize_correspondent(correspondent: str | None) -> str:
    """Like sanitize_title, but returns "" (not the "Dokument" fallback) for
    blank/unusable input, since the correspondent is an optional field that
    is simply omitted from the filename when unknown."""
    if not correspondent:
        return ""
    cleaned = sanitize_title(correspondent)
    return "" if cleaned == "Dokument" else cleaned


def build_filename(
    title: str,
    issue_date: date | None,
    processed_on: date,
    ext: str = "pdf",
    correspondent: str | None = None,
) -> str:
    """Build the canonical `YYYY-MM-DD [Absender - ]Titel.pdf` filename.
    Falls back to the processing date (with a visible marker) when no issue
    date could be determined, so the sortable date-prefix convention never
    breaks. `correspondent` (who issued/sent the document) is optional and,
    when present, is prefixed to the title with " - " so documents from the
    same sender sort/scan together at a glance without cluttering the free
    text title itself. `ext` defaults to pdf (OCR always produces a
    searchable PDF) but callers may pass the original extension for the rare
    case where OCR produced nothing usable at all and the raw scan bytes are
    being filed as-is."""
    clean_title = sanitize_title(title)
    clean_correspondent = sanitize_correspondent(correspondent)
    ext = ext.lstrip(".")
    date_str = issue_date.isoformat() if issue_date is not None else processed_on.isoformat()

    name_core = f"{clean_correspondent} - {clean_title}" if clean_correspondent else clean_title
    if issue_date is None:
        name_core = f"{name_core} (Datum unsicher)"

    budget = MAX_FILENAME_LENGTH - len(date_str) - len(ext) - 2  # spaces + dot
    if budget > 0 and len(name_core) > budget:
        name_core = name_core[:budget].rstrip()

    return f"{date_str} {name_core}.{ext}"


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
