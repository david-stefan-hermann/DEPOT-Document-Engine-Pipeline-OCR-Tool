from __future__ import annotations

import json
import logging

import ollama
from pydantic import ValidationError

from depot.models import ClassificationResult
from depot.naming import closest_existing_leaf

log = logging.getLogger(__name__)

MAX_OCR_CHARS = 3500

# Above this similarity ratio, a proposed folder is treated as referring to an
# already-existing one (e.g. "Rechnung" vs "Rechnungen") and gets redirected
# there automatically instead of creating a near-duplicate or falling back.
NEAR_DUPLICATE_THRESHOLD = 0.85

_SYSTEM_PROMPT = """\
Du hilfst dabei, gescannte Briefe/Dokumente automatisch in eine bestehende, \
handgepflegte Nextcloud-Ordnerstruktur einzusortieren.

Regeln:
- Bevorzuge IMMER einen bestehenden, passenden Ordner gegenueber dem Anlegen \
eines neuen. Erfinde keine Ordner, die es schon in aehnlicher Form gibt.
- Schlage nur dann einen neuen Ordner vor, wenn wirklich kein bestehender \
Ordner inhaltlich passt. Ein neuer Ordnername muss zum Stil und der Sprache \
der bestehenden Ordner passen (z.B. Grossschreibung, Singular/Plural-Konvention).
- "folder" ist immer der vollstaendige Pfad inklusive des Wurzelordners "Dokumente", \
exakt in der Schreibweise wie in der Ordnerliste unten, z.B. \
"Dokumente/Gesundheit/Krankenkasse" oder "Dokumente/Motorrad/Rechnungen". \
Ein neu vorgeschlagener Ordner muss ebenfalls mit "Dokumente/" beginnen.
- "title" ist NUR ein kurzer, praegnanter Titel ohne Datum, ohne Dateiendung \
und ohne Rechnungs-/Kundennummern, z.B. "Stromrechnung Juli" oder \
"Bussgeldbescheid". Referenznummern gehoeren NIEMALS in den Titel.
- "issue_date" ist das Ausstellungs-/Rechnungsdatum des Dokuments im Format \
YYYY-MM-DD, oder null falls nicht ermittelbar. Deutsche Datumsangaben im Text \
sind TT.MM.JJJJ (Tag zuerst) - wandle sie sorgfaeltig um, ohne Ziffern zu \
vertauschen. Beispiel: "31.07.2026" im Text bedeutet issue_date "2026-07-31" \
(Jahr-Monat-Tag), NICHT "3107-07-20" oder aehnliche Vertauschungen.
- "confidence" ist deine eigene Einschaetzung (0.0-1.0), wie sicher du bei \
Ordner UND Titel bist. Sei ehrlich niedrig, wenn der Text schlecht lesbar \
oder mehrdeutig ist.
- Antworte AUSSCHLIESSLICH mit einem JSON-Objekt passend zum vorgegebenen Schema.
"""


def _build_messages(ocr_text: str, original_filename: str, existing_folders: list[str]) -> list[dict]:
    folder_list = "\n".join(f"- {f}" for f in sorted(existing_folders)) or "(noch keine Ordner vorhanden)"
    truncated_text = ocr_text[:MAX_OCR_CHARS]

    user_prompt = f"""\
Bestehende Ordner unter Dokumente/:
{folder_list}

Urspruenglicher Dateiname des Scans (kann bereits ein Hinweis auf Inhalt/Datum sein):
{original_filename}

Erkannter Text (OCR, ggf. gekuerzt):
---
{truncated_text}
---
"""
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _resolve_folder(result: ClassificationResult, existing_folders: list[str]) -> tuple[ClassificationResult, list[str]]:
    """Cross-checks the model's folder choice against the real folder listing,
    which it cannot be trusted to get exactly right on its own:

    - If it named a real, existing folder: trust it (regardless of whether it
      thought that folder was "new").
    - If not, but a real folder has a near-identical name (e.g. it proposed
      creating "Rechnung" when "Rechnungen" already exists): redirect to the
      existing one instead of creating a near-duplicate or bailing out to the
      fallback folder. This is filed automatically like any other match; the
      redirect is recorded as a log tag so it can be spot-checked later.
    - If it claimed an existing folder that doesn't resemble anything real:
      that's an unreliable hallucination, so confidence is zeroed and it
      falls back to the configured fallback folder instead.

    Returns the (possibly updated) result plus tags for the log.
    """
    tags: list[str] = []
    folder = result.folder
    is_new_folder = result.is_new_folder
    confidence = result.confidence

    if folder in existing_folders:
        is_new_folder = False
    else:
        match = closest_existing_leaf(folder, existing_folders)
        if match is not None and match[1] >= NEAR_DUPLICATE_THRESHOLD:
            tags.append(f"AUTO-REDIRECTED (vorgeschlagen: {folder} -> genutzt: {match[0]})")
            folder = match[0]
            is_new_folder = False
        elif not is_new_folder:
            log.warning(
                "Model claimed existing folder %r but it is not in the live listing "
                "and nothing close matches; treating as unreliable.",
                folder,
            )
            tags.append("HALLUCINATED-FOLDER")
            confidence = 0.0
        # else: is_new_folder=True with no close match -> genuine new folder, kept as-is.

    updated = result.model_copy(update={"folder": folder, "is_new_folder": is_new_folder, "confidence": confidence})
    return updated, tags


def classify(
    ocr_text: str,
    original_filename: str,
    existing_folders: list[str],
    ollama_host: str,
    model: str,
    timeout: float = 120.0,
) -> tuple[ClassificationResult, list[str]]:
    """Ask the local LLM to classify one document. Returns the (confidence-
    adjusted) result plus a list of tags for the log (near-duplicate folder
    warnings, hallucination detection, etc.). Raises on infrastructure
    failures (unreachable Ollama, invalid response) so the caller can treat
    those as transient and retry/fallback accordingly."""
    client = ollama.Client(host=ollama_host, timeout=timeout)
    messages = _build_messages(ocr_text, original_filename, existing_folders)

    response = client.chat(
        model=model,
        messages=messages,
        format=ClassificationResult.model_json_schema(),
        options={"temperature": 0.1},
    )

    raw_content = response["message"]["content"]
    try:
        payload = json.loads(raw_content)
        result = ClassificationResult.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise RuntimeError(f"Model returned invalid classification JSON: {exc}") from exc

    result, tags = _resolve_folder(result, existing_folders)
    return result, tags
