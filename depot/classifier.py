from __future__ import annotations

import json
import logging

import ollama
from pydantic import ValidationError

from depot.models import ClassificationResult
from depot.naming import closest_existing_leaf

log = logging.getLogger(__name__)

MAX_OCR_CHARS = 3500

# Above this similarity ratio, a proposed new folder is treated as a likely
# near-duplicate of an existing one (e.g. "Rechnung" vs "Rechnungen") and its
# confidence is downgraded rather than trusted outright.
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
- "title" ist ein kurzer, praegnanter Titel ohne Datum und ohne Dateiendung, \
z.B. "Stromrechnung Juli" oder "Bussgeldbescheid".
- "issue_date" ist das Ausstellungs-/Rechnungsdatum des Dokuments im Format \
YYYY-MM-DD, oder null falls nicht ermittelbar.
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


def _adjust_confidence(result: ClassificationResult, existing_folders: list[str]) -> tuple[float, list[str]]:
    """Cross-checks the model's self-reported confidence against facts it
    cannot be trusted to judge itself. Returns (adjusted_confidence, tags)."""
    tags: list[str] = []
    confidence = result.confidence

    if not result.is_new_folder and result.folder not in existing_folders:
        log.warning(
            "Model claimed existing folder %r but it is not in the live listing; "
            "treating as unreliable.",
            result.folder,
        )
        tags.append("HALLUCINATED-FOLDER")
        confidence = 0.0

    if result.is_new_folder:
        match = closest_existing_leaf(result.folder, existing_folders)
        if match is not None and match[1] >= NEAR_DUPLICATE_THRESHOLD:
            tags.append(f"SIMILAR-FOLDER-EXISTS? ({match[0]})")
            confidence = min(confidence, 0.4)

    return confidence, tags


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

    adjusted_confidence, tags = _adjust_confidence(result, existing_folders)
    result = result.model_copy(update={"confidence": adjusted_confidence})
    return result, tags
