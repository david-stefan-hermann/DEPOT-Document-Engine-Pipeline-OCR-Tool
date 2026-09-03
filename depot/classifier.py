from __future__ import annotations

import json
import logging
from datetime import date
from typing import NamedTuple

import ollama
from pydantic import ValidationError

from depot.models import ContentExtraction, FolderStepDecision
from depot.naming import closest_existing_leaf

log = logging.getLogger(__name__)

MAX_OCR_CHARS = 3500

# Above this similarity ratio, a proposed folder name is treated as referring
# to an already-existing sibling (e.g. "Rechnung" vs "Rechnungen") and gets
# redirected/corrected instead of creating a near-duplicate or giving up.
NEAR_DUPLICATE_THRESHOLD = 0.85

# Safety cap on how many levels the folder walk will descend. Not the normal
# termination path (that's reaching a leaf or the model saying "stay"/
# "new_folder") - just a guard against something looping pathologically.
MAX_DEPTH = 12

# When the model hallucinates a folder choice (a "descend" target that isn't
# one of the offered children with no close fuzzy match, or a "new_folder"
# with no name), that step's *reported* confidence is not trustworthy - a
# model that confabulates a folder is not meaningfully more reliable when it
# also claims to be 95% sure about it. Hard-cap the step confidence instead
# of trusting the model's own number, so these cases reliably fall through
# to the fallback folder (below CONFIDENCE_THRESHOLD) instead of silently
# landing one level too shallow with a falsely high confidence.
INVALID_CHOICE_CONFIDENCE_CAP = 0.2


class ClassificationOutcome(NamedTuple):
    folder: str
    is_new_folder: bool
    title: str
    confidence: float
    issue_date: date | None = None
    correspondent: str | None = None


_CONTENT_SYSTEM_PROMPT = """\
Du extrahierst Kerninformationen aus einem gescannten Dokument.

Regeln:
- "correspondent" ist der Absender/Aussteller des Dokuments (Firma, Behoerde, \
Institution) - kurz und wiedererkennbar, z.B. "Stadtwerke Muenchen" statt \
"Stadtwerke Muenchen Servicegesellschaft mbH". Steht im Briefkopf/der \
Absenderzeile z.B. "Finanzamt Muenchen" oder nur "Finanzamt", nutze GENAU \
das als correspondent (nicht null, nur weil kein Firmenname im Sinne einer \
GmbH vorliegt - auch Behoerden, Aemter und Kassen sind ein correspondent). \
Nur bei WIRKLICH keinem erkennbaren Absender (z.B. private Notizen) ist \
correspondent null. Der Absender darf NICHT nochmal im "title" wiederholt \
werden.
- "title" ist ein kurzer, praegnanter Betreff OHNE den Absendernamen (der \
steht bereits in "correspondent"), ohne Datum, ohne Dateiendung und ohne \
Rechnungs-/Kundennummern, z.B. "Stromrechnung Juli" oder "Bussgeldbescheid". \
Referenznummern gehoeren NIEMALS in den Titel.
- "issue_date" ist das Ausstellungs-/Rechnungsdatum des Dokuments im Format \
YYYY-MM-DD, oder null falls nicht ermittelbar. Deutsche Datumsangaben im Text \
sind TT.MM.JJJJ (Tag zuerst) - wandle sie sorgfaeltig um, ohne Ziffern zu \
vertauschen. Beispiel: "31.07.2026" im Text bedeutet issue_date "2026-07-31" \
(Jahr-Monat-Tag), NICHT "3107-07-20" oder aehnliche Vertauschungen.
- "confidence" ist deine eigene Einschaetzung (0.0-1.0), wie sicher du bei \
Titel UND Datum bist. Sei ehrlich niedrig, wenn der Text schlecht lesbar \
oder mehrdeutig ist.
- Antworte AUSSCHLIESSLICH mit einem JSON-Objekt passend zum vorgegebenen Schema.
"""

_FOLDER_STEP_SYSTEM_PROMPT = """\
Du hilfst dabei, ein gescanntes Dokument in eine bestehende, handgepflegte \
Nextcloud-Ordnerstruktur einzusortieren - Schritt fuer Schritt, eine Ebene \
nach der anderen.

Du bekommst die AKTUELLE Ordner-Ebene und deren direkte Unterordner. \
Entscheide NUR, was auf DIESER Ebene als naechstes passiert:
- "descend": einer der angebotenen Unterordner passt eindeutig besser als \
die aktuelle Ebene - dann geht es dort eine Ebene tiefer weiter. \
"folder_name" muss EXAKT einem der oben angebotenen Namen entsprechen - \
WICHTIG: erfinde hier NIEMALS einen Namen, der nicht woertlich in der Liste \
steht, auch wenn er passender klaenge. Ist kein Name aus der Liste wirklich \
passend, nutze stattdessen "stay" oder "new_folder".
- "stay": keiner der angebotenen Unterordner passt besser als die aktuelle \
Ebene selbst - das Dokument wird direkt hier abgelegt.
- "new_folder": keiner der angebotenen Unterordner passt, aber ein neuer, \
sinnvoll benannter Unterordner ist hier gerechtfertigt (Stil/Sprache/ \
Gross-Kleinschreibung der bestehenden Ordner beachten) - das ist der \
richtige Weg fuer einen Ordnernamen, der dir zwar sinnvoll erscheint, aber \
NICHT in der Liste der angebotenen Unterordner steht. "folder_name" ist NUR \
der Name des neuen Ordners, kein Pfad.
- "confidence" ist deine Einschaetzung (0.0-1.0), wie sicher du bei DIESER \
EINEN Entscheidung bist.
- Antworte AUSSCHLIESSLICH mit einem JSON-Objekt passend zum vorgegebenen Schema.
"""


def _build_content_messages(ocr_text: str, original_filename: str) -> list[dict]:
    truncated_text = ocr_text[:MAX_OCR_CHARS]
    user_prompt = f"""\
Urspruenglicher Dateiname des Scans (kann bereits ein Hinweis auf Inhalt/Datum sein):
{original_filename}

Erkannter Text (OCR, ggf. gekuerzt):
---
{truncated_text}
---
"""
    return [
        {"role": "system", "content": _CONTENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _build_folder_step_messages(
    ocr_text: str, original_filename: str, current_path: str, children: list[str]
) -> list[dict]:
    children_list = "\n".join(f"- {c}" for c in sorted(children)) or "(keine Unterordner vorhanden)"
    truncated_text = ocr_text[:MAX_OCR_CHARS]
    user_prompt = f"""\
Aktuelle Ebene: {current_path}
Direkte Unterordner dieser Ebene:
{children_list}

Urspruenglicher Dateiname des Scans:
{original_filename}

Erkannter Text (OCR, ggf. gekuerzt):
---
{truncated_text}
---
"""
    return [
        {"role": "system", "content": _FOLDER_STEP_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _children_of(existing_folders: list[str], parent: str) -> list[str]:
    """Direct child leaf names (not full paths) of `parent` within the flat
    folder listing."""
    prefix = f"{parent}/"
    children = set()
    for f in existing_folders:
        if f.startswith(prefix):
            rest = f[len(prefix):]
            if rest and "/" not in rest:
                children.add(rest)
    return sorted(children)


def extract_content(
    ocr_text: str,
    original_filename: str,
    ollama_host: str,
    model: str,
    timeout: float = 120.0,
) -> ContentExtraction:
    client = ollama.Client(host=ollama_host, timeout=timeout)
    messages = _build_content_messages(ocr_text, original_filename)
    response = client.chat(
        model=model,
        messages=messages,
        format=ContentExtraction.model_json_schema(),
        options={"temperature": 0.1},
    )
    raw_content = response["message"]["content"]
    try:
        payload = json.loads(raw_content)
        return ContentExtraction.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise RuntimeError(f"Model returned invalid content-extraction JSON: {exc}") from exc


def _decide_folder_step(
    ocr_text: str,
    original_filename: str,
    current_path: str,
    children: list[str],
    ollama_host: str,
    model: str,
    timeout: float = 120.0,
) -> FolderStepDecision:
    client = ollama.Client(host=ollama_host, timeout=timeout)
    messages = _build_folder_step_messages(ocr_text, original_filename, current_path, children)
    response = client.chat(
        model=model,
        messages=messages,
        format=FolderStepDecision.model_json_schema(),
        options={"temperature": 0.1},
    )
    raw_content = response["message"]["content"]
    try:
        payload = json.loads(raw_content)
        return FolderStepDecision.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise RuntimeError(f"Model returned invalid folder-step JSON: {exc}") from exc


def _walk_folder_tree(
    ocr_text: str,
    original_filename: str,
    existing_folders: list[str],
    dokumente_root: str,
    ollama_host: str,
    model: str,
    timeout: float = 120.0,
) -> tuple[str, bool, float, list[str]]:
    """Descends the Dokumente/ tree one level at a time, asking the model at
    each level to pick a direction from a small, focused candidate set
    (that level's direct children only) instead of the entire tree at once.
    Returns (folder, is_new_folder, confidence, tags)."""
    current_path = dokumente_root
    confidences: list[float] = []
    tags: list[str] = []
    is_new_folder = False

    for _ in range(MAX_DEPTH):
        children = _children_of(existing_folders, current_path)
        if not children:
            break  # leaf reached, nothing to ask about

        decision = _decide_folder_step(
            ocr_text, original_filename, current_path, children, ollama_host, model, timeout
        )
        confidences.append(decision.confidence)

        if decision.action == "stay":
            break

        if decision.action == "descend":
            if decision.folder_name in children:
                current_path = f"{current_path}/{decision.folder_name}"
                continue
            match = closest_existing_leaf(decision.folder_name or "", children)
            if match is not None and match[1] >= NEAR_DUPLICATE_THRESHOLD:
                tags.append(f"AUTO-KORRIGIERT ({decision.folder_name} -> {match[0]})")
                current_path = f"{current_path}/{match[0]}"
                continue
            log.warning(
                "Model chose non-existent child %r at %r with no close match; staying.",
                decision.folder_name, current_path,
            )
            tags.append("UNGUELTIGE-ORDNERWAHL")
            confidences[-1] = min(confidences[-1], INVALID_CHOICE_CONFIDENCE_CAP)
            break

        if decision.action == "new_folder":
            if not decision.folder_name:
                tags.append("UNGUELTIGE-ORDNERWAHL")
                confidences[-1] = min(confidences[-1], INVALID_CHOICE_CONFIDENCE_CAP)
                break
            match = closest_existing_leaf(decision.folder_name, children)
            if match is not None and match[1] >= NEAR_DUPLICATE_THRESHOLD:
                tags.append(f"AUTO-REDIRECTED (vorgeschlagen: {decision.folder_name} -> genutzt: {match[0]})")
                current_path = f"{current_path}/{match[0]}"
            else:
                current_path = f"{current_path}/{decision.folder_name}"
                is_new_folder = True
            break

    confidence = min(confidences) if confidences else 1.0
    return current_path, is_new_folder, confidence, tags


def classify(
    ocr_text: str,
    original_filename: str,
    existing_folders: list[str],
    ollama_host: str,
    model: str,
    dokumente_root: str = "Dokumente",
    timeout: float = 120.0,
) -> tuple[ClassificationOutcome, list[str]]:
    """Classifies one document: extracts title/date independently of the
    folder structure, then walks the Dokumente/ tree level by level to find
    (or create) the right destination folder. Raises on infrastructure
    failures (unreachable Ollama, invalid response) so the caller can treat
    those as transient and retry/fallback accordingly."""
    content = extract_content(ocr_text, original_filename, ollama_host, model, timeout)
    folder, is_new_folder, folder_confidence, tags = _walk_folder_tree(
        ocr_text, original_filename, existing_folders, dokumente_root, ollama_host, model, timeout
    )
    overall_confidence = min(content.confidence, folder_confidence)
    outcome = ClassificationOutcome(
        folder=folder,
        is_new_folder=is_new_folder,
        title=content.title,
        issue_date=content.issue_date,
        correspondent=content.correspondent,
        confidence=overall_confidence,
    )
    return outcome, tags
