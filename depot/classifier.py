from __future__ import annotations

import json
import logging
from datetime import date
from typing import NamedTuple

import anthropic
import ollama
from pydantic import ValidationError

from depot.models import AnthropicFolderDecision, ContentExtraction, FolderStepDecision
from depot.naming import closest_existing_leaf

log = logging.getLogger(__name__)

MAX_OCR_CHARS = 3500

# temperature=0.1 (without a fixed seed) still produced visibly different
# answers for the exact same document across repeated runs (confirmed via a
# live A/B test: the same letter's title flip-flopped between two phrasings
# across 4 runs at temperature=0.1, but was byte-identical across 4 runs at
# temperature=0 + a fixed seed). Since there is no benefit to creative
# variation here - a given document should always file the same way - both
# calls use fully deterministic sampling.
_OLLAMA_OPTIONS = {"temperature": 0.0, "seed": 42}

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

# Above this similarity ratio, an extracted correspondent is treated as
# already having a home somewhere in the existing tree (e.g. correspondent
# "Bucher Grundstuecksservice GmbH" vs. an existing folder leaf "Bucher
# Grundstuecksservice"), and the folder walk starts there directly instead
# of at the Dokumente root. Slightly higher than NEAR_DUPLICATE_THRESHOLD on
# purpose: this searches leaf names across the WHOLE tree (not just a
# handful of siblings at one level), so the larger candidate pool deserves a
# bit more caution against an accidental false-positive match. Real case
# this fixes: a small model, given the top-level folder list alone (no
# insight into what's actually inside each one), confidently but wrongly
# filed salary slips from a clearly-named employer under Finanzen/Vermoegen/
# instead of the employer's own existing folder under Arbeit/ - once there,
# every subsequent level had exactly one child, so "descend" was the only
# option and each step still reported confidence 1.0, masking how wrong the
# very first (real, multi-way) choice was.
CORRESPONDENT_FOLDER_MATCH_THRESHOLD = 0.87


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
Institution) - PFLICHTFELD, darf so gut wie nie leer sein. Kurz und \
wiedererkennbar, z.B. "Stadtwerke Muenchen" statt "Stadtwerke Muenchen \
Servicegesellschaft mbH". Suche AKTIV im Briefkopf, in der Absenderzeile \
oder der Fusszeile nach einem Firmen-/Behoerden-/Kassennamen. Auch Aemter, \
Kassen und Vereine zaehlen als correspondent, nicht nur Firmen im \
GmbH-Sinne: steht im Text z.B. "Finanzamt Muenchen" oder nur "Finanzamt", \
nutze GENAU das. NUR wenn im GESAMTEN Text wirklich kein einziger \
Absenderhinweis existiert (z.B. eine private handschriftliche Notiz ganz \
ohne Briefkopf), ist ein leerer String "" erlaubt - das ist der \
Ausnahmefall, nicht der Normalfall. Der Absender darf NICHT nochmal im \
"title" wiederholt werden.
- "title" ist ein kurzer, praegnanter Betreff OHNE den Absendernamen (der \
steht bereits in "correspondent"), ohne Datum, ohne Dateiendung und ohne \
Rechnungs-/Kundennummern, z.B. "Stromrechnung Juli" oder "Bussgeldbescheid". \
Referenznummern gehoeren NIEMALS in den Titel.
- Hat das Dokument einen offiziellen Formular-/Dokumenttyp-Namen (Rechnung, \
Bescheid, Bescheinigung, Mahnung, Pruefbericht, Vertrag, ...), nutze GENAU \
diesen als Kern des Titels. Ist es dagegen ein freier, persoenlich \
adressierter Brief OHNE einen solchen offiziellen Dokumenttyp (erkennbar an \
"Sehr geehrte(r) ...", einer direkten Anrede, einem freien Anliegen statt \
einem Formular), leite den Titel aus dem TATSAECHLICHEN Anliegen/Thema des \
Brieftexts ab (worum es inhaltlich geht) - NIEMALS eine generische \
Bezeichnung wie "Schreiben", "Mitteilung" oder "Buergerbrief" verwenden, \
die nur die Textsorte statt des Inhalts benennt.
- Bezieht sich das Dokument erkennbar auf ein konkretes physisches Objekt, \
das der Nutzer mehrfach besitzen koennte (z.B. ein Fahrzeug, ein \
Geraet), und steht im Text eine eindeutige Kennung dafuer (amtliches \
Kennzeichen, Seriennummer, Fahrgestellnummer), nimm diese Kennung mit in \
den Titel auf - das unterscheidet sonst gleichnamige Dokumente \
(z.B. "Pruefbericht B-XY 123" statt nur "Pruefbericht"). Das ist KEINE \
Rechnungs-/Kundennummer und faellt nicht unter das Verbot oben.
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

_ANTHROPIC_FOLDER_SYSTEM_PROMPT = """\
Du sortierst ein gescanntes Dokument in eine bestehende, handgepflegte \
Nextcloud-Ordnerstruktur ein.

Du bekommst NUR: den extrahierten Absender, den Titel, und die \
VOLLSTAENDIGE flache Liste aller existierenden Ordnerpfade - bewusst KEINEN \
Dokumentinhalt (Datenschutz: der eigentliche Dokumenttext bleibt lokal).

Antworte als JSON:
- "action": "existing" wenn ein vorhandener Ordner aus der Liste wirklich \
passt, sonst "new_folder".
- "folder": bei "existing" EXAKT einer der Pfade aus der Liste. Bei \
"new_folder" der EXAKT existierende Elternordner (ebenfalls woertlich aus \
der Liste), unter dem der neue Ordner angelegt werden soll - erfinde \
diesen Elternpfad NIEMALS.
- "new_folder_name": nur bei "new_folder" gesetzt, NUR der Name des neuen \
Unterordners (kein Pfad), im Stil der bestehenden Ordner.
- "confidence": deine ehrliche Einschaetzung (0.0-1.0). Ist keine \
Kategorie wirklich eindeutig, wähle eine plausible existierende Ober- \
kategorie (z.B. den passenden Themen-Hauptordner) statt eine falsche \
Unterkategorie zu erraten, und melde entsprechend moderate statt maximale \
Konfidenz.
- "reasoning": kurze Begruendung (1-2 Saetze), warum dieser Ordner passt.
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
        options=_OLLAMA_OPTIONS,
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
        options=_OLLAMA_OPTIONS,
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
    correspondent: str = "",
) -> tuple[str, bool, float, list[str]]:
    """Descends the Dokumente/ tree one level at a time, asking the model at
    each level to pick a direction from a small, focused candidate set
    (that level's direct children only) instead of the entire tree at once.
    Returns (folder, is_new_folder, confidence, tags).

    If `correspondent` closely matches an existing folder's leaf name
    anywhere in the tree, the walk starts there directly instead of at
    dokumente_root - see CORRESPONDENT_FOLDER_MATCH_THRESHOLD for why."""
    current_path = dokumente_root
    confidences: list[float] = []
    tags: list[str] = []
    is_new_folder = False

    if correspondent:
        match = closest_existing_leaf(correspondent, existing_folders)
        if match is not None and match[1] >= CORRESPONDENT_FOLDER_MATCH_THRESHOLD:
            current_path = match[0]
            tags.append(f"ABSENDER-ORDNER-GEFUNDEN ({correspondent} -> {match[0]})")

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


def _build_anthropic_folder_user_content(
    correspondent: str, title: str, existing_folders: list[str]
) -> str:
    folder_list = "\n".join(sorted(existing_folders)) or "(keine Ordner vorhanden)"
    return f"""\
Absender: {correspondent or "(kein Absender erkannt)"}
Titel: {title}

Vollstaendige Liste existierender Ordner ({len(existing_folders)} Stueck):
{folder_list}
"""


def classify_folder_via_anthropic(
    correspondent: str,
    title: str,
    existing_folders: list[str],
    dokumente_root: str,
    anthropic_api_key: str | None,
    anthropic_model: str,
    timeout: float = 60.0,
) -> tuple[str, bool, float, list[str]]:
    """Single-shot cloud classification: unlike _walk_folder_tree, hands the
    WHOLE existing folder tree to the model in one call instead of walking
    it level by level - a frontier model doesn't need the small-model
    workaround that hierarchical descent exists for. Sends ONLY
    correspondent + title + the folder-path list, never the OCR text or
    original filename, so the actual document content never leaves the
    local network.

    On ANY failure (no API key configured, network error, rate limit,
    invalid response), returns confidence=0.0 instead of raising, so the
    caller's existing confidence-threshold check routes the document to the
    fallback folder rather than retrying the whole pipeline run - the
    already-locally-extracted title/correspondent/date are not lost, only
    the filing decision falls back to "needs manual review"."""
    if not anthropic_api_key:
        log.warning("use_anthropic_classifier is enabled but no ANTHROPIC_API_KEY is configured.")
        return dokumente_root, False, 0.0, ["ANTHROPIC-NICHT-ERREICHBAR"]

    try:
        client = anthropic.Anthropic(api_key=anthropic_api_key, timeout=timeout)
        user_content = _build_anthropic_folder_user_content(correspondent, title, existing_folders)
        # No temperature/seed knob here (unlike _OLLAMA_OPTIONS above):
        # current-generation Claude models removed sampling parameters from
        # the API entirely (confirmed against the installed SDK - `create`/
        # `parse` no longer accept temperature/top_p/top_k at all). Some
        # run-to-run variance in the exact folder choice is possible, but
        # observed live to stay within sensible options (e.g. "Finanzen" vs.
        # the more specific "Finanzen/Steuern"), not wrong ones.
        response = client.messages.parse(
            model=anthropic_model,
            max_tokens=1024,
            system=_ANTHROPIC_FOLDER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
            output_format=AnthropicFolderDecision,
        )
        decision = response.parsed_output
    except Exception as exc:
        log.warning("Anthropic folder classification failed (%s); routing to fallback.", exc)
        return dokumente_root, False, 0.0, ["ANTHROPIC-NICHT-ERREICHBAR"]

    if decision.action == "existing":
        if decision.folder in existing_folders:
            return decision.folder, False, decision.confidence, []
        match = closest_existing_leaf(decision.folder, existing_folders)
        if match is not None and match[1] >= NEAR_DUPLICATE_THRESHOLD:
            return match[0], False, decision.confidence, [
                f"AUTO-KORRIGIERT ({decision.folder} -> {match[0]})"
            ]
        log.warning("Anthropic chose non-existent folder %r with no close match.", decision.folder)
        return dokumente_root, False, INVALID_CHOICE_CONFIDENCE_CAP, ["UNGUELTIGE-ORDNERWAHL"]

    # action == "new_folder"
    if not decision.new_folder_name:
        return dokumente_root, False, INVALID_CHOICE_CONFIDENCE_CAP, ["UNGUELTIGE-ORDNERWAHL"]

    parent = decision.folder
    if parent == dokumente_root or parent in existing_folders:
        return f"{parent}/{decision.new_folder_name}", True, decision.confidence, []

    match = closest_existing_leaf(parent, existing_folders)
    if match is not None and match[1] >= NEAR_DUPLICATE_THRESHOLD:
        return f"{match[0]}/{decision.new_folder_name}", True, decision.confidence, [
            f"AUTO-KORRIGIERT ({parent} -> {match[0]})"
        ]
    log.warning("Anthropic proposed new folder under non-existent parent %r.", parent)
    return dokumente_root, False, INVALID_CHOICE_CONFIDENCE_CAP, ["UNGUELTIGE-ORDNERWAHL"]


def classify_via_anthropic(
    ocr_text: str,
    original_filename: str,
    existing_folders: list[str],
    ollama_host: str,
    model: str,
    anthropic_api_key: str | None,
    anthropic_model: str,
    dokumente_root: str = "Dokumente",
    timeout: float = 120.0,
) -> tuple[ClassificationOutcome, list[str]]:
    """Same contract as classify(), but the folder decision is delegated to
    Anthropic (classify_folder_via_anthropic) instead of the local
    hierarchical walk. title/correspondent/issue_date extraction still runs
    fully locally via extract_content() - only correspondent+title+folder
    names ever reach the cloud call."""
    content = extract_content(ocr_text, original_filename, ollama_host, model, timeout)
    folder, is_new_folder, folder_confidence, tags = classify_folder_via_anthropic(
        content.correspondent, content.title, existing_folders, dokumente_root,
        anthropic_api_key, anthropic_model,
    )
    overall_confidence = min(content.confidence, folder_confidence)
    outcome = ClassificationOutcome(
        folder=folder,
        is_new_folder=is_new_folder,
        title=content.title,
        issue_date=content.issue_date,
        correspondent=content.correspondent or None,
        confidence=overall_confidence,
    )
    return outcome, tags


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
        ocr_text, original_filename, existing_folders, dokumente_root, ollama_host, model, timeout,
        correspondent=content.correspondent,
    )
    overall_confidence = min(content.confidence, folder_confidence)
    outcome = ClassificationOutcome(
        folder=folder,
        is_new_folder=is_new_folder,
        title=content.title,
        issue_date=content.issue_date,
        correspondent=content.correspondent or None,
        confidence=overall_confidence,
    )
    return outcome, tags
