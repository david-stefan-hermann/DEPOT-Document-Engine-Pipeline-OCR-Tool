from __future__ import annotations

import json
import logging
import re
from datetime import date
from typing import Callable, NamedTuple

import anthropic
import ollama
from pydantic import ValidationError

from depot import candidates as candidate_search
from depot.candidates import STRONG_EVIDENCE, Candidate, DocumentQuery
from depot.embeddings import Embedder, cosine
from depot.models import (
    AnthropicFolderDecision,
    ContentExtraction,
    FolderPick,
    FolderStepDecision,
    extraction_json_schema,
)
from depot.naming import closest_existing_leaf, known_correspondents, normalize_correspondent, restore_umlauts

log = logging.getLogger(__name__)

MAX_OCR_CHARS = 3500

# temperature=0.1 (without a fixed seed) still produced visibly different
# answers for the exact same document across repeated runs (confirmed via a
# live A/B test: the same letter's title flip-flopped between two phrasings
# across 4 runs at temperature=0.1, but was byte-identical across 4 runs at
# temperature=0 + a fixed seed). Since there is no benefit to creative
# variation here - a given document should always file the same way - both
# calls use fully deterministic sampling.
#
# num_ctx is pinned rather than left to the server default (4096 when this
# was written): the extraction prompt alone measured ~2900 tokens, and a
# prompt that outgrows the context is silently truncated by Ollama. 8192
# was verified to still sit fully in VRAM on the 6 GB card (4.99 GB). It
# must be identical for every call - Ollama reloads the model (~5 s)
# whenever it changes.
_OLLAMA_OPTIONS = {"temperature": 0.0, "seed": 42, "num_ctx": 8192}

# How long Ollama keeps the model in memory after a call. The server default
# (5 min) meant the first document of nearly every batch paid a measured
# ~11 s model load.
OLLAMA_KEEP_ALIVE = "30m"

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

_YEAR_FOLDER = re.compile(r"(19|20)\d{2}")

# The model's own confidence says little (it reports 0.9+ for right and
# wrong answers alike), so how sure a filing decision is follows from what
# backs it instead:
# - the chosen folder already holds documents of this sender/kind;
CONFIDENCE_BACKED = 0.9
# - words and meaning agree: the chosen folder is the best match both by
#   word overlap and for the embedding model (two independent signals).
#   Measured on 120 documents: where both pointed at the same folder it was
#   the right one in 63 of 69 cases (not counting documents the owner had
#   deliberately filed somewhere unusual) - as reliable as sender evidence;
CONFIDENCE_AGREED = 0.8
# - a new subfolder inside a backed folder;
CONFIDENCE_BACKED_NEW_FOLDER = 0.7
# - nothing in the tree supports the choice: a plausible suggestion, but
#   below the default CONFIDENCE_THRESHOLD, so the document goes to the
#   review folder (with the suggestion in its log line) rather than being
#   filed on a guess. Once it has been filed by hand, the next document of
#   that kind finds it there.
CONFIDENCE_UNBACKED = 0.5

# With strong candidates present, weaker ones are only distraction.
_SHOWN_SCORE_SHARE = 0.4
_MAX_SHOWN_WITH_STRONG = 6


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
- "correspondent" ist der Absender/Aussteller des Dokuments (Firma, Behörde, \
Institution) - PFLICHTFELD, darf so gut wie nie leer sein. Kurz und \
wiedererkennbar, z.B. "Stadtwerke München" statt "Stadtwerke München \
Servicegesellschaft mbH". Suche AKTIV im Briefkopf, in der Absenderzeile \
oder der Fußzeile nach einem Firmen-/Behörden-/Kassennamen. Auch Ämter, \
Kassen und Vereine zählen als correspondent, nicht nur Firmen im \
GmbH-Sinne: steht im Text z.B. "Finanzamt München" oder nur "Finanzamt", \
nutze GENAU das. NUR wenn im GESAMTEN Text wirklich kein einziger \
Absenderhinweis existiert (z.B. eine private handschriftliche Notiz ganz \
ohne Briefkopf), ist ein leerer String "" erlaubt - das ist der \
Ausnahmefall, nicht der Normalfall. Der Absender darf NICHT nochmal im \
"title" wiederholt werden.
- "title" ist ein kurzer, prägnanter Betreff OHNE den Absendernamen (der \
steht bereits in "correspondent"), ohne das Ausstellungsdatum, ohne \
Dateiendung und ohne Rechnungs-/Kundennummern, z.B. "Stromrechnung Juli 2026". \
Referenznummern gehören NIEMALS in den Titel.
- Schreibe Titel und Absender mit echten Umlauten und ß, so wie sie im \
Dokument stehen ("Änderung", "über", "Bußgeld") - NIEMALS die Ersatzschreibung \
"ae", "oe", "ue", "ss". E-Mail-Adressen, Internetadressen und Telefonnummern \
gehören weder in den Absender noch in den Titel.
- Gilt das Dokument für einen bestimmten ZEITRAUM (Abrechnungsmonat, Quartal, \
Jahr) - z.B. Gehalts-/Entgeltabrechnung, Zuzahlungsrechnung, Kontoauszug, \
Nebenkostenabrechnung, Beitragsrechnung -, dann gehört dieser Zeitraum IMMER \
ausgeschrieben in den Titel: "Entgeltabrechnung August 2026", \
"Zuzahlungsrechnung Juli 2026", "Nebenkostenabrechnung 2025". Der Zeitraum \
ist das, was mehrere gleichnamige Dokumente desselben Absenders unterscheidet; \
er ist NICHT das Ausstellungsdatum.
- Bei Bußgeldbescheiden, Verwarnungen und Anhörungen im Straßenverkehr gehören \
das amtliche Kennzeichen und der geforderte Gesamtbetrag in den Titel: \
"Bußgeldbescheid B-XY 123 - 28,50 EUR".
- Hat das Dokument einen offiziellen Formular-/Dokumenttyp-Namen (Rechnung, \
Bescheid, Bescheinigung, Mahnung, Prüfbericht, Vertrag, ...), nutze GENAU \
diesen als Kern des Titels. Ist es dagegen ein freier, persönlich \
adressierter Brief OHNE einen solchen offiziellen Dokumenttyp (erkennbar an \
"Sehr geehrte(r) ...", einer direkten Anrede, einem freien Anliegen statt \
einem Formular), leite den Titel aus dem TATSÄCHLICHEN Anliegen/Thema des \
Brieftexts ab (worum es inhaltlich geht) - NIEMALS eine generische \
Bezeichnung wie "Schreiben", "Mitteilung" oder "Bürgerbrief" verwenden, \
die nur die Textsorte statt des Inhalts benennt.
- Bezieht sich das Dokument erkennbar auf ein konkretes physisches Objekt, \
das der Nutzer mehrfach besitzen könnte (z.B. ein Fahrzeug, ein \
Gerät), und steht im Text eine eindeutige Kennung dafür (amtliches \
Kennzeichen, Seriennummer, Fahrgestellnummer), nimm diese Kennung mit in \
den Titel auf - das unterscheidet sonst gleichnamige Dokumente \
(z.B. "Prüfbericht B-XY 123" statt nur "Prüfbericht"). Das ist KEINE \
Rechnungs-/Kundennummer und fällt nicht unter das Verbot oben.
- "issue_date" ist das Ausstellungs-/Erstellungsdatum DIESES Dokuments selbst \
(wann es geschrieben/gedruckt/verschickt wurde) - NICHT irgendein anderes \
Datum, das im Dokument zufällig vorkommt. Formulare wie Gehalts-/ \
Entgeltabrechnungen enthalten oft MEHRERE Datumsangaben, die NICHTS mit dem \
Ausstellungsdatum zu tun haben: Geburtsdatum, Eintrittsdatum, Austrittsdatum, \
Referenzdatum u.ä. - diese Personaldaten sind NIEMALS issue_date, auch wenn \
sie im selben Zeitraum liegen wie das Dokument. Suche stattdessen gezielt \
nach einem Feld, das wörtlich "Datum" heißt (oft in einer Kopfzeile nahe \
"Seite"/"Kundennummer"/"Kostenstelle") oder dem Datum am Ende/in der \
Fußzeile des Schreibens. Bist du zwischen mehreren Datumsangaben unsicher, \
welches das echte Ausstellungsdatum ist, setze issue_date auf null und \
senke die confidence, statt zu raten oder Ziffern aus verschiedenen Daten \
zu vermischen. Format YYYY-MM-DD, oder null falls nicht sicher ermittelbar. \
Deutsche Datumsangaben im Text sind TT.MM.JJJJ (Tag zuerst) - wandle sie \
sorgfältig um, ohne Ziffern zu vertauschen. Beispiel: "31.07.2026" im Text \
bedeutet issue_date "2026-07-31" (Jahr-Monat-Tag), NICHT "3107-07-20" oder \
ähnliche Vertauschungen.
- "keywords" sind 3 bis 6 allgemeine Stichworte zu Dokumentart und Thema \
(z.B. ["Rechnung", "Strom", "Jahresabrechnung"]), die helfen, das Dokument \
einer Ablage-Kategorie zuzuordnen. KEINE personenbezogenen Angaben: keine \
Personennamen, Adressen, Nummern, Beträge oder Datumsangaben.
- "summary" beschreibt in zwei bis drei kurzen Sätzen, WORUM es in dem \
Dokument geht und zu welchem Lebensbereich/Gegenstand es gehört (z.B. \
"Monatliche Gehaltsabrechnung eines Arbeitgebers." oder "Schreiben einer \
Kfz-Versicherung zur Beitragsänderung für ein Motorrad."). Fahrzeugmodell, \
Gerätetyp oder Produktname dürfen genannt werden. KEINE personenbezogenen \
Angaben: keine Personennamen, Adressen, Kennzeichen, Nummern, Beträge oder \
Datumsangaben.
- "confidence" ist deine eigene Einschätzung (0.0-1.0), wie sicher du bei \
Titel UND Datum bist. Sei ehrlich niedrig, wenn der Text schlecht lesbar \
oder mehrdeutig ist.
- Antworte AUSSCHLIESSLICH mit einem JSON-Objekt passend zum vorgegebenen Schema.
"""

_ANTHROPIC_FOLDER_SYSTEM_PROMPT = """\
Du sortierst ein gescanntes Dokument in eine bestehende, handgepflegte \
Nextcloud-Ordnerstruktur ein.

Du bekommst NUR: den extrahierten Absender, den Titel, ggf. den Dateinamen \
bzw. PDF-Titel, den der Nutzer/Aussteller dem Dokument gegeben hat, ein \
paar allgemeine Stichworte, eine kurze lokal erzeugte Zusammenfassung, und \
die VOLLSTAENDIGE flache Liste aller existierenden Ordnerpfade - bewusst \
KEINEN Dokumentinhalt (Datenschutz: der eigentliche Dokumenttext bleibt \
lokal). Fehlt der Absender, stuetze dich auf Dateiname, PDF-Titel, \
Stichworte und Zusammenfassung.

Unter "Hinweise aus der bestehenden Ablage" steht, was lokal ueber den \
Inhalt der Ordner bekannt ist: in welchen Ordnern bereits Dokumente \
DESSELBEN ABSENDERS liegen und welche Ordner aehnliche Dokumente \
enthalten. Das ist das staerkste Signal ueberhaupt - der Nutzer legt \
Dokumente eines Absenders fast immer wieder am selben Ort ab, auch wenn \
der Ordner anders heisst als der Absender (z.B. Abrechnungen einer \
Depotbank im Ordner des Brokers). Liegen dort schon Dokumente desselben \
Absenders, waehle diesen Ordner und lege KEINEN neuen an, ausser Titel und \
Zusammenfassung sprechen eindeutig fuer ein voellig anderes Thema.

Reine Jahres-Unterordner (2024, 2025, ...) fehlen in der Liste absichtlich; \
das Jahr wird nach deiner Wahl automatisch aus dem Dokumentdatum bestimmt. \
Schlage niemals einen Jahresordner als neuen Ordner vor.

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

Du bekommst den erkannten Text, den urspruenglichen Dateinamen, den bereits \
ermittelten Absender und Titel des Dokuments sowie die AKTUELLE Ordner-Ebene \
und deren direkte Unterordner. Nutze Titel, Absender und Dateiname ZUSAMMEN \
mit dem Text; ist kein Text vorhanden (z.B. bei einem Foto), entscheide \
allein anhand von Titel und Dateiname. \
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


def _build_content_messages(
    ocr_text: str, original_filename: str, pdf_title: str | None = None
) -> list[dict]:
    truncated_text = ocr_text[:MAX_OCR_CHARS]
    pdf_title_line = f"\nTitel laut PDF-Metadaten: {pdf_title}" if pdf_title else ""
    user_prompt = f"""\
Urspruenglicher Dateiname des Scans (kann bereits ein Hinweis auf Inhalt/Datum sein):
{original_filename}{pdf_title_line}

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
    ocr_text: str,
    original_filename: str,
    current_path: str,
    children: list[str],
    title: str = "",
    correspondent: str = "",
    pdf_title: str | None = None,
) -> list[dict]:
    children_list = "\n".join(f"- {c}" for c in sorted(children)) or "(keine Unterordner vorhanden)"
    truncated_text = ocr_text[:MAX_OCR_CHARS] or "(kein Text erkannt)"
    pdf_title_line = f"Titel laut PDF-Metadaten: {pdf_title}\n" if pdf_title else ""
    # Order matters for speed: everything that stays the same for one
    # document across all steps of the walk (the long text) comes before the
    # part that changes per step, so Ollama reuses the already evaluated
    # prefix - measured ~0.1 s instead of ~4.6 s prompt evaluation for every
    # step after the first.
    #
    # It is split into two turns on purpose. Simply appending the level to
    # the document in ONE message made the model answer "stay" at the root
    # for 3 of 4 real documents that the old level-first prompt routed
    # correctly; as a separate final turn it decided like the old prompt
    # again. Don't merge these without re-running tools/eval.py.
    document_prompt = f"""\
Hier ist das Dokument, das einsortiert werden soll.

Erkannter Text (OCR, ggf. gekuerzt):
---
{truncated_text}
---

Urspruenglicher Dateiname des Scans: {original_filename}
{pdf_title_line}Absender des Dokuments: {correspondent or "(nicht erkannt)"}
Titel des Dokuments: {title or "(nicht ermittelt)"}
"""
    level_prompt = f"""\
Aktuelle Ebene: {current_path}
Direkte Unterordner dieser Ebene:
{children_list}
"""
    return [
        {"role": "system", "content": _FOLDER_STEP_SYSTEM_PROMPT},
        {"role": "user", "content": document_prompt},
        {"role": "assistant", "content": "Verstanden. Nenne mir die aktuelle Ebene und ihre Unterordner."},
        {"role": "user", "content": level_prompt},
    ]


_PICK_TASK = """\
Neue Aufgabe zum selben Dokument: Es soll in eine bestehende, handgepflegte \
Ordnerstruktur einsortiert werden. Waehle den Ordner, in den es gehoert.

{candidate_block}Hauptordner der Ablage{fallback_note}:
{top_level_block}

Regeln:
- "folder" ist der vollstaendige Pfad GENAU EINES Ordners aus den Listen \
oben, woertlich uebernommen.
- Das Dokument gehoert dorthin, wo bereits gleichartige Dokumente liegen: \
gleicher Absender, gleiches Thema oder derselbe Gegenstand (z.B. dasselbe \
Fahrzeug, derselbe Vertrag). Die Beispieldateien zeigen, was in einem Ordner \
tatsaechlich abgelegt ist - sie sind aussagekraeftiger als der Ordnername.
- Waehle den SPEZIFISCHSTEN passenden Ordner. Einen Hauptordner nur, wenn \
keiner der Ordner mit Beispielen wirklich passt.
- "new_folder_name": im Normalfall null. Nur setzen, wenn das Dokument zwar \
in den gewaehlten Ordner gehoert, dort aber ein NEUER Unterordner dafuer \
angelegt werden soll (z.B. ein neuer Absender oder Vorgang) - dann NUR der \
Name des neuen Unterordners, kein Pfad, im Stil der bestehenden Ordner.
- "confidence": deine ehrliche Einschaetzung (0.0-1.0). Niedrig, wenn \
mehrere Ordner aehnlich gut passen oder keiner wirklich passt.
- Antworte AUSSCHLIESSLICH mit einem JSON-Objekt passend zum vorgegebenen Schema.
"""


def _document_turns(
    ocr_text: str, original_filename: str, pdf_title: str | None, content: ContentExtraction
) -> list[dict]:
    """The extraction conversation, replayed: the same system prompt and
    document message extract_content() sent, followed by what was extracted.
    A follow-up task appended to this shares its whole prefix with the
    extraction call that ran just before it, so Ollama does not evaluate
    the document text a second time - and the follow-up sees the document's
    text AND its title/correspondent."""
    extracted = {
        "title": content.title,
        "correspondent": content.correspondent,
        "issue_date": content.issue_date.isoformat() if content.issue_date else None,
        "keywords": content.keywords,
    }
    return [
        *_build_content_messages(ocr_text, original_filename, pdf_title),
        {"role": "assistant", "content": json.dumps(extracted, ensure_ascii=False)},
    ]


def _build_pick_messages(
    ocr_text: str,
    original_filename: str,
    pdf_title: str | None,
    content: ContentExtraction,
    ranked: list[Candidate],
    top_level: list[str],
) -> tuple[list[dict], list[str]]:
    """Messages for the one decision "which of these folders", plus the
    offered folder paths (candidates first, then the remaining top-level
    folders)."""
    options: list[str] = []
    candidate_lines: list[str] = []
    for candidate in ranked:
        options.append(candidate.path)
        candidate_lines.append(f"- {candidate.path}")
        candidate_lines += [f"    z.B. {example[:90]}" for example in candidate.examples]
    top_level_lines = []
    for path in top_level:
        if path in options:
            continue
        options.append(path)
        top_level_lines.append(f"- {path}")

    candidate_block = ""
    if candidate_lines:
        candidate_block = (
            "Ordner, in denen bereits aehnliche Dokumente liegen (darunter jeweils Beispiele dort "
            "abgelegter Dateien):\n" + "\n".join(candidate_lines) + "\n\n"
        )
    task = _PICK_TASK.format(
        candidate_block=candidate_block,
        fallback_note=" (falls keiner der Ordner oben passt)" if candidate_lines else "",
        top_level_block="\n".join(top_level_lines) or "(keine weiteren)",
    )
    messages = [*_document_turns(ocr_text, original_filename, pdf_title, content), {"role": "user", "content": task}]
    return messages, options


def _pick_folder(
    ocr_text: str,
    original_filename: str,
    pdf_title: str | None,
    content: ContentExtraction,
    ranked: list[Candidate],
    existing_folders: list[str],
    dokumente_root: str,
    ollama_host: str,
    model: str,
    timeout: float = 120.0,
) -> tuple[str, bool, float, list[str], Candidate | None]:
    """One model call choosing among the ranked candidates (with example
    files) and the top-level folders. Returns (folder, is_new_folder,
    confidence, tags, chosen candidate or None if a top-level folder was
    chosen). The answer is constrained by the schema to the offered paths,
    so an invented folder is impossible by construction."""
    top_level = [f"{dokumente_root}/{name}" for name in _children_of(existing_folders, dokumente_root)]
    messages, options = _build_pick_messages(ocr_text, original_filename, pdf_title, content, ranked, top_level)
    if not options:
        return dokumente_root, False, 1.0, [], None

    schema = FolderPick.model_json_schema()
    schema["properties"]["folder"] = {"type": "string", "enum": options}
    raw_content = _chat("folder-pick", ollama_host, model, messages, schema, timeout)
    try:
        pick = FolderPick.model_validate(json.loads(raw_content))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise RuntimeError(f"Model returned invalid folder-pick JSON: {exc}") from exc

    if pick.folder not in options:
        # Unreachable while the schema is enforced; never trust it blindly.
        return dokumente_root, False, INVALID_CHOICE_CONFIDENCE_CAP, ["UNGUELTIGE-ORDNERWAHL"], None
    folder = pick.folder
    chosen = next((c for c in ranked if c.path == folder), None)
    tags: list[str] = []

    if pick.new_folder_name:
        siblings = _children_of(existing_folders, folder)
        match = closest_existing_leaf(pick.new_folder_name, siblings)
        if match is not None and match[1] >= NEAR_DUPLICATE_THRESHOLD:
            tags.append(f"AUTO-REDIRECTED (vorgeschlagen: {pick.new_folder_name} -> genutzt: {match[0]})")
            return f"{folder}/{match[0]}", False, pick.confidence, tags, chosen
        return f"{folder}/{pick.new_folder_name}", True, pick.confidence, tags, chosen
    return folder, False, pick.confidence, tags, chosen


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


def _chat(label: str, ollama_host: str, model: str, messages: list[dict], schema: dict, timeout: float) -> str:
    client = ollama.Client(host=ollama_host, timeout=timeout)
    response = client.chat(
        model=model,
        messages=messages,
        format=schema,
        options=_OLLAMA_OPTIONS,
        keep_alive=OLLAMA_KEEP_ALIVE,
    )
    ns = 1e9
    log.info(
        "ollama %s: load=%.1fs prompt=%s tok/%.1fs output=%s tok/%.1fs",
        label,
        (response.get("load_duration") or 0) / ns,
        response.get("prompt_eval_count"),
        (response.get("prompt_eval_duration") or 0) / ns,
        response.get("eval_count"),
        (response.get("eval_duration") or 0) / ns,
    )
    return response["message"]["content"]


def preload_model(ollama_host: str, model: str, timeout: float = 120.0) -> None:
    """Asks Ollama to load the model without generating anything, so a cold
    model loads while OCR is still running instead of afterwards. Best
    effort only - a failure here just means the first real call loads it."""
    try:
        ollama.Client(host=ollama_host, timeout=timeout).generate(
            model=model,
            prompt="",
            options={"num_ctx": _OLLAMA_OPTIONS["num_ctx"]},
            keep_alive=OLLAMA_KEEP_ALIVE,
        )
    except Exception as exc:
        log.debug("Model preload failed (ignored): %s", exc)


def extract_content(
    ocr_text: str,
    original_filename: str,
    ollama_host: str,
    model: str,
    timeout: float = 120.0,
    pdf_title: str | None = None,
    summary: bool = False,
) -> ContentExtraction:
    """`summary`: also have the model write the short summary the cloud
    classifier gets - skipped otherwise, nothing local reads it."""
    messages = _build_content_messages(ocr_text, original_filename, pdf_title)
    raw_content = _chat("extract", ollama_host, model, messages, extraction_json_schema(summary), timeout)
    try:
        payload = json.loads(raw_content)
        content = ContentExtraction.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise RuntimeError(f"Model returned invalid content-extraction JSON: {exc}") from exc
    return content.model_copy(update={
        "title": restore_umlauts(content.title, ocr_text),
        "correspondent": restore_umlauts(content.correspondent, ocr_text),
    })


def _decide_folder_step(
    ocr_text: str,
    original_filename: str,
    current_path: str,
    children: list[str],
    ollama_host: str,
    model: str,
    timeout: float = 120.0,
    title: str = "",
    correspondent: str = "",
    pdf_title: str | None = None,
) -> FolderStepDecision:
    messages = _build_folder_step_messages(
        ocr_text, original_filename, current_path, children, title, correspondent, pdf_title
    )
    raw_content = _chat(
        "folder-step", ollama_host, model, messages, FolderStepDecision.model_json_schema(), timeout
    )
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
    title: str = "",
    pdf_title: str | None = None,
    start_path: str | None = None,
    issue_date: date | None = None,
    by_year_only: bool = False,
) -> tuple[str, bool, float, list[str]]:
    """Descends the Dokumente/ tree one level at a time, asking the model at
    each level to pick a direction from a small, focused candidate set
    (that level's direct children only) instead of the entire tree at once.
    Returns (folder, is_new_folder, confidence, tags).

    The descent begins at `start_path` (an already chosen folder), or at
    dokumente_root without one.

    Folders that are split purely by year are not a question for the model:
    with a known `issue_date` the matching year is entered (or created)
    directly. With `by_year_only` that is all that happens - no model call,
    for a start folder that was already chosen on evidence."""
    current_path = start_path or dokumente_root
    confidences: list[float] = []
    tags: list[str] = []
    is_new_folder = False

    for _ in range(MAX_DEPTH):
        children = _children_of(existing_folders, current_path)
        if not children:
            break  # leaf reached, nothing to ask about

        if issue_date is not None and all(_YEAR_FOLDER.fullmatch(c) for c in children):
            year = str(issue_date.year)
            if year in children:
                current_path = f"{current_path}/{year}"
                continue
            if len(children) >= 2:
                current_path = f"{current_path}/{year}"
                is_new_folder = True
                break
        if by_year_only:
            break

        decision = _decide_folder_step(
            ocr_text, original_filename, current_path, children, ollama_host, model, timeout,
            title=title, correspondent=correspondent, pdf_title=pdf_title,
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
    correspondent: str,
    title: str,
    existing_folders: list[str],
    filename_title: str | None = None,
    pdf_title: str | None = None,
    keywords: list[str] | tuple[str, ...] = (),
    summary: str = "",
    hints: list[str] | tuple[str, ...] = (),
) -> str:
    folder_list = "\n".join(sorted(existing_folders)) or "(keine Ordner vorhanden)"
    extra = ""
    if filename_title:
        extra += f"Dateiname des Scans: {filename_title}\n"
    if pdf_title:
        extra += f"PDF-Titel: {pdf_title}\n"
    if keywords:
        extra += f"Stichworte: {', '.join(keywords)}\n"
    if summary:
        extra += f"Zusammenfassung: {summary}\n"
    extra += "\nHinweise aus der bestehenden Ablage:\n" + (
        "\n".join(f"- {hint}" for hint in hints) if hints
        else "- (nichts Passendes gefunden: weder Dokumente dieses Absenders noch aehnliche)"
    ) + "\n"
    return f"""\
Absender: {correspondent or "(kein Absender erkannt)"}
Titel: {title}
{extra}
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
    filename_title: str | None = None,
    pdf_title: str | None = None,
    keywords: list[str] | tuple[str, ...] = (),
    summary: str = "",
    hints: list[str] | tuple[str, ...] = (),
) -> tuple[str, bool, float, list[str]]:
    """Single-shot cloud classification: unlike _walk_folder_tree, hands the
    WHOLE existing folder tree to the model in one call instead of walking
    it level by level - a frontier model doesn't need the small-model
    workaround that hierarchical descent exists for. Sends ONLY
    correspondent + title + the folder-path list, plus (when available) the
    name the user gave the file, the PDF's metadata title, a few locally
    generated general topic keywords, a locally generated two-sentence
    summary (no names/numbers) and `hints` - which folders already hold
    documents of this sender or similar ones, as counts, never as file
    names. Never the OCR text, so the actual document content never leaves
    the local network.

    On ANY failure (no API key configured, network error, rate limit,
    invalid response), returns confidence=0.0 instead of raising, so the
    caller's existing confidence-threshold check routes the document to the
    fallback folder rather than retrying the whole pipeline run - the
    already-locally-extracted title/correspondent/date are not lost, only
    the filing decision falls back to "needs manual review"."""
    if not anthropic_api_key:
        log.warning("The cloud classifier is switched on but no ANTHROPIC_API_KEY is configured.")
        return dokumente_root, False, 0.0, ["ANTHROPIC-NICHT-ERREICHBAR"]

    try:
        client = anthropic.Anthropic(api_key=anthropic_api_key, timeout=timeout)
        user_content = _build_anthropic_folder_user_content(
            correspondent, title, existing_folders, filename_title, pdf_title, keywords, summary, hints
        )
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

    # A year folder named anyway (they are not offered, see
    # classify_via_anthropic) means the folder above it: the year follows
    # from the document's date, not from the model.
    folder, action = decision.folder, decision.action
    above, _, leaf = folder.rpartition("/")
    if folder not in existing_folders and above and _YEAR_FOLDER.fullmatch(leaf):
        folder = above
    if action == "new_folder" and folder != dokumente_root and _YEAR_FOLDER.fullmatch(decision.new_folder_name or ""):
        action = "existing"

    if action == "existing":
        if folder in existing_folders:
            return folder, False, decision.confidence, []
        match = closest_existing_leaf(folder, existing_folders)
        if match is not None and match[1] >= NEAR_DUPLICATE_THRESHOLD:
            return match[0], False, decision.confidence, [
                f"AUTO-KORRIGIERT ({folder} -> {match[0]})"
            ]
        log.warning("Anthropic chose non-existent folder %r with no close match.", folder)
        return dokumente_root, False, INVALID_CHOICE_CONFIDENCE_CAP, ["UNGUELTIGE-ORDNERWAHL"]

    # action == "new_folder"
    if not decision.new_folder_name:
        return dokumente_root, False, INVALID_CHOICE_CONFIDENCE_CAP, ["UNGUELTIGE-ORDNERWAHL"]

    parent = folder
    if parent == dokumente_root or parent in existing_folders:
        return f"{parent}/{decision.new_folder_name}", True, decision.confidence, []

    match = closest_existing_leaf(parent, existing_folders)
    if match is not None and match[1] >= NEAR_DUPLICATE_THRESHOLD:
        return f"{match[0]}/{decision.new_folder_name}", True, decision.confidence, [
            f"AUTO-KORRIGIERT ({parent} -> {match[0]})"
        ]
    log.warning("Anthropic proposed new folder under non-existent parent %r.", parent)
    return dokumente_root, False, INVALID_CHOICE_CONFIDENCE_CAP, ["UNGUELTIGE-ORDNERWAHL"]


def _cloud_hints(ranked: list[Candidate]) -> list[str]:
    """The local shortlist as the cloud gets it: folder paths (which it has
    anyway) with how many documents of this sender are filed there and how
    strong the overall match is - no file names, no document content."""
    hints = []
    for candidate in ranked:
        parts = []
        if candidate.sender_files:
            parts.append(
                "1 Dokument desselben Absenders" if candidate.sender_files == 1
                else f"{candidate.sender_files} Dokumente desselben Absenders"
            )
        if candidate.strength >= STRONG_EVIDENCE:
            parts.append("starke Aehnlichkeit zu den dort abgelegten Dokumenten")
        elif not parts:
            parts.append("schwache Aehnlichkeit")
        hints.append(f"{candidate.path}: {', '.join(parts)}")
    return hints


def _shortlist(
    content: ContentExtraction,
    ocr_text: str,
    existing_folders: list[str],
    folder_files: dict[str, list[str]] | None,
    dokumente_root: str,
    filename_title: str | None,
    pdf_title: str | None,
    embedder: Embedder | None,
) -> tuple[list[Candidate], dict[str, float] | None]:
    """The folders that already hold similar documents, best first
    (candidates.rank_candidates; with an `embedder`, blended with semantic
    similarity), and those similarities (None without an embedder)."""
    query = DocumentQuery(
        correspondent=content.correspondent,
        title=content.title,
        keywords=content.keywords,
        filename_title=filename_title or "",
        pdf_title=pdf_title or "",
        text=ocr_text,
    )
    semantic = None
    if embedder is not None:
        semantic = semantic_similarities(embedder, query, existing_folders, folder_files, dokumente_root)
    return candidate_search.rank_candidates(query, existing_folders, folder_files, semantic=semantic), semantic


def _prepare_content(
    content: ContentExtraction,
    folder_files: dict[str, list[str]] | None,
    resolve_date: Callable[[date | None], date | None] | None,
) -> ContentExtraction:
    """The extraction as used for filing: sender in its canonical spelling,
    issue date checked against the document."""
    return content.model_copy(update={
        "correspondent": normalize_correspondent(
            content.correspondent, known_correspondents(folder_files or {})
        ),
        "issue_date": resolve_date(content.issue_date) if resolve_date else content.issue_date,
    })


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
    content: ContentExtraction | None = None,
    filename_title: str | None = None,
    pdf_title: str | None = None,
    folder_files: dict[str, list[str]] | None = None,
    resolve_date: Callable[[date | None], date | None] | None = None,
    embedder: Embedder | None = None,
) -> tuple[ClassificationOutcome, list[str]]:
    """Same contract as classify(), but the folder decision is delegated to
    Anthropic (classify_folder_via_anthropic) instead of the local model.
    title/correspondent/issue_date/summary extraction still runs fully
    locally via extract_content() - see classify_folder_via_anthropic for
    exactly what reaches the cloud call.

    The local candidate search runs here too, as the source of the hints:
    which folders already hold documents of this sender or similar ones.
    `folder_files` itself stays local - no filename of an already filed
    document is ever sent, only counts per folder. Year subfolders are not
    offered to the cloud at all; the year follows from the document's date,
    as in classify()."""
    if content is None:
        content = extract_content(ocr_text, original_filename, ollama_host, model, timeout, pdf_title, summary=True)
    content = _prepare_content(content, folder_files, resolve_date)

    ranked, _ = _shortlist(
        content, ocr_text, existing_folders, folder_files, dokumente_root, filename_title, pdf_title, embedder
    )
    offered = sorted(candidate_search.collapsed_folders(existing_folders, folder_files)) or existing_folders

    folder, is_new_folder, folder_confidence, tags = classify_folder_via_anthropic(
        content.correspondent, content.title, offered, dokumente_root,
        anthropic_api_key, anthropic_model,
        filename_title=filename_title, pdf_title=pdf_title, keywords=content.keywords,
        summary=content.summary, hints=_cloud_hints(ranked),
    )
    chosen = next((c for c in ranked if c.path == folder), None)
    if not is_new_folder and folder != dokumente_root:
        folder, is_new_folder, _, _ = _walk_folder_tree(
            ocr_text, original_filename, existing_folders, dokumente_root, ollama_host, model, timeout,
            start_path=folder, issue_date=content.issue_date, by_year_only=True,
        )
    if chosen is not None:
        rank = next(n for n, c in enumerate(ranked, 1) if c.path == chosen.path)
        tags.append(f"KANDIDAT-{rank} (Beleg {chosen.strength:.1f}, Absender-Dateien {chosen.sender_files})")
    elif "ANTHROPIC-NICHT-ERREICHBAR" not in tags:
        tags.append("OHNE-KANDIDAT")
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


def classify_with_cloud_when_unsure(
    confidence_threshold: float,
    cloud_min_confidence: float,
    anthropic_api_key: str | None,
    anthropic_model: str,
    **classify_args,
) -> tuple[ClassificationOutcome, list[str]]:
    """Local first, cloud second: classify() decides; only when its result
    would go to the review folder (confidence below `confidence_threshold`,
    i.e. nothing in the tree backs the choice) the cloud classifier is asked
    too, and its answer is filed if it is at least `cloud_min_confidence`
    sure. Otherwise the document goes to review as before, carrying the
    cloud's folder as the suggestion (the local one if the cloud named no
    usable folder).

    The cloud's result can never be surer than the local extraction
    (title/date) it builds on, so with an extraction below
    `cloud_min_confidence` it is not asked at all - nothing leaves the
    machine for an answer that could not be filed.

    Measured on 120 documents (cloud asked for the 32 the local path left
    unfiled; threshold 0.9): 10 more filed, 7 of them exactly right, 2 one
    level off, 1 wrong. Accepting every cloud answer instead filed 31 and
    got 12 of them wrong."""
    content = classify_args.pop("content", None)
    if content is None:
        content = extract_content(
            classify_args["ocr_text"], classify_args["original_filename"], classify_args["ollama_host"],
            classify_args["model"], classify_args.get("timeout", 120.0), classify_args.get("pdf_title"),
            summary=True,
        )
    outcome, tags = classify(**classify_args, content=content)
    if outcome.confidence >= confidence_threshold or content.confidence < cloud_min_confidence:
        return outcome, tags

    # Without an API key this warns and comes back as not reachable.
    cloud_outcome, cloud_tags = classify_via_anthropic(
        **classify_args, content=content, anthropic_api_key=anthropic_api_key, anthropic_model=anthropic_model,
    )
    if "ANTHROPIC-NICHT-ERREICHBAR" in cloud_tags:
        return outcome, tags + ["ANTHROPIC-NICHT-ERREICHBAR"]
    if "UNGUELTIGE-ORDNERWAHL" in cloud_tags:
        # No usable folder from the cloud: the local suggestion stays.
        return outcome, tags + ["CLOUD-UNSICHER", "UNGUELTIGE-ORDNERWAHL"]
    if cloud_outcome.confidence >= cloud_min_confidence:
        return cloud_outcome, ["CLOUD-ENTSCHEIDUNG", *cloud_tags]
    # Not sure enough either: review folder, with the cloud's folder as the
    # suggestion - and no surer than the local result, which is below the
    # filing threshold whatever that is set to.
    unsure = cloud_outcome._replace(confidence=min(cloud_outcome.confidence, outcome.confidence))
    return unsure, ["CLOUD-UNSICHER", *cloud_tags]


def semantic_similarities(
    embedder: Embedder,
    query: DocumentQuery,
    existing_folders: list[str],
    folder_files: dict[str, list[str]] | None,
    dokumente_root: str,
) -> dict[str, float] | None:
    """Cosine similarity between the document and every shortlist-able
    folder (candidates.collapsed_folders), or None if the embedding model
    could not be reached - the shortlist then stays lexical; a model that is
    really down makes the following chat call fail as a transient error
    anyway."""
    files_of = candidate_search.collapsed_folders(existing_folders, folder_files)
    if not files_of:
        return None
    paths = list(files_of)
    texts = [candidate_search.folder_text(path, dokumente_root, files_of[path]) for path in paths]
    try:
        vectors = embedder.embed([candidate_search.document_text(query), *texts])
    except Exception as exc:
        log.warning("Embedding with %s failed (%s); the shortlist is lexical only.", embedder.model, exc)
        return None
    document = vectors[0]
    return {path: cosine(document, vector) for path, vector in zip(paths, vectors[1:])}


def classify(
    ocr_text: str,
    original_filename: str,
    existing_folders: list[str],
    ollama_host: str,
    model: str,
    dokumente_root: str = "Dokumente",
    timeout: float = 120.0,
    content: ContentExtraction | None = None,
    pdf_title: str | None = None,
    folder_files: dict[str, list[str]] | None = None,
    resolve_date: Callable[[date | None], date | None] | None = None,
    filename_title: str | None = None,
    embedder: Embedder | None = None,
) -> tuple[ClassificationOutcome, list[str]]:
    """Classifies one document:
    1. extract title/date/correspondent, independent of the folder structure;
    2. shortlist the folders that already hold similar documents
       (candidates.rank_candidates - deterministic, from folder names and
       the names of the files in them; with an `embedder`, blended with the
       semantic similarity of an embedding model);
    3. one model call picks among those candidates and the top-level folders;
    4. from the picked folder, descend further only where it still has
       subfolders.
    Raises on infrastructure failures (unreachable Ollama, invalid response)
    so the caller can treat those as transient and retry/fallback
    accordingly.

    Pass `content` to skip the extraction call when title/correspondent are
    already known some other way (e.g. taken from the filename because OCR
    found no text). `folder_files` maps each existing folder to the names of
    the files directly in it; without it the shortlist can only go by folder
    names. `resolve_date` turns the model's issue date into the one to trust
    (see signals.resolve_issue_date)."""
    if content is None:
        content = extract_content(ocr_text, original_filename, ollama_host, model, timeout, pdf_title)
    content = _prepare_content(content, folder_files, resolve_date)

    ranked, semantic = _shortlist(
        content, ocr_text, existing_folders, folder_files, dokumente_root, filename_title, pdf_title, embedder
    )
    if ranked and ranked[0].strength >= STRONG_EVIDENCE:
        floor = _SHOWN_SCORE_SHARE * ranked[0].score
        ranked = [c for c in ranked if c.score >= floor][:_MAX_SHOWN_WITH_STRONG]

    folder, is_new_folder, _, tags, chosen = _pick_folder(
        ocr_text, original_filename, pdf_title, content, ranked, existing_folders, dokumente_root,
        ollama_host, model, timeout,
    )
    # Backed by evidence: the folder matches strongly, or earlier documents
    # of this very sender are filed there (in the evaluation the single most
    # reliable sign - right in 31 of 33 cases).
    backed = chosen is not None and (chosen.strength >= STRONG_EVIDENCE or chosen.sender_files > 0)
    # Or two independent signals agree on it: best by words AND for the
    # embedding model (see CONFIDENCE_AGREED).
    agreed = (
        chosen is not None and semantic is not None and chosen.lexical_rank == 1
        and chosen.path == max(semantic, key=semantic.get)
    )
    if is_new_folder:
        folder_confidence = CONFIDENCE_BACKED_NEW_FOLDER if backed else CONFIDENCE_UNBACKED
    else:
        # A folder chosen on evidence is final except for its year
        # subfolders; a bare top-level folder is only the start of the
        # level-by-level descent.
        folder, is_new_folder, _, walk_tags = _walk_folder_tree(
            ocr_text, original_filename, existing_folders, dokumente_root, ollama_host, model, timeout,
            correspondent=content.correspondent, title=content.title, pdf_title=pdf_title,
            start_path=folder, issue_date=content.issue_date, by_year_only=chosen is not None,
        )
        tags += walk_tags
        if backed:
            folder_confidence = CONFIDENCE_BACKED
        elif agreed:
            folder_confidence = CONFIDENCE_AGREED
            tags.append("WORT-UND-BEDEUTUNG-EINIG")
        else:
            folder_confidence = CONFIDENCE_UNBACKED
        if "UNGUELTIGE-ORDNERWAHL" in walk_tags:
            folder_confidence = min(folder_confidence, INVALID_CHOICE_CONFIDENCE_CAP)
    if chosen is not None:
        rank = next((n for n, c in enumerate(ranked, 1) if c.path == chosen.path), 0)
        tags.append(f"KANDIDAT-{rank} (Beleg {chosen.strength:.1f}, Absender-Dateien {chosen.sender_files})")
    else:
        tags.append("OHNE-KANDIDAT")

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
