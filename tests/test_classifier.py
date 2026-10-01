import json
from datetime import date

import ollama
import pytest

from depot import classifier
from depot.models import AnthropicFolderDecision, ContentExtraction, FolderStepDecision


# ---- _children_of ----------------------------------------------------

def test_children_of_returns_direct_children_only():
    folders = [
        "Dokumente/Gesundheit",
        "Dokumente/Gesundheit/Krankenkasse",
        "Dokumente/Gesundheit/Krankenkasse/Rechnungen",
        "Dokumente/Motorrad",
    ]
    assert classifier._children_of(folders, "Dokumente") == ["Gesundheit", "Motorrad"]
    assert classifier._children_of(folders, "Dokumente/Gesundheit") == ["Krankenkasse"]
    assert classifier._children_of(folders, "Dokumente/Gesundheit/Krankenkasse") == ["Rechnungen"]


def test_children_of_leaf_returns_empty():
    folders = ["Dokumente/Gesundheit"]
    assert classifier._children_of(folders, "Dokumente/Gesundheit") == []


# ---- _walk_folder_tree --------------------------------------------------

EXISTING_FOLDERS = [
    "Dokumente/Gesundheit",
    "Dokumente/Gesundheit/Krankenkasse",
    "Dokumente/Motorrad",
    "Dokumente/Motorrad/Rechnungen",
    "Dokumente/Games",
    "Dokumente/Games/Amiibo-main",
]


def _decision(action, folder_name=None, confidence=0.9):
    return FolderStepDecision.model_validate(
        {"action": action, "folder_name": folder_name, "confidence": confidence}
    )


def test_walk_descends_then_stays(monkeypatch):
    calls = []

    def fake_decide(ocr_text, original_filename, current_path, children, ollama_host, model, timeout=120.0, **kwargs):
        calls.append(current_path)
        if current_path == "Dokumente":
            return _decision("descend", "Gesundheit", confidence=0.9)
        return _decision("stay", confidence=0.8)

    monkeypatch.setattr(classifier, "_decide_folder_step", fake_decide)

    folder, is_new, confidence, tags = classifier._walk_folder_tree(
        "text", "scan.pdf", EXISTING_FOLDERS, "Dokumente", "http://fake", "model"
    )
    assert folder == "Dokumente/Gesundheit"
    assert is_new is False
    assert confidence == 0.8  # min of the two steps
    assert tags == []
    assert calls == ["Dokumente", "Dokumente/Gesundheit"]


def test_walk_never_visits_irrelevant_branch(monkeypatch):
    """The whole point of the redesign: at the top level the model only
    ever sees Dokumente's direct children, so an irrelevant deep branch
    like Games/Amiibo is never even offered as a candidate."""
    seen_children = []

    def fake_decide(ocr_text, original_filename, current_path, children, ollama_host, model, timeout=120.0, **kwargs):
        seen_children.append(children)
        return _decision("descend", "Gesundheit") if current_path == "Dokumente" else _decision("stay")

    monkeypatch.setattr(classifier, "_decide_folder_step", fake_decide)

    classifier._walk_folder_tree("text", "scan.pdf", EXISTING_FOLDERS, "Dokumente", "http://fake", "model")

    assert seen_children[0] == ["Games", "Gesundheit", "Motorrad"]
    assert "Amiibo-main" not in seen_children[0]


def test_walk_stops_at_leaf_without_llm_call(monkeypatch):
    calls = []

    def fake_decide(*args, **kwargs):
        calls.append(1)
        return _decision("descend", "Krankenkasse")

    monkeypatch.setattr(classifier, "_decide_folder_step", fake_decide)

    folder, is_new, confidence, tags = classifier._walk_folder_tree(
        "text", "scan.pdf", EXISTING_FOLDERS, "Dokumente/Gesundheit", "http://fake", "model"
    )
    # Dokumente/Gesundheit/Krankenkasse has no children -> loop ends without
    # ever calling _decide_folder_step again after reaching it.
    assert folder == "Dokumente/Gesundheit/Krankenkasse"
    assert len(calls) == 1


def test_walk_corrects_near_duplicate_descend_choice(monkeypatch):
    def fake_decide(ocr_text, original_filename, current_path, children, ollama_host, model, timeout=120.0, **kwargs):
        if current_path == "Dokumente":
            return _decision("descend", "Motorad")  # typo of "Motorrad"
        return _decision("stay")

    monkeypatch.setattr(classifier, "_decide_folder_step", fake_decide)

    folder, is_new, confidence, tags = classifier._walk_folder_tree(
        "text", "scan.pdf", EXISTING_FOLDERS, "Dokumente", "http://fake", "model"
    )
    assert folder == "Dokumente/Motorrad/Rechnungen" or folder == "Dokumente/Motorrad"
    assert any("AUTO-KORRIGIERT" in t for t in tags)


def test_walk_treats_invalid_descend_as_stay(monkeypatch):
    def fake_decide(*args, **kwargs):
        return _decision("descend", "CompletelyUnrelatedName")

    monkeypatch.setattr(classifier, "_decide_folder_step", fake_decide)

    folder, is_new, confidence, tags = classifier._walk_folder_tree(
        "text", "scan.pdf", EXISTING_FOLDERS, "Dokumente", "http://fake", "model"
    )
    assert folder == "Dokumente"
    assert "UNGUELTIGE-ORDNERWAHL" in tags


def test_walk_creates_new_folder(monkeypatch):
    def fake_decide(ocr_text, original_filename, current_path, children, ollama_host, model, timeout=120.0, **kwargs):
        return _decision("new_folder", "Versicherung", confidence=0.85)

    monkeypatch.setattr(classifier, "_decide_folder_step", fake_decide)

    folder, is_new, confidence, tags = classifier._walk_folder_tree(
        "text", "scan.pdf", EXISTING_FOLDERS, "Dokumente", "http://fake", "model"
    )
    assert folder == "Dokumente/Versicherung"
    assert is_new is True
    assert tags == []


def test_walk_redirects_near_duplicate_new_folder_to_existing_sibling(monkeypatch):
    def fake_decide(ocr_text, original_filename, current_path, children, ollama_host, model, timeout=120.0, **kwargs):
        return _decision("new_folder", "Rechnung")  # "Rechnungen" already exists under Motorrad... but we're at root

    monkeypatch.setattr(classifier, "_decide_folder_step", fake_decide)

    folder, is_new, confidence, tags = classifier._walk_folder_tree(
        "text", "scan.pdf", ["Dokumente/Rechnungen"], "Dokumente", "http://fake", "model"
    )
    assert folder == "Dokumente/Rechnungen"
    assert is_new is False
    assert any("AUTO-REDIRECTED" in t for t in tags)


def test_walk_with_no_top_level_folders_stays_at_root():
    folder, is_new, confidence, tags = classifier._walk_folder_tree(
        "text", "scan.pdf", [], "Dokumente", "http://fake", "model"
    )
    assert folder == "Dokumente"
    assert is_new is False
    assert confidence == 1.0
    assert tags == []


# ---- extract_content / _decide_folder_step (real ollama call, mocked) ----

class _FakeClient:
    def __init__(self, response_payload):
        self._payload = response_payload

    def __call__(self, *args, **kwargs):
        return self

    def chat(self, model, messages, format, options, **kwargs):
        return {"message": {"content": json.dumps(self._payload)}}


def test_extract_content_parses_valid_response(monkeypatch):
    canned = {
        "title": "Stromrechnung Juli",
        "correspondent": "Stadtwerke München",
        "issue_date": "2026-07-15",
        "confidence": 0.9,
    }
    monkeypatch.setattr(ollama, "Client", lambda *a, **k: _FakeClient(canned))

    result = classifier.extract_content("ocr text", "scan.pdf", "http://fake", "model")
    assert isinstance(result, ContentExtraction)
    assert result.title == "Stromrechnung Juli"
    assert result.correspondent == "Stadtwerke München"
    assert result.issue_date == date(2026, 7, 15)


def test_extract_content_raises_on_invalid_json(monkeypatch):
    class BadClient:
        def chat(self, model, messages, format, options, **kwargs):
            return {"message": {"content": "not json"}}

    monkeypatch.setattr(ollama, "Client", lambda *a, **k: BadClient())

    with pytest.raises(RuntimeError):
        classifier.extract_content("ocr text", "scan.pdf", "http://fake", "model")


def test_decide_folder_step_parses_valid_response(monkeypatch):
    canned = {"action": "descend", "folder_name": "Gesundheit", "confidence": 0.9}
    monkeypatch.setattr(ollama, "Client", lambda *a, **k: _FakeClient(canned))

    result = classifier._decide_folder_step(
        "ocr text", "scan.pdf", "Dokumente", ["Gesundheit", "Motorrad"], "http://fake", "model"
    )
    assert isinstance(result, FolderStepDecision)
    assert result.action == "descend"
    assert result.folder_name == "Gesundheit"


# ---- classify() end-to-end (mocked at the extract_content/_walk_folder_tree level) ----

def _fake_pick(folder, is_new=False, confidence=0.95, tags=(), chosen=None):
    return lambda *a, **k: (folder, is_new, confidence, list(tags), chosen)


def test_classify_combines_content_and_folder_walk(monkeypatch):
    monkeypatch.setattr(
        classifier, "extract_content",
        lambda *a, **k: ContentExtraction(
            title="Arztrechnung", correspondent="Dr. Müller", issue_date=date(2026, 6, 10), confidence=0.8
        ),
    )
    walk_calls = []

    def fake_walk(*a, **k):
        walk_calls.append(k)
        return ("Dokumente/Gesundheit/Krankenkasse", False, 0.95, [])

    monkeypatch.setattr(classifier, "_pick_folder", _fake_pick("Dokumente/Gesundheit"))
    monkeypatch.setattr(classifier, "_walk_folder_tree", fake_walk)

    outcome, tags = classifier.classify(
        ocr_text="text",
        original_filename="scan.pdf",
        existing_folders=EXISTING_FOLDERS,
        ollama_host="http://fake",
        model="model",
    )
    assert outcome.folder == "Dokumente/Gesundheit/Krankenkasse"
    assert outcome.title == "Arztrechnung"
    assert outcome.correspondent == "Dr. Müller"
    assert outcome.issue_date == date(2026, 6, 10)
    # a top-level folder picked without any candidate backing it: a
    # suggestion only, whatever confidence the model itself reported
    assert outcome.confidence == classifier.CONFIDENCE_UNBACKED
    assert tags == ["OHNE-KANDIDAT"]
    # the level-by-level descent continues from the picked top-level folder
    assert walk_calls[0]["start_path"] == "Dokumente/Gesundheit"
    assert walk_calls[0]["by_year_only"] is False
    assert walk_calls[0]["correspondent"] == "Dr. Müller"


def test_classify_converts_empty_correspondent_to_none_on_outcome(monkeypatch):
    monkeypatch.setattr(
        classifier, "extract_content",
        lambda *a, **k: ContentExtraction(title="Notiz", correspondent="", confidence=0.5),
    )
    monkeypatch.setattr(classifier, "_pick_folder", _fake_pick("Dokumente"))
    monkeypatch.setattr(
        classifier, "_walk_folder_tree",
        lambda *a, **k: ("Dokumente", False, 0.5, []),
    )

    outcome, _ = classifier.classify(
        ocr_text="text", original_filename="scan.pdf", existing_folders=EXISTING_FOLDERS,
        ollama_host="http://fake", model="model",
    )
    assert outcome.correspondent is None


# ---- classify_folder_via_anthropic (cloud call, mocked) --------------------

class _FakeAnthropicResponse:
    def __init__(self, parsed_output):
        self.parsed_output = parsed_output


class _FakeAnthropicMessages:
    def __init__(self, parsed_output=None, exc=None):
        self._parsed_output = parsed_output
        self._exc = exc
        self.last_kwargs = None

    def parse(self, **kwargs):
        self.last_kwargs = kwargs
        if self._exc is not None:
            raise self._exc
        return _FakeAnthropicResponse(self._parsed_output)


class _FakeAnthropicClient:
    def __init__(self, parsed_output=None, exc=None):
        self.messages = _FakeAnthropicMessages(parsed_output, exc)


def _decision_anthropic(action, folder="Dokumente", new_folder_name=None, confidence=0.9):
    return AnthropicFolderDecision(
        action=action, folder=folder, new_folder_name=new_folder_name, confidence=confidence
    )


def test_classify_folder_via_anthropic_existing_folder(monkeypatch):
    fake_client = _FakeAnthropicClient(
        parsed_output=_decision_anthropic("existing", folder="Dokumente/Gesundheit")
    )
    monkeypatch.setattr(classifier.anthropic, "Anthropic", lambda **kwargs: fake_client)

    folder, is_new, confidence, tags = classifier.classify_folder_via_anthropic(
        "Techniker Krankenkasse", "Mitgliedsbescheinigung", EXISTING_FOLDERS, "Dokumente",
        "sk-ant-fake", "claude-haiku-4-5",
    )
    assert folder == "Dokumente/Gesundheit"
    assert is_new is False
    assert confidence == 0.9
    assert tags == []
    # privacy contract: only correspondent/title/folder-list ever get sent,
    # never OCR text (the function signature doesn't even accept it)
    sent = fake_client.messages.last_kwargs
    assert "Techniker Krankenkasse" in sent["messages"][0]["content"]
    assert "Mitgliedsbescheinigung" in sent["messages"][0]["content"]


def test_classify_folder_via_anthropic_new_folder_under_valid_parent(monkeypatch):
    fake_client = _FakeAnthropicClient(
        parsed_output=_decision_anthropic("new_folder", folder="Dokumente/Gesundheit", new_folder_name="Zahnarzt")
    )
    monkeypatch.setattr(classifier.anthropic, "Anthropic", lambda **kwargs: fake_client)

    folder, is_new, confidence, tags = classifier.classify_folder_via_anthropic(
        "Dr. Beispiel", "Rechnung", EXISTING_FOLDERS, "Dokumente", "sk-ant-fake", "claude-haiku-4-5",
    )
    assert folder == "Dokumente/Gesundheit/Zahnarzt"
    assert is_new is True
    assert tags == []


def test_classify_folder_via_anthropic_hallucinated_existing_folder_gets_fuzzy_corrected(monkeypatch):
    fake_client = _FakeAnthropicClient(
        parsed_output=_decision_anthropic("existing", folder="Dokumente/Gesundheiten")  # close typo
    )
    monkeypatch.setattr(classifier.anthropic, "Anthropic", lambda **kwargs: fake_client)

    folder, is_new, confidence, tags = classifier.classify_folder_via_anthropic(
        "Foo", "Bar", EXISTING_FOLDERS, "Dokumente", "sk-ant-fake", "claude-haiku-4-5",
    )
    assert folder == "Dokumente/Gesundheit"
    assert any("AUTO-KORRIGIERT" in t for t in tags)


def test_classify_folder_via_anthropic_hallucinated_folder_no_match_is_capped(monkeypatch):
    fake_client = _FakeAnthropicClient(
        parsed_output=_decision_anthropic("existing", folder="Dokumente/Vollkommen-Erfunden", confidence=0.99)
    )
    monkeypatch.setattr(classifier.anthropic, "Anthropic", lambda **kwargs: fake_client)

    folder, is_new, confidence, tags = classifier.classify_folder_via_anthropic(
        "Foo", "Bar", EXISTING_FOLDERS, "Dokumente", "sk-ant-fake", "claude-haiku-4-5",
    )
    assert folder == "Dokumente"
    assert confidence == classifier.INVALID_CHOICE_CONFIDENCE_CAP
    assert "UNGUELTIGE-ORDNERWAHL" in tags


def test_classify_folder_via_anthropic_missing_new_folder_name_is_invalid(monkeypatch):
    fake_client = _FakeAnthropicClient(
        parsed_output=_decision_anthropic("new_folder", folder="Dokumente/Gesundheit", new_folder_name=None)
    )
    monkeypatch.setattr(classifier.anthropic, "Anthropic", lambda **kwargs: fake_client)

    folder, is_new, confidence, tags = classifier.classify_folder_via_anthropic(
        "Foo", "Bar", EXISTING_FOLDERS, "Dokumente", "sk-ant-fake", "claude-haiku-4-5",
    )
    assert confidence == classifier.INVALID_CHOICE_CONFIDENCE_CAP
    assert "UNGUELTIGE-ORDNERWAHL" in tags


def test_classify_folder_via_anthropic_no_api_key_configured(monkeypatch):
    folder, is_new, confidence, tags = classifier.classify_folder_via_anthropic(
        "Foo", "Bar", EXISTING_FOLDERS, "Dokumente", None, "claude-haiku-4-5",
    )
    assert confidence == 0.0
    assert "ANTHROPIC-NICHT-ERREICHBAR" in tags


def test_classify_folder_via_anthropic_call_failure_falls_back_gracefully(monkeypatch):
    fake_client = _FakeAnthropicClient(exc=RuntimeError("network down"))
    monkeypatch.setattr(classifier.anthropic, "Anthropic", lambda **kwargs: fake_client)

    folder, is_new, confidence, tags = classifier.classify_folder_via_anthropic(
        "Foo", "Bar", EXISTING_FOLDERS, "Dokumente", "sk-ant-fake", "claude-haiku-4-5",
    )
    # Must never raise - the caller's confidence-threshold check routes this
    # to Unsortiert instead of retrying the whole pipeline run.
    assert confidence == 0.0
    assert "ANTHROPIC-NICHT-ERREICHBAR" in tags


# ---- classify_via_anthropic (end-to-end, mocked) ---------------------------

def test_classify_via_anthropic_combines_content_and_cloud_folder_decision(monkeypatch):
    monkeypatch.setattr(
        classifier, "extract_content",
        lambda *a, **k: ContentExtraction(
            title="Mitgliedsbescheinigung", correspondent="Techniker Krankenkasse",
            issue_date=date(2026, 6, 10), confidence=0.8,
        ),
    )
    calls = []

    def fake_folder_via_anthropic(*a, **k):
        calls.append(a)
        return ("Dokumente/Gesundheit", False, 0.95, [])

    monkeypatch.setattr(classifier, "classify_folder_via_anthropic", fake_folder_via_anthropic)

    outcome, tags = classifier.classify_via_anthropic(
        ocr_text="geheimer volltext, darf nicht an anthropic gehen",
        original_filename="scan.pdf",
        existing_folders=EXISTING_FOLDERS,
        ollama_host="http://fake",
        model="model",
        anthropic_api_key="sk-ant-fake",
        anthropic_model="claude-haiku-4-5",
    )
    assert outcome.folder == "Dokumente/Gesundheit"
    assert outcome.title == "Mitgliedsbescheinigung"
    assert outcome.correspondent == "Techniker Krankenkasse"
    assert outcome.confidence == 0.8  # min(content=0.8, folder=0.95)
    assert tags == []
    # only correspondent + title were passed to the cloud call - no ocr_text
    assert calls[0][:2] == ("Techniker Krankenkasse", "Mitgliedsbescheinigung")
    assert calls[0][2] == EXISTING_FOLDERS
    assert calls[0][3] == "Dokumente"


# ---- title signals in the prompts -------------------------------------------

def test_folder_step_prompt_contains_title_and_correspondent_after_the_text():
    """The folder decision must see the document's title/sender, not only
    its raw text - and the (long, per-document constant) text must come
    first so Ollama can reuse it across the steps of one walk."""
    messages = classifier._build_folder_step_messages(
        "OCR-VOLLTEXT", "scan.pdf", "Dokumente/Gesundheit", ["Krankenkasse"],
        title="Mitgliedsbescheinigung", correspondent="Techniker Krankenkasse", pdf_title="Ihre Bescheinigung",
    )
    document, level = messages[1]["content"], messages[-1]["content"]
    assert "OCR-VOLLTEXT" in document
    assert "Titel des Dokuments: Mitgliedsbescheinigung" in document
    assert "Absender des Dokuments: Techniker Krankenkasse" in document
    assert "Titel laut PDF-Metadaten: Ihre Bescheinigung" in document
    # the per-step part is a separate, final turn - nothing of it may leak
    # into the constant document turn, or the prefix cache is lost
    assert "Aktuelle Ebene: Dokumente/Gesundheit" in level and "- Krankenkasse" in level
    assert "Aktuelle Ebene" not in document
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]


def test_folder_step_prompt_without_text_says_so():
    messages = classifier._build_folder_step_messages(
        "", "motorrad anhaenger.jpg", "Dokumente", ["Motorrad"], title="motorrad anhaenger"
    )
    assert "(kein Text erkannt)" in messages[1]["content"]


def test_walk_passes_title_and_correspondent_to_every_step(monkeypatch):
    seen = []

    def fake_decide(ocr_text, original_filename, current_path, children, ollama_host, model, timeout=120.0, **kwargs):
        seen.append(kwargs)
        return _decision("stay")

    monkeypatch.setattr(classifier, "_decide_folder_step", fake_decide)

    classifier._walk_folder_tree(
        "text", "scan.pdf", EXISTING_FOLDERS, "Dokumente", "http://fake", "model",
        correspondent="Werkstatt Beispiel", title="Inspektionsrechnung", pdf_title=None,
    )

    assert seen == [{"title": "Inspektionsrechnung", "correspondent": "Werkstatt Beispiel", "pdf_title": None}]


def test_classify_with_given_content_skips_extraction(monkeypatch):
    """OCR found nothing but the filename is descriptive: the caller passes
    a content built from the filename, and no extraction call is made."""
    def _must_not_run(*a, **k):
        raise AssertionError("extract_content must not be called when content is given")

    monkeypatch.setattr(classifier, "extract_content", _must_not_run)
    walk_kwargs = []

    def fake_walk(*a, **k):
        walk_kwargs.append(k)
        return ("Dokumente/Motorrad", False, 0.9, [])

    monkeypatch.setattr(classifier, "_pick_folder", _fake_pick("Dokumente/Motorrad"))
    monkeypatch.setattr(classifier, "_walk_folder_tree", fake_walk)

    outcome, _ = classifier.classify(
        ocr_text="", original_filename="motorrad anhaenger.jpg", existing_folders=EXISTING_FOLDERS,
        ollama_host="http://fake", model="model",
        content=ContentExtraction(title="motorrad anhaenger", correspondent="", confidence=0.7),
    )

    assert outcome.folder == "Dokumente/Motorrad"
    assert outcome.title == "motorrad anhaenger"
    assert outcome.confidence == classifier.CONFIDENCE_UNBACKED
    assert walk_kwargs[0]["title"] == "motorrad anhaenger"


def test_ollama_calls_pin_context_and_keep_the_model_loaded(monkeypatch):
    seen = {}

    class RecordingClient:
        def chat(self, **kwargs):
            seen.update(kwargs)
            return {"message": {"content": json.dumps({"title": "T", "correspondent": "", "confidence": 0.9})}}

    monkeypatch.setattr(ollama, "Client", lambda *a, **k: RecordingClient())

    classifier.extract_content("ocr text", "scan.pdf", "http://fake", "model")

    assert seen["keep_alive"] == classifier.OLLAMA_KEEP_ALIVE
    assert seen["options"]["num_ctx"] == 8192
    assert "keywords" in seen["format"]["required"]


def test_preload_never_raises(monkeypatch):
    def unreachable(*a, **k):
        raise ConnectionError("no ollama here")

    monkeypatch.setattr(ollama, "Client", unreachable)

    classifier.preload_model("http://fake", "model")  # must not raise


def test_cloud_call_gets_filename_pdf_title_and_keywords_but_no_document_text(monkeypatch):
    fake_client = _FakeAnthropicClient(
        parsed_output=_decision_anthropic("existing", folder="Dokumente/Gesundheit")
    )
    monkeypatch.setattr(classifier.anthropic, "Anthropic", lambda **kwargs: fake_client)
    monkeypatch.setattr(
        classifier, "extract_content",
        lambda *a, **k: ContentExtraction(
            title="Beendigung Zusatztarif", correspondent="", confidence=0.8,
            keywords=["Krankenversicherung", "Kuendigung"],
        ),
    )

    classifier.classify_via_anthropic(
        ocr_text="GEHEIMER VOLLTEXT",
        original_filename="Wir muessen leider die Teilnahme beenden.pdf",
        existing_folders=EXISTING_FOLDERS,
        ollama_host="http://fake",
        model="model",
        anthropic_api_key="sk-ant-fake",
        anthropic_model="claude-haiku-4-5",
        filename_title="Wir muessen leider die Teilnahme beenden",
        pdf_title="Ende der Teilnahme",
    )

    sent = fake_client.messages.last_kwargs["messages"][0]["content"]
    assert "Dateiname des Scans: Wir muessen leider die Teilnahme beenden" in sent
    assert "PDF-Titel: Ende der Teilnahme" in sent
    assert "Stichworte: Krankenversicherung, Kuendigung" in sent
    assert "GEHEIMER VOLLTEXT" not in sent


# ---- candidate shortlist + single pick ----------------------------------------

TREE = [
    "Dokumente/Versicherungen",
    "Dokumente/Fahrzeuge",
    "Dokumente/Fahrzeuge/MT-07",
    "Dokumente/Fahrzeuge/MT-07/Versicherung",
    "Dokumente/Finanzen",
    "Dokumente/Finanzen/Depot",
    "Dokumente/Finanzen/Depot/2024",
    "Dokumente/Finanzen/Depot/2025",
]
TREE_FILES = {
    "Dokumente/Fahrzeuge/MT-07/Versicherung": [
        "2025-03-01 Beispiel Versicherung - Beitragsrechnung.pdf",
        "2024-03-01 Beispiel Versicherung - Versicherungsschein.pdf",
    ],
    "Dokumente/Finanzen/Depot/2025": ["2025-02-01 Musterbank - Wertpapierabrechnung.pdf"],
}
INSURANCE_LETTER = ContentExtraction(
    title="Beitragsrechnung", correspondent="Beispiel Versicherung", issue_date=date(2026, 3, 1),
    confidence=0.9, keywords=["Versicherung", "Beitrag"],
)


class _PickClient:
    """Answers the folder-pick call with a fixed choice and records it."""

    def __init__(self, answer):
        self._answer = answer
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {"message": {"content": json.dumps(self._answer)}}


def _ranked(content=INSURANCE_LETTER):
    from depot.candidates import DocumentQuery, rank_candidates

    return rank_candidates(
        DocumentQuery(correspondent=content.correspondent, title=content.title, keywords=content.keywords),
        TREE, TREE_FILES,
    )


def test_pick_offers_candidates_with_examples_then_the_top_level_folders(monkeypatch):
    client = _PickClient(
        {"folder": "Dokumente/Fahrzeuge/MT-07/Versicherung", "new_folder_name": None, "confidence": 0.9}
    )
    monkeypatch.setattr(ollama, "Client", lambda *a, **k: client)
    ranked = _ranked()

    folder, is_new, confidence, tags, chosen = classifier._pick_folder(
        "OCR-VOLLTEXT", "scan.pdf", None, INSURANCE_LETTER, ranked, TREE, "Dokumente", "http://fake", "model"
    )

    assert folder == "Dokumente/Fahrzeuge/MT-07/Versicherung"
    assert (is_new, confidence, tags) == (False, 0.9, [])
    assert chosen is ranked[0]
    task = client.calls[0]["messages"][-1]["content"]
    assert "- Dokumente/Fahrzeuge/MT-07/Versicherung\n" in task
    assert "z.B. 2025-03-01 Beispiel Versicherung - Beitragsrechnung.pdf" in task
    # every top-level folder is offered exactly once, after the candidates
    for top in ("Dokumente/Fahrzeuge", "Dokumente/Finanzen", "Dokumente/Versicherungen"):
        assert task.count(f"- {top}\n") == 1
    assert task.index("Dokumente/Fahrzeuge/MT-07/Versicherung") < task.index("Hauptordner der Ablage")


def test_pick_shares_its_prefix_with_the_extraction_call(monkeypatch):
    """The decision is a follow-up turn of the extraction conversation, so
    Ollama does not evaluate the document text again - and the decision
    sees the text together with the extracted title and sender."""
    client = _PickClient({"folder": "Dokumente/Fahrzeuge/MT-07/Versicherung", "confidence": 0.9})
    monkeypatch.setattr(ollama, "Client", lambda *a, **k: client)

    classifier._pick_folder(
        "OCR-VOLLTEXT", "scan.pdf", "PDF-Titel", INSURANCE_LETTER, _ranked(), TREE, "Dokumente",
        "http://fake", "model",
    )

    messages = client.calls[0]["messages"]
    assert messages[:2] == classifier._build_content_messages("OCR-VOLLTEXT", "scan.pdf", "PDF-Titel")
    assert messages[2]["role"] == "assistant"
    assert json.loads(messages[2]["content"])["correspondent"] == "Beispiel Versicherung"
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]


def test_pick_answer_is_restricted_to_the_offered_folders(monkeypatch):
    """The answer is the folder path itself, limited by the schema to what
    was offered - asked for a list number instead, the model's choice
    followed the order of the list rather than the document."""
    client = _PickClient({"folder": "Dokumente/Fahrzeuge/MT-07/Versicherung", "confidence": 0.8})
    monkeypatch.setattr(ollama, "Client", lambda *a, **k: client)
    ranked = _ranked()

    classifier._pick_folder(
        "text", "scan.pdf", None, INSURANCE_LETTER, ranked, TREE, "Dokumente", "http://fake", "model"
    )

    allowed = client.calls[0]["format"]["properties"]["folder"]["enum"]
    assert allowed[:len(ranked)] == [c.path for c in ranked]
    assert set(allowed) == {c.path for c in ranked} | {
        "Dokumente/Versicherungen", "Dokumente/Fahrzeuge", "Dokumente/Finanzen"
    }
    assert len(allowed) == len(set(allowed))


def test_pick_of_a_folder_that_was_not_offered_is_not_trusted(monkeypatch):
    client = _PickClient({"folder": "Dokumente/Frei Erfunden", "confidence": 0.99})
    monkeypatch.setattr(ollama, "Client", lambda *a, **k: client)

    folder, is_new, confidence, tags, chosen = classifier._pick_folder(
        "text", "scan.pdf", None, INSURANCE_LETTER, _ranked(), TREE, "Dokumente", "http://fake", "model"
    )

    assert (folder, is_new, chosen) == ("Dokumente", False, None)
    assert confidence == classifier.INVALID_CHOICE_CONFIDENCE_CAP
    assert "UNGUELTIGE-ORDNERWAHL" in tags


def test_pick_of_a_top_level_folder_reports_no_candidate(monkeypatch):
    insurance_only = [c for c in _ranked() if c.path != "Dokumente/Finanzen"]
    client = _PickClient({"folder": "Dokumente/Finanzen", "confidence": 0.6})
    monkeypatch.setattr(ollama, "Client", lambda *a, **k: client)

    folder, _, _, _, chosen = classifier._pick_folder(
        "text", "scan.pdf", None, INSURANCE_LETTER, insurance_only, TREE, "Dokumente", "http://fake", "model"
    )

    assert chosen is None
    assert folder == "Dokumente/Finanzen"


def test_pick_can_create_a_new_subfolder_under_the_chosen_folder(monkeypatch):
    client = _PickClient({
        "folder": "Dokumente/Fahrzeuge/MT-07/Versicherung", "new_folder_name": "Schaden 2026", "confidence": 0.8,
    })
    monkeypatch.setattr(ollama, "Client", lambda *a, **k: client)

    folder, is_new, _, _, _ = classifier._pick_folder(
        "text", "scan.pdf", None, INSURANCE_LETTER, _ranked(), TREE, "Dokumente", "http://fake", "model"
    )

    assert folder == "Dokumente/Fahrzeuge/MT-07/Versicherung/Schaden 2026"
    assert is_new is True


def test_pick_redirects_a_near_duplicate_new_subfolder(monkeypatch):
    ranked = _ranked()
    client = _PickClient(
        {"folder": "Dokumente/Fahrzeuge/MT-07", "new_folder_name": "Versicherungen", "confidence": 0.8}
    )
    monkeypatch.setattr(ollama, "Client", lambda *a, **k: client)

    folder, is_new, _, tags, _ = classifier._pick_folder(
        "text", "scan.pdf", None, INSURANCE_LETTER, ranked, TREE, "Dokumente", "http://fake", "model"
    )

    assert folder == "Dokumente/Fahrzeuge/MT-07/Versicherung"
    assert is_new is False
    assert any("AUTO-REDIRECTED" in t for t in tags)


def test_pick_without_any_folders_stays_at_the_root():
    folder, is_new, confidence, tags, chosen = classifier._pick_folder(
        "text", "scan.pdf", None, INSURANCE_LETTER, [], [], "Dokumente", "http://fake", "model"
    )
    assert (folder, is_new, chosen) == ("Dokumente", False, None)


# ---- year subfolders are not a question for the model -------------------------

def _no_model_call(*a, **k):
    raise AssertionError("year folders must be resolved without asking the model")


def test_walk_enters_the_matching_year_folder_by_itself(monkeypatch):
    monkeypatch.setattr(classifier, "_decide_folder_step", _no_model_call)

    folder, is_new, confidence, tags = classifier._walk_folder_tree(
        "text", "scan.pdf", TREE, "Dokumente", "http://fake", "model",
        start_path="Dokumente/Finanzen/Depot", issue_date=date(2025, 2, 1),
    )

    assert (folder, is_new) == ("Dokumente/Finanzen/Depot/2025", False)


def test_walk_creates_the_missing_year_in_a_folder_split_by_year(monkeypatch):
    monkeypatch.setattr(classifier, "_decide_folder_step", _no_model_call)

    folder, is_new, confidence, tags = classifier._walk_folder_tree(
        "text", "scan.pdf", TREE, "Dokumente", "http://fake", "model",
        start_path="Dokumente/Finanzen/Depot", issue_date=date(2026, 1, 15),
    )

    assert (folder, is_new) == ("Dokumente/Finanzen/Depot/2026", True)


def test_walk_asks_the_model_about_year_folders_when_the_date_is_unknown(monkeypatch):
    asked = []

    def fake_decide(ocr_text, original_filename, current_path, children, *a, **k):
        asked.append(children)
        return _decision("stay")

    monkeypatch.setattr(classifier, "_decide_folder_step", fake_decide)

    folder, _, _, _ = classifier._walk_folder_tree(
        "text", "scan.pdf", TREE, "Dokumente", "http://fake", "model",
        start_path="Dokumente/Finanzen/Depot", issue_date=None,
    )

    assert asked == [["2024", "2025"]]
    assert folder == "Dokumente/Finanzen/Depot"


# ---- classify(): the whole local flow -------------------------------------------

def test_classify_picks_among_candidates_then_descends_by_year(monkeypatch):
    statement = ContentExtraction(
        title="Wertpapierabrechnung", correspondent="Musterbank AG", issue_date=date(2025, 6, 1), confidence=0.9
    )
    picks = []

    def fake_pick(ocr_text, original_filename, pdf_title, content, ranked, *a, **k):
        picks.append((content, ranked))
        depot = next(c for c in ranked if c.path == "Dokumente/Finanzen/Depot")
        return (depot.path, False, 0.9, [], depot)

    monkeypatch.setattr(classifier, "_pick_folder", fake_pick)
    monkeypatch.setattr(classifier, "_decide_folder_step", _no_model_call)

    outcome, tags = classifier.classify(
        ocr_text="Wertpapierabrechnung Musterbank", original_filename="scan.pdf", existing_folders=TREE,
        ollama_host="http://fake", model="model", content=statement, folder_files=TREE_FILES,
    )

    assert outcome.folder == "Dokumente/Finanzen/Depot/2025"
    assert outcome.is_new_folder is False
    # the sender is filed under its canonical spelling, without the legal form
    assert outcome.correspondent == "Musterbank"
    assert picks[0][0].correspondent == "Musterbank"
    # year folders are folded into their parent: never offered on their own
    assert not any(c.path.endswith(("/2024", "/2025")) for c in picks[0][1])
    # backed by the sender's earlier statement in that folder
    assert outcome.confidence == classifier.CONFIDENCE_BACKED
    assert any(t.startswith("KANDIDAT-1 (Beleg ") for t in tags)


def test_classify_uses_the_resolved_date_for_the_year_folder(monkeypatch):
    """The year folder must follow the date that was checked against the
    document, not whatever the model first said."""
    statement = ContentExtraction(
        title="Wertpapierabrechnung", correspondent="Musterbank", issue_date=date(2024, 1, 1), confidence=0.9
    )
    monkeypatch.setattr(classifier, "_pick_folder", _fake_pick("Dokumente/Finanzen/Depot"))
    monkeypatch.setattr(classifier, "_decide_folder_step", _no_model_call)

    outcome, _ = classifier.classify(
        ocr_text="text", original_filename="scan.pdf", existing_folders=TREE,
        ollama_host="http://fake", model="model", content=statement, folder_files=TREE_FILES,
        resolve_date=lambda d: date(2025, 3, 3),
    )

    assert outcome.issue_date == date(2025, 3, 3)
    assert outcome.folder == "Dokumente/Finanzen/Depot/2025"


def _classify_insurance_letter(monkeypatch, pick):
    monkeypatch.setattr(classifier, "_pick_folder", pick)
    monkeypatch.setattr(classifier, "_decide_folder_step", lambda *a, **k: _decision("stay", confidence=0.99))
    return classifier.classify(
        ocr_text="text", original_filename="scan.pdf", existing_folders=TREE,
        ollama_host="http://fake", model="model", content=INSURANCE_LETTER, folder_files=TREE_FILES,
    )


def test_confidence_follows_the_evidence_not_the_models_own_number(monkeypatch):
    """The model reports 0.9+ for right and wrong answers alike. What counts
    is whether the chosen folder already holds documents like this one."""
    def pick_backed(ocr_text, original_filename, pdf_title, content, ranked, *a, **k):
        best = ranked[0]
        assert best.path == "Dokumente/Fahrzeuge/MT-07/Versicherung"
        return (best.path, False, 0.3, [], best)  # the model's own 0.3 is ignored

    outcome, tags = _classify_insurance_letter(monkeypatch, pick_backed)

    assert outcome.folder == "Dokumente/Fahrzeuge/MT-07/Versicherung"
    assert outcome.confidence == classifier.CONFIDENCE_BACKED


def test_top_level_choice_despite_candidates_is_only_a_suggestion(monkeypatch):
    outcome, tags = _classify_insurance_letter(
        monkeypatch, _fake_pick("Dokumente/Versicherungen", confidence=0.99)
    )

    assert outcome.folder == "Dokumente/Versicherungen"
    assert outcome.confidence == classifier.CONFIDENCE_UNBACKED
    assert "OHNE-KANDIDAT" in tags


def test_weakly_matching_candidate_is_only_a_suggestion(monkeypatch):
    from depot.candidates import Candidate

    weak = Candidate(path="Dokumente/Finanzen", score=3.0, strength=1.0)
    outcome, tags = _classify_insurance_letter(monkeypatch, _fake_pick(weak.path, confidence=0.99, chosen=weak))

    assert outcome.confidence == classifier.CONFIDENCE_UNBACKED


def test_new_subfolder_in_a_backed_folder_is_filed_with_reduced_confidence(monkeypatch):
    def pick_new(ocr_text, original_filename, pdf_title, content, ranked, *a, **k):
        return (f"{ranked[0].path}/Schaden 2026", True, 0.9, [], ranked[0])

    outcome, tags = _classify_insurance_letter(monkeypatch, pick_new)

    assert outcome.is_new_folder is True
    assert outcome.confidence == classifier.CONFIDENCE_BACKED_NEW_FOLDER


def test_backed_folder_is_not_descended_any_further_by_the_model(monkeypatch):
    """Seen in the evaluation: after a correct pick the level-by-level walk
    went on into an unrelated, very specific subfolder."""
    from depot.candidates import Candidate

    def _no_step(*a, **k):
        raise AssertionError("a folder chosen on evidence must not be descended by another model call")

    strong = Candidate(path="Dokumente/Fahrzeuge/MT-07", score=50.0, strength=9.0)
    monkeypatch.setattr(classifier, "_pick_folder", _fake_pick(strong.path, chosen=strong))
    monkeypatch.setattr(classifier, "_decide_folder_step", _no_step)

    outcome, _ = classifier.classify(
        ocr_text="text", original_filename="scan.pdf", existing_folders=TREE,
        ollama_host="http://fake", model="model", content=INSURANCE_LETTER, folder_files=TREE_FILES,
    )

    assert outcome.folder == "Dokumente/Fahrzeuge/MT-07"


# ---- semantic shortlist (embedding model) ------------------------------------

from depot.candidates import DocumentQuery  # noqa: E402

class _FakeEmbedder:
    """Returns a fixed similarity for one folder text and a low one for the rest."""

    model = "fake-embed"

    def __init__(self, favourite: str | None = None, fail: bool = False):
        self.favourite = favourite
        self.fail = fail
        self.texts: list[str] = []

    def embed(self, texts):
        if self.fail:
            raise ConnectionError("embedding model unreachable")
        self.texts = texts
        vectors = []
        for text in texts:
            if text.startswith("Absender:") or (self.favourite and self.favourite in text):
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return vectors


def test_semantic_similarities_cover_exactly_the_shortlistable_folders():
    embedder = _FakeEmbedder(favourite="Ordner: Finanzen > Depot")
    sims = classifier.semantic_similarities(
        embedder, DocumentQuery(correspondent="X", title="Depotauszug", text="text"), TREE, TREE_FILES, "Dokumente"
    )
    from depot.candidates import collapsed_folders

    assert set(sims) == set(collapsed_folders(TREE, TREE_FILES))
    assert sims["Dokumente/Finanzen/Depot"] == pytest.approx(1.0)
    assert all(v == pytest.approx(0.0) for p, v in sims.items() if p != "Dokumente/Finanzen/Depot")
    assert embedder.texts[0].startswith("Absender: X\nTitel: Depotauszug")
    assert any(t.startswith("Ordner: Finanzen > Depot\nDateien: ") for t in embedder.texts[1:])


def test_semantic_similarities_are_skipped_when_the_embedding_model_fails(caplog):
    sims = classifier.semantic_similarities(
        _FakeEmbedder(fail=True), DocumentQuery(title="x"), TREE, TREE_FILES, "Dokumente"
    )
    assert sims is None
    assert "lexical only" in caplog.text


def test_classify_hands_the_similarities_to_the_candidate_search(monkeypatch):
    seen = {}
    real_rank = classifier.candidate_search.rank_candidates

    def spy_rank(query, folders, folder_files=None, limit=8, semantic=None):
        seen["semantic"] = semantic
        return real_rank(query, folders, folder_files, limit, semantic)

    monkeypatch.setattr(classifier.candidate_search, "rank_candidates", spy_rank)
    monkeypatch.setattr(classifier, "_pick_folder", _fake_pick("Dokumente/Finanzen/Depot"))
    monkeypatch.setattr(classifier, "_decide_folder_step", _no_model_call)
    content = ContentExtraction(
        title="Depotauszug", correspondent="Musterbank", issue_date=date(2025, 1, 1), confidence=0.9
    )

    classifier.classify(
        ocr_text="text", original_filename="scan.pdf", existing_folders=TREE, ollama_host="http://fake",
        model="model", content=content, folder_files=TREE_FILES, embedder=_FakeEmbedder("Ordner: Finanzen > Depot"),
    )
    assert seen["semantic"]["Dokumente/Finanzen/Depot"] == pytest.approx(1.0)

    classifier.classify(
        ocr_text="text", original_filename="scan.pdf", existing_folders=TREE, ollama_host="http://fake",
        model="model", content=content, folder_files=TREE_FILES,
    )
    assert seen["semantic"] is None  # no embedder: lexical only, as before


def test_agreement_of_words_and_meaning_is_evidence_enough_to_file(monkeypatch):
    """A folder with only moderate word overlap and no document of the
    sender would go to Unsortiert - unless the embedding model independently
    picks the same folder as the best match."""
    letter = ContentExtraction(title="Rechnung Inspektion", correspondent="Werkstatt Nord", confidence=0.9)

    def pick_first(ocr_text, original_filename, pdf_title, content, ranked, *a, **k):
        best = ranked[0]
        assert best.lexical_rank == 1
        assert best.strength < classifier.STRONG_EVIDENCE and best.sender_files == 0
        return (best.path, False, 0.9, [], best)

    monkeypatch.setattr(classifier, "_pick_folder", pick_first)
    monkeypatch.setattr(classifier, "_decide_folder_step", lambda *a, **k: _decision("stay", confidence=0.99))
    kwargs = dict(
        ocr_text="text", original_filename="scan.pdf", existing_folders=TREE, ollama_host="http://fake",
        model="model", content=letter, folder_files=TREE_FILES,
    )

    outcome_plain, tags_plain = classifier.classify(**kwargs)
    assert outcome_plain.confidence == classifier.CONFIDENCE_UNBACKED

    favourite = "Ordner: " + " > ".join(outcome_plain.folder.split("/")[1:])
    outcome_agreed, tags_agreed = classifier.classify(**kwargs, embedder=_FakeEmbedder(favourite))
    assert outcome_agreed.folder == outcome_plain.folder
    assert outcome_agreed.confidence == classifier.CONFIDENCE_AGREED
    assert "WORT-UND-BEDEUTUNG-EINIG" in tags_agreed

    # the embedding model preferring some other folder changes nothing
    outcome_other, tags_other = classifier.classify(**kwargs, embedder=_FakeEmbedder("Ordner: Gesundheit"))
    assert outcome_other.confidence == classifier.CONFIDENCE_UNBACKED
    assert "WORT-UND-BEDEUTUNG-EINIG" not in tags_other
