import json
from datetime import date

import ollama
import pytest

from depot import classifier
from depot.models import ContentExtraction, FolderStepDecision


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

    def fake_decide(ocr_text, original_filename, current_path, children, ollama_host, model, timeout=120.0):
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

    def fake_decide(ocr_text, original_filename, current_path, children, ollama_host, model, timeout=120.0):
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
    def fake_decide(ocr_text, original_filename, current_path, children, ollama_host, model, timeout=120.0):
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
    def fake_decide(ocr_text, original_filename, current_path, children, ollama_host, model, timeout=120.0):
        return _decision("new_folder", "Versicherung", confidence=0.85)

    monkeypatch.setattr(classifier, "_decide_folder_step", fake_decide)

    folder, is_new, confidence, tags = classifier._walk_folder_tree(
        "text", "scan.pdf", EXISTING_FOLDERS, "Dokumente", "http://fake", "model"
    )
    assert folder == "Dokumente/Versicherung"
    assert is_new is True
    assert tags == []


def test_walk_redirects_near_duplicate_new_folder_to_existing_sibling(monkeypatch):
    def fake_decide(ocr_text, original_filename, current_path, children, ollama_host, model, timeout=120.0):
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

    def chat(self, model, messages, format, options):
        return {"message": {"content": json.dumps(self._payload)}}


def test_extract_content_parses_valid_response(monkeypatch):
    canned = {"title": "Stromrechnung Juli", "issue_date": "2026-07-15", "confidence": 0.9}
    monkeypatch.setattr(ollama, "Client", lambda *a, **k: _FakeClient(canned))

    result = classifier.extract_content("ocr text", "scan.pdf", "http://fake", "model")
    assert isinstance(result, ContentExtraction)
    assert result.title == "Stromrechnung Juli"
    assert result.issue_date == date(2026, 7, 15)


def test_extract_content_raises_on_invalid_json(monkeypatch):
    class BadClient:
        def chat(self, model, messages, format, options):
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

def test_classify_combines_content_and_folder_walk(monkeypatch):
    monkeypatch.setattr(
        classifier, "extract_content",
        lambda *a, **k: ContentExtraction(title="Arztrechnung", issue_date=date(2026, 6, 10), confidence=0.8),
    )
    monkeypatch.setattr(
        classifier, "_walk_folder_tree",
        lambda *a, **k: ("Dokumente/Gesundheit/Krankenkasse", False, 0.95, []),
    )

    outcome, tags = classifier.classify(
        ocr_text="text",
        original_filename="scan.pdf",
        existing_folders=EXISTING_FOLDERS,
        ollama_host="http://fake",
        model="model",
    )
    assert outcome.folder == "Dokumente/Gesundheit/Krankenkasse"
    assert outcome.title == "Arztrechnung"
    assert outcome.issue_date == date(2026, 6, 10)
    assert outcome.confidence == 0.8  # min(content=0.8, folder=0.95)
    assert tags == []
