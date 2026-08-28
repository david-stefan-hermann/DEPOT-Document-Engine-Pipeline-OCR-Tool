import json

import ollama
import pytest

from depot import classifier
from depot.models import ClassificationResult


EXISTING_FOLDERS = ["Gesundheit/Krankenkasse", "Motorrad/Rechnungen", "Energie/Rechnungen"]


def _result(**overrides) -> ClassificationResult:
    data = dict(
        folder="Gesundheit/Krankenkasse",
        is_new_folder=False,
        title="Arztrechnung",
        issue_date=None,
        confidence=0.9,
        reasoning="",
    )
    data.update(overrides)
    return ClassificationResult.model_validate(data)


def test_resolve_folder_trusts_valid_existing_folder():
    result = _result(folder="Gesundheit/Krankenkasse", is_new_folder=False, confidence=0.9)
    resolved, tags = classifier._resolve_folder(result, EXISTING_FOLDERS)
    assert resolved.confidence == 0.9
    assert resolved.folder == "Gesundheit/Krankenkasse"
    assert resolved.is_new_folder is False
    assert tags == []


def test_resolve_folder_zeroes_hallucinated_folder():
    result = _result(folder="Erfundener/Ordner", is_new_folder=False, confidence=0.95)
    resolved, tags = classifier._resolve_folder(result, EXISTING_FOLDERS)
    assert resolved.confidence == 0.0
    assert "HALLUCINATED-FOLDER" in tags


def test_resolve_folder_redirects_near_duplicate_new_folder_to_existing():
    result = _result(folder="Motorrad/Rechnung", is_new_folder=True, confidence=0.9)
    resolved, tags = classifier._resolve_folder(result, EXISTING_FOLDERS)
    # Confidence is kept as-is (not downgraded) since the file is actually
    # filed under the existing folder, not dumped to a fallback.
    assert resolved.confidence == 0.9
    assert resolved.folder == "Motorrad/Rechnungen"
    assert resolved.is_new_folder is False
    assert any("AUTO-REDIRECTED" in t for t in tags)


def test_resolve_folder_keeps_genuinely_new_folder():
    result = _result(folder="Versicherung/KFZ", is_new_folder=True, confidence=0.8)
    resolved, tags = classifier._resolve_folder(result, EXISTING_FOLDERS)
    assert resolved.confidence == 0.8
    assert resolved.folder == "Versicherung/KFZ"
    assert resolved.is_new_folder is True
    assert tags == []


def test_classify_parses_valid_ollama_response(monkeypatch):
    canned = {
        "folder": "Gesundheit/Krankenkasse",
        "is_new_folder": False,
        "title": "Arztrechnung Juni",
        "issue_date": "2026-06-10",
        "confidence": 0.88,
        "reasoning": "Absender ist die Krankenkasse.",
    }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def chat(self, model, messages, format, options):
            return {"message": {"content": json.dumps(canned)}}

    monkeypatch.setattr(ollama, "Client", FakeClient)

    result, tags = classifier.classify(
        ocr_text="Sehr geehrter Herr Muster, anbei Ihre Arztrechnung...",
        original_filename="scan001.pdf",
        existing_folders=EXISTING_FOLDERS,
        ollama_host="http://fake:11434",
        model="fake-model",
    )
    assert result.folder == "Gesundheit/Krankenkasse"
    assert result.confidence == 0.88
    assert tags == []


def test_classify_raises_on_invalid_json(monkeypatch):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def chat(self, model, messages, format, options):
            return {"message": {"content": "not json"}}

    monkeypatch.setattr(ollama, "Client", FakeClient)

    with pytest.raises(RuntimeError):
        classifier.classify(
            ocr_text="...",
            original_filename="scan001.pdf",
            existing_folders=EXISTING_FOLDERS,
            ollama_host="http://fake:11434",
            model="fake-model",
        )
