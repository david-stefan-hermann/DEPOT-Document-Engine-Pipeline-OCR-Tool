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


def test_adjust_confidence_trusts_valid_existing_folder():
    result = _result(folder="Gesundheit/Krankenkasse", is_new_folder=False, confidence=0.9)
    confidence, tags = classifier._adjust_confidence(result, EXISTING_FOLDERS)
    assert confidence == 0.9
    assert tags == []


def test_adjust_confidence_zeroes_hallucinated_folder():
    result = _result(folder="Erfundener/Ordner", is_new_folder=False, confidence=0.95)
    confidence, tags = classifier._adjust_confidence(result, EXISTING_FOLDERS)
    assert confidence == 0.0
    assert "HALLUCINATED-FOLDER" in tags


def test_adjust_confidence_downgrades_near_duplicate_new_folder():
    result = _result(folder="Motorrad/Rechnung", is_new_folder=True, confidence=0.9)
    confidence, tags = classifier._adjust_confidence(result, EXISTING_FOLDERS)
    assert confidence <= 0.4
    assert any("SIMILAR-FOLDER-EXISTS" in t for t in tags)


def test_adjust_confidence_keeps_genuinely_new_folder():
    result = _result(folder="Versicherung/KFZ", is_new_folder=True, confidence=0.8)
    confidence, tags = classifier._adjust_confidence(result, EXISTING_FOLDERS)
    assert confidence == 0.8
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
