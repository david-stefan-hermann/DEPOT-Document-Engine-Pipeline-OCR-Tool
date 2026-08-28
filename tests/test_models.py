import pytest
from pydantic import ValidationError

from depot.models import ClassificationResult


def test_valid_result_parses():
    result = ClassificationResult.model_validate(
        {
            "folder": "/Gesundheit/Krankenkasse/",
            "is_new_folder": False,
            "title": "  Arztrechnung  ",
            "issue_date": "2026-03-05",
            "confidence": 0.92,
            "reasoning": "Klarer Absender und Betreff.",
        }
    )
    assert result.folder == "Gesundheit/Krankenkasse"
    assert result.title == "Arztrechnung"
    assert result.issue_date.isoformat() == "2026-03-05"


def test_confidence_out_of_range_rejected():
    with pytest.raises(ValidationError):
        ClassificationResult.model_validate(
            {
                "folder": "X",
                "is_new_folder": False,
                "title": "Y",
                "confidence": 1.5,
            }
        )


def test_missing_issue_date_is_none():
    result = ClassificationResult.model_validate(
        {
            "folder": "Unsortiert",
            "is_new_folder": False,
            "title": "Unklares Dokument",
            "confidence": 0.1,
        }
    )
    assert result.issue_date is None


def test_empty_folder_rejected():
    with pytest.raises(ValidationError):
        ClassificationResult.model_validate(
            {
                "folder": "",
                "is_new_folder": False,
                "title": "Y",
                "confidence": 0.5,
            }
        )
