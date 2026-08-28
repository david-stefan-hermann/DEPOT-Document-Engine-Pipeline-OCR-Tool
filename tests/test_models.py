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


def test_confidence_percentage_int_is_rescaled():
    result = ClassificationResult.model_validate(
        {
            "folder": "X",
            "is_new_folder": False,
            "title": "Y",
            "confidence": 95,
        }
    )
    assert result.confidence == 0.95


def test_confidence_wildly_out_of_range_is_clamped():
    result = ClassificationResult.model_validate(
        {
            "folder": "X",
            "is_new_folder": False,
            "title": "Y",
            "confidence": 500,
        }
    )
    assert result.confidence == 1.0


def test_confidence_negative_is_clamped():
    result = ClassificationResult.model_validate(
        {
            "folder": "X",
            "is_new_folder": False,
            "title": "Y",
            "confidence": -0.2,
        }
    )
    assert result.confidence == 0.0


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


def test_implausible_future_date_is_discarded():
    result = ClassificationResult.model_validate(
        {
            "folder": "Gesundheit/Diabetes/VitalAire",
            "is_new_folder": False,
            "title": "Zuzahlungsrechnung",
            "issue_date": "3107-07-20",
            "confidence": 0.8,
        }
    )
    assert result.issue_date is None


def test_implausible_ancient_date_is_discarded():
    result = ClassificationResult.model_validate(
        {
            "folder": "Gesundheit",
            "is_new_folder": False,
            "title": "Rechnung",
            "issue_date": "1850-01-01",
            "confidence": 0.8,
        }
    )
    assert result.issue_date is None


def test_plausible_old_date_is_kept():
    result = ClassificationResult.model_validate(
        {
            "folder": "Gesundheit",
            "is_new_folder": False,
            "title": "Rechnung",
            "issue_date": "1998-05-01",
            "confidence": 0.8,
        }
    )
    assert result.issue_date.isoformat() == "1998-05-01"


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
