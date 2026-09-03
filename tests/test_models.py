import pytest
from pydantic import ValidationError

from depot.models import ContentExtraction, FolderStepDecision


# ---- ContentExtraction ----------------------------------------------------

def test_content_extraction_parses_and_strips_title():
    result = ContentExtraction.model_validate(
        {
            "title": "  Arztrechnung  ",
            "correspondent": "Dr. Müller",
            "issue_date": "2026-03-05",
            "confidence": 0.92,
            "reasoning": "Klarer Absender und Betreff.",
        }
    )
    assert result.title == "Arztrechnung"
    assert result.issue_date.isoformat() == "2026-03-05"


def test_content_confidence_percentage_int_is_rescaled():
    result = ContentExtraction.model_validate({"title": "Y", "correspondent": "", "confidence": 95})
    assert result.confidence == 0.95


def test_content_confidence_wildly_out_of_range_is_clamped():
    result = ContentExtraction.model_validate({"title": "Y", "correspondent": "", "confidence": 500})
    assert result.confidence == 1.0


def test_content_confidence_negative_is_clamped():
    result = ContentExtraction.model_validate({"title": "Y", "correspondent": "", "confidence": -0.2})
    assert result.confidence == 0.0


def test_content_missing_issue_date_is_none():
    result = ContentExtraction.model_validate({"title": "Unklares Dokument", "correspondent": "", "confidence": 0.1})
    assert result.issue_date is None


def test_content_implausible_future_date_is_discarded():
    result = ContentExtraction.model_validate(
        {"title": "Zuzahlungsrechnung", "correspondent": "", "issue_date": "3107-07-20", "confidence": 0.8}
    )
    assert result.issue_date is None


def test_content_implausible_ancient_date_is_discarded():
    result = ContentExtraction.model_validate(
        {"title": "Rechnung", "correspondent": "", "issue_date": "1850-01-01", "confidence": 0.8}
    )
    assert result.issue_date is None


def test_content_plausible_old_date_is_kept():
    result = ContentExtraction.model_validate(
        {"title": "Rechnung", "correspondent": "", "issue_date": "1998-05-01", "confidence": 0.8}
    )
    assert result.issue_date.isoformat() == "1998-05-01"


def test_content_empty_title_rejected():
    with pytest.raises(ValidationError):
        ContentExtraction.model_validate({"title": "", "correspondent": "", "confidence": 0.5})


def test_content_correspondent_is_stripped():
    result = ContentExtraction.model_validate(
        {"title": "Stromrechnung Juli", "correspondent": "  Stadtwerke München  ", "confidence": 0.9}
    )
    assert result.correspondent == "Stadtwerke München"


def test_content_correspondent_is_required():
    with pytest.raises(ValidationError):
        ContentExtraction.model_validate({"title": "Notiz", "confidence": 0.5})


def test_content_blank_correspondent_becomes_empty_string():
    result = ContentExtraction.model_validate({"title": "Notiz", "correspondent": "   ", "confidence": 0.5})
    assert result.correspondent == ""


# ---- FolderStepDecision ----------------------------------------------------

def test_folder_step_decision_parses_descend():
    result = FolderStepDecision.model_validate(
        {"action": "descend", "folder_name": "  Gesundheit  ", "confidence": 0.9}
    )
    assert result.action == "descend"
    assert result.folder_name == "Gesundheit"


def test_folder_step_decision_stay_with_no_folder_name():
    result = FolderStepDecision.model_validate({"action": "stay", "confidence": 0.7})
    assert result.action == "stay"
    assert result.folder_name is None


def test_folder_step_decision_invalid_action_rejected():
    with pytest.raises(ValidationError):
        FolderStepDecision.model_validate({"action": "explode", "confidence": 0.5})


def test_folder_step_decision_confidence_percentage_is_rescaled():
    result = FolderStepDecision.model_validate({"action": "stay", "confidence": 80})
    assert result.confidence == 0.8
