from datetime import date

import pymupdf as fitz
import pytest

from depot.signals import (
    FilenameSignals,
    analyze_filename,
    find_dates,
    read_pdf_metadata,
    resolve_issue_date,
    usable_pdf_title,
)

TODAY = date(2026, 9, 30)


# ---- analyze_filename ------------------------------------------------------

@pytest.mark.parametrize("name", [
    "SCN_20260828_220109.pdf",
    "IMG-20260101-WA0007.jpg",
    "scan1.pdf",
    "Scan 3.pdf",
    "WhatsApp Image 2026-09-01 at 12.30.45.jpeg",
])
def test_machine_generated_names_carry_no_title(name):
    assert analyze_filename(name).title is None
    assert analyze_filename(name).date is None


def test_scanner_timestamp_is_only_a_scan_date():
    assert analyze_filename("SCN_20260828_220109.pdf") == FilenameSignals(scan_date=date(2026, 8, 28))


def test_user_name_with_leading_iso_date():
    assert analyze_filename("2026-09-25 Blitzerfoto.jpg") == FilenameSignals(
        title="Blitzerfoto", date=date(2026, 9, 25)
    )


def test_user_name_with_trailing_german_date_and_time():
    result = analyze_filename("Mustermann, Max, Antrag, 15.07 EUR, 28-09-2026 13-52-56.pdf")
    assert result.title == "Mustermann, Max, Antrag, 15.07 EUR"
    assert result.date == date(2026, 9, 28)


def test_user_name_without_date():
    assert analyze_filename("motorrad anhänger standschiene 1.jpg") == FilenameSignals(
        title="motorrad anhänger standschiene 1"
    )
    assert analyze_filename("Windows_11_Key.pdf").title == "Windows 11 Key"


def test_copy_counter_is_not_part_of_the_title():
    assert analyze_filename("Antrag auf Zuzahlungsbefreiung-2.pdf").title == "Antrag auf Zuzahlungsbefreiung"


def test_short_number_groups_are_not_a_date():
    assert analyze_filename("Seite 1-2-3.pdf").date is None


# ---- find_dates ------------------------------------------------------------

def test_find_dates_covers_german_document_formats():
    text = "Datum 14.08.2026, geb. 01.02.85, am 3. März 2026, Frist 2026-07-31, OCR: 14. 09. 2026"
    assert find_dates(text) == {
        date(2026, 8, 14), date(1985, 2, 1), date(2026, 3, 3), date(2026, 7, 31), date(2026, 9, 14),
    }


def test_find_dates_ignores_impossible_dates():
    assert find_dates("Tel. 99.99.2026, Betrag 31.02.2026") == set()


# ---- PDF metadata ----------------------------------------------------------

def test_usable_pdf_title_filters_tool_defaults():
    assert usable_pdf_title("Ende der Familienversicherung") == "Ende der Familienversicherung"
    assert usable_pdf_title("Microsoft Word - Dokument1") is None
    assert usable_pdf_title("SCN_0001.pdf") is None
    assert usable_pdf_title("") is None
    assert usable_pdf_title(None) is None


def test_read_pdf_metadata(tmp_path):
    path = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.set_metadata({"title": "Beitragsanpassung", "creationDate": "D:20260210114041+01'00'"})
    doc.save(path)
    doc.close()

    assert read_pdf_metadata(path) == ("Beitragsanpassung", date(2026, 2, 10))


def test_read_pdf_metadata_never_raises(tmp_path):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-not-really")
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"jpeg")

    assert read_pdf_metadata(broken) == (None, None)
    assert read_pdf_metadata(image) == (None, None)


# ---- resolve_issue_date ----------------------------------------------------

def test_model_date_that_occurs_in_the_text_is_kept():
    result = resolve_issue_date(date(2026, 8, 21), "Datum 21.08.2026", FilenameSignals(), None, TODAY)
    assert result == (date(2026, 8, 21), "model")


def test_model_date_not_in_the_text_is_discarded():
    result = resolve_issue_date(
        date(2024, 8, 21), "Eintritt 01.03.2024 Datum 21.08.2026", FilenameSignals(), None, TODAY
    )
    assert result == (None, None)


def test_model_date_is_kept_when_the_text_has_no_checkable_date():
    result = resolve_issue_date(date(2026, 7, 1), "Abrechnung fuer Juli 2026", FilenameSignals(), None, TODAY)
    assert result == (date(2026, 7, 1), "model")


def test_future_date_is_discarded_even_if_it_occurs_in_the_text():
    """Real case: an election notice was filed under the election day."""
    result = resolve_issue_date(
        date(2026, 10, 20), "Wahltag: 20.10.2026", FilenameSignals(), None, TODAY
    )
    assert result == (None, None)


def test_date_after_the_scan_timestamp_is_discarded():
    scanned = FilenameSignals(scan_date=date(2026, 8, 28))
    result = resolve_issue_date(date(2026, 9, 20), "Termin 20.09.2026", scanned, None, TODAY)
    assert result == (None, None)


def test_filename_date_is_the_fallback():
    named = FilenameSignals(title="Antrag", date=date(2026, 9, 28))
    assert resolve_issue_date(None, "kein Datum im Text", named, None, TODAY) == (date(2026, 9, 28), "filename")


def test_filename_date_validates_a_model_date_missing_from_the_text():
    named = FilenameSignals(title="Antrag", date=date(2026, 9, 28))
    assert resolve_issue_date(date(2026, 9, 28), "Stand 01.01.2020", named, None, TODAY) == (
        date(2026, 9, 28), "model"
    )


def test_pdf_creation_date_is_the_last_fallback():
    assert resolve_issue_date(None, "", FilenameSignals(), date(2026, 2, 10), TODAY) == (
        date(2026, 2, 10), "metadata"
    )
    assert resolve_issue_date(None, "", FilenameSignals(), None, TODAY) == (None, None)
