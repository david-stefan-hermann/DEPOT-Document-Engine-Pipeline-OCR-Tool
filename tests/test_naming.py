from datetime import date

from depot import naming


def test_sanitize_title_strips_invalid_chars():
    assert naming.sanitize_title('Stromrechnung: Juli/2026 "Mahnung"?') == "Stromrechnung Juli2026 Mahnung"


def test_sanitize_title_collapses_whitespace():
    assert naming.sanitize_title("Bußgeld   bescheid\n\nOrdnungsamt") == "Bußgeld bescheid Ordnungsamt"


def test_sanitize_title_falls_back_when_empty():
    assert naming.sanitize_title("///???") == "Dokument"


def test_build_filename_with_issue_date():
    name = naming.build_filename("Stromrechnung Juli", date(2026, 7, 15), date(2026, 8, 28))
    assert name == "2026-07-15 Stromrechnung Juli.pdf"


def test_build_filename_without_issue_date_marks_uncertain():
    name = naming.build_filename("Kontoauszug", None, date(2026, 8, 28))
    assert name == "2026-08-28 Kontoauszug (Datum unsicher).pdf"


def test_build_filename_respects_custom_extension():
    name = naming.build_filename("Foto", None, date(2026, 8, 28), ext=".jpg")
    assert name.endswith(".jpg")


def test_resolve_collision_no_conflict():
    assert naming.resolve_collision("2026-07-15 Miete.pdf", set()) == "2026-07-15 Miete.pdf"


def test_resolve_collision_increments_counter():
    existing = {"2026-07-15 Miete.pdf", "2026-07-15 Miete (2).pdf"}
    assert naming.resolve_collision("2026-07-15 Miete.pdf", existing) == "2026-07-15 Miete (3).pdf"


def test_folder_similarity_identical():
    assert naming.folder_similarity("Rechnungen", "Rechnungen") == 1.0


def test_folder_similarity_near_duplicate_is_high():
    assert naming.folder_similarity("Rechnung", "Rechnungen") > 0.85


def test_folder_similarity_unrelated_is_low():
    assert naming.folder_similarity("Gesundheit", "Motorrad") < 0.5


def test_closest_existing_leaf_finds_best_match():
    existing = ["Gesundheit/Krankenkasse", "Motorrad/Rechnungen", "Energie/Rechnungen"]
    best = naming.closest_existing_leaf("Motorrad/Rechnung", existing)
    assert best is not None
    match, ratio = best
    assert match in ("Motorrad/Rechnungen", "Energie/Rechnungen")
    assert ratio > 0.85


def test_closest_existing_leaf_empty_list():
    assert naming.closest_existing_leaf("Neu/Ordner", []) is None
