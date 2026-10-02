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


def test_build_filename_with_correspondent():
    name = naming.build_filename(
        "Stromrechnung Juli", date(2026, 7, 15), date(2026, 8, 28), correspondent="Stadtwerke München"
    )
    assert name == "2026-07-15 Stadtwerke München - Stromrechnung Juli.pdf"


def test_build_filename_without_correspondent_unchanged():
    name = naming.build_filename("Stromrechnung Juli", date(2026, 7, 15), date(2026, 8, 28), correspondent=None)
    assert name == "2026-07-15 Stromrechnung Juli.pdf"


def test_build_filename_blank_correspondent_is_omitted():
    name = naming.build_filename("Stromrechnung Juli", date(2026, 7, 15), date(2026, 8, 28), correspondent="   ")
    assert name == "2026-07-15 Stromrechnung Juli.pdf"


def test_build_filename_correspondent_and_uncertain_date():
    name = naming.build_filename("Kontoauszug", None, date(2026, 8, 28), correspondent="Sparkasse")
    assert name == "2026-08-28 Sparkasse - Kontoauszug (Datum unsicher).pdf"


def test_build_filename_truncates_overly_long_result():
    long_title = "Sehr " * 60 + "langer Titel"
    name = naming.build_filename(long_title, date(2026, 7, 15), date(2026, 8, 28), correspondent="Ein Absender")
    assert len(name) <= naming.MAX_FILENAME_LENGTH
    assert name.startswith("2026-07-15 Ein Absender - Sehr ")


def test_sanitize_correspondent_strips_invalid_chars():
    assert naming.sanitize_correspondent('Stadtwerke: München?') == "Stadtwerke München"


def test_sanitize_correspondent_blank_returns_empty_string():
    assert naming.sanitize_correspondent("") == ""
    assert naming.sanitize_correspondent(None) == ""
    assert naming.sanitize_correspondent("///???") == ""


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


def test_duplicate_filename_marks_the_first_copys_name():
    assert naming.duplicate_filename("2026-09-29 ROLAND - Antrag.pdf", ".pdf") == "2026-09-29 ROLAND - Antrag (Duplikat).pdf"
    # the duplicate is stored as the raw scan, so it keeps its own extension
    assert naming.duplicate_filename("2026-09-29 Foto.pdf", "jpg") == "2026-09-29 Foto (Duplikat).jpg"


# ---- correspondent normalization ------------------------------------------------

def test_strip_legal_form_and_address():
    assert naming.strip_legal_form("Stadtwerke Musterstadt Servicegesellschaft mbH") == "Stadtwerke Musterstadt Servicegesellschaft"
    assert naming.strip_legal_form("Muster GmbH & Co. KG") == "Muster"
    assert naming.strip_legal_form("Bezirkswahlamt Musterbezirk, 12345 Berlin") == "Bezirkswahlamt Musterbezirk"
    assert naming.strip_legal_form("Sportverein Beispiel e.V.") == "Sportverein Beispiel"
    assert naming.strip_legal_form("Finanzamt") == "Finanzamt"


def test_strip_legal_form_leaves_names_that_only_look_like_one():
    # "AG" at the start is a court (Amtsgericht), not a stock corporation
    assert naming.strip_legal_form("AG Charlottenburg") == "AG Charlottenburg"
    # part of a hyphenated name, not a separate word
    assert naming.strip_legal_form("Beispiel Rechtsschutz-Versicherungs-AG") == "Beispiel Rechtsschutz-Versicherungs-AG"


def test_known_correspondents_come_from_depots_own_filenames():
    files = {
        "Dokumente/A": [
            "2026-01-01 Gesundkasse - Bescheid.pdf", "2026-02-01 Gesundkasse - Rechnung.pdf", "irgendwas.pdf",
        ],
        "Dokumente/B": ["2026-03-01 Muster GmbH - Abrechnung.pdf", "2026-03-02 Nur ein Titel.pdf"],
    }
    assert naming.known_correspondents(files) == ["Gesundkasse", "Muster"]


def test_normalize_correspondent_takes_the_spelling_already_in_use():
    known = ["Techniker Krankenkasse", "Muster"]
    assert naming.normalize_correspondent("Techniker Krankenkase", known) == "Techniker Krankenkasse"
    assert naming.normalize_correspondent("Muster GmbH", known) == "Muster"
    assert naming.normalize_correspondent("Ganz Anderer Absender AG", known) == "Ganz Anderer Absender"
    assert naming.normalize_correspondent("", known) == ""


# ---- umlauts and sender cleanup ---------------------------------------------

def test_restore_umlauts_only_where_the_document_writes_them():
    from depot.naming import restore_umlauts

    text = "Schreiben über die Änderung Ihrer Steuerklasse. Aktuelle Bußgeldstelle, Straße 1, Wasser"
    assert restore_umlauts("Schreiben ueber Aenderung der Steuerklasse", text) == "Schreiben über Änderung der Steuerklasse"
    assert restore_umlauts("Bussgeldstelle Strasse", text) == "Bußgeldstelle Straße"
    # words that merely contain ae/oe/ue/ss keep their spelling
    assert restore_umlauts("Steuerklasse aktuelle Wasser", text) == "Steuerklasse aktuelle Wasser"
    # the document itself uses the plain spelling: leave it
    assert restore_umlauts("Mueller Rechnung", "Rechnung von Mueller und Müller") == "Mueller Rechnung"
    # not in the document at all: no guessing
    assert restore_umlauts("Pruefbericht", "Bericht") == "Pruefbericht"
    assert restore_umlauts("Pruefbericht", "") == "Pruefbericht"


def test_sender_loses_mail_addresses_and_the_letter_left_behind():
    from depot.naming import normalize_correspondent

    assert normalize_correspondent("Polizei Berlin L bussgeldstelle@bowi.berlin.de") == "Polizei Berlin"
    assert normalize_correspondent("Musterwerke GmbH www.musterwerke.de") == "Musterwerke"
    assert normalize_correspondent("Kanzlei B") == "Kanzlei B"  # no address involved: untouched
