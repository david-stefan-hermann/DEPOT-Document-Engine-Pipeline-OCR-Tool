from depot.candidates import DocumentQuery, fold, rank_candidates, tokenize

FOLDERS = [
    "Dokumente/Versicherungen",
    "Dokumente/Fahrzeuge",
    "Dokumente/Fahrzeuge/MT-07",
    "Dokumente/Fahrzeuge/MT-07/Versicherung",
    "Dokumente/Fahrzeuge/MT-07/Rechnungen",
    "Dokumente/Gesundheit",
    "Dokumente/Gesundheit/Krankenkasse",
    "Dokumente/Finanzen",
    "Dokumente/Finanzen/Steuern",
    "Dokumente/Arbeit",
    "Dokumente/Arbeit/Muster Grundstücksservice",
]

FILES = {
    "Dokumente/Fahrzeuge/MT-07/Versicherung": [
        "2025-03-01 Beispiel Versicherung - Beitragsrechnung.pdf",
        "2024-03-01 Beispiel Versicherung - Versicherungsschein.pdf",
    ],
    "Dokumente/Fahrzeuge/MT-07/Rechnungen": ["2025-05-10 Werkstatt Nord - Inspektion.pdf"],
    "Dokumente/Gesundheit/Krankenkasse": [
        "2026-01-22 Gesundkasse - Bestätigung für die Steuererklärung.pdf",
        "2026-04-02 Gesundkasse - Ende Familienversicherung.pdf",
    ],
    "Dokumente/Finanzen/Steuern": ["2025-06-01 Finanzamt Musterstadt - Einkommensteuerbescheid.pdf"],
}


def _paths(query: DocumentQuery, **kwargs) -> list[str]:
    return [c.path for c in rank_candidates(query, FOLDERS, FILES, **kwargs)]


def test_tokenize_folds_umlauts_and_joins_hyphenated_names():
    assert fold("Grundstücksservice") == fold("GRUNDSTUECKSSERVICE")
    assert "mt07" in tokenize("Yamaha MT-07")
    # legal forms, function words and bare numbers carry no meaning
    assert tokenize("Die Muster GmbH & Co. KG 2026") == ["muster"]


def test_sender_history_beats_a_merely_similar_folder_name():
    """The real failure this exists for: an insurer's letter about a vehicle
    went to the general "Versicherungen" folder although every earlier
    letter from that insurer lies with the vehicle."""
    paths = _paths(DocumentQuery(correspondent="Beispiel Versicherung", title="Beitragsrechnung"))
    assert paths[0] == "Dokumente/Fahrzeuge/MT-07/Versicherung"
    assert "Dokumente/Versicherungen" in paths


def test_sender_without_a_folder_of_its_own_is_found_through_filed_documents():
    """No folder is called "Finanzamt" - but a filed document of that sender
    shows where its letters go."""
    paths = _paths(DocumentQuery(correspondent="Finanzamt Musterstadt", title="Änderung der Steuerklasse"))
    assert paths[0] == "Dokumente/Finanzen/Steuern"


def test_folder_name_occurring_in_the_document_text_is_a_candidate():
    paths = _paths(DocumentQuery(
        correspondent="Werkstatt Süd", title="Rechnung",
        text="Rechnung für Ihre Yamaha MT-07, amtliches Kennzeichen ...",
    ))
    assert paths[0] == "Dokumente/Fahrzeuge/MT-07/Rechnungen"


def test_sender_matching_a_folder_name_is_found_without_any_files():
    paths = _paths(DocumentQuery(correspondent="Muster Grundstuecksservice GmbH", title="Entgeltabrechnung"))
    assert paths[0] == "Dokumente/Arbeit/Muster Grundstücksservice"


def test_candidates_carry_related_example_files_and_the_sender_count():
    (best, *_) = rank_candidates(
        DocumentQuery(correspondent="Gesundkasse", title="Ende der Familienversicherung"), FOLDERS, FILES
    )
    assert best.path == "Dokumente/Gesundheit/Krankenkasse"
    assert best.sender_files == 2
    assert best.examples[0] == "2026-04-02 Gesundkasse - Ende Familienversicherung.pdf"


def test_folder_with_only_subfolders_shows_them_as_examples():
    ranked = rank_candidates(DocumentQuery(title="MT-07 Unterlagen"), FOLDERS, FILES)
    parent = next(c for c in ranked if c.path == "Dokumente/Fahrzeuge/MT-07")
    assert parent.examples == ["Unterordner: Rechnungen, Versicherung"]


def test_works_with_folder_names_only():
    ranked = rank_candidates(DocumentQuery(correspondent="Gesundkasse", title="Krankenkasse Beitrag"), FOLDERS, None)
    assert ranked[0].path == "Dokumente/Gesundheit/Krankenkasse"
    assert ranked[0].examples == []


def test_nothing_in_common_yields_no_candidates():
    assert rank_candidates(DocumentQuery(correspondent="Xyzzy", title="Qwertz"), FOLDERS, FILES) == []
    assert rank_candidates(DocumentQuery(title="Rechnung"), [], {}) == []


def test_limit_caps_the_shortlist():
    assert len(_paths(DocumentQuery(title="Versicherung Rechnung Steuern Krankenkasse"), limit=2)) == 2


# ---- year folders, parent vs. subfolder, evidence strength ---------------------

BANK_FOLDERS = [
    "Dokumente/Finanzen",
    "Dokumente/Finanzen/Musterbank",
    "Dokumente/Finanzen/Musterbank/2024",
    "Dokumente/Finanzen/Musterbank/2025",
    "Dokumente/Arbeit",
    "Dokumente/Arbeit/Beispiel Logistik",
    "Dokumente/Arbeit/Beispiel Logistik/Arbeitgeber",
    "Dokumente/Arbeit/Beispiel Logistik/Technik",
    "Dokumente/Archiv/Seite.html_files",
]
BANK_FILES = {
    "Dokumente/Finanzen/Musterbank/2024": ["2024-01-04-Wertpapierabrechnung-A.pdf", "2024-02-01-Wertpapierabrechnung-B.pdf"],
    "Dokumente/Finanzen/Musterbank/2025": ["2025-01-02-Wertpapierabrechnung-A.pdf"],
    "Dokumente/Arbeit/Beispiel Logistik": ["Beispiel Logistik.xlsx"],
    "Dokumente/Arbeit/Beispiel Logistik/Arbeitgeber": ["2023_02_Gehaltsabrechnung.pdf", "2023_03_Gehaltsabrechnung.pdf"],
    "Dokumente/Arbeit/Beispiel Logistik/Technik": ["Netzplan.drawio"],
    "Dokumente/Archiv/Seite.html_files": ["wertpapierabrechnung.css"],
}


def test_year_folders_are_folded_into_their_parent():
    """Which year a document goes into follows from its date - the choice
    is only ever between the folders above."""
    ranked = rank_candidates(
        DocumentQuery(correspondent="Musterbank", title="Wertpapierabrechnung"), BANK_FOLDERS, BANK_FILES
    )
    assert ranked[0].path == "Dokumente/Finanzen/Musterbank"
    assert not any(c.path.endswith(("/2024", "/2025")) for c in ranked)
    # the files of the year folders are the parent's examples
    assert ranked[0].examples[0].endswith("Wertpapierabrechnung-A.pdf")


def test_saved_web_page_asset_folders_are_never_candidates():
    ranked = rank_candidates(DocumentQuery(title="Wertpapierabrechnung"), BANK_FOLDERS, BANK_FILES)
    assert not any("html_files" in c.path for c in ranked)


def test_subfolder_holding_similar_documents_goes_before_the_senders_own_folder():
    """The folder named after the employer only holds a spreadsheet and the
    subfolders; the payslips are in one of those."""
    ranked = rank_candidates(
        DocumentQuery(correspondent="Beispiel Logistik GmbH", title="Gehaltsabrechnung März"),
        BANK_FOLDERS, BANK_FILES,
    )
    paths = [c.path for c in ranked]
    assert paths[0] == "Dokumente/Arbeit/Beispiel Logistik/Arbeitgeber"
    assert paths.index("Dokumente/Arbeit/Beispiel Logistik") < paths.index("Dokumente/Arbeit/Beispiel Logistik/Technik")


def test_strength_separates_sender_evidence_from_coincidental_words():
    from depot.candidates import STRONG_EVIDENCE

    (by_sender, *_) = rank_candidates(
        DocumentQuery(correspondent="Musterbank", title="Wertpapierabrechnung"), BANK_FOLDERS, BANK_FILES
    )
    (by_chance, *_) = rank_candidates(
        DocumentQuery(correspondent="Unbekannt", title="Netzplan Skizze"), BANK_FOLDERS, BANK_FILES
    )
    assert by_sender.strength >= STRONG_EVIDENCE
    assert 0 < by_chance.strength < STRONG_EVIDENCE


def test_sender_count_ignores_files_that_only_share_a_common_word():
    """A city name in both sender and an unrelated filename is not "a file
    of this sender" (real case: "... Berlin" matched a polling notice)."""
    folders = ["Dokumente/Behörden", "Dokumente/Gesundheit"]
    files = {"Dokumente/Behörden": ["2026-09-20 Bezirkswahlamt Musterbezirk Berlin - Wahlbenachrichtigung.pdf"]}
    ranked = rank_candidates(DocumentQuery(correspondent="Praxiszentrum Berlin-Mitte", title="Bericht"), folders, files)
    assert all(c.sender_files == 0 for c in ranked)

