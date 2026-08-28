from datetime import date

from depot.depotlog import DepotLog, is_log_file, log_filename


def test_is_log_file_matches_prefix():
    assert is_log_file("DEPOT Dateilog 28-08-2026.txt", "DEPOT Dateilog")
    assert not is_log_file("2026-07-15 Stromrechnung.pdf", "DEPOT Dateilog")


def test_log_filename_format():
    assert log_filename("DEPOT Dateilog", date(2026, 8, 28)) == "DEPOT Dateilog 28-08-2026.txt"


def test_append_creates_new_daily_log(client):
    log = DepotLog(client, "Scan-Eingang", "DEPOT Dateilog")
    log.append("scan1.pdf", "-> Dokumente/Energie/... | confidence=0.90", on_date=date(2026, 8, 28))

    content = client.get("Scan-Eingang/DEPOT Dateilog 28-08-2026.txt")
    assert content is not None
    text = content.decode("utf-8")
    assert "scan1.pdf" in text
    assert "confidence=0.90" in text


def test_append_appends_to_same_day_log(client):
    log = DepotLog(client, "Scan-Eingang", "DEPOT Dateilog")
    log.append("scan1.pdf", "erster Eintrag", on_date=date(2026, 8, 28))
    log.append("scan2.pdf", "zweiter Eintrag", on_date=date(2026, 8, 28))

    text = client.get("Scan-Eingang/DEPOT Dateilog 28-08-2026.txt").decode("utf-8")
    lines = [l for l in text.splitlines() if l.strip()]
    assert len(lines) == 2
    assert "scan1.pdf" in lines[0]
    assert "scan2.pdf" in lines[1]


def test_append_uses_separate_file_per_day(client):
    log = DepotLog(client, "Scan-Eingang", "DEPOT Dateilog")
    log.append("scan1.pdf", "gestern", on_date=date(2026, 8, 27))
    log.append("scan2.pdf", "heute", on_date=date(2026, 8, 28))

    assert client.get("Scan-Eingang/DEPOT Dateilog 27-08-2026.txt") is not None
    assert client.get("Scan-Eingang/DEPOT Dateilog 28-08-2026.txt") is not None


def test_append_includes_tags(client):
    log = DepotLog(client, "Scan-Eingang", "DEPOT Dateilog")
    log.append("scan1.pdf", "-> Dokumente/Unsortiert/x.pdf", tags=["UNSORTIERT", "OCR-FEHLGESCHLAGEN"], on_date=date(2026, 8, 28))

    text = client.get("Scan-Eingang/DEPOT Dateilog 28-08-2026.txt").decode("utf-8")
    assert "UNSORTIERT" in text
    assert "OCR-FEHLGESCHLAGEN" in text
