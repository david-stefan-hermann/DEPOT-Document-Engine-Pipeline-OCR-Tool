from datetime import date, time

from depot.depotlog import DepotLog, is_log_file, log_filename


def test_is_log_file_matches_prefix():
    assert is_log_file("DEPOT Dateilog 28-08-2026 14-30-05.txt", "DEPOT Dateilog")
    assert not is_log_file("2026-07-15 Stromrechnung.pdf", "DEPOT Dateilog")


def test_log_filename_format():
    assert (
        log_filename("DEPOT Dateilog", date(2026, 8, 28), time(14, 30, 5))
        == "DEPOT Dateilog 28-08-2026 14-30-05.txt"
    )


def test_append_writes_into_config_subfolder(client):
    log = DepotLog(client, "Scan-Eingang", "DEPOT Dateilog", "Config")
    log.append(
        "scan1.pdf",
        "-> Dokumente/Energie/... | confidence=0.90",
        on_date=date(2026, 8, 28),
        on_time=time(14, 30, 5),
    )

    content = client.get("Scan-Eingang/Config/DEPOT Dateilog 28-08-2026 14-30-05.txt")
    assert content is not None
    text = content.decode("utf-8")
    assert "scan1.pdf" in text
    assert "confidence=0.90" in text


def test_append_uses_default_config_subfolder(client):
    log = DepotLog(client, "Scan-Eingang", "DEPOT Dateilog")
    log.append("scan1.pdf", "erster Eintrag", on_date=date(2026, 8, 28), on_time=time(9, 0, 0))

    assert client.get("Scan-Eingang/Config/DEPOT Dateilog 28-08-2026 09-00-00.txt") is not None


def test_append_creates_a_separate_file_per_event(client):
    log = DepotLog(client, "Scan-Eingang", "DEPOT Dateilog", "Config")
    log.append("scan1.pdf", "erster Eintrag", on_date=date(2026, 8, 28), on_time=time(9, 0, 0))
    log.append("scan2.pdf", "zweiter Eintrag", on_date=date(2026, 8, 28), on_time=time(9, 0, 1))

    text1 = client.get("Scan-Eingang/Config/DEPOT Dateilog 28-08-2026 09-00-00.txt").decode("utf-8")
    text2 = client.get("Scan-Eingang/Config/DEPOT Dateilog 28-08-2026 09-00-01.txt").decode("utf-8")
    assert "scan1.pdf" in text1
    assert "scan2.pdf" in text2


def test_append_disambiguates_same_second_collisions(client):
    log = DepotLog(client, "Scan-Eingang", "DEPOT Dateilog", "Config")
    log.append("scan1.pdf", "erster Eintrag", on_date=date(2026, 8, 28), on_time=time(9, 0, 0))
    log.append("scan2.pdf", "zweiter Eintrag", on_date=date(2026, 8, 28), on_time=time(9, 0, 0))

    assert client.get("Scan-Eingang/Config/DEPOT Dateilog 28-08-2026 09-00-00.txt") is not None
    assert client.get("Scan-Eingang/Config/DEPOT Dateilog 28-08-2026 09-00-00 (2).txt") is not None


def test_append_includes_tags(client):
    log = DepotLog(client, "Scan-Eingang", "DEPOT Dateilog", "Config")
    log.append(
        "scan1.pdf",
        "confidence=0.10",
        tags=["UNSORTIERT", "OCR-FEHLGESCHLAGEN"],
        on_date=date(2026, 8, 28),
        on_time=time(9, 0, 0),
    )

    text = client.get("Scan-Eingang/Config/DEPOT Dateilog 28-08-2026 09-00-00.txt").decode("utf-8")
    assert "UNSORTIERT" in text
    assert "OCR-FEHLGESCHLAGEN" in text


def test_append_path_is_always_the_last_thing_on_the_line(client):
    log = DepotLog(client, "Scan-Eingang", "DEPOT Dateilog", "Config")
    log.append(
        "scan1.pdf",
        "confidence=0.90",
        tags=["NEUER-ORDNER"],
        path="Dokumente/Energie/Rechnungen/2026-07-15 Stromrechnung.pdf",
        on_date=date(2026, 8, 28),
        on_time=time(9, 0, 0),
    )

    text = client.get("Scan-Eingang/Config/DEPOT Dateilog 28-08-2026 09-00-00.txt").decode("utf-8")
    line = [l for l in text.splitlines() if l.strip()][0]
    assert line.endswith("Dokumente/Energie/Rechnungen/2026-07-15 Stromrechnung.pdf")


def test_append_without_path_has_no_trailing_pipe(client):
    log = DepotLog(client, "Scan-Eingang", "DEPOT Dateilog", "Config")
    log.append(
        "scan1.pdf", "Fehler (1/3 Versuche): boom", tags=["FEHLER"],
        on_date=date(2026, 8, 28), on_time=time(9, 0, 0),
    )

    text = client.get("Scan-Eingang/Config/DEPOT Dateilog 28-08-2026 09-00-00.txt").decode("utf-8")
    line = [l for l in text.splitlines() if l.strip()][0]
    assert not line.rstrip().endswith("|")


def test_append_recovers_from_transient_write_failure(monkeypatch, client):
    import depot.depotlog as depotlog_module

    monkeypatch.setattr(depotlog_module, "_WRITE_RETRY_DELAY_SECONDS", 0.01)

    log = DepotLog(client, "Scan-Eingang", "DEPOT Dateilog", "Config")
    real_put = client.put
    calls = {"n": 0}

    def flaky_put(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("423 Locked")
        return real_put(*args, **kwargs)

    monkeypatch.setattr(client, "put", flaky_put)

    log.append("scan1.pdf", "confidence=0.90", on_date=date(2026, 8, 28), on_time=time(9, 0, 0))  # must not raise

    text = client.get("Scan-Eingang/Config/DEPOT Dateilog 28-08-2026 09-00-00.txt").decode("utf-8")
    assert "scan1.pdf" in text
    assert calls["n"] == 2


def test_append_gives_up_gracefully_without_raising(monkeypatch, client):
    import depot.depotlog as depotlog_module

    monkeypatch.setattr(depotlog_module, "_WRITE_RETRY_DELAY_SECONDS", 0.01)

    log = DepotLog(client, "Scan-Eingang", "DEPOT Dateilog", "Config")

    def always_locked(*args, **kwargs):
        raise RuntimeError("423 Locked")

    monkeypatch.setattr(client, "put", always_locked)

    # Must not raise, even though every write attempt fails - a document that
    # was already successfully filed must not be treated as a failure just
    # because the audit-log entry couldn't be written.
    log.append("scan1.pdf", "confidence=0.90", on_date=date(2026, 8, 28), on_time=time(9, 0, 0))
