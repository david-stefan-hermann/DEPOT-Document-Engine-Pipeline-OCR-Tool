import queue
import time
from pathlib import Path

import httpx
import pytest

from depot import classifier, ocr
from depot.config import Config
from depot.depotlog import DepotLog
from depot.models import ClassificationResult, OcrResult
from depot.pipeline import Pipeline
from depot.state import StateStore

BASE_URL = "https://nc.example.test/remote.php/dav/files/testuser"


def make_config(tmp_path, **overrides) -> Config:
    defaults = dict(
        nextcloud_webdav_url=BASE_URL,
        nextcloud_user="testuser",
        nextcloud_app_password="app-password",
        scan_eingang_local_path=str(tmp_path),
        scan_eingang_webdav_path="Scan-Eingang",
        dokumente_webdav_root="Dokumente",
        fallback_folder="Dokumente/Unsortiert",
        error_folder="Dokumente/_Fehlerhaft",
        ollama_host="http://fake:11434",
        ollama_model="fake-model",
        confidence_threshold=0.6,
        log_file_prefix="DEPOT Dateilog",
        ocr_language="deu",
        max_concurrent_jobs=1,
        state_db_path=str(tmp_path / "state.sqlite3"),
        supported_extensions=frozenset({".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}),
    )
    defaults.update(overrides)
    return Config(**defaults)


@pytest.fixture
def pipeline(tmp_path, fake_server, client):
    config = make_config(tmp_path)
    depot_log = DepotLog(client, config.scan_eingang_webdav_path, config.log_file_prefix)
    state = StateStore(config.state_db_path)
    p = Pipeline(config, webdav=client, depot_log=depot_log, state=state)
    yield p
    state.close()


def _make_scan(tmp_path, name="scan1.pdf", content=b"%PDF-raw-scan") -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path


def _make_ocr_pdf(tmp_path, name="ocr-out.pdf", content=b"%PDF-with-text-layer") -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path


def _seed_source_on_server(pipeline, client, scan_path: Path):
    client.mkcol(pipeline.config.scan_eingang_webdav_path)
    client.put(f"{pipeline.config.scan_eingang_webdav_path}/{scan_path.name}", scan_path.read_bytes())


def test_happy_path_files_into_existing_folder(monkeypatch, tmp_path, pipeline, fake_server, client):
    scan = _make_scan(tmp_path)
    _seed_source_on_server(pipeline, client, scan)
    ocr_pdf = _make_ocr_pdf(tmp_path)

    monkeypatch.setattr(
        ocr, "process_file",
        lambda path, language: OcrResult(
            text="Ihre Krankenkasse informiert Sie ueber...",
            page_count=1,
            ocr_pdf_path=str(ocr_pdf),
            ocr_failed=False,
        ),
    )
    monkeypatch.setattr(
        classifier, "classify",
        lambda **kwargs: (
            ClassificationResult(
                folder="Dokumente/Gesundheit/Krankenkasse",
                is_new_folder=False,
                title="Mitgliedsbescheinigung",
                issue_date=None,
                confidence=0.9,
            ),
            [],
        ),
    )
    client.mkcol("Dokumente/Gesundheit/Krankenkasse")

    pipeline.process_one(scan)

    files = {p for p in fake_server.files if p.startswith("Dokumente/Gesundheit/Krankenkasse/")}
    assert len(files) == 1
    assert client.get(f"Scan-Eingang/{scan.name}") is None  # source removed
    log_text = client.get("Scan-Eingang/DEPOT Dateilog " + time.strftime("%d-%m-%Y") + ".txt")
    assert log_text is not None
    assert "scan1.pdf" in log_text.decode("utf-8")


def test_low_confidence_goes_to_fallback(monkeypatch, tmp_path, pipeline, fake_server, client):
    scan = _make_scan(tmp_path)
    _seed_source_on_server(pipeline, client, scan)
    ocr_pdf = _make_ocr_pdf(tmp_path)

    monkeypatch.setattr(
        ocr, "process_file",
        lambda path, language: OcrResult(text="unklarer Text", page_count=1, ocr_pdf_path=str(ocr_pdf), ocr_failed=False),
    )
    monkeypatch.setattr(
        classifier, "classify",
        lambda **kwargs: (
            ClassificationResult(folder="Irgendwas", is_new_folder=True, title="Unklar", confidence=0.3),
            [],
        ),
    )

    pipeline.process_one(scan)

    fallback_files = {p for p in fake_server.files if p.startswith("Dokumente/Unsortiert/")}
    assert len(fallback_files) == 1


def test_ocr_failure_skips_classifier_and_goes_to_fallback(monkeypatch, tmp_path, pipeline, fake_server, client):
    scan = _make_scan(tmp_path)
    _seed_source_on_server(pipeline, client, scan)

    monkeypatch.setattr(
        ocr, "process_file",
        lambda path, language: OcrResult(text="", page_count=1, ocr_pdf_path=str(path), ocr_failed=True),
    )

    def _should_not_be_called(**kwargs):
        raise AssertionError("classifier.classify must not be called when OCR failed")

    monkeypatch.setattr(classifier, "classify", _should_not_be_called)

    pipeline.process_one(scan)

    fallback_files = {p for p in fake_server.files if p.startswith("Dokumente/Unsortiert/")}
    assert len(fallback_files) == 1
    # OCR totally failed -> original bytes/extension preserved, not forced to .pdf
    assert list(fallback_files)[0].endswith(".pdf")  # scan.pdf input, so still .pdf here


def test_new_folder_is_created_and_tagged(monkeypatch, tmp_path, pipeline, fake_server, client):
    scan = _make_scan(tmp_path)
    _seed_source_on_server(pipeline, client, scan)
    ocr_pdf = _make_ocr_pdf(tmp_path)

    monkeypatch.setattr(
        ocr, "process_file",
        lambda path, language: OcrResult(text="Rechnung fuer Ersatzteile...", page_count=1, ocr_pdf_path=str(ocr_pdf), ocr_failed=False),
    )
    monkeypatch.setattr(
        classifier, "classify",
        lambda **kwargs: (
            ClassificationResult(folder="Dokumente/Motorrad/Ersatzteile", is_new_folder=True, title="Ersatzteilrechnung", confidence=0.85),
            [],
        ),
    )

    pipeline.process_one(scan)

    assert "Dokumente/Motorrad/Ersatzteile" in fake_server.collections
    log_text = client.get("Scan-Eingang/DEPOT Dateilog " + time.strftime("%d-%m-%Y") + ".txt").decode("utf-8")
    assert "NEUER-ORDNER" in log_text


def test_filename_collision_gets_suffix(monkeypatch, tmp_path, pipeline, fake_server, client):
    scan = _make_scan(tmp_path)
    _seed_source_on_server(pipeline, client, scan)
    ocr_pdf = _make_ocr_pdf(tmp_path)

    from datetime import date
    monkeypatch.setattr(
        ocr, "process_file",
        lambda path, language: OcrResult(text="Stromrechnung Juli", page_count=1, ocr_pdf_path=str(ocr_pdf), ocr_failed=False),
    )
    monkeypatch.setattr(
        classifier, "classify",
        lambda **kwargs: (
            ClassificationResult(folder="Dokumente/Energie/Rechnungen", is_new_folder=False, title="Stromrechnung", issue_date=date(2026, 7, 15), confidence=0.9),
            [],
        ),
    )
    client.mkcol("Dokumente/Energie/Rechnungen")
    client.put("Dokumente/Energie/Rechnungen/2026-07-15 Stromrechnung.pdf", b"already-there")

    pipeline.process_one(scan)

    names = {p.rsplit("/", 1)[-1] for p in fake_server.files if p.startswith("Dokumente/Energie/Rechnungen/")}
    assert "2026-07-15 Stromrechnung.pdf" in names
    assert "2026-07-15 Stromrechnung (2).pdf" in names


def test_permanent_failure_quarantines_after_max_attempts(monkeypatch, tmp_path, pipeline, fake_server, client):
    scan = _make_scan(tmp_path)
    _seed_source_on_server(pipeline, client, scan)

    def _boom(path, language):
        raise ValueError("corrupt file, cannot even open it")

    monkeypatch.setattr(ocr, "process_file", _boom)

    for _ in range(3):
        pipeline.process_one(scan)

    quarantined = {p for p in fake_server.files if p.startswith("Dokumente/_Fehlerhaft/")}
    assert len(quarantined) == 1
    assert client.get(f"Scan-Eingang/{scan.name}") is None


def test_transient_failure_is_requeued_not_quarantined(monkeypatch, tmp_path, pipeline, fake_server, client):
    import depot.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "_TRANSIENT_RETRY_DELAY_SECONDS", 0.05)

    scan = _make_scan(tmp_path)
    _seed_source_on_server(pipeline, client, scan)

    def _unreachable(path, language):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(ocr, "process_file", _unreachable)

    q: "queue.Queue" = queue.Queue()
    pipeline.process_one(scan, requeue=q)

    requeued = q.get(timeout=1.0)
    assert requeued == scan
    # must NOT count toward the permanent-failure quarantine limit
    assert pipeline.state.should_quarantine(scan.name) is False
