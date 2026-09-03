import queue
import time
from pathlib import Path

import httpx
import pytest

from depot import classifier, ocr
from depot.classifier import ClassificationOutcome
from depot.config import Config
from depot.depotlog import DepotLog
from depot.models import ContentExtraction, OcrResult
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
        anthropic_api_key=None,
        anthropic_model="claude-haiku-4-5",
        log_file_prefix="DEPOT Dateilog",
        config_file_name="DEPOT Config.json",
        config_subfolder="Config",
        processed_subfolder="Processed",
        ocr_language="deu",
        max_concurrent_jobs=1,
        state_db_path=str(tmp_path / "state.sqlite3"),
        supported_extensions=frozenset({".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}),
    )
    defaults.update(overrides)
    return Config(**defaults)


def _make_pipeline(tmp_path, client, **config_overrides) -> Pipeline:
    config = make_config(tmp_path, **config_overrides)
    depot_log = DepotLog(
        client, config.scan_eingang_webdav_path, config.log_file_prefix, config.config_subfolder
    )
    state = StateStore(config.state_db_path)
    return Pipeline(config, webdav=client, depot_log=depot_log, state=state)


@pytest.fixture
def pipeline(tmp_path, fake_server, client):
    p = _make_pipeline(tmp_path, client)
    yield p
    p.state.close()


def _make_scan(tmp_path, name="scan1.pdf", content=b"%PDF-raw-scan") -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path


def _make_ocr_pdf(tmp_path, name="ocr-out.pdf", content=b"%PDF-with-text-layer") -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path


def _get_log_text(fake_server, client) -> str:
    """The pipeline now writes one log file per processed event (timestamped
    to the second) into Scan-Eingang/Config/, instead of one shared daily
    file - fetch whichever single log file a test's single process_one()
    call produced."""
    log_paths = [p for p in fake_server.files if p.startswith("Scan-Eingang/Config/DEPOT Dateilog ")]
    assert len(log_paths) == 1, f"expected exactly one log file, found {log_paths}"
    return client.get(log_paths[0]).decode("utf-8")


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
            ClassificationOutcome(
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
    assert "scan1.pdf" in _get_log_text(fake_server, client)


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
            ClassificationOutcome(folder="Irgendwas", is_new_folder=True, title="Unklar", confidence=0.3),
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
            ClassificationOutcome(folder="Dokumente/Motorrad/Ersatzteile", is_new_folder=True, title="Ersatzteilrechnung", confidence=0.85),
            [],
        ),
    )

    pipeline.process_one(scan)

    assert "Dokumente/Motorrad/Ersatzteile" in fake_server.collections
    assert "NEUER-ORDNER" in _get_log_text(fake_server, client)


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
            ClassificationOutcome(folder="Dokumente/Energie/Rechnungen", is_new_folder=False, title="Stromrechnung", issue_date=date(2026, 7, 15), confidence=0.9),
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


def test_folder_cache_avoids_repeated_webdav_calls(pipeline, client):
    client.mkcol("Dokumente/Gesundheit")

    call_count = 0
    real_list = client.list_folders_recursive

    def counting_list(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return real_list(*args, **kwargs)

    pipeline.webdav.list_folders_recursive = counting_list

    first = pipeline._get_existing_folders()
    second = pipeline._get_existing_folders()

    assert call_count == 1
    assert first == second


def test_folder_cache_refreshes_after_ttl(monkeypatch, pipeline, client):
    import depot.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "FOLDER_CACHE_TTL_SECONDS", 0.01)
    client.mkcol("Dokumente/Gesundheit")

    call_count = 0
    real_list = client.list_folders_recursive

    def counting_list(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return real_list(*args, **kwargs)

    pipeline.webdav.list_folders_recursive = counting_list

    pipeline._get_existing_folders()
    time.sleep(0.05)
    pipeline._get_existing_folders()

    assert call_count == 2


def test_remember_folder_makes_new_folder_visible_before_ttl(pipeline, client):
    pipeline._get_existing_folders()  # populate cache
    pipeline._remember_folder("Dokumente/Versicherung/KFZ")

    assert "Dokumente/Versicherung/KFZ" in pipeline._get_existing_folders()
    # the real folder must not have been created on the server by this alone
    assert "Dokumente/Versicherung/KFZ" not in client.list_folders_recursive("Dokumente")


def test_new_folder_is_immediately_visible_to_next_document(monkeypatch, tmp_path, pipeline, fake_server, client):
    from datetime import date

    monkeypatch.setattr(
        ocr, "process_file",
        lambda path, language: OcrResult(text="text", page_count=1, ocr_pdf_path=str(path), ocr_failed=False),
    )

    calls = []

    def fake_classify(**kwargs):
        calls.append(kwargs["existing_folders"])
        return (
            ClassificationOutcome(
                folder="Dokumente/Versicherung/KFZ",
                is_new_folder=True,
                title="Kfz-Versicherung",
                issue_date=date(2026, 1, 1),
                confidence=0.9,
            ),
            [],
        )

    monkeypatch.setattr(classifier, "classify", fake_classify)

    scan1 = _make_scan(tmp_path, name="scan1.pdf")
    scan2 = _make_scan(tmp_path, name="scan2.pdf")
    _seed_source_on_server(pipeline, client, scan1)
    _seed_source_on_server(pipeline, client, scan2)

    pipeline.process_one(scan1)
    pipeline.process_one(scan2)

    assert "Dokumente/Versicherung/KFZ" not in calls[0]
    assert "Dokumente/Versicherung/KFZ" in calls[1]


def test_excluded_folders_from_config_file_are_never_offered(tmp_path, pipeline, client):
    import json

    client.mkcol("Dokumente/Games/Amiibo-main")
    client.mkcol("Dokumente/Gesundheit")
    config_dir = tmp_path / "Config"
    config_dir.mkdir()
    (config_dir / "DEPOT Config.json").write_text(
        json.dumps({"excluded_folders": ["Dokumente/Games"]}), encoding="utf-8"
    )

    folders = pipeline._get_existing_folders()

    assert "Dokumente/Gesundheit" in folders
    assert not any(f.startswith("Dokumente/Games") for f in folders)


def test_scan_eingang_is_always_excluded_even_when_nested_under_dokumente(tmp_path, client):
    """If Scan-Eingang lives inside Dokumente/, it must never be offered as
    a filing target - otherwise the classifier could file a document back
    into the folder the watcher watches, reprocessing it in a loop. This
    exclusion is structural, not dependent on the user's own
    excluded_folders config."""
    client.mkcol("Dokumente/Scan Eingang")
    client.mkcol("Dokumente/Scan Eingang/Depot Config")
    client.mkcol("Dokumente/Gesundheit")

    p = _make_pipeline(tmp_path, client, scan_eingang_webdav_path="Dokumente/Scan Eingang")

    folders = p._get_existing_folders()

    assert "Dokumente/Gesundheit" in folders
    assert not any(f.startswith("Dokumente/Scan Eingang") for f in folders)
    p.state.close()


def test_fallback_folder_is_never_offered_as_classification_target(pipeline, client):
    """Seen in production: the model choosing Unsortiert deliberately (high
    confidence, no tags) instead of it being reached only via the
    confidence-threshold fallback - defeats its purpose as a "needs review"
    bucket. fallback_folder must be structurally excluded, like Scan-Eingang."""
    client.mkcol("Dokumente/Unsortiert")
    client.mkcol("Dokumente/Gesundheit")

    folders = pipeline._get_existing_folders()

    assert "Dokumente/Gesundheit" in folders
    assert "Dokumente/Unsortiert" not in folders


# ---- file_into_dokumente / save_processed_copy switches (DEPOT Config.json) ---

def _write_processing_switches(tmp_path, config_subfolder="Config", **switches):
    import json
    config_dir = tmp_path / config_subfolder
    config_dir.mkdir(exist_ok=True)
    (config_dir / "DEPOT Config.json").write_text(json.dumps(switches), encoding="utf-8")


def test_filing_disabled_skips_folder_walk_and_only_extracts_content(monkeypatch, tmp_path, fake_server, client):
    from datetime import date
    scan = _make_scan(tmp_path)
    _write_processing_switches(tmp_path, file_into_dokumente=False, save_processed_copy=True)
    p = _make_pipeline(tmp_path, client)
    _seed_source_on_server(p, client, scan)
    ocr_pdf = _make_ocr_pdf(tmp_path)

    monkeypatch.setattr(
        ocr, "process_file",
        lambda path, language: OcrResult(text="Stromrechnung Juli", page_count=1, ocr_pdf_path=str(ocr_pdf), ocr_failed=False),
    )

    def _classify_should_not_be_called(**kwargs):
        raise AssertionError("classifier.classify must not be called when filing is disabled")

    monkeypatch.setattr(classifier, "classify", _classify_should_not_be_called)
    monkeypatch.setattr(
        classifier, "extract_content",
        lambda *a, **k: ContentExtraction(
            title="Stromrechnung Juli", correspondent="Stadtwerke", issue_date=date(2026, 7, 15), confidence=0.9
        ),
    )

    p.process_one(scan)

    assert not any(f.startswith("Dokumente/") for f in fake_server.files)
    processed = [f for f in fake_server.files if f.startswith("Scan-Eingang/Config/Processed/")]
    assert len(processed) == 1
    assert "Stadtwerke" in processed[0]
    assert client.get(f"Scan-Eingang/{scan.name}") is None  # source still removed
    p.state.close()


def test_save_processed_copy_alongside_normal_filing(monkeypatch, tmp_path, fake_server, client):
    _write_processing_switches(tmp_path, save_processed_copy=True)
    p = _make_pipeline(tmp_path, client)
    scan = _make_scan(tmp_path)
    _seed_source_on_server(p, client, scan)
    ocr_pdf = _make_ocr_pdf(tmp_path)

    monkeypatch.setattr(
        ocr, "process_file",
        lambda path, language: OcrResult(text="text", page_count=1, ocr_pdf_path=str(ocr_pdf), ocr_failed=False),
    )
    monkeypatch.setattr(
        classifier, "classify",
        lambda **kwargs: (
            ClassificationOutcome(
                folder="Dokumente/Gesundheit", is_new_folder=False, title="Rezept", confidence=0.9
            ),
            [],
        ),
    )

    p.process_one(scan)

    filed = [f for f in fake_server.files if f.startswith("Dokumente/Gesundheit/")]
    processed = [f for f in fake_server.files if f.startswith("Scan-Eingang/Config/Processed/")]
    assert len(filed) == 1
    assert len(processed) == 1
    p.state.close()


def test_switch_takes_effect_on_next_document_without_restart(monkeypatch, tmp_path, fake_server, client):
    """The whole point of reading DEPOT Config.json fresh per document
    instead of once at startup: editing it in Nextcloud mid-batch changes
    behavior for the very next scan, no container restart needed."""
    p = _make_pipeline(tmp_path, client)
    scan1 = _make_scan(tmp_path, name="scan1.pdf")
    scan2 = _make_scan(tmp_path, name="scan2.pdf")
    _seed_source_on_server(p, client, scan1)
    _seed_source_on_server(p, client, scan2)
    ocr_pdf = _make_ocr_pdf(tmp_path)

    monkeypatch.setattr(
        ocr, "process_file",
        lambda path, language: OcrResult(text="text", page_count=1, ocr_pdf_path=str(ocr_pdf), ocr_failed=False),
    )
    monkeypatch.setattr(
        classifier, "classify",
        lambda **kwargs: (
            ClassificationOutcome(folder="Dokumente/Gesundheit", is_new_folder=False, title="Doc", confidence=0.9),
            [],
        ),
    )
    monkeypatch.setattr(
        classifier, "extract_content",
        lambda *a, **k: ContentExtraction(title="Doc", correspondent="", confidence=0.9),
    )

    p.process_one(scan1)  # no DEPOT Config.json yet -> default: filing on
    _make_ocr_pdf(tmp_path)  # scan1's processing deleted the (non-raw) OCR output file
    _write_processing_switches(tmp_path, file_into_dokumente=False, save_processed_copy=True)
    p.process_one(scan2)  # same running pipeline, switch flipped between calls

    filed = [f for f in fake_server.files if f.startswith("Dokumente/Gesundheit/")]
    processed = [f for f in fake_server.files if f.startswith("Scan-Eingang/Config/Processed/")]
    assert len(filed) == 1  # only scan1
    assert len(processed) == 1  # only scan2


def test_use_anthropic_classifier_switch_routes_to_cloud_folder_decision(monkeypatch, tmp_path, fake_server, client):
    """When the switch is on, the pipeline must call classify_via_anthropic
    (cloud folder decision) instead of classify (local walk) - not both."""
    _write_processing_switches(tmp_path, use_anthropic_classifier=True)
    p = _make_pipeline(tmp_path, client, anthropic_api_key="sk-ant-fake")
    scan = _make_scan(tmp_path)
    _seed_source_on_server(p, client, scan)
    ocr_pdf = _make_ocr_pdf(tmp_path)

    monkeypatch.setattr(
        ocr, "process_file",
        lambda path, language: OcrResult(text="text", page_count=1, ocr_pdf_path=str(ocr_pdf), ocr_failed=False),
    )

    def _classify_should_not_be_called(**kwargs):
        raise AssertionError("classify (local walk) must not be called when use_anthropic_classifier is on")

    monkeypatch.setattr(classifier, "classify", _classify_should_not_be_called)

    via_anthropic_calls = []

    def fake_classify_via_anthropic(**kwargs):
        via_anthropic_calls.append(kwargs)
        return (
            ClassificationOutcome(
                folder="Dokumente/Gesundheit", is_new_folder=False, title="Rezept", confidence=0.9
            ),
            [],
        )

    monkeypatch.setattr(classifier, "classify_via_anthropic", fake_classify_via_anthropic)

    p.process_one(scan)

    assert len(via_anthropic_calls) == 1
    assert via_anthropic_calls[0]["anthropic_api_key"] == "sk-ant-fake"
    filed = [f for f in fake_server.files if f.startswith("Dokumente/Gesundheit/")]
    assert len(filed) == 1
    p.state.close()
    p.state.close()
