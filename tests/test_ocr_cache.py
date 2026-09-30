import json
import time

from depot.models import OcrResult
from depot.ocr_cache import OcrCache


def _scan(tmp_path, name="scan.pdf"):
    path = tmp_path / name
    path.write_bytes(b"%PDF-raw")
    return path


def test_produced_pdf_moves_into_the_cache_and_comes_back(tmp_path):
    cache = OcrCache(tmp_path / "cache")
    scan = _scan(tmp_path)
    produced = tmp_path / "depot-ocr-out.pdf"
    produced.write_bytes(b"%PDF-with-text")

    stored = cache.put("k1", OcrResult(text="Hallo Welt", page_count=1, ocr_pdf_path=str(produced), ocr_failed=False), scan)

    assert not produced.exists()
    assert stored.ocr_pdf_path == str(tmp_path / "cache" / "k1.pdf")
    assert (tmp_path / "cache" / "k1.pdf").read_bytes() == b"%PDF-with-text"

    again = cache.get("k1", scan)
    assert again == stored

    cache.discard("k1")
    assert cache.get("k1", scan) is None
    assert not list((tmp_path / "cache").iterdir())


def test_result_that_archives_the_input_file_stores_only_the_text(tmp_path):
    cache = OcrCache(tmp_path / "cache")
    scan = _scan(tmp_path)
    result = OcrResult(text="digital", page_count=3, ocr_pdf_path=str(scan), ocr_failed=False, born_digital=True)

    stored = cache.put("k2", result, scan)

    assert stored == result
    assert scan.exists()  # the input file is never moved
    assert not (tmp_path / "cache" / "k2.pdf").exists()
    # the same content under a new name: the result points at the current file
    renamed = _scan(tmp_path, "renamed.pdf")
    hit = cache.get("k2", renamed)
    assert hit.ocr_pdf_path == str(renamed)
    assert hit.born_digital is True
    assert hit.text == "digital"


def test_failed_ocr_is_cached_too(tmp_path):
    cache = OcrCache(tmp_path / "cache")
    scan = _scan(tmp_path, "photo.jpg")
    cache.put("k3", OcrResult(text="", page_count=1, ocr_pdf_path=str(scan), ocr_failed=True), scan)

    assert cache.get("k3", scan).ocr_failed is True


def test_entry_whose_pdf_is_missing_is_a_miss(tmp_path):
    cache = OcrCache(tmp_path / "cache")
    scan = _scan(tmp_path)
    produced = tmp_path / "out.pdf"
    produced.write_bytes(b"%PDF")
    cache.put("k4", OcrResult(text="t", page_count=1, ocr_pdf_path=str(produced), ocr_failed=False), scan)
    (tmp_path / "cache" / "k4.pdf").unlink()

    assert cache.get("k4", scan) is None
    assert not (tmp_path / "cache" / "k4.json").exists()


def test_prune_drops_old_entries_and_stray_files(tmp_path):
    cache_dir = tmp_path / "cache"
    cache = OcrCache(cache_dir)
    scan = _scan(tmp_path)
    for key in ("old", "new"):
        produced = tmp_path / f"{key}.pdf"
        produced.write_bytes(b"%PDF")
        cache.put(key, OcrResult(text=key, page_count=1, ocr_pdf_path=str(produced), ocr_failed=False), scan)
    meta = cache_dir / "old.json"
    data = json.loads(meta.read_text(encoding="utf-8"))
    data["created"] = time.time() - 30 * 24 * 3600
    meta.write_text(json.dumps(data), encoding="utf-8")
    (cache_dir / "stray.pdf").write_bytes(b"%PDF")

    removed = cache.prune(max_age_seconds=7 * 24 * 3600)

    assert removed == 1
    assert cache.get("old", scan) is None
    assert cache.get("new", scan) is not None
    assert not (cache_dir / "stray.pdf").exists()
