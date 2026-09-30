"""Keeps the OCR result of a scan until that scan has been filed, so a
retry after a transient failure (Ollama or Nextcloud unreachable while
the document waited for classification or upload) or a container
restart mid-batch does not run OCR - the slowest stage by far - a second
time for the same file."""
from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

from depot.models import OcrResult

log = logging.getLogger(__name__)

# Entries this old belong to scans that never made it through processing
# (quarantined, removed by hand, ...) and are dropped at startup.
MAX_AGE_SECONDS = 7 * 24 * 3600


class OcrCache:
    """One entry per content hash: the recognized text plus, when OCR
    produced a new searchable PDF, that PDF. Results that reuse the input
    file as the file to archive (born-digital PDFs, files without any
    recognizable text) store only the text."""

    def __init__(self, directory: str | Path):
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _paths(self, key: str) -> tuple[Path, Path]:
        return self._dir / f"{key}.json", self._dir / f"{key}.pdf"

    def get(self, key: str, input_path: Path) -> OcrResult | None:
        """The cached result for `key`, with `input_path` as the file to
        archive where the original result used its own input file."""
        meta_path, pdf_path = self._paths(key)
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if meta.get("has_pdf"):
            if not pdf_path.is_file():
                self.discard(key)
                return None
            archive_path = pdf_path
        else:
            archive_path = input_path
        return OcrResult(
            text=meta["text"],
            page_count=meta["page_count"],
            ocr_pdf_path=str(archive_path),
            ocr_failed=meta["ocr_failed"],
            born_digital=meta.get("born_digital", False),
        )

    def put(self, key: str, result: OcrResult, input_path: Path) -> OcrResult:
        """Store `result`, moving a produced PDF into the cache. Returns the
        result as it should be used from now on (pointing at the moved
        file)."""
        meta_path, pdf_path = self._paths(key)
        produced = Path(result.ocr_pdf_path)
        has_pdf = produced != input_path
        if has_pdf:
            shutil.move(str(produced), str(pdf_path))
        meta = {
            "text": result.text,
            "page_count": result.page_count,
            "ocr_failed": result.ocr_failed,
            "born_digital": result.born_digital,
            "has_pdf": has_pdf,
            "source_name": input_path.name,
            "created": time.time(),
        }
        tmp = meta_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        tmp.replace(meta_path)
        return result.model_copy(update={"ocr_pdf_path": str(pdf_path if has_pdf else input_path)})

    def discard(self, key: str) -> None:
        for path in self._paths(key):
            path.unlink(missing_ok=True)

    def prune(self, max_age_seconds: float = MAX_AGE_SECONDS) -> int:
        """Drop entries older than `max_age_seconds` (and stray files
        without metadata). Returns how many entries were removed."""
        cutoff = time.time() - max_age_seconds
        removed = 0
        for meta_path in self._dir.glob("*.json"):
            try:
                created = json.loads(meta_path.read_text(encoding="utf-8")).get("created", 0)
            except (OSError, ValueError):
                created = 0
            if created < cutoff:
                self.discard(meta_path.stem)
                removed += 1
        for pdf_path in self._dir.glob("*.pdf"):
            if not pdf_path.with_suffix(".json").exists():
                pdf_path.unlink(missing_ok=True)
        if removed:
            log.info("Dropped %d stale OCR cache entries.", removed)
        return removed
