from __future__ import annotations

import logging
import subprocess
import tempfile
import uuid
from pathlib import Path

import img2pdf
import pymupdf as fitz

from depot.models import OcrResult

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Below this average word count per page, treat OCR as having effectively failed
# (blank page, fully unreadable scan, camera pointed at the wrong thing, etc.).
MIN_WORDS_PER_PAGE = 5


def _as_pdf(input_path: Path, work_dir: Path) -> Path:
    """Return a PDF path for the given input, wrapping loose images losslessly."""
    if input_path.suffix.lower() == ".pdf":
        return input_path
    pdf_path = work_dir / (input_path.stem + "__source.pdf")
    pdf_path.write_bytes(img2pdf.convert(str(input_path)))
    return pdf_path


def _word_count(text: str) -> int:
    return len(text.split())


def _run_ocrmypdf(src_pdf: Path, out_pdf: Path, sidecar: Path, language: str, force: bool) -> None:
    cmd = [
        "ocrmypdf",
        "--language", language,
        "--deskew",
        "--clean",
        "--rotate-pages",
        "--sidecar", str(sidecar),
        "--output-type", "pdf",
    ]
    cmd.append("--force-ocr" if force else "--skip-text")
    cmd += [str(src_pdf), str(out_pdf)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    # ocrmypdf exit code 6 = "input file already has text, --skip-text produced
    # no new OCR" style soft-warnings on some versions; treat only a hard
    # failure (no output file at all) as fatal, everything else is inspected
    # via the resulting text.
    if result.returncode != 0 and not out_pdf.exists():
        raise RuntimeError(
            f"ocrmypdf failed (exit {result.returncode}): {result.stderr.strip()}"
        )


def process_file(input_path: Path, language: str = "deu") -> OcrResult:
    """Run OCR on a single scan (PDF or image), returning extracted text plus
    a new searchable PDF. Always produces a PDF as output, even for image
    inputs, so the archived document ends up text-searchable in Nextcloud.
    """
    with tempfile.TemporaryDirectory(prefix="depot-ocr-") as tmp:
        work_dir = Path(tmp)
        src_pdf = _as_pdf(input_path, work_dir)

        out_pdf = work_dir / "out.pdf"
        sidecar = work_dir / "out.txt"

        try:
            _run_ocrmypdf(src_pdf, out_pdf, sidecar, language, force=False)
        except RuntimeError as exc:
            log.warning("ocrmypdf first pass failed for %s: %s", input_path.name, exc)
            out_pdf.unlink(missing_ok=True)

        text = sidecar.read_text(encoding="utf-8", errors="replace") if sidecar.exists() else ""
        page_count = _safe_page_count(out_pdf if out_pdf.exists() else src_pdf)

        if not out_pdf.exists() or _word_count(text) < MIN_WORDS_PER_PAGE * max(page_count, 1):
            # Retry with forced OCR: covers the case where an existing garbage
            # text layer caused --skip-text to skip real recognition.
            log.info("Retrying %s with --force-ocr", input_path.name)
            try:
                _run_ocrmypdf(src_pdf, out_pdf, sidecar, language, force=True)
                text = sidecar.read_text(encoding="utf-8", errors="replace") if sidecar.exists() else ""
                page_count = _safe_page_count(out_pdf if out_pdf.exists() else src_pdf)
            except RuntimeError as exc:
                log.error("ocrmypdf forced pass failed for %s: %s", input_path.name, exc)

        ocr_failed = not out_pdf.exists() or _word_count(text) < MIN_WORDS_PER_PAGE * max(page_count, 1)

        # Persist the produced PDF outside the temp dir so callers can use it
        # after this function returns (the TemporaryDirectory is cleaned up
        # on exit).
        persisted_pdf = Path(tempfile.gettempdir()) / f"depot-ocr-out-{uuid.uuid4().hex}.pdf"
        if out_pdf.exists():
            persisted_pdf.write_bytes(out_pdf.read_bytes())
        else:
            persisted_pdf = input_path  # nothing usable was produced

        return OcrResult(
            text=text.strip(),
            page_count=page_count,
            ocr_pdf_path=str(persisted_pdf),
            ocr_failed=ocr_failed,
        )


def _safe_page_count(pdf_path: Path) -> int:
    try:
        with fitz.open(pdf_path) as doc:
            return doc.page_count
    except Exception:
        return 1
