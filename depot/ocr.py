from __future__ import annotations

import io
import logging
import re
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

# What ocrmypdf writes into the sidecar file for a page it skipped because
# the page already had text ("[OCR skipped on page(s) 1-3]"). Never real
# document text.
_SKIPPED_MARKER = re.compile(r"\[OCR skipped on page\(s\) [^\]]*\]")


def _as_pdf(input_path: Path, work_dir: Path) -> Path:
    """Return a PDF path for the given input, wrapping loose images losslessly."""
    if input_path.suffix.lower() == ".pdf":
        return input_path
    pdf_path = work_dir / (input_path.stem + "__source.pdf")
    try:
        pdf_path.write_bytes(img2pdf.convert(str(input_path)))
    except Exception as exc:
        # img2pdf refuses images with an alpha channel (typical for PNG
        # screenshots). Flatten onto white and wrap that instead - only the
        # temporary OCR input is affected, never the archived original.
        log.info("img2pdf rejected %s (%s); retrying without alpha channel.", input_path.name, exc)
        from PIL import Image

        with Image.open(input_path) as img:
            rgba = img.convert("RGBA")
            flat = Image.new("RGB", rgba.size, "white")
            flat.paste(rgba, mask=rgba.getchannel("A"))
            buffer = io.BytesIO()
            flat.save(buffer, format="PNG")
        pdf_path.write_bytes(img2pdf.convert(buffer.getvalue()))
    return pdf_path


def _word_count(text: str) -> int:
    return len(text.split())


def _page_texts(pdf_path: Path) -> list[str] | None:
    """The existing text layer of each page, or None if the file can't be
    opened as a PDF at all."""
    pages = _inspect_pages(pdf_path)
    return None if pages is None else [text for text, _ in pages]


def _inspect_pages(pdf_path: Path) -> list[tuple[str, bool]] | None:
    """Per page: (existing text layer, whether the page contains an image).
    None if the file can't be opened as a PDF at all."""
    try:
        with fitz.open(pdf_path) as doc:
            return [(page.get_text(), bool(page.get_images())) for page in doc]
    except Exception:
        return None


def _is_born_digital(pages: list[tuple[str, bool]]) -> bool:
    """True if there is nothing on any page for OCR to find that the PDF's
    own text layer doesn't already provide: every page either has text, or
    has no image that could be hiding some (an intentionally blank page)."""
    words = [_word_count(text) for text, _ in pages]
    if any(not count and has_image for count, (_, has_image) in zip(words, pages)):
        return False
    pages_with_text = sum(1 for count in words if count)
    return pages_with_text > 0 and sum(words) >= MIN_WORDS_PER_PAGE * pages_with_text


def _run_ocrmypdf(src_pdf: Path, out_pdf: Path, sidecar: Path, language: str, force: bool) -> None:
    # No --clean (unpaper): measured with tools/ocr_bench.py on 8 real scans
    # it took a third of the OCR time and left the recognized text identical
    # in 6 of 7 documents (97 % identical in the seventh). --deskew stays:
    # it costs about as much, but it is what makes a crooked phone photo
    # readable, and the test scans were all straight.
    cmd = [
        "ocrmypdf",
        "--language", language,
        "--deskew",
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


def _read_result_text(out_pdf: Path, sidecar: Path, had_text_layer: bool) -> str:
    """The text of an ocrmypdf run. For an input that already had text on
    some pages, the sidecar only holds a "[OCR skipped ...]" marker for
    those pages instead of their text, so the text is read back from the
    produced PDF itself (existing layer + newly recognized pages alike)."""
    if had_text_layer and out_pdf.exists():
        pages = _page_texts(out_pdf)
        if pages is not None:
            return "\n".join(pages)
    text = sidecar.read_text(encoding="utf-8", errors="replace") if sidecar.exists() else ""
    return _SKIPPED_MARKER.sub("", text)


def process_file(input_path: Path, language: str = "deu") -> OcrResult:
    """Get the text of a single scan (PDF or image) plus the file to archive.

    - A PDF whose pages all already have real text (born-digital; blank
      pages without any image don't count against it) is not run through
      OCR at all: its own text is used and the original file is
      archived unchanged. Previously such files were rasterized and
      re-recognized, replacing perfect text with OCR guesses, multiplying
      the file size and taking minutes for long documents.
    - Everything else goes through ocrmypdf, producing a searchable PDF.
    - If no usable text comes out (a photo, a blank page), the original file
      is archived as-is rather than a deskewed/re-encoded PDF of it.
    """
    is_pdf = input_path.suffix.lower() == ".pdf"
    source_pages = _inspect_pages(input_path) if is_pdf else []

    if source_pages and _is_born_digital(source_pages):
        log.info("%s already has a complete text layer; skipping OCR.", input_path.name)
        return OcrResult(
            text="\n".join(text for text, _ in source_pages).strip(),
            page_count=len(source_pages),
            ocr_pdf_path=str(input_path),
            ocr_failed=False,
            born_digital=True,
        )

    # None = a PDF pymupdf couldn't read; assume it may have text so the
    # forced retry below stays available for it.
    had_text_layer = is_pdf and (
        source_pages is None or any(_word_count(text) for text, _ in source_pages)
    )

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

        text = _read_result_text(out_pdf, sidecar, had_text_layer)
        page_count = _safe_page_count(out_pdf if out_pdf.exists() else src_pdf)

        too_little_text = not out_pdf.exists() or _word_count(text) < MIN_WORDS_PER_PAGE * max(page_count, 1)
        if too_little_text and had_text_layer:
            # Retry with forced OCR: covers the case where an existing garbage
            # text layer caused --skip-text to skip real recognition. Pointless
            # for inputs without any text layer (images, plain scans) - there
            # the first pass already recognized every page.
            log.info("Retrying %s with --force-ocr", input_path.name)
            try:
                _run_ocrmypdf(src_pdf, out_pdf, sidecar, language, force=True)
                text = _read_result_text(out_pdf, sidecar, had_text_layer=False)
                page_count = _safe_page_count(out_pdf if out_pdf.exists() else src_pdf)
            except RuntimeError as exc:
                log.error("ocrmypdf forced pass failed for %s: %s", input_path.name, exc)

        ocr_failed = not out_pdf.exists() or _word_count(text) < MIN_WORDS_PER_PAGE * max(page_count, 1)

        if ocr_failed:
            persisted_pdf = input_path  # nothing worth keeping was produced
        else:
            # Persist the produced PDF outside the temp dir so callers can use
            # it after this function returns (the TemporaryDirectory is
            # cleaned up on exit).
            persisted_pdf = Path(tempfile.gettempdir()) / f"depot-ocr-out-{uuid.uuid4().hex}.pdf"
            persisted_pdf.write_bytes(out_pdf.read_bytes())

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
