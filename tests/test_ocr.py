"""ocr.process_file without a real Tesseract: _run_ocrmypdf is replaced by a
stand-in that behaves like ocrmypdf does for the case under test."""
import shutil
from pathlib import Path

import pymupdf as fitz

from depot import ocr

LETTER = "Sehr geehrter Herr Mustermann, hiermit bestaetigen wir Ihnen den Eingang Ihres Antrags vom 14.08.2026."
SCANNED = "<scanned page>"  # a page that is only an image, no text layer


def _make_pdf(path: Path, page_texts: list[str]) -> Path:
    doc = fitz.open()
    for text in page_texts:
        page = doc.new_page()
        if text == SCANNED:
            pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 40, 40), False)
            pix.clear_with(200)
            page.insert_image(page.rect, pixmap=pix)
        elif text:
            page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()
    return path


def _make_png(path: Path, alpha: bool = False) -> Path:
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 40, 40), alpha)
    pix.clear_with(255)
    pix.save(path)
    return path


class FakeOcrmypdf:
    """Records every call; produces the input as output plus a sidecar."""

    def __init__(self, sidecar_text: str = "", forced_sidecar_text: str | None = None):
        self.calls: list[bool] = []
        self._sidecar_text = sidecar_text
        self._forced_sidecar_text = forced_sidecar_text

    def __call__(self, src_pdf, out_pdf, sidecar, language, force):
        self.calls.append(force)
        shutil.copyfile(src_pdf, out_pdf)
        text = self._forced_sidecar_text if force and self._forced_sidecar_text is not None else self._sidecar_text
        sidecar.write_text(text, encoding="utf-8")


def _must_not_run(*a, **k):
    raise AssertionError("ocrmypdf must not be run for this input")


def test_born_digital_pdf_skips_ocr_and_keeps_the_original(monkeypatch, tmp_path):
    """Real case: multi-page born-digital PDFs were rasterized via
    --force-ocr (a 130-page PDF became 169 MB and took ~10 minutes)."""
    pdf = _make_pdf(tmp_path / "brief.pdf", [LETTER, LETTER, LETTER])
    monkeypatch.setattr(ocr, "_run_ocrmypdf", _must_not_run)

    result = ocr.process_file(pdf)

    assert result.ocr_failed is False
    assert result.born_digital is True
    assert result.ocr_pdf_path == str(pdf)  # archived unchanged
    assert result.page_count == 3
    assert "bestaetigen" in result.text


def test_single_page_born_digital_pdf_yields_its_real_text(monkeypatch, tmp_path):
    """Real case: for a one-page PDF with a text layer the model was given
    ocrmypdf's "[OCR skipped on page(s) 1]" placeholder as document text."""
    pdf = _make_pdf(tmp_path / "brief.pdf", [LETTER])
    monkeypatch.setattr(ocr, "_run_ocrmypdf", _must_not_run)

    result = ocr.process_file(pdf)

    assert "OCR skipped" not in result.text
    assert "14.08.2026" in result.text


def test_blank_pages_do_not_stop_a_pdf_from_counting_as_born_digital(monkeypatch, tmp_path):
    """A letter with an empty back page has nothing for OCR to find."""
    pdf = _make_pdf(tmp_path / "brief.pdf", [LETTER, "", LETTER])
    monkeypatch.setattr(ocr, "_run_ocrmypdf", _must_not_run)

    result = ocr.process_file(pdf)

    assert result.born_digital is True
    assert result.ocr_pdf_path == str(pdf)


def test_scanned_pdf_without_text_layer_uses_the_sidecar(monkeypatch, tmp_path):
    pdf = _make_pdf(tmp_path / "scan.pdf", [SCANNED])
    fake = FakeOcrmypdf(sidecar_text="Erkannter Text aus der Texterkennung mit genug Woertern")
    monkeypatch.setattr(ocr, "_run_ocrmypdf", fake)

    result = ocr.process_file(pdf)

    assert fake.calls == [False]
    assert result.ocr_failed is False
    assert result.born_digital is False
    assert result.text.startswith("Erkannter Text")
    assert result.ocr_pdf_path != str(pdf)
    Path(result.ocr_pdf_path).unlink()


def test_mixed_pdf_reads_text_from_the_output_not_the_sidecar_marker(monkeypatch, tmp_path):
    """One page with a text layer, one without: ocrmypdf skips the first and
    only writes a marker for it into the sidecar. The real text must come
    from the produced PDF, and nothing may be rasterized via --force-ocr."""
    pdf = _make_pdf(tmp_path / "mixed.pdf", [LETTER, SCANNED])
    fake = FakeOcrmypdf(sidecar_text="[OCR skipped on page(s) 1]\f")
    monkeypatch.setattr(ocr, "_run_ocrmypdf", fake)

    result = ocr.process_file(pdf)

    assert fake.calls == [False]
    assert result.ocr_failed is False
    assert "bestaetigen" in result.text
    assert "OCR skipped" not in result.text
    Path(result.ocr_pdf_path).unlink()


def test_garbage_text_layer_still_gets_the_forced_retry(monkeypatch, tmp_path):
    pdf = _make_pdf(tmp_path / "scan-with-junk-layer.pdf", ["x", "y"])
    fake = FakeOcrmypdf(
        sidecar_text="[OCR skipped on page(s) 1-2]",
        forced_sidecar_text="Jetzt wirklich erkannter Text der beiden Seiten mit ausreichend vielen Woertern",
    )
    monkeypatch.setattr(ocr, "_run_ocrmypdf", fake)

    result = ocr.process_file(pdf)

    assert fake.calls == [False, True]
    assert result.ocr_failed is False
    assert result.text.startswith("Jetzt wirklich")
    Path(result.ocr_pdf_path).unlink()


def test_image_without_text_runs_ocr_once_and_keeps_the_original_image(monkeypatch, tmp_path):
    """A photo: no second (identical) OCR pass, and the original image is
    archived rather than a deskewed PDF of it."""
    image = _make_png(tmp_path / "foto.png")
    fake = FakeOcrmypdf(sidecar_text="")
    monkeypatch.setattr(ocr, "_run_ocrmypdf", fake)

    result = ocr.process_file(image)

    assert fake.calls == [False]
    assert result.ocr_failed is True
    assert result.ocr_pdf_path == str(image)


def test_image_with_alpha_channel_is_still_ocred(monkeypatch, tmp_path):
    image = _make_png(tmp_path / "screenshot.png", alpha=True)
    fake = FakeOcrmypdf(sidecar_text="Text eines Bildschirmfotos mit einigen lesbaren Woertern darin")
    monkeypatch.setattr(ocr, "_run_ocrmypdf", fake)

    result = ocr.process_file(image)

    assert fake.calls == [False]
    assert result.ocr_failed is False
    Path(result.ocr_pdf_path).unlink()


def test_unreadable_pdf_falls_back_to_the_original(monkeypatch, tmp_path):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-not-really")

    def failing(src_pdf, out_pdf, sidecar, language, force):
        raise RuntimeError("ocrmypdf failed (exit 2)")

    monkeypatch.setattr(ocr, "_run_ocrmypdf", failing)

    result = ocr.process_file(broken)

    assert result.ocr_failed is True
    assert result.ocr_pdf_path == str(broken)
