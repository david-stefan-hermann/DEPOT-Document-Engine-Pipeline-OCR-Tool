"""Measure what each ocrmypdf option costs and what it buys, on real scans.

DEPOT's OCR flags (--deskew, --clean, --rotate-pages, language) were chosen
by expectation, not measurement. This runs a handful of flag variants over
sample scans and reports, per variant: wall time, size of the produced PDF,
number of recognized words and how much the recognized text differs from
the reference variant ("aktuell", the flags DEPOT uses today).

Every run uses --force-ocr, so a PDF that already has a text layer (e.g. one
DEPOT filed earlier) is treated like a fresh scan: the existing layer is
ignored and the page images are recognized anew. Loose images (jpg/png) are
wrapped into a PDF first, like the pipeline does.

Needs tesseract etc., so it runs where DEPOT runs:

    # on the server, inside the container (the image contains tools/)
    docker compose exec depot python tools/ocr_bench.py /nextcloud-data/Dokumente/<...>.pdf ...

    # or locally via the image, with a folder of sample scans mounted
    docker build -t depot-bench .
    docker run --rm -v <samples>:/samples -v <repo>/eval-results:/out depot-bench \\
        python tools/ocr_bench.py /samples --out /out/ocr-bench.json

Result: a table on stdout plus a JSON file with every single run.
Sample files never leave the machine; the JSON holds only file names,
durations and counts, no text.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depot import ocr  # noqa: E402

# name -> (extra ocrmypdf flags, language). The pipeline's own flags live in
# ocr._run_ocrmypdf; "aktuell" mirrors them.
VARIANTS: dict[str, tuple[list[str], str]] = {
    "aktuell": (["--deskew", "--rotate-pages"], "deu"),
    "mit-clean": (["--deskew", "--clean", "--rotate-pages"], "deu"),
    "ohne-deskew": (["--rotate-pages"], "deu"),
    "ohne-rotate": (["--deskew"], "deu"),
    "nur-ocr": ([], "deu"),
    "optimize0": (["--deskew", "--rotate-pages", "--optimize", "0"], "deu"),
    "deu+eng": (["--deskew", "--rotate-pages"], "deu+eng"),
    "jobs1": (["--deskew", "--rotate-pages", "--jobs", "1"], "deu"),
    "jobs4": (["--deskew", "--rotate-pages", "--jobs", "4"], "deu"),
}
REFERENCE = "aktuell"


def run_variant(src_pdf: Path, flags: list[str], language: str, work: Path) -> dict:
    out_pdf = work / "out.pdf"
    sidecar = work / "out.txt"
    cmd = [
        "ocrmypdf", "--language", language, *flags,
        "--force-ocr", "--sidecar", str(sidecar), "--output-type", "pdf",
        str(src_pdf), str(out_pdf),
    ]
    started = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    seconds = time.perf_counter() - started
    text = sidecar.read_text(encoding="utf-8", errors="replace") if sidecar.exists() else ""
    return {
        "seconds": round(seconds, 2),
        "exit": proc.returncode,
        "size_kb": out_pdf.stat().st_size // 1024 if out_pdf.exists() else None,
        "words": len(text.split()),
        "_text": text,
        "stderr_tail": proc.stderr.strip()[-300:] if proc.returncode else "",
    }


def word_overlap(a: str, b: str) -> float:
    """Share of the reference's distinct words (3+ letters) also recognized
    in the other text - 1.0 means nothing the reference found is missing."""
    ref = {w.strip(".,;:()").casefold() for w in a.split() if len(w.strip(".,;:()")) >= 3}
    other = {w.strip(".,;:()").casefold() for w in b.split() if len(w.strip(".,;:()")) >= 3}
    if not ref:
        return 1.0 if not other else 0.0  # nothing to find (a photo): equal if both found nothing
    return round(len(ref & other) / len(ref), 3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("samples", nargs="+", help="scan files, or directories of them")
    parser.add_argument("--variants", default=",".join(VARIANTS), help="comma-separated subset of: " + ", ".join(VARIANTS))
    parser.add_argument("--out", default="eval-results/ocr-bench.json")
    args = parser.parse_args()

    files: list[Path] = []
    for sample in args.samples:
        p = Path(sample)
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in {".pdf", *ocr.IMAGE_EXTENSIONS})
        else:
            files.append(p)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        parser.error(f"unknown variant(s): {', '.join(unknown)}")
    if REFERENCE not in variants:
        variants.insert(0, REFERENCE)

    results: list[dict] = []
    for f in files:
        with tempfile.TemporaryDirectory(prefix="ocr-bench-") as tmp:
            work = Path(tmp)
            src_pdf = ocr._as_pdf(f, work)
            texts: dict[str, str] = {}
            for name in variants:
                flags, language = VARIANTS[name]
                run_dir = work / name
                run_dir.mkdir()
                r = run_variant(src_pdf, flags, language, run_dir)
                texts[name] = r.pop("_text")
                r.update(file=f.name, variant=name)
                results.append(r)
                print(f"{f.name:28} {name:12} {r['seconds']:7.1f}s {str(r['size_kb']):>7} KB {r['words']:6} words"
                      + (f"  exit={r['exit']} {r['stderr_tail']}" if r["exit"] else ""), flush=True)
            for r in results:
                if r["file"] == f.name:
                    r["overlap_with_reference"] = word_overlap(texts[REFERENCE], texts[r["variant"]])

    # summary per variant: totals over all files, relative to the reference
    print()
    print(f"{'Variante':12} {'Zeit':>8} {'rel.':>6} {'Größe':>9} {'Wörter':>7} {'Textübereinstimmung':>20}")
    ref_time = sum(r["seconds"] for r in results if r["variant"] == REFERENCE) or 1.0
    for name in variants:
        rows = [r for r in results if r["variant"] == name]
        total = sum(r["seconds"] for r in rows)
        size = sum(r["size_kb"] or 0 for r in rows)
        words = sum(r["words"] for r in rows)
        overlap = sum(r["overlap_with_reference"] for r in rows) / len(rows) if rows else 0
        print(f"{name:12} {total:7.1f}s {total / ref_time:6.2f} {size:7} KB {words:7} {overlap:20.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(results)} Läufe -> {out}")


if __name__ == "__main__":
    main()
