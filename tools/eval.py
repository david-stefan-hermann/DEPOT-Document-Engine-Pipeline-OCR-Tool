"""Measures how well the classifier files documents, using an existing,
hand-sorted Dokumente/ tree as ground truth: every PDF in it that has a text
layer is a test case whose correct folder is known (the one it is in).

    py tools/eval.py --tree "C:/Users/me/Nextcloud/Dokumente" --limit 40

Runs the LOCAL classifier only (Ollama); document text never leaves the
machines it would reach in normal operation. Results (which contain real
document names) are written to eval-results/, which is gitignored - never
commit them, this repository is public.

Not part of the container image; a development tool.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import pymupdf as fitz

MIN_WORDS = 40
MAX_PAGES_READ = 5


def _read_text(pdf: Path) -> str:
    try:
        with fitz.open(pdf) as doc:
            return "\n".join(doc[i].get_text() for i in range(min(doc.page_count, MAX_PAGES_READ)))
    except Exception:
        return ""


def _excluded_prefixes(tree: Path, root: str, extra: list[str]) -> list[str]:
    excluded = [f"{root}/Scan Eingang", f"{root}/Unsortiert", *extra]
    config_file = tree / "Scan Eingang" / "Depot Config" / "DEPOT Config.json"
    if config_file.is_file():
        try:
            excluded += json.loads(config_file.read_text(encoding="utf-8")).get("excluded_folders", [])
        except (OSError, json.JSONDecodeError):
            pass
    return [e.strip("/") for e in excluded]


def _is_excluded(rel: str, excluded: list[str]) -> bool:
    return any(rel == e or rel.startswith(f"{e}/") for e in excluded)


def _verdict(truth: str, predicted: str, confident: bool) -> str:
    if not confident:
        return "unsortiert"
    if predicted == truth:
        return "exakt"
    if predicted.startswith(f"{truth}/") or truth.startswith(f"{predicted}/"):
        deeper, shallower = (predicted, truth) if len(predicted) > len(truth) else (truth, predicted)
        if deeper[len(shallower) + 1:].count("/") == 0:
            return "eine-ebene-daneben"
    return "falsch"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tree", required=True, type=Path, help="local path of the hand-sorted Dokumente folder")
    parser.add_argument("--limit", type=int, default=40, help="number of documents to sample")
    parser.add_argument("--seed", type=int, default=1, help="sampling seed (same seed = same documents)")
    parser.add_argument("--keep-filenames", action="store_true",
                        help="pass the real filename to the classifier instead of a neutral 'scan.pdf'")
    parser.add_argument("--exclude", action="append", default=[], help="additional folder prefix to exclude")
    parser.add_argument("--threshold", type=float, default=float(os.environ.get("CONFIDENCE_THRESHOLD", "0.6")))
    parser.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M"))
    parser.add_argument("--depot-path", type=Path, default=Path(__file__).resolve().parent.parent,
                        help="checkout whose depot/ package is evaluated (for comparing two versions)")
    parser.add_argument("--label", default="", help="name for this run, used in the output filename")
    args = parser.parse_args()

    sys.path.insert(0, str(args.depot_path))
    from depot import classifier  # noqa: E402  (imported from the chosen checkout)

    tree: Path = args.tree.resolve()
    root = tree.name
    excluded = _excluded_prefixes(tree, root, args.exclude)

    folders: list[str] = []
    pdfs: list[tuple[Path, str]] = []
    for current, dirnames, filenames in os.walk(tree):
        rel = Path(current).relative_to(tree).as_posix()
        here = root if rel == "." else f"{root}/{rel}"
        if _is_excluded(here, excluded):
            dirnames[:] = []
            continue
        if here != root:
            folders.append(here)
        pdfs += [(Path(current) / f, here) for f in sorted(filenames) if f.lower().endswith(".pdf")]

    random.Random(args.seed).shuffle(pdfs)
    cases: list[tuple[Path, str, str]] = []
    for pdf, truth in pdfs:
        if len(cases) >= args.limit:
            break
        text = _read_text(pdf)
        if len(text.split()) >= MIN_WORDS and truth != root:
            cases.append((pdf, truth, text))

    print(f"{len(folders)} candidate folders, {len(cases)} test documents "
          f"(seed {args.seed}, {'real' if args.keep_filenames else 'neutral'} filenames), depot from {args.depot_path}")

    rows = []
    for i, (pdf, truth, text) in enumerate(cases, 1):
        started = time.perf_counter()
        try:
            outcome, tags = classifier.classify(
                ocr_text=text,
                original_filename=pdf.name if args.keep_filenames else "scan.pdf",
                existing_folders=folders,
                ollama_host=args.ollama_host,
                model=args.model,
                dokumente_root=root,
            )
            predicted, confidence = outcome.folder, outcome.confidence
            title, correspondent = outcome.title, outcome.correspondent or ""
            verdict = _verdict(truth, predicted, confidence >= args.threshold)
        except Exception as exc:
            predicted, confidence, title, correspondent, tags = f"ERROR: {exc}", 0.0, "", "", []
            verdict = "fehler"
        seconds = time.perf_counter() - started
        rows.append(dict(file=pdf.name, truth=truth, predicted=predicted, confidence=f"{confidence:.2f}",
                         verdict=verdict, top_level_ok=predicted.split("/")[1:2] == truth.split("/")[1:2],
                         seconds=f"{seconds:.1f}", title=title, correspondent=correspondent, tags="; ".join(tags)))
        print(f"[{i:3d}/{len(cases)}] {verdict:18s} {seconds:5.1f}s  {truth}  ->  {predicted}  ({confidence:.2f})")

    if not rows:
        print("No usable test documents found.")
        return

    n = len(rows)
    count = lambda v: sum(r["verdict"] == v for r in rows)  # noqa: E731
    filed = [r for r in rows if r["verdict"] not in ("unsortiert", "fehler")]
    print("\n=== Ergebnis ===")
    print(f"exakter Ordner:            {count('exakt'):3d} / {n}  ({count('exakt') / n:.0%})")
    print(f"eine Ebene daneben:        {count('eine-ebene-daneben'):3d} / {n}")
    print(f"richtiger Top-Level:       {sum(r['top_level_ok'] for r in filed):3d} / {n}")
    print(f"Unsortiert (unsicher):     {count('unsortiert'):3d} / {n}")
    print(f"FALSCH bei hoher Konfidenz:{count('falsch'):3d} / {n}  ({count('falsch') / n:.0%})")
    print(f"Fehler:                    {count('fehler'):3d} / {n}")
    print(f"Sekunden pro Dokument:     {sum(float(r['seconds']) for r in rows) / n:.1f}")

    out_dir = Path(__file__).resolve().parent.parent / "eval-results"
    out_dir.mkdir(exist_ok=True)
    label = f"-{args.label}" if args.label else ""
    out_file = out_dir / f"eval-{datetime.now():%Y%m%d-%H%M%S}{label}.csv"
    with out_file.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nDetails: {out_file}")


if __name__ == "__main__":
    main()
