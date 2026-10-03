"""Measures how well the classifier files documents, using an existing,
hand-sorted Dokumente/ tree as ground truth: every PDF in it that has a text
layer is a test case whose correct folder is known (the one it is in).

    py tools/eval.py --tree "C:/Users/me/Nextcloud/Dokumente" --limit 100

Each test document is removed from the folder index before it is classified
(leave-one-out), so the classifier never sees the document itself lying in
its own target folder - only its neighbours, as for a newly arriving scan.

Stages (--stage):
  extract     only run the content extraction and cache it
  candidates  extraction (cached) + deterministic candidate search, no
              folder decision: is the right folder among the candidates?
  full        everything, as in production (default)

Runs the LOCAL classifier (Ollama) unless --cloud is given; document text
never leaves the machines it would reach in normal operation. With --cloud
the folder decision is made by Anthropic exactly as in production with
`use_anthropic_classifier`: sender, title, keywords and the folder list of
every test document are sent there (never the text) - API costs apply. Results and the extraction
cache (which contain real document names) go to eval-results/, which is
gitignored - never commit them, this repository is public.

Not part of the container image; a development tool.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import random
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pymupdf as fitz

MIN_WORDS = 40
MAX_PAGES_READ = 5
RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval-results"


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


def _scan_tree(tree: Path, root: str, excluded: list[str]) -> dict[str, list[str]]:
    """Folder path -> names of the files directly inside it."""
    folder_files: dict[str, list[str]] = {}
    for current, dirnames, filenames in os.walk(tree):
        rel = Path(current).relative_to(tree).as_posix()
        here = root if rel == "." else f"{root}/{rel}"
        if _is_excluded(here, excluded):
            dirnames[:] = []
            continue
        folder_files[here] = sorted(filenames)
    return folder_files


class ExtractionCache:
    """Content extraction is the slow, deterministic part (~10 s per
    document) - cache it so candidate search and folder decision can be
    iterated on without repeating it."""

    def __init__(self, classifier, enabled: bool):
        self._classifier = classifier
        self._enabled = enabled
        self._dir = RESULTS_DIR / "extraction-cache"
        self._dir.mkdir(parents=True, exist_ok=True)

    def get(self, text: str, filename: str, host: str, model: str):
        from depot.models import ContentExtraction

        prompt = self._classifier._CONTENT_SYSTEM_PROMPT
        key = hashlib.sha1(f"{model}\0{prompt}\0{filename}\0{text}".encode("utf-8")).hexdigest()
        path = self._dir / f"{key}.json"
        if self._enabled and path.is_file():
            return ContentExtraction.model_validate_json(path.read_text(encoding="utf-8")), 0.0
        started = time.perf_counter()
        # With the summary, so the same cached extraction serves --cloud too
        # (it is the last field: what comes before it is generated the same).
        content = self._classifier.extract_content(text, filename, host, model, summary=True)
        seconds = time.perf_counter() - started
        path.write_text(content.model_dump_json(), encoding="utf-8")
        return content, seconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tree", required=True, type=Path, help="local path of the hand-sorted Dokumente folder")
    parser.add_argument("--limit", type=int, default=100, help="number of documents to sample")
    parser.add_argument("--seed", type=int, default=7, help="sampling seed (same seed = same documents)")
    parser.add_argument("--stage", choices=("extract", "candidates", "full"), default="full")
    parser.add_argument("--keep-filenames", action="store_true",
                        help="pass the real filename to the classifier instead of a neutral 'scan.pdf'")
    parser.add_argument("--no-extract-cache", action="store_true",
                        help="always re-run the extraction (for realistic timings)")
    parser.add_argument("--exclude", action="append", default=[], help="additional folder prefix to exclude")
    parser.add_argument("--threshold", type=float, default=float(os.environ.get("CONFIDENCE_THRESHOLD", "0.6")))
    parser.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M"))
    parser.add_argument("--embedding-model", default=os.environ.get("EMBEDDING_MODEL", ""),
                        help="Ollama embedding model for the semantic half of the shortlist ('' = lexical only)")
    parser.add_argument("--cloud", action="store_true",
                        help="folder decision via Anthropic (needs ANTHROPIC_API_KEY, e.g. from .env)")
    parser.add_argument("--anthropic-model", default=os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5"))
    parser.add_argument("--depot-path", type=Path, default=Path(__file__).resolve().parent.parent,
                        help="checkout whose depot/ package is evaluated (for comparing two versions)")
    parser.add_argument("--label", default="", help="name for this run, used in the output filename")
    args = parser.parse_args()

    sys.path.insert(0, str(args.depot_path))
    from depot import classifier, signals  # noqa: E402  (imported from the chosen checkout)

    try:
        from depot import candidates as candidates_module  # noqa: E402
    except ImportError:
        candidates_module = None
    embedder = None
    if args.embedding_model:
        from depot.embeddings import Embedder  # noqa: E402

        embedder = Embedder(args.ollama_host, args.embedding_model, RESULTS_DIR / "embedding-cache.sqlite3")

    anthropic_api_key = None
    if args.cloud:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
        anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            parser.error("--cloud needs ANTHROPIC_API_KEY (environment or .env)")

    tree: Path = args.tree.resolve()
    root = tree.name
    excluded = _excluded_prefixes(tree, root, args.exclude)
    folder_files = _scan_tree(tree, root, excluded)
    folders = [f for f in folder_files if f != root]

    pdfs = [(tree.parent / folder / name, folder)
            for folder, names in folder_files.items() for name in names if name.lower().endswith(".pdf")]
    pdfs.sort()
    random.Random(args.seed).shuffle(pdfs)
    cases: list[tuple[Path, str, str]] = []
    for pdf, truth in pdfs:
        if len(cases) >= args.limit:
            break
        text = _read_text(pdf)
        if len(text.split()) >= MIN_WORDS and truth != root:
            cases.append((pdf, truth, text))

    print(f"{len(folders)} candidate folders, {len(cases)} test documents "
          f"(seed {args.seed}, {'real' if args.keep_filenames else 'neutral'} filenames), "
          f"stage {args.stage}, depot from {args.depot_path}", flush=True)

    cache = ExtractionCache(classifier, enabled=not args.no_extract_cache)
    classify_params = inspect.signature(classifier.classify).parameters
    rows = []
    for i, (pdf, truth, text) in enumerate(cases, 1):
        filename = pdf.name if args.keep_filenames else "scan.pdf"
        name_signals = signals.analyze_filename(filename)
        # Leave-one-out: the document must not find itself in its target folder.
        loo_files = dict(folder_files)
        loo_files[truth] = [n for n in folder_files[truth] if n != pdf.name]

        row = dict(file=pdf.name, truth=truth)
        try:
            content, extract_seconds = cache.get(text, filename, args.ollama_host, args.model)
            row.update(title=content.title, correspondent=content.correspondent,
                       keywords=", ".join(getattr(content, "keywords", [])))
            if args.stage == "extract":
                print(f"[{i:3d}/{len(cases)}] {extract_seconds:5.1f}s  {content.correspondent} | {content.title}", flush=True)
                continue

            if candidates_module is not None:
                query = candidates_module.DocumentQuery(
                    correspondent=content.correspondent, title=content.title,
                    keywords=list(content.keywords), filename_title=name_signals.title or "", text=text,
                )
                semantic = None
                if embedder is not None:
                    semantic = classifier.semantic_similarities(embedder, query, folders, loo_files, root)
                ranked = candidates_module.rank_candidates(query, folders, loo_files, limit=10, semantic=semantic)
                paths = [c.path for c in ranked]
                rank = paths.index(truth) + 1 if truth in paths else 0
                ancestor_rank = next(
                    (n for n, p in enumerate(paths, 1) if p == truth or truth.startswith(f"{p}/")), 0
                )
                row.update(cand_rank=rank, cand_ancestor_rank=ancestor_rank, cand_count=len(paths))
            if args.stage == "candidates":
                print(f"[{i:3d}/{len(cases)}] rank={row.get('cand_rank', '-'):>2} "
                      f"ancestor_rank={row.get('cand_ancestor_rank', '-'):>2}  {truth}", flush=True)
                rows.append(row)
                continue

            started = time.perf_counter()
            kwargs = dict(
                ocr_text=text, original_filename=filename, existing_folders=folders,
                ollama_host=args.ollama_host, model=args.model, dokumente_root=root, content=content,
            )
            if "folder_files" in classify_params:
                kwargs["folder_files"] = loo_files
            if "resolve_date" in classify_params:
                kwargs["resolve_date"] = lambda d: signals.resolve_issue_date(
                    d, text, name_signals, None, date.today())[0]
            if args.cloud:
                kwargs.pop("folder_files", None)
                outcome, tags = classifier.classify_via_anthropic(
                    **kwargs, folder_files=loo_files, filename_title=name_signals.title,
                    anthropic_api_key=anthropic_api_key, anthropic_model=args.anthropic_model,
                    embedder=embedder,
                )
                if "ANTHROPIC-NICHT-ERREICHBAR" in tags:
                    raise RuntimeError("Anthropic call failed")
            else:
                if embedder is not None and "embedder" in classify_params:
                    kwargs["embedder"] = embedder
                outcome, tags = classifier.classify(**kwargs)
            seconds = extract_seconds + time.perf_counter() - started
            verdict = _verdict(truth, outcome.folder, outcome.confidence >= args.threshold)
            row.update(predicted=outcome.folder, confidence=f"{outcome.confidence:.2f}", verdict=verdict,
                       top_level_ok=outcome.folder.split("/")[1:2] == truth.split("/")[1:2],
                       new_folder=outcome.is_new_folder, seconds=f"{seconds:.1f}", tags="; ".join(tags))
        except Exception as exc:
            row.update(predicted=f"ERROR: {exc}", confidence="0.00", verdict="fehler", top_level_ok=False,
                       new_folder=False, seconds="0.0", tags="")
            if args.stage != "full":
                print(f"[{i:3d}/{len(cases)}] ERROR {exc}", flush=True)
                continue
        rows.append(row)
        print(f"[{i:3d}/{len(cases)}] {row['verdict']:18s} {row['seconds']:>5}s  {truth}  ->  "
              f"{row['predicted']}  ({row['confidence']})", flush=True)

    if not rows:
        return
    n = len(rows)
    print("\n=== Ergebnis ===")
    if "cand_rank" in rows[0]:
        for k in (1, 3, 5, 10):
            exact = sum(0 < r.get("cand_rank", 0) <= k for r in rows)
            near = sum(0 < r.get("cand_ancestor_rank", 0) <= k for r in rows)
            print(f"Kandidaten Top-{k:<2}: richtiger Ordner {exact:3d} / {n} ({exact / n:.0%}), "
                  f"Ordner oder ein Vorfahre {near:3d} / {n} ({near / n:.0%})")
    if args.stage == "full":
        count = lambda v: sum(r["verdict"] == v for r in rows)  # noqa: E731
        filed = [r for r in rows if r["verdict"] not in ("unsortiert", "fehler")]
        print(f"exakter Ordner:            {count('exakt'):3d} / {n}  ({count('exakt') / n:.0%})")
        print(f"eine Ebene daneben:        {count('eine-ebene-daneben'):3d} / {n}")
        print(f"richtiger Top-Level:       {sum(r['top_level_ok'] for r in filed):3d} / {n}")
        print(f"Unsortiert (unsicher):     {count('unsortiert'):3d} / {n}")
        print(f"FALSCH bei hoher Konfidenz:{count('falsch'):3d} / {n}  ({count('falsch') / n:.0%})")
        print(f"Fehler:                    {count('fehler'):3d} / {n}")
        print(f"Sekunden pro Dokument:     {sum(float(r['seconds']) for r in rows) / n:.1f}"
              f"{'' if args.no_extract_cache else '  (Extraktion ggf. aus dem Cache - nicht repraesentativ)'}")

    label = f"-{args.label}" if args.label else ""
    out_file = RESULTS_DIR / f"eval-{datetime.now():%Y%m%d-%H%M%S}-{args.stage}{label}.csv"
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with out_file.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nDetails: {out_file}")


if __name__ == "__main__":
    main()
