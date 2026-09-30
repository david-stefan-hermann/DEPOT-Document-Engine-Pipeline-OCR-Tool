"""Deterministic shortlist of folders a document might belong in.

The level-by-level walk only ever showed the model folder NAMES, so the very
first choice at the root was a guess about what each top-level folder might
contain - the cause of most real misfilings (an insurance letter about a
vehicle went to the general insurance folder instead of to the vehicle,
where every similar document already was). This module looks at what is
actually in the folders: their names AND the names of the files already
filed there, and ranks the folders by how much they have in common with the
new document. The model then chooses among a handful of concrete, evidenced
candidates instead of guessing a branch.

Purely lexical and local - no model call, no network. Because it is derived
from the live tree on every run, it also learns from every manual
correction: move a file to the right folder and the next document of that
kind finds it there.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

MAX_TEXT_CHARS = 6000

# From this strength on a candidate is backed by real evidence - typically
# the sender's name in the folder path or in several filed documents. Below
# it, matches are mostly coincidental words. Measured on 120 hand-filed
# documents: with the best candidate above this line it was the right
# folder in 76 % of cases (88 % not counting documents the owner had
# deliberately filed somewhere unusual), below it in 20 %.
STRONG_EVIDENCE = 5.0
DEFAULT_LIMIT = 8
EXAMPLES_PER_CANDIDATE = 3

# Query parts, by how much a match on them says about where a document goes.
_WEIGHT_CORRESPONDENT = 3.0
_WEIGHT_TITLE = 2.0
_WEIGHT_KEYWORD = 1.0
_WEIGHT_TEXT = 1.0

# Where in a folder the match was found. A folder BELOW one named after the
# sender counts almost like that folder itself: the documents usually are in
# such a subfolder ("<Arbeitgeber>/Arbeitgeber", "<Bank>/2024"), while the
# sender's own folder only holds the subfolders. Which of the two it is, is
# then decided by what is filed in them.
_FIELD_LEAF = 1.5
_FIELD_ANCESTOR = 1.2
_FIELD_FILES = 1.0
_FIELD_SUBTREE = 0.4

_YEAR = re.compile(r"(19|20)\d{2}")
# Asset folders browsers create next to a saved web page - never a filing target.
_SAVED_PAGE_ASSETS = re.compile(r"(_files|-Dateien)$")

_UMLAUTS = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})
_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

# Legal forms and address noise that say nothing about who a sender is.
LEGAL_FORMS = frozenset({
    "gmbh", "ggmbh", "mbh", "ag", "kg", "kgaa", "ohg", "ug", "se", "gbr", "ev", "eg", "co", "ltd", "inc",
    "llc", "plc", "sarl", "bv",
})
_STOPWORDS = LEGAL_FORMS | frozenset({
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer", "eines", "einem", "einen",
    "und", "oder", "fuer", "von", "vom", "mit", "auf", "aus", "bei", "zur", "zum", "ueber", "unter",
    "nach", "vor", "ist", "sind", "wir", "sie", "ihr", "ihre", "ihren", "ihrem", "ihres", "ihnen",
    "the", "and", "for", "pdf", "jpg", "jpeg", "png", "scan", "scn", "img", "dokument", "dokumente",
    "document", "kopie", "copy", "neu", "alt", "datum", "unsicher", "duplikat", "seite", "herr", "frau",
})


def fold(text: str) -> str:
    """Case-, umlaut- and normalization-insensitive form for comparisons
    ("Grundstücksservice" and "Grundstuecksservice" must meet)."""
    return unicodedata.normalize("NFC", text).casefold().translate(_UMLAUTS)


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for group in _TOKEN.findall(fold(text)):
        parts = group.split("-")
        if len(parts) > 1:
            tokens.append("".join(parts))  # "MT-07" also as "mt07"
        tokens.extend(parts)
    return [t for t in tokens if len(t) >= 3 and not t.isdigit() and t not in _STOPWORDS]


def _similarity(a: str, b: str) -> float:
    """How strongly two tokens refer to the same thing: identical, one an
    inflection/compound of the other ("Rechnung"/"Rechnungen",
    "Stromrechnung"/"Rechnung"), or sharing a long stem."""
    if a == b:
        return 1.0
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) < 5:
        return 0.0
    if longer.startswith(shorter):
        return 0.7
    if shorter in longer:
        return 0.5
    prefix = 0
    for x, y in zip(a, b):
        if x != y:
            break
        prefix += 1
    return 0.5 if prefix >= 6 else 0.0


@dataclass(frozen=True)
class DocumentQuery:
    correspondent: str = ""
    title: str = ""
    keywords: list[str] = field(default_factory=list)
    filename_title: str = ""
    pdf_title: str = ""
    text: str = ""


@dataclass(frozen=True)
class Candidate:
    path: str
    score: float
    # The score in units that mean the same in a small and a large tree
    # (1.0 = one maximally distinctive word matching exactly in the folder's
    # files) - what STRONG_EVIDENCE is measured in.
    strength: float = 0.0
    # Names of files already in that folder that are most related to the
    # document (or, for a folder holding only subfolders, their names).
    examples: list[str] = field(default_factory=list)
    # How many files directly in that folder carry the document's sender.
    sender_files: int = 0


class _Folder:
    __slots__ = ("path", "leaf_tokens", "ancestor_tokens", "file_counts", "subtree_counts", "files", "children")

    def __init__(self, path: str, root: str, filenames: list[str]):
        self.path = path
        segments = path[len(root) + 1:].split("/") if path != root else []
        self.leaf_tokens = set(tokenize(segments[-1])) if segments else set()
        self.ancestor_tokens = {t for s in segments[:-1] for t in tokenize(s)}
        self.files = [(name, set(tokenize(name.rsplit(".", 1)[0]))) for name in filenames]
        self.file_counts: Counter[str] = Counter(t for _, toks in self.files for t in toks)
        self.subtree_counts: Counter[str] = Counter()
        self.children: list[str] = []


def _strength(count: int) -> float:
    """One file with a matching name is a hint, several are a pattern."""
    return 1.0 - 0.5 ** count if count else 0.0


_REORDER_WINDOW = 30


def _subfolders_with_more_evidence_first(
    scored: list[tuple[float, _Folder]], file_evidence: dict[str, float]
) -> list[tuple[float, _Folder]]:
    """A folder named after the sender outranks its own subfolders on the
    name match alone - but if similar documents are filed in one of those
    subfolders and not in the folder itself, the subfolder is the better
    guess and goes first."""
    ordered = list(scored)
    position = 0
    while position < len(ordered):
        parent = ordered[position][1].path
        better = [
            item for item in ordered[position + 1:]
            if item[1].path.startswith(f"{parent}/") and file_evidence[item[1].path] > file_evidence[parent]
        ]
        if better:
            best = max(better, key=lambda item: file_evidence[item[1].path])
            ordered.remove(best)
            ordered.insert(position, best)
        position += 1
    return ordered


def rank_candidates(
    query: DocumentQuery,
    folders: list[str],
    folder_files: dict[str, list[str]] | None = None,
    limit: int = DEFAULT_LIMIT,
) -> list[Candidate]:
    """The `limit` folders that have the most in common with the document,
    best first. Empty if nothing matches at all.

    Subfolders that are just a year ("2024") never appear on their own:
    their files count for the folder above, and which year a document goes
    into follows from its date, not from a choice."""
    if not folders:
        return []
    folder_files = folder_files or {}
    known = set(folders)
    root = min(folders, key=len).split("/")[0]

    files_of: dict[str, list[str]] = {}
    for path in folders:
        if any(_SAVED_PAGE_ASSETS.search(segment) for segment in path.split("/")):
            continue
        parent, _, leaf = path.rpartition("/")
        target = parent if _YEAR.fullmatch(leaf) and parent in known else path
        files_of.setdefault(target, []).extend(folder_files.get(path, []))

    index = {path: _Folder(path, root, filenames) for path, filenames in files_of.items()}
    for path, folder in index.items():
        parent = path.rsplit("/", 1)[0]
        if parent in index:
            index[parent].children.append(path.rsplit("/", 1)[1])
        # files further down still say something about what this branch is for
        ancestor, depth = parent, 1
        while ancestor in index:
            weight = 0.5 ** (depth - 1)
            for token, count in folder.file_counts.items():
                index[ancestor].subtree_counts[token] += count * weight
            ancestor, depth = ancestor.rsplit("/", 1)[0], depth + 1

    document_frequency: Counter[str] = Counter()
    for folder in index.values():
        document_frequency.update(folder.leaf_tokens | set(folder.file_counts))
    total = len(index)

    def idf(token: str) -> float:
        # a token no folder knows is as distinctive as one found exactly once
        return math.log(1.0 + total / max(document_frequency[token], 1))

    weights: dict[str, float] = {}
    for text, weight in (
        (query.correspondent, _WEIGHT_CORRESPONDENT),
        (query.title, _WEIGHT_TITLE),
        (query.filename_title, _WEIGHT_TITLE),
        (query.pdf_title, _WEIGHT_TITLE),
        (" ".join(query.keywords), _WEIGHT_KEYWORD),
    ):
        for token in tokenize(text):
            weights[token] = max(weights.get(token, 0.0), weight)

    # Each query token's counterparts in the index vocabulary, with similarity.
    related: dict[str, dict[str, float]] = {}
    for token in weights:
        matches = {}
        for vocab_token in document_frequency:
            similarity = _similarity(token, vocab_token)
            if similarity:
                matches[vocab_token] = similarity
        related[token] = matches
    # Folder names that literally occur in the document text ("MT-07", the
    # name of an insurer or employer) - exact matches only, text is noisy.
    path_vocabulary = {t for f in index.values() for t in f.leaf_tokens | f.ancestor_tokens}
    text_tokens = (set(tokenize(query.text[:MAX_TEXT_CHARS])) - set(weights)) & path_vocabulary

    sender_tokens = set(tokenize(query.correspondent))
    sender_total = sum(idf(t) for t in sender_tokens)

    scored: list[tuple[float, _Folder]] = []
    # How well the files in a folder match WHAT the document is (title,
    # keywords) - the sender is left out: below a folder named after the
    # sender, every subfolder shares that match, and a lone spreadsheet
    # carrying the sender's name must not outweigh a subfolder full of
    # documents of the same kind.
    file_evidence: dict[str, float] = {}
    for folder in index.values():
        score = 0.0
        evidence = 0.0
        for token, weight in weights.items():
            best = 0.0
            best_files = 0.0
            for vocab_token, similarity in related[token].items():
                base = similarity * idf(vocab_token)
                files = _FIELD_FILES * _strength(folder.file_counts.get(vocab_token, 0))
                value = base * (
                    _FIELD_LEAF * (vocab_token in folder.leaf_tokens)
                    + _FIELD_ANCESTOR * (vocab_token in folder.ancestor_tokens)
                    + files
                    + _FIELD_SUBTREE * _strength(round(folder.subtree_counts.get(vocab_token, 0)))
                )
                best = max(best, value)
                best_files = max(best_files, base * files)
            score += weight * best
            if token not in sender_tokens:
                evidence += weight * best_files
        file_evidence[folder.path] = evidence
        for token in text_tokens:
            score += _WEIGHT_TEXT * idf(token) * (
                _FIELD_LEAF * (token in folder.leaf_tokens) + _FIELD_ANCESTOR * (token in folder.ancestor_tokens)
            )
        if score > 0:
            scored.append((score, folder))

    scored.sort(key=lambda item: (-item[0], item[1].path))
    scored = _subfolders_with_more_evidence_first(scored[:_REORDER_WINDOW], file_evidence)
    query_tokens = set(weights)
    result = []
    for score, folder in scored[:limit]:
        ranked_files = sorted(folder.files, key=lambda f: (-sum(weights[t] for t in f[1] & query_tokens), f[0]))
        examples = [name for name, _ in ranked_files[:EXAMPLES_PER_CANDIDATE]]
        if not examples and folder.children:
            examples = [f"Unterordner: {', '.join(sorted(folder.children)[:6])}"]
        sender_files = 0
        if sender_total:
            sender_files = sum(
                1 for _, tokens in folder.files
                if sum(idf(t) for t in sender_tokens & tokens) >= 0.6 * sender_total
            )
        result.append(Candidate(
            path=folder.path, score=score, strength=score / math.log(1.0 + total),
            examples=examples, sender_files=sender_files,
        ))
    return result
