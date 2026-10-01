from __future__ import annotations

import hashlib
import logging
import queue
import threading
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable

import httpx

from depot import classifier, depotlog, folder_index, naming, ocr, scan_config, signals
from depot.config import Config
from depot.depotlog import DepotLog
from depot.embeddings import Embedder
from depot.models import ContentExtraction, OcrResult
from depot.ocr_cache import OcrCache
from depot.state import StateStore
from depot.webdav import PreconditionFailed, WebDavClient

log = logging.getLogger(__name__)

TRANSIENT_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.TimeoutException,
    ConnectionError,
)

# In-memory only (deliberately not persisted): how many times a transient
# infra failure (Ollama/WebDAV unreachable) has been retried for a given
# file in this process's lifetime. Once these are used up the file simply
# stays in the inbox until the periodic sweep queues it again.
_TRANSIENT_RETRY_DELAY_SECONDS = 30.0
MAX_TRANSIENT_RETRIES = 5

# How long the Dokumente/ folder listing is cached before being re-fetched
# from WebDAV. Folders created by DEPOT itself are added to the cache
# immediately (see _remember_folder), so this only bounds staleness for
# folders the user creates/renames by hand while a batch is running.
FOLDER_CACHE_TTL_SECONDS = 300.0
# Reading the tree from the local mount costs milliseconds instead of one
# WebDAV request per folder, so it can be refreshed far more often.
LOCAL_FOLDER_CACHE_TTL_SECONDS = 15.0

# Content confidence of a document whose title comes from its filename alone
# because OCR found no text (a photo). Above the default threshold on
# purpose - the user named the file, that is real information - but the
# folder decision's own confidence still applies on top.
FILENAME_ONLY_CONFIDENCE = 0.7


@dataclass
class PreparedDocument:
    """What the OCR stage hands to the LLM stage for one scan: everything
    that could be learned about the file without the model."""

    path: Path
    content_hash: str
    ocr_result: OcrResult
    ocr_seconds: float
    name_signals: signals.FilenameSignals
    pdf_title: str | None
    pdf_created: date | None
    file_into_dokumente: bool
    save_processed_copy: bool
    use_anthropic_classifier: bool


class Pipeline:
    """Two stages, so the CPU and the GPU work at the same time:

    1. OCR stage (`max_concurrent_jobs` threads): duplicate check, filename
       and PDF-metadata signals, OCR. Its result is kept in the OCR cache
       until the scan is filed, so a retry never repeats the OCR.
    2. LLM stage (exactly one thread): classification, filename, upload,
       deletion of the source scan. One thread because the model holds one
       document's context at a time - a second one would evict the first
       from Ollama's prompt cache.

    While document n is being classified, document n+1 is already being
    recognized.
    """

    def __init__(
        self,
        config: Config,
        webdav: WebDavClient | None = None,
        depot_log: DepotLog | None = None,
        state: StateStore | None = None,
    ):
        self.config = config
        self.webdav = webdav or WebDavClient(
            config.nextcloud_webdav_url,
            config.nextcloud_user,
            config.nextcloud_app_password,
        )
        self.depot_log = depot_log or DepotLog(
            self.webdav, config.scan_eingang_webdav_path, config.log_file_prefix, config.config_subfolder
        )
        self.state = state or StateStore(config.state_db_path)
        scratch = Path(config.state_db_path).parent
        self.ocr_cache = OcrCache(config.ocr_cache_dir or str(scratch / "ocr-cache"))
        self.ocr_cache.prune()
        self.embedder: Embedder | None = (
            Embedder(config.ollama_host, config.embedding_model, scratch / "embeddings.sqlite3")
            if config.embedding_model
            else None
        )
        self._transient_retries: dict[str, int] = {}
        self._transient_lock = threading.Lock()
        self._folder_cache: list[str] | None = None
        self._folder_cache_time: float = 0.0
        self._folder_cache_ttl: float = FOLDER_CACHE_TTL_SECONDS
        # Names of the files in each cached folder - what the candidate
        # search reads "what is this folder for" from. None when the tree
        # can only be listed via WebDAV (folder names only).
        self._folder_files: dict[str, list[str]] | None = None
        self._folder_cache_lock = threading.Lock()
        # Every folder known to exist on the server (unfiltered, unlike
        # _folder_cache) - lets uploads skip the per-path-segment existence
        # checks of mkcol for folders that are certainly there.
        self._known_folders: set[str] = set()
        self._mount_root = (
            folder_index.local_mount_root(config.scan_eingang_local_path, config.scan_eingang_webdav_path)
            if config.use_local_folder_listing
            else None
        )
        # Files somewhere between "taken from the queue" and "filed or
        # failed" - in either stage or waiting between them.
        self._in_flight: set[str] = set()
        self._in_flight_lock = threading.Lock()

    def _get_existing_folders(self) -> list[str]:
        with self._folder_cache_lock:
            now = time.monotonic()
            stale = self._folder_cache is None or (now - self._folder_cache_time) > self._folder_cache_ttl
            if stale:
                folders, folder_files, self._folder_cache_ttl = self._list_all_folders()
                self._known_folders = set(folders)
                excluded = scan_config.load_excluded_folders(
                    self.config.scan_eingang_local_path,
                    self.config.config_subfolder,
                    self.config.config_file_name,
                )
                # Structural, non-optional exclusions (on top of whatever the
                # user configured):
                # - the scan inbox itself must never be offered as a filing
                #   target. This matters most when Scan-Eingang lives inside
                #   Dokumente/ - without this, the classifier could file a
                #   document straight back into (or under) the folder the
                #   watcher watches, which would pick it up again and
                #   reprocess it in a loop.
                # - the low-confidence fallback folder must never be a
                #   candidate the model can deliberately choose - it's meant
                #   to be reached only via the confidence-threshold check
                #   below, as a visible "needs review" signal. Seen in
                #   production: the model happily filing a document there on
                #   its own with confidence 0.95 and no tags, silently
                #   defeating the point of a review bucket.
                excluded = [*excluded, self.config.scan_eingang_webdav_path, self.config.fallback_folder]
                self._folder_cache = scan_config.filter_excluded(folders, excluded)
                self._folder_files = (
                    None if folder_files is None
                    else {f: folder_files.get(f, []) for f in self._folder_cache}
                )
                self._folder_cache_time = now
            return list(self._folder_cache)

    def _get_folder_files(self) -> dict[str, list[str]] | None:
        """File names per folder, as of the listing _get_existing_folders()
        last returned."""
        with self._folder_cache_lock:
            return self._folder_files

    def _list_all_folders(self) -> tuple[list[str], dict[str, list[str]] | None, float]:
        """Every folder under Dokumente/, the file names in each (if known)
        and how long that listing may be cached: from the local mount when
        it covers the tree, otherwise via WebDAV as before (folders only)."""
        if self._mount_root is not None:
            tree = folder_index.scan_local_tree(self._mount_root, self.config.dokumente_webdav_root)
            if tree is not None:
                return list(tree), tree, LOCAL_FOLDER_CACHE_TTL_SECONDS
        return (
            self.webdav.list_folders_recursive(self.config.dokumente_webdav_root),
            None,
            FOLDER_CACHE_TTL_SECONDS,
        )

    def _remember_folder(self, folder: str) -> None:
        """Make a just-created (or just-confirmed) folder visible to the next
        classification immediately, without waiting for the cache TTL."""
        with self._folder_cache_lock:
            if self._folder_cache is not None and folder not in self._folder_cache:
                self._folder_cache.append(folder)

    def close(self) -> None:
        self.webdav.close()
        self.state.close()
        if self.embedder is not None:
            self.embedder.close()

    def _preload_models(self) -> None:
        classifier.preload_model(self.config.ollama_host, self.config.ollama_model)
        if self.embedder is not None:
            self.embedder.preload()

    # ---- workers -----------------------------------------------------------

    def run_workers(self, in_queue: "queue.Queue[Path]") -> list[threading.Thread]:
        """Start the OCR threads (`max_concurrent_jobs`) and the single LLM
        thread; see the class docstring."""
        llm_queue: "queue.Queue[PreparedDocument]" = queue.Queue()
        workers = []
        for i in range(max(1, self.config.max_concurrent_jobs)):
            t = threading.Thread(
                target=self._ocr_loop, args=(in_queue, llm_queue), daemon=True, name=f"depot-ocr-{i}"
            )
            t.start()
            workers.append(t)
        t = threading.Thread(target=self._llm_loop, args=(llm_queue, in_queue), daemon=True, name="depot-llm")
        t.start()
        workers.append(t)
        return workers

    def _ocr_loop(self, in_queue: "queue.Queue[Path]", llm_queue: "queue.Queue[PreparedDocument]") -> None:
        while True:
            path = in_queue.get()
            try:
                self.process_one(path, in_queue, hand_off=llm_queue.put)
            except Exception:
                log.exception("Unhandled error processing %s", path)
            finally:
                in_queue.task_done()

    def _llm_loop(self, llm_queue: "queue.Queue[PreparedDocument]", requeue: "queue.Queue[Path]") -> None:
        while True:
            prepared = llm_queue.get()
            try:
                self._finish_guarded(prepared, requeue)
            except Exception:
                log.exception("Unhandled error finishing %s", prepared.path)
            finally:
                llm_queue.task_done()

    def process_one(
        self,
        path: Path,
        requeue: "queue.Queue[Path] | None" = None,
        hand_off: "Callable[[PreparedDocument], None] | None" = None,
    ) -> None:
        """Run the OCR stage for one scan and then either hand the result to
        the LLM stage (`hand_off`) or, without one, finish it right here."""
        original_name = path.name
        # The watcher can report one file twice (created + moved, or sweep +
        # event). Seen in production: the second run started a second after
        # the first had filed and deleted the scan, failed with "No such
        # file" and was logged/counted as a processing error.
        with self._in_flight_lock:
            if original_name in self._in_flight:
                log.info("Skipping %s: already being processed.", original_name)
                return
            self._in_flight.add(original_name)
        handed_off = False
        try:
            if not path.exists():
                log.info("Skipping %s: no longer in the scan inbox.", original_name)
                return
            log.info("Processing %s", original_name)

            try:
                prepared = self._prepare(path)
            except TRANSIENT_EXCEPTIONS as exc:
                self._handle_transient_failure(path, requeue, exc)
                return
            except Exception as exc:
                self._handle_permanent_failure(path, exc)
                return
            if prepared is None:  # a duplicate - filed as such, nothing left to do
                self._mark_done(original_name)
            elif hand_off is not None:
                hand_off(prepared)
                handed_off = True
            else:
                self._finish_guarded(prepared, requeue)
        finally:
            if not handed_off:
                self._release(original_name)

    def _finish_guarded(self, prepared: PreparedDocument, requeue: "queue.Queue[Path] | None") -> None:
        """The LLM stage for one prepared scan, with the same failure
        handling as the OCR stage. A transient failure sends the path back
        to the start of the queue - where the OCR cache makes the first
        stage nearly free."""
        path = prepared.path
        try:
            if not path.exists():
                # Taken out of the inbox by hand while it waited its turn.
                log.info("Skipping %s: removed from the scan inbox while waiting.", path.name)
                self.ocr_cache.discard(prepared.content_hash)
                return
            self._finish(prepared)
            self._mark_done(path.name)
        except TRANSIENT_EXCEPTIONS as exc:
            self._handle_transient_failure(path, requeue, exc)
        except Exception as exc:
            self._handle_permanent_failure(path, exc)
        finally:
            self._release(path.name)

    def _release(self, original_name: str) -> None:
        with self._in_flight_lock:
            self._in_flight.discard(original_name)

    def _mark_done(self, original_name: str) -> None:
        self.state.reset(original_name)
        with self._transient_lock:
            self._transient_retries.pop(original_name, None)

    def _handle_transient_failure(
        self, path: Path, requeue: "queue.Queue[Path] | None", exc: Exception
    ) -> None:
        original_name = path.name
        with self._transient_lock:
            retries = self._transient_retries.get(original_name, 0) + 1
            self._transient_retries[original_name] = retries

        log.warning("Transient failure for %s (%d/%d): %s", original_name, retries, MAX_TRANSIENT_RETRIES, exc)
        self.depot_log.append(
            original_name,
            f"Vorübergehender Fehler ({retries}/{MAX_TRANSIENT_RETRIES}): {exc}",
            tags=[depotlog.TAG_ERROR],
        )
        if retries >= MAX_TRANSIENT_RETRIES or requeue is None:
            log.error("Giving up on transient retries for %s until the next sweep", original_name)
            with self._transient_lock:
                self._transient_retries.pop(original_name, None)
            return
        timer = threading.Timer(_TRANSIENT_RETRY_DELAY_SECONDS, requeue.put, args=(path,))
        timer.daemon = True
        timer.start()

    def _handle_permanent_failure(self, path: Path, exc: Exception) -> None:
        original_name = path.name
        log.exception("Permanent-looking failure for %s", original_name)
        count = self.state.increment_failure(original_name)

        if self.state.should_quarantine(original_name):
            try:
                dest_rel = self._quarantine(path)
                self.state.reset(original_name)
                self.depot_log.append(
                    original_name,
                    f"Nach {count} Fehlversuchen quarantänisiert: {exc}",
                    tags=[depotlog.TAG_QUARANTINED],
                    path=dest_rel,
                )
            except Exception:
                log.exception("Failed to quarantine %s", original_name)
        else:
            self.depot_log.append(
                original_name,
                f"Fehler ({count}/3 Versuche): {exc}",
                tags=[depotlog.TAG_ERROR],
            )

    def _quarantine(self, path: Path) -> str:
        data = path.read_bytes()
        dest_rel = self._put_with_collision_resolution(self.config.error_folder, path.name, data)
        self._delete_source(path.name)
        self.ocr_cache.discard(hashlib.sha256(data).hexdigest())
        return dest_rel

    def _delete_source(self, original_name: str) -> None:
        # Hardcoded to the scan inbox on purpose: this is the ONLY place in
        # the whole codebase that ever calls webdav.delete(), and it must
        # never be reachable with a path derived from anywhere else (e.g.
        # Dokumente/). Folders are never deleted anywhere in this codebase.
        src_rel = f"{self.config.scan_eingang_webdav_path}/{original_name}"
        self.webdav.delete(src_rel)

    # ---- uploads -----------------------------------------------------------

    def _names_in_folder(self, folder: str) -> set[str]:
        """The file names already in `folder`, creating the folder if it
        does not exist. Read from the local mount when the folder is there
        (no request at all); otherwise one WebDAV listing."""
        if self._mount_root is not None:
            local_dir = self._mount_root / folder
            if local_dir.is_dir():
                names = {entry.name for entry in local_dir.iterdir() if entry.is_file()}
                with self._folder_cache_lock:
                    self._known_folders.add(folder)
                return names
        with self._folder_cache_lock:
            assumed_existing = folder in self._known_folders
        if not assumed_existing:
            self.webdav.mkcol(folder)
        return {
            e.path.rsplit("/", 1)[-1]
            for e in self.webdav.list_dir(folder)
            if not e.is_collection
        }

    def _put_with_collision_resolution(self, folder: str, desired_name: str, data: bytes) -> str:
        """Upload under `desired_name`, or ' (2)', ' (3)', ... if that name
        is taken. The upload itself refuses to overwrite, so a name taken
        between the listing and the upload is caught too."""
        existing_names = self._names_in_folder(folder)
        with self._folder_cache_lock:
            assumed_existing = folder in self._known_folders
        for attempt in range(2):
            final_name = naming.resolve_collision(desired_name, existing_names)
            dest_rel = f"{folder}/{final_name}"
            try:
                self.webdav.put(dest_rel, data, overwrite=False)
            except PreconditionFailed:
                if attempt:
                    raise
                # Written by someone else in the meantime; list again, via
                # WebDAV this time, and pick the next free name.
                existing_names = {
                    e.path.rsplit("/", 1)[-1] for e in self.webdav.list_dir(folder) if not e.is_collection
                }
                existing_names.add(final_name)
                continue
            except RuntimeError:
                if not assumed_existing:
                    raise
                # The folder was listed a moment ago but is gone now (removed or
                # renamed by hand in the meantime): create it and try once more.
                self.webdav.mkcol(folder)
                self.webdav.put(dest_rel, data, overwrite=False)
            break
        with self._folder_cache_lock:
            self._known_folders.add(folder)
        return dest_rel

    def _processed_folder(self) -> str:
        cfg = self.config
        return f"{cfg.scan_eingang_webdav_path}/{cfg.config_subfolder}/{cfg.processed_subfolder}"

    def _file_duplicate(self, path: Path, prior_dest: str, file_into_dokumente: bool) -> None:
        """A scan whose exact bytes were already filed (and are still where
        they were filed) is not processed again: it goes to the review
        folder under the first copy's name, visibly marked, so nothing
        disappears silently but no second "real" copy is created either."""
        folder = self.config.fallback_folder if file_into_dokumente else self._processed_folder()
        name = naming.duplicate_filename(prior_dest.rsplit("/", 1)[-1], path.suffix)
        dest_rel = self._put_with_collision_resolution(folder, name, path.read_bytes())
        self._delete_source(path.name)
        tags = [depotlog.TAG_DUPLICATE]
        if file_into_dokumente:
            tags.append(depotlog.TAG_UNSORTED)
        self.depot_log.append(path.name, f"Duplikat von {prior_dest}", tags=tags, path=dest_rel)
        log.info("Duplicate %s (same content as %s) -> %s", path.name, prior_dest, dest_rel)

    # ---- stage 1: everything before the model --------------------------------

    def _prepare(self, path: Path) -> PreparedDocument | None:
        """Duplicate check, title signals and OCR. None if the scan turned
        out to be a duplicate (already filed as such)."""
        cfg = self.config
        original_name = path.name

        # Read fresh per document (cheap local file read) rather than at
        # startup, so the user can toggle these directly in DEPOT
        # Config.json in Nextcloud and have it take effect on the very next
        # scan, the same way excluded_folders already works.
        file_into_dokumente, save_processed_copy, use_anthropic_classifier = (
            scan_config.load_processing_switches(
                cfg.scan_eingang_local_path, cfg.config_subfolder, cfg.config_file_name
            )
        )

        with path.open("rb") as fh:
            content_hash = hashlib.file_digest(fh, "sha256").hexdigest()
        prior_dest = self.state.find_processed(content_hash)
        if prior_dest is not None and self.webdav.exists(prior_dest):
            if self.state.source_pending_delete(content_hash):
                # Our own earlier attempt: uploaded, then lost the connection
                # before it could remove the scan. Only that step is left.
                self._delete_source(original_name)
                self.state.mark_source_deleted(content_hash)
                self.ocr_cache.discard(content_hash)
                self.depot_log.append(
                    original_name, "Ablage war bereits erfolgt; Scan jetzt aus dem Eingang entfernt",
                    tags=[], path=prior_dest,
                )
                log.info("%s had already been filed to %s; removed the source scan.", original_name, prior_dest)
                return None
            self._file_duplicate(path, prior_dest, file_into_dokumente)
            return None

        # What the document already says about itself without any OCR/LLM.
        name_signals = signals.analyze_filename(original_name)
        pdf_title, pdf_created = signals.read_pdf_metadata(path)

        started = time.perf_counter()
        ocr_result = self.ocr_cache.get(content_hash, path)
        if ocr_result is not None:
            log.info("Reusing the OCR result of an earlier attempt for %s", original_name)
        else:
            # Let cold models load while OCR runs instead of after it.
            threading.Thread(target=self._preload_models, daemon=True, name="depot-preload").start()
            ocr_result = self.ocr_cache.put(content_hash, ocr.process_file(path, cfg.ocr_language), path)
        ocr_seconds = time.perf_counter() - started

        return PreparedDocument(
            path=path,
            content_hash=content_hash,
            ocr_result=ocr_result,
            ocr_seconds=ocr_seconds,
            name_signals=name_signals,
            pdf_title=pdf_title,
            pdf_created=pdf_created,
            file_into_dokumente=file_into_dokumente,
            save_processed_copy=save_processed_copy,
            use_anthropic_classifier=use_anthropic_classifier,
        )

    # ---- stage 2: classification, filename, upload ---------------------------

    def _finish(self, prepared: PreparedDocument) -> None:
        cfg = self.config
        path = prepared.path
        original_name = path.name
        today = date.today()
        ocr_result = prepared.ocr_result
        name_signals = prepared.name_signals
        file_into_dokumente = prepared.file_into_dokumente

        produced_path = Path(ocr_result.ocr_pdf_path)
        using_raw_original = produced_path == path
        ext = path.suffix.lstrip(".") if using_raw_original else "pdf"

        tags: list[str] = []
        target_folder: str | None = None
        model_date: date | None = None
        suggestion: str | None = None

        def resolve_date(candidate: date | None) -> tuple[date | None, str | None]:
            return signals.resolve_issue_date(
                candidate,
                ocr_result.text,
                name_signals,
                # Only a born-digital PDF's creation date says when the
                # document was issued; for a scan it is just the scan time.
                prepared.pdf_created if ocr_result.born_digital else None,
                today,
            )

        # No text, but the user gave the file a real name: that name is the
        # document's title, and is enough to file it by. (Previously every
        # such file - typically a photo - went to Unsortiert unclassified.)
        content: ContentExtraction | None = None
        if ocr_result.ocr_failed:
            tags.append(depotlog.TAG_OCR_FAILED)
            if name_signals.title:
                content = ContentExtraction(
                    title=name_signals.title, correspondent="", confidence=FILENAME_ONLY_CONFIDENCE
                )
                tags.append(depotlog.TAG_FILENAME_ONLY)

        started = time.perf_counter()
        if ocr_result.ocr_failed and content is None:
            title = path.stem
            correspondent = None
            confidence = 0.0
            if file_into_dokumente:
                target_folder = cfg.fallback_folder
                tags.append(depotlog.TAG_UNSORTED)
        elif file_into_dokumente:
            existing_folders = self._get_existing_folders()
            classify_args = dict(
                ocr_text=ocr_result.text,
                original_filename=original_name,
                existing_folders=existing_folders,
                ollama_host=cfg.ollama_host,
                model=cfg.ollama_model,
                dokumente_root=cfg.dokumente_webdav_root,
                content=content,
                pdf_title=prepared.pdf_title,
                filename_title=name_signals.title,
                folder_files=self._get_folder_files(),
                # The folder decision needs the date too (year subfolders).
                resolve_date=lambda candidate: resolve_date(candidate)[0],
            )
            if prepared.use_anthropic_classifier:
                result, classifier_tags = classifier.classify_via_anthropic(
                    **classify_args,
                    anthropic_api_key=cfg.anthropic_api_key,
                    anthropic_model=cfg.anthropic_model,
                )
            else:
                result, classifier_tags = classifier.classify(**classify_args, embedder=self.embedder)
            tags += classifier_tags
            confidence = result.confidence
            title = result.title
            correspondent = result.correspondent
            model_date = result.issue_date

            if confidence < cfg.confidence_threshold:
                target_folder = cfg.fallback_folder
                tags.append(depotlog.TAG_UNSORTED)
                # Not sure enough to file it there - but worth telling.
                if result.folder != cfg.dokumente_webdav_root:
                    suggestion = result.folder
            else:
                target_folder = result.folder
                if result.is_new_folder:
                    tags.append(depotlog.TAG_NEW_FOLDER)
        else:
            # Filing is switched off: only title/date/correspondent for the
            # filename are needed, not a filing decision - skip the folder
            # walk entirely (saves the folder listing and several Ollama
            # calls it would otherwise cost).
            if content is None:
                content = classifier.extract_content(
                    ocr_result.text, original_name, cfg.ollama_host, cfg.ollama_model,
                    pdf_title=prepared.pdf_title,
                )
            confidence = content.confidence
            title = content.title
            correspondent = naming.normalize_correspondent(content.correspondent)
            model_date = content.issue_date
            tags.append(depotlog.TAG_FILING_DISABLED)
        llm_seconds = time.perf_counter() - started

        issue_date, date_source = resolve_date(model_date)
        if issue_date is None:
            tags.append(depotlog.TAG_DATE_UNCERTAIN)
        elif date_source == "filename":
            tags.append(depotlog.TAG_DATE_FROM_FILENAME)
        elif date_source == "metadata":
            tags.append(depotlog.TAG_DATE_FROM_METADATA)

        started = time.perf_counter()
        desired_name = naming.build_filename(title, issue_date, today, ext=ext, correspondent=correspondent)
        produced_bytes = produced_path.read_bytes()

        dest_rel: str | None = None
        if target_folder is not None:
            self._remember_folder(target_folder)
            dest_rel = self._put_with_collision_resolution(target_folder, desired_name, produced_bytes)

        processed_rel: str | None = None
        if prepared.save_processed_copy:
            processed_rel = self._put_with_collision_resolution(
                self._processed_folder(), desired_name, produced_bytes
            )
            tags.append(depotlog.TAG_PROCESSED_COPY)

        if dest_rel is None and processed_rel is None:
            # Defense in depth: load_processing_switches() already forces
            # file_into_dokumente back to True when DEPOT Config.json has
            # both switches false, so this should be unreachable - but if it
            # ever happens anyway, refuse to delete the source rather than
            # lose the processed document. Caught by process_one() as a
            # permanent-looking failure; the scan stays in Scan-Eingang.
            raise RuntimeError(
                "Neither filing nor the processed-copy produced a stored destination; "
                "refusing to delete the source scan."
            )

        # Recorded before the delete: should the connection drop in between,
        # the next attempt removes the scan instead of filing it again.
        self.state.record_processed(prepared.content_hash, dest_rel or processed_rel, source_deleted=False)
        self._delete_source(original_name)
        self.state.mark_source_deleted(prepared.content_hash)
        dav_seconds = time.perf_counter() - started

        # Filed - the cached OCR result (and its PDF) is no longer needed.
        self.ocr_cache.discard(prepared.content_hash)

        message = (
            f"confidence={confidence:.2f} | "
            f"ocr={prepared.ocr_seconds:.1f}s llm={llm_seconds:.1f}s dav={dav_seconds:.1f}s"
        )
        if suggestion:
            message += f" | Vorschlag: {suggestion}"
        if dest_rel and processed_rel:
            message += f" | Kopie: {processed_rel}"
        log_path = dest_rel or processed_rel

        self.depot_log.append(original_name, message, tags=tags, path=log_path)
        log.info("Processed %s -> %s (confidence=%.2f)", original_name, log_path, confidence)
