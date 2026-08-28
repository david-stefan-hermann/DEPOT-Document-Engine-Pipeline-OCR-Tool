from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import date
from pathlib import Path

import httpx

from depot import classifier, depotlog, naming, ocr
from depot.config import Config
from depot.depotlog import DepotLog
from depot.state import StateStore
from depot.webdav import WebDavClient

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
# file in this process's lifetime.
_TRANSIENT_RETRY_DELAY_SECONDS = 30.0
MAX_TRANSIENT_RETRIES = 5

# How long the Dokumente/ folder listing is cached before being re-fetched
# from WebDAV. Folders created by DEPOT itself are added to the cache
# immediately (see _remember_folder), so this only bounds staleness for
# folders the user creates/renames by hand while a batch is running.
FOLDER_CACHE_TTL_SECONDS = 300.0


class Pipeline:
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
            self.webdav, config.scan_eingang_webdav_path, config.log_file_prefix
        )
        self.state = state or StateStore(config.state_db_path)
        self._transient_retries: dict[str, int] = {}
        self._transient_lock = threading.Lock()
        self._folder_cache: list[str] | None = None
        self._folder_cache_time: float = 0.0
        self._folder_cache_lock = threading.Lock()

    def _get_existing_folders(self) -> list[str]:
        with self._folder_cache_lock:
            now = time.monotonic()
            stale = self._folder_cache is None or (now - self._folder_cache_time) > FOLDER_CACHE_TTL_SECONDS
            if stale:
                self._folder_cache = self.webdav.list_folders_recursive(self.config.dokumente_webdav_root)
                self._folder_cache_time = now
            return list(self._folder_cache)

    def _remember_folder(self, folder: str) -> None:
        """Make a just-created (or just-confirmed) folder visible to the next
        classification immediately, without waiting for the cache TTL."""
        with self._folder_cache_lock:
            if self._folder_cache is not None and folder not in self._folder_cache:
                self._folder_cache.append(folder)

    def close(self) -> None:
        self.webdav.close()
        self.state.close()

    def run_workers(self, in_queue: "queue.Queue[Path]") -> list[threading.Thread]:
        workers = []
        for i in range(max(1, self.config.max_concurrent_jobs)):
            t = threading.Thread(
                target=self._worker_loop, args=(in_queue,), daemon=True, name=f"depot-worker-{i}"
            )
            t.start()
            workers.append(t)
        return workers

    def _worker_loop(self, in_queue: "queue.Queue[Path]") -> None:
        while True:
            path = in_queue.get()
            try:
                self.process_one(path, in_queue)
            except Exception:
                log.exception("Unhandled error processing %s", path)
            finally:
                in_queue.task_done()

    def process_one(self, path: Path, requeue: "queue.Queue[Path] | None" = None) -> None:
        original_name = path.name
        log.info("Processing %s", original_name)

        try:
            self._process(path)
            self.state.reset(original_name)
            with self._transient_lock:
                self._transient_retries.pop(original_name, None)
        except TRANSIENT_EXCEPTIONS as exc:
            self._handle_transient_failure(path, requeue, exc)
        except Exception as exc:
            self._handle_permanent_failure(path, exc)

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
            log.error("Giving up on transient retries for %s", original_name)
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
                self._quarantine(path)
                self.state.reset(original_name)
                self.depot_log.append(
                    original_name,
                    f"Nach {count} Fehlversuchen quarantänisiert: {exc}",
                    tags=[depotlog.TAG_QUARANTINED],
                )
            except Exception:
                log.exception("Failed to quarantine %s", original_name)
        else:
            self.depot_log.append(
                original_name,
                f"Fehler ({count}/3 Versuche): {exc}",
                tags=[depotlog.TAG_ERROR],
            )

    def _quarantine(self, path: Path) -> None:
        self.webdav.mkcol(self.config.error_folder)
        existing = {
            e.path.rsplit("/", 1)[-1]
            for e in self.webdav.list_dir(self.config.error_folder)
            if not e.is_collection
        }
        final_name = naming.resolve_collision(path.name, existing)
        dest_rel = f"{self.config.error_folder}/{final_name}"
        self.webdav.put(dest_rel, path.read_bytes())
        self._delete_source(path.name)

    def _delete_source(self, original_name: str) -> None:
        src_rel = f"{self.config.scan_eingang_webdav_path}/{original_name}"
        self.webdav.delete(src_rel)

    def _process(self, path: Path) -> None:
        cfg = self.config
        original_name = path.name
        today = date.today()

        ocr_result = ocr.process_file(path, cfg.ocr_language)
        produced_path = Path(ocr_result.ocr_pdf_path)
        using_raw_original = produced_path == path
        ext = path.suffix.lstrip(".") if using_raw_original else "pdf"

        tags: list[str] = []

        if ocr_result.ocr_failed:
            target_folder = cfg.fallback_folder
            title = path.stem
            issue_date = None
            confidence = 0.0
            tags += [depotlog.TAG_OCR_FAILED, depotlog.TAG_UNSORTED]
        else:
            existing_folders = self._get_existing_folders()
            result, classifier_tags = classifier.classify(
                ocr_text=ocr_result.text,
                original_filename=original_name,
                existing_folders=existing_folders,
                ollama_host=cfg.ollama_host,
                model=cfg.ollama_model,
            )
            tags += classifier_tags
            confidence = result.confidence
            title = result.title
            issue_date = result.issue_date

            if confidence < cfg.confidence_threshold:
                target_folder = cfg.fallback_folder
                tags.append(depotlog.TAG_UNSORTED)
            else:
                target_folder = result.folder
                if result.is_new_folder:
                    tags.append(depotlog.TAG_NEW_FOLDER)

        if issue_date is None:
            tags.append(depotlog.TAG_DATE_UNCERTAIN)

        self.webdav.mkcol(target_folder)
        self._remember_folder(target_folder)
        desired_name = naming.build_filename(title, issue_date, today, ext=ext)
        existing_names = {
            e.path.rsplit("/", 1)[-1]
            for e in self.webdav.list_dir(target_folder)
            if not e.is_collection
        }
        final_name = naming.resolve_collision(desired_name, existing_names)
        dest_rel = f"{target_folder}/{final_name}"

        self.webdav.put(dest_rel, produced_path.read_bytes())
        self._delete_source(original_name)

        if not using_raw_original:
            produced_path.unlink(missing_ok=True)

        self.depot_log.append(
            original_name,
            f"-> {dest_rel} | confidence={confidence:.2f}",
            tags=tags,
        )
        log.info("Filed %s -> %s (confidence=%.2f)", original_name, dest_rel, confidence)
