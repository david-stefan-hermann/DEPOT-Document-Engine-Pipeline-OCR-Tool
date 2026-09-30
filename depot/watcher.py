from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from depot.depotlog import is_log_file
from depot.scan_config import is_config_file

log = logging.getLogger(__name__)

STABLE_CHECK_INTERVAL = 0.5
STABLE_CHECKS_REQUIRED = 4  # ~2s of unchanged size before considering a file "done"

# A file the periodic sweep finds is only queued once it has not been
# modified for this long - anything younger is either still being written
# or already on its way through the watcher's own debounce.
SWEEP_MIN_AGE_SECONDS = 60.0


class ScanWatcher:
    """Watches the local (read-only) Scan-Eingang mount for new files and
    feeds fully-written, supported scans into a processing queue. Also
    performs a startup sweep so files that arrived while the container was
    down get picked up too, and a periodic sweep for anything that is still
    lying there afterwards: a file whose filesystem event never arrived
    (inotify does not see every write on every kind of mount), or one whose
    processing was given up after repeated transient failures - the next
    sweep is its next attempt."""

    def __init__(
        self,
        local_path: str,
        supported_extensions: frozenset[str],
        log_file_prefix: str,
        config_file_name: str,
        out_queue: "queue.Queue[Path]",
    ):
        self._local_path = Path(local_path)
        self._supported_extensions = supported_extensions
        self._log_file_prefix = log_file_prefix
        self._config_file_name = config_file_name
        self._out_queue = out_queue
        self._pending: set[Path] = set()
        self._pending_lock = threading.Lock()
        self._observer = Observer()

    def _should_consider(self, path: Path) -> bool:
        if not path.is_file():
            return False
        if is_log_file(path.name, self._log_file_prefix):
            return False
        if is_config_file(path.name, self._config_file_name):
            return False
        return path.suffix.lower() in self._supported_extensions

    def _debounce_and_enqueue(self, path: Path) -> None:
        with self._pending_lock:
            if path in self._pending:
                return
            self._pending.add(path)
        try:
            last_size = -1
            stable_count = 0
            while stable_count < STABLE_CHECKS_REQUIRED:
                time.sleep(STABLE_CHECK_INTERVAL)
                try:
                    size = path.stat().st_size
                except FileNotFoundError:
                    log.debug("File disappeared before it stabilized: %s", path)
                    return
                if size == last_size:
                    stable_count += 1
                else:
                    stable_count = 0
                    last_size = size
            log.info("New scan ready: %s", path.name)
            self._out_queue.put(path)
        finally:
            with self._pending_lock:
                self._pending.discard(path)

    def startup_sweep(self) -> None:
        if not self._local_path.is_dir():
            log.warning("Scan-Eingang path does not exist yet: %s", self._local_path)
            return
        for entry in sorted(self._local_path.iterdir()):
            if self._should_consider(entry):
                log.info("Startup sweep found: %s", entry.name)
                self._out_queue.put(entry)

    def sweep(self, min_age_seconds: float = SWEEP_MIN_AGE_SECONDS) -> list[Path]:
        """Queue every supported file in the inbox that has been unchanged
        for `min_age_seconds` and is not already being debounced. Returns
        the files it queued. The queue itself ignores paths that are
        already waiting in it, and the pipeline skips files it is
        currently processing."""
        if not self._local_path.is_dir():
            return []
        now = time.time()
        with self._pending_lock:
            pending = set(self._pending)
        found: list[Path] = []
        for entry in sorted(self._local_path.iterdir()):
            if entry in pending or not self._should_consider(entry):
                continue
            try:
                if now - entry.stat().st_mtime < min_age_seconds:
                    continue
            except FileNotFoundError:
                continue
            found.append(entry)
            self._out_queue.put(entry)
        if found:
            log.info("Sweep found %d file(s) in the inbox: %s", len(found), ", ".join(p.name for p in found))
        return found

    def start_periodic_sweep(self, interval_seconds: float) -> threading.Thread | None:
        """Run sweep() every `interval_seconds` in a background thread.
        Disabled (returns None) for a non-positive interval."""
        if interval_seconds <= 0:
            return None

        def _loop() -> None:
            while True:
                time.sleep(interval_seconds)
                try:
                    self.sweep()
                except Exception:
                    log.exception("Periodic sweep failed")

        thread = threading.Thread(target=_loop, daemon=True, name="depot-sweep")
        thread.start()
        log.info("Sweeping %s every %.0f s for files left behind", self._local_path, interval_seconds)
        return thread

    def start(self) -> None:
        if not self._local_path.is_dir():
            raise RuntimeError(
                f"Scan-Eingang path does not exist or is not a directory: {self._local_path}. "
                "Check SCAN_EINGANG_LOCAL_PATH and the bind mount."
            )
        handler = _Handler(self)
        self._observer.schedule(handler, str(self._local_path), recursive=False)
        self._observer.start()
        log.info("Watching %s for new scans", self._local_path)

    def stop(self) -> None:
        self._observer.stop()
        self._observer.join()


class _Handler(FileSystemEventHandler):
    def __init__(self, watcher: ScanWatcher):
        self._watcher = watcher

    def on_created(self, event) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if self._watcher._should_consider(path):
            threading.Thread(
                target=self._watcher._debounce_and_enqueue, args=(path,), daemon=True
            ).start()

    def on_moved(self, event) -> None:
        if event.is_directory:
            return
        path = Path(event.dest_path)
        if self._watcher._should_consider(path):
            threading.Thread(
                target=self._watcher._debounce_and_enqueue, args=(path,), daemon=True
            ).start()
