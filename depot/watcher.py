from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from depot.depotlog import is_log_file

log = logging.getLogger(__name__)

STABLE_CHECK_INTERVAL = 0.5
STABLE_CHECKS_REQUIRED = 4  # ~2s of unchanged size before considering a file "done"


class ScanWatcher:
    """Watches the local (read-only) Scan-Eingang mount for new files and
    feeds fully-written, supported scans into a processing queue. Also
    performs a startup sweep so files that arrived while the container was
    down get picked up too."""

    def __init__(
        self,
        local_path: str,
        supported_extensions: frozenset[str],
        log_file_prefix: str,
        out_queue: "queue.Queue[Path]",
    ):
        self._local_path = Path(local_path)
        self._supported_extensions = supported_extensions
        self._log_file_prefix = log_file_prefix
        self._out_queue = out_queue
        self._pending: set[Path] = set()
        self._pending_lock = threading.Lock()
        self._observer = Observer()

    def _should_consider(self, path: Path) -> bool:
        if not path.is_file():
            return False
        if is_log_file(path.name, self._log_file_prefix):
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

    def start(self) -> None:
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
