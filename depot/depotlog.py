from __future__ import annotations

import logging
import threading
import time
from datetime import date, datetime, time as dtime

from depot.naming import resolve_collision
from depot.webdav import WebDavClient

log = logging.getLogger(__name__)

TAG_OCR_FAILED = "OCR-FEHLGESCHLAGEN"
TAG_UNSORTED = "UNSORTIERT"
TAG_NEW_FOLDER = "NEUER-ORDNER"
TAG_DATE_UNCERTAIN = "DATUM-UNSICHER"
TAG_ERROR = "FEHLER"
TAG_QUARANTINED = "FEHLER-QUARANTAENE"
TAG_SKIPPED = "UEBERSPRUNGEN"

# The log write is a best-effort side channel, not part of the actual filing
# operation (which has already completed by the time append() is called). A
# transient write failure here (e.g. a brief network hiccup) must never be
# treated as a pipeline failure for an already-successfully-filed document.
_WRITE_RETRIES = 3
_WRITE_RETRY_DELAY_SECONDS = 2.0


def is_log_file(filename: str, prefix: str) -> bool:
    return prefix in filename


def log_filename(prefix: str, on_date: date, on_time: dtime) -> str:
    return f"{prefix} {on_date:%d-%m-%Y} {on_time:%H-%M-%S}.txt"


class DepotLog:
    """Writes one log file per processed file (event) into
    Scan-Eingang/<config_subfolder>, as the audit trail replacing the review
    step. Filename keeps the existing "DEPOT Dateilog DD-MM-YYYY" pattern
    with the time (HH-MM-SS) appended, so every processing event gets its
    own file instead of a shared, ever-growing daily log. Writes go through
    WebDAV since the local bind mount is read-only.
    """

    def __init__(
        self,
        webdav: WebDavClient,
        scan_eingang_webdav_path: str,
        prefix: str,
        config_subfolder: str = "Config",
    ):
        self._webdav = webdav
        self._folder_path = f"{scan_eingang_webdav_path.strip('/')}/{config_subfolder.strip('/')}"
        self._prefix = prefix
        self._folder_ensured = False
        self._used_names: set[str] = set()
        self._lock = threading.Lock()

    def _ensure_folder(self) -> None:
        if not self._folder_ensured:
            self._webdav.mkcol(self._folder_path)
            self._folder_ensured = True

    def _reserve_filename(self, on_date: date, on_time: dtime) -> str:
        """Picks a filename for this event, disambiguating against every
        filename this instance has already used in the current process (e.g.
        two workers logging within the same second)."""
        desired = log_filename(self._prefix, on_date, on_time)
        with self._lock:
            final = resolve_collision(desired, self._used_names)
            self._used_names.add(final)
        return final

    def append(
        self,
        original_filename: str,
        message: str,
        tags: list[str] | None = None,
        path: str | None = None,
        on_date: date | None = None,
        on_time: dtime | None = None,
    ) -> None:
        """`path`, when given (the Nextcloud destination path of a filed or
        quarantined document), is always the last thing on the line with
        nothing after it — so it can be selected/copied by jumping to the
        end of the line, without trimming trailing decoration first."""
        now = datetime.now()
        on_date = on_date or now.date()
        on_time = on_time or now.time()
        timestamp = on_time.strftime("%H:%M:%S")
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        path_str = f" | {path}" if path else ""
        line = f"{timestamp} | {original_filename} | {message}{tag_str}{path_str}\n"

        filename = self._reserve_filename(on_date, on_time)
        rel_path = f"{self._folder_path}/{filename}"

        for attempt in range(1, _WRITE_RETRIES + 1):
            try:
                self._ensure_folder()
                self._webdav.put(rel_path, line.encode("utf-8"))
                return
            except Exception as exc:
                if attempt < _WRITE_RETRIES:
                    log.warning(
                        "Log write to %s failed (attempt %d/%d, retrying): %s",
                        rel_path, attempt, _WRITE_RETRIES, exc,
                    )
                    time.sleep(_WRITE_RETRY_DELAY_SECONDS)
                else:
                    log.error(
                        "Giving up writing log entry to %s after %d attempts: %s. "
                        "Entry that could not be written: %r",
                        rel_path, _WRITE_RETRIES, exc, line,
                    )
