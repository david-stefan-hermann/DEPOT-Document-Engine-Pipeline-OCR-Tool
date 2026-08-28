from __future__ import annotations

import logging
import time
from datetime import date, datetime

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
# transient write failure here (e.g. the user has the log file open in
# Nextcloud's editor and it's briefly locked) must never be treated as a
# pipeline failure for an already-successfully-filed document.
_WRITE_RETRIES = 3
_WRITE_RETRY_DELAY_SECONDS = 2.0


def is_log_file(filename: str, prefix: str) -> bool:
    return prefix in filename


def log_filename(prefix: str, on_date: date) -> str:
    return f"{prefix} {on_date:%d-%m-%Y}.txt"


class DepotLog:
    """Appends one line per processed file to a daily log kept alongside the
    scans in Scan-Eingang itself, as the audit trail replacing the review
    step. Writes go through WebDAV since the local bind mount is read-only.
    """

    def __init__(self, webdav: WebDavClient, scan_eingang_webdav_path: str, prefix: str):
        self._webdav = webdav
        self._scan_eingang_path = scan_eingang_webdav_path.strip("/")
        self._prefix = prefix

    def _rel_path(self, on_date: date) -> str:
        return f"{self._scan_eingang_path}/{log_filename(self._prefix, on_date)}"

    def append(
        self,
        original_filename: str,
        message: str,
        tags: list[str] | None = None,
        path: str | None = None,
        on_date: date | None = None,
    ) -> None:
        """`path`, when given (the Nextcloud destination path of a filed or
        quarantined document), is always the last thing on the line with
        nothing after it — so it can be selected/copied by jumping to the
        end of the line, without trimming trailing decoration first."""
        on_date = on_date or date.today()
        timestamp = datetime.now().strftime("%H:%M:%S")
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        path_str = f" | {path}" if path else ""
        line = f"{timestamp} | {original_filename} | {message}{tag_str}{path_str}\n"

        rel_path = self._rel_path(on_date)

        for attempt in range(1, _WRITE_RETRIES + 1):
            try:
                existing = self._webdav.get(rel_path) or b""
                self._webdav.put(rel_path, existing + line.encode("utf-8"))
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
