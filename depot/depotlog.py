from __future__ import annotations

from datetime import date, datetime

from depot.webdav import WebDavClient

TAG_OCR_FAILED = "OCR-FEHLGESCHLAGEN"
TAG_UNSORTED = "UNSORTIERT"
TAG_NEW_FOLDER = "NEUER-ORDNER"
TAG_DATE_UNCERTAIN = "DATUM-UNSICHER"
TAG_ERROR = "FEHLER"
TAG_QUARANTINED = "FEHLER-QUARANTAENE"
TAG_SKIPPED = "UEBERSPRUNGEN"


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
        existing = self._webdav.get(rel_path) or b""
        self._webdav.put(rel_path, existing + line.encode("utf-8"))
