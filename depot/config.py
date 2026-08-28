from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@dataclass(frozen=True)
class Config:
    nextcloud_webdav_url: str
    nextcloud_user: str
    nextcloud_app_password: str

    scan_eingang_local_path: str
    scan_eingang_webdav_path: str

    dokumente_webdav_root: str
    fallback_folder: str
    error_folder: str

    ollama_host: str
    ollama_model: str
    confidence_threshold: float

    log_file_prefix: str
    ocr_language: str
    max_concurrent_jobs: int
    state_db_path: str

    supported_extensions: frozenset[str]

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            nextcloud_webdav_url=_require("NEXTCLOUD_WEBDAV_URL").rstrip("/"),
            nextcloud_user=_require("NEXTCLOUD_USER"),
            nextcloud_app_password=_require("NEXTCLOUD_APP_PASSWORD"),
            scan_eingang_local_path=_require("SCAN_EINGANG_LOCAL_PATH"),
            scan_eingang_webdav_path=os.environ.get(
                "SCAN_EINGANG_WEBDAV_PATH", "Scan-Eingang"
            ).strip("/"),
            dokumente_webdav_root=os.environ.get(
                "DOKUMENTE_WEBDAV_ROOT", "Dokumente"
            ).strip("/"),
            fallback_folder=os.environ.get(
                "FALLBACK_FOLDER", "Dokumente/Unsortiert"
            ).strip("/"),
            error_folder=os.environ.get(
                "ERROR_FOLDER", "Dokumente/_Fehlerhaft"
            ).strip("/"),
            ollama_host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
            ollama_model=os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M"),
            confidence_threshold=float(os.environ.get("CONFIDENCE_THRESHOLD", "0.6")),
            log_file_prefix=os.environ.get("LOG_FILE_PREFIX", "DEPOT Dateilog"),
            ocr_language=os.environ.get("OCR_LANGUAGE", "deu"),
            max_concurrent_jobs=int(os.environ.get("MAX_CONCURRENT_JOBS", "1")),
            state_db_path=os.environ.get(
                "STATE_DB_PATH", "/scratch/depot-state.sqlite3"
            ),
            supported_extensions=frozenset(
                {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
            ),
        )
