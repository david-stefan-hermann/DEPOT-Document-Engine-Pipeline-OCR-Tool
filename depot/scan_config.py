from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def is_config_file(filename: str, config_file_name: str) -> bool:
    return filename == config_file_name


def load_excluded_folders(
    scan_eingang_local_path: str, config_subfolder: str, config_file_name: str
) -> list[str]:
    """Reads `excluded_folders` from the user-editable DEPOT Config.json in
    Scan-Eingang/<config_subfolder>, if present. Missing file or malformed
    content just means "no exclusions" rather than a hard failure — this is
    a convenience knob, not critical configuration."""
    config_path = Path(scan_eingang_local_path) / config_subfolder / config_file_name
    if not config_path.is_file():
        return []
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not read/parse %s; ignoring: %s", config_path, exc)
        return []

    excluded = data.get("excluded_folders", []) if isinstance(data, dict) else None
    if not isinstance(excluded, list) or not all(isinstance(f, str) for f in excluded):
        log.warning("%s: 'excluded_folders' must be a list of strings; ignoring.", config_path)
        return []
    return [f.strip().strip("/") for f in excluded if f.strip()]


def filter_excluded(folders: list[str], excluded_prefixes: list[str]) -> list[str]:
    """Removes any folder that is, or is nested under, one of the excluded
    prefixes. The excluded folders themselves are untouched on Nextcloud —
    they're just never offered as a classification target."""
    if not excluded_prefixes:
        return folders
    return [
        f for f in folders
        if not any(f == prefix or f.startswith(f"{prefix}/") for prefix in excluded_prefixes)
    ]
