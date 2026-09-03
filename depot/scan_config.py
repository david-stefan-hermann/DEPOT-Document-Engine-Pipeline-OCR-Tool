from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def is_config_file(filename: str, config_file_name: str) -> bool:
    return filename == config_file_name


def _load_json(scan_eingang_local_path: str, config_subfolder: str, config_file_name: str) -> dict:
    """Reads and parses the user-editable DEPOT Config.json in
    Scan-Eingang/<config_subfolder>, if present. Missing file, malformed
    JSON, or a non-object top level just means "use defaults everywhere"
    rather than a hard failure - this file is a convenience knob edited by
    hand in Nextcloud, not critical configuration, so a mistake in it must
    never take the pipeline down."""
    config_path = Path(scan_eingang_local_path) / config_subfolder / config_file_name
    if not config_path.is_file():
        return {}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not read/parse %s; ignoring: %s", config_path, exc)
        return {}
    if not isinstance(data, dict):
        log.warning("%s: expected a JSON object at the top level; ignoring.", config_path)
        return {}
    return data


def load_excluded_folders(
    scan_eingang_local_path: str, config_subfolder: str, config_file_name: str
) -> list[str]:
    """Reads `excluded_folders` from DEPOT Config.json."""
    data = _load_json(scan_eingang_local_path, config_subfolder, config_file_name)
    excluded = data.get("excluded_folders", [])
    if not isinstance(excluded, list) or not all(isinstance(f, str) for f in excluded):
        log.warning("%s: 'excluded_folders' must be a list of strings; ignoring.", config_file_name)
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


def load_processing_switches(
    scan_eingang_local_path: str, config_subfolder: str, config_file_name: str
) -> tuple[bool, bool, bool]:
    """Reads `file_into_dokumente`/`save_processed_copy`/`use_anthropic_classifier`
    from DEPOT Config.json, e.g.:
        { "file_into_dokumente": true, "save_processed_copy": false, "use_anthropic_classifier": false }
    Read fresh on every call (a cheap local file read) rather than cached,
    so toggling a switch in the file takes effect on the very next document
    instead of waiting on the folder-listing cache TTL. Missing file/keys
    default to (True, False, False) - the original fixed behavior. If
    file_into_dokumente and save_processed_copy would both end up False,
    DEPOT would have nowhere to put a processed document before deleting
    the source scan - file_into_dokumente wins instead, with a warning,
    rather than ever silently discarding one."""
    data = _load_json(scan_eingang_local_path, config_subfolder, config_file_name)

    file_into_dokumente = data.get("file_into_dokumente", True)
    if not isinstance(file_into_dokumente, bool):
        log.warning("%s: 'file_into_dokumente' must be true/false; using default true.", config_file_name)
        file_into_dokumente = True

    save_processed_copy = data.get("save_processed_copy", False)
    if not isinstance(save_processed_copy, bool):
        log.warning("%s: 'save_processed_copy' must be true/false; using default false.", config_file_name)
        save_processed_copy = False

    use_anthropic_classifier = data.get("use_anthropic_classifier", False)
    if not isinstance(use_anthropic_classifier, bool):
        log.warning(
            "%s: 'use_anthropic_classifier' must be true/false; using default false.", config_file_name
        )
        use_anthropic_classifier = False

    if not file_into_dokumente and not save_processed_copy:
        log.warning(
            "%s: both file_into_dokumente and save_processed_copy are false; "
            "falling back to file_into_dokumente=true so processed documents aren't discarded.",
            config_file_name,
        )
        file_into_dokumente = True

    return file_into_dokumente, save_processed_copy, use_anthropic_classifier
