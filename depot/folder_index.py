"""Reads the existing Dokumente/ tree straight from the local read-only
bind mount instead of asking Nextcloud for it folder by folder."""
from __future__ import annotations

import os
from pathlib import Path


def local_mount_root(scan_eingang_local_path: str, scan_eingang_webdav_path: str) -> Path | None:
    """The local directory corresponding to the WebDAV root, derived from
    the two ways the scan inbox is already configured (its local path ends
    with its WebDAV-relative path). None if they don't line up that way,
    e.g. when only the inbox itself is mounted."""
    local = scan_eingang_local_path.replace("\\", "/").rstrip("/")
    suffix = "/" + scan_eingang_webdav_path.strip("/")
    if not local.endswith(suffix):
        return None
    return Path(local[: -len(suffix)] or "/")


def list_local_folders(mount_root: Path, webdav_root: str) -> list[str] | None:
    """WebDAV-relative paths of every subfolder under `webdav_root` (itself
    excluded) - the same result as WebDavClient.list_folders_recursive, from
    one local directory walk. None if that folder isn't there locally."""
    webdav_root = webdav_root.strip("/")
    start = mount_root / webdav_root
    if not start.is_dir():
        return None
    result: list[str] = []
    for current, dirnames, _ in os.walk(start):
        rel = Path(current).relative_to(start).as_posix()
        base = webdav_root if rel == "." else f"{webdav_root}/{rel}"
        result.extend(f"{base}/{name}" for name in dirnames)
    return result
