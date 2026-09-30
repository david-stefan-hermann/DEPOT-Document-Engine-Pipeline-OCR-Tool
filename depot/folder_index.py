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


def scan_local_tree(mount_root: Path, webdav_root: str) -> dict[str, list[str]] | None:
    """Every subfolder under `webdav_root` (itself excluded) as a
    WebDAV-relative path, mapped to the names of the files directly inside
    it - from one local directory walk. None if that folder isn't there
    locally."""
    webdav_root = webdav_root.strip("/")
    start = mount_root / webdav_root
    if not start.is_dir():
        return None
    result: dict[str, list[str]] = {}
    for current, _, filenames in os.walk(start):
        rel = Path(current).relative_to(start).as_posix()
        if rel != ".":
            result[f"{webdav_root}/{rel}"] = sorted(filenames)
    return result


def list_local_folders(mount_root: Path, webdav_root: str) -> list[str] | None:
    """WebDAV-relative paths of every subfolder under `webdav_root` - the
    same result as WebDavClient.list_folders_recursive."""
    tree = scan_local_tree(mount_root, webdav_root)
    return None if tree is None else list(tree)
