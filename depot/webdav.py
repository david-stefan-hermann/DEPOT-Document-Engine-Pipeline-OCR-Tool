from __future__ import annotations

import logging
from dataclasses import dataclass
from urllib.parse import quote, urlparse
from xml.etree import ElementTree as ET

import httpx

log = logging.getLogger(__name__)

_DAV_NS = "{DAV:}"

_PROPFIND_BODY = b"""<?xml version="1.0" encoding="utf-8"?>
<d:propfind xmlns:d="DAV:">
  <d:prop>
    <d:resourcetype/>
    <d:displayname/>
  </d:prop>
</d:propfind>
"""


@dataclass(frozen=True)
class Entry:
    path: str  # relative to the WebDAV root, no leading/trailing slash
    is_collection: bool


class WebDavClient:
    """Minimal WebDAV client tailored to Nextcloud's quirks (no Depth:infinity
    support on PROPFIND for large trees, so recursive listing is done via
    repeated Depth:1 requests).
    """

    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._base_path = urlparse(self._base_url).path.rstrip("/")
        self._client = httpx.Client(
            auth=(username, password),
            timeout=timeout,
            follow_redirects=True,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "WebDavClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _url_for(self, rel_path: str) -> str:
        rel_path = rel_path.strip("/")
        quoted = quote(rel_path, safe="/")
        return f"{self._base_url}/{quoted}" if rel_path else self._base_url

    def _rel_path_from_href(self, href: str) -> str:
        path = urlparse(href).path
        if path.startswith(self._base_path):
            path = path[len(self._base_path):]
        from urllib.parse import unquote

        return unquote(path).strip("/")

    def check_connection(self) -> None:
        resp = self._client.request(
            "PROPFIND",
            self._url_for(""),
            headers={"Depth": "0"},
            content=_PROPFIND_BODY,
        )
        if resp.status_code >= 300:
            raise RuntimeError(
                f"WebDAV connectivity check failed: HTTP {resp.status_code} {resp.text[:300]}"
            )

    def list_dir(self, rel_path: str) -> list[Entry]:
        """List immediate children of rel_path. Returns [] if the folder does
        not exist (404) rather than raising, since callers use this both for
        existence checks and for tree walks."""
        resp = self._client.request(
            "PROPFIND",
            self._url_for(rel_path),
            headers={"Depth": "1"},
            content=_PROPFIND_BODY,
        )
        if resp.status_code == 404:
            return []
        if resp.status_code >= 300:
            raise RuntimeError(
                f"PROPFIND {rel_path!r} failed: HTTP {resp.status_code} {resp.text[:300]}"
            )

        root = ET.fromstring(resp.content)
        self_rel = rel_path.strip("/")
        entries: list[Entry] = []
        for response in root.findall(f"{_DAV_NS}response"):
            href = response.findtext(f"{_DAV_NS}href") or ""
            child_rel = self._rel_path_from_href(href)
            if child_rel == self_rel:
                continue  # PROPFIND Depth:1 includes the queried collection itself
            resourcetype = response.find(f".//{_DAV_NS}resourcetype")
            is_collection = resourcetype is not None and (
                resourcetype.find(f"{_DAV_NS}collection") is not None
            )
            entries.append(Entry(path=child_rel, is_collection=is_collection))
        return entries

    def list_folders_recursive(self, root_path: str) -> list[str]:
        """Return relative paths of every subfolder under root_path (root_path
        itself excluded), fetched fresh via repeated Depth:1 PROPFINDs."""
        result: list[str] = []
        queue = [root_path.strip("/")]
        while queue:
            current = queue.pop(0)
            for entry in self.list_dir(current):
                if entry.is_collection:
                    result.append(entry.path)
                    queue.append(entry.path)
        return result

    def exists(self, rel_path: str) -> bool:
        resp = self._client.request(
            "PROPFIND",
            self._url_for(rel_path),
            headers={"Depth": "0"},
            content=_PROPFIND_BODY,
        )
        return resp.status_code < 300

    def mkcol(self, rel_path: str) -> None:
        """Create a collection (folder), creating missing parent folders too."""
        rel_path = rel_path.strip("/")
        parts = rel_path.split("/")
        built = ""
        for part in parts:
            built = f"{built}/{part}" if built else part
            if self.exists(built):
                continue
            resp = self._client.request("MKCOL", self._url_for(built))
            if resp.status_code not in (201, 405):  # 405 = already exists (race)
                raise RuntimeError(
                    f"MKCOL {built!r} failed: HTTP {resp.status_code} {resp.text[:300]}"
                )

    def get(self, rel_path: str) -> bytes | None:
        resp = self._client.get(self._url_for(rel_path))
        if resp.status_code == 404:
            return None
        if resp.status_code >= 300:
            raise RuntimeError(
                f"GET {rel_path!r} failed: HTTP {resp.status_code} {resp.text[:300]}"
            )
        return resp.content

    def put(self, rel_path: str, data: bytes) -> None:
        resp = self._client.put(self._url_for(rel_path), content=data)
        if resp.status_code not in (200, 201, 204):
            raise RuntimeError(
                f"PUT {rel_path!r} failed: HTTP {resp.status_code} {resp.text[:300]}"
            )

    def delete(self, rel_path: str) -> None:
        resp = self._client.delete(self._url_for(rel_path))
        if resp.status_code not in (200, 204, 404):
            raise RuntimeError(
                f"DELETE {rel_path!r} failed: HTTP {resp.status_code} {resp.text[:300]}"
            )

    def move(self, src_rel_path: str, dst_rel_path: str, overwrite: bool = False) -> None:
        resp = self._client.request(
            "MOVE",
            self._url_for(src_rel_path),
            headers={
                "Destination": self._url_for(dst_rel_path),
                "Overwrite": "T" if overwrite else "F",
            },
        )
        if resp.status_code not in (201, 204):
            raise RuntimeError(
                f"MOVE {src_rel_path!r} -> {dst_rel_path!r} failed: "
                f"HTTP {resp.status_code} {resp.text[:300]}"
            )
