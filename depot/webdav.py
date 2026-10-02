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


_OC_NS = "{http://owncloud.org/ns}"

_FILEID_BODY = b"""<?xml version="1.0" encoding="utf-8"?>
<d:propfind xmlns:d="DAV:" xmlns:oc="http://owncloud.org/ns">
  <d:prop><oc:fileid/></d:prop>
</d:propfind>
"""

_TAGS_BODY = b"""<?xml version="1.0" encoding="utf-8"?>
<d:propfind xmlns:d="DAV:" xmlns:oc="http://owncloud.org/ns">
  <d:prop><oc:id/><oc:display-name/></d:prop>
</d:propfind>
"""


class PreconditionFailed(RuntimeError):
    """A conditional request (PUT with If-None-Match: *) was refused because
    the target already exists."""


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
        self._tag_cache: dict[str, str] = {}

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

    def put(self, rel_path: str, data: bytes, overwrite: bool = True) -> None:
        """Upload a file. With overwrite=False the server refuses to replace
        an existing file (If-None-Match: *) and PreconditionFailed is
        raised instead - the guard against two writers picking the same
        name at the same moment."""
        headers = {} if overwrite else {"If-None-Match": "*"}
        resp = self._client.put(self._url_for(rel_path), content=data, headers=headers)
        if resp.status_code == 412 and not overwrite:
            raise PreconditionFailed(f"PUT {rel_path!r}: a file with that name already exists")
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

    # ---- Nextcloud system tags ("Tags" in the Files app) -------------------
    # Not part of the files WebDAV tree: tags live under <dav>/systemtags and
    # are attached to a file by its numeric id under
    # <dav>/systemtags-relations/files/<fileid>/<tagid>.

    def _dav_url(self, path: str) -> str:
        root, sep, _ = self._base_url.partition("/remote.php/dav")
        if not sep:
            raise RuntimeError("Tags need a Nextcloud WebDAV URL (…/remote.php/dav/files/<user>)")
        return f"{root}/remote.php/dav/{path.lstrip('/')}"

    def file_id(self, rel_path: str) -> str | None:
        resp = self._client.request(
            "PROPFIND", self._url_for(rel_path), headers={"Depth": "0"}, content=_FILEID_BODY
        )
        if resp.status_code >= 300:
            return None
        return ET.fromstring(resp.content).findtext(f".//{_OC_NS}fileid") or None

    def _tag_ids(self) -> dict[str, str]:
        resp = self._client.request(
            "PROPFIND", self._dav_url("systemtags"), headers={"Depth": "1"}, content=_TAGS_BODY
        )
        if resp.status_code >= 300:
            raise RuntimeError(f"Listing tags failed: HTTP {resp.status_code} {resp.text[:300]}")
        tags: dict[str, str] = {}
        for response in ET.fromstring(resp.content).findall(f"{_DAV_NS}response"):
            tag_id = response.findtext(f".//{_OC_NS}id")
            name = response.findtext(f".//{_OC_NS}display-name")
            if tag_id and name:
                tags[name] = tag_id
        return tags

    def ensure_tag(self, name: str) -> str:
        """The id of the tag called `name`, creating the tag if needed."""
        if name not in self._tag_cache:
            self._tag_cache = self._tag_ids()
        if name not in self._tag_cache:
            resp = self._client.post(
                self._dav_url("systemtags"),
                json={"name": name, "userVisible": True, "userAssignable": True},
            )
            if resp.status_code == 201 and resp.headers.get("Content-Location"):
                self._tag_cache[name] = resp.headers["Content-Location"].rstrip("/").rsplit("/", 1)[-1]
            elif resp.status_code == 409:  # created by someone else just now
                self._tag_cache = self._tag_ids()
            else:
                raise RuntimeError(f"Creating tag {name!r} failed: HTTP {resp.status_code} {resp.text[:300]}")
        return self._tag_cache[name]

    def tag_file(self, rel_path: str, names: list[str]) -> None:
        """Attach the tags `names` to the file (creating missing tags)."""
        file_id = self.file_id(rel_path)
        if file_id is None:
            raise RuntimeError(f"No file id for {rel_path!r}; cannot tag it")
        for name in names:
            tag_id = self.ensure_tag(name)
            resp = self._client.put(self._dav_url(f"systemtags-relations/files/{file_id}/{tag_id}"))
            if resp.status_code not in (201, 204, 409):  # 409 = already tagged
                raise RuntimeError(f"Tagging {rel_path!r} with {name!r} failed: HTTP {resp.status_code}")

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
