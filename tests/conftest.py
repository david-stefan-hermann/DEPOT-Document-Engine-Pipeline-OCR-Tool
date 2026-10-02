from urllib.parse import urlparse, unquote
from xml.sax.saxutils import escape

import httpx
import pytest

from depot.webdav import WebDavClient

BASE_URL = "https://nc.example.test/remote.php/dav/files/testuser"
BASE_PATH = urlparse(BASE_URL).path


def _parent(rel_path: str) -> str:
    return rel_path.rsplit("/", 1)[0] if "/" in rel_path else ""


class FakeNextcloud:
    """A minimal in-memory stand-in for a Nextcloud WebDAV endpoint, enough to
    exercise PROPFIND XML parsing, MKCOL idempotency, PUT/GET/DELETE and MOVE
    without needing a real server."""

    def __init__(self):
        self.collections: set[str] = set()
        self.files: dict[str, bytes] = {}
        self.tags: dict[str, str] = {}  # tag id -> name
        self.file_tags: dict[str, set[str]] = {}  # rel path -> tag names
        self._file_ids: dict[str, str] = {}
        self.tags_broken = False

    def _file_id(self, rel: str) -> str:
        return self._file_ids.setdefault(rel, str(1000 + len(self._file_ids)))

    def _handle_tags(self, request: httpx.Request) -> httpx.Response:
        if self.tags_broken:
            return httpx.Response(503)
        path = unquote(request.url.path)
        if path.rstrip("/").endswith("/systemtags"):
            if request.method == "PROPFIND":
                items = "".join(
                    f"<d:response><d:href>{path}/{tid}</d:href><d:propstat><d:prop><oc:id>{tid}</oc:id>"
                    f"<oc:display-name>{escape(name)}</oc:display-name></d:prop></d:propstat></d:response>"
                    for tid, name in self.tags.items()
                )
                body = f'<d:multistatus xmlns:d="DAV:" xmlns:oc="http://owncloud.org/ns">{items}</d:multistatus>'
                return httpx.Response(207, content=body.encode("utf-8"))
            if request.method == "POST":
                import json as _json

                name = _json.loads(request.content)["name"]
                if name in self.tags.values():
                    return httpx.Response(409)
                tid = str(len(self.tags) + 1)
                self.tags[tid] = name
                return httpx.Response(201, headers={"Content-Location": f"/remote.php/dav/systemtags/{tid}"})
        if "/systemtags-relations/files/" in path and request.method == "PUT":
            file_id, tag_id = path.rstrip("/").rsplit("/", 2)[-2:]
            rel = next((r for r, fid in self._file_ids.items() if fid == file_id), None)
            if rel is None or tag_id not in self.tags:
                return httpx.Response(404)
            names = self.file_tags.setdefault(rel, set())
            if self.tags[tag_id] in names:
                return httpx.Response(409)
            names.add(self.tags[tag_id])
            return httpx.Response(201)
        return httpx.Response(400)

    def _rel(self, url: httpx.URL) -> str:
        path = url.path
        if path.startswith(BASE_PATH):
            path = path[len(BASE_PATH):]
        return unquote(path).strip("/")

    def _multistatus(self, entries: list[tuple[str, bool]]) -> bytes:
        items = []
        for rel, is_collection in entries:
            href = f"{BASE_PATH}/{rel}" if rel else BASE_PATH
            restype = "<d:collection/>" if is_collection else ""
            items.append(
                f"<d:response><d:href>{escape(href)}</d:href><d:propstat>"
                f"<d:prop><d:resourcetype>{restype}</d:resourcetype></d:prop>"
                f"<d:status>HTTP/1.1 200 OK</d:status></d:propstat></d:response>"
            )
        body = "<d:multistatus xmlns:d=\"DAV:\">" + "".join(items) + "</d:multistatus>"
        return body.encode("utf-8")

    def handle(self, request: httpx.Request) -> httpx.Response:
        if "/systemtags" in request.url.path:
            return self._handle_tags(request)
        rel = self._rel(request.url)
        method = request.method

        if method == "PROPFIND":
            exists = rel == "" or rel in self.collections or rel in self.files
            if not exists:
                return httpx.Response(404)
            if b"fileid" in request.content:
                body = (
                    '<d:multistatus xmlns:d="DAV:" xmlns:oc="http://owncloud.org/ns"><d:response><d:propstat>'
                    f"<d:prop><oc:fileid>{self._file_id(rel)}</oc:fileid></d:prop></d:propstat></d:response>"
                    "</d:multistatus>"
                )
                return httpx.Response(207, content=body.encode("utf-8"))
            is_coll = rel == "" or rel in self.collections
            entries = [(rel, is_coll)]
            if request.headers.get("Depth") == "1" and is_coll:
                for c in self.collections:
                    if _parent(c) == rel:
                        entries.append((c, True))
                for f in self.files:
                    if _parent(f) == rel:
                        entries.append((f, False))
            return httpx.Response(207, content=self._multistatus(entries))

        if method == "MKCOL":
            if rel in self.collections:
                return httpx.Response(405)
            self.collections.add(rel)
            return httpx.Response(201)

        if method == "PUT":
            if request.headers.get("If-None-Match") == "*" and rel in self.files:
                return httpx.Response(412)
            self.files[rel] = request.content
            return httpx.Response(201)

        if method == "GET":
            if rel in self.files:
                return httpx.Response(200, content=self.files[rel])
            return httpx.Response(404)

        if method == "DELETE":
            self.files.pop(rel, None)
            self.collections.discard(rel)
            return httpx.Response(204)

        if method == "MOVE":
            dest_url = httpx.URL(request.headers["Destination"])
            dest_rel = self._rel(dest_url)
            if rel in self.files:
                self.files[dest_rel] = self.files.pop(rel)
            elif rel in self.collections:
                self.collections.discard(rel)
                self.collections.add(dest_rel)
            return httpx.Response(201)

        return httpx.Response(400)


@pytest.fixture(autouse=True)
def _no_model_preload(monkeypatch):
    """The pipeline warms up the Ollama model in a background thread for
    every document; tests have no Ollama server to warm up."""
    from depot import classifier

    monkeypatch.setattr(classifier, "preload_model", lambda *a, **k: None)


@pytest.fixture
def fake_server():
    return FakeNextcloud()


@pytest.fixture
def client(fake_server):
    transport = httpx.MockTransport(fake_server.handle)
    return WebDavClient(BASE_URL, "testuser", "app-password", transport=transport)
