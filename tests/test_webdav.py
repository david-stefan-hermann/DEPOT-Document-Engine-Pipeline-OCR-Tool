def test_check_connection_ok(client):
    client.check_connection()  # should not raise


def test_list_dir_on_missing_folder_returns_empty(client):
    assert client.list_dir("Dokumente") == []


def test_mkcol_creates_missing_parents(client, fake_server):
    client.mkcol("Dokumente/Gesundheit/Krankenkasse")
    assert "Dokumente" in fake_server.collections
    assert "Dokumente/Gesundheit" in fake_server.collections
    assert "Dokumente/Gesundheit/Krankenkasse" in fake_server.collections


def test_mkcol_is_idempotent(client):
    client.mkcol("Dokumente/Motorrad")
    client.mkcol("Dokumente/Motorrad")  # must not raise on second call


def test_list_folders_recursive(client):
    client.mkcol("Dokumente/Gesundheit/Krankenkasse")
    client.mkcol("Dokumente/Motorrad/Rechnungen")

    folders = set(client.list_folders_recursive("Dokumente"))
    assert folders == {
        "Dokumente/Gesundheit",
        "Dokumente/Gesundheit/Krankenkasse",
        "Dokumente/Motorrad",
        "Dokumente/Motorrad/Rechnungen",
    }


def test_put_get_roundtrip(client):
    client.mkcol("Dokumente/Energie")
    client.put("Dokumente/Energie/2026-07-15 Stromrechnung.pdf", b"%PDF-fake-bytes")
    assert client.get("Dokumente/Energie/2026-07-15 Stromrechnung.pdf") == b"%PDF-fake-bytes"


def test_get_missing_file_returns_none(client):
    assert client.get("Dokumente/does-not-exist.pdf") is None


def test_delete_removes_file(client):
    client.mkcol("Scan-Eingang")
    client.put("Scan-Eingang/scan.pdf", b"data")
    client.delete("Scan-Eingang/scan.pdf")
    assert client.get("Scan-Eingang/scan.pdf") is None


def test_exists_true_and_false(client):
    client.mkcol("Dokumente/Unsortiert")
    assert client.exists("Dokumente/Unsortiert") is True
    assert client.exists("Dokumente/Nirgendwo") is False


def test_folder_names_with_umlauts_and_spaces(client):
    client.mkcol("Dokumente/Straßenverkehr/Bußgeldbescheide")
    client.put("Dokumente/Straßenverkehr/Bußgeldbescheide/2026-01-01 Bescheid.pdf", b"x")
    assert client.get("Dokumente/Straßenverkehr/Bußgeldbescheide/2026-01-01 Bescheid.pdf") == b"x"
    folders = client.list_folders_recursive("Dokumente")
    assert "Dokumente/Straßenverkehr/Bußgeldbescheide" in folders


def test_tag_file_creates_missing_tags_once_and_is_idempotent(fake_server, client):
    client.mkcol("Dokumente")
    client.put("Dokumente/a.pdf", b"x")
    client.put("Dokumente/b.pdf", b"y")

    client.tag_file("Dokumente/a.pdf", ["Depot", "Neu"])
    client.tag_file("Dokumente/a.pdf", ["Depot"])  # already tagged: no error
    client.tag_file("Dokumente/b.pdf", ["Depot", "Datum unsicher"])

    assert fake_server.file_tags == {
        "Dokumente/a.pdf": {"Depot", "Neu"},
        "Dokumente/b.pdf": {"Depot", "Datum unsicher"},
    }
    assert sorted(fake_server.tags.values()) == ["Datum unsicher", "Depot", "Neu"]


def test_tag_file_uses_a_tag_that_already_exists_on_the_server(fake_server, client):
    fake_server.tags["7"] = "Depot"
    client.put("a.pdf", b"x")
    client.tag_file("a.pdf", ["Depot"])
    assert list(fake_server.tags.items()) == [("7", "Depot")]
    assert fake_server.file_tags == {"a.pdf": {"Depot"}}


def test_tag_file_on_a_missing_file_raises(fake_server, client):
    import pytest

    with pytest.raises(RuntimeError, match="No file id"):
        client.tag_file("gone.pdf", ["Depot"])
