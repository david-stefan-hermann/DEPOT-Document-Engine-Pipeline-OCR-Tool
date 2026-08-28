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
