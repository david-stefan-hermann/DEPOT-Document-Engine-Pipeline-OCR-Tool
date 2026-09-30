from pathlib import Path

from depot.folder_index import list_local_folders, local_mount_root


def test_mount_root_is_derived_from_the_two_inbox_paths():
    assert local_mount_root("/nextcloud-data/Dokumente/Scan Eingang", "Dokumente/Scan Eingang") == Path(
        "/nextcloud-data"
    )
    assert local_mount_root("/nextcloud-data/Dokumente/Scan Eingang/", "/Dokumente/Scan Eingang/") == Path(
        "/nextcloud-data"
    )


def test_mount_root_is_none_when_the_paths_do_not_line_up():
    # only the inbox itself is mounted, under a different name
    assert local_mount_root("/scans", "Dokumente/Scan Eingang") is None


def test_list_local_folders_matches_webdav_style_paths(tmp_path):
    (tmp_path / "Dokumente" / "Gesundheit" / "Krankenkasse").mkdir(parents=True)
    (tmp_path / "Dokumente" / "Motorrad").mkdir()
    (tmp_path / "Dokumente" / "Motorrad" / "rechnung.pdf").write_bytes(b"x")
    (tmp_path / "Anderes").mkdir()

    assert sorted(list_local_folders(tmp_path, "Dokumente")) == [
        "Dokumente/Gesundheit",
        "Dokumente/Gesundheit/Krankenkasse",
        "Dokumente/Motorrad",
    ]


def test_list_local_folders_returns_none_when_root_is_missing(tmp_path):
    assert list_local_folders(tmp_path, "Dokumente") is None
