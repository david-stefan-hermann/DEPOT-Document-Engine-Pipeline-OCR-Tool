import json

from depot import scan_config


def test_is_config_file_exact_match_only():
    assert scan_config.is_config_file("DEPOT Config.json", "DEPOT Config.json")
    assert not scan_config.is_config_file("scan1.pdf", "DEPOT Config.json")
    assert not scan_config.is_config_file("DEPOT Config.json.bak", "DEPOT Config.json")


def test_load_excluded_folders_missing_file_returns_empty(tmp_path):
    assert scan_config.load_excluded_folders(str(tmp_path), "Config", "DEPOT Config.json") == []


def test_load_excluded_folders_reads_list(tmp_path):
    config_dir = tmp_path / "Config"
    config_dir.mkdir()
    (config_dir / "DEPOT Config.json").write_text(
        json.dumps({"excluded_folders": ["Dokumente/Games", "Dokumente/Media/"]}),
        encoding="utf-8",
    )
    result = scan_config.load_excluded_folders(str(tmp_path), "Config", "DEPOT Config.json")
    assert result == ["Dokumente/Games", "Dokumente/Media"]


def test_load_excluded_folders_malformed_json_returns_empty(tmp_path):
    config_dir = tmp_path / "Config"
    config_dir.mkdir()
    (config_dir / "DEPOT Config.json").write_text("{not valid json", encoding="utf-8")
    assert scan_config.load_excluded_folders(str(tmp_path), "Config", "DEPOT Config.json") == []


def test_load_excluded_folders_wrong_type_returns_empty(tmp_path):
    config_dir = tmp_path / "Config"
    config_dir.mkdir()
    (config_dir / "DEPOT Config.json").write_text(
        json.dumps({"excluded_folders": "Dokumente/Games"}), encoding="utf-8"
    )
    assert scan_config.load_excluded_folders(str(tmp_path), "Config", "DEPOT Config.json") == []


def test_load_excluded_folders_missing_key_returns_empty(tmp_path):
    config_dir = tmp_path / "Config"
    config_dir.mkdir()
    (config_dir / "DEPOT Config.json").write_text(json.dumps({}), encoding="utf-8")
    assert scan_config.load_excluded_folders(str(tmp_path), "Config", "DEPOT Config.json") == []


def test_filter_excluded_removes_exact_and_nested():
    folders = [
        "Dokumente/Games",
        "Dokumente/Games/Amiibo-main",
        "Dokumente/Games/Amiibo-main/Amiibo NFC",
        "Dokumente/Gesundheit",
        "Dokumente/Gesundheit/Krankenkasse",
    ]
    result = scan_config.filter_excluded(folders, ["Dokumente/Games"])
    assert result == ["Dokumente/Gesundheit", "Dokumente/Gesundheit/Krankenkasse"]


def test_filter_excluded_does_not_remove_similarly_named_sibling():
    folders = ["Dokumente/Games", "Dokumente/Games2"]
    result = scan_config.filter_excluded(folders, ["Dokumente/Games"])
    assert result == ["Dokumente/Games2"]


def test_filter_excluded_no_exclusions_returns_same_list():
    folders = ["Dokumente/Gesundheit"]
    assert scan_config.filter_excluded(folders, []) == folders
