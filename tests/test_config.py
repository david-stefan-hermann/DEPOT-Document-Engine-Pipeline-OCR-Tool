import pytest

from depot.config import Config


def _set_required_env(monkeypatch, tmp_path):
    monkeypatch.setenv("NEXTCLOUD_WEBDAV_URL", "https://nc.example.test/remote.php/dav/files/testuser")
    monkeypatch.setenv("NEXTCLOUD_USER", "testuser")
    monkeypatch.setenv("NEXTCLOUD_APP_PASSWORD", "app-password")
    monkeypatch.setenv("SCAN_EINGANG_LOCAL_PATH", str(tmp_path))


def test_from_env_defaults(monkeypatch, tmp_path):
    _set_required_env(monkeypatch, tmp_path)
    config = Config.from_env()
    assert config.scan_eingang_webdav_path == "Dokumente/Scan Eingang"
    assert config.config_subfolder == "Depot Config"
    assert config.processed_subfolder == "Processed"
    assert config.file_into_dokumente is True
    assert config.save_processed_copy is False
    # error_folder defaults inside the scan inbox's config subfolder, not
    # under Dokumente/, so it's never offered to the classifier as a target.
    assert config.error_folder == "Dokumente/Scan Eingang/Depot Config/_Fehlerhaft"


def test_from_env_error_folder_follows_custom_scan_eingang_and_config_subfolder(monkeypatch, tmp_path):
    _set_required_env(monkeypatch, tmp_path)
    monkeypatch.setenv("SCAN_EINGANG_WEBDAV_PATH", "Custom-Eingang")
    monkeypatch.setenv("CONFIG_SUBFOLDER", "Custom Config")
    config = Config.from_env()
    assert config.error_folder == "Custom-Eingang/Custom Config/_Fehlerhaft"


def test_from_env_error_folder_explicit_override_wins(monkeypatch, tmp_path):
    _set_required_env(monkeypatch, tmp_path)
    monkeypatch.setenv("ERROR_FOLDER", "Dokumente/_Fehlerhaft")
    config = Config.from_env()
    assert config.error_folder == "Dokumente/_Fehlerhaft"


def test_from_env_bool_flags_parsed(monkeypatch, tmp_path):
    _set_required_env(monkeypatch, tmp_path)
    monkeypatch.setenv("SAVE_PROCESSED_COPY", "true")
    monkeypatch.setenv("FILE_INTO_DOKUMENTE", "false")
    config = Config.from_env()
    assert config.file_into_dokumente is False
    assert config.save_processed_copy is True


def test_from_env_rejects_both_filing_switches_disabled(monkeypatch, tmp_path):
    _set_required_env(monkeypatch, tmp_path)
    monkeypatch.setenv("FILE_INTO_DOKUMENTE", "false")
    monkeypatch.setenv("SAVE_PROCESSED_COPY", "false")
    with pytest.raises(RuntimeError, match="nowhere to put"):
        Config.from_env()
