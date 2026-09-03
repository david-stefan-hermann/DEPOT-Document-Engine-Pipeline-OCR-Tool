from depot.config import Config


def _set_required_env(monkeypatch, tmp_path):
    monkeypatch.setenv("NEXTCLOUD_WEBDAV_URL", "https://nc.example.test/remote.php/dav/files/testuser")
    monkeypatch.setenv("NEXTCLOUD_USER", "testuser")
    monkeypatch.setenv("NEXTCLOUD_APP_PASSWORD", "app-password")
    monkeypatch.setenv("SCAN_EINGANG_LOCAL_PATH", str(tmp_path))


def test_from_env_defaults(monkeypatch, tmp_path):
    _set_required_env(monkeypatch, tmp_path)
    # A real .env may exist in the repo root (gitignored, holds a real
    # secret for manual local testing) - explicitly clear so this test's
    # "unset means None" assertion doesn't depend on the developer's local
    # filesystem state, and so a real key never ends up compared/asserted on.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config = Config.from_env()
    assert config.scan_eingang_webdav_path == "Dokumente/Scan Eingang"
    assert config.config_subfolder == "Depot Config"
    assert config.processed_subfolder == "Processed"
    # error_folder defaults inside the scan inbox's config subfolder, not
    # under Dokumente/, so it's never offered to the classifier as a target.
    assert config.error_folder == "Dokumente/Scan Eingang/Depot Config/_Fehlerhaft"
    assert config.anthropic_api_key is None
    assert config.anthropic_model == "claude-haiku-4-5"


def test_from_env_reads_anthropic_api_key(monkeypatch, tmp_path):
    _set_required_env(monkeypatch, tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-value")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-5")
    config = Config.from_env()
    assert config.anthropic_api_key == "sk-ant-test-value"
    assert config.anthropic_model == "claude-sonnet-5"


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
