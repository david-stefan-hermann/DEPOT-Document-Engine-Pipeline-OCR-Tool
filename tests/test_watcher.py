import queue

import pytest

from depot.watcher import ScanWatcher


def _make_watcher(path):
    return ScanWatcher(
        local_path=str(path),
        supported_extensions=frozenset({".pdf"}),
        log_file_prefix="DEPOT Dateilog",
        config_file_name="DEPOT Config.json",
        out_queue=queue.Queue(),
    )


def test_start_raises_clear_error_when_path_missing(tmp_path):
    missing = tmp_path / "does-not-exist"
    watcher = _make_watcher(missing)

    with pytest.raises(RuntimeError, match="does not exist"):
        watcher.start()


def test_start_raises_clear_error_when_path_is_a_file(tmp_path):
    a_file = tmp_path / "not-a-directory"
    a_file.write_text("oops")
    watcher = _make_watcher(a_file)

    with pytest.raises(RuntimeError, match="does not exist"):
        watcher.start()


def test_startup_sweep_warns_but_does_not_raise_when_path_missing(tmp_path, caplog):
    missing = tmp_path / "does-not-exist"
    watcher = _make_watcher(missing)

    watcher.startup_sweep()  # must not raise


def test_config_file_is_ignored_by_startup_sweep(tmp_path):
    (tmp_path / "DEPOT Config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "scan1.pdf").write_bytes(b"%PDF-1.4")
    watcher = _make_watcher(tmp_path)

    watcher.startup_sweep()

    queued = [watcher._out_queue.get_nowait().name for _ in range(watcher._out_queue.qsize())]
    assert queued == ["scan1.pdf"]
