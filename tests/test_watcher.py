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


# ---- periodic sweep --------------------------------------------------------

def _queued_names(watcher):
    return [watcher._out_queue.get_nowait().name for _ in range(watcher._out_queue.qsize())]


def test_sweep_queues_only_files_that_have_settled(tmp_path):
    import os
    import time

    old = tmp_path / "old.pdf"
    old.write_bytes(b"%PDF-1.4")
    os.utime(old, (time.time() - 300, time.time() - 300))
    (tmp_path / "fresh.pdf").write_bytes(b"%PDF-1.4")  # just written - maybe still being written
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "DEPOT Config.json").write_text("{}", encoding="utf-8")
    watcher = _make_watcher(tmp_path)

    found = watcher.sweep(min_age_seconds=60)

    assert [p.name for p in found] == ["old.pdf"]
    assert _queued_names(watcher) == ["old.pdf"]


def test_sweep_skips_a_file_the_watcher_is_still_debouncing(tmp_path):
    import os
    import time

    settled = tmp_path / "settled.pdf"
    settled.write_bytes(b"%PDF-1.4")
    os.utime(settled, (time.time() - 300, time.time() - 300))
    watcher = _make_watcher(tmp_path)
    watcher._pending.add(settled)

    assert watcher.sweep(min_age_seconds=60) == []


def test_sweep_does_not_queue_a_file_that_is_already_waiting(tmp_path):
    import os
    import time

    from depot.workqueue import WorkQueue

    scan = tmp_path / "scan.pdf"
    scan.write_bytes(b"%PDF-1.4")
    os.utime(scan, (time.time() - 300, time.time() - 300))
    watcher = _make_watcher(tmp_path)
    watcher._out_queue = WorkQueue()

    watcher.sweep(min_age_seconds=0)
    watcher.sweep(min_age_seconds=0)

    assert _queued_names(watcher) == ["scan.pdf"]


def test_sweep_on_a_missing_inbox_is_a_noop(tmp_path):
    watcher = _make_watcher(tmp_path / "gone")
    assert watcher.sweep() == []


def test_periodic_sweep_is_disabled_for_a_non_positive_interval(tmp_path):
    watcher = _make_watcher(tmp_path)
    assert watcher.start_periodic_sweep(0) is None
    assert watcher.start_periodic_sweep(-1) is None
