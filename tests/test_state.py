from depot.state import StateStore, MAX_PERMANENT_FAILURES


def test_increment_failure_counts_up(tmp_path):
    store = StateStore(str(tmp_path / "state.sqlite3"))
    assert store.increment_failure("scan.pdf") == 1
    assert store.increment_failure("scan.pdf") == 2
    store.close()


def test_should_quarantine_after_threshold(tmp_path):
    store = StateStore(str(tmp_path / "state.sqlite3"))
    for _ in range(MAX_PERMANENT_FAILURES - 1):
        store.increment_failure("scan.pdf")
    assert store.should_quarantine("scan.pdf") is False
    store.increment_failure("scan.pdf")
    assert store.should_quarantine("scan.pdf") is True
    store.close()


def test_reset_clears_failure_count(tmp_path):
    store = StateStore(str(tmp_path / "state.sqlite3"))
    store.increment_failure("scan.pdf")
    store.increment_failure("scan.pdf")
    store.reset("scan.pdf")
    assert store.should_quarantine("scan.pdf") is False
    assert store.increment_failure("scan.pdf") == 1
    store.close()


def test_persists_across_reconnect(tmp_path):
    db_path = str(tmp_path / "state.sqlite3")
    store1 = StateStore(db_path)
    store1.increment_failure("scan.pdf")
    store1.close()

    store2 = StateStore(db_path)
    assert store2.increment_failure("scan.pdf") == 2
    store2.close()
