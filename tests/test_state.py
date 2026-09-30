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


def test_processed_hash_is_remembered_across_reconnect(tmp_path):
    db_path = str(tmp_path / "state.sqlite3")
    store1 = StateStore(db_path)
    assert store1.find_processed("abc") is None
    store1.record_processed("abc", "Dokumente/A/x.pdf")
    store1.close()

    store2 = StateStore(db_path)
    assert store2.find_processed("abc") == "Dokumente/A/x.pdf"
    store2.record_processed("abc", "Dokumente/B/x.pdf")
    assert store2.find_processed("abc") == "Dokumente/B/x.pdf"
    store2.close()


def test_source_pending_delete_until_marked(tmp_path):
    store = StateStore(str(tmp_path / "state.sqlite3"))
    assert store.source_pending_delete("abc") is False  # unknown hash
    store.record_processed("abc", "Dokumente/A/x.pdf", source_deleted=False)
    assert store.source_pending_delete("abc") is True
    store.mark_source_deleted("abc")
    assert store.source_pending_delete("abc") is False
    # the default records a completed filing
    store.record_processed("def", "Dokumente/A/y.pdf")
    assert store.source_pending_delete("def") is False
    store.close()


def test_opens_a_database_from_before_the_source_deleted_column(tmp_path):
    import sqlite3

    db_path = str(tmp_path / "state.sqlite3")
    conn = sqlite3.connect(db_path)
    conn.executescript(
        "CREATE TABLE failures (filename TEXT PRIMARY KEY, count INTEGER NOT NULL DEFAULT 0);"
        "CREATE TABLE processed (sha256 TEXT PRIMARY KEY, dest_path TEXT NOT NULL);"
        "INSERT INTO processed VALUES ('abc', 'Dokumente/A/x.pdf');"
    )
    conn.commit()
    conn.close()

    store = StateStore(db_path)
    assert store.find_processed("abc") == "Dokumente/A/x.pdf"
    assert store.source_pending_delete("abc") is False  # old rows count as completed
    store.record_processed("def", "Dokumente/B/y.pdf", source_deleted=False)
    assert store.source_pending_delete("def") is True
    store.close()
    StateStore(db_path).close()  # applying the migration twice is harmless
