import threading
from pathlib import Path

from depot.workqueue import WorkQueue


def test_a_waiting_path_is_not_queued_twice():
    q = WorkQueue()
    a, b = Path("a.pdf"), Path("b.pdf")
    q.put(a)
    q.put(b)
    q.put(a)  # already waiting

    assert q.qsize() == 2
    assert q.is_waiting(a) is True
    assert q.get() == a
    assert q.is_waiting(a) is False
    assert q.get() == b
    assert q.empty()


def test_a_path_can_be_queued_again_once_taken_out():
    q = WorkQueue()
    a = Path("a.pdf")
    q.put(a)
    assert q.get() == a
    q.put(a)
    assert q.get() == a


def test_join_does_not_wait_for_the_ignored_put():
    q = WorkQueue()
    a = Path("a.pdf")
    q.put(a)
    q.put(a)
    q.get()
    q.task_done()

    finished = threading.Event()
    threading.Thread(target=lambda: (q.join(), finished.set()), daemon=True).start()
    assert finished.wait(timeout=2.0), "join() hung: the ignored put was still counted as unfinished"
