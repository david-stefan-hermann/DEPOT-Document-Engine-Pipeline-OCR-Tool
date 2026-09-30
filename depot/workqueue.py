from __future__ import annotations

import queue
from pathlib import Path


class WorkQueue(queue.Queue):
    """A queue of scan paths that holds each path at most once.

    The same file reaches the queue from several directions - watcher
    event, startup sweep, periodic sweep, transient-failure retry - and a
    path queued twice would be processed twice (or, if it fails, count a
    failure twice per attempt). Putting a path that is already waiting is
    a no-op; once taken out it may be put again."""

    def _init(self, maxsize: int) -> None:
        super()._init(maxsize)
        self._waiting: set[Path] = set()

    # _put/_get run under the queue's own mutex, so the set stays consistent
    # with the deque without a second lock. (put() itself can't be
    # overridden for this: it takes that mutex, which is not reentrant.)
    def _put(self, item: Path) -> None:
        if item in self._waiting:
            # put() adds one to unfinished_tasks right after this; nothing
            # was added, so take it back.
            self.unfinished_tasks -= 1
            return
        self._waiting.add(item)
        super()._put(item)

    def _get(self) -> Path:
        item = super()._get()
        self._waiting.discard(item)
        return item

    def is_waiting(self, item: Path) -> bool:
        with self.mutex:
            return item in self._waiting
