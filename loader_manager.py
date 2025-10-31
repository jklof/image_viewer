import collections
from PySide6.QtCore import (
    QObject,
    Qt,
    QRunnable,
    QThreadPool,
    Signal,
    Slot,
    QMutex,
    QMutexLocker,
    QWaitCondition,
)
from PySide6.QtGui import QPixmap
from ui_components import THUMBNAIL_SIZE

import logging

logger = logging.getLogger(__name__)

# --- Cache size large enough to hold many images ---
THUMBNAIL_CACHE_SIZE = 2000


class ThumbnailCache:
    """A simple thread-safe LRU cache for QPixmap objects."""

    def __init__(self, size):
        self.cache = collections.OrderedDict()
        self.size = size
        self.mutex = QMutex()

    def get(self, key):
        with QMutexLocker(self.mutex):
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
        return None

    def put(self, key, value):
        with QMutexLocker(self.mutex):
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.size:
                self.cache.popitem(last=False)


# Global cache instance
thumbnail_cache = ThumbnailCache(THUMBNAIL_CACHE_SIZE)


class Worker(QRunnable):
    """A persistent worker that pulls jobs from the LoaderManager's queue."""

    def __init__(self, manager: "LoaderManager"):
        super().__init__()
        self.manager = manager
        self.setAutoDelete(True)

    def run(self):
        """The main worker loop."""
        while True:
            filepath = self.manager.get_next_job()
            if filepath is None:
                # Sentinel value received, manager is shutting down.
                logger.info("Worker received shutdown signal.")
                break

            # --- Perform the slow I/O and processing ---
            pixmap = QPixmap(filepath)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(
                    THUMBNAIL_SIZE,
                    THUMBNAIL_SIZE,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.manager.job_finished(filepath, scaled_pixmap)
            else:
                self.manager.job_finished(filepath, None)


class LoaderManager(QObject):
    """
    Manages a thread pool with a single FIFO queue.
    Requests are deduplicated - if already queued or cached, they're ignored.
    """

    thumbnail_loaded = Signal(str)

    def __init__(self):
        super().__init__()
        self.thread_pool = QThreadPool()
        self.is_shutting_down = False

        # --- Single queue, FIFO order ---
        self.mutex = QMutex()
        self.wait_condition = QWaitCondition()
        self.queue = collections.deque()
        self.queued_paths = set()  # Track what's already queued to avoid duplicates

        # I/O bound tasks work better with 4-8 threads
        num_workers = 8  # self.thread_pool.maxThreadCount()
        for _ in range(num_workers):
            worker = Worker(self)
            self.thread_pool.start(worker)

    @Slot(str)
    def request_thumbnail(self, filepath: str):
        """Request a thumbnail. Ignores if already cached or queued."""
        with QMutexLocker(self.mutex):
            # Already cached? Nothing to do
            if thumbnail_cache.get(filepath):
                return

            # Already queued? Nothing to do
            if filepath in self.queued_paths or self.is_shutting_down:
                return

            # Add to queue
            self.queue.append(filepath)
            self.queued_paths.add(filepath)
            self.wait_condition.wakeOne()

    def get_next_job(self) -> str | None:
        """Called by workers to get the next available job."""
        with QMutexLocker(self.mutex):
            while not self.queue:
                if self.is_shutting_down:
                    return None
                self.wait_condition.wait(self.mutex)

            filepath = self.queue.popleft()
            self.queued_paths.discard(filepath)
            return filepath

    def job_finished(self, filepath: str, pixmap: QPixmap | None):
        """Called by workers when a job completes."""
        if pixmap:
            thumbnail_cache.put(filepath, pixmap)
            self.thumbnail_loaded.emit(filepath)

    def shutdown(self):
        """Signals workers to terminate gracefully."""
        with QMutexLocker(self.mutex):
            self.is_shutting_down = True
            self.wait_condition.wakeAll()
        self.thread_pool.waitForDone()


# A single global instance of the loader manager
loader_manager = LoaderManager()
