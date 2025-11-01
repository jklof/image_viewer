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
    QSemaphore,  # NEW IMPORT
)
from PySide6.QtGui import QPixmap, QImage
from ui_components import THUMBNAIL_SIZE

import logging

logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
THUMBNAIL_CACHE_SIZE = 2000
NUM_WORKERS = 8


class ThumbnailCache:
    """A simple thread-safe LRU cache for QPixmap objects."""

    def __init__(self, size):
        self.cache = collections.OrderedDict()
        self.size = size
        self.lock = QMutex()

    def get(self, key):
        with QMutexLocker(self.lock):
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
        return None

    def put(self, key, value):
        with QMutexLocker(self.lock):
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.size:
                self.cache.popitem(last=False)


# Global cache instance
thumbnail_cache = ThumbnailCache(THUMBNAIL_CACHE_SIZE)


class PersistentWorker(QRunnable):
    """A persistent worker that continuously pulls jobs from the manager."""

    def __init__(self, manager: "LoaderManager"):
        super().__init__()
        self.manager = manager
        # This worker should not be deleted by the thread pool when it's done
        self.setAutoDelete(False)

    def run(self):
        # We check for the shutdown flag after a small timeout in get_next_job
        # or when we are finally woken up during the shutdown sequence.
        while not self.manager._is_shutting_down:
            filepath = self.manager.get_next_job()
            if filepath:
                # --- Actual Work Logic ---
                if thumbnail_cache.get(filepath):
                    self.manager.job_finished(filepath, None)  # Still mark as finished
                    continue

                image = QImage(filepath)
                pixmap = None
                if not image.isNull():
                    scaled = image.scaled(
                        THUMBNAIL_SIZE,
                        THUMBNAIL_SIZE,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                    pixmap = QPixmap.fromImage(scaled)
                else:
                    logger.warning(f"Failed to load image: {filepath}")

                self.manager.job_finished(filepath, pixmap)


class LoaderManager(QObject):
    """
    Manages a thread pool using QSemaphore for job synchronization,
    ensuring a simple, race-free producer/consumer pattern.
    """

    thumbnail_loaded = Signal(str)

    def __init__(self):
        super().__init__()
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(NUM_WORKERS)
        self._is_shutting_down = False

        self.mutex = QMutex()
        self.queue = collections.deque()
        self.pending_jobs = set()

        # NEW: A semaphore acts as a counter for available jobs (starts at 0)
        self.semaphore = QSemaphore(0)

        # Start the persistent workers
        for _ in range(NUM_WORKERS):
            worker = PersistentWorker(self)
            self.thread_pool.start(worker)

    @Slot(str)
    def request_thumbnail(self, filepath: str):
        """Request a thumbnail. Fast and non-blocking."""
        if self._is_shutting_down:
            return

        if thumbnail_cache.get(filepath):
            return

        # --- Producer Logic: Add job and release semaphore ---
        with QMutexLocker(self.mutex):
            # Check again inside lock for pending status
            if filepath in self.pending_jobs:
                return

            self.pending_jobs.add(filepath)
            # Add to the FRONT of the queue (LIFO priority).
            self.queue.appendleft(filepath)

        # Signal that one more job is available. This unblocks a waiting worker.
        self.semaphore.release(1)

    def get_next_job(self) -> str | None:
        """
        Called by workers. Blocks until a job is available, or checks
        for shutdown after a brief timeout.
        """
        # Consumer Logic: Block until a job is available (semaphore counter > 0)
        # Use a timeout (50ms) to check the shutdown flag periodically
        if not self.semaphore.tryAcquire(1, 50):
            with QMutexLocker(self.mutex):
                # If we timed out on acquire, check the shutdown flag and exit loop
                if self._is_shutting_down:
                    return None
            return None  # Timed out, worker will loop and try acquiring again

        # If acquire succeeded, a job is guaranteed to be in the queue.
        with QMutexLocker(self.mutex):
            # We check shutdown again, though highly unlikely after a successful acquire
            if self._is_shutting_down:
                # If we acquired but are shutting down, release the resource and exit
                self.semaphore.release(1)
                return None

            # Take from the FRONT of the queue (LIFO).
            return self.queue.popleft()

    def job_finished(self, filepath: str, pixmap: QPixmap | None):
        """Called by a worker when it completes a job."""
        if self._is_shutting_down:
            return

        if pixmap:
            thumbnail_cache.put(filepath, pixmap)
            self.thumbnail_loaded.emit(filepath)

        with QMutexLocker(self.mutex):
            self.pending_jobs.discard(filepath)
            # NO NEED FOR WAKE SIGNAL: The worker loops back immediately,
            # and if another job is available, it will acquire the semaphore.

    def shutdown(self):
        """Gracefully shuts down the loader."""
        logger.info("LoaderManager shutting down...")
        with QMutexLocker(self.mutex):
            self._is_shutting_down = True
            self.queue.clear()
            # Release the semaphore multiple times to ensure all NUM_WORKERS
            # threads (and any blocked on acquiring) wake up and see the flag.
            self.semaphore.release(NUM_WORKERS)
        self.thread_pool.waitForDone(5000)
        logger.info("LoaderManager shut down complete.")


# A single global instance of the loader manager
loader_manager = LoaderManager()
