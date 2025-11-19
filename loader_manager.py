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
    QSemaphore,
    QMetaObject,
    Q_ARG,
)
from PySide6.QtGui import QPixmap, QImage
from ui_components import THUMBNAIL_SIZE

import logging
import traceback  # Import traceback for safe logging

logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
THUMBNAIL_CACHE_SIZE = 2000
NUM_WORKERS = 8
MAX_PENDING_JOBS = 100  # New: Limit queue size


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
        """
        EXCEPTION SAFETY:
        This entire method is wrapped to prevent exceptions from propagating
        back to the QThreadPool during application shutdown.
        """
        try:
            while not self.manager._is_shutting_down:
                try:
                    filepath = self.manager.get_next_job()
                    if filepath:
                        # --- Actual Work Logic ---
                        # Check cache first
                        if thumbnail_cache.get(filepath):
                            self.manager.job_finished(filepath, None)
                            continue

                        # Load as QImage (Thread Safe)
                        image = QImage(filepath)
                        if not image.isNull():
                            scaled = image.scaled(
                                THUMBNAIL_SIZE,
                                THUMBNAIL_SIZE,
                                Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.SmoothTransformation,
                            )
                            self.manager.job_finished(filepath, scaled)
                        else:
                            # Even if loading fails, we mark job as finished
                            self.manager.job_finished(filepath, None)
                except Exception:
                    # Catch individual job failures so the worker keeps running
                    # unless it's a shutdown issue.
                    if self.manager._is_shutting_down:
                        break
                    logger.debug(f"Worker encountered error processing job:\n{traceback.format_exc()}")

        except (KeyboardInterrupt, SystemExit):
            # Allow standard python exit signals to pass cleanly
            pass
        except Exception:
            # Catch catastrophic failures (e.g. QImage bindings destroyed)
            # Log only if not shutting down, otherwise suppress to avoid noise
            if not self.manager._is_shutting_down:
                logger.error(f"PersistentWorker thread crashed:\n{traceback.format_exc()}")


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

            # --- NEW: Cap the queue size ---
            # If queue is full, drop the oldest item (the one at the bottom/right)
            # This ensures we only process what is currently on screen (LIFO)
            while len(self.queue) >= MAX_PENDING_JOBS:
                try:
                    discarded = self.queue.pop()  # Remove oldest
                    self.pending_jobs.discard(discarded)
                    # We removed an item but the semaphore count is still high.
                    # The worker will handle this empty slot gracefully.
                except IndexError:
                    break
            # -------------------------------

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

            # --- NEW: Safe pop ---
            try:
                # Take from the FRONT of the queue (LIFO).
                return self.queue.popleft()
            except IndexError:
                # This happens if we dropped items in request_thumbnail.
                # The semaphore let us in, but the queue is empty.
                # Return None so the worker simply loops again.
                return None
            # ---------------------

    def job_finished(self, filepath: str, image: QImage | None):
        """
        Called by a worker when it completes a job.
        NOTE: This method might be called from the worker thread context.
        We must be careful to handle QImage -> QPixmap conversion on the main thread.
        """
        if self._is_shutting_down:
            return

        # We need to handle the QImage -> QPixmap conversion carefully.
        # If this method runs in the worker thread, we can't create QPixmap here.
        # Strategy: Use QMetaObject.invokeMethod to push the result to the main thread

        if image:
            # Enqueue the conversion to run on the main thread
            QMetaObject.invokeMethod(
                self, "_handle_finished_job", Qt.QueuedConnection, Q_ARG(str, filepath), Q_ARG(QImage, image)
            )

        with QMutexLocker(self.mutex):
            self.pending_jobs.discard(filepath)
            # NO NEED FOR WAKE SIGNAL: The worker loops back immediately,
            # and if another job is available, it will acquire the semaphore.

    @Slot(str, QImage)
    def _handle_finished_job(self, filepath: str, image: QImage):
        if self._is_shutting_down:
            return
        try:
            pixmap = QPixmap.fromImage(image)
            thumbnail_cache.put(filepath, pixmap)
            self.thumbnail_loaded.emit(filepath)
        except Exception:
            # If QPixmap creation fails (e.g. during shutdown), ignore
            pass

    def shutdown(self):
        """Gracefully shuts down the loader."""
        logger.info("LoaderManager shutting down...")

        # 1. Set Flag
        with QMutexLocker(self.mutex):
            self._is_shutting_down = True
            self.queue.clear()
            self.pending_jobs.clear()

        # 2. Wake up all workers so they see the flag
        # We release enough semaphores for all workers + buffer
        self.semaphore.release(NUM_WORKERS * 2)

        # 3. Wait for threads
        logger.info("Waiting for thread pool to clear...")
        # Don't wait forever; if a thread is stuck on a bad C-call, we proceed to exit
        is_cleared = self.thread_pool.waitForDone(2000)

        if not is_cleared:
            logger.warning("Thread pool did not clear instantly (likely harmless during exit).")

        logger.info("LoaderManager shut down complete.")


# A single global instance of the loader manager
loader_manager = LoaderManager()
