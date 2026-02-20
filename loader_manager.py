import collections
import logging
import threading

from PySide6.QtCore import (
    QObject,
    Qt,
    QRunnable,
    QThreadPool,
    Signal,
    Slot,
    QThread,
)
from PySide6.QtGui import QPixmap, QImage, QColor
from ui_components import THUMBNAIL_SIZE

logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
THUMBNAIL_CACHE_SIZE = 2000
# High worker count is safe here; Condition variables prevent CPU busy loops
NUM_WORKERS = max(2, QThread.idealThreadCount() - 1)
MAX_PENDING_JOBS = 100


class ThumbnailCache:
    """Thread-safe cache optimized for LOCKLESS reads on the UI Thread."""

    def __init__(self, size):
        self.cache = {}
        self.cleanup_queue = collections.deque()
        self.size = size
        self.lock = threading.Lock()

    def get(self, key):
        # FAST PATH: Dictionary `.get()` is atomic in Python.
        # No lock required! The UI Thread never waits for background workers here.
        return self.cache.get(key)

    def put(self, key, value):
        with self.lock:
            if key in self.cache:
                return

            self.cache[key] = value
            self.cleanup_queue.append(key)

            # Simple FIFO cleanup is perfectly fine for thumbnails
            # and avoids mutating the dictionary on reads.
            if len(self.cache) > self.size:
                oldest_key = self.cleanup_queue.popleft()
                self.cache.pop(oldest_key, None)


thumbnail_cache = ThumbnailCache(THUMBNAIL_CACHE_SIZE)


class PersistentWorker(QRunnable):
    def __init__(self, manager: "LoaderManager"):
        super().__init__()
        self.manager = manager
        self.setAutoDelete(False)

    def run(self):
        try:
            while not self.manager._is_shutting_down:
                filepath = self.manager.get_next_job()
                if not filepath:
                    continue

                try:
                    # 1. Fast path cache check
                    if thumbnail_cache.get(filepath):
                        continue

                    # 2. Reverted to QImage: It is significantly faster for
                    # rapid I/O decoding than QImageReader on most systems.
                    image = QImage(filepath)

                    if not image.isNull():
                        # 3. Scale on the background thread
                        if image.width() > THUMBNAIL_SIZE or image.height() > THUMBNAIL_SIZE:
                            image = image.scaled(
                                THUMBNAIL_SIZE,
                                THUMBNAIL_SIZE,
                                Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.SmoothTransformation,
                            )

                        # 4. Convert to optimal GPU texture format on the background thread!
                        # This turns QPixmap.fromImage() on the Main Thread into an instant O(1) memory copy.
                        image = image.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)

                        self.manager._internal_worker_result.emit(filepath, image)
                    else:
                        self.manager._internal_worker_result.emit(filepath, QImage())

                except Exception:
                    pass

        except Exception:
            pass


class LoaderManager(QObject):
    thumbnail_loaded = Signal(str)
    _internal_worker_result = Signal(str, QImage)

    def __init__(self):
        super().__init__()
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(NUM_WORKERS)
        self._is_shutting_down = False

        # --- ROCK SOLID SYNCHRONIZATION ---
        # Replacing the fragile Semaphore with a Condition variable.
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.queue = collections.deque()
        self.pending_jobs = set()

        self._internal_worker_result.connect(self._on_worker_result_ready)

    def initialize(self):
        if self._is_shutting_down:
            return
        logger.info("LoaderManager initializing workers...")
        for _ in range(NUM_WORKERS):
            worker = PersistentWorker(self)
            self.thread_pool.start(worker)

    @Slot(str)
    def request_thumbnail(self, filepath: str):
        if self._is_shutting_down:
            return
        if thumbnail_cache.get(filepath):
            return

        with self.condition:
            if filepath in self.pending_jobs:
                return

            # Enforce max size (LIFO drop oldest job from the right)
            while len(self.queue) >= MAX_PENDING_JOBS:
                try:
                    discarded = self.queue.pop()
                    self.pending_jobs.discard(discarded)
                except IndexError:
                    break

            self.pending_jobs.add(filepath)
            self.queue.appendleft(filepath)  # Add newest to the left

            # Wake up exactly one sleeping worker
            self.condition.notify()

    def get_next_job(self) -> str | None:
        with self.condition:
            # Sleep perfectly without spinning the CPU while the queue is empty
            while len(self.queue) == 0 and not self._is_shutting_down:
                self.condition.wait(0.5)

            if self._is_shutting_down or len(self.queue) == 0:
                return None

            # Return newest job
            return self.queue.popleft()

    @Slot(str, QImage)
    def _on_worker_result_ready(self, filepath: str, image: QImage):
        if self._is_shutting_down:
            return

        with self.lock:
            self.pending_jobs.discard(filepath)

        if not image.isNull():
            pixmap = QPixmap.fromImage(image)
            thumbnail_cache.put(filepath, pixmap)
            self.thumbnail_loaded.emit(filepath)
        else:
            error_pixmap = QPixmap(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
            error_pixmap.fill(QColor(30, 30, 30))
            thumbnail_cache.put(filepath, error_pixmap)
            self.thumbnail_loaded.emit(filepath)

    def shutdown(self):
        self._is_shutting_down = True
        with self.condition:
            self.queue.clear()
            self.pending_jobs.clear()
            self.condition.notify_all()  # Wake all threads so they exit cleanly
        self.thread_pool.waitForDone(1500)


loader_manager = LoaderManager()