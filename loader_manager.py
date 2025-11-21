import collections
import logging

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
    QThread,
)
from PySide6.QtGui import QPixmap, QImage, QColor
from ui_components import THUMBNAIL_SIZE

logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
THUMBNAIL_CACHE_SIZE = 2000
# Leave 1 core free for the Main Thread/OS to keep mouse movement smooth
NUM_WORKERS = max(2, QThread.idealThreadCount() - 1)
MAX_PENDING_JOBS = 100


class ThumbnailCache:
    """Thread-safe LRU cache."""

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
                    # 1. Check cache (Fast path)
                    if thumbnail_cache.get(filepath):
                        continue

                    # 2. Load Image (Blocking IO, but safe in thread)
                    # QImage(path) is robust and avoids Windows file-locking contention
                    image = QImage(filepath)

                    if not image.isNull():
                        # 3. Scale HIGH QUALITY (Worker Thread CPU)
                        if image.width() > THUMBNAIL_SIZE or image.height() > THUMBNAIL_SIZE:
                            image = image.scaled(
                                THUMBNAIL_SIZE,
                                THUMBNAIL_SIZE,
                                Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.SmoothTransformation,
                            )

                        # 4. Send to Manager via Signal
                        # Qt handles the thread-safe handover to the main thread
                        self.manager._internal_worker_result.emit(filepath, image)
                    else:
                        # Handle corrupt file result
                        self.manager._internal_worker_result.emit(filepath, QImage())

                except Exception:
                    pass  # Skip corrupt files silently

        except Exception:
            pass


class LoaderManager(QObject):
    # Public signal for UI
    thumbnail_loaded = Signal(str)

    # Internal signal to bridge Worker Thread -> Main Thread
    _internal_worker_result = Signal(str, QImage)

    def __init__(self):
        super().__init__()
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(NUM_WORKERS)
        self._is_shutting_down = False

        self.mutex = QMutex()
        self.queue = collections.deque()
        self.pending_jobs = set()
        self.semaphore = QSemaphore(0)

        # Connect the worker signal to the main thread slot
        self._internal_worker_result.connect(self._on_worker_result_ready)

    def initialize(self):
        """Starts the worker threads."""
        if self._is_shutting_down:
            return

        logger.info("LoaderManager initializing workers...")
        for _ in range(NUM_WORKERS):
            worker = PersistentWorker(self)
            self.thread_pool.start(worker)

    @Slot(str)
    def request_thumbnail(self, filepath: str):
        """Called by UI (Main Thread)."""
        if self._is_shutting_down:
            return
        if thumbnail_cache.get(filepath):
            return

        with QMutexLocker(self.mutex):
            if filepath in self.pending_jobs:
                return

            # LIFO Logic: Drop old requests if queue is full
            # This ensures the user sees what they are looking at NOW.
            while len(self.queue) >= MAX_PENDING_JOBS:
                try:
                    discarded = self.queue.pop()
                    self.pending_jobs.discard(discarded)
                except IndexError:
                    break

            self.pending_jobs.add(filepath)
            self.queue.appendleft(filepath)

        self.semaphore.release(1)

    def get_next_job(self) -> str | None:
        """Called by Worker (Worker Thread)."""
        if not self.semaphore.tryAcquire(1, 50):
            return None

        with QMutexLocker(self.mutex):
            if self._is_shutting_down:
                return None
            try:
                return self.queue.popleft()
            except IndexError:
                return None

    @Slot(str, QImage)
    def _on_worker_result_ready(self, filepath: str, image: QImage):
        """
        Runs on MAIN THREAD.
        """
        if self._is_shutting_down:
            return

        # Cleanup pending set
        with QMutexLocker(self.mutex):
            self.pending_jobs.discard(filepath)

        if not image.isNull():
            # Texture upload (fast)
            pixmap = QPixmap.fromImage(image)
            thumbnail_cache.put(filepath, pixmap)
            self.thumbnail_loaded.emit(filepath)
        else:
            # Placeholder for corrupt images
            error_pixmap = QPixmap(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
            error_pixmap.fill(QColor(30, 30, 30))
            thumbnail_cache.put(filepath, error_pixmap)
            self.thumbnail_loaded.emit(filepath)

    def shutdown(self):
        self._is_shutting_down = True
        with QMutexLocker(self.mutex):
            self.queue.clear()
            self.pending_jobs.clear()
        self.semaphore.release(NUM_WORKERS * 2)
        self.thread_pool.waitForDone(1000)


loader_manager = LoaderManager()
