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
    QSemaphore,
    QSize,
)
from PySide6.QtGui import QPixmap, QImage, QColor, QImageReader
from ui_components import THUMBNAIL_SIZE

logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
THUMBNAIL_CACHE_SIZE = 2000
# Cap at 4 to prevent SSD I/O saturation and GUI thread starvation on high-core CPUs
NUM_WORKERS = 4
MAX_PENDING_JOBS = 100


class ThumbnailCache:
    """A thread-safe cache optimized for LOCKLESS reads."""

    def __init__(self, size):
        self.cache = {}
        self.queue = collections.deque()
        self.size = size
        self.lock = threading.Lock()

    def get(self, key):
        # FAST PATH: Dictionary `.get()` is atomic in Python (GIL protected).
        # No lock required! This prevents UI stuttering during rapid scrolling.
        # We sacrifice strict LRU access-updates for massive speed gains.
        return self.cache.get(key)

    def put(self, key, value):
        with self.lock:
            if key in self.cache:
                return
            self.cache[key] = value
            self.queue.append(key)
            if len(self.cache) > self.size:
                oldest = self.queue.popleft()
                self.cache.pop(oldest, None)


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
                    if thumbnail_cache.get(filepath):
                        continue

                    reader = QImageReader(filepath)
                    reader.setAutoTransform(True)

                    if not reader.canRead():
                        self.manager._internal_worker_result.emit(filepath, QImage())
                        continue

                    # Pre-scale to save RAM during decode
                    original_size = reader.size()
                    rough_target = THUMBNAIL_SIZE * 3
                    if original_size.width() > rough_target or original_size.height() > rough_target:
                        scaled_size = original_size.scaled(
                            rough_target, rough_target, Qt.AspectRatioMode.KeepAspectRatio
                        )
                        reader.setScaledSize(scaled_size)

                    image = reader.read()

                    if not image.isNull():
                        if image.width() > THUMBNAIL_SIZE or image.height() > THUMBNAIL_SIZE:
                            image = image.scaled(
                                THUMBNAIL_SIZE,
                                THUMBNAIL_SIZE,
                                Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.SmoothTransformation,
                            )

                        # Convert to GPU-optimal format on the worker thread
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

        self.lock = threading.Lock()
        self.queue = collections.deque()
        self.pending_jobs = set()
        self.semaphore = QSemaphore(0)

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

        # LOCKLESS FAST-PATHS: Checking sets/dicts is atomic in Python.
        if filepath in self.pending_jobs:
            return
        if filepath in thumbnail_cache.cache:
            return

        with self.lock:
            # Re-check inside lock
            if filepath in self.pending_jobs:
                return

            added_to_semaphore = False

            if len(self.queue) < MAX_PENDING_JOBS:
                added_to_semaphore = True
            else:
                try:
                    discarded = self.queue.pop()
                    self.pending_jobs.discard(discarded)
                except IndexError:
                    pass

            self.pending_jobs.add(filepath)
            self.queue.appendleft(filepath)

        if added_to_semaphore:
            self.semaphore.release(1)

    def get_next_job(self) -> str | None:
        if not self.semaphore.tryAcquire(1, 50):
            return None

        with self.lock:
            if self._is_shutting_down:
                return None
            try:
                return self.queue.popleft()
            except IndexError:
                return None

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
        with self.lock:
            self.queue.clear()
            self.pending_jobs.clear()
        self.semaphore.release(NUM_WORKERS * 2)
        self.thread_pool.waitForDone(1000)


loader_manager = LoaderManager()