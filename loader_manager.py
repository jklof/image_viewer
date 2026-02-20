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
    QThread,
    QSize,
)
from PySide6.QtGui import QPixmap, QImage, QColor, QImageReader
from ui_components import THUMBNAIL_SIZE

logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
THUMBNAIL_CACHE_SIZE = 2000
NUM_WORKERS = max(2, QThread.idealThreadCount() - 1)
MAX_PENDING_JOBS = 100


class ThumbnailCache:
    """Thread-safe LRU cache using Python's native threading.Lock."""

    def __init__(self, size):
        self.cache = collections.OrderedDict()
        self.size = size
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
        return None

    def put(self, key, value):
        with self.lock:
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

                    # 2. QImageReader for efficient decoding
                    reader = QImageReader(filepath)
                    reader.setAutoTransform(True)  # Fix EXIF rotation issues

                    if not reader.canRead():
                        self.manager._internal_worker_result.emit(filepath, QImage())
                        continue

                    # 3. Rough scale down on load to save RAM (e.g. 3x thumbnail size)
                    original_size = reader.size()
                    rough_target = THUMBNAIL_SIZE * 3
                    if original_size.width() > rough_target or original_size.height() > rough_target:
                        scaled_size = original_size.scaled(
                            rough_target, rough_target, Qt.AspectRatioMode.KeepAspectRatio
                        )
                        reader.setScaledSize(scaled_size)

                    image = reader.read()

                    if not image.isNull():
                        # 4. High-quality smooth scaling
                        if image.width() > THUMBNAIL_SIZE or image.height() > THUMBNAIL_SIZE:
                            image = image.scaled(
                                THUMBNAIL_SIZE,
                                THUMBNAIL_SIZE,
                                Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.SmoothTransformation,
                            )

                        # 5. VERY IMPORTANT: Convert to optimal GPU format on the worker thread.
                        # This prevents the Main UI Thread from stuttering during QPixmap creation.
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
        if thumbnail_cache.get(filepath):
            return

        with self.lock:
            if filepath in self.pending_jobs:
                return

            added_to_semaphore = False

            if len(self.queue) < MAX_PENDING_JOBS:
                added_to_semaphore = True
            else:
                # Discard oldest without increasing total queue count
                try:
                    discarded = self.queue.pop()
                    self.pending_jobs.discard(discarded)
                except IndexError:
                    pass

            self.pending_jobs.add(filepath)
            self.queue.appendleft(filepath)

        # CRITICAL FIX: Only release the semaphore if the queue ACTUALLY grew.
        # This prevents 100% CPU busy loops when rapidly scrolling.
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
            # This is now instantaneous because of ARGB32_Premultiplied format
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