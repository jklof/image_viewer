import collections
import logging
import threading

import cv2
import numpy as np

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


class LRUThumbnailCache:
    """Thread-safe LRU cache using OrderedDict."""

    def __init__(self, capacity: int):
        self.cache = collections.OrderedDict()
        self.capacity = capacity
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

    def put(self, key, value):
        with self.lock:
            self.cache[key] = value
            self.cache.move_to_end(key)
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)


thumbnail_cache = LRUThumbnailCache(THUMBNAIL_CACHE_SIZE)


class LoadThumbnailTask(QRunnable):
    """A discrete, fire-and-forget task for the QThreadPool."""

    def __init__(self, filepath: str, manager: "LoaderManager"):
        super().__init__()
        self.filepath = filepath
        self.manager = manager
        self.setAutoDelete(True)

    def run(self):
        if self.manager._is_shutting_down:
            return

        try:
            # 1. Fast path cache check
            if thumbnail_cache.get(self.filepath):
                return

            ext = self.filepath.lower()
            image = QImage()

            if ext.endswith(".mp4"):
                cap = cv2.VideoCapture(self.filepath)
                if cap.isOpened():
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * 0.1))
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        h, w, ch = frame.shape
                        bytes_per_line = ch * w
                        tmp_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                        image = tmp_img.copy()
                cap.release()
            else:
                image = QImage(self.filepath)

            if not image.isNull():
                if image.width() > THUMBNAIL_SIZE or image.height() > THUMBNAIL_SIZE:
                    image = image.scaled(
                        THUMBNAIL_SIZE,
                        THUMBNAIL_SIZE,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                image = image.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
                self.manager._internal_worker_result.emit(self.filepath, image)
            else:
                self.manager._internal_worker_result.emit(self.filepath, QImage())

        except Exception as e:
            logger.error(f"Thumbnail load failed for {self.filepath}: {e}")
            self.manager._internal_worker_result.emit(self.filepath, QImage())


class LoaderManager(QObject):
    thumbnail_loaded = Signal(str)
    _internal_worker_result = Signal(str, QImage)

    def __init__(self):
        super().__init__()
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(NUM_WORKERS)
        self._is_shutting_down = False
        self._task_counter = 0  # Used to ensure newer requests jump to the front of the queue

        # Keep track of active tasks to prevent duplicate queuing
        self.lock = threading.Lock()
        self.pending_jobs = set()

        self._internal_worker_result.connect(self._on_worker_result_ready)

    def initialize(self):
        self._is_shutting_down = False

    @Slot(str)
    def request_thumbnail(self, filepath: str):
        if self._is_shutting_down:
            return

        if thumbnail_cache.get(filepath):
            return

        with self.lock:
            if filepath in self.pending_jobs:
                return
            self.pending_jobs.add(filepath)

            # Increment to ensure this new task gets the highest execution priority
            self._task_counter += 1
            current_priority = self._task_counter

        task = LoadThumbnailTask(filepath, self)
        # Passing priority to QThreadPool forces LIFO behavior (snappy scrolling)
        self.thread_pool.start(task, current_priority)

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
        self.thread_pool.clear()
        self.thread_pool.waitForDone(1500)


loader_manager = LoaderManager()
