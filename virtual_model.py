from PySide6.QtCore import QAbstractListModel, QModelIndex, Qt, Slot, QMimeData, QUrl
from PySide6.QtGui import QIcon

import logging

logger = logging.getLogger(__name__)

from constants import FILEPATH_ROLE, SCORE_ROLE

from ui_components import create_placeholder_icon
from loader_manager import loader_manager, thumbnail_cache


class ImageResultModel(QAbstractListModel):
    """
    A lazy-loading virtualized list model with intelligent prefetching.
    Supports drag-and-drop operations by providing file URLs.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.results_data = []
        self.placeholder_icon = create_placeholder_icon()
        self._filepath_to_row_map = {}

        loader_manager.thumbnail_loaded.connect(self.on_thumbnail_ready)

    def rowCount(self, parent=QModelIndex()):
        return len(self.results_data)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row = index.row()
        if row >= len(self.results_data):
            return None

        score, filepath = self.results_data[row]

        if role == SCORE_ROLE:
            return score
        if role == FILEPATH_ROLE:
            return filepath

        if role == Qt.ItemDataRole.DecorationRole:
            cached_pixmap = thumbnail_cache.get(filepath)

            if cached_pixmap:
                # --- FIX 2: Return QPixmap directly, avoid QIcon wrapper ---
                return cached_pixmap
            else:
                loader_manager.request_thumbnail(filepath)
                # Use a cached pixmap for the placeholder too if possible,
                # but returning the icon here is fine for the loading state
                return self.placeholder_icon

        return None

    # --- NEW: Drag and Drop Support ---
    def flags(self, index):
        default_flags = super().flags(index)
        if index.isValid():
            return default_flags | Qt.ItemFlag.ItemIsDragEnabled
        return default_flags

    def mimeTypes(self):
        return ["text/uri-list"]

    def mimeData(self, indexes):
        mime_data = QMimeData()
        urls = []
        for index in indexes:
            if index.isValid():
                filepath = self.data(index, FILEPATH_ROLE)
                urls.append(QUrl.fromLocalFile(filepath))
        mime_data.setUrls(urls)
        return mime_data

    # --- End Drag and Drop Support ---

    @Slot(str)
    def on_thumbnail_ready(self, filepath: str):
        row = self._filepath_to_row_map.get(filepath)
        if row is not None:
            index = self.createIndex(row, 0)
            self.dataChanged.emit(index, index, [Qt.ItemDataRole.DecorationRole])

    def set_results(self, results: list):
        self.beginResetModel()
        self.results_data = results
        self._filepath_to_row_map = {filepath: i for i, (_, filepath) in enumerate(results)}
        self._last_prefetch_row = -1
        self.endResetModel()

    def clear(self):
        self.beginResetModel()
        self.results_data = []
        self._filepath_to_row_map = {}
        self._last_prefetch_row = -1
        self.endResetModel()

    def set_prefetch_radius(self, radius: int):
        self._prefetch_radius = max(0, radius)
        logger.info(f"Prefetch radius set to {self._prefetch_radius}")

    def set_prefetch_threshold(self, threshold: int):
        self._prefetch_threshold = max(1, threshold)
        logger.info(f"Prefetch threshold set to {self._prefetch_threshold}")
