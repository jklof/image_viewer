from PySide6.QtCore import QAbstractListModel, QModelIndex, Qt, Slot, QMimeData, QUrl
import logging
import collections

from constants import FILEPATH_ROLE, SCORE_ROLE
from loader_manager import loader_manager, thumbnail_cache
from ui_components import create_placeholder_pixmap

logger = logging.getLogger(__name__)


class ImageResultModel(QAbstractListModel):
    """
    A lazy-loading virtualized list model.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.results_data = []
        # Pre-allocate one placeholder
        self.placeholder_pixmap = create_placeholder_pixmap()
        # Map filepath to list of row indices (handles duplicate filepaths)
        self._filepath_to_row_map = collections.defaultdict(list)
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
            # 1. Fast Path: Check Cache
            cached = thumbnail_cache.get(filepath)
            if cached:
                return cached

            # 2. Request Load
            # Since we use LIFO in the loader, it is safe to spam requests here.
            loader_manager.request_thumbnail(filepath)

            # 3. Return None (Delegate will draw the placeholder)
            return None

        return None

    # --- Drag and Drop Support ---
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

    @Slot(str)
    def on_thumbnail_ready(self, filepath: str):
        rows = self._filepath_to_row_map.get(filepath)
        if rows:
            # Emit dataChanged for all rows with this filepath
            for row in rows:
                index = self.createIndex(row, 0)
                self.dataChanged.emit(index, index, [Qt.ItemDataRole.DecorationRole])

    def set_results(self, results: list):
        self.beginResetModel()
        self.results_data = results
        # Map filepath to list of row indices to handle duplicate filepaths
        self._filepath_to_row_map = collections.defaultdict(list)
        for i, (_, filepath) in enumerate(results):
            self._filepath_to_row_map[filepath].append(i)
        self.endResetModel()

    def clear(self):
        self.beginResetModel()
        self.results_data = []
        self._filepath_to_row_map = collections.defaultdict(list)
        self.endResetModel()
