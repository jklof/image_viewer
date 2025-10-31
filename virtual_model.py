from PySide6.QtCore import QAbstractListModel, QModelIndex, Qt, Slot
from PySide6.QtGui import QIcon

import logging

logger = logging.getLogger(__name__)

from ui_components import FILEPATH_ROLE, SCORE_ROLE, create_placeholder_icon
from loader_manager import loader_manager, thumbnail_cache


class ImageResultModel(QAbstractListModel):
    """
    A lazy-loading virtualized list model with intelligent prefetching.
    Requests thumbnails on-demand and prefetches nearby items for smooth scrolling.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.results_data = []
        self.placeholder_icon = create_placeholder_icon()
        self._filepath_to_row_map = {}

        # Prefetching configuration
        self._prefetch_radius = 5  # Number of items to prefetch ahead/behind
        self._last_prefetch_row = -1  # Track last prefetch position to avoid duplicates
        self._prefetch_threshold = 2  # Only prefetch if moved this many rows

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
            # Check the cache for the requested item
            cached_pixmap = thumbnail_cache.get(filepath)

            if cached_pixmap:
                # Cache hit - return immediately
                icon = QIcon(cached_pixmap)
                # Prefetch nearby items for smooth scrolling
                self._prefetch_nearby(row)
                return icon
            else:
                # Cache miss - request this item and prefetch nearby
                loader_manager.request_thumbnail(filepath)
                self._prefetch_nearby(row)
                return self.placeholder_icon
        return None

    def _prefetch_nearby(self, center_row: int):
        """
        Intelligently prefetch thumbnails for items near the given row.
        Only prefetches if we've moved significantly since last prefetch.

        Args:
            center_row: The current row being viewed
        """
        # Check if we've moved enough to warrant a new prefetch
        if abs(center_row - self._last_prefetch_row) < self._prefetch_threshold:
            return

        self._last_prefetch_row = center_row

        # Calculate prefetch range with bounds checking
        start_row = max(0, center_row - self._prefetch_radius)
        end_row = min(len(self.results_data), center_row + self._prefetch_radius + 1)

        # Request thumbnails for items in range (already cached items are ignored by loader)
        prefetch_count = 0
        for row in range(start_row, end_row):
            if row < len(self.results_data):
                _, filepath = self.results_data[row]
                # The loader_manager will automatically ignore already cached/queued items
                loader_manager.request_thumbnail(filepath)
                prefetch_count += 1

        # Optional: Log prefetch activity for debugging
        logger.debug(f"Prefetched {prefetch_count} items around row {center_row} (range: {start_row}-{end_row})")

    @Slot(str)
    def on_thumbnail_ready(self, filepath: str):
        """Called when a thumbnail finishes loading."""
        row = self._filepath_to_row_map.get(filepath)
        if row is not None:
            index = self.createIndex(row, 0)
            self.dataChanged.emit(index, index, [Qt.ItemDataRole.DecorationRole])

    def set_results(self, results: list):
        """Load new search results and reset prefetch state."""
        self.beginResetModel()
        self.results_data = results
        self._filepath_to_row_map = {filepath: i for i, (_, filepath) in enumerate(results)}
        # Reset prefetch tracking for new data
        self._last_prefetch_row = -1
        self.endResetModel()

    def clear(self):
        """Clear all results and reset state."""
        self.beginResetModel()
        self.results_data = []
        self._filepath_to_row_map = {}
        self._last_prefetch_row = -1
        self.endResetModel()

    def set_prefetch_radius(self, radius: int):
        """
        Adjust the prefetch radius dynamically.

        Args:
            radius: Number of items to prefetch ahead/behind (default: 5)
        """
        self._prefetch_radius = max(0, radius)
        logger.info(f"Prefetch radius set to {self._prefetch_radius}")

    def set_prefetch_threshold(self, threshold: int):
        """
        Adjust how far the user must scroll before triggering new prefetch.

        Args:
            threshold: Minimum row movement to trigger prefetch (default: 2)
        """
        self._prefetch_threshold = max(1, threshold)
        logger.info(f"Prefetch threshold set to {self._prefetch_threshold}")
