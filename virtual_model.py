from PySide6.QtCore import QAbstractListModel, QModelIndex, Qt, Slot, QMimeData, QUrl
import logging
import collections

from constants import FILEPATH_ROLE, SCORE_ROLE, TAGS_ROLE, DUP_COUNT_ROLE
from loader_manager import get_loader_manager, thumbnail_cache
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
        # Map sha256 to list of row indices (handles duplicate content shown
        # as separate rows in Sort-by-Date). Used for SHA-aware tag toggles.
        self._sha_to_rows_map = collections.defaultdict(list)
        get_loader_manager().thumbnail_loaded.connect(self.on_thumbnail_ready)

    def rowCount(self, parent=QModelIndex()):
        return len(self.results_data)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row = index.row()
        if row >= len(self.results_data):
            return None

        score, filepath, tags = self.results_data[row][:3]

        if role == SCORE_ROLE:
            return score
        if role == FILEPATH_ROLE:
            return filepath
        if role == TAGS_ROLE:
            return "marked" in tags

        if role == DUP_COUNT_ROLE:
            # result tuple is (score, filepath, tags, dup_count[, sha256])
            # guard against old 3-tuples during transition
            return self.results_data[row][3] if len(self.results_data[row]) > 3 else 1

        if role == Qt.ItemDataRole.DecorationRole:
            # 1. Fast Path: Check Cache
            cached = thumbnail_cache.get(filepath)
            if cached:
                return cached

            # 2. Request Load
            # Since we use LIFO in the loader, it is safe to spam requests here.
            get_loader_manager().request_thumbnail(filepath)

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
        self._sha_to_rows_map = collections.defaultdict(list)
        for i, item in enumerate(results):
            filepath = item[1]
            self._filepath_to_row_map[filepath].append(i)
            if len(item) >= 5 and item[4]:
                self._sha_to_rows_map[item[4]].append(i)
        self.endResetModel()

    def clear(self):
        self.beginResetModel()
        self.results_data = []
        self._filepath_to_row_map = collections.defaultdict(list)
        self._sha_to_rows_map = collections.defaultdict(list)
        self.endResetModel()

    def toggle_tag_for_filepaths(self, filepaths: list[str]):
        """Optimistically update tags for specified filepaths, regardless of current sort order.

        SHA-aware: tags belong to content hashes, so toggling one filepath
        updates ALL rows sharing its sha256. This keeps Sort-by-Date (which
        shows every filepath) consistent when only one copy is selected.
        Each affected row is toggled exactly once.
        """
        # Resolve target rows: expand filepaths -> SHAs -> all sibling rows.
        target_rows: set[int] = set()
        for filepath in filepaths:
            rows = self._filepath_to_row_map.get(filepath, [])
            if not rows:
                continue
            # Collect SHAs for these rows (5th tuple element, may be missing)
            shas = set()
            for row in rows:
                if 0 <= row < len(self.results_data):
                    item = self.results_data[row]
                    if len(item) >= 5 and item[4]:
                        shas.add(item[4])
            if shas:
                for sha in shas:
                    target_rows.update(self._sha_to_rows_map.get(sha, []))
            else:
                # No SHA info (legacy 4-tuples): fall back to filepath rows
                target_rows.update(rows)

        for row in sorted(target_rows):
            if 0 <= row < len(self.results_data):
                item = self.results_data[row]
                score, fp, tags = item[0], item[1], item[2]
                dup_count = item[3] if len(item) > 3 else 1
                sha = item[4] if len(item) >= 5 else ""

                # Convert to set for safer manipulation (Also fixes Issue #6)
                tag_set = set(tags.split(",")) if tags else set()
                if "marked" in tag_set:
                    tag_set.discard("marked")
                else:
                    tag_set.add("marked")

                new_tags = ",".join(filter(None, tag_set))

                if sha:
                    self.results_data[row] = (score, fp, new_tags, dup_count, sha)
                else:
                    self.results_data[row] = (score, fp, new_tags, dup_count)
                index = self.createIndex(row, 0)
                self.dataChanged.emit(index, index, [TAGS_ROLE])
