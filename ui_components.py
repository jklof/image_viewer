from pathlib import Path
from PySide6.QtWidgets import QStyledItemDelegate, QStyle
from PySide6.QtGui import QPixmap, QIcon, QPainter, QFont, QColor, QPalette, QPen, QFontMetrics
from PySide6.QtCore import Qt, QSize, QRect, QModelIndex
from PySide6.QtGui import QStandardItem

from constants import (
    THUMBNAIL_SIZE,
    ITEM_WIDTH,
    ITEM_HEIGHT,
    FILEPATH_ROLE,
    SCORE_ROLE,
    PIXMAP_ROLE,
)


def create_list_item(score: float, filepath: str) -> QStandardItem:
    item = QStandardItem("")
    item.setFlags(Qt.ItemFlag.ItemIsEnabled)
    item.setData(filepath, FILEPATH_ROLE)
    item.setData(score, SCORE_ROLE)
    return item


def create_placeholder_icon() -> QIcon:
    pixmap = QPixmap(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
    pixmap.fill(QColor(45, 45, 45))
    return QIcon(pixmap)


class SearchResultDelegate(QStyledItemDelegate):
    """
    Highly optimized delegate with cached objects and minimal allocations.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.placeholder_icon = create_placeholder_icon()

        # Pre-cache fonts
        self.score_font = QFont()
        self.score_font.setBold(True)
        self.filename_font = QFont()

        # Cache font metrics (expensive to create repeatedly)
        self._filename_metrics = QFontMetrics(self.filename_font)

        # Pre-cache pens (avoid creating in paint loop)
        self.score_pen = QPen(QColor("#55aaff"))
        self.text_pen = QPen()

        # Pre-cache commonly used colors
        self.hover_color = QColor(85, 170, 255, 60)

        # Cache size hint (never changes)
        self._size_hint = QSize(ITEM_WIDTH, ITEM_HEIGHT)

        # Pre-calculate rect offsets (avoid repeated calculations)
        self._icon_y_offset = (THUMBNAIL_SIZE - THUMBNAIL_SIZE) / 2
        self._score_y = THUMBNAIL_SIZE + 5
        self._filename_y = THUMBNAIL_SIZE + 25
        self._text_margin = 5

    def sizeHint(self, option, index):
        """Return pre-cached size."""
        return self._size_hint

    def paint(self, painter: QPainter, option, index):
        """Optimized paint with minimal allocations and state changes."""
        # Early exit for invalid index
        if not index.isValid():
            return

        painter.save()

        # Get data (these are fast lookups)
        filepath = index.data(FILEPATH_ROLE)
        score = index.data(SCORE_ROLE)
        icon = index.data(Qt.ItemDataRole.DecorationRole)

        item_rect = option.rect

        # Draw selection/hover background (only if needed)
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(item_rect, option.palette.highlight())
        elif option.state & QStyle.StateFlag.State_MouseOver:
            painter.fillRect(item_rect, self.hover_color)

        # Draw thumbnail (center-aligned)
        pixmap = icon.pixmap(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
        icon_x = item_rect.x() + (item_rect.width() - pixmap.width()) // 2
        icon_y = item_rect.y() + int(self._icon_y_offset)
        painter.drawPixmap(icon_x, icon_y, pixmap)

        # Draw score text
        painter.setFont(self.score_font)
        painter.setPen(self.score_pen)
        score_rect = QRect(
            item_rect.x() + self._text_margin,
            item_rect.y() + self._score_y,
            item_rect.width() - 2 * self._text_margin,
            20,
        )
        # Format score once
        painter.drawText(
            score_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"{score:.4f}",
        )

        # Draw filename (elided)
        self.text_pen.setColor(option.palette.text().color())
        painter.setFont(self.filename_font)
        painter.setPen(self.text_pen)

        filename_rect = QRect(
            item_rect.x() + self._text_margin,
            item_rect.y() + self._filename_y,
            item_rect.width() - 2 * self._text_margin,
            20,
        )

        # Use cached metrics for elision
        filename = Path(filepath).name
        elided = self._filename_metrics.elidedText(filename, Qt.TextElideMode.ElideRight, filename_rect.width())

        painter.drawText(
            filename_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            elided,
        )

        painter.restore()

    def updateEditorGeometry(self, editor, option, index):
        """Override to prevent editor creation (not needed for this view)."""
        pass
