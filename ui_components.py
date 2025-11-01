from pathlib import Path
from PySide6.QtWidgets import QStyledItemDelegate, QStyle
from PySide6.QtGui import QPixmap, QIcon, QPainter, QFont, QColor, QPalette, QPen
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
    """An optimized delegate that pre-caches fonts and pens."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.placeholder_icon = create_placeholder_icon()

        self.score_font = QFont()
        self.score_font.setBold(True)
        self.filename_font = QFont()

        self.score_pen = QPen(QColor("#55aaff"))
        self.text_pen = QPen()

    def sizeHint(self, option, index):
        return QSize(ITEM_WIDTH, ITEM_HEIGHT)

    def paint(self, painter: QPainter, option, index):
        painter.save()

        filepath = index.data(FILEPATH_ROLE)
        score = index.data(SCORE_ROLE)

        pixmap = index.data(PIXMAP_ROLE)
        if not pixmap or pixmap.isNull():
            icon = index.data(Qt.ItemDataRole.DecorationRole)
            if not icon or icon.isNull():
                icon = self.placeholder_icon
            pixmap = icon.pixmap(THUMBNAIL_SIZE, THUMBNAIL_SIZE)

        item_rect = option.rect
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(item_rect, option.palette.highlight())
        elif option.state & QStyle.StateFlag.State_MouseOver:
            highlight_color = option.palette.highlight().color()
            highlight_color.setAlpha(60)
            painter.fillRect(item_rect, highlight_color)

        centered_icon_rect = QRect(
            int(item_rect.x() + (item_rect.width() - pixmap.width()) / 2),
            int(item_rect.y() + (THUMBNAIL_SIZE - pixmap.height()) / 2),
            pixmap.width(),
            pixmap.height(),
        )
        painter.drawPixmap(centered_icon_rect, pixmap)

        painter.setFont(self.score_font)
        painter.setPen(self.score_pen)
        score_text = f"{score:.4f}"
        score_rect = QRect(
            item_rect.x() + 5,
            item_rect.y() + THUMBNAIL_SIZE + 5,
            item_rect.width() - 10,
            20,
        )
        painter.drawText(
            score_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            score_text,
        )

        self.text_pen.setColor(option.palette.text().color())
        painter.setFont(self.filename_font)
        painter.setPen(self.text_pen)
        filename_text = Path(filepath).name
        filename_rect = QRect(
            item_rect.x() + 5,
            item_rect.y() + THUMBNAIL_SIZE + 25,
            item_rect.width() - 10,
            20,
        )
        font_metrics = painter.fontMetrics()
        elided_text = font_metrics.elidedText(filename_text, Qt.TextElideMode.ElideRight, filename_rect.width())
        painter.drawText(
            filename_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            elided_text,
        )

        painter.restore()
