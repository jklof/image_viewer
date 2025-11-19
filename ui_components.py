import os
from pathlib import Path
from PySide6.QtWidgets import QStyledItemDelegate, QStyle, QLabel
from PySide6.QtGui import QPixmap, QIcon, QPainter, QFont, QColor, QPen
from PySide6.QtCore import Qt, QSize, QRect

from constants import THUMBNAIL_SIZE, ITEM_WIDTH, FILEPATH_ROLE, SCORE_ROLE


def create_placeholder_icon() -> QIcon:
    pixmap = QPixmap(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
    pixmap.fill(QColor(40, 40, 40))
    painter = QPainter(pixmap)
    painter.setPen(QColor(100, 100, 100))
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "Loading...")
    painter.end()
    return QIcon(pixmap)


class SearchResultDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.placeholder_icon = create_placeholder_icon()
        self.score_font = QFont()
        self.score_font.setBold(True)
        self.score_font.setPointSize(10)
        self.filename_font = QFont()
        self.filename_font.setPointSize(9)
        self.score_pen = QPen(QColor("#55aaff"))
        self.text_pen = QPen()
        self.hover_color = QColor(85, 170, 255, 60)
        self._size_hint = QSize(ITEM_WIDTH, THUMBNAIL_SIZE + 50)
        self._padding = 5
        self._thumbnail_top_margin = 5
        self._score_top_margin = 5

    def sizeHint(self, option, index):
        return self._size_hint

    def paint(self, painter: QPainter, option, index):
        if not index.isValid():
            return

        painter.save()
        painter.setClipRect(option.rect)
        item_rect = option.rect

        # Background
        bg_color = QColor(40, 40, 40)
        if option.state & QStyle.StateFlag.State_Selected:
            bg_color = option.palette.highlight().color()
        elif option.state & QStyle.StateFlag.State_MouseOver:
            bg_color = self.hover_color
        painter.fillRect(item_rect, bg_color)

        # Thumbnail
        icon = index.data(Qt.ItemDataRole.DecorationRole)
        pixmap = icon.pixmap(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
        icon_x = item_rect.x() + (item_rect.width() - pixmap.width()) // 2
        icon_y = item_rect.y() + self._thumbnail_top_margin
        painter.drawPixmap(icon_x, icon_y, pixmap)

        # Score
        score = index.data(SCORE_ROLE)
        if score > 0:
            painter.setFont(self.score_font)
            painter.setPen(self.score_pen)
            score_rect = QRect(
                item_rect.left(), icon_y + THUMBNAIL_SIZE + self._score_top_margin, item_rect.width(), 15
            )
            painter.drawText(score_rect, Qt.AlignmentFlag.AlignCenter, f"{score:.2f}")

        # Filename
        filepath = index.data(FILEPATH_ROLE)
        filename = os.path.basename(filepath)
        painter.setFont(self.filename_font)
        self.text_pen.setColor(option.palette.text().color())
        painter.setPen(self.text_pen)

        filename_rect = QRect(
            item_rect.left() + self._padding, item_rect.bottom() - 20, item_rect.width() - (self._padding * 2), 20
        )
        elided = painter.fontMetrics().elidedText(filename, Qt.TextElideMode.ElideRight, filename_rect.width())
        painter.drawText(filename_rect, Qt.AlignmentFlag.AlignCenter, elided)

        painter.restore()
