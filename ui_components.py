import os
from pathlib import Path
from PySide6.QtWidgets import QStyledItemDelegate, QStyle, QLabel, QListView
from PySide6.QtGui import QPixmap, QIcon, QPainter, QFont, QColor, QPen
from PySide6.QtCore import Qt, QSize, QRect, QTimer

from constants import THUMBNAIL_SIZE, ITEM_WIDTH, FILEPATH_ROLE, SCORE_ROLE


def create_placeholder_icon() -> QIcon:
    """Creates a dark gray placeholder icon with text."""
    pixmap = QPixmap(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
    pixmap.fill(QColor(40, 40, 40))
    painter = QPainter(pixmap)
    painter.setPen(QColor(100, 100, 100))
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "Loading...")
    painter.end()
    return QIcon(pixmap)


class SmoothListView(QListView):
    """
    A QListView that handles resizing more efficiently.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        # Timer to debounce the layout calculation
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._finalize_resize)

    def resizeEvent(self, event):
        # 1. Switch to "Fixed" to prevent jumpiness during drag
        if self.resizeMode() == QListView.ResizeMode.Adjust:
            self.setResizeMode(QListView.ResizeMode.Fixed)

        # 2. Restart debounce timer
        self._resize_timer.start(100)

        super().resizeEvent(event)

    def _finalize_resize(self):
        # 3. Re-enable Adjust
        self.setResizeMode(QListView.ResizeMode.Adjust)
        # 4. Force re-layout
        self.doItemsLayout()


class SearchResultDelegate(QStyledItemDelegate):
    """
    A custom delegate to render image search results.
    Optimized to draw QPixmaps directly instead of converting to QIcons.
    """

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

        # Cache the size hint to avoid recalculations
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

        # 1. Draw Background
        bg_color = QColor(40, 40, 40)
        if option.state & QStyle.StateFlag.State_Selected:
            bg_color = option.palette.highlight().color()
        elif option.state & QStyle.StateFlag.State_MouseOver:
            bg_color = self.hover_color
        painter.fillRect(item_rect, bg_color)

        # 2. Draw Thumbnail (Optimized & Centered)
        data_variant = index.data(Qt.ItemDataRole.DecorationRole)

        pixmap = None
        if isinstance(data_variant, QIcon):
            pixmap = data_variant.pixmap(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
        elif isinstance(data_variant, QPixmap):
            pixmap = data_variant

        if pixmap and not pixmap.isNull():
            # Center Horizontally
            icon_x = item_rect.x() + (item_rect.width() - pixmap.width()) // 2

            # --- ASPECT RATIO FIX: Center Vertically ---
            # The reserved height for the image is THUMBNAIL_SIZE.
            # We calculate offset if the image is shorter (e.g. landscape)
            vertical_offset = (THUMBNAIL_SIZE - pixmap.height()) // 2
            icon_y = item_rect.y() + self._thumbnail_top_margin + vertical_offset
            # -------------------------------------------

            painter.drawPixmap(icon_x, icon_y, pixmap)

        # 3. Draw Score
        score = index.data(SCORE_ROLE)
        if isinstance(score, (int, float)) and score > 0:
            painter.setFont(self.score_font)
            painter.setPen(self.score_pen)
            # Position score below the reserved thumbnail area
            score_rect = QRect(
                item_rect.left(),
                item_rect.y() + self._thumbnail_top_margin + THUMBNAIL_SIZE + self._score_top_margin,
                item_rect.width(),
                15,
            )
            painter.drawText(score_rect, Qt.AlignmentFlag.AlignCenter, f"{score:.2f}")

        # 4. Draw Filename
        filepath = index.data(FILEPATH_ROLE)
        if filepath:
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
