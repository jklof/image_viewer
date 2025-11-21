import os
from PySide6.QtWidgets import QStyledItemDelegate, QStyle, QListView
from PySide6.QtGui import QPixmap, QPainter, QFont, QColor, QPen, QBrush
from PySide6.QtCore import Qt, QSize, QRect, QTimer

from constants import THUMBNAIL_SIZE, ITEM_WIDTH, ITEM_HEIGHT, FILEPATH_ROLE, SCORE_ROLE


def create_placeholder_pixmap() -> QPixmap:
    """Creates a dark gray placeholder PIXMAP (not Icon)."""
    pixmap = QPixmap(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
    pixmap.fill(QColor(40, 40, 40))
    painter = QPainter(pixmap)
    painter.setPen(QColor(100, 100, 100))
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "Loading...")
    painter.end()
    return pixmap


class SmoothListView(QListView):
    """
    A QListView with optimizations for grid scrolling.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._finalize_resize)

        # WINDOWS OPTIMIZATION: Reduce paint events during scroll
        self.setVerticalScrollMode(QListView.ScrollMode.ScrollPerPixel)
        self.setUniformItemSizes(True)

    def resizeEvent(self, event):
        # Prevent layout trashing during resize
        if self.resizeMode() == QListView.ResizeMode.Adjust:
            self.setResizeMode(QListView.ResizeMode.Fixed)
        self._resize_timer.start(100)
        super().resizeEvent(event)

    def _finalize_resize(self):
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.doItemsLayout()


class SearchResultDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)

        # --- OPTIMIZATION 1: Pre-allocate GDI objects ---
        self.placeholder_pixmap = create_placeholder_pixmap()

        self.score_font = QFont()
        self.score_font.setBold(True)
        self.score_font.setPointSize(10)

        self.filename_font = QFont()
        self.filename_font.setPointSize(9)

        self.score_pen = QPen(QColor("#55aaff"))
        self.text_pen_color = QColor(220, 220, 220)  # Cache color, apply to painter directly

        self.bg_brush_normal = QBrush(QColor(40, 40, 40))
        self.bg_brush_hover = QBrush(QColor(85, 170, 255, 60))

        # Cache size hint
        self._size_hint = QSize(ITEM_WIDTH, ITEM_HEIGHT)

        # Pre-calculate offsets to avoid math in loop
        self._thumbnail_top_margin = 5
        self._score_height = 15
        self._text_height = 20
        self._padding = 5

    def sizeHint(self, option, index):
        return self._size_hint

    def paint(self, painter: QPainter, option, index):
        if not index.isValid():
            return

        # Save state only once
        painter.save()

        # Clip is essential for performance
        painter.setClipRect(option.rect)
        item_rect = option.rect

        # --- OPTIMIZATION 2: Fast Background Drawing ---
        # Avoid creating QColor/QBrush here. Use pre-allocated brushes.
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(item_rect, option.palette.highlight())
        elif option.state & QStyle.StateFlag.State_MouseOver:
            painter.fillRect(item_rect, self.bg_brush_hover)
        else:
            painter.fillRect(item_rect, self.bg_brush_normal)

        # --- OPTIMIZATION 3: Simplified Image Handling ---
        # The model now returns QPixmap directly (or None). No QIcon conversion.
        pixmap = index.data(Qt.ItemDataRole.DecorationRole)

        if pixmap is None:
            pixmap = self.placeholder_pixmap

        # Center calculations (Integer arithmetic is fast, but good to be clean)
        # X Center
        icon_x = item_rect.x() + (item_rect.width() - pixmap.width()) // 2
        # Y Center (within the thumbnail area)
        icon_y = item_rect.y() + self._thumbnail_top_margin + (THUMBNAIL_SIZE - pixmap.height()) // 2

        painter.drawPixmap(icon_x, icon_y, pixmap)

        # --- OPTIMIZATION 4: Score Drawing ---
        score = index.data(SCORE_ROLE)
        if isinstance(score, float) and score > 0:
            painter.setFont(self.score_font)
            painter.setPen(self.score_pen)

            score_rect = QRect(
                item_rect.left(),
                item_rect.y() + self._thumbnail_top_margin + THUMBNAIL_SIZE + 5,
                item_rect.width(),
                self._score_height,
            )
            painter.drawText(score_rect, Qt.AlignmentFlag.AlignCenter, f"{score:.2f}")

        # --- OPTIMIZATION 5: Filename Drawing ---
        # Only draw text if we really have to
        filepath = index.data(FILEPATH_ROLE)
        if filepath:
            filename = os.path.basename(filepath)
            painter.setFont(self.filename_font)
            painter.setPen(self.text_pen_color)

            filename_rect = QRect(
                item_rect.left() + self._padding,
                item_rect.bottom() - self._text_height,
                item_rect.width() - (self._padding * 2),
                self._text_height,
            )
            # Elision is expensive, but necessary.
            # We rely on Qt's internal caching for font metrics.
            elided_text = painter.fontMetrics().elidedText(filename, Qt.TextElideMode.ElideRight, filename_rect.width())
            painter.drawText(filename_rect, Qt.AlignmentFlag.AlignCenter, elided_text)

        painter.restore()
