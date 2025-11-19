import os
from pathlib import Path
from PySide6.QtWidgets import QStyledItemDelegate, QStyle, QLabel, QFrame
from PySide6.QtGui import QPixmap, QIcon, QPainter, QFont, QColor, QPalette, QPen, QFontMetrics
from PySide6.QtCore import Qt, QSize, QRect, QModelIndex

from constants import (
    THUMBNAIL_SIZE,
    ITEM_WIDTH,
    FILEPATH_ROLE,
    SCORE_ROLE,
)


def create_thumbnail_label_with_border(size: QSize) -> QLabel:
    """
    Creates a QLabel designed to display a thumbnail, with a border and background.
    """
    label = QLabel()
    label.setFixedSize(size)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setStyleSheet("QLabel { border: 1px solid #55aaff; border-radius: 4px; background-color: #333; }")
    # Fill with a placeholder pixmap to ensure border is visible even when no image
    placeholder_pixmap = QPixmap(size)
    placeholder_pixmap.fill(QColor("#333"))
    label.setPixmap(placeholder_pixmap)
    return label


def create_list_item(score: float, filepath: str) -> "QStandardItem":  # Forward declaration for QStandardItem
    from PySide6.QtGui import QStandardItem  # Import here to avoid circular dependency if needed

    item = QStandardItem("")
    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)  # Make selectable for context menu
    item.setData(filepath, FILEPATH_ROLE)
    item.setData(score, SCORE_ROLE)
    return item


def create_placeholder_icon() -> QIcon:
    """Creates a placeholder icon with text."""
    pixmap = QPixmap(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
    pixmap.fill(QColor(40, 40, 40)) # Dark gray background
    
    painter = QPainter(pixmap)
    painter.setPen(QColor(100, 100, 100)) # Lighter gray text
    font = painter.font()
    font.setPointSize(10)
    painter.setFont(font)
    
    # Draw "Loading..." in the center
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "Loading...")
    painter.end()
    
    return QIcon(pixmap)


class SearchResultDelegate(QStyledItemDelegate):
    """
    Highly optimized delegate with cached objects and minimal allocations.
    Adjusted for the new UI look with more spacing and clearer text.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.placeholder_icon = create_placeholder_icon()

        # Pre-cache fonts
        self.score_font = QFont()
        self.score_font.setBold(True)
        self.score_font.setPointSize(10)  # Smaller font for score
        self.filename_font = QFont()
        self.filename_font.setPointSize(9)  # Smaller font for filename

        # Cache font metrics (expensive to create repeatedly)
        self._filename_metrics = QFontMetrics(self.filename_font)

        # Pre-cache pens (avoid creating in paint loop)
        self.score_pen = QPen(QColor("#55aaff"))  # Blue color for score
        # Initializing text_pen without a color; its color will be set dynamically
        # in paint() using option.palette.text().color() for theme compatibility.
        self.text_pen = QPen()

        # Pre-cache commonly used colors
        self.hover_color = QColor(85, 170, 255, 60)  # Light blue with transparency for hover

        # Cache size hint (never changes)
        # Increased height slightly for more spacing
        self._size_hint = QSize(ITEM_WIDTH, THUMBNAIL_SIZE + 50)  # Adjusted height

        # Pre-calculate rect offsets (avoid repeated calculations)
        self._padding = 5  # Padding around elements
        self._thumbnail_top_margin = 5
        self._score_top_margin = 5  # Space between thumbnail and score
        self._filename_top_margin = 2  # Space between score and filename

    def sizeHint(self, option, index):
        """Return pre-cached size."""
        return self._size_hint

    def paint(self, painter: QPainter, option, index):
        """Optimized paint with minimal allocations and state changes."""
        # Early exit for invalid index
        if not index.isValid():
            return

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        item_rect = option.rect
        # Adjust item_rect for internal padding if desired, or handle padding within each element.
        # For this design, we'll use padding for text elements relative to the item_rect.

        # Draw selection/hover background (only if needed)
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(item_rect, option.palette.highlight())
        elif option.state & QStyle.StateFlag.State_MouseOver:
            painter.fillRect(item_rect, self.hover_color)
        else:
            # Draw a subtle background for the item for better separation
            painter.fillRect(item_rect, QColor(40, 40, 40))  # Dark background for each item

        # Draw thumbnail (center-aligned)
        icon = index.data(Qt.ItemDataRole.DecorationRole)
        pixmap = icon.pixmap(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
        icon_x = item_rect.x() + (item_rect.width() - pixmap.width()) // 2
        icon_y = item_rect.y() + self._thumbnail_top_margin  # Adjusted top margin for thumbnail
        painter.drawPixmap(icon_x, icon_y, pixmap)

        # Draw score text
        score = index.data(SCORE_ROLE)
        painter.setFont(self.score_font)
        painter.setPen(self.score_pen)

        score_text = f"{score:.4f}"
        # Measure text width to align it correctly
        score_width = painter.fontMetrics().horizontalAdvance(score_text)
        # Center horizontally under the thumbnail
        score_x = item_rect.x() + (item_rect.width() - score_width) // 2
        score_y = icon_y + THUMBNAIL_SIZE + self._score_top_margin

        painter.drawText(
            score_x,
            score_y + painter.fontMetrics().ascent(),  # Adjust for baseline
            score_text,
        )

        # Draw filename (elided)
        filepath = index.data(FILEPATH_ROLE)

        filename = os.path.basename(filepath) 
        
        # Set the pen color using the palette for theme compatibility
        self.text_pen.setColor(option.palette.text().color())
        painter.setFont(self.filename_font)
        painter.setPen(self.text_pen)

        # The filename rect now starts after the score and is centered
        filename_rect_width = item_rect.width() - 2 * self._padding
        filename_rect_x = item_rect.x() + self._padding
        filename_rect_y = score_y + painter.fontMetrics().height() + self._filename_top_margin

        elided = self._filename_metrics.elidedText(filename, Qt.TextElideMode.ElideRight, filename_rect_width)

        painter.drawText(
            QRect(filename_rect_x, filename_rect_y, filename_rect_width, self._filename_metrics.height()),
            Qt.AlignmentFlag.AlignCenter,
            elided,
        )

        painter.restore()

    def updateEditorGeometry(self, editor, option, index):
        """Override to prevent editor creation (not needed for this view)."""
        pass
