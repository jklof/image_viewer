# ui_components.py

from pathlib import Path
from PySide6.QtWidgets import QStyledItemDelegate, QStyle
# --- UPDATED IMPORT: Added QPalette ---
from PySide6.QtGui import QPixmap, QIcon, QPainter, QFont, QColor, QPalette
from PySide6.QtCore import Qt, QSize, QRect

# The size of the icon/thumbnail we will generate and display
THUMBNAIL_SIZE = 150 
# The total size of each item in the grid, allowing for padding
ITEM_WIDTH = 180
ITEM_HEIGHT = 210

# Custom data roles to store our specific data in the model
FILEPATH_ROLE = Qt.ItemDataRole.UserRole + 1
SCORE_ROLE = Qt.ItemDataRole.UserRole + 2

def create_list_item(score: float, filepath: str):
    """
    Helper function to create a QStandardItem with an icon and our custom data.
    This replaces the need for a full QWidget for each result.
    """
    from PySide6.QtGui import QStandardItem
    
    # Create the item and set its icon
    pixmap = QPixmap(filepath).scaled(
        THUMBNAIL_SIZE, THUMBNAIL_SIZE, 
        Qt.AspectRatioMode.KeepAspectRatio, 
        Qt.TransformationMode.SmoothTransformation
    )
    item = QStandardItem(QIcon(pixmap), "") # Text is drawn by the delegate
    
    # Make the item non-editable and non-selectable by text
    item.setFlags(Qt.ItemFlag.ItemIsEnabled)
    
    # Store our custom data in the item using the defined roles
    item.setData(filepath, FILEPATH_ROLE)
    item.setData(score, SCORE_ROLE)
    
    return item


class SearchResultDelegate(QStyledItemDelegate):
    """
    A custom delegate to control the rendering of each item in the QListView.
    This class is responsible for drawing the thumbnail, score, and filename.
    """
    def sizeHint(self, option, index):
        """Returns the size of each item."""
        return QSize(ITEM_WIDTH, ITEM_HEIGHT)

    def paint(self, painter: QPainter, option, index):
        """
        Paints the contents of a single item.
        """
        painter.save()

        # --- Get Data ---
        filepath = index.data(FILEPATH_ROLE)
        score = index.data(SCORE_ROLE)
        icon = index.data(Qt.ItemDataRole.DecorationRole)
        
        item_rect = option.rect 
        
        # --- Draw Background and Selection (CORRECTED) ---
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(item_rect, option.palette.highlight())
        elif option.state & QStyle.StateFlag.State_MouseOver:
            # --- CORRECTED LOGIC FOR HOVER ---
            # Get the highlight color from the palette
            highlight_color = option.palette.highlight().color()
            # Set its alpha to make it semi-transparent for the hover effect
            highlight_color.setAlpha(60) 
            painter.fillRect(item_rect, highlight_color)
            # --- End of Correction ---
        
        # --- Draw Icon (Thumbnail) ---
        icon_rect = QRect(item_rect.x(), item_rect.y(), ITEM_WIDTH, THUMBNAIL_SIZE)
        pixmap = icon.pixmap(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
        centered_icon_rect = QRect(
            int(icon_rect.x() + (icon_rect.width() - pixmap.width()) / 2),
            int(icon_rect.y() + (icon_rect.height() - pixmap.height()) / 2),
            pixmap.width(),
            pixmap.height()
        )
        painter.drawPixmap(centered_icon_rect, pixmap)

        # --- Draw Score Text ---
        score_font = QFont()
        score_font.setBold(True)
        painter.setFont(score_font)
        painter.setPen(QColor("#55aaff"))
        
        score_text = f"{score:.4f}"
        score_rect = QRect(
            item_rect.x() + 5, 
            item_rect.y() + THUMBNAIL_SIZE + 5, 
            item_rect.width() - 10, 
            20
        )
        painter.drawText(score_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, score_text)

        # --- Draw Filename Text (with eliding) ---
        filename_font = QFont()
        painter.setFont(filename_font)
        painter.setPen(option.palette.text().color())
        
        filename_text = Path(filepath).name
        filename_rect = QRect(
            item_rect.x() + 5, 
            item_rect.y() + THUMBNAIL_SIZE + 25, 
            item_rect.width() - 10, 
            20
        )

        font_metrics = painter.fontMetrics()
        elided_text = font_metrics.elidedText(
            filename_text, 
            Qt.TextElideMode.ElideRight, 
            filename_rect.width()
        )
        painter.drawText(filename_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, elided_text)

        painter.restore()