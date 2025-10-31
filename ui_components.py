from pathlib import Path
from PySide6.QtWidgets import QStyledItemDelegate, QStyle
from PySide6.QtGui import QPixmap, QIcon, QPainter, QFont, QColor, QPalette
from PySide6.QtCore import Qt, QSize, QRect, QModelIndex
from PySide6.QtGui import QStandardItem

THUMBNAIL_SIZE = 150
ITEM_WIDTH = 180
ITEM_HEIGHT = 210

FILEPATH_ROLE = Qt.ItemDataRole.UserRole + 1
SCORE_ROLE = Qt.ItemDataRole.UserRole + 2
SHA256_ROLE = Qt.ItemDataRole.UserRole + 3


def create_list_item(score: float, filepath: str, sha256: str) -> QStandardItem:
    item = QStandardItem("")
    item.setFlags(Qt.ItemFlag.ItemIsEnabled)
    item.setData(filepath, FILEPATH_ROLE)
    item.setData(score, SCORE_ROLE)
    item.setData(sha256, SHA256_ROLE)
    return item


def create_placeholder_icon() -> QIcon:
    pixmap = QPixmap(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
    # Using a dark color for the placeholder
    pixmap.fill(QColor(45, 45, 45))
    return QIcon(pixmap)


class SearchResultDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.placeholder_icon = create_placeholder_icon

    def sizeHint(self, option, index):
        return QSize(ITEM_WIDTH, ITEM_HEIGHT)

    def paint(self, painter: QPainter, option, index):
        painter.save()

        filepath = index.data(FILEPATH_ROLE)
        score = index.data(SCORE_ROLE)
        icon = index.data(Qt.ItemDataRole.DecorationRole)
        if not icon or icon.isNull():
            icon = self.placeholder_icon

        item_rect = option.rect
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(item_rect, option.palette.highlight())
        elif option.state & QStyle.StateFlag.State_MouseOver:
            highlight_color = option.palette.highlight().color()
            highlight_color.setAlpha(60)
            painter.fillRect(item_rect, highlight_color)

        icon_rect = QRect(item_rect.x(), item_rect.y(), ITEM_WIDTH, THUMBNAIL_SIZE)
        pixmap = icon.pixmap(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
        centered_icon_rect = QRect(
            int(icon_rect.x() + (icon_rect.width() - pixmap.width()) / 2),
            int(icon_rect.y() + (icon_rect.height() - pixmap.height()) / 2),
            pixmap.width(),
            pixmap.height(),
        )
        painter.drawPixmap(centered_icon_rect, pixmap)

        score_font = QFont()
        score_font.setBold(True)
        painter.setFont(score_font)
        painter.setPen(QColor("#55aaff"))
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

        filename_font = QFont()
        painter.setFont(filename_font)
        painter.setPen(option.palette.text().color())
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
