from PySide6.QtCore import Signal, Qt, Slot, QPoint
from PySide6.QtGui import QMouseEvent, QPainter, QRegion, QColor, QBrush, QPolygon
from PySide6.QtWidgets import QWidget
from loader_manager import get_loader_manager, thumbnail_cache
class NavThumbnail(QWidget):
    # ... (No changes to NavThumbnail class) ...
    clicked = Signal()

    def __init__(self, direction="next", parent=None):
        super().__init__(parent)
        self.direction = direction
        self.filepath = None
        self.setFixedWidth(160)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        get_loader_manager().thumbnail_loaded.connect(self._on_thumbnail_loaded)
        self._is_hovered = False

    def set_filepath(self, path: str | None):
        self.filepath = path
        self.setVisible(path is not None)
        if path and not thumbnail_cache.get(path):
            get_loader_manager().request_thumbnail(path)
        self.update()

    @Slot(str)
    def _on_thumbnail_loaded(self, path: str):
        if path == self.filepath:
            self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self.filepath:
            self.clicked.emit()

    def enterEvent(self, event):
        self._is_hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._is_hovered = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        if not self.filepath:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        draw_rect = self.rect().adjusted(5, 5, -5, -5)

        path = QRegion(draw_rect, QRegion.RegionType.Rectangle)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(45, 45, 45))
        painter.drawRoundedRect(draw_rect, 8, 8)

        pixmap = thumbnail_cache.get(self.filepath)
        if pixmap:
            scaled = pixmap.scaled(
                draw_rect.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            x = draw_rect.center().x() - scaled.width() // 2
            y = draw_rect.center().y() - scaled.height() // 2

            painter.save()
            painter.setClipRect(draw_rect)
            painter.drawPixmap(x, y, scaled)
            painter.restore()

        if self._is_hovered:
            painter.setBrush(QColor(255, 255, 255, 40))
            painter.drawRoundedRect(draw_rect, 8, 8)

        arrow_size = 16
        cx, cy = draw_rect.center().x(), draw_rect.center().y()

        painter.setBrush(QColor(0, 0, 0, 150))
        painter.drawEllipse(QPoint(cx, cy), arrow_size + 4, arrow_size + 4)

        painter.setBrush(QBrush(QColor(255, 255, 255, 240)))
        if self.direction == "prev":
            points = [QPoint(cx + 4, cy - 8), QPoint(cx - 6, cy), QPoint(cx + 4, cy + 8)]
        else:
            points = [QPoint(cx - 4, cy - 8), QPoint(cx + 6, cy), QPoint(cx - 4, cy + 8)]
        painter.drawPolygon(QPolygon(points))


