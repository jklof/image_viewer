from PySide6.QtCore import Signal, Qt, Slot, QPoint
from PySide6.QtGui import QMouseEvent, QPainter, QColor, QBrush, QPolygon, QWheelEvent
from PySide6.QtWidgets import QWidget
from loader_manager import get_loader_manager, thumbnail_cache


class NavThumbnail(QWidget):
    clicked = Signal()
    resized = Signal()

    COLLAPSED_WIDTH = 38
    EXPANDED_WIDTH = 130
    BTN_HEIGHT = 68

    def __init__(self, direction="next", parent=None):
        super().__init__(parent)
        self.direction = direction
        self.filepath = None
        self._is_hovered = False

        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # Prevents stealing keyboard focus from viewer
        self.setFixedSize(self.COLLAPSED_WIDTH, self.BTN_HEIGHT)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)

        get_loader_manager().thumbnail_loaded.connect(self._on_thumbnail_loaded)

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

    def enterEvent(self, event):
        self._is_hovered = True
        self.setFixedSize(self.EXPANDED_WIDTH, self.BTN_HEIGHT)
        self.resized.emit()
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._is_hovered = False
        self.setFixedSize(self.COLLAPSED_WIDTH, self.BTN_HEIGHT)
        self.resized.emit()
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self.filepath:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        # Absorb double clicks and trigger navigation; prevents closing SingleMediaViewer
        if event.button() == Qt.MouseButton.LeftButton and self.filepath:
            self.clicked.emit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        # Forward wheel scrolling to viewer for next/prev navigation
        event.ignore()

    def paintEvent(self, event):
        if not self.filepath:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect().adjusted(1, 1, -1, -1)
        radius = 8

        # Background fill and border
        if self._is_hovered:
            painter.setBrush(QColor(20, 20, 20, 230))
            painter.setPen(QColor(85, 170, 255, 180))  # Accent border
        else:
            painter.setBrush(QColor(25, 25, 25, 160))
            painter.setPen(QColor(255, 255, 255, 40))

        painter.drawRoundedRect(rect, radius, radius)

        # Draw thumbnail when expanded on hover
        if self._is_hovered:
            thumb_w = 76
            thumb_h = 56
            if self.direction == "prev":
                thumb_x = rect.right() - thumb_w - 6
            else:
                thumb_x = rect.left() + 6
            thumb_y = rect.center().y() - thumb_h // 2
            thumb_rect = rect.adjusted(0, 0, 0, 0)
            thumb_rect.setLeft(thumb_x)
            thumb_rect.setTop(thumb_y)
            thumb_rect.setWidth(thumb_w)
            thumb_rect.setHeight(thumb_h)

            pixmap = thumbnail_cache.get(self.filepath)
            if pixmap:
                scaled = pixmap.scaled(
                    thumb_w, thumb_h,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                px = thumb_x + (thumb_w - scaled.width()) // 2
                py = thumb_y + (thumb_h - scaled.height()) // 2

                painter.save()
                painter.setClipRect(thumb_rect)
                painter.drawPixmap(px, py, scaled)
                painter.restore()
            else:
                painter.fillRect(thumb_rect, QColor(40, 40, 40))

        # Draw Chevron Arrow
        arrow_w = 8
        arrow_h = 14
        if not self._is_hovered:
            cx = rect.center().x()
        else:
            if self.direction == "prev":
                cx = rect.left() + 18
            else:
                cx = rect.right() - 18
        cy = rect.center().y()

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(255, 255, 255, 230 if self._is_hovered else 180)))

        if self.direction == "prev":
            points = [
                QPoint(cx + 4, cy - arrow_h // 2),
                QPoint(cx - arrow_w // 2, cy),
                QPoint(cx + 4, cy + arrow_h // 2),
            ]
        else:
            points = [
                QPoint(cx - 4, cy - arrow_h // 2),
                QPoint(cx + arrow_w // 2, cy),
                QPoint(cx - 4, cy + arrow_h // 2),
            ]
        painter.drawPolygon(QPolygon(points))
