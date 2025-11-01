from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QColor, QBrush, QPen
from PySide6.QtCore import Qt, QTimer, Property, QRectF, QEasingCurve
import math

import logging

logger = logging.getLogger(__name__)


class PulsingSpinner(QWidget):
    """
    A custom widget that draws a pulsing ring animation.
    Its color can be configured when the animation is started.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(150, 150)  # Fixed size for the spinner itself
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setVisible(False)

        self._current_scale = 0.0  # From 0.0 to 1.0, controls pulse
        self._current_alpha = 0  # From 0 to 255, controls fade
        # --- Store the ring color as an instance attribute ---
        self._ring_color = QColor(85, 170, 255)  # Default to blue

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_animation)
        self._animation_start_time = 0
        self._animation_duration = 1500  # Duration of one pulse cycle in ms
        self._is_animating = False

    def _update_animation(self):
        if not self._is_animating:
            return

        elapsed_time = self._timer.interval()  # Time since last update
        self._animation_start_time += elapsed_time

        progress = (self._animation_start_time % self._animation_duration) / self._animation_duration

        eased_progress = QEasingCurve(QEasingCurve.Type.OutSine).valueForProgress(progress)

        if eased_progress < 0.5:
            self._current_scale = eased_progress * 2
        else:
            self._current_scale = (1 - eased_progress) * 2

        self._current_alpha = int(255 * (1 - eased_progress))

        self.update()

    # --- The method now accepts a color argument ---
    def start_animation(self, color: QColor = QColor(85, 170, 255)):
        """
        Starts the animation with the specified color.
        Args:
            color: The QColor to use for the pulsing ring. Defaults to blue.
        """
        logger.info(f"Starting pulsing spinner animation with color {color.name()}.")
        self._ring_color = color  # Set the color for the upcoming animation
        if not self._is_animating:
            self._is_animating = True
            self._animation_start_time = 0
            self._timer.start(16)  # ~60 FPS
        self.setVisible(True)
        self.update()  # Trigger an immediate repaint with the new color

    def stop_animation(self):
        if self._is_animating:
            logger.info("Stopping pulsing spinner animation.")
            self._is_animating = False
            self._timer.stop()
            self.setVisible(False)
            self._current_scale = 0.0
            self._current_alpha = 0
            self.update()

    def paintEvent(self, event):
        if not self._is_animating:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        base_outer_radius = self.width() * 0.4
        base_inner_radius = self.width() * 0.3

        current_outer_radius = base_outer_radius * (0.5 + 0.5 * self._current_scale)
        current_inner_radius = base_inner_radius * (0.5 + 0.5 * self._current_scale)

        # --- Use the stored color and apply the calculated alpha ---
        ring_color = QColor(self._ring_color)  # Make a copy to modify alpha
        ring_color.setAlpha(self._current_alpha)

        ring_width = current_outer_radius - current_inner_radius
        pen = QPen(ring_color)
        pen.setWidthF(ring_width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)

        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        center_x = self.width() / 2
        center_y = self.height() / 2
        radius = current_inner_radius + ring_width / 2

        painter.drawEllipse(QRectF(center_x - radius, center_y - radius, radius * 2, radius * 2))
