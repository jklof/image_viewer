import pyqtgraph as pg
import numpy as np
from pathlib import Path

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame
from PySide6.QtGui import QPixmap, QCursor
from PySide6.QtCore import Qt, Signal, Slot, QPoint

# Ensure PySide6 compatibility
pg.setConfigOption("leftButtonPan", False)  # Use right-click for pan, left for selecting/hover


class QtImageTooltip(QFrame):
    """
    A custom frameless widget to act as a fast, persistent image tooltip.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        # Use ToolTip flag to ensure it appears over other windows
        self.setWindowFlags(
            Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint | Qt.WindowType.NoDropShadowWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMouseTracking(True)
        self.setVisible(False)
        self.setFixedSize(200, 200)  # Fixed size for the thumbnail area

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedSize(190, 190)
        # Styled to match a dark theme look
        self.image_label.setStyleSheet(
            "border: 2px solid rgba(85, 170, 255, 200); background-color: rgba(20, 20, 20, 220); border-radius: 5px;"
        )

        layout.addWidget(self.image_label)

    @Slot(str, QPoint)
    def show_image(self, filepath: str, global_pos: QPoint):
        pixmap = QPixmap(filepath)
        if pixmap.isNull():
            return

        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled_pixmap)

        # Position the tooltip slightly offset from the mouse
        self.move(global_pos + QPoint(20, 20))
        self.setVisible(True)

    def hide_tooltip(self):
        self.setVisible(False)


class QtVisualizer(QWidget):
    """
    A native Qt Widget using pyqtgraph for high-performance 2D scatter plotting.
    Handles data visualization, pan/zoom, and native image tooltips.
    """

    data_loaded = Signal(int)
    status_update = Signal(str)
    image_search_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot_widget = pg.PlotWidget()
        self.plot_item = self.plot_widget.plotItem

        # Set up plot defaults
        self.plot_item.setTitle("UMAP 2D Image Embeddings (Double-click a point to search by similarity)")
        self.plot_item.setLabel("bottom", "UMAP Dimension 1")
        self.plot_item.setLabel("left", "UMAP Dimension 2")
        self.plot_item.showGrid(x=True, y=True, alpha=0.5)

        layout = QVBoxLayout(self)
        layout.addWidget(self.plot_widget)

        self.scatter_plot = pg.ScatterPlotItem(
            size=7,
            pen=pg.mkPen(None),
            brush=pg.mkBrush(255, 255, 255, 150),
            hoverable=True,
            tip=None,
        )
        self.plot_widget.addItem(self.scatter_plot)

        self.scatter_plot.sigHovered.connect(self._on_point_hovered_or_unhovered)
        self.plot_item.scene().sigMouseClicked.connect(self._on_plot_clicked)

        self.all_data: list = []
        # Passing 'self' as parent helps with automatic cleanup,
        # though we still need manual visibility management due to ToolTip flag.
        self.tooltip = QtImageTooltip(self)

        # Guard flag to prevent race conditions with hover events during view transitions
        self._is_active = False

    def showEvent(self, event):
        """Called when the widget is being shown."""
        super().showEvent(event)
        self._is_active = True

    def hideEvent(self, event):
        """Called when the widget is being hidden."""
        super().hideEvent(event)
        self._is_active = False
        self.tooltip.hide_tooltip()

    @Slot(object, object)
    def _on_point_hovered_or_unhovered(self, plot_item, points):
        # Immediately return if we are not the active view
        if not self._is_active:
            return

        if len(points) == 0:
            self.tooltip.hide_tooltip()
            self.status_update.emit("Backend ready. You can now search.")
            return

        point = points[0]
        data_index = int(point.data())
        _, _, cluster, filepath = self.all_data[data_index]
        global_mouse_pos = QCursor.pos()
        self.tooltip.show_image(filepath, global_mouse_pos)
        self.status_update.emit(f"Hover: Cluster={cluster} | File: {Path(filepath).name}")

    @Slot(object)
    def _on_plot_clicked(self, event):
        if not event.double():
            return

        points = self.scatter_plot.pointsAt(event.pos())
        if points.size == 0:
            return

        point = points[0]
        data_index = int(point.data())

        if 0 <= data_index < len(self.all_data):
            _, _, _, filepath = self.all_data[data_index]

            # --- CRITICAL: Immediately disable further hover events ---
            self._is_active = False
            self.tooltip.hide_tooltip()

            self.image_search_requested.emit(filepath)

    @Slot(list)
    def load_plot_data(self, plot_data: list):
        self.tooltip.hide_tooltip()
        self.all_data = plot_data

        if not plot_data:
            self.scatter_plot.setData([])
            self.data_loaded.emit(0)
            return

        x = np.array([d[0] for d in plot_data])
        y = np.array([d[1] for d in plot_data])
        clusters = np.array([d[2] for d in plot_data])
        max_cluster = max(clusters)

        if max_cluster >= 0:
            colors_list = [
                (255, 0, 0),
                (255, 255, 0),
                (0, 255, 0),
                (0, 255, 255),
                (0, 0, 255),
                (255, 0, 255),
            ]
            cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, len(colors_list)), color=colors_list)
            all_colors = np.empty((len(plot_data), 4), dtype=np.uint8)
            non_noise_indices = clusters != -1
            cluster_indices = clusters[non_noise_indices]

            if len(cluster_indices) > 0:
                normalized_clusters = (cluster_indices % len(colors_list)) / len(colors_list)
                mapped_colors = cmap.map(normalized_clusters, "byte")
                all_colors[non_noise_indices] = mapped_colors
        else:
            all_colors = np.array([pg.mkColor("w")] * len(clusters))

        noise_indices = clusters == -1
        gray_color = np.array([100, 100, 100, 150], dtype=np.uint8)
        all_colors[noise_indices] = gray_color

        self.scatter_plot.setData(
            x=x,
            y=y,
            data=np.arange(len(x)),
            brush=all_colors,
            symbol="o",
            size=7,
        )

        self.plot_item.autoRange()
        self.data_loaded.emit(len(plot_data))
