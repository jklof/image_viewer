# qt_visualizer.py

import pyqtgraph as pg
import numpy as np
from pathlib import Path

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame, QApplication
from PySide6.QtGui import QPixmap, QCursor # --- ADDED QCursor ---
from PySide6.QtCore import Qt, Signal, Slot, QPoint 

# Ensure PySide6 compatibility
pg.setConfigOption('leftButtonPan', False) # Use right-click for pan, left for selecting/hover

class QtImageTooltip(QFrame):
    """
    A custom frameless widget to act as a fast, persistent image tooltip.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        # Use ToolTip flag to ensure it appears over other windows
        self.setWindowFlags(Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint | Qt.WindowType.NoDropShadowWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMouseTracking(True)
        self.setVisible(False)
        self.setFixedSize(200, 200) # Fixed size for the thumbnail area

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedSize(190, 190)
        # Styled to match a dark theme look
        self.image_label.setStyleSheet("border: 2px solid rgba(85, 170, 255, 200); background-color: rgba(20, 20, 20, 220); border-radius: 5px;")
        
        layout.addWidget(self.image_label)

    @Slot(str, QPoint)
    def show_image(self, filepath: str, global_pos: QPoint):
        pixmap = QPixmap(filepath)
        if pixmap.isNull():
            return
        
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
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
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot_widget = pg.PlotWidget()
        self.plot_item = self.plot_widget.plotItem
        
        # Set up plot defaults
        self.plot_item.setTitle("UMAP 2D Image Embeddings (Drag to Pan, Scroll to Zoom)")
        self.plot_item.setLabel('bottom', 'UMAP Dimension 1')
        self.plot_item.setLabel('left', 'UMAP Dimension 2')
        self.plot_item.showGrid(x=True, y=True, alpha=0.5)

        layout = QVBoxLayout(self)
        layout.addWidget(self.plot_widget)

        # pyqtgraph setup for fast scatter plotting
        self.scatter_plot = pg.ScatterPlotItem(
            size=7, 
            pen=pg.mkPen(None), # No outline
            brush=pg.mkBrush(255, 255, 255, 150), # Default brush
            hoverable=True, 
            tip=None # Custom tooltip is handled manually
        )
        self.plot_widget.addItem(self.scatter_plot)

        # Use sigHovered, which handles both point-on and point-off events
        self.scatter_plot.sigHovered.connect(self._on_point_hovered_or_unhovered)

        # Data storage: [(x, y, cluster, filepath), ...]
        self.all_data: list = [] 
        self.tooltip = QtImageTooltip()

    @Slot(object, object)
    def _on_point_hovered_or_unhovered(self, plot_item, points):
        """
        Handles the sigHovered signal.
        - If points is not empty, it's a hover on a point.
        - If points is empty, it's a hover off a point (unhover).
        """
        # --- CORRECTED: Use len() for NumPy array check to avoid ValueError ---
        if len(points) == 0: 
            # Unhover event
            self.tooltip.hide_tooltip()
            # Safely attempt to reset the status bar text
            try:
                QApplication.instance().mainWindow().statusBar().showMessage("Backend ready. You can now search.")
            except AttributeError:
                pass
            return
            
        # Hover event
        point = points[0]
        data_index = int(point.data()) # Gets the stored index
        
        # Look up the data from our storage
        x, y, cluster, filepath = self.all_data[data_index]
        
        # --- CORRECTED: Use QCursor.pos() to get the global mouse position and avoid AttributeError ---
        global_mouse_pos = QCursor.pos()

        # Display the image tooltip
        self.tooltip.show_image(filepath, global_mouse_pos)
        
        # Update the main window's status bar
        try:
            QApplication.instance().mainWindow().statusBar().showMessage(
                f"Hover: Cluster={cluster} | File: {Path(filepath).name}"
            )
        except AttributeError:
            # Fallback if mainWindow() is not defined on QApplication
            pass

    @Slot(list)
    def load_plot_data(self, plot_data: list):
        """
        Loads the data, generates the colors, and updates the plot.
        """
        self.tooltip.hide_tooltip()
        self.all_data = plot_data
        
        if not plot_data:
            self.scatter_plot.setData([])
            self.data_loaded.emit(0)
            return

        # Separate the columns
        x = np.array([d[0] for d in plot_data])
        y = np.array([d[1] for d in plot_data])
        clusters = np.array([d[2] for d in plot_data])
        
        # Generate colors based on clusters
        max_cluster = max(clusters)
        
        # --- Using a manually defined, robust ColorMap to avoid FileNotFoundError ---
        if max_cluster >= 0:
            # Define 6 anchor colors for a rainbow effect
            colors_list = [
                (255, 0, 0),     # Red
                (255, 255, 0),   # Yellow
                (0, 255, 0),     # Green
                (0, 255, 255),   # Cyan
                (0, 0, 255),     # Blue
                (255, 0, 255)    # Magenta
            ]
            
            # Create a ColorMap instance
            cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, len(colors_list)), color=colors_list)
            
            # Pre-allocate array for all points with 4 channels (R, G, B, A)
            all_colors = np.empty((len(plot_data), 4), dtype=np.uint8)

            # --- Map only non-noise points ---
            non_noise_indices = clusters != -1
            cluster_indices = clusters[non_noise_indices]
            
            if len(cluster_indices) > 0:
                # Normalize cluster IDs (mod len(colors_list)) to the range [0, 1] for the ColorMap
                normalized_clusters = (cluster_indices % len(colors_list)) / len(colors_list)
                
                # --- CORRECTED: Use 'byte' mode to get the RGBA array and avoid KeyError: 'qrgb' ---
                mapped_colors = cmap.map(normalized_clusters, 'byte')
                all_colors[non_noise_indices] = mapped_colors
            else:
                 # If all points are noise, all_colors will be set to gray below
                 pass
        
        else:
            # If there are no positive clusters, initialize as white/default
            all_colors = np.array([pg.mkColor('w')] * len(clusters))

        # Set noise points (-1) to a different color (semi-transparent gray)
        noise_indices = clusters == -1
        # Set the color for noise points to semi-transparent gray (RGB=100, Alpha=150)
        gray_color = np.array([100, 100, 100, 150], dtype=np.uint8) 
        all_colors[noise_indices] = gray_color
        
        # Update the scatter plot data. The 'data' parameter stores the index 
        # which is retrieved in _on_point_hovered.
        self.scatter_plot.setData(x=x, y=y, data=np.arange(len(x)),
                                 # --- Pass the numpy array of colors ---
                                 brush=all_colors, symbol='o', size=7) 
        
        # Auto-range the view
        self.plot_item.autoRange()

        self.data_loaded.emit(len(plot_data))