import logging
import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QLabel,
    QStatusBar,
    QMessageBox,
    QListView,
    QMenu,
    QStackedWidget,
)
from PySide6.QtGui import QAction, QScreen
from PySide6.QtCore import QThread, Signal, Qt, Slot

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from backend import BackendWorker
from ui_components import SearchResultDelegate, FILEPATH_ROLE
from qt_visualizer import QtVisualizer
from virtual_model import ImageResultModel
from loader_manager import loader_manager


class MainWindow(QMainWindow):
    """The main window for the AI Image Explorer application."""

    request_text_search = Signal(str)
    request_image_search = Signal(str)
    request_visualization = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Image Explorer")

        screen = QApplication.primaryScreen()
        if screen:
            screen_geometry = screen.availableGeometry()
            self.setGeometry(
                50,
                50,
                int(screen_geometry.width() * 0.8),
                int(screen_geometry.height() * 0.8),
            )
        else:
            self.setGeometry(50, 50, 1200, 800)

        self.worker_thread = QThread()
        self.backend_worker = BackendWorker()
        self.backend_worker.moveToThread(self.worker_thread)

        self.query_image_path = None

        self._init_ui()
        self._connect_signals()

        self.worker_thread.start()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        controls_layout = QHBoxLayout()
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Enter text to re-order all images by similarity...")
        self.text_search_btn = QPushButton("Order by Text")
        self.image_search_btn = QPushButton("Order by Image")
        self.selected_image_label = QLabel("No image selected.")
        self.selected_image_label.setWordWrap(True)

        controls_layout.addWidget(self.search_bar)
        controls_layout.addWidget(self.text_search_btn)
        controls_layout.addWidget(self.image_search_btn)
        controls_layout.addWidget(self.selected_image_label)

        self.visualize_btn = QPushButton("Visualize Embeddings")
        controls_layout.addStretch()
        controls_layout.addWidget(self.visualize_btn)

        main_layout.addLayout(controls_layout)
        self.content_stack = QStackedWidget()

        self.init_label = QLabel("Initializing, please wait...")
        self.init_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.init_label.setStyleSheet("font-size: 16px; color: #aaa;")

        self.results_view = QListView()
        self.results_view.setViewMode(QListView.ViewMode.IconMode)
        self.results_view.setResizeMode(QListView.ResizeMode.Adjust)
        self.results_view.setMovement(QListView.Movement.Static)
        self.results_view.setSpacing(10)
        self.results_model = ImageResultModel(self)
        self.results_view.setModel(self.results_model)
        self.results_view.setItemDelegate(SearchResultDelegate(self))
        self.results_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        self.loading_label = QLabel("Ordering all images, please wait...")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_label.setStyleSheet("font-size: 16px; color: #aaa;")

        self.visualizer_widget = QtVisualizer()

        self.content_stack.addWidget(self.init_label)
        self.content_stack.addWidget(self.results_view)
        self.content_stack.addWidget(self.loading_label)
        self.content_stack.addWidget(self.visualizer_widget)

        self.content_stack.setCurrentWidget(self.init_label)
        main_layout.addWidget(self.content_stack)

        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Starting backend thread...")
        self.set_controls_enabled(False)

    def _connect_signals(self):
        self.worker_thread.started.connect(self.backend_worker.initialize)
        self.search_bar.returnPressed.connect(self.start_text_search)
        self.text_search_btn.clicked.connect(self.start_text_search)
        self.image_search_btn.clicked.connect(self.select_image_for_search)
        self.visualize_btn.clicked.connect(self.start_visualization)

        self.request_text_search.connect(self.backend_worker.perform_text_search)
        self.request_image_search.connect(self.backend_worker.perform_image_search)
        self.request_visualization.connect(self.backend_worker.request_visualization_data)

        self.backend_worker.initialized.connect(self.on_backend_initialized)
        self.backend_worker.error.connect(self.on_backend_error)
        self.backend_worker.results_ready.connect(self.display_results)
        self.backend_worker.status_update.connect(self.statusBar().showMessage)
        self.backend_worker.visualization_data_ready.connect(self.on_visualization_data_ready)

        self.visualizer_widget.data_loaded.connect(self.on_visualization_loaded)
        self.visualizer_widget.status_update.connect(self.statusBar().showMessage)

        self.worker_thread.finished.connect(self.backend_worker.shutdown)
        self.worker_thread.finished.connect(self.backend_worker.deleteLater)
        self.results_view.customContextMenuRequested.connect(self.on_results_context_menu)

    def on_backend_initialized(self):
        self.content_stack.setCurrentWidget(self.results_view)
        self.statusBar().showMessage("Backend ready. Enter text or select an image to begin exploring.")
        self.set_controls_enabled(True)

    def on_backend_error(self, err_msg):
        QMessageBox.critical(self, "Backend Error", f"A critical error occurred: {err_msg}")
        self.statusBar().showMessage("Backend failed to initialize. Please restart.")
        self.init_label.setText("Initialization Failed. Please restart.")

    def start_visualization(self):
        self.clear_results()
        self.content_stack.setCurrentWidget(self.loading_label)
        self.loading_label.setText("Calculating UMAP coordinates and clusters...")
        self.set_controls_enabled(False)
        self.request_visualization.emit()

    @Slot(list)
    def on_visualization_data_ready(self, plot_data: list):
        if not plot_data:
            self.content_stack.setCurrentWidget(self.results_view)
            self.set_controls_enabled(True)
            return
        self.visualizer_widget.load_plot_data(plot_data)

    @Slot(int)
    def on_visualization_loaded(self, count: int):
        self.content_stack.setCurrentWidget(self.visualizer_widget)
        self.loading_label.setText("Ordering all images, please wait...")
        self.set_controls_enabled(True)
        self.statusBar().showMessage(f"Visualization complete. Plotted {count} points.")

    def start_text_search(self):
        query = self.search_bar.text()
        if not query:
            return
        self.clear_results()
        self.content_stack.setCurrentWidget(self.loading_label)
        self.set_controls_enabled(False)
        self.request_text_search.emit(query)

    def select_image_for_search(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Image to Order By", "", "Image Files (*.png *.jpg *.jpeg)"
        )
        if filepath:
            self.execute_image_search(filepath)

    def execute_image_search(self, filepath: str):
        self.query_image_path = filepath
        self.selected_image_label.setText(f"Selected: ...{Path(filepath).name}")
        self.clear_results()
        self.content_stack.setCurrentWidget(self.loading_label)
        self.set_controls_enabled(False)
        self.request_image_search.emit(self.query_image_path)

    def display_results(self, results: list):
        self.content_stack.setCurrentWidget(self.results_view)
        self.results_model.set_results(results)
        self.statusBar().showMessage(f"Ordering complete. Displaying all {len(results)} images.")
        self.set_controls_enabled(True)

    def clear_results(self):
        self.results_model.clear()

    def on_results_context_menu(self, pos):
        index = self.results_view.indexAt(pos)
        if not index.isValid():
            return
        filepath = index.data(FILEPATH_ROLE)
        context_menu = QMenu(self)
        reorder_action = QAction("Order by Similarity to This", self)
        reorder_action.triggered.connect(lambda: self.execute_image_search(filepath))
        context_menu.addAction(reorder_action)
        context_menu.addSeparator()
        copy_path_action = QAction("Copy Full Path", self)
        copy_path_action.triggered.connect(lambda: QApplication.clipboard().setText(filepath))
        context_menu.addAction(copy_path_action)
        copy_image_action = QAction("Copy Image", self)
        copy_image_action.triggered.connect(lambda: self.copy_image_to_clipboard(filepath))
        context_menu.addAction(copy_image_action)
        context_menu.exec(self.results_view.viewport().mapToGlobal(pos))

    def copy_image_to_clipboard(self, filepath: str):
        from PySide6.QtGui import QPixmap

        pixmap = QPixmap(filepath)
        if not pixmap.isNull():
            QApplication.clipboard().setPixmap(pixmap)

    def set_controls_enabled(self, enabled: bool):
        self.search_bar.setEnabled(enabled)
        self.text_search_btn.setEnabled(enabled)
        self.image_search_btn.setEnabled(enabled)
        self.visualize_btn.setEnabled(enabled)

    def closeEvent(self, event):
        """Ensure all background threads are shut down cleanly."""
        logger.info("Main window closing. Shutting down background services.")

        loader_manager.shutdown()

        self.worker_thread.quit()
        if not self.worker_thread.wait(5000):
            logger.warning("Backend worker thread did not shut down cleanly. Terminating.")
            self.worker_thread.terminate()

        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    try:
        import qdarktheme

        app.setStyleSheet(qdarktheme.load_stylesheet())
    except ImportError:
        logger.info("For a dark theme, install with: pip install pyqtdarktheme")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
