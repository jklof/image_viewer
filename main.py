# main.py

import logging
import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QGridLayout, QScrollArea,
    QFileDialog, QLabel, QStatusBar, QMessageBox
)
from PySide6.QtCore import QThread, Signal

import logging
logger = logging.getLogger(__name__)


# Local imports from our other project files
from backend import BackendWorker
from ui_components import SearchResultWidget

# --- Constants ---
RESULTS_PER_ROW = 5


class MainWindow(QMainWindow):
    """The main window for the AI Image Search application."""
    
    request_text_search = Signal(str)
    request_image_search = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Image Search")
        self.setGeometry(100, 100, 1200, 800)
        
        self.worker_thread = QThread()
        self.backend_worker = BackendWorker()
        self.backend_worker.moveToThread(self.worker_thread)
        
        self.query_image_path = None
        self._init_ui()
        self._connect_signals()
        
        self.worker_thread.start()

    def _init_ui(self):
        # This method is unchanged
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        controls_layout = QHBoxLayout()
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Enter text description...")
        self.text_search_btn = QPushButton("Search Text")
        self.image_search_btn = QPushButton("Select Image to Search")
        self.selected_image_label = QLabel("No image selected.")
        self.selected_image_label.setWordWrap(True)
        controls_layout.addWidget(self.search_bar)
        controls_layout.addWidget(self.text_search_btn)
        controls_layout.addWidget(self.image_search_btn)
        controls_layout.addWidget(self.selected_image_label)
        main_layout.addLayout(controls_layout)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.results_container = QWidget()
        self.results_layout = QGridLayout(self.results_container)
        self.scroll_area.setWidget(self.results_container)
        main_layout.addWidget(self.scroll_area)
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Starting backend thread...")
        self.search_bar.setEnabled(False)
        self.text_search_btn.setEnabled(False)
        self.image_search_btn.setEnabled(False)

    def _connect_signals(self):
        # This method is unchanged
        self.worker_thread.started.connect(self.backend_worker.initialize)
        self.search_bar.returnPressed.connect(self.start_text_search)
        self.text_search_btn.clicked.connect(self.start_text_search)
        self.image_search_btn.clicked.connect(self.select_image_for_search)
        self.request_text_search.connect(self.backend_worker.perform_text_search)
        self.request_image_search.connect(self.backend_worker.perform_image_search)
        self.backend_worker.initialized.connect(self.on_backend_initialized)
        self.backend_worker.error.connect(self.on_backend_error)
        self.backend_worker.results_ready.connect(self.display_results)
        self.backend_worker.status_update.connect(self.statusBar().showMessage)
        self.worker_thread.finished.connect(self.backend_worker.shutdown)
        self.worker_thread.finished.connect(self.backend_worker.deleteLater)

    def on_backend_initialized(self):
        logger.info("Backend initialized successfully.")
        self.statusBar().showMessage("Backend ready. You can now search.")
        self.search_bar.setEnabled(True)
        self.text_search_btn.setEnabled(True)
        self.image_search_btn.setEnabled(True)

    def on_backend_error(self, err_msg):
        logger.error(f"Backend error: {err_msg}")
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setText("A critical error occurred in the backend.")
        msg_box.setInformativeText("The application may not function correctly. Please check the console for details.")
        msg_box.setDetailedText(err_msg)
        msg_box.exec()
        self.statusBar().showMessage("Backend failed to initialize. Please restart.")

    def start_text_search(self):
        # This method is unchanged
        query = self.search_bar.text()
        if not query: return
        self.clear_results()
        self.text_search_btn.setEnabled(False)
        self.image_search_btn.setEnabled(False)
        self.request_text_search.emit(query)

    def select_image_for_search(self):
        """Opens a file dialog to choose an image."""
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Query Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if filepath:
            # Trigger the actual search logic using the selected path
            self.execute_image_search(filepath)
    
    def on_find_similar_clicked(self, filepath: str):
        """Slot that is triggered by a result widget's context menu."""
        logger.info(f"Context menu clicked: Find images similar to {filepath}")
        self.execute_image_search(filepath)

    def execute_image_search(self, filepath: str):
        """Central method to start an image search from any source."""
        self.query_image_path = filepath
        self.selected_image_label.setText(f"Selected: ...{Path(filepath).name}")
        
        self.clear_results()
        self.text_search_btn.setEnabled(False)
        self.image_search_btn.setEnabled(False)
        self.request_image_search.emit(self.query_image_path)

    def display_results(self, results: list):
        """Slot to display search results. Now connects the context menu signal."""
        self.clear_results()
        if not results:
            self.statusBar().showMessage("Search complete. No results found.")
        else:
            for i, (score, path) in enumerate(results):
                row, col = divmod(i, RESULTS_PER_ROW)
                result_widget = SearchResultWidget(score, path)
                
                # --- NEW: Connect the widget's signal to our new slot ---
                result_widget.find_similar_requested.connect(self.on_find_similar_clicked)
                
                self.results_layout.addWidget(result_widget, row, col)
            self.statusBar().showMessage(f"Search complete. Found {len(results)} results.")
        
        self.text_search_btn.setEnabled(True)
        self.image_search_btn.setEnabled(True)
        
    def clear_results(self):
        # This method is unchanged
        while self.results_layout.count():
            child = self.results_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def closeEvent(self, event):
        # This method is unchanged
        logger.info("Main window closing. Quitting worker thread.")
        self.worker_thread.quit()
        success = self.worker_thread.wait(5000)
        if not success:
            logger.warning("Warning: Worker thread did not shut down cleanly. Terminating.")
            self.worker_thread.terminate()
        event.accept()


# --- Application Entry Point ---
if __name__ == "__main__":
    logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

    logger.info("Starting AI Image Search application.")
    # This block is unchanged
    app = QApplication(sys.argv)
    try:
        import qdarktheme
        app.setStyleSheet(qdarktheme.load_stylesheet())
    except ImportError:
        logger.info("Theme library not found. For a dark theme, install with: pip install pyqtdarktheme")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
    logger.info("Quitting AI Image Search application.")