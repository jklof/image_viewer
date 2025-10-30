# main.py

import logging
import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QFileDialog, QLabel, QStatusBar, QMessageBox,
    QListView, QMenu, QStackedWidget, QSpinBox
)
# --- UPDATED IMPORT: Added QScreen for screen geometry ---
from PySide6.QtGui import QStandardItemModel, QAction, QScreen
from PySide6.QtCore import QThread, Signal, Qt, Slot

import logging
logger = logging.getLogger(__name__)


# Local imports from our other project files
from backend import BackendWorker
from ui_components import SearchResultDelegate, create_list_item, FILEPATH_ROLE
# --- ADDED LOCAL IMPORT for native visualizer ---
from qt_visualizer import QtVisualizer


class MainWindow(QMainWindow):
    """The main window for the AI Image Search application."""
    
    # --- MODIFIED --- Signals now carry an integer for the number of results.
    request_text_search = Signal(str, int)
    request_image_search = Signal(str, int)
    
    # --- MODIFIED SIGNAL: No parameters needed for native viz ---
    request_visualization = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Image Search")
        
        # --- Smart window placement logic ---
        # Get the primary screen from the application
        screen = QApplication.primaryScreen()
        if screen:
            # Get the available geometry (excludes taskbar, etc.)
            screen_geometry = screen.availableGeometry()
            screen_width = screen_geometry.width()
            screen_height = screen_geometry.height()

            win_width = int(screen_width * 6 / 7) - 50
            win_height = int(screen_height * 6 / 7) - 50
            
            # Set the window geometry
            self.setGeometry(50, 50, win_width, win_height)
        else:
            # Fallback for systems with no primary screen detected
            self.setGeometry(50, 50, 800, 600)
        # --- End of new placement logic ---
        
        self.worker_thread = QThread()
        self.backend_worker = BackendWorker()
        self.backend_worker.moveToThread(self.worker_thread)
        
        self.query_image_path = None
        
        self._init_ui()
        self._connect_signals()
        
        self.worker_thread.start()

    def _init_ui(self):
        # This method is modified to add the new SpinBox and Visualize button
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

        # --- ADDED: UI for configuring max results ---
        self.results_count_label = QLabel("Max Results:")
        self.results_spinbox = QSpinBox()
        self.results_spinbox.setRange(1, 500)
        self.results_spinbox.setValue(50)
        self.results_spinbox.setToolTip("Set the maximum number of images to return in a search.")
        
        # --- ADDED: Visualize Button ---
        self.visualize_btn = QPushButton("Visualize Embeddings")
        self.visualize_btn.setToolTip("Generate an interactive 2D UMAP visualization of all embeddings.")

        controls_layout.addStretch() # Pushes the following widgets to the right
        controls_layout.addWidget(self.results_count_label)
        controls_layout.addWidget(self.results_spinbox)
        controls_layout.addWidget(self.visualize_btn)
        # --- END OF ADDED UI ---
        
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
        self.results_model = QStandardItemModel()
        self.results_view.setModel(self.results_model)
        self.results_view.setItemDelegate(SearchResultDelegate(self))
        self.results_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        self.loading_label = QLabel("Searching, please wait...")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_label.setStyleSheet("font-size: 16px; color: #aaa;")
        
        # --- ADDED: Native Visualizer Widget ---
        self.visualizer_widget = QtVisualizer()

        self.content_stack.addWidget(self.init_label)
        self.content_stack.addWidget(self.results_view)
        self.content_stack.addWidget(self.loading_label)
        self.content_stack.addWidget(self.visualizer_widget) # Add native visualizer
        
        self.content_stack.setCurrentWidget(self.init_label)
        
        main_layout.addWidget(self.content_stack)

        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Starting backend thread...")
        self.search_bar.setEnabled(False)
        self.text_search_btn.setEnabled(False)
        self.image_search_btn.setEnabled(False)
        self.results_spinbox.setEnabled(False) 
        self.visualize_btn.setEnabled(False) # Disable button during init

    def _connect_signals(self):
        # UI Input connections
        self.worker_thread.started.connect(self.backend_worker.initialize)
        self.search_bar.returnPressed.connect(self.start_text_search)
        self.text_search_btn.clicked.connect(self.start_text_search)
        self.image_search_btn.clicked.connect(self.select_image_for_search)
        self.visualize_btn.clicked.connect(self.start_visualization) # NEW CONNECTION
        
        # Request connections (UI -> Backend Worker)
        self.request_text_search.connect(self.backend_worker.perform_text_search)
        self.request_image_search.connect(self.backend_worker.perform_image_search)
        self.request_visualization.connect(self.backend_worker.request_visualization_data) 
        
        # Response connections (Backend Worker -> UI)
        self.backend_worker.initialized.connect(self.on_backend_initialized)
        self.backend_worker.error.connect(self.on_backend_error)
        self.backend_worker.results_ready.connect(self.display_results)
        self.backend_worker.status_update.connect(self.statusBar().showMessage)
        self.backend_worker.visualization_data_ready.connect(self.on_visualization_data_ready)
        self.visualizer_widget.data_loaded.connect(self.on_visualization_loaded) # Native widget loaded signal
        
        # Cleanup connections
        self.worker_thread.finished.connect(self.backend_worker.shutdown)
        self.worker_thread.finished.connect(self.backend_worker.deleteLater)
        self.results_view.customContextMenuRequested.connect(self.on_results_context_menu)

    def on_backend_initialized(self):
        # This method is modified to enable the spinbox and visualize button
        logger.info("Backend initialized successfully.")
        
        self.content_stack.setCurrentWidget(self.results_view)
        
        self.statusBar().showMessage("Backend ready. You can now search or visualize.")
        self.search_bar.setEnabled(True)
        self.text_search_btn.setEnabled(True)
        self.image_search_btn.setEnabled(True)
        self.results_spinbox.setEnabled(True)
        self.visualize_btn.setEnabled(True) # ENABLE VISUALIZE BUTTON

    def on_backend_error(self, err_msg):
        # This method is unchanged
        logger.error(f"Backend error: {err_msg}")
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setText("A critical error occurred in the backend.")
        msg_box.setInformativeText("The application may not function correctly. Please check the console for details.")
        msg_box.setDetailedText(err_msg)
        msg_box.exec()
        self.statusBar().showMessage("Backend failed to initialize. Please restart.")
        self.init_label.setText("Initialization Failed. Please restart.")
        
    # --- ADDED METHOD: Visualization Workflow Start ---
    def start_visualization(self):
        self.clear_results()
        self.content_stack.setCurrentWidget(self.loading_label)
        self.loading_label.setText("Calculating UMAP coordinates and clusters...")
        self.visualize_btn.setEnabled(False)
        self.statusBar().showMessage("Starting visualization calculation...")
        
        # Request data calculation from the worker thread
        self.request_visualization.emit()

    # --- ADDED METHOD: Visualization Workflow Data Ready (Native) ---
    @Slot(list)
    def on_visualization_data_ready(self, plot_data: list):
        if not plot_data:
            # Error was handled in backend, just revert UI state
            self.content_stack.setCurrentWidget(self.results_view)
            self.visualize_btn.setEnabled(True)
            return

        # Pass the calculated data to the native pyqtgraph widget
        self.visualizer_widget.load_plot_data(plot_data)

    # --- ADDED METHOD: Visualization Workflow Load Complete ---
    @Slot(int)
    def on_visualization_loaded(self, count: int):
        self.content_stack.setCurrentWidget(self.visualizer_widget)
        self.loading_label.setText("Searching, please wait...") # Reset loading text
        self.visualize_btn.setEnabled(True)
        self.statusBar().showMessage(f"Visualization complete. Plotted {count} points.")

    def start_text_search(self):
        # This method is modified to send the number of results
        query = self.search_bar.text()
        if not query: return
        self.clear_results()
        self.content_stack.setCurrentWidget(self.loading_label)
        self.text_search_btn.setEnabled(False)
        self.image_search_btn.setEnabled(False)
        self.visualize_btn.setEnabled(False) # Disable during search
        # --- MODIFIED --- Get value from spinbox and emit it with the signal.
        max_results = self.results_spinbox.value()
        self.request_text_search.emit(query, max_results)

    def select_image_for_search(self):
        # This method is unchanged
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Query Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if filepath:
            self.execute_image_search(filepath)
    
    def execute_image_search(self, filepath: str):
        # This method is modified to send the number of results
        self.query_image_path = filepath
        self.selected_image_label.setText(f"Selected: ...{Path(filepath).name}")
        
        self.clear_results()
        self.content_stack.setCurrentWidget(self.loading_label)
        self.text_search_btn.setEnabled(False)
        self.image_search_btn.setEnabled(False)
        self.visualize_btn.setEnabled(False) # Disable during search
        # --- MODIFIED --- Get value from spinbox and emit it with the signal.
        max_results = self.results_spinbox.value()
        self.request_image_search.emit(self.query_image_path, max_results)

    def display_results(self, results: list):
        # This method is unchanged
        self.content_stack.setCurrentWidget(self.results_view)
        self.clear_results()

        if not results:
            self.statusBar().showMessage("Search complete. No results found.")
        else:
            for score, path in results:
                item = create_list_item(score, path)
                self.results_model.appendRow(item)
            
            self.statusBar().showMessage(f"Search complete. Found {len(results)} results.")
        
        self.text_search_btn.setEnabled(True)
        self.image_search_btn.setEnabled(True)
        self.visualize_btn.setEnabled(True) # Re-enable after search
        
    def clear_results(self):
        # This method is unchanged
        self.results_model.clear()

    def on_results_context_menu(self, pos):
        # This method is modified to change the action text
        index = self.results_view.indexAt(pos)
        if not index.isValid():
            return

        filepath = index.data(FILEPATH_ROLE)
        
        context_menu = QMenu(self)
        
        find_similar_action = QAction("Find Similar", self)
        find_similar_action.triggered.connect(lambda: self.execute_image_search(filepath))
        context_menu.addAction(find_similar_action)
        
        context_menu.addSeparator()

        # --- MODIFIED --- Action text is now more descriptive.
        copy_path_action = QAction("Copy Full Path", self)
        copy_path_action.triggered.connect(lambda: self.copy_filepath_to_clipboard(filepath))
        context_menu.addAction(copy_path_action)

        copy_image_action = QAction("Copy Image", self)
        copy_image_action.triggered.connect(lambda: self.copy_image_to_clipboard(filepath))
        context_menu.addAction(copy_image_action)
        
        context_menu.exec(self.results_view.viewport().mapToGlobal(pos))

    # --- MODIFIED --- Function renamed for clarity and behavior changed.
    def copy_filepath_to_clipboard(self, filepath: str):
        # --- MODIFIED --- Now copies the full path, not just the filename.
        QApplication.clipboard().setText(filepath)
        logger.info(f"Copied full path to clipboard: {filepath}")

    def copy_image_to_clipboard(self, filepath: str):
        # This method is unchanged
        from PySide6.QtGui import QPixmap
        pixmap = QPixmap(filepath)
        if not pixmap.isNull():
            QApplication.clipboard().setPixmap(pixmap)
            logger.info(f"Copied image to clipboard: {filepath}")
        else:
            logger.warning(f"Could not load image for clipboard: {filepath}")

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
    
    # This is a safe way to set a global reference to the main window
    # for the QtVisualizer to access the status bar.
    class CustomApplication(QApplication):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._main_window = None

        def setMainWindow(self, window):
            self._main_window = window

        def mainWindow(self):
            return self._main_window
            
    logger.info("Starting AI Image Search application.")
    app = CustomApplication(sys.argv)
    try:
        import qdarktheme
        app.setStyleSheet(qdarktheme.load_stylesheet())
    except ImportError:
        logger.info("Theme library not found. For a dark theme, install with: pip install pyqtdarktheme")
        
    window = MainWindow()
    app.setMainWindow(window) # Set the main window reference
    
    window.show()
    sys.exit(app.exec())
    logger.info("Quitting AI Image Search application.")