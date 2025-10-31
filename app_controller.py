import logging
import typing
import traceback

from PySide6.QtCore import QObject, QThread, Slot, Signal

from backend import BackendWorker

# Forward-declaring the view type for type hinting to avoid circular imports
if typing.TYPE_CHECKING:
    from main_window import MainWindow

logger = logging.getLogger(__name__)


class AppController(QObject):
    """
    The central controller for the application.
    Owns the backend worker and orchestrates the flow of data and state
    changes between the view (MainWindow) and the backend.
    """

    # --- Signals to communicate with the worker thread ---
    request_text_search = Signal(str)
    request_image_search = Signal(str)
    request_visualization = Signal()
    request_shutdown = Signal()

    def __init__(self, main_window: "MainWindow"):
        super().__init__()

        if not main_window:
            raise ValueError("AppController requires a MainWindow instance.")

        self.window = main_window

        # --- Backend Thread Management ---
        self.worker_thread = QThread()
        self.backend_worker = BackendWorker()
        self.backend_worker.moveToThread(self.worker_thread)

        self._connect_signals()

    def initialize_app(self):
        """Starts the backend thread and the initialization process."""
        self.window.update_status_bar("Starting backend thread...")
        self.worker_thread.start()

    def _connect_signals(self):
        """Connects signals from the view and backend to the controller's slots."""
        # --- View to Controller ---
        self.window.text_search_triggered.connect(self.on_text_search_requested)
        self.window.image_search_triggered.connect(self.on_image_search_requested)
        self.window.visualization_triggered.connect(self.on_visualization_requested)
        self.window.closing.connect(self.on_main_window_closing)

        # --- Controller to Backend ---
        # Connect the new signals to the worker's slots. This ensures the calls
        # are queued and executed in the worker thread.
        self.worker_thread.started.connect(self.backend_worker.initialize)
        self.request_text_search.connect(self.backend_worker.perform_text_search)
        self.request_image_search.connect(self.backend_worker.perform_image_search)
        self.request_visualization.connect(self.backend_worker.request_visualization_data)
        self.request_shutdown.connect(self.backend_worker.shutdown)

        # --- Backend to Controller ---
        self.backend_worker.initialized.connect(self.on_backend_initialized)
        self.backend_worker.error.connect(self.on_backend_error)
        self.backend_worker.results_ready.connect(self.on_results_ready)
        self.backend_worker.status_update.connect(self.window.update_status_bar)
        self.backend_worker.visualization_data_ready.connect(self.on_visualization_data_ready)

        # --- Visualization Widget to Controller/View ---
        self.window.visualizer_widget.data_loaded.connect(self.on_visualization_loaded)
        self.window.visualizer_widget.status_update.connect(self.window.update_status_bar)

    @Slot()
    def on_backend_initialized(self):
        self.window.show_results_view()
        self.window.set_controls_enabled(True)
        self.window.update_status_bar("Backend ready. Enter text or select an image to explore.")

    @Slot(str)
    def on_backend_error(self, error_message: str):
        self.window.show_critical_error("Backend Error", f"A critical error occurred: {error_message}")
        self.window.update_status_bar("Backend failed to initialize. Please restart.")
        self.window.show_initialization_failed_state()

    @Slot(str)
    def on_text_search_requested(self, query: str):
        if not query:
            return
        self.window.clear_results()
        self.window.show_loading_state("Ordering all images, please wait...")
        self.window.set_controls_enabled(False)
        self.request_text_search.emit(query)

    @Slot(str)
    def on_image_search_requested(self, image_path: str):
        self.window.update_selected_image_label(image_path)
        self.window.clear_results()
        self.window.show_loading_state("Ordering all images, please wait...")
        self.window.set_controls_enabled(False)
        self.request_image_search.emit(image_path)

    @Slot()
    def on_visualization_requested(self):
        self.window.clear_results()
        self.window.show_loading_state("Loading visualization data...")
        self.window.set_controls_enabled(False)
        self.request_visualization.emit()

    @Slot(list)
    def on_results_ready(self, results: list):
        self.window.set_results_data(results)
        self.window.show_results_view()
        self.window.update_status_bar(f"Ordering complete. Displaying all {len(results)} images.")
        self.window.set_controls_enabled(True)

    @Slot(list)
    def on_visualization_data_ready(self, plot_data: list):
        if not plot_data:
            self.window.show_results_view()
            self.window.set_controls_enabled(True)
            self.window.update_status_bar("Visualization failed: No images found to plot.")
            return
        self.window.visualizer_widget.load_plot_data(plot_data)

    @Slot(int)
    def on_visualization_loaded(self, count: int):
        self.window.show_visualizer_view()
        self.window.set_controls_enabled(True)
        self.window.update_status_bar(f"Visualization complete. Plotted {count} points.")

    @Slot()
    def on_main_window_closing(self):
        """Cleanly shuts down the backend thread."""
        logger.info("Controller received close signal. Shutting down backend worker.")
        self.request_shutdown.emit()
        self.worker_thread.quit()
        if not self.worker_thread.wait(5000):
            logger.warning("Backend worker thread did not shut down cleanly. Terminating.")
            self.worker_thread.terminate()
        else:
            logger.info("Backend worker thread shut down successfully.")
