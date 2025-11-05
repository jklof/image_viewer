import logging
import typing
import traceback

from PySide6.QtCore import QObject, QThread, Slot, Signal

from backend import BackendWorker
from sync_worker import SyncWorker

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
    request_composite_search = Signal(list)
    request_random_search = Signal()
    request_visualization = Signal()
    request_shutdown = Signal()
    request_reload = Signal()  # Signal to the BackendWorker to reload its data

    def __init__(self, main_window: "MainWindow", use_cpu_only: bool = False):
        super().__init__()

        if not main_window:
            raise ValueError("AppController requires a MainWindow instance.")

        self.window = main_window

        # --- Backend Thread Management ---
        self.worker_thread = QThread()
        self.backend_worker = BackendWorker(use_cpu_only=use_cpu_only)
        self.backend_worker.moveToThread(self.worker_thread)

        # --- Sync Thread Management ---
        self.sync_thread = None
        self.sync_worker = None

        self._connect_signals()

    def initialize_app(self):
        """Starts the backend thread and the initialization process."""
        self.window.update_status_bar("Starting backend thread...")
        self.worker_thread.start()

    def _connect_signals(self):
        """Connects signals from the view and backend to the controller's slots."""
        # --- View to Controller ---
        self.window.composite_search_triggered.connect(self.on_composite_search_requested)
        self.window.visualization_triggered.connect(self.on_visualization_requested)
        self.window.closing.connect(self.on_main_window_closing)
        self.window.random_order_triggered.connect(self.on_random_order_requested)
        self.window.sync_triggered.connect(self.on_sync_requested)
        self.window.sync_cancel_triggered.connect(self.on_sync_cancel_requested)

        # --- Controller to Backend ---
        self.worker_thread.started.connect(self.backend_worker.initialize)
        self.request_composite_search.connect(self.backend_worker.perform_composite_search)
        self.request_random_search.connect(self.backend_worker.perform_random_search)
        self.request_visualization.connect(self.backend_worker.request_visualization_data)
        self.request_shutdown.connect(self.backend_worker.shutdown)
        self.request_reload.connect(self.backend_worker.perform_reload)

        # --- Backend to Controller ---
        self.backend_worker.initialized.connect(self.on_backend_initialized)
        self.backend_worker.error.connect(self.on_backend_error)
        self.backend_worker.results_ready.connect(self.on_results_ready)
        self.backend_worker.status_update.connect(self.window.update_status_bar)
        self.backend_worker.visualization_data_ready.connect(self.on_visualization_data_ready)
        self.backend_worker.reloaded.connect(self.on_backend_reloaded)

        # --- Visualization Widget to Controller/View ---
        self.window.visualizer_widget.data_loaded.connect(self.on_visualization_loaded)
        self.window.visualizer_widget.status_update.connect(self.window.update_status_bar)

    @Slot()
    def on_backend_initialized(self):
        self.window.clear_results()
        self.window.show_loading_state("Loading initial random order...")
        self.window.set_controls_enabled(False)
        self.request_random_search.emit()

    @Slot(str)
    def on_backend_error(self, error_message: str):
        self.window.show_critical_error_state()
        self.window.update_status_bar("Backend failed to initialize. Please restart.")
        self.window.show_critical_error("Backend Error", f"A critical error occurred: {error_message}")

    @Slot(list)
    def on_composite_search_requested(self, query_elements: list):
        if not query_elements:
            self.window.show_loading_state("No query elements provided, showing random order...")
            self.request_random_search.emit()
            return

        self.window.clear_results()
        self.window.show_loading_state("Constructing query and ordering images...")
        self.window.set_controls_enabled(False)
        self.request_composite_search.emit(query_elements)

    @Slot()
    def on_random_order_requested(self):
        self.window.clear_results()
        self.window.show_loading_state("Randomly reordering images...")
        self.window.set_controls_enabled(False)
        self.request_random_search.emit()

    @Slot()
    def on_visualization_requested(self):
        self.window.show_loading_state("Loading visualization data...")
        self.window.set_controls_enabled(False)
        self.request_visualization.emit()

    @Slot()
    def on_sync_requested(self):
        if self.sync_thread and self.sync_thread.isRunning():
            logger.warning("Sync requested, but one is already in progress.")
            return

        logger.info("Sync process requested by user.")
        self.window.set_sync_controls_enabled(False)
        self.window.show_sync_active_view()

        self.sync_thread = QThread()
        self.sync_worker = SyncWorker(use_cpu_only=self.backend_worker.use_cpu_only)
        self.sync_worker.moveToThread(self.sync_thread)

        # Connect worker signals
        self.sync_worker.status_update.connect(self.window.update_sync_status)
        self.sync_worker.progress_update.connect(self.window.update_sync_progress)
        self.sync_worker.finished.connect(self.on_sync_finished)
        self.sync_worker.error.connect(self.on_backend_error)  # Can reuse the same error handler
        self.sync_thread.started.connect(self.sync_worker.run)

        self.sync_thread.start()

    @Slot()
    def on_sync_cancel_requested(self):
        if self.sync_worker:
            logger.info("Controller forwarding cancel request to sync worker.")
            self.sync_worker.cancel()
            self.window.update_sync_status("Cancelling...")
            self.window.set_sync_cancel_button_enabled(False)

    @Slot(str, str)
    def on_sync_finished(self, result: str, message: str):
        logger.info(f"Sync process finished. Result: {result}, Message: {message}")
        self.window.update_status_bar(message)

        # Clean up the thread and worker
        if self.sync_thread:
            self.sync_thread.quit()
            self.sync_thread.wait()
            self.sync_thread = None
            self.sync_worker = None

        # Always reload data to ensure the UI reflects the on-disk database state,
        # regardless of the sync outcome (success, cancelled, or failed).
        self.window.update_sync_status("Reloading data...")
        self.request_reload.emit()

    @Slot()
    def on_backend_reloaded(self):
        logger.info("Backend has confirmed data reload. Refreshing view.")
        self.window.show_sync_idle_view()
        self.window.set_sync_controls_enabled(True)
        # Trigger a new random search to show the new/updated data
        self.on_random_order_requested()

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
        logger.info("Controller received close signal. Shutting down worker threads.")
        if self.sync_thread and self.sync_thread.isRunning():
            logger.info("Window closing, cancelling active sync.")
            self.on_sync_cancel_requested()
            self.sync_thread.quit()
            if not self.sync_thread.wait(2000):
                logger.warning("Sync worker thread did not shut down cleanly. Terminating.")
                self.sync_thread.terminate()

        self.request_shutdown.emit()
        self.worker_thread.quit()
        if not self.worker_thread.wait(5000):
            logger.warning("Backend worker thread did not shut down cleanly. Terminating.")
            self.worker_thread.terminate()
        else:
            logger.info("Backend worker thread shut down successfully.")
