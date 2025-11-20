import logging
import queue
import threading

from PySide6.QtCore import QObject, QThread, Slot, Signal

from backend import BackendWorker, BackendSignals
from sync_worker import SyncWorker

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from main_window import MainWindow

logger = logging.getLogger(__name__)


class AppController(QObject):
    """
    The central controller for the application.
    Orchestrates UI, a background sync worker, and a background backend worker.
    """

    def __init__(self, main_window: "MainWindow", use_cpu_only: bool = False):
        super().__init__()

        if not main_window:
            raise ValueError("AppController requires a MainWindow instance.")

        self.window = main_window

        self.backend_job_queue = queue.Queue()
        self.backend_signals = BackendSignals()
        self.backend_worker = BackendWorker(
            signals=self.backend_signals, job_queue=self.backend_job_queue, use_cpu_only=use_cpu_only
        )
        self.backend_thread = threading.Thread(target=self.backend_worker.run, daemon=True)

        self.sync_thread = None
        self.sync_worker = None
        self._cached_visualization_data = None
        self._visualization_data_dirty = True

        self._connect_signals()

    def initialize_app(self):
        """Starts the backend thread."""
        self.window.update_status_bar("Starting backend thread...")
        self.backend_thread.start()

    def _connect_signals(self):
        """Connects signals from the view and backend to the controller's slots."""
        # View to Controller
        self.window.composite_search_triggered.connect(self.on_composite_search_requested)
        self.window.visualization_triggered.connect(self.on_visualization_requested)
        self.window.closing.connect(self.on_main_window_closing)
        self.window.random_order_triggered.connect(self.on_random_order_requested)
        self.window.sort_by_date_triggered.connect(self.on_sort_by_date_requested)
        self.window.sync_triggered.connect(self.on_sync_requested)
        self.window.sync_cancel_triggered.connect(self.on_sync_cancel_requested)

        # Backend Signals to Controller Slots
        self.backend_signals.initialized.connect(self.on_backend_initialized)
        self.backend_signals.error.connect(self.on_backend_error)
        self.backend_signals.results_ready.connect(self.on_results_ready)
        self.backend_signals.status_update.connect(self.window.update_status_bar)
        self.backend_signals.visualization_data_ready.connect(self.on_visualization_data_ready)
        self.backend_signals.reloaded.connect(self.on_backend_reloaded)

        # Visualization Widget
        self.window.visualizer_widget.data_loaded.connect(self.on_visualization_loaded)
        self.window.visualizer_widget.status_update.connect(self.window.update_status_bar)

    @Slot()
    def on_backend_initialized(self):
        self.on_random_order_requested()

    @Slot(str)
    def on_backend_error(self, error_message: str):
        self.window.show_critical_error_state()
        self.window.update_status_bar("Backend failed. Please restart.")
        self.window.show_critical_error("Backend Error", f"A critical error occurred: {error_message}")

    @Slot(list)
    def on_composite_search_requested(self, query_elements: list):
        self.window.clear_results()
        self.window.show_loading_state("Constructing query...")
        self.window.set_controls_enabled(False)
        self.backend_job_queue.put(("composite_search", query_elements))

    @Slot()
    def on_random_order_requested(self):
        self.window.clear_results()
        self.window.show_loading_state("Randomly reordering...")
        self.window.set_controls_enabled(False)
        self.backend_job_queue.put(("random_search", None))

    @Slot()
    def on_sort_by_date_requested(self):
        self.window.clear_results()
        self.window.show_loading_state("Sorting images by date...")
        self.window.set_controls_enabled(False)
        self.backend_job_queue.put(("sort_by_date", None))

    @Slot()
    def on_visualization_requested(self):
        if not self._visualization_data_dirty and self._cached_visualization_data is not None:
            self.window.show_visualizer_view()
            self.window.update_status_bar(
                f"Visualization ready. Plotted {len(self._cached_visualization_data)} points."
            )
            return
        self.window.show_loading_state("Loading visualization data...")
        self.window.set_controls_enabled(False)
        self.backend_job_queue.put(("visualization_data", None))

    @Slot(list)
    def on_results_ready(self, results: list):
        if not results:
            self.window.show_sync_prompt_view()
            return

        self.window.set_results_data(results)
        self.window.show_results_view()
        self.window.update_status_bar(f"Ordering complete. Displaying all {len(results)} images.")
        self.window.set_controls_enabled(True)
        self.window.set_sync_controls_enabled(True)

    @Slot(list)
    def on_visualization_data_ready(self, plot_data: list):
        self._cached_visualization_data = plot_data
        self._visualization_data_dirty = False
        if not plot_data:
            self.window.show_results_view()
            self.window.set_controls_enabled(True)
            self.window.update_status_bar("Visualization failed: No images to plot.")
            return
        self.window.visualizer_widget.load_plot_data(plot_data)

    @Slot(int)
    def on_visualization_loaded(self, count: int):
        self.window.show_visualizer_view()
        self.window.set_controls_enabled(True)
        self.window.update_status_bar(f"Visualization complete. Plotted {count} points.")

    @Slot()
    def on_sync_requested(self):
        if self.sync_thread and self.sync_thread.isRunning():
            return

        # Explicitly disable visualization during sync to prevent DB locking
        # and wasted computation.
        self.window.visualize_btn.setEnabled(False)

        self.window.set_sync_controls_enabled(False)
        self.window.show_sync_active_view()
        self.window.update_status_bar("Sync started. You can continue searching on existing data.")
        self.sync_thread = QThread()
        self.sync_worker = SyncWorker(use_cpu_only=self.backend_worker.use_cpu_only)
        self.sync_worker.moveToThread(self.sync_thread)
        self.sync_worker.status_update.connect(self.window.update_sync_status)
        self.sync_worker.progress_update.connect(self.window.update_sync_progress)
        self.sync_worker.finished.connect(self.on_sync_finished)
        self.sync_worker.error.connect(self.on_backend_error)

        # Connect the QThread's own finished signal for safe, asynchronous cleanup.
        self.sync_thread.finished.connect(self._on_sync_thread_finished)

        self.sync_thread.started.connect(self.sync_worker.run)
        self.sync_thread.start()

    @Slot()
    def on_sync_cancel_requested(self):
        if self.sync_worker:
            self.sync_worker.cancel()
            self.window.update_sync_status("Cancelling...")
            self.window.set_sync_cancel_button_enabled(False)

    @Slot(str, str)
    def on_sync_finished(self, result: str, message: str):
        logger.info(f"Sync finished with result: {result}, message: {message}")
        self.window.update_status_bar(message)
        self._visualization_data_dirty = True
        self._cached_visualization_data = None

        # 1. Tell the thread to quit. This is a non-blocking request.
        if self.sync_thread:
            self.sync_thread.quit()
        # 2. ALWAYS reload, because DB state might have partially changed
        # even on cancel/error.
        self.window.update_sync_status("Reloading data...")
        self.backend_job_queue.put(("reload", None))

    @Slot()
    def _on_sync_thread_finished(self):
        """
        This slot is connected to the QThread.finished signal for safe, asynchronous cleanup.
        It runs on the main thread only after the sync thread's event loop has fully terminated.
        """
        logger.info("Sync thread has finished. Cleaning up worker and thread objects.")
        # By setting these to None, we allow a new sync operation to start.
        self.sync_worker = None
        self.sync_thread = None

    @Slot()
    def on_backend_reloaded(self):
        logger.info("Backend reloaded data.")
        self.window.show_sync_idle_view()
        self.window.set_sync_controls_enabled(True)
        self.window.update_status_bar("Data reloaded. Displaying updated images.")

        # Now that sync is done and data is loaded, it is safe to visualize.
        self.window.visualize_btn.setEnabled(True)

        self.on_random_order_requested()

    @Slot()
    def on_main_window_closing(self):
        logger.info("Controller received close signal. Shutting down worker threads.")
        if self.sync_thread and self.sync_thread.isRunning():
            self.on_sync_cancel_requested()
            self.sync_thread.quit()
            # The wait() is acceptable here because the entire app is closing.
            if not self.sync_thread.wait(2000):
                self.sync_thread.terminate()
        self.backend_worker.shutdown()
        self.backend_thread.join(timeout=5)
        logger.info("Backend worker thread shut down.")
