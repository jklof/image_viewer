import logging
import queue
import threading
import os
import shutil
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Slot, Signal, QTimer
from PySide6.QtWidgets import QFileDialog, QMessageBox

from backend import BackendWorker, BackendSignals
from sync_worker import SyncWorker
from config_utils import get_scan_directories, save_config, load_config

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from main_window import MainWindow

logger = logging.getLogger(__name__)




class AppController(QObject):
    """
    The central controller for the application.
    Orchestrates UI, a background sync worker, and a background backend worker.
    """

    # Signal to indicate when preferences are saved
    preferences_saved = Signal(bool)

    # Signals for background thread UI updates
    move_completed = Signal(str)
    delete_completed = Signal(list, str)

    def __init__(self, main_window: "MainWindow", use_cpu_only: bool = False):
        super().__init__()

        if not main_window:
            raise ValueError("AppController requires a MainWindow instance.")

        self.window = main_window
        self._use_cpu_only = use_cpu_only  # Store for soft restart

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
        self._tagged_only_filter = False
        self._current_results_generation = 0
        self._pending_scroll_state = None
        self._pending_single_view_fp = None

        self._connect_signals()

    def initialize_app(self):
        """Starts the backend thread."""
        self.window.update_status_bar("Starting backend thread...")
        self.backend_thread.start()

    def _connect_signals(self):
        """Connects signals from the view and backend to the controller's slots."""
        # Connect internal controller signals
        self.move_completed.connect(self._on_move_completed)
        self.delete_completed.connect(self._on_delete_completed)

        # View to Controller
        self.window.composite_search_triggered.connect(self.on_composite_search_requested)
        self.window.visualization_triggered.connect(self.on_visualization_requested)
        self.window.closing.connect(self.on_main_window_closing)
        self.window.random_order_triggered.connect(self.on_random_order_requested)
        self.window.sort_by_date_triggered.connect(self.on_sort_by_date_requested)
        self.window.sync_triggered.connect(self.on_sync_requested)
        self.window.sync_cancel_triggered.connect(self.on_sync_cancel_requested)

        # Add the new preferences_saved signal connection
        self.window.preferences_saved.connect(self.on_preferences_saved)

        # Backend Signals to Controller Slots
        self.backend_signals.initialized.connect(self.on_backend_initialized)
        self.backend_signals.error.connect(self.on_backend_error)
        self.backend_signals.warning.connect(self.on_backend_warning)
        self.backend_signals.results_ready.connect(self.on_results_ready)
        self.backend_signals.status_update.connect(self.window.update_status_bar)
        self.backend_signals.visualization_data_ready.connect(self.on_visualization_data_ready)
        self.backend_signals.reloaded.connect(self.on_backend_reloaded)
        self.backend_signals.tag_operation_failed.connect(self.on_tag_operation_failed)
        self.backend_signals.deletion_completed.connect(self.on_backend_deletion_completed)
        self.backend_signals.duplicate_paths_ready.connect(self.window._on_duplicate_paths_ready)
        self.backend_signals.source_resolution_ready.connect(self.on_source_resolution_ready)

        # Duplicate management
        self.window.dedup_info_requested.connect(self.on_dedup_info_requested)
        self.window.manage_duplicates_requested.connect(self.on_manage_duplicates_requested)

        # ComfyUI source lineage (window -> controller only; never re-connect
        # on soft restart, same rule as duplicate management above)
        self.window.resolve_sources_requested.connect(self.on_resolve_sources_requested)

        # Visualization Widget
        self.window.visualizer_widget.data_loaded.connect(self.on_visualization_loaded)
        self.window.visualizer_widget.status_update.connect(self.window.update_status_bar)

        # Tagging signals
        self.window.toggle_tags_requested.connect(self.on_toggle_tags_requested)
        self.window.untag_all_requested.connect(self.on_untag_all_requested)
        self.window.move_tagged_requested.connect(self.on_move_tagged_requested)
        self.window.delete_tagged_requested.connect(self.on_delete_tagged_requested)
        self.window.show_tagged_only_btn.toggled.connect(self.on_show_tagged_only_toggled)

    def _disconnect_backend_signals(self, signals):
        """Disconnect all backend signal slots for the given signals object."""
        try:
            signals.initialized.disconnect(self.on_backend_initialized)
            signals.error.disconnect(self.on_backend_error)
            signals.warning.disconnect(self.on_backend_warning)
            signals.results_ready.disconnect(self.on_results_ready)
            signals.status_update.disconnect(self.window.update_status_bar)
            signals.visualization_data_ready.disconnect(self.on_visualization_data_ready)
            signals.reloaded.disconnect(self.on_backend_reloaded)
            signals.tag_operation_failed.disconnect(self.on_tag_operation_failed)
            signals.deletion_completed.disconnect(self.on_backend_deletion_completed)
            signals.duplicate_paths_ready.disconnect(self.window._on_duplicate_paths_ready)
            signals.source_resolution_ready.disconnect(self.on_source_resolution_ready)
        except (RuntimeError, AttributeError):
            pass  # Ignore disconnection errors

    @Slot(bool)
    def on_preferences_saved(self, requires_restart: bool):
        """
        Handle soft restart when preferences are saved that require backend restart.

        Args:
            requires_restart: True if database path or model ID was changed
        """
        if not requires_restart:
            return  # No restart needed for directory changes

        logger.info("Soft restart initiated due to database/model changes.")

        # Store old signals before reassignment
        old_signals = self.backend_signals

        # Disconnect old signals immediately BEFORE shutting down thread to prevent race conditions
        self._disconnect_backend_signals(old_signals)

        # Disable controls during restart
        self.window.set_controls_enabled(False)

        # Cancel and wait for any running sync operation
        if self.sync_thread and self.sync_thread.isRunning():
            logger.info("Cancelling active sync operation for soft restart...")
            self.on_sync_cancel_requested()
            # Wait for sync thread to fully terminate to avoid embedder race
            if not self.sync_thread.wait(15000):  # Wait up to 15 seconds
                logger.warning("Sync thread did not exit gracefully, forcing termination")
                self.sync_thread.terminate()
                self.sync_thread.wait() # Ensure it's dead
            self.sync_thread = None
            self.sync_worker = None

        # Shutdown current backend
        logger.info("Shutting down current backend worker...")
        self.backend_worker.shutdown()
        # Ensure backend thread is fully joined before proceeding
        self.backend_thread.join(timeout=10)

        # Check if backend thread is still alive (race condition protection)
        if self.backend_thread.is_alive():
            logger.error("Backend thread did not shut down gracefully within timeout.")
            self.window.show_critical_error(
                "Restart Failed", "The backend thread did not shut down properly. Please restart the application."
            )
            return

        # Re-initialize backend components with new configuration
        logger.info("Re-initializing backend with new configuration...")
        self.backend_job_queue = queue.Queue()
        self.backend_signals = BackendSignals()
        self.backend_worker = BackendWorker(
            signals=self.backend_signals, job_queue=self.backend_job_queue, use_cpu_only=self._use_cpu_only
        )
        self.backend_thread = threading.Thread(target=self.backend_worker.run, daemon=True)

        # Reconnect signals for the new backend
        self._connect_backend_signals()

        # Restart backend thread
        logger.info("Starting new backend worker thread...")
        self.backend_thread.start()

        # Clear cached data as it may be invalid with new configuration
        self._cached_visualization_data = None
        self._visualization_data_dirty = True
        self._pending_single_view_fp = None
        self._pending_scroll_state = None

        logger.info("Soft restart completed successfully.")

    def _connect_backend_signals(self):
        """Reconnect backend signals after soft restart.

        NOTE: window -> controller connections (dedup, tagging, etc.) live
        ONLY in _connect_signals and must NOT be re-connected here, otherwise
        every soft restart adds a duplicate slot (e.g. triple file deletion).
        """
        # Connect new backend signals
        self.backend_signals.initialized.connect(self.on_backend_initialized)
        self.backend_signals.error.connect(self.on_backend_error)
        self.backend_signals.warning.connect(self.on_backend_warning)
        self.backend_signals.results_ready.connect(self.on_results_ready)
        self.backend_signals.status_update.connect(self.window.update_status_bar)
        self.backend_signals.visualization_data_ready.connect(self.on_visualization_data_ready)
        self.backend_signals.reloaded.connect(self.on_backend_reloaded)
        self.backend_signals.tag_operation_failed.connect(self.on_tag_operation_failed)
        self.backend_signals.deletion_completed.connect(self.on_backend_deletion_completed)
        self.backend_signals.duplicate_paths_ready.connect(self.window._on_duplicate_paths_ready)
        self.backend_signals.source_resolution_ready.connect(self.on_source_resolution_ready)

    @Slot()
    def on_backend_initialized(self):
        self.on_sort_by_date_requested()

    @Slot(str)
    def on_backend_error(self, error_message: str):
        self.window.show_critical_error_state()
        self.window.update_status_bar("Backend failed. Please restart.")
        self.window.show_critical_error("Backend Error", f"A critical error occurred: {error_message}")

    @Slot(str)
    def on_backend_warning(self, message: str):
        """Displays a non-fatal warning to the user without locking the UI."""
        self.window.update_status_bar(message)
        QMessageBox.warning(self.window, "Search Failed", message)

    def _capture_single_view_anchor(self):
        if self.window.content_stack.currentWidget() is self.window.single_image_view_widget:
            self._pending_single_view_fp = self.window.single_image_view_widget.current_filepath
        else:
            self._pending_single_view_fp = None

    @Slot(list)
    def on_composite_search_requested(self, query_elements: list):
        self._capture_single_view_anchor()
        self.window.clear_results()
        self.window.show_loading_state("Constructing query...")
        self.window.set_controls_enabled(False)
        # Wrap payload in dict to include the filter state
        payload = {"query_elements": query_elements, "tagged_only": self._tagged_only_filter}
        self.backend_job_queue.put(("composite_search", payload))

    @Slot()
    def on_random_order_requested(self):
        self._capture_single_view_anchor()
        self.window.clear_results()
        self.window.show_loading_state("Randomly reordering...")
        self.window.set_controls_enabled(False)
        # Pass the filter state
        self.backend_job_queue.put(("random_search", {"tagged_only": self._tagged_only_filter}))

    @Slot()
    def on_sort_by_date_requested(self):
        self._capture_single_view_anchor()
        self.window.clear_results()
        self.window.show_loading_state("Sorting images by date...")
        self.window.set_controls_enabled(False)
        # Pass the filter state
        self.backend_job_queue.put(("sort_by_date", {"tagged_only": self._tagged_only_filter}))

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
        self._current_results_generation = (self._current_results_generation + 1) % 1000000

        anchor_fp = self._pending_single_view_fp
        self._pending_single_view_fp = None  # Consume latch

        pending_scroll = self._pending_scroll_state
        self._pending_scroll_state = None  # Consume scroll state

        if not results:
            self.window.current_single_view_index = -1
            if self._tagged_only_filter:
                self.window.show_no_tags_view()
            else:
                self.window.show_sync_prompt_view()
            return

        was_in_single_view = (anchor_fp is not None) or (
            self.window.content_stack.currentWidget() is self.window.single_image_view_widget
        )
        if not anchor_fp and was_in_single_view:
            anchor_fp = self.window.single_image_view_widget.current_filepath

        # Update model
        self.window.results_model.set_results(results)

        if was_in_single_view and anchor_fp:
            # Re-anchor to the active file at its new row location
            self.window.handle_dataset_updated(preferred_filepath=anchor_fp, file_deleted=False)
            self.window.update_status_bar(f"Data reloaded. Viewing {Path(anchor_fp).name}.")
        else:
            # Grid View branch
            self.window.current_single_view_index = -1  # Reset stale index
            self.window.show_results_view()
            self.window.update_status_bar(f"Ordering complete. Displaying all {len(results)} images.")
            if pending_scroll is not None:
                val, max_val = pending_scroll
                QTimer.singleShot(50, lambda: self.window.restore_scroll_state(val, max_val))

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
        # Pass the shared embedder to avoid CUDA OOM from double instantiation
        self.sync_worker = SyncWorker(use_cpu_only=self._use_cpu_only, shared_embedder=self.backend_worker.embedder)
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

        self.on_sort_by_date_requested()

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

    @Slot(list)
    def on_toggle_tags_requested(self, indices: list):
        """Optimistically toggle tags for selected images using filepaths."""
        if self.sync_thread and self.sync_thread.isRunning():
            self.window.update_status_bar("Cannot toggle tags while sync is running.")
            return

        if not indices:
            return

        # Extract filepaths for the selected items
        filepaths = []
        for index in indices:
            row = index.row()
            if 0 <= row < len(self.window.results_model.results_data):
                item = self.window.results_model.results_data[row]
                filepaths.append(item[1])

        if not filepaths:
            return

        # 1. Optimistically update UI via filepaths (safe from row shifting)
        self.window.results_model.toggle_tag_for_filepaths(filepaths)

        # 2. Dispatch job to backend
        self.backend_job_queue.put(("toggle_tags", {
            "filepath_list": filepaths, 
            "tag_name": "marked",
            "generation": self._current_results_generation
        }))

    @Slot(list, int)
    def on_tag_operation_failed(self, filepaths: list, generation: int):
        if generation != self._current_results_generation:
            # The model has been completely replaced since this request; 
            # the optimistic UI state is gone, so no rollback needed.
            return
            
        self.window.update_status_bar("Database error: Failed to toggle tags. Rolling back.")
        # Toggling them a second time reverts the optimistic UI update
        self.window.results_model.toggle_tag_for_filepaths(filepaths)

    @Slot()
    def on_untag_all_requested(self):
        """Remove all tags from all images."""
        if self.sync_thread and self.sync_thread.isRunning():
            QMessageBox.warning(
                self.window, "Sync in Progress", "Cannot modify tags while a background sync is running."
            )
            return

        reply = QMessageBox.question(
            self.window,
            "Untag All",
            "Are you sure you want to remove the 'marked' tag from all images?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.backend_job_queue.put(("untag_all", {"tag_name": "marked"}))
            # Trigger a refresh to update the UI
            self.on_random_order_requested()

    @Slot()
    def on_move_tagged_requested(self):
        """Move tagged files to a user-selected directory."""
        if self.sync_thread and self.sync_thread.isRunning():
            QMessageBox.warning(
                self.window,
                "Sync in Progress",
                "Cannot move files while a background sync is running. Please wait or cancel the sync.",
            )
            return

        # Get tagged filepaths from the model
        tagged_filepaths = []
        for item in self.window.results_model.results_data:
            filepath = item[1]
            tags = item[2]
            if "marked" in tags:
                tagged_filepaths.append(filepath)

        if not tagged_filepaths:
            QMessageBox.information(self.window, "Move Tagged Files", "No files are tagged.")
            return

        # Prompt user for destination directory
        destination = QFileDialog.getExistingDirectory(self.window, "Select Destination Directory for Tagged Files")

        if not destination:
            return  # User cancelled

        # Boundary check: verify destination is within tracked directories
        tracked_dirs = get_scan_directories()
        dest_path = Path(destination).resolve()
        is_within_tracked = any(
            dest_path.is_relative_to(Path(d).resolve()) or dest_path == Path(d).resolve() for d in tracked_dirs
        )

        if not is_within_tracked:
            reply = QMessageBox.question(
                self.window,
                "Destination Not Tracked",
                f"The directory '{destination}' is not currently in your scan directories.\n\n"
                "Would you like to add it to the scan directories?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Yes:
                # Update config to include new directory
                config = load_config()
                if destination not in config.get("directories", []):
                    config.setdefault("directories", []).append(destination)
                    save_config(config)
                    logger.info(f"Added '{destination}' to scan directories.")
            else:
                return  # User declined

        # Spawn thread to move files
        self.window.update_status_bar(f"Moving {len(tagged_filepaths)} tagged files...")
        self.window.set_controls_enabled(False)

        move_thread = threading.Thread(
            target=self._move_files_thread, args=(tagged_filepaths, destination), daemon=True
        )
        move_thread.start()

    @Slot(str)
    def _on_move_completed(self, message: str):
        self.on_sync_requested()
        self.window.update_status_bar(message)

    def _move_files_thread(self, filepaths: list[str], destination: str):
        """Thread function to move files with collision handling."""
        moved_count = 0
        error_count = 0
        
        tracked_dirs = get_scan_directories()

        for filepath in filepaths:
            try:
                src = Path(filepath)
                
                # Boundary check before destructive operation
                src_resolved = src.resolve()
                is_within_tracked = False
                for d in tracked_dirs:
                    try:
                        d_resolved = Path(d).resolve()
                        if src_resolved.is_relative_to(d_resolved) or src_resolved == d_resolved:
                            is_within_tracked = True
                            break
                    except ValueError:
                        continue

                if not is_within_tracked:
                    logger.warning(f"File {filepath} is outside tracked directories, skipping move.")
                    error_count += 1
                    continue

                dest_dir = Path(destination)
                dest_file = dest_dir / src.name

                # Handle filename collisions
                if dest_file.exists():
                    stem = src.stem
                    suffix = src.suffix
                    counter = 1
                    while dest_file.exists():
                        dest_file = dest_dir / f"{stem}_{counter}{suffix}"
                        counter += 1

                shutil.move(str(src), str(dest_file))
                moved_count += 1
            except Exception as e:
                logger.error(f"Failed to move {filepath}: {e}")
                error_count += 1

        # Report results on main thread
        message = f"Moved {moved_count} files."
        if error_count > 0:
            message += f" {error_count} errors occurred."

        # Emit signal to safely update UI from main thread
        self.move_completed.emit(message)

    @Slot()
    def on_delete_tagged_requested(self):
        """Delete tagged files with a critical warning."""
        if self.sync_thread and self.sync_thread.isRunning():
            QMessageBox.warning(
                self.window,
                "Sync in Progress",
                "Cannot delete files while a background sync is running. Please wait or cancel the sync.",
            )
            return

        # Get tagged filepaths from the model
        tagged_filepaths = []
        for item in self.window.results_model.results_data:
            filepath = item[1]
            tags = item[2]
            if "marked" in tags:
                tagged_filepaths.append(filepath)

        if not tagged_filepaths:
            QMessageBox.information(self.window, "Delete Tagged Files", "No files are tagged.")
            return

        # Critical warning
        reply = QMessageBox.warning(
            self.window,
            "Delete Tagged Files",
            f"WARNING: This will permanently delete {len(tagged_filepaths)} files from disk.\n\n"
            "This action cannot be undone!\n\n"
            "Are you absolutely sure you want to proceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Spawn thread to delete files
        self.window.update_status_bar(f"Deleting {len(tagged_filepaths)} tagged files...")
        self.window.set_controls_enabled(False)

        delete_thread = threading.Thread(target=self._delete_files_thread, args=(tagged_filepaths,), daemon=True)
        delete_thread.start()

    @Slot(list, str)
    def _on_delete_completed(self, deleted_filepaths: list, message: str):
        is_in_single_view = (
            self.window.content_stack.currentWidget() is self.window.single_image_view_widget
        )
        active_fp = self.window.single_image_view_widget.current_filepath if is_in_single_view else None

        # Save scroll state
        scroll_val, scroll_max = self.window.get_scroll_state()

        # Refresh B (in-place): filter deleted rows and decrement surviving
        # ×N badges without re-querying, so the user's active search /
        # sort / filter context is preserved.
        deleted_filepaths_set = set(deleted_filepaths)
        current = list(self.window.results_model.results_data)

        # Map deleted filepaths -> sha (from pre-delete model rows) so we
        # know how many copies of each content hash were removed.
        from collections import Counter
        fp_to_sha: dict[str, str] = {}
        for result in current:
            if len(result) >= 5 and result[4]:
                fp_to_sha.setdefault(result[1], result[4])
        removed_sha_counts = Counter(
            fp_to_sha[fp] for fp in deleted_filepaths if fp in fp_to_sha
        )

        new_results = []
        for result in current:
            filepath = result[1]
            if filepath in deleted_filepaths_set:
                continue
            if len(result) >= 5 and removed_sha_counts:
                score, fp, tags, dup_count, sha = (
                    result[0], result[1], result[2], result[3], result[4],
                )
                if sha in removed_sha_counts:
                    dup_count = max(1, dup_count - removed_sha_counts[sha])
                new_results.append((score, fp, tags, dup_count, sha))
            else:
                new_results.append(result)

        self.window.results_model.set_results(new_results)
        self.window.update_status_bar(message)

        # Consume any stale pending scroll (e.g. set by
        # on_manage_duplicates_requested) so it doesn't misfire on the next
        # on_results_ready. Scroll is handled explicitly below.
        self._pending_scroll_state = None

        if is_in_single_view:
            if active_fp in deleted_filepaths_set:
                # Active file deleted: advance to clamped index
                self.window.handle_dataset_updated(preferred_filepath=None, file_deleted=True)
            else:
                # Surviving file: re-anchor row index, refresh dup badges and thumbnails
                self.window.handle_dataset_updated(preferred_filepath=active_fp, file_deleted=False)
        else:
            # Restore scroll state
            self.window.restore_scroll_state(scroll_val, scroll_max)
        # We DON'T enable controls here, because the DB deletion job might still be in the backend queue.
        # We wait for on_backend_deletion_completed.

    def _delete_files_thread(self, filepaths: list[str]):
        """Thread function to delete files."""
        deleted_count = 0
        error_count = 0
        deleted_filepaths = []
        
        tracked_dirs = get_scan_directories()

        for filepath in filepaths:
            try:
                # Boundary check before destructive operation
                src = Path(filepath).resolve()
                is_within_tracked = False
                for d in tracked_dirs:
                    try:
                        d_resolved = Path(d).resolve()
                        if src.is_relative_to(d_resolved) or src == d_resolved:
                            is_within_tracked = True
                            break
                    except ValueError:
                        continue

                if not is_within_tracked:
                    logger.warning(f"File {filepath} is outside tracked directories, skipping deletion.")
                    error_count += 1
                    continue

                os.remove(str(src))
                deleted_count += 1
                deleted_filepaths.append(filepath)
            except Exception as e:
                logger.error(f"Failed to delete {filepath}: {e}")
                error_count += 1

        # Report results
        message = f"Deleted {deleted_count} files."
        if error_count > 0:
            message += f" {error_count} errors occurred."

        # Send targeted deletion job to backend to remove DB rows
        if deleted_filepaths:
            self.backend_job_queue.put(("delete_target_filepaths", {"filepath_list": deleted_filepaths}))

        # Emit signal with deleted filepaths - filtering done on main thread
        self.delete_completed.emit(deleted_filepaths, message)

    @Slot(bool)
    def on_show_tagged_only_toggled(self, checked: bool):
        """Filter results to show only tagged images."""
        # Trigger a refresh with the tagged_only filter
        self._tagged_only_filter = checked

        # Preserve single-view anchor across the loading overlay (nit fix).
        self._capture_single_view_anchor()

        # Determine current search mode and re-trigger with filter
        # For simplicity, we'll just trigger a random search with the filter
        self.window.clear_results()
        self.window.show_loading_state("Filtering..." if checked else "Loading all images...")
        self.window.set_controls_enabled(False)
        self.backend_job_queue.put(("random_search", {"tagged_only": checked}))

    @Slot(str)
    def on_dedup_info_requested(self, filepath: str):
        self.backend_job_queue.put(("get_duplicate_paths", {"filepath": filepath}))

    @Slot(str, list)
    def on_resolve_sources_requested(self, target_filepath: str, source_filenames: list):
        """Forward a ComfyUI source-lineage request to the backend worker."""
        if not target_filepath or not source_filenames:
            return
        self.backend_job_queue.put(
            ("resolve_sources", {"target_filepath": target_filepath, "source_filenames": list(source_filenames)})
        )

    @Slot(str, list)
    def on_source_resolution_ready(self, target_filepath: str, resolved_items: list):
        """Forward backend lineage results to the window (stale-guarded there)."""
        self.window.on_source_resolution_ready(target_filepath, resolved_items)

    @Slot(list)
    def on_manage_duplicates_requested(self, filepaths: list):
        if not filepaths:
            return
        if self.sync_thread and self.sync_thread.isRunning():
            QMessageBox.warning(
                self.window,
                "Sync in progress",
                "Cannot delete files while a sync is running.",
            )
            return
        self._pending_scroll_state = self.window.get_scroll_state()
        self.window.update_status_bar(f"Deleting {len(filepaths)} duplicate(s)\u2026")
        self.window.set_controls_enabled(False)
        import threading
        t = threading.Thread(
            target=self._delete_files_thread, args=(filepaths,), daemon=True
        )
        t.start()

    @Slot()
    def on_backend_deletion_completed(self):
        # Refresh B: the model was already filtered in-place in
        # _on_delete_completed (with dup_counts decremented), so just
        # re-enable controls. Do NOT force Sort-by-Date — that would wipe
        # the user's active search / random / tagged-only context.
        logger.info("Backend finished DB deletion job.")
        self.window.set_controls_enabled(True)
        self.window.update_status_bar("Deletion complete. View updated in place.")
