import logging
import traceback
from PySide6.QtCore import QObject, Signal, Slot
from image_db import ImageDatabase
from ml_core import ImageEmbedder
from config_utils import get_scan_directories, get_db_path

logger = logging.getLogger(__name__)


class SyncWorker(QObject):
    """
    A worker that runs the database synchronization in a separate thread.
    It creates its own instances of the embedder and database to ensure
    thread safety and isolation from the main application's backend.
    """

    status_update = Signal(str)
    progress_update = Signal(str, int, int)  # stage_name, current, total
    finished = Signal(str, str)  # result_string, message
    error = Signal(str)

    def __init__(self, use_cpu_only: bool):
        super().__init__()
        self.use_cpu_only = use_cpu_only
        self.db = None
        self.embedder = None
        self._is_cancelled = False

    @Slot()
    def run(self):
        """The main entry point for the worker's execution."""
        result = "failed"
        message = "An unknown error occurred."
        try:
            self._is_cancelled = False
            logger.info("SyncWorker starting.")
            self.status_update.emit("Initializing...")
            db_path = get_db_path()

            self.embedder = ImageEmbedder(use_cpu_only=self.use_cpu_only)
            self.db = ImageDatabase(db_path=db_path, embedder=self.embedder)

            if self._is_cancelled:
                result, message = "cancelled", "Sync cancelled during initialization."
                return

            directories_to_scan = get_scan_directories()
            if not directories_to_scan:
                result, message = "success", "No directories configured to scan."
                return

            self.db.reconcile_database(
                configured_dirs=directories_to_scan,
                progress_callback=self.handle_progress,
                status_callback=self.handle_status,
                check_cancelled_callback=self.is_cancelled,
            )

            result, message = "success", "Synchronization complete."

        except self.db.InterruptedError:
            result, message = "cancelled", "Synchronization cancelled."
        except Exception as e:
            logger.error("--- AN ERROR OCCURRED DURING SYNC ---")
            logger.error(traceback.format_exc())
            self.error.emit(traceback.format_exc())
            result, message = "failed", "An error occurred during synchronization."
        finally:
            self.finished.emit(result, message)
            self.cleanup()

    @Slot()
    def cancel(self):
        """Flags the worker to cancel its operation at the next opportunity."""
        logger.info("SyncWorker received cancellation request.")
        self._is_cancelled = True
        if self.db:
            self.db.cancel_sync()  # Propagate cancellation to the DB layer

    def is_cancelled(self) -> bool:
        """Callback for the DB layer to check for cancellation."""
        return self._is_cancelled

    def handle_progress(self, stage: str, current: int, total: int):
        self.progress_update.emit(stage, current, total)

    def handle_status(self, message: str):
        self.status_update.emit(message)

    def cleanup(self):
        """Closes database connections and cleans up resources."""
        if self.db:
            self.db.close()
            self.db = None
        self.embedder = None
        logger.info("SyncWorker cleaned up resources.")
