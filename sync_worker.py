import logging
import traceback
from PySide6.QtCore import QObject, Signal, Slot
from image_db import ImageDatabase
from ml_core import ImageEmbedder
from config_utils import get_scan_directories, get_db_path, get_model_id

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ml_core import ImageEmbedder

logger = logging.getLogger(__name__)


class SyncWorker(QObject):
    """
    A worker that runs the database synchronization in a separate thread.
    Can use a shared embedder to avoid CUDA OOM issues, or create its own
    if no shared embedder is provided.
    """

    status_update = Signal(str)
    progress_update = Signal(str, int, int)  # stage_name, current, total
    finished = Signal(str, str)  # result_string, message
    error = Signal(str)

    def __init__(self, use_cpu_only: bool, shared_embedder: "ImageEmbedder" = None):
        super().__init__()
        self.use_cpu_only = use_cpu_only
        self.shared_embedder = shared_embedder
        self.db = None
        self.embedder = None
        self._is_cancelled = False
        self._owns_embedder = False

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
            model_id = get_model_id()

            # Use shared embedder if provided, otherwise create our own
            if self.shared_embedder is not None:
                logger.info("Using shared embedder to avoid CUDA OOM.")
                self.embedder = self.shared_embedder
                self._owns_embedder = False
                self.status_update.emit("Using shared model...")
            else:
                self.status_update.emit(f"Loading model '{model_id}'...")
                self.embedder = ImageEmbedder(model_id=model_id, use_cpu_only=self.use_cpu_only)
                self._owns_embedder = True

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
            )

            result, message = "success", "Synchronization complete."

        except self.db.InterruptedError if self.db else Exception:
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

    def handle_progress(self, stage: str, current: int, total: int):
        self.progress_update.emit(stage, current, total)

    def handle_status(self, message: str):
        self.status_update.emit(message)

    def cleanup(self):
        """Closes database connections and cleans up resources."""
        # Close database properly
        if self.db is not None:
            self.db.close()
            self.db = None

        # Only unload embedder if we own it (not shared)
        if self._owns_embedder and self.embedder is not None:
            self.embedder.unload()
        
        self.embedder = None
        logger.info("SyncWorker cleaned up resources.")
