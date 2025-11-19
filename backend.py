import logging
import traceback
import queue
from pathlib import Path
import random

from PySide6.QtCore import QObject, Signal
import numpy as np
from PIL import Image

from image_db import ImageDatabase
from ml_core import ImageEmbedder
from config_utils import get_db_path, get_model_id

logger = logging.getLogger(__name__)


class BackendSignals(QObject):
    """Holds signals that the backend worker thread can emit."""

    error = Signal(str)
    initialized = Signal()
    reloaded = Signal()
    results_ready = Signal(list)
    status_update = Signal(str)
    visualization_data_ready = Signal(list)


class BackendWorker:
    """
    Runs in a standard Python thread, consuming jobs from a queue.
    This avoids all QThread/CUDA conflicts.
    """

    def __init__(self, signals: BackendSignals, job_queue: queue.Queue, use_cpu_only: bool):
        self.signals = signals
        self.job_queue = job_queue
        self.use_cpu_only = use_cpu_only
        self.db: ImageDatabase | None = None
        self.embedder: ImageEmbedder | None = None
        self._shutdown = False

    def run(self):
        """The main loop for the worker thread."""
        # --- 1. Initialization ---
        try:
            logger.info("BackendWorker thread started. Initializing...")
            self.signals.status_update.emit("Initializing backend...")

            db_path = get_db_path()
            model_id = get_model_id()

            self.signals.status_update.emit(f"Loading model '{model_id}'...")
            self.embedder = ImageEmbedder(model_id=model_id, use_cpu_only=self.use_cpu_only)

            self.signals.status_update.emit("Connecting to database...")
            self.db = ImageDatabase(db_path=db_path, embedder=self.embedder)
            logger.info("Backend initialized successfully.")
            self.signals.initialized.emit()
        except Exception:
            error_msg = traceback.format_exc()
            logger.error(f"--- AN ERROR OCCURRED DURING INITIALIZATION ---\n{error_msg}")
            self.signals.error.emit(error_msg)
            return  # Exit thread on catastrophic failure

        # --- 2. Job Processing Loop ---
        while not self._shutdown:
            try:
                # Wait for a job to appear on the queue
                job_type, payload = self.job_queue.get()
                if job_type == "shutdown":
                    break
                handler = getattr(self, f"handle_{job_type}", None)
                if handler:
                    handler(payload)
                else:
                    logger.warning(f"Unknown job type received: {job_type}")
            except Exception:
                # This is a fallback for unexpected errors in the loop itself.
                # Individual handlers have their own, more specific error handling.
                error_msg = traceback.format_exc()
                logger.error(f"--- AN UNHANDLED ERROR OCCURRED IN BACKEND WORKER LOOP ---\n{error_msg}")
                self.signals.error.emit(error_msg)

        logger.info("BackendWorker thread shutting down.")

    # --- Job Handlers ---
    def handle_composite_search(self, query_elements: list):
        try:
            if not self.db or not self.embedder:
                return
            logger.info(f"Performing composite search with {len(query_elements)} elements.")
            self.signals.status_update.emit(f"Building query from {len(query_elements)} elements...")

            combined_vector = np.zeros(self.embedder.embedding_shape, dtype=self.embedder.embedding_dtype)
            successful_elements = 0
            failed_elements = []

            for element in query_elements:
                embedding = None
                if element["type"] == "text":
                    try:
                        embedding = self.embedder.embed_text(element["value"])
                    except Exception as e:
                        logger.warning(
                            f"Could not embed text '{element['value'][:50]}...' for composite query. Error: {e}"
                        )
                        failed_elements.append(("text", element["value"][:50], str(e)))
                        continue
                elif element["type"] == "image":
                    try:
                        image = Image.open(element["value"]).convert("RGB")
                        embedding = self.embedder.embed_image(image)
                    except Exception as e:
                        logger.warning(
                            f"Could not load image {element['value']} for composite query. Skipping. Error: {e}"
                        )
                        failed_elements.append(("image", element["value"], str(e)))
                        continue

                if embedding is not None:
                    combined_vector += embedding * element["weight"]
                    successful_elements += 1

            # Check if any elements were successfully processed
            if failed_elements:
                failed_summary = ", ".join(
                    [f"{t} ({Path(v).name if t == 'image' else v})" for t, v, _ in failed_elements]
                )
                logger.warning(f"Failed to process {len(failed_elements)} query element(s): {failed_summary}")
                self.signals.status_update.emit(
                    f"Warning: {len(failed_elements)} of {len(query_elements)} query elements failed to load"
                )

            norm = np.linalg.norm(combined_vector)
            if norm > 0:
                logger.info(f"Successfully processed {successful_elements}/{len(query_elements)} query elements.")
                final_query_vector = combined_vector / norm
                results = self.db._perform_search(final_query_vector, -1)
                self.signals.results_ready.emit(results)
            else:
                # All elements failed or resulted in zero vector
                if successful_elements == 0 and len(query_elements) > 0:
                    error_msg = (
                        f"All {len(query_elements)} query element(s) failed to load or process. Cannot perform search."
                    )
                    logger.error(error_msg)
                    self.signals.error.emit(error_msg)
                    # Still show random results so the UI isn't empty
                    self.handle_random_search(None)
                else:
                    logger.warning("Composite query resulted in a zero vector. Falling back to random order.")
                    self.signals.status_update.emit("Query resulted in zero vector - showing random order")
                    self.handle_random_search(None)
        except Exception:
            error_msg = traceback.format_exc()
            logger.error(f"--- AN ERROR OCCURRED DURING COMPOSITE SEARCH ---\n{error_msg}")
            self.signals.error.emit(error_msg)

    def handle_random_search(self, _):
        try:
            if not self.db:
                return
            self.signals.status_update.emit("Randomly ordering all images...")
            filepaths = self.db.get_all_unique_filepaths()
            random.shuffle(filepaths)
            results = [(0.0, path) for path in filepaths]
            self.signals.results_ready.emit(results)
        except Exception:
            error_msg = traceback.format_exc()
            logger.error(f"--- AN ERROR OCCURRED DURING RANDOM SEARCH ---\n{error_msg}")
            self.signals.error.emit(error_msg)

    def handle_sort_by_date(self, _):
        try:
            if not self.db:
                return
            self.signals.status_update.emit("Sorting all images by date...")
            files_with_mtime = self.db.get_all_filepaths_with_mtime()
            files_with_mtime.sort(key=lambda x: x[1], reverse=True)
            results = [(0.0, path) for path, mtime in files_with_mtime]
            self.signals.results_ready.emit(results)
        except Exception:
            error_msg = traceback.format_exc()
            logger.error(f"--- AN ERROR OCCURRED DURING SORT BY DATE ---\n{error_msg}")
            self.signals.error.emit(error_msg)

    def handle_visualization_data(self, _):
        try:
            if not self.db:
                return
            self.signals.status_update.emit("Loading pre-calculated visualization data...")
            plot_data = self.db.get_visualization_data()
            self.signals.visualization_data_ready.emit(plot_data or [])
        except Exception:
            error_msg = traceback.format_exc()
            logger.error(f"--- AN ERROR OCCURRED LOADING VISUALIZATION DATA ---\n{error_msg}")
            self.signals.error.emit(error_msg)

    def handle_reload(self, _):
        try:
            if not self.db:
                return
            self.signals.status_update.emit("Reloading image data from database...")
            self.db._load_embeddings_into_memory()
            self.signals.reloaded.emit()
        except Exception:
            error_msg = traceback.format_exc()
            logger.error(f"--- AN ERROR OCCURRED DURING DATA RELOAD ---\n{error_msg}")
            self.signals.error.emit(error_msg)

    def shutdown(self):
        self._shutdown = True
        # Post a final shutdown job to unblock the queue.get() if it's waiting
        try:
            self.job_queue.put(("shutdown", None), block=False)
        except queue.Full:
            # If the queue is somehow full, it doesn't matter,
            # the worker will eventually see the _shutdown flag.
            pass
