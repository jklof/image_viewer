import logging
import traceback
import queue
import threading
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
    warning = Signal(str)
    initialized = Signal()
    reloaded = Signal()
    results_ready = Signal(list)
    status_update = Signal(str)
    visualization_data_ready = Signal(list)
    tag_operation_failed = Signal(list, int)
    deletion_completed = Signal()
    duplicate_paths_ready = Signal(list)  # list of dicts from get_duplicate_paths


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
        self._shutdown_event = threading.Event()

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
            self.db._verify_model_compatibility()
            logger.info("Backend initialized successfully.")
            self.signals.initialized.emit()
        except (KeyboardInterrupt, SystemExit):
            return
        except Exception:
            error_msg = traceback.format_exc()
            logger.error(f"--- AN ERROR OCCURRED DURING INITIALIZATION ---\n{error_msg}")
            self.signals.error.emit(error_msg)
            return  # Exit thread on catastrophic failure

        # --- 2. Job Processing Loop ---
        while not self._shutdown_event.is_set():
            try:
                # Wait for a job to appear on the queue
                try:
                    job_type, payload = self.job_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                if job_type == "shutdown":
                    break

                handler = getattr(self, f"handle_{job_type}", None)
                if handler:
                    handler(payload)
                else:
                    logger.warning(f"Unknown job type received: {job_type}")

            except (KeyboardInterrupt, SystemExit):
                logger.info("BackendWorker stopping due to interrupt/exit signal.")
                break
            except Exception as e:
                # Catch generic errors in the loop to keep thread alive
                if self._shutdown_event.is_set():
                    break  # Ignore errors if we are shutting down

                error_msg = traceback.format_exc()
                logger.error(f"--- AN UNHANDLED ERROR OCCURRED IN BACKEND WORKER LOOP ---\n{error_msg}")
                self.signals.warning.emit(str(e))

        logger.info("BackendWorker thread shutting down.")

    # --- Job Handlers ---
    def handle_composite_search(self, payload):
        try:
            if not self.db or not self.embedder:
                return

            # Handle both list (legacy) and dict (new) formats
            if isinstance(payload, dict):
                query_elements = payload.get("query_elements", [])
                tagged_only = payload.get("tagged_only", False)
            else:
                query_elements = payload
                tagged_only = False

            logger.info(f"Performing composite search with {len(query_elements)} elements. tagged_only={tagged_only}")
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
                if tagged_only:
                    results = [(s, f, t) for s, f, t in results if "marked" in t.split(",")]
                # Add dup_count + sha to composite search results.
                # 5-tuple (score, path, tags, dup_count, sha) lets the model
                # update all same-SHA rows optimistically (Sort-by-Date shows
                # every filepath, not just paths[0]).
                results_with_dup = []
                for score, path, tags in results:
                    sha = self.db._filepath_to_sha_cache.get(path, "")
                    dup_count = self.db._sha_to_dup_count_cache.get(sha, 1)
                    results_with_dup.append((score, path, tags, dup_count, sha))
                self.signals.results_ready.emit(results_with_dup)
            else:
                # All elements failed or resulted in zero vector
                if successful_elements == 0 and len(query_elements) > 0:
                    error_msg = (
                        f"All {len(query_elements)} query element(s) failed to load or process. Cannot perform search."
                    )
                    logger.warning(error_msg)
                    self.signals.warning.emit(error_msg)
                    # Safely load random results to clear the loading UI spinner
                    self.handle_random_search({"tagged_only": tagged_only})
                else:
                    logger.warning("Composite query resulted in a zero vector. Falling back to random order.")
                    self.signals.status_update.emit("Query resulted in zero vector - showing random order")
                    self.handle_random_search({"tagged_only": tagged_only})
        except Exception:
            error_msg = traceback.format_exc()
            logger.error(f"--- AN ERROR OCCURRED DURING COMPOSITE SEARCH ---\n{error_msg}")
            self.signals.error.emit(error_msg)

    def handle_random_search(self, payload):
        try:
            if not self.db:
                return
            tagged_only = payload.get("tagged_only", False) if isinstance(payload, dict) else False
            self.signals.status_update.emit("Randomly ordering all images...")
            filepaths_with_tags = self.db.get_all_unique_filepaths()
            random.shuffle(filepaths_with_tags)
            if tagged_only:
                filepaths_with_tags = [t for t in filepaths_with_tags if "marked" in t[1].split(",")]
            results = []
            for entry in filepaths_with_tags:
                # Backward compat: accept (path, tags, dup) or (path, tags, dup, sha)
                if len(entry) >= 4:
                    path, tags, _dup, sha = entry[0], entry[1], entry[2], entry[3]
                else:
                    path, tags = entry[0], entry[1]
                    sha = self.db._filepath_to_sha_cache.get(path, "")
                dup_count = self.db._sha_to_dup_count_cache.get(sha, 1)
                results.append((0.0, path, tags, dup_count, sha))
            self.signals.results_ready.emit(results)
        except Exception:
            logger.error(traceback.format_exc())

    def handle_sort_by_date(self, payload):
        try:
            if not self.db:
                return
            tagged_only = payload.get("tagged_only", False) if isinstance(payload, dict) else False
            self.signals.status_update.emit("Sorting all images by date...")
            files_with_mtime_tags = self.db.get_all_filepaths_with_mtime()
            files_with_mtime_tags.sort(key=lambda x: x[1], reverse=True)
            if tagged_only:
                files_with_mtime_tags = [t for t in files_with_mtime_tags if "marked" in t[2].split(",")]
            results = []
            for entry in files_with_mtime_tags:
                # Backward compat: (path, mtime, tags, dup) or (path, mtime, tags, dup, sha)
                if len(entry) >= 5:
                    path, tags, dup_count, sha = entry[0], entry[2], entry[3], entry[4]
                else:
                    path, tags, dup_count = entry[0], entry[2], entry[3]
                    sha = self.db._filepath_to_sha_cache.get(path, "")
                results.append((0.0, path, tags, dup_count, sha))
            self.signals.results_ready.emit(results)
        except Exception:
            logger.error(traceback.format_exc())

    def handle_visualization_data(self, _):
        try:
            if not self.db:
                return

            # Check if shutdown requested during long op
            def check_stop():
                if self._shutdown_event.is_set():
                    raise Exception("Backend shutdown requested")

            self.signals.status_update.emit("Checking visualization data integrity...")

            # Ensure data exists (re-calculate UMAP if dirty)
            self.db.ensure_visualization_data(
                status_callback=lambda msg: self.signals.status_update.emit(msg), check_cancelled_callback=check_stop
            )

            self.signals.status_update.emit("Loading visualization data...")
            plot_data = self.db.get_visualization_data()
            self.signals.visualization_data_ready.emit(plot_data or [])
        except Exception as e:
            # Log and notify UI
            logger.error(traceback.format_exc())
            self.signals.warning.emit(f"Visualization failed: {str(e)}")

    def handle_reload(self, _):
        try:
            if not self.db:
                return
            self.signals.status_update.emit("Reloading image data from database...")
            self.db._load_embeddings_into_memory()
            self.signals.reloaded.emit()
        except Exception:
            logger.error(traceback.format_exc())

    def handle_toggle_tags(self, payload: dict):
        filepaths = payload.get("filepath_list", [])
        tag_name = payload.get("tag_name", "marked")
        generation = payload.get("generation", 0)
        try:
            if not self.db:
                return
            if filepaths:
                self.db.toggle_tag(filepaths, tag_name)
                logger.info(f"Toggled tag '{tag_name}' for {len(filepaths)} images.")
        except Exception:
            logger.error(traceback.format_exc())
            # Emit the failed filepaths back to the UI for rollback
            self.signals.tag_operation_failed.emit(filepaths, generation)

    def handle_untag_all(self, payload: dict):
        try:
            if not self.db:
                return
            tag_name = payload.get("tag_name", "marked")
            self.db.untag_all(tag_name)
            logger.info(f"Removed tag '{tag_name}' from all images.")
        except Exception:
            logger.error(traceback.format_exc())

    def handle_delete_target_filepaths(self, payload: dict):
        try:
            if not self.db:
                return
            filepath_list = payload.get("filepath_list", [])
            if filepath_list:
                self.db.delete_target_filepaths(filepath_list)
                logger.info(f"Deleted {len(filepath_list)} filepaths from database.")
                self.signals.deletion_completed.emit()
        except Exception:
            logger.error(traceback.format_exc())

    def handle_get_duplicate_paths(self, payload: dict):
        try:
            if not self.db:
                return
            filepath = payload.get("filepath", "")
            paths = self.db.get_duplicate_paths(filepath)
            self.signals.duplicate_paths_ready.emit(paths)
        except Exception:
            logger.error(traceback.format_exc())

    def shutdown(self):
        self._shutdown_event.set()
        try:
            self.job_queue.put(("shutdown", None), block=False)
        except (queue.Full, Exception):
            pass

        # Release GPU memory by unloading the embedder
        if self.embedder is not None:
            self.embedder.unload()
            self.embedder = None

        # Close database connections and free memory
        if self.db is not None:
            self.db.close()
            self.db = None
