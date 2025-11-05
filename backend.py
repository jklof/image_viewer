import traceback
from pathlib import Path
import random
from PySide6.QtCore import QObject, Signal, Slot
import numpy as np
from PIL import Image
from umap import UMAP
import hdbscan
from image_db import ImageDatabase
from ml_core import ImageEmbedder, EMBEDDING_SHAPE
from config_utils import get_db_path
import logging

logger = logging.getLogger(__name__)


class BackendWorker(QObject):
    error = Signal(str)
    initialized = Signal()
    reloaded = Signal()  # Emitted after a successful data reload
    results_ready = Signal(list)
    status_update = Signal(str)
    visualization_data_ready = Signal(list)

    def __init__(self, use_cpu_only: bool = False):
        super().__init__()
        self.db = None
        self.embedder = None
        self.use_cpu_only = use_cpu_only
        logger.info(f"BackendWorker constructed in main thread. CPU-only mode: {self.use_cpu_only}")

    @Slot()
    def initialize(self):
        try:
            logger.info("BackendWorker.initialize() starting in worker thread.")
            self.status_update.emit("Initializing backend... This may take a moment.")
            db_path = get_db_path()
            self.embedder = ImageEmbedder(use_cpu_only=self.use_cpu_only)
            self.db = ImageDatabase(db_path=db_path, embedder=self.embedder)
            self.initialized.emit()
        except Exception as e:
            logger.error("--- AN ERROR OCCURRED DURING INITIALIZATION ---")
            logger.error(traceback.format_exc())
            self.error.emit(traceback.format_exc())

    @Slot()
    def perform_reload(self):
        """
        Reloads the in-memory embeddings from the database.
        Used after a synchronization process has completed.
        """
        try:
            logger.info("BackendWorker received request to reload data.")
            self.status_update.emit("Reloading image data from database...")
            if not self.db:
                self.error.emit("Cannot reload, database connection is not available.")
                return

            self.db._load_embeddings_into_memory()
            self.status_update.emit("Data reload complete.")
            self.reloaded.emit()  # Signal completion
        except Exception as e:
            logger.error("--- AN ERROR OCCURRED DURING DATA RELOAD ---")
            logger.error(traceback.format_exc())
            self.error.emit(traceback.format_exc())

    @Slot(list)
    def perform_composite_search(self, query_elements: list):
        if not self.db or not self.embedder:
            return
        try:
            logger.info(f"Performing composite search with {len(query_elements)} elements.")
            self.status_update.emit(f"Building query from {len(query_elements)} elements...")

            # Initialize a zero vector for the final query
            combined_vector = np.zeros(EMBEDDING_SHAPE, dtype=np.float32)

            for element in query_elements:
                element_type = element["type"]
                value = element["value"]
                weight = element["weight"]

                embedding = None
                if element_type == "text":
                    embedding = self.embedder.embed_text(value)
                elif element_type == "image":
                    try:
                        image = Image.open(value).convert("RGB")
                        embedding = self.embedder.embed_image(image)
                    except Exception as e:
                        logger.warning(f"Could not load image {value} for composite query. Skipping. Error: {e}")
                        continue

                if embedding is not None:
                    combined_vector += embedding * weight

            # Normalize the final combined vector
            norm = np.linalg.norm(combined_vector)
            if norm > 0:
                final_query_vector = combined_vector / norm
            else:
                # If all weights cancel out or elements fail, we have a zero vector.
                # In this case, we cannot perform a search. Emit empty results.
                logger.warning("Composite query resulted in a zero vector. Cannot perform search.")
                self.results_ready.emit([])
                return

            results = self.db._perform_search(final_query_vector, -1)
            self.results_ready.emit(results)
        except Exception as e:
            logger.error("--- AN ERROR OCCURRED DURING COMPOSITE SEARCH ---")
            logger.error(traceback.format_exc())
            self.error.emit(traceback.format_exc())

    @Slot()
    def perform_random_search(self):
        if not self.db:
            return
        try:
            logger.info("Performing random ordering of all images.")
            self.status_update.emit("Randomly ordering all images...")
            random.seed()
            filepaths = self.db.get_all_unique_filepaths()
            random.shuffle(filepaths)
            results = [(0.0, path) for path in filepaths]
            self.results_ready.emit(results)
        except Exception as e:
            logger.error("--- AN ERROR OCCURRED DURING RANDOM SEARCH ---")
            logger.error(traceback.format_exc())
            self.error.emit(traceback.format_exc())

    @Slot()
    def request_visualization_data(self):
        if not self.db:
            self.error.emit("Database not initialized for visualization request.")
            return
        self.status_update.emit("Loading pre-calculated visualization data...")
        try:
            plot_data = self.db.get_visualization_data()

            if not plot_data:
                self.status_update.emit("Visualization failed: No data found. Please run a sync from the command line.")
                self.visualization_data_ready.emit([])
                return

            self.status_update.emit(f"Visualization data ready for {len(plot_data)} images.")
            self.visualization_data_ready.emit(plot_data)
        except Exception as e:
            logger.error("Error loading visualization data from database", exc_info=True)
            self.error.emit(traceback.format_exc())

    @Slot()
    def shutdown(self):
        if self.db:
            logger.info("Backend shutting down, closing DB connection.")
            self.db.close()
