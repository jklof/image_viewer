import traceback
from pathlib import Path
from PySide6.QtCore import QObject, Signal, Slot
import numpy as np
from umap import UMAP
import hdbscan
from image_db import ImageDatabase
from ml_core import ImageEmbedder
import logging

logger = logging.getLogger(__name__)

DB_PATH = "images.db"


class BackendWorker(QObject):
    error = Signal(str)
    initialized = Signal()
    results_ready = Signal(list)
    status_update = Signal(str)
    visualization_data_ready = Signal(list)

    def __init__(self):
        super().__init__()
        self.db = None
        self.embedder = None
        logger.info("BackendWorker constructed in main thread.")

    @Slot()
    def initialize(self):
        try:
            logger.info("BackendWorker.initialize() starting in worker thread.")
            self.status_update.emit("Initializing backend... This may take a moment.")
            self.embedder = ImageEmbedder()
            self.db = ImageDatabase(db_path=DB_PATH, embedder=self.embedder)
            self.initialized.emit()
        except Exception as e:
            logger.error("--- AN ERROR OCCURRED IN THE BACKEND THREAD ---")
            logger.error(traceback.format_exc())
            self.error.emit(traceback.format_exc())

    # ... rest of the file is identical ...
    @Slot(str)
    def perform_text_search(self, query: str):
        if not self.db:
            return
        logger.info(f"Performing text search to order all images by: '{query}'")
        self.status_update.emit(f"Ordering all images by: '{query}'...")
        results = self.db.search_by_text(text_query=query)
        self.results_ready.emit(results)

    @Slot(str)
    def perform_image_search(self, image_path: str):
        if not self.db:
            return
        logger.info(f"Performing image search to order all images by: {image_path}")
        self.status_update.emit(f"Ordering all images by similarity to {Path(image_path).name}...")
        results = self.db.search_similar_images(image_path=image_path)
        self.results_ready.emit(results)

    @Slot()
    def request_visualization_data(self):
        """
        --- MODIFIED: This method now directly queries the pre-calculated data ---
        """
        if not self.db:
            self.error.emit("Database not initialized for visualization request.")
            return
        self.status_update.emit("Loading pre-calculated visualization data...")
        try:
            # Directly query the pre-calculated data from the database
            plot_data = self.db.get_visualization_data()

            if not plot_data:
                # This can happen if sync hasn't run or there are no images.
                self.status_update.emit("Visualization failed: No data found. Please run a sync from the command line.")
                self.visualization_data_ready.emit([])  # Emit empty list to clear the view
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
