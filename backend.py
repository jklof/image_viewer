# backend.py

import traceback
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

# Updated local imports
from image_db import ImageDatabase
from ml_core import ImageEmbedder

import logging
logger = logging.getLogger(__name__)

DB_PATH = "images.db"
MAX_SEARCH_RESULTS = 50

class BackendWorker(QObject):
    """
    A dedicated worker that owns the ImageDatabase and ImageEmbedder instances.
    It lives in a separate thread and communicates with the UI via signals.
    """
    error = Signal(str)
    initialized = Signal()
    results_ready = Signal(list)
    status_update = Signal(str)

    def __init__(self):
        super().__init__()
        self.db = None
        self.embedder = None

    @Slot()
    def initialize(self):
        """
        Creates the ImageEmbedder and ImageDatabase instances.
        This MUST be run in the worker thread.
        This includes robust error handling to prevent silent failures.
        """
        try:
            self.status_update.emit("Initializing backend... This may take a moment.")
            
            # Instantiate both core components in the correct thread
            self.embedder = ImageEmbedder()
            self.db = ImageDatabase(db_path=DB_PATH, embedder=self.embedder)
            
            self.initialized.emit()
        except Exception as e:
            # This is critical for debugging silent startup failures.
            logger.error("--- AN ERROR OCCURRED IN THE BACKEND THREAD ---")
            logger.error(traceback.format_exc())
            logger.error("-------------------------------------------------")
            self.error.emit(traceback.format_exc())

    @Slot(str)
    def perform_text_search(self, query: str):
        if not self.db: return
        logger.info(f"Performing text search for: '{query}'")
        self.status_update.emit(f"Searching for: '{query}'...")
        results = self.db.search_by_text(text_query=query, top_k=MAX_SEARCH_RESULTS)
        self.results_ready.emit(results)

    @Slot(str)
    def perform_image_search(self, image_path: str):
        if not self.db: return
        logger.info(f"Performing image search for: {image_path}")
        self.status_update.emit(f"Searching for images similar to {Path(image_path).name}...")
        results = self.db.search_similar_images(image_path=image_path, top_k=MAX_SEARCH_RESULTS)

        self.results_ready.emit(results)
    
    @Slot()
    def shutdown(self):
        """Safely closes the database connection in the correct thread."""
        if self.db:
            logger.info("Backend shutting down, closing DB connection.")
            self.db.close()