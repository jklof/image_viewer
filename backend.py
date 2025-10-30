# backend.py

import traceback
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

import numpy as np
# --- CORRECT CANONICAL IMPORTS for a clean environment ---
from umap import UMAP 
import hdbscan 

# Updated local imports
from image_db import ImageDatabase
from ml_core import ImageEmbedder

import logging
logger = logging.getLogger(__name__)

DB_PATH = "images.db"
# --- REMOVED --- The hardcoded search limit is now controlled by the UI.
# MAX_SEARCH_RESULTS = 50

class BackendWorker(QObject):
    """
    A dedicated worker that owns the ImageDatabase and ImageEmbedder instances.
    It lives in a separate thread and communicates with the UI via signals.
    """
    error = Signal(str)
    initialized = Signal()
    results_ready = Signal(list)
    status_update = Signal(str)
    
    # --- MODIFIED SIGNAL: Emits processed data for the native visualization ---
    # The list contains: [(x_coord, y_coord, cluster_id, filepath), ...]
    visualization_data_ready = Signal(list) 

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

    # --- MODIFIED --- Slot now accepts an integer for the number of results.
    @Slot(str, int)
    def perform_text_search(self, query: str, top_k: int):
        if not self.db: return
        logger.info(f"Performing text search for: '{query}' with top_k={top_k}")
        self.status_update.emit(f"Searching for: '{query}'...")
        # --- MODIFIED --- Pass the top_k value from the UI to the database search.
        results = self.db.search_by_text(text_query=query, top_k=top_k)
        self.results_ready.emit(results)

    # --- MODIFIED --- Slot now accepts an integer for the number of results.
    @Slot(str, int)
    def perform_image_search(self, image_path: str, top_k: int):
        if not self.db: return
        logger.info(f"Performing image search for: {image_path} with top_k={top_k}")
        self.status_update.emit(f"Searching for images similar to {Path(image_path).name}...")
        # --- MODIFIED --- Pass the top_k value from the UI to the database search.
        results = self.db.search_similar_images(image_path=image_path, top_k=top_k)

        self.results_ready.emit(results)
    
    # --- ADDED SLOT: Performs UMAP/HDBSCAN in the worker thread ---
    @Slot()
    def request_visualization_data(self):
        if not self.db: 
            self.error.emit("Database not initialized for visualization request.")
            return

        self.status_update.emit("Preparing 2D visualization data (UMAP/HDBSCAN)...")
        
        try:
            # 1. Fetch data
            embedding_dicts = self.db.get_all_embeddings_with_filepaths()
            if not embedding_dicts:
                self.status_update.emit("Visualization failed: No images found in database.")
                return

            all_embeddings = np.array([item["embedding"] for item in embedding_dicts])
            filepaths = [item["filepath"] for item in embedding_dicts]
            
            # 2. UMAP (Dimensionality Reduction)
            logger.info("Performing UMAP...")
            reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42) 
            coords_2d = reducer.fit_transform(all_embeddings)
            
            # 3. HDBSCAN (Clustering)
            logger.info("Clustering with HDBSCAN...")
            clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=None)
            cluster_labels = clusterer.fit_predict(coords_2d)

            # 4. Prepare the final list for the UI thread
            plot_data = []
            for i, fp in enumerate(filepaths):
                # Data format: (x, y, cluster, filepath)
                plot_data.append((float(coords_2d[i, 0]), float(coords_2d[i, 1]), int(cluster_labels[i]), fp))
            
            self.status_update.emit(f"Visualization data ready for {len(plot_data)} images.")
            self.visualization_data_ready.emit(plot_data)
            
        except (ImportError, AttributeError) as e:
            # Catch both import errors and attribute errors for robustness
            msg = (
                f"Visualization failed due to missing or incorrectly installed package: {type(e).__name__}.\n"
                "Please ensure all visualization packages are installed:\n"
                "pip install umap-learn hdbscan pyqtgraph"
            )
            logger.error(msg)
            logger.error(traceback.format_exc())
            self.error.emit(msg)
        except Exception as e:
            logger.error("--- ERROR IN VISUALIZATION DATA PREPARATION ---")
            logger.error(traceback.format_exc())
            self.error.emit(traceback.format_exc())
    
    @Slot()
    def shutdown(self):
        """Safely closes the database connection in the correct thread."""
        if self.db:
            logger.info("Backend shutting down, closing DB connection.")
            self.db.close()