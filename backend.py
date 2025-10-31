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
        if not self.db:
            self.error.emit("Database not initialized for visualization request.")
            return
        self.status_update.emit("Preparing 2D visualization data (UMAP/HDBSCAN)...")
        try:
            embedding_dicts = self.db.get_all_embeddings_with_filepaths()
            if not embedding_dicts:
                self.status_update.emit("Visualization failed: No images found in database.")
                return
            all_embeddings = np.array([item["embedding"] for item in embedding_dicts])
            filepaths = [item["filepath"] for item in embedding_dicts]
            logger.info("Performing UMAP...")
            reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
            coords_2d = reducer.fit_transform(all_embeddings)
            logger.info("Clustering with HDBSCAN...")
            clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=None)
            cluster_labels = clusterer.fit_predict(coords_2d)
            plot_data = []
            for i, fp in enumerate(filepaths):
                plot_data.append(
                    (
                        float(coords_2d[i, 0]),
                        float(coords_2d[i, 1]),
                        int(cluster_labels[i]),
                        fp,
                    )
                )
            self.status_update.emit(f"Visualization data ready for {len(plot_data)} images.")
            self.visualization_data_ready.emit(plot_data)
        except (ImportError, AttributeError) as e:
            msg = (
                f"Visualization failed: {type(e).__name__}.\n"
                "Please install all visualization packages:\n"
                "pip install umap-learn hdbscan pyqtgraph"
            )
            logger.error(msg, exc_info=True)
            self.error.emit(msg)
        except Exception as e:
            logger.error("Error in visualization data preparation", exc_info=True)
            self.error.emit(traceback.format_exc())

    @Slot()
    def shutdown(self):
        if self.db:
            logger.info("Backend shutting down, closing DB connection.")
            self.db.close()
