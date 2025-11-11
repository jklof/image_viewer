import hashlib
import logging
import sqlite3
import threading
import queue
import concurrent.futures
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Callable

import numpy as np
from PIL import Image
import umap
import hdbscan

from ml_core import ImageEmbedder

logger = logging.getLogger(__name__)

# --- Configuration for the new parallel pipeline ---
# Hashing workers are only used for the small set of added/modified files now.
HASHING_WORKER_COUNT = max(1, os.cpu_count() // 2)
EMBEDDING_BATCH_SIZE = 32
PIPELINE_QUEUE_SIZE = 256


# NOTE: Worker now accepts a Path object that is NOT necessarily resolved,
# but is the path string we want to store in the DB.
def _targeted_hashing_worker(filepath: Path, mtime: float) -> tuple[Path, str, float] | None:
    """
    Worker to perform the hash for the targeted file.
    The filepath passed is the canonical path string that will be stored in the DB.
    """
    try:
        # File I/O for hashing
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        # Returns the canonical Path object, the SHA, and the original mtime
        return filepath, sha256_hash.hexdigest(), mtime
    except (IOError, FileNotFoundError, OSError):
        return None


class EmbeddingConsumerThread(threading.Thread):
    """
    A dedicated consumer thread that pulls file data from a queue, generates
    embeddings for new images, and writes all results to the database.
    This isolates all GPU and database write operations to a single thread.
    """

    def __init__(self, db_path: str, work_queue: queue.Queue, embedder: ImageEmbedder):
        super().__init__(daemon=True)
        self.db_path = db_path
        self.work_queue = work_queue
        self.embedder = embedder
        self.conn = None

    def run(self):
        """The main loop for the consumer thread."""
        # --- Each thread must have its own DB connection ---
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA foreign_keys = ON;")

        batch = []
        while True:
            try:
                # Block until an item is available or timeout occurs
                item = self.work_queue.get(timeout=0.5)

                if item is None:  # Sentinel value received
                    logger.info("Embedding thread received shutdown signal.")
                    if batch:
                        self._process_batch(batch)  # Process any remaining items
                    break

                batch.append(item)
                if len(batch) >= EMBEDDING_BATCH_SIZE:
                    self._process_batch(batch)
                    batch = []

            except queue.Empty:
                # Timeout occurred, process the current batch if any items exist
                if batch:
                    logger.debug(f"Processing partial batch of size {len(batch)} due to timeout.")
                    self._process_batch(batch)
                    batch = []

        if self.conn:
            self.conn.close()
            logger.info("Embedding thread database connection closed.")

    def _process_batch(self, batch: List[tuple[Path, str, float]]):
        """
        Processes a batch of files: finds new images, embeds them,
        and commits all file information to the database.
        """
        if not self.conn:
            return

        # Separate data for easier processing
        # Note: fp is now the canonical (resolved) Path object from the worker
        filepaths = [item[0] for item in batch]
        shas = [item[1] for item in batch]
        mtimes = [item[2] for item in batch]

        # --- Efficiently find which SHAs are genuinely new ---
        shas_to_check = list(set(shas))
        placeholders = ", ".join("?" for _ in shas_to_check)
        cursor = self.conn.execute(
            f"SELECT sha256 FROM embeddings WHERE sha256 IN ({placeholders})",
            shas_to_check,
        )
        existing_shas_in_db = {row[0] for row in cursor.fetchall()}

        # --- Identify the images that need embedding ---
        new_image_data = []  # List of (sha, filepath) for new images
        sha_to_filepath_map = defaultdict(list)
        for fp, sha in zip(filepaths, shas):
            sha_to_filepath_map[sha].append(fp)

        for sha in shas_to_check:
            if sha not in existing_shas_in_db:
                # Pick the first filepath as a representative for this SHA
                representative_filepath = sha_to_filepath_map[sha][0]
                new_image_data.append((sha, representative_filepath))

        # --- Embed new images (if any) ---
        new_embeddings_to_commit = []
        if new_image_data:
            shas_to_embed = [item[0] for item in new_image_data]
            image_paths_to_load = [item[1] for item in new_image_data]

            pil_images, valid_shas = [], []
            for sha, path in zip(shas_to_embed, image_paths_to_load):
                try:
                    pil_images.append(Image.open(path).convert("RGB"))
                    valid_shas.append(sha)
                except Exception as e:
                    logger.warning(f"Could not load image {path} for embedding. Skipping. Error: {e}")

            if pil_images:
                embeddings = self.embedder.embed_batch(pil_images)
                new_embeddings_to_commit = [(sha, emb.tobytes()) for sha, emb in zip(valid_shas, embeddings)]

        # --- Commit everything in a single transaction ---
        with self.conn:
            if new_embeddings_to_commit:
                self.conn.executemany(
                    "INSERT OR IGNORE INTO embeddings (sha256, embedding) VALUES (?, ?)",
                    new_embeddings_to_commit,
                )

            # Update all filepaths in the batch (both new and duplicates)
            filepath_data = [(str(fp), sha, mt) for fp, sha, mt in zip(filepaths, shas, mtimes)]
            self.conn.executemany(
                "INSERT OR REPLACE INTO filepaths (filepath, sha256, mtime) VALUES (?, ?, ?)", filepath_data
            )


class ImageDatabase:
    """
    Manages a database of image embeddings for local AI-powered searching.
    All ML logic is handled by the provided 'ImageEmbedder' instance.
    This version pre-loads embeddings into memory for fast, vectorized searching.
    """

    class InterruptedError(Exception):
        """Custom exception for clean cancellation handling."""

        pass

    class ModelMismatchError(Exception):
        """Custom exception for when the DB model and app model conflict."""

        pass

    def __init__(self, db_path="images.db", embedder: ImageEmbedder = None):
        if not isinstance(embedder, ImageEmbedder):
            raise TypeError("An instance of ImageEmbedder is required for database operations.")

        self.db_path = db_path
        self.embedder = embedder
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self._create_tables()
        self._verify_model_compatibility()

        self._shas_in_order: List[str] = []
        self._embedding_matrix: np.ndarray | None = None
        self._cancel_flag = threading.Event()  # For cancelling sync
        self._load_embeddings_into_memory()

    def _create_tables(self):
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    sha256 TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS filepaths (
                    filepath TEXT PRIMARY KEY,
                    sha256 TEXT NOT NULL,
                    mtime REAL NOT NULL,
                    FOREIGN KEY (sha256) REFERENCES embeddings (sha256) ON DELETE CASCADE
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS visualization (
                    sha256 TEXT PRIMARY KEY,
                    coord_x REAL NOT NULL,
                    coord_y REAL NOT NULL,
                    cluster_id INTEGER NOT NULL,
                    FOREIGN KEY (sha256) REFERENCES embeddings (sha256) ON DELETE CASCADE
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )

    def _get_metadata(self, key: str) -> str | None:
        """Retrieves a value from the metadata table."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT value FROM metadata WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row[0] if row else None

    def _set_metadata(self, key: str, value: str):
        """Sets a key-value pair in the metadata table."""
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", (key, value))

    def _verify_model_compatibility(self):
        """Checks if the embedder model matches the one stored in the DB."""
        db_model_id = self._get_metadata("model_id")
        # If a model ID is stored and it doesn't match the current embedder, raise an error.
        if db_model_id and db_model_id != self.embedder.model_id:
            raise self.ModelMismatchError(
                f"Database was created with model '{db_model_id}', but application is configured "
                f"to use '{self.embedder.model_id}'. Please update config.yml or run a new sync."
            )

    def _reconcile_model_id(self):
        """
        Ensures the DB is stamped with the correct model ID. If the model has
        changed, this will wipe all incompatible data before proceeding.
        """
        db_model_id = self._get_metadata("model_id")
        config_model_id = self.embedder.model_id

        if db_model_id != config_model_id:
            if db_model_id is not None:
                logger.warning(
                    f"CONFIG-DB MODEL MISMATCH: Config model is '{config_model_id}' but DB was built "
                    f"with '{db_model_id}'. All existing embeddings will be deleted to rebuild the database."
                )
                with self.conn:
                    self.conn.execute("DELETE FROM embeddings")
                    self.conn.execute("DELETE FROM visualization")
            else:
                logger.info(f"Stamping new database with model ID: '{config_model_id}'")

            self._set_metadata("model_id", config_model_id)

    def _load_embeddings_into_memory(self):
        logger.info("Loading embeddings from database into memory...")
        cursor = self.conn.cursor()
        cursor.execute("SELECT sha256, embedding FROM embeddings")

        rows = cursor.fetchall()
        if not rows:
            logger.info("No embeddings found in the database.")
            self._shas_in_order = []
            self._embedding_matrix = None
            return

        self._shas_in_order = [row[0] for row in rows]
        embeddings_list = [self._reconstruct_embedding(row[1]) for row in rows]
        self._embedding_matrix = np.vstack(embeddings_list)
        logger.info(f"Loaded {len(self._shas_in_order)} embeddings into memory cache.")

    def _reconstruct_embedding(self, embedding_blob: bytes) -> np.ndarray:
        return np.frombuffer(embedding_blob, dtype=self.embedder.embedding_dtype).reshape(self.embedder.embedding_shape)

    def cancel_sync(self):
        """Sets a flag to gracefully interrupt the sync process."""
        self._cancel_flag.set()

    def reconcile_database(
        self,
        configured_dirs: list[str],
        progress_callback: Callable = None,
        status_callback: Callable = None,
    ):
        """
        A resilient, parallelized synchronization method with progress reporting and cancellation.
        """
        self._cancel_flag.clear()  # Reset flag at the start of a new sync

        def _check_cancelled():
            if self._cancel_flag.is_set():
                raise self.InterruptedError("Synchronization was cancelled.")

        # Re-initializing executor here for targeted use
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=HASHING_WORKER_COUNT)
        try:
            logger.info("Starting database synchronization...")

            # --- Phase 0: Model ID Reconciliation ---
            self._reconcile_model_id()
            _check_cancelled()

            # --- Phase 1: Discovery (Optimized String-Based Check) ---
            if status_callback:
                status_callback("Discovering files...")

            cursor = self.conn.cursor()
            # db_files is our canonical reference: {path_str: mtime}
            # NOTE: These path strings were GENERATED via .resolve() in the past.
            cursor.execute("SELECT filepath, mtime FROM filepaths")
            db_files = {row[0]: row[1] for row in cursor.fetchall()}
            db_paths = set(db_files.keys())

            logger.info("Scanning configured directories for image files...")

            all_image_paths = []
            for directory in configured_dirs:
                path = Path(directory)
                if not path.is_dir():
                    continue
                for p in path.rglob("*"):
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        all_image_paths.append(p)
                _check_cancelled()

            logger.info(f"Discovered {len(all_image_paths)} image files on disk.")

            # --- Phase 2: SINGLE-THREADED STRING/MTIME CHECK (AVOIDS resolve() bottleneck) ---
            # We must use the *current* string representation for lookup.
            # This is the trade-off: FAST but risks missing a symlink change.
            if status_callback:
                status_callback(f"Analyzing {len(all_image_paths)} files for changes (String Check)...")

            disk_files_candidate: Dict[str, float] = {}
            # Store the Path object corresponding to the candidate string
            candidate_path_obj: Dict[str, Path] = {}

            for p in all_image_paths:
                try:
                    # **SPEEDUP HERE**: Use str(p) or str(p.absolute()) (faster than resolve())
                    # Using str(p.absolute()) is safer for cross-OS/mount consistency.
                    candidate_path_str = str(p.absolute())
                    disk_files_candidate[candidate_path_str] = p.stat().st_mtime
                    candidate_path_obj[candidate_path_str] = p
                except FileNotFoundError:
                    continue
            _check_cancelled()

            logger.info("Comparing database records with on-disk files...")

            # --- Perform Comparison and Identify Sets ---
            disk_paths_candidate = set(disk_files_candidate.keys())

            # A file is REMOVED if its canonical DB path is NOT in the new candidates
            removed_paths = db_paths - disk_paths_candidate

            added_paths_candidate = disk_paths_candidate - db_paths  # New path string
            potential_modified = disk_paths_candidate.intersection(db_paths)

            modified_paths_candidate = {p for p in potential_modified if disk_files_candidate[p] > db_files[p]}

            paths_to_hash_candidate = added_paths_candidate.union(modified_paths_candidate)

            # --- Phase 3: Immediate Deletions ---
            if removed_paths:
                if status_callback:
                    status_callback(f"Removing {len(removed_paths)} old files...")
                with self.conn:
                    placeholders = ", ".join("?" for _ in removed_paths)
                    self.conn.execute(f"DELETE FROM filepaths WHERE filepath IN ({placeholders})", list(removed_paths))

            if not paths_to_hash_candidate:
                # ... (No new or modified files to process, jump to cleanup/visualization)
                logger.info("No new or modified files to process.")
                self._cleanup_orphaned_embeddings()
                if status_callback:
                    status_callback("Updating visualization...")
                self._update_visualization_data(
                    status_callback=status_callback, check_cancelled_callback=_check_cancelled
                )
                self._load_embeddings_into_memory()
                return

            # --- Phase C & 4: Targeted Parallel Hashing and Embedding Pipeline ---
            logger.info(f"Hashing {len(paths_to_hash_candidate)} new or modified files...")

            work_queue = queue.Queue(maxsize=PIPELINE_QUEUE_SIZE)
            consumer = EmbeddingConsumerThread(self.db_path, work_queue, self.embedder)
            consumer.start()

            if status_callback:
                status_callback(f"Hashing {len(paths_to_hash_candidate)} files...")

            # Prepare the jobs for the Targeted Hashing Worker: (Path object, mtime)
            hashing_jobs = []
            for candidate_str in paths_to_hash_candidate:
                path_obj = candidate_path_obj[candidate_str]
                mtime = disk_files_candidate[candidate_str]
                hashing_jobs.append((path_obj, mtime))

            future_to_path = {
                executor.submit(_targeted_hashing_worker, path_obj, mtime): path_obj for path_obj, mtime in hashing_jobs
            }

            # Collect results from the Hashing Worker and feed the Embedding Consumer
            for i, future in enumerate(concurrent.futures.as_completed(future_to_path)):
                _check_cancelled()
                result = future.result()  # result is (Path, SHA, mtime)
                if result:
                    # Pass directly to the embedding queue
                    while True:
                        _check_cancelled()
                        try:
                            # Path object is the one used for the DB key
                            work_queue.put(result, timeout=0.2)
                            break
                        except queue.Full:
                            continue

                if progress_callback:
                    progress_callback("hashing", i + 1, len(hashing_jobs))

            # --- Phase 5: Coordinated Shutdown ---
            if status_callback:
                status_callback("Embedding new images...")
            work_queue.put(None)
            while consumer.is_alive():
                _check_cancelled()
                consumer.join(timeout=0.2)

            _check_cancelled()
            logger.info("Cleanup: Removing orphaned embeddings...")
            self._cleanup_orphaned_embeddings()

            logger.info("Recalculating visualization data...")

            if status_callback:
                status_callback("Updating visualization...")
            self._update_visualization_data(status_callback=status_callback, check_cancelled_callback=_check_cancelled)

            self._load_embeddings_into_memory()

        finally:
            logger.info("Shutting down process pool executor.")
            executor.shutdown(wait=True, cancel_futures=True)
            logger.info("Process pool executor shut down.")

    def _cleanup_orphaned_embeddings(self):
        """Removes embeddings that are no longer referenced by any file."""
        with self.conn:
            logger.info("Cleaning up orphaned embeddings...")
            res = self.conn.execute(
                "DELETE FROM embeddings WHERE sha256 NOT IN (SELECT DISTINCT sha256 FROM filepaths)"
            )
            if res.rowcount > 0:
                logger.info(f"Removed {res.rowcount} orphaned embeddings.")

    def _update_visualization_data(self, status_callback: Callable = None, check_cancelled_callback: Callable = None):
        """
        Calculates and stores UMAP/HDBSCAN data if the set of images has changed.
        """
        logger.info("Checking if visualization data needs to be updated...")
        cursor = self.conn.cursor()
        cursor.execute("SELECT sha256 FROM embeddings")
        embedding_shas = {row[0] for row in cursor.fetchall()}
        cursor.execute("SELECT sha256 FROM visualization")
        visualization_shas = {row[0] for row in cursor.fetchall()}

        if embedding_shas == visualization_shas:
            logger.info("Visualization data is already up to date.")
            return

        logger.info("Change detected. Recalculating all visualization data...")
        embedding_data = self.get_all_embeddings_with_shas()
        if not embedding_data:
            with self.conn:
                self.conn.execute("DELETE FROM visualization")
            return

        all_embeddings = np.array([item["embedding"] for item in embedding_data])
        all_shas = [item["sha256"] for item in embedding_data]

        if status_callback:
            status_callback("Reducing dimensions (UMAP)...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, verbose=False, n_jobs=-1)
        coords_2d = reducer.fit_transform(all_embeddings)
        if check_cancelled_callback:
            check_cancelled_callback()

        if status_callback:
            status_callback("Clustering (HDBSCAN)...")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=None, core_dist_n_jobs=-1)
        cluster_labels = clusterer.fit_predict(coords_2d)
        if check_cancelled_callback:
            check_cancelled_callback()

        vis_data = [
            (all_shas[i], float(coords_2d[i, 0]), float(coords_2d[i, 1]), int(cluster_labels[i]))
            for i in range(len(all_shas))
        ]

        with self.conn:
            self.conn.execute("DELETE FROM visualization")
            self.conn.executemany(
                "INSERT INTO visualization (sha256, coord_x, coord_y, cluster_id) VALUES (?, ?, ?, ?)", vis_data
            )
        logger.info("Visualization data successfully updated.")

    def get_all_embeddings_with_filepaths(self) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT e.embedding, f.filepath FROM embeddings e JOIN (
                SELECT sha256, MIN(filepath) as filepath FROM filepaths GROUP BY sha256
            ) f ON e.sha256 = f.sha256
            """
        )
        return [
            {"embedding": self._reconstruct_embedding(eb).flatten(), "filepath": fp} for eb, fp in cursor.fetchall()
        ]

    def get_all_embeddings_with_shas(self) -> List[Dict]:
        """Fetches all embeddings and their SHAs, optimized for visualization calculation."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT sha256, embedding FROM embeddings")
        return [
            {"sha256": sha, "embedding": self._reconstruct_embedding(eb).flatten()} for sha, eb in cursor.fetchall()
        ]

    def _perform_search(self, query_embedding: np.ndarray, top_k: int) -> List[tuple[float, str]]:
        if self._embedding_matrix is None or len(self._embedding_matrix) == 0:
            return []
        similarities = np.dot(query_embedding, self._embedding_matrix.T).flatten()
        if top_k != -1 and top_k < len(similarities):
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_results_unsorted = [(similarities[i], self._shas_in_order[i]) for i in top_indices]
            top_results = sorted(top_results_unsorted, key=lambda x: x[0], reverse=True)
        else:
            all_indices = np.argsort(similarities)[::-1]
            top_results = [(similarities[i], self._shas_in_order[i]) for i in all_indices]

        sha_to_path_map = defaultdict(list)
        cursor = self.conn.cursor()
        cursor.execute("SELECT sha256, filepath FROM filepaths")
        for sha, path in cursor.fetchall():
            sha_to_path_map[sha].append(path)
        final_results = [
            (float(score), sha_to_path_map[sha][0]) for score, sha in top_results if sha in sha_to_path_map
        ]
        return final_results if top_k == -1 else final_results[:top_k]

    def get_all_unique_filepaths(self) -> List[str]:
        """
        Retrieves a list of all unique filepaths from the database, one per SHA.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT MIN(filepath) FROM filepaths GROUP BY sha256")
        # fetchall() returns a list of tuples, e.g., [('path1',), ('path2',)]
        return [row[0] for row in cursor.fetchall()]

    def get_all_filepaths_with_mtime(self) -> List[tuple[str, float]]:
        """Retrieves all filepaths and their modification times."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT filepath, mtime FROM filepaths")
        return cursor.fetchall()

    def search_similar_images(self, image_path: str, top_k: int = -1):
        query_path = Path(image_path)
        try:
            image = Image.open(query_path).convert("RGB")
        except FileNotFoundError:
            logger.error(f"Query image not found at '{image_path}'")
            return []
        query_embedding = self.embedder.embed_image(image)

        if top_k == -1:
            return self._perform_search(query_embedding, -1)

        similar_images = self._perform_search(query_embedding, top_k + 1)
        other_results = [(s, p) for s, p in similar_images if Path(p).resolve() != query_path.resolve()]
        final_results = [(1.0, str(query_path))] + other_results
        return final_results[:top_k]

    def search_by_text(self, text_query: str, top_k: int = -1):
        query_embedding = self.embedder.embed_text(text_query)
        return self._perform_search(query_embedding, top_k)

    def close(self):
        if self.conn:
            logger.info("Closing database connection.")
            self.conn.close()

    # --- QUERY PRE-CALCULATED VISUALIZATION DATA ---
    def get_visualization_data(self) -> List[tuple[float, float, int, str]]:
        """
        Retrieves pre-calculated visualization data directly from the database.
        """
        cursor = self.conn.cursor()
        # Join visualization data with a representative filepath for each sha
        cursor.execute(
            """
            SELECT
                v.coord_x,
                v.coord_y,
                v.cluster_id,
                f.filepath
            FROM visualization v
            JOIN (
                SELECT sha256, MIN(filepath) as filepath FROM filepaths GROUP BY sha256
            ) f ON v.sha256 = f.sha256
            """
        )
        return cursor.fetchall()
