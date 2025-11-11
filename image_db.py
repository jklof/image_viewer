import hashlib
import logging
import sqlite3
import threading
import queue
import concurrent.futures
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Callable, Tuple

import numpy as np
from PIL import Image
import umap
import hdbscan

from ml_core import ImageEmbedder

logger = logging.getLogger(__name__)

# --- Configuration for the new parallel pipeline ---
HASHING_WORKER_COUNT = max(1, os.cpu_count() // 2)
EMBEDDING_BATCH_SIZE = 32
PIPELINE_QUEUE_SIZE = 256
SQLITE_VARIABLE_LIMIT = 900  # A safe limit for IN clauses


def _targeted_hashing_worker(filepath: Path, mtime: float) -> tuple[Path, str, float] | None:
    """
    Worker to perform the hash for a targeted file.
    """
    try:
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return filepath, sha256_hash.hexdigest(), mtime
    except (IOError, FileNotFoundError, OSError) as e:
        # Log failures for traceability instead of failing silently.
        logger.debug(f"Hashing failed for {filepath}: {e}")
        return None


class EmbeddingConsumerThread(threading.Thread):
    """
    A dedicated consumer thread that pulls file data from a queue, generates
    embeddings for new images, and writes all results to the database.
    """

    def __init__(self, db_path: str, work_queue: queue.Queue, embedder: ImageEmbedder):
        super().__init__()
        self.db_path = db_path
        self.work_queue = work_queue
        self.embedder = embedder
        self.conn = None

    def run(self):
        """The main loop for the consumer thread."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA foreign_keys = ON;")

        batch = []
        while True:
            try:
                item = self.work_queue.get(timeout=0.5)
                if item is None:
                    if batch:
                        self._process_batch(batch)
                    break
                batch.append(item)
                if len(batch) >= EMBEDDING_BATCH_SIZE:
                    self._process_batch(batch)
                    batch = []
            except queue.Empty:
                if batch:
                    self._process_batch(batch)
                    batch = []
        if self.conn:
            self.conn.close()

    def _process_batch(self, batch: List[tuple[Path, str, float]]):
        """Processes a batch of files: finds new images, embeds them, and commits."""
        # Add guard clause to prevent crash on empty batch.
        if not batch:
            return

        filepaths, shas, mtimes = zip(*batch)

        # Process SHA lookups in chunks to avoid SQLite variable limits.
        shas_to_check = list(set(shas))
        existing_shas_in_db = set()
        for i in range(0, len(shas_to_check), SQLITE_VARIABLE_LIMIT):
            chunk = shas_to_check[i : i + SQLITE_VARIABLE_LIMIT]
            placeholders = ", ".join("?" for _ in chunk)
            cursor = self.conn.execute(
                f"SELECT sha256 FROM embeddings WHERE sha256 IN ({placeholders})",
                chunk,
            )
            existing_shas_in_db.update(row[0] for row in cursor.fetchall())

        new_image_data = []
        sha_to_filepath_map = defaultdict(list)
        for fp, sha in zip(filepaths, shas):
            sha_to_filepath_map[sha].append(fp)
        for sha in shas_to_check:
            if sha not in existing_shas_in_db:
                # Deterministically pick the representative path by sorting.
                representative_filepath = sorted(sha_to_filepath_map[sha])[0]
                new_image_data.append((sha, representative_filepath))

        new_embeddings_to_commit = []
        if new_image_data:
            shas_to_embed, image_paths_to_load = zip(*new_image_data)
            pil_images, valid_shas = [], []
            for sha, path in zip(shas_to_embed, image_paths_to_load):
                try:
                    pil_images.append(Image.open(path).convert("RGB"))
                    valid_shas.append(sha)
                except Exception as e:
                    logger.warning(f"Could not load image {path} for embedding. Skipping. Error: {e}")
            if pil_images:
                try:
                    embeddings = self.embedder.embed_batch(pil_images)
                    new_embeddings_to_commit = [(sha, emb.tobytes()) for sha, emb in zip(valid_shas, embeddings)]
                except Exception as e:
                    logger.error(f"Failed to embed a batch of {len(pil_images)} images. Error: {e}", exc_info=True)
        with self.conn:
            if new_embeddings_to_commit:
                self.conn.executemany(
                    "INSERT OR IGNORE INTO embeddings (sha256, embedding) VALUES (?, ?)",
                    new_embeddings_to_commit,
                )
            filepath_data = [(str(fp), sha, mt) for fp, sha, mt in zip(filepaths, shas, mtimes)]
            self.conn.executemany(
                "INSERT OR REPLACE INTO filepaths (filepath, sha256, mtime) VALUES (?, ?, ?)", filepath_data
            )


class ImageDatabase:
    """
    Manages a database of image embeddings for local AI-powered searching.
    """

    class InterruptedError(Exception):
        """Custom exception for clean cancellation handling."""

    class ModelMismatchError(Exception):
        """Custom exception for when the DB model and app model conflict."""

    def __init__(self, db_path="images.db", embedder: ImageEmbedder = None):
        if not isinstance(embedder, ImageEmbedder):
            raise TypeError("An instance of ImageEmbedder is required for database operations.")
        self.db_path = db_path
        self.embedder = embedder

        self._create_tables()
        self._verify_model_compatibility()
        self._shas_in_order: List[str] = []
        self._embedding_matrix: np.ndarray | None = None
        self._sha_to_path_map_cache: Dict | None = None
        self._cancel_flag = threading.Event()
        self._load_embeddings_into_memory()

    # --- Database Connection Management ---

    def _get_db_connection(self):
        """Creates a new, thread-safe database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    # --- Table and Metadata Management ---

    def _create_tables(self):
        with self._get_db_connection() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS embeddings (sha256 TEXT PRIMARY KEY, embedding BLOB NOT NULL)")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS filepaths (
                    filepath TEXT PRIMARY KEY,
                    sha256 TEXT NOT NULL,
                    mtime REAL NOT NULL,
                    FOREIGN KEY (sha256) REFERENCES embeddings (sha256) ON DELETE CASCADE
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS visualization (
                    sha256 TEXT PRIMARY KEY,
                    coord_x REAL NOT NULL,
                    coord_y REAL NOT NULL,
                    cluster_id INTEGER NOT NULL,
                    FOREIGN KEY (sha256) REFERENCES embeddings (sha256) ON DELETE CASCADE
                )"""
            )
            conn.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    def _get_metadata(self, key: str) -> str | None:
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM metadata WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row[0] if row else None

    def _set_metadata(self, key: str, value: str):
        with self._get_db_connection() as conn:
            conn.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", (key, value))

    def _verify_model_compatibility(self):
        db_model_id = self._get_metadata("model_id")
        if db_model_id and db_model_id != self.embedder.model_id:
            raise self.ModelMismatchError(
                f"Database was created with model '{db_model_id}', but application is configured "
                f"to use '{self.embedder.model_id}'. Please update config.yml or run a new sync."
            )

    # --- Synchronization Coordinator ---

    def reconcile_database(
        self,
        configured_dirs: List[str],
        progress_callback: Callable = None,
        status_callback: Callable = None,
    ):
        """A resilient, parallelized synchronization method with progress reporting and cancellation."""
        self._cancel_flag.clear()

        def _check_cancelled():
            if self._cancel_flag.is_set():
                raise self.InterruptedError("Synchronization was cancelled.")

        executor = concurrent.futures.ProcessPoolExecutor(max_workers=HASHING_WORKER_COUNT)
        try:
            logger.info("Starting database synchronization...")
            self._reconcile_model_id()
            _check_cancelled()

            if status_callback:
                status_callback("Discovering files...")
            db_files = self._get_tracked_files_from_db()
            disk_files, candidate_paths = self._discover_image_files_on_disk(configured_dirs, _check_cancelled)
            _check_cancelled()

            if status_callback:
                status_callback(f"Analyzing {len(disk_files)} files for changes...")
            changes = self._calculate_file_changes(db_files, disk_files)

            if status_callback and changes["removed"]:
                status_callback(f"Removing {len(changes['removed'])} old files...")
            self._remove_deleted_files_from_db(changes["removed"])

            if not changes["to_hash"]:
                logger.info("No new or modified files to process.")
                self._finalize_sync(status_callback, _check_cancelled)
                return

            self._run_hashing_and_embedding_pipeline(
                changes["to_hash"],
                disk_files,
                candidate_paths,
                executor,
                progress_callback,
                status_callback,
                _check_cancelled,
            )

            self._finalize_sync(status_callback, _check_cancelled)
        finally:
            logger.info("Shutting down process pool executor.")
            executor.shutdown(wait=True, cancel_futures=True)
            logger.info("Process pool executor shut down.")

    # --- Synchronization Phases ---

    def _reconcile_model_id(self):
        db_model_id = self._get_metadata("model_id")
        config_model_id = self.embedder.model_id
        if db_model_id != config_model_id:
            if db_model_id is not None:
                logger.warning(
                    f"CONFIG-DB MODEL MISMATCH: Config model is '{config_model_id}' but DB was built with '{db_model_id}'. All existing embeddings will be deleted to rebuild the database."
                )
                with self._get_db_connection() as conn:
                    conn.execute("DELETE FROM embeddings")
                    conn.execute("DELETE FROM visualization")
            else:
                logger.info(f"Stamping new database with model ID: '{config_model_id}'")
            self._set_metadata("model_id", config_model_id)

    def _get_tracked_files_from_db(self) -> Dict[str, float]:
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT filepath, mtime FROM filepaths")
            return {row[0]: row[1] for row in cursor.fetchall()}

    def _discover_image_files_on_disk(
        self, configured_dirs: List[str], check_cancelled: Callable
    ) -> Tuple[Dict[str, float], Dict[str, Path]]:
        logger.info("Scanning configured directories for image files...")
        disk_files_candidate: Dict[str, float] = {}
        candidate_path_obj: Dict[str, Path] = {}

        for directory in configured_dirs:
            path = Path(directory)
            if not path.is_dir():
                continue
            for p in path.rglob("*"):
                if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    try:
                        absolute_path = p.absolute()
                        candidate_path_str = str(absolute_path)
                        disk_files_candidate[candidate_path_str] = p.stat().st_mtime
                        candidate_path_obj[candidate_path_str] = absolute_path
                    except FileNotFoundError:
                        continue
            check_cancelled()
        logger.info(f"Discovered {len(disk_files_candidate)} image files on disk.")
        return disk_files_candidate, candidate_path_obj

    def _calculate_file_changes(self, db_files: Dict[str, float], disk_files: Dict[str, float]) -> Dict[str, Set[str]]:
        db_paths = set(db_files.keys())
        disk_paths = set(disk_files.keys())

        removed_paths = db_paths - disk_paths
        added_paths = disk_paths - db_paths
        potential_modified = disk_paths.intersection(db_paths)
        modified_paths = {p for p in potential_modified if disk_files[p] > db_files[p]}

        return {"removed": removed_paths, "to_hash": added_paths.union(modified_paths)}

    def _remove_deleted_files_from_db(self, removed_paths: Set[str]):
        if not removed_paths:
            return
        with self._get_db_connection() as conn:
            placeholders = ", ".join("?" for _ in removed_paths)
            conn.execute(f"DELETE FROM filepaths WHERE filepath IN ({placeholders})", list(removed_paths))

    def _run_hashing_and_embedding_pipeline(
        self,
        paths_to_hash: Set[str],
        disk_files: Dict[str, float],
        candidate_paths: Dict[str, Path],
        executor,
        progress_callback: Callable,
        status_callback: Callable,
        check_cancelled: Callable,
    ):
        logger.info(f"Hashing {len(paths_to_hash)} new or modified files...")
        if status_callback:
            status_callback(f"Hashing {len(paths_to_hash)} files...")

        work_queue = queue.Queue(maxsize=PIPELINE_QUEUE_SIZE)
        consumer = EmbeddingConsumerThread(self.db_path, work_queue, self.embedder)
        consumer.start()

        hashing_jobs = [(candidate_paths[p], disk_files[p]) for p in paths_to_hash]
        future_to_path = {
            executor.submit(_targeted_hashing_worker, path_obj, mtime): path_obj for path_obj, mtime in hashing_jobs
        }

        for i, future in enumerate(concurrent.futures.as_completed(future_to_path)):
            check_cancelled()
            result = future.result()
            if result:
                while True:
                    try:
                        work_queue.put(result, timeout=1.0)
                        break
                    except queue.Full:
                        check_cancelled()
                        continue
            if progress_callback:
                progress_callback("hashing", i + 1, len(hashing_jobs))

        if status_callback:
            status_callback("Embedding new images...")
        work_queue.put(None)
        consumer.join()
        check_cancelled()

    def _finalize_sync(self, status_callback: Callable, check_cancelled: Callable):
        logger.info("Finalizing sync: cleaning up and updating data...")
        self._cleanup_orphaned_embeddings()
        if status_callback:
            status_callback("Updating visualization...")
        self._update_visualization_data(status_callback=status_callback, check_cancelled_callback=check_cancelled)
        self._load_embeddings_into_memory()

    # --- Data Loading and Cleanup ---

    def _load_embeddings_into_memory(self):
        self._sha_to_path_map_cache = None
        logger.info("Loading embeddings from database into memory...")
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT sha256, embedding FROM embeddings")
            rows = cursor.fetchall()
        if not rows:
            self._shas_in_order, self._embedding_matrix = [], None
            return
        self._shas_in_order, embeddings_blobs = zip(*rows)
        embeddings_list = [self._reconstruct_embedding(b) for b in embeddings_blobs]
        self._embedding_matrix = np.vstack(embeddings_list)
        logger.info(f"Loaded {len(self._shas_in_order)} embeddings into memory cache.")

    def _cleanup_orphaned_embeddings(self):
        with self._get_db_connection() as conn:
            res = conn.execute("DELETE FROM embeddings WHERE sha256 NOT IN (SELECT DISTINCT sha256 FROM filepaths)")
            if res.rowcount > 0:
                logger.info(f"Removed {res.rowcount} orphaned embeddings.")

    def _reconstruct_embedding(self, embedding_blob: bytes) -> np.ndarray:
        return np.frombuffer(embedding_blob, dtype=self.embedder.embedding_dtype).reshape(self.embedder.embedding_shape)

    # --- Visualization Management ---

    def _update_visualization_data(self, status_callback: Callable = None, check_cancelled_callback: Callable = None):
        logger.info("Checking if visualization data needs to be updated...")
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
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
            with self._get_db_connection() as conn:
                conn.execute("DELETE FROM visualization")
            return

        all_embeddings = np.array([item["embedding"] for item in embedding_data])
        all_shas = [item["sha256"] for item in embedding_data]

        try:
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
        except Exception as e:
            logger.error(f"Visualization calculation failed. Skipping update. Error: {e}", exc_info=True)
            return

        vis_data = [(sh, float(c[0]), float(c[1]), int(l)) for sh, c, l in zip(all_shas, coords_2d, cluster_labels)]
        with self._get_db_connection() as conn:
            conn.execute("DELETE FROM visualization")
            conn.executemany(
                "INSERT INTO visualization (sha256, coord_x, coord_y, cluster_id) VALUES (?, ?, ?, ?)", vis_data
            )
        logger.info("Visualization data successfully updated.")

    # --- Search and Query Methods ---

    def _build_sha_to_path_map(self):
        logger.info("Building SHA-to-filepath cache...")
        sha_to_path_map = defaultdict(list)
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT sha256, filepath FROM filepaths")
            for sha, path in cursor.fetchall():
                sha_to_path_map[sha].append(path)
        self._sha_to_path_map_cache = sha_to_path_map
        logger.info(f"Cache built with {len(self._sha_to_path_map_cache)} unique SHAs.")

    def _perform_search(self, query_embedding: np.ndarray, top_k: int) -> List[tuple[float, str]]:
        if self._embedding_matrix is None or len(self._embedding_matrix) == 0:
            return []
        if self._sha_to_path_map_cache is None:
            self._build_sha_to_path_map()

        similarities = np.dot(query_embedding, self._embedding_matrix.T).flatten()
        if top_k != -1 and top_k < len(similarities):
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_results_unsorted = [(similarities[i], self._shas_in_order[i]) for i in top_indices]
            top_results = sorted(top_results_unsorted, key=lambda x: x[0], reverse=True)
        else:
            all_indices = np.argsort(similarities)[::-1]
            top_results = [(similarities[i], self._shas_in_order[i]) for i in all_indices]

        final_results = [
            (float(score), self._sha_to_path_map_cache[sha][0])
            for score, sha in top_results
            if sha in self._sha_to_path_map_cache
        ]
        return final_results if top_k == -1 else final_results[:top_k]

    def search_similar_images(self, image_path: str, top_k: int = -1):
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            logger.error(f"Query image not found at '{image_path}'")
            return []
        query_embedding = self.embedder.embed_image(image)
        if top_k == -1:
            return self._perform_search(query_embedding, -1)
        similar_images = self._perform_search(query_embedding, top_k + 1)
        other_results = [(s, p) for s, p in similar_images if Path(p).resolve() != Path(image_path).resolve()]
        return [(1.0, image_path)] + other_results[: top_k - 1]

    def search_by_text(self, text_query: str, top_k: int = -1):
        query_embedding = self.embedder.embed_text(text_query)
        return self._perform_search(query_embedding, top_k)

    # --- Utility and Public Interface Methods ---

    def cancel_sync(self):
        self._cancel_flag.set()

    def get_all_unique_filepaths(self) -> List[str]:
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MIN(filepath) FROM filepaths GROUP BY sha256")
            return [row[0] for row in cursor.fetchall()]

    def get_all_filepaths_with_mtime(self) -> List[tuple[str, float]]:
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT filepath, mtime FROM filepaths")
            return cursor.fetchall()

    def get_all_embeddings_with_shas(self) -> List[Dict]:
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT sha256, embedding FROM embeddings")
            return [
                {"sha256": sha, "embedding": self._reconstruct_embedding(eb).flatten()} for sha, eb in cursor.fetchall()
            ]

    def get_visualization_data(self) -> List[tuple[float, float, int, str]]:
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT v.coord_x, v.coord_y, v.cluster_id, f.filepath
                FROM visualization v
                JOIN (SELECT sha256, MIN(filepath) as filepath FROM filepaths GROUP BY sha256) f
                ON v.sha256 = f.sha256
                """
            )
            return cursor.fetchall()
