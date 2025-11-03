import hashlib
import logging
import sqlite3
import threading
import queue
import concurrent.futures
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
from PIL import Image
from tqdm import tqdm

# --- NEW IMPORTS for visualization calculation ---
import umap
import hdbscan

from ml_core import EMBEDDING_DTYPE, EMBEDDING_SHAPE, ImageEmbedder

logger = logging.getLogger(__name__)

# --- Configuration for the new parallel pipeline ---
# Number of CPU cores to use for hashing files.
HASHING_WORKER_COUNT = max(1, os.cpu_count() // 2)
# How many items the embedding thread will pull from the queue at once.
EMBEDDING_BATCH_SIZE = 32
# Max size of the queue connecting the hashers and the embedder.
# This prevents hashers from getting too far ahead and using too much memory.
PIPELINE_QUEUE_SIZE = 256


def _hash_file_worker(filepath: Path) -> tuple[Path, str, float] | None:
    """
    A standalone function for use in a ProcessPoolExecutor.
    Calculates SHA256 and mtime for a given file path.
    Returns None if the file cannot be read.
    """
    try:
        mtime = filepath.stat().st_mtime
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return filepath, sha256_hash.hexdigest(), mtime
    except (IOError, FileNotFoundError):
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

    def __init__(self, db_path="images.db", embedder: ImageEmbedder = None):
        if not isinstance(embedder, ImageEmbedder):
            raise TypeError("An instance of ImageEmbedder is required for database operations.")

        self.db_path = db_path
        self.embedder = embedder
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self._create_tables()

        self._shas_in_order: List[str] = []
        self._embedding_matrix: np.ndarray | None = None
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
            # --- NEW TABLE FOR VISUALIZATION DATA ---
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
        return np.frombuffer(embedding_blob, dtype=EMBEDDING_DTYPE).reshape(EMBEDDING_SHAPE)

    def reconcile_database(self, configured_dirs: list[str]):
        """
        A resilient, parallelized synchronization method.
        - Hashes files on multiple CPU cores.
        - Embeds new images on the GPU concurrently.
        - Work is checkpointed to the DB, making the process resumable.
        """
        logger.info("Starting database synchronization...")

        # --- Phase 1: Discovery (Fast, Serial) ---
        cursor = self.conn.cursor()
        cursor.execute("SELECT filepath, mtime FROM filepaths")

        # --- MODIFICATION 1 (FIX): Avoid slow .resolve() on every DB entry. ---
        # This is now a very fast, in-memory operation that just loads strings,
        # avoiding hundreds of thousands of slow filesystem calls.
        db_files = {row[0]: row[1] for row in cursor.fetchall()}

        logger.info("Discovering image files on disk...")
        all_image_paths = []
        with tqdm(desc="Discovering files", unit=" files") as pbar:
            for directory in configured_dirs:
                path = Path(directory)
                if not path.is_dir():
                    logger.warning(f"Configured directory '{directory}' does not exist. Skipping.")
                    continue

                pbar.set_description(f"Scanning in {path.name}")
                for p in path.rglob("*"):
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        all_image_paths.append(p)
                        pbar.update(1)

        logger.info(f"Found {len(all_image_paths)} total image files on disk.")

        # --- MODIFICATION 2: Use strings as keys to match db_files. ---
        disk_files: Dict[str, float] = {}
        for p in tqdm(all_image_paths, desc="Checking file modification times"):
            try:
                # Resolve the path and convert to a canonical string for comparison.
                disk_files[str(p.resolve())] = p.stat().st_mtime
            except FileNotFoundError:
                continue

        # Now both sets contain canonical path strings, making the comparison very fast.
        db_paths, disk_paths = set(db_files.keys()), set(disk_files.keys())
        removed_paths = db_paths - disk_paths
        added_paths = disk_paths - db_paths
        potential_modified_paths = disk_paths.intersection(db_paths)

        # Create a set of modified paths using string comparison first.
        modified_string_paths = {p for p in potential_modified_paths if disk_files[p] > db_files[p]}

        logger.info(
            f"Found {len(added_paths)} new, {len(removed_paths)} removed, "
            f"and {len(modified_string_paths)} modified files."
        )

        # --- Phase 2: Immediate Deletions ---
        if removed_paths:
            logger.info(f"Removing {len(removed_paths)} missing files from database...")
            with self.conn:
                placeholders = ", ".join("?" for _ in removed_paths)
                # The 'removed_paths' set already contains strings, so this is correct.
                self.conn.execute(
                    f"DELETE FROM filepaths WHERE filepath IN ({placeholders})",
                    list(removed_paths),
                )

        # --- MODIFICATION 3: Convert path strings back to Path objects for processing. ---
        # The processing pool expects Path objects, so we convert the final set of strings.
        string_paths_to_process = list(added_paths.union(modified_string_paths))
        paths_to_process = [Path(p) for p in string_paths_to_process]

        if not paths_to_process:
            logger.info("No new or modified files to process.")
            self._cleanup_orphaned_embeddings()
            self._update_visualization_data()
            return

        # --- Phase 3: Setup Parallel Pipeline ---
        work_queue = queue.Queue(maxsize=PIPELINE_QUEUE_SIZE)
        consumer = EmbeddingConsumerThread(self.db_path, work_queue, self.embedder)
        consumer.start()

        # --- Phase 4: Run Hashing and Feed the Pipeline ---
        logger.info(f"Hashing {len(paths_to_process)} files using {HASHING_WORKER_COUNT} processes...")
        processed_count = 0
        with tqdm(total=len(paths_to_process), desc="Processing files") as pbar:
            with concurrent.futures.ProcessPoolExecutor(max_workers=HASHING_WORKER_COUNT) as executor:
                future_to_path = {executor.submit(_hash_file_worker, path): path for path in paths_to_process}

                for future in concurrent.futures.as_completed(future_to_path):
                    result = future.result()
                    if result:
                        work_queue.put(result)

                    pbar.update(1)
                    processed_count += 1

        logger.info(f"Hashing complete. Processed {processed_count} files.")

        # --- Phase 5: Coordinated Shutdown ---
        logger.info("Waiting for embedding and database writes to complete...")
        work_queue.put(None)
        consumer.join()

        logger.info("All file processing tasks complete.")
        self._cleanup_orphaned_embeddings()

        # --- NEW: Final Step - Update visualization data if needed ---
        self._update_visualization_data()

        self._load_embeddings_into_memory()

    def _cleanup_orphaned_embeddings(self):
        """Removes embeddings that are no longer referenced by any file."""
        with self.conn:
            logger.info("Cleaning up orphaned embeddings...")
            res = self.conn.execute(
                "DELETE FROM embeddings WHERE sha256 NOT IN (SELECT DISTINCT sha256 FROM filepaths)"
            )
            if res.rowcount > 0:
                logger.info(f"Removed {res.rowcount} orphaned embeddings.")

    # --- NEW METHOD to update visualization data ---
    def _update_visualization_data(self):
        """
        Calculates and stores UMAP/HDBSCAN data if the set of images has changed.
        """
        logger.info("Checking if visualization data needs to be updated...")
        cursor = self.conn.cursor()

        # Get all current shas from the main embeddings table
        cursor.execute("SELECT sha256 FROM embeddings")
        embedding_shas = {row[0] for row in cursor.fetchall()}

        # Get all shas that already have visualization data
        cursor.execute("SELECT sha256 FROM visualization")
        visualization_shas = {row[0] for row in cursor.fetchall()}

        # If the sets are identical, no work is needed
        if embedding_shas == visualization_shas:
            logger.info("Visualization data is already up to date.")
            return

        logger.info("Change detected. Recalculating all visualization data...")

        embedding_data = self.get_all_embeddings_with_shas()
        if not embedding_data:
            logger.warning("No embeddings found to generate visualization.")
            # Ensure the table is empty if there are no embeddings
            with self.conn:
                self.conn.execute("DELETE FROM visualization")
            return

        all_embeddings = np.array([item["embedding"] for item in embedding_data])
        all_shas = [item["sha256"] for item in embedding_data]

        logger.info("Performing UMAP reduction...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, verbose=True, n_jobs=-1)
        coords_2d = reducer.fit_transform(all_embeddings)

        logger.info("Clustering with HDBSCAN...")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=None, core_dist_n_jobs=-1)
        cluster_labels = clusterer.fit_predict(coords_2d)

        # Prepare data for bulk insertion
        vis_data_to_commit = [
            (
                all_shas[i],
                float(coords_2d[i, 0]),
                float(coords_2d[i, 1]),
                int(cluster_labels[i]),
            )
            for i in range(len(all_shas))
        ]

        logger.info(f"Saving visualization data for {len(vis_data_to_commit)} points...")
        with self.conn:
            # Clear old data and insert the fresh calculations
            self.conn.execute("DELETE FROM visualization")
            self.conn.executemany(
                "INSERT INTO visualization (sha256, coord_x, coord_y, cluster_id) VALUES (?, ?, ?, ?)",
                vis_data_to_commit,
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

    # --- NEW HELPER METHOD for visualization ---
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
