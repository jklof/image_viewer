import hashlib
import logging
import sqlite3
import threading
import queue
import concurrent.futures
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

# Use a string for the type hint to avoid runtime import
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ml_core import ImageEmbedder

logger = logging.getLogger(__name__)

# --- Configuration for the new parallel pipeline ---
HASHING_WORKER_COUNT = max(1, os.cpu_count() // 2)
EMBEDDING_BATCH_SIZE = 64
PIPELINE_QUEUE_SIZE = 256
SQLITE_VARIABLE_LIMIT = 900  # A safe limit for IN clauses


def _targeted_hashing_worker(filepath: Path, mtime: float) -> tuple[Path, str, float] | None:
    """
    Worker to perform the hash for a targeted file.
    """
    try:
        sha256_hash = hashlib.sha256()
        buffer_size = 1024 * 1024  # 1MB buffer
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(buffer_size), b""):
                sha256_hash.update(byte_block)
        return filepath, sha256_hash.hexdigest(), mtime
    except (IOError, FileNotFoundError, OSError) as e:
        logger.warning(f"Hashing failed for {filepath}: {e}")
        return None


class EmbeddingConsumerThread(threading.Thread):
    """
    A dedicated consumer thread that pulls file data from a queue, generates
    embeddings for new images, and writes all results to the database.
    """

    def __init__(self, db_path: str, work_queue: queue.Queue, embedder: "ImageEmbedder", cancel_flag: threading.Event):
        super().__init__()
        self.db_path = db_path
        self.work_queue = work_queue
        self.embedder = embedder
        self.cancel_flag = cancel_flag
        self.conn = None
        self.setName("EmbeddingConsumer")

    def run(self):
        """The main loop for the consumer thread."""
        logger.info("EmbeddingConsumerThread started.")
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA foreign_keys = ON;")

        batch = []
        try:
            while not self.cancel_flag.is_set():
                try:
                    item = self.work_queue.get(timeout=0.5)
                    if item is None:  # Graceful shutdown signal
                        if batch:
                            logger.info(f"Processing final batch of {len(batch)} items before shutdown.")
                            self._process_batch(batch)
                        break  # Exit the loop

                    batch.append(item)
                    if len(batch) >= EMBEDDING_BATCH_SIZE:
                        self._process_batch(batch)
                        batch = []
                except queue.Empty:
                    # Timeout is not an error. If we have a partial batch, process it.
                    if batch:
                        self._process_batch(batch)
                        batch = []

        except Exception as e:
            logger.error(f"Unhandled exception in EmbeddingConsumerThread: {e}", exc_info=True)
        finally:
            # This block runs on BOTH graceful shutdown and cancellation.
            if self.cancel_flag.is_set():
                # If we were cancelled, drain the queue of any pending data.
                # This is critical to unblock the producer thread which might be
                # waiting on a full queue to put the `None` sentinel.
                logger.info("Cancellation detected. Draining work queue to unblock producer...")
                while not self.work_queue.empty():
                    try:
                        self.work_queue.get_nowait()
                    except queue.Empty:
                        break  # Safeguard against race condition

            if self.conn:
                self.conn.close()
                logger.info("Database connection closed by EmbeddingConsumerThread.")
            logger.info("EmbeddingConsumerThread finished.")

    def _process_batch(self, batch: list[tuple[Path, str, float]]):
        if not batch:
            return
        # A safeguard check, although the main loop is the primary gatekeeper.
        if self.cancel_flag.is_set():
            return

        logger.debug(f"Processing batch of {len(batch)} hashed files.")
        filepaths, shas, mtimes = zip(*batch)
        shas_to_check = list(set(shas))
        existing_shas_in_db = set()

        # Check which SHAs already exist
        for i in range(0, len(shas_to_check), SQLITE_VARIABLE_LIMIT):
            chunk = shas_to_check[i : i + SQLITE_VARIABLE_LIMIT]
            placeholders = ", ".join("?" for _ in chunk)
            cursor = self.conn.execute(f"SELECT sha256 FROM embeddings WHERE sha256 IN ({placeholders})", chunk)
            existing_shas_in_db.update(row[0] for row in cursor.fetchall())

        new_image_data = []
        sha_to_filepath_map = defaultdict(list)
        for fp, sha in zip(filepaths, shas):
            sha_to_filepath_map[sha].append(fp)

        for sha in shas_to_check:
            if sha not in existing_shas_in_db:
                # We pick one filepath to represent this SHA for loading purposes
                new_image_data.append((sha, sorted(sha_to_filepath_map[sha])[0]))

        new_embeddings_to_commit = []
        # Track SHAs that we successfully generate embeddings for in this run
        valid_new_shas = set()

        if new_image_data:
            logger.info(f"Found {len(new_image_data)} new images in batch to embed.")
            shas_to_embed, image_paths_to_load = zip(*new_image_data)
            pil_images, temp_valid_shas = [], []

            for sha, path in zip(shas_to_embed, image_paths_to_load):
                try:
                    pil_images.append(Image.open(path).convert("RGB"))
                    temp_valid_shas.append(sha)
                except Exception as e:
                    logger.warning(f"Could not load image {path}. Skipping. Error: {e}")
                    # Note: We do NOT add this SHA to temp_valid_shas

            if pil_images:
                try:
                    embeddings = self.embedder.embed_batch(pil_images)
                    new_embeddings_to_commit = [(sha, emb.tobytes()) for sha, emb in zip(temp_valid_shas, embeddings)]
                    valid_new_shas.update(temp_valid_shas)
                except Exception as e:
                    logger.error(f"Failed to embed batch. Error: {e}", exc_info=True)
                    # If the whole batch fails embedding, valid_new_shas remains empty

        with self.conn:
            if new_embeddings_to_commit:
                self.conn.executemany(
                    "INSERT OR IGNORE INTO embeddings (sha256, embedding) VALUES (?, ?)", new_embeddings_to_commit
                )

            # Calculate the set of SHAs that are legally present in the embeddings table now.
            # 1. SHAs that were there before (existing_shas_in_db)
            # 2. SHAs we just added successfully (valid_new_shas)
            safe_shas = existing_shas_in_db.union(valid_new_shas)

            filepath_data = []
            for fp, sha, mt in zip(filepaths, shas, mtimes):
                if sha in safe_shas:
                    filepath_data.append((str(fp), sha, mt))
                else:
                    # This happens if the image was 'new' but failed to load/embed.
                    # We skip inserting it into filepaths to prevent FK violation.
                    pass

            if filepath_data:
                self.conn.executemany(
                    "INSERT OR REPLACE INTO filepaths (filepath, sha256, mtime) VALUES (?, ?, ?)", filepath_data
                )

        logger.debug(f"Batch of {len(batch)} items fully processed and committed.")


class ImageDatabase:
    class InterruptedError(Exception):
        pass

    class ModelMismatchError(Exception):
        pass

    def __init__(self, db_path="images.db", embedder: "ImageEmbedder" = None):
        if embedder is None:
            raise TypeError("ImageEmbedder is required.")
        self.db_path, self.embedder = db_path, embedder
        self._create_tables()
        self._verify_model_compatibility()
        self._shas_in_order, self._embedding_matrix = [], None
        self._sha_to_path_map_cache = None
        self._cancel_flag = threading.Event()
        self._load_embeddings_into_memory()

    def _get_db_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _create_tables(self):
        with self._get_db_connection() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS embeddings (sha256 TEXT PRIMARY KEY, embedding BLOB NOT NULL)")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS filepaths (filepath TEXT PRIMARY KEY, sha256 TEXT NOT NULL, mtime REAL NOT NULL, FOREIGN KEY (sha256) REFERENCES embeddings (sha256) ON DELETE CASCADE)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS visualization (sha256 TEXT PRIMARY KEY, coord_x REAL NOT NULL, coord_y REAL NOT NULL, cluster_id INTEGER NOT NULL, FOREIGN KEY (sha256) REFERENCES embeddings (sha256) ON DELETE CASCADE)"
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
            raise self.ModelMismatchError(f"DB model is '{db_model_id}', but app is using '{self.embedder.model_id}'.")

    def reconcile_database(self, configured_dirs: list[str], progress_callback=None, status_callback=None):
        self._cancel_flag.clear()

        def _check_cancelled():
            if self._cancel_flag.is_set():
                raise self.InterruptedError("Sync cancelled.")

        executor = concurrent.futures.ProcessPoolExecutor(max_workers=HASHING_WORKER_COUNT)
        try:
            logger.info("Starting database synchronization...")
            self._reconcile_model_id()
            _check_cancelled()
            if status_callback:
                status_callback("Discovering files...")
            db_files = self._get_tracked_files_from_db()
            disk_files = self._discover_files_on_disk(configured_dirs, status_callback, _check_cancelled)
            _check_cancelled()
            if status_callback:
                status_callback(f"Analyzing {len(disk_files)} files...")
            changes = self._calculate_file_changes(db_files, disk_files)
            logger.info(f"Found {len(changes['to_hash'])} to hash, {len(changes['removed'])} to remove.")
            self._remove_deleted_files_from_db(changes["removed"], status_callback)
            if not changes["to_hash"]:
                logger.info("No new or modified files to process.")
                self._finalize_sync(status_callback, _check_cancelled)
                return
            self._run_hashing_and_embedding_pipeline(
                changes["to_hash"], disk_files, executor, progress_callback, status_callback, _check_cancelled
            )
            self._finalize_sync(status_callback, _check_cancelled)
        except self.InterruptedError as e:
            logger.warning(f"Synchronization cancelled by user: {e}")
            raise
        finally:
            logger.info("Shutting down process pool executor.")
            executor.shutdown(wait=True, cancel_futures=True)

    def _reconcile_model_id(self):
        db_model_id, config_model_id = self._get_metadata("model_id"), self.embedder.model_id
        if db_model_id != config_model_id:
            if db_model_id is not None:
                logger.warning(f"CONFIG-DB MODEL MISMATCH. Deleting all embeddings.")
                with self._get_db_connection() as conn:
                    conn.execute("DELETE FROM embeddings")
                    conn.execute("DELETE FROM visualization")
            self._set_metadata("model_id", config_model_id)

    def _get_tracked_files_from_db(self) -> dict[str, float]:
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT filepath, mtime FROM filepaths")
            return {row[0]: row[1] for row in cursor.fetchall()}

    def _discover_files_on_disk(
        self, configured_dirs: list[str], status_callback: callable, check_cancelled: callable
    ) -> dict[str, float]:
        disk_files: dict[str, float] = {}
        for directory in configured_dirs:
            check_cancelled()
            path_obj = Path(directory)
            if not path_obj.is_dir():
                continue
            if status_callback:
                status_callback(f"Scanning {directory}...")
            for p in path_obj.rglob("*"):
                if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    try:
                        disk_files[str(p.absolute())] = p.stat().st_mtime
                    except FileNotFoundError:
                        continue
        return disk_files

    def _calculate_file_changes(self, db_files: dict[str, float], disk_files: dict[str, float]) -> dict[str, set[str]]:
        db_paths, disk_paths = set(db_files.keys()), set(disk_files.keys())
        return {
            "removed": db_paths - disk_paths,
            "to_hash": (disk_paths - db_paths).union(
                {p for p in disk_paths.intersection(db_paths) if disk_files[p] > db_files[p]}
            ),
        }

    def _remove_deleted_files_from_db(self, removed_paths: set[str], status_callback: callable):
        if not removed_paths:
            return
        if status_callback:
            status_callback(f"Removing {len(removed_paths)} old files...")
        with self._get_db_connection() as conn:
            removed_list = list(removed_paths)
            for i in range(0, len(removed_list), SQLITE_VARIABLE_LIMIT):
                chunk = removed_list[i : i + SQLITE_VARIABLE_LIMIT]
                conn.execute(f"DELETE FROM filepaths WHERE filepath IN ({','.join('?'*len(chunk))})", chunk)

    def _run_hashing_and_embedding_pipeline(
        self,
        paths_to_hash: set[str],
        disk_files: dict[str, float],
        executor,
        progress_callback,
        status_callback,
        check_cancelled,
    ):
        total_jobs = len(paths_to_hash)
        if status_callback:
            status_callback(f"Hashing {total_jobs} files...")
        work_queue = queue.Queue(maxsize=PIPELINE_QUEUE_SIZE)
        consumer = EmbeddingConsumerThread(self.db_path, work_queue, self.embedder, self._cancel_flag)
        consumer.start()
        job_args = [(Path(p), disk_files[p]) for p in paths_to_hash]
        completed_jobs = 0
        try:
            results_iterator = executor.map(_targeted_hashing_worker, *(zip(*job_args)))
            for result in results_iterator:
                check_cancelled()
                if result:
                    # This loop also prevents blocking forever, as it will break
                    # if the cancel_flag is set.
                    while not self._cancel_flag.is_set():
                        try:
                            work_queue.put(result, timeout=1.0)
                            break
                        except queue.Full:
                            continue
                completed_jobs += 1
                if progress_callback:
                    progress_callback("hashing", completed_jobs, total_jobs)
        finally:
            logger.info("Hashing jobs finished or cancelled. Signaling consumer.")
            # This is now safe. If the queue is full and we were cancelled, the
            # consumer will drain it, making space. If we finished normally,
            # the consumer is waiting for this sentinel.
            work_queue.put(None)

            consumer.join()
            logger.info("Embedding consumer has finished.")
            check_cancelled()  # Final check before proceeding

    def _finalize_sync(self, status_callback, check_cancelled):
        logger.info("Finalizing sync...")
        self._cleanup_orphaned_embeddings()
        if status_callback:
            status_callback("Updating visualization...")
        self._update_visualization_data(status_callback=status_callback, check_cancelled_callback=check_cancelled)
        self._load_embeddings_into_memory()

    def _load_embeddings_into_memory(self):
        self._sha_to_path_map_cache = None
        logger.info("Loading embeddings into memory...")
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT sha256, embedding FROM embeddings")
            rows = cursor.fetchall()
        if not rows:
            self._shas_in_order, self._embedding_matrix = [], None
            return
        self._shas_in_order, embeddings_blobs = zip(*rows)
        self._embedding_matrix = np.vstack([self._reconstruct_embedding(b) for b in embeddings_blobs])
        logger.info(f"Loaded {len(self._shas_in_order)} embeddings into memory.")

    def _cleanup_orphaned_embeddings(self):
        with self._get_db_connection() as conn:
            res = conn.execute("DELETE FROM embeddings WHERE sha256 NOT IN (SELECT DISTINCT sha256 FROM filepaths)")
            if res.rowcount > 0:
                logger.info(f"Removed {res.rowcount} orphaned embeddings.")

    def _reconstruct_embedding(self, blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=self.embedder.embedding_dtype).reshape(self.embedder.embedding_shape)

    def _update_visualization_data(self, status_callback=None, check_cancelled_callback=None):
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT sha256 FROM embeddings")
            embedding_shas = {row[0] for row in cursor.fetchall()}
            cursor.execute("SELECT sha256 FROM visualization")
            visualization_shas = {row[0] for row in cursor.fetchall()}
        if embedding_shas == visualization_shas:
            return
        logger.info("Recalculating all visualization data...")
        embedding_data = self.get_all_embeddings_with_shas()
        if not embedding_data:
            with self._get_db_connection() as conn:
                conn.execute("DELETE FROM visualization")
            return
        all_embeddings = np.array([item["embedding"] for item in embedding_data])
        all_shas = [item["sha256"] for item in embedding_data]
        if status_callback:
            status_callback("Importing visualization libraries...")
        import umap
        import hdbscan

        try:
            if check_cancelled_callback:
                check_cancelled_callback()
            if status_callback:
                status_callback("Reducing dimensions (UMAP)...")
            coords_2d = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, n_jobs=-1).fit_transform(all_embeddings)
            if check_cancelled_callback:
                check_cancelled_callback()
            if status_callback:
                status_callback("Clustering (HDBSCAN)...")
            cluster_labels = hdbscan.HDBSCAN(min_cluster_size=5, core_dist_n_jobs=-1).fit_predict(coords_2d)
            if check_cancelled_callback:
                check_cancelled_callback()
        except Exception as e:
            logger.error(f"Visualization calculation failed. Skipping. Error: {e}", exc_info=True)
            return
        vis_data = [(sh, float(c[0]), float(c[1]), int(l)) for sh, c, l in zip(all_shas, coords_2d, cluster_labels)]
        with self._get_db_connection() as conn:
            conn.execute("DELETE FROM visualization")
            conn.executemany(
                "INSERT INTO visualization (sha256, coord_x, coord_y, cluster_id) VALUES (?, ?, ?, ?)", vis_data
            )
        logger.info("Visualization data successfully updated.")

    def _build_sha_to_path_map(self):
        sha_to_path_map = defaultdict(list)
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT sha256, filepath FROM filepaths")
            for sha, path in cursor.fetchall():
                sha_to_path_map[sha].append(path)
        self._sha_to_path_map_cache = sha_to_path_map

    def _perform_search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[float, str]]:
        if self._embedding_matrix is None or len(self._embedding_matrix) == 0:
            return []
        if self._sha_to_path_map_cache is None:
            self._build_sha_to_path_map()
        similarities = np.dot(query_embedding, self._embedding_matrix.T).flatten()
        if top_k != -1 and top_k < len(similarities):
            indices = np.argpartition(similarities, -top_k)[-top_k:]
            results = sorted(
                [(similarities[i], self._shas_in_order[i]) for i in indices], key=lambda x: x[0], reverse=True
            )
        else:
            indices = np.argsort(similarities)[::-1]
            results = [(similarities[i], self._shas_in_order[i]) for i in indices]
        final = [
            (float(s), self._sha_to_path_map_cache[sha][0]) for s, sha in results if sha in self._sha_to_path_map_cache
        ]
        return final if top_k == -1 else final[:top_k]

    def search_similar_images(self, image_path: str, top_k: int = -1):
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            return []
        query = self.embedder.embed_image(image)
        if top_k == -1:
            return self._perform_search(query, -1)
        results = self._perform_search(query, top_k + 1)
        other = [(s, p) for s, p in results if Path(p).resolve() != Path(image_path).resolve()]
        return [(1.0, image_path)] + other[: top_k - 1]

    def search_by_text(self, text_query: str, top_k: int = -1):
        return self._perform_search(self.embedder.embed_text(text_query), top_k)

    def cancel_sync(self):
        self._cancel_flag.set()

    def get_all_unique_filepaths(self) -> list[str]:
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MIN(filepath) FROM filepaths GROUP BY sha256")
            return [row[0] for row in cursor.fetchall()]

    def get_all_filepaths_with_mtime(self) -> list[tuple[str, float]]:
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT filepath, mtime FROM filepaths")
            return cursor.fetchall()

    def get_all_embeddings_with_shas(self) -> list[dict]:
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT sha256, embedding FROM embeddings")
            return [
                {"sha256": sha, "embedding": self._reconstruct_embedding(eb).flatten()} for sha, eb in cursor.fetchall()
            ]

    def get_visualization_data(self) -> list[tuple[float, float, int, str]]:
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT v.coord_x, v.coord_y, v.cluster_id, f.filepath FROM visualization v JOIN (SELECT sha256, MIN(filepath) as filepath FROM filepaths GROUP BY sha256) f ON v.sha256 = f.sha256"
            )
            return cursor.fetchall()
