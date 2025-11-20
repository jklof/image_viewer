import hashlib
import logging
import sqlite3
import threading
import queue
import concurrent.futures
import os
import io  # Required for byte manipulation
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ml_core import ImageEmbedder

logger = logging.getLogger(__name__)

# --- Configuration ---
HASHING_WORKER_COUNT = max(1, os.cpu_count() // 2)
EMBEDDING_BATCH_SIZE = 64
PIPELINE_QUEUE_SIZE = 256
SQLITE_VARIABLE_LIMIT = 900
MAX_IMAGE_PIXELS = 80 * 1000 * 1000
PREPROCESS_TARGET_SIZE = (256, 256)  # Size for CLIP pre-processing


def _targeted_hashing_and_resize_worker(filepath: Path, mtime: float) -> tuple[Path, str, float, bytes | None] | None:
    """
    Worker to perform the hash AND pre-resize the image to bytes.
    Returns: (filepath, sha256, mtime, resized_image_bytes)
    """
    try:
        # 1. Hashing
        sha256_hash = hashlib.sha256()
        buffer_size = 1024 * 1024
        try:
            with open(filepath, "rb") as f:
                while chunk := f.read(buffer_size):
                    sha256_hash.update(chunk)
        except (IOError, OSError):
            return None

        file_hash = sha256_hash.hexdigest()

        # 2. Pre-processing (Resize in parallel process to save main thread CPU)
        img_bytes = None
        try:
            # We only attempt to open if it looks like an image
            if filepath.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                img = Image.open(filepath)

                # Fast reject massive images
                if (img.width * img.height) <= MAX_IMAGE_PIXELS:
                    img = img.convert("RGB")
                    # Resize now. This is CPU intensive, perfect for the worker pool.
                    img.thumbnail(PREPROCESS_TARGET_SIZE, Image.Resampling.LANCZOS)

                    # Save to bytes to pass over process boundary
                    out_buffer = io.BytesIO()
                    img.save(out_buffer, format="JPEG", quality=90)
                    img_bytes = out_buffer.getvalue()
        except Exception:
            # If image load/resize fails, we still return the hash so DB tracks the file,
            # but img_bytes is None so we skip embedding generation.
            pass

        return filepath, file_hash, mtime, img_bytes

    except Exception:
        return None


class EmbeddingConsumerThread(threading.Thread):
    """
    Consumes (path, hash, mtime, bytes) tuples.
    Writes filepaths to DB. Generates embeddings for new hashes using the pre-resized bytes.
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
        logger.info("EmbeddingConsumerThread started.")
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA foreign_keys = ON;")

            batch = []
            while not self.cancel_flag.is_set():
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
                except Exception:
                    if self.cancel_flag.is_set():
                        break
                    raise

        except Exception as e:
            logger.error(f"Unhandled exception in EmbeddingConsumerThread: {e}", exc_info=True)
        finally:
            if self.cancel_flag.is_set():
                try:
                    while not self.work_queue.empty():
                        self.work_queue.get_nowait()
                except Exception:
                    pass
            if self.conn:
                try:
                    self.conn.close()
                except Exception:
                    pass
            logger.info("EmbeddingConsumerThread finished.")

    def _process_batch(self, batch: list[tuple[Path, str, float, bytes | None]]):
        if not batch or self.cancel_flag.is_set():
            return

        try:
            filepaths, shas, mtimes, img_byte_list = zip(*batch)
            shas_to_check = list(set(shas))
            existing_shas_in_db = set()

            # Check existence
            for i in range(0, len(shas_to_check), SQLITE_VARIABLE_LIMIT):
                chunk = shas_to_check[i : i + SQLITE_VARIABLE_LIMIT]
                placeholders = ", ".join("?" for _ in chunk)
                cursor = self.conn.execute(f"SELECT sha256 FROM embeddings WHERE sha256 IN ({placeholders})", chunk)
                existing_shas_in_db.update(row[0] for row in cursor.fetchall())

            new_embeddings_to_commit = []
            valid_new_shas = set()

            # Prepare images for embedding
            pil_images = []
            temp_valid_shas = []

            for sha, img_bytes in zip(shas, img_byte_list):
                # Embed if: Not in DB, Image data exists, Not already queued
                if sha not in existing_shas_in_db and img_bytes is not None and sha not in temp_valid_shas:
                    try:
                        # Fast load from memory
                        img = Image.open(io.BytesIO(img_bytes))
                        pil_images.append(img)
                        temp_valid_shas.append(sha)
                    except Exception:
                        pass

            if pil_images:
                try:
                    embeddings = self.embedder.embed_batch(pil_images)
                    new_embeddings_to_commit = [(sha, emb.tobytes()) for sha, emb in zip(temp_valid_shas, embeddings)]
                    valid_new_shas.update(temp_valid_shas)
                except Exception as e:
                    logger.error(f"Failed to embed batch: {e}")

            with self.conn:
                if new_embeddings_to_commit:
                    self.conn.executemany(
                        "INSERT OR IGNORE INTO embeddings (sha256, embedding) VALUES (?, ?)", new_embeddings_to_commit
                    )

                safe_shas = existing_shas_in_db.union(valid_new_shas)
                filepath_data = []

                for fp, sha, mt in zip(filepaths, shas, mtimes):
                    if sha in safe_shas:
                        filepath_data.append((str(fp), sha, mt))

                if filepath_data:
                    self.conn.executemany(
                        "INSERT OR REPLACE INTO filepaths (filepath, sha256, mtime) VALUES (?, ?, ?)", filepath_data
                    )

        except Exception as e:
            logger.error(f"Error in batch processing: {e}")


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
            # --- OPTIMIZATION: Index for Orphan Cleanup and Foreign Key checks ---
            conn.execute("CREATE INDEX IF NOT EXISTS idx_filepaths_sha256 ON filepaths(sha256)")
            # ---------------------------------------------------------------------
            conn.execute(
                "CREATE TABLE IF NOT EXISTS visualization (sha256 TEXT PRIMARY KEY, coord_x REAL NOT NULL, coord_y REAL NOT NULL, cluster_id INTEGER NOT NULL, FOREIGN KEY (sha256) REFERENCES embeddings (sha256) ON DELETE CASCADE)"
            )
            conn.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    def _get_metadata(self, key: str) -> str | None:
        with self._get_db_connection() as conn:
            row = conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
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
            executor.shutdown(wait=True, cancel_futures=True)

    def _reconcile_model_id(self):
        db_model_id, config_model_id = self._get_metadata("model_id"), self.embedder.model_id
        if db_model_id != config_model_id:
            if db_model_id is not None:
                with self._get_db_connection() as conn:
                    conn.execute("DELETE FROM embeddings")
                    conn.execute("DELETE FROM visualization")
            self._set_metadata("model_id", config_model_id)

    def _get_tracked_files_from_db(self) -> dict[str, float]:
        with self._get_db_connection() as conn:
            return {r[0]: r[1] for r in conn.execute("SELECT filepath, mtime FROM filepaths").fetchall()}

    def _discover_files_on_disk(self, configured_dirs, status_callback, check_cancelled) -> dict[str, float]:
        disk_files = {}
        for directory in configured_dirs:
            check_cancelled()
            path_obj = Path(directory)
            if not path_obj.is_dir():
                continue
            if status_callback:
                status_callback(f"Scanning {directory}...")
            for p in path_obj.rglob("*"):
                if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                    try:
                        disk_files[str(p.absolute())] = p.stat().st_mtime
                    except OSError:
                        continue
        return disk_files

    def _calculate_file_changes(self, db_files, disk_files):
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

        count = len(removed_paths)
        if status_callback:
            status_callback(f"Removing {count} old files...")

        # --- OPTIMIZATION: Temporary Table for Bulk Deletion ---
        with self._get_db_connection() as conn:
            conn.execute("CREATE TEMPORARY TABLE IF NOT EXISTS temp_delete_paths (filepath TEXT PRIMARY KEY)")
            params = [(p,) for p in removed_paths]
            conn.executemany("INSERT OR IGNORE INTO temp_delete_paths (filepath) VALUES (?)", params)
            conn.execute("DELETE FROM filepaths WHERE filepath IN (SELECT filepath FROM temp_delete_paths)")
            conn.execute("DROP TABLE temp_delete_paths")
        # -------------------------------------------------------

    def _run_hashing_and_embedding_pipeline(
        self, paths_to_hash, disk_files, executor, progress_callback, status_callback, check_cancelled
    ):
        total_jobs = len(paths_to_hash)
        if status_callback:
            status_callback(f"Hashing {total_jobs} files...")
        work_queue = queue.Queue(maxsize=PIPELINE_QUEUE_SIZE)
        consumer = EmbeddingConsumerThread(self.db_path, work_queue, self.embedder, self._cancel_flag)
        consumer.start()

        job_args = [(Path(p), disk_files[p]) for p in paths_to_hash]
        completed = 0
        try:
            # Use the new Worker that returns 4 items
            results = executor.map(_targeted_hashing_and_resize_worker, *(zip(*job_args)))
            for result in results:
                check_cancelled()
                if result:
                    while not self._cancel_flag.is_set():
                        try:
                            work_queue.put(result, timeout=1.0)
                            break
                        except queue.Full:
                            continue
                completed += 1
                if progress_callback:
                    progress_callback("hashing", completed, total_jobs)
        finally:
            work_queue.put(None)
            consumer.join()
            check_cancelled()

    def _finalize_sync(self, status_callback, check_cancelled):
        self._cleanup_orphaned_embeddings()
        # We no longer automatically update visualization here.
        # It will be calculated Just-In-Time when the user requests it.
        self._load_embeddings_into_memory()

    def _load_embeddings_into_memory(self):
        self._sha_to_path_map_cache = None
        with self._get_db_connection() as conn:
            rows = conn.execute("SELECT sha256, embedding FROM embeddings").fetchall()
        if not rows:
            self._shas_in_order, self._embedding_matrix = [], None
            return
        self._shas_in_order, blobs = zip(*rows)
        self._embedding_matrix = np.vstack([self._reconstruct_embedding(b) for b in blobs])

    def _cleanup_orphaned_embeddings(self):
        # --- OPTIMIZATION: Use NOT EXISTS and cleanup visualization too ---
        with self._get_db_connection() as conn:
            conn.execute(
                "DELETE FROM embeddings WHERE NOT EXISTS (SELECT 1 FROM filepaths WHERE filepaths.sha256 = embeddings.sha256)"
            )
            # Also remove visualization entries that no longer have a corresponding embedding
            conn.execute("DELETE FROM visualization WHERE sha256 NOT IN (SELECT sha256 FROM embeddings)")
        # ------------------------------------------------------------------

    def _reconstruct_embedding(self, blob):
        return np.frombuffer(blob, dtype=self.embedder.embedding_dtype).reshape(self.embedder.embedding_shape)

    def ensure_visualization_data(self, status_callback=None, check_cancelled_callback=None):
        """
        Calculates visualization data using Incremental PCA and Landmark UMAP
        to optimize for speed and memory usage on large datasets.
        """
        try:
            # 1. Check if update is needed (Same as before)
            with self._get_db_connection() as conn:
                emb_count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
                vis_count = conn.execute("SELECT COUNT(*) FROM visualization").fetchone()[0]

            # Simple check: if counts match, we assume we are good.
            # (Robustness improvement: check specific SHAs if you prefer strictness)
            if emb_count > 0 and emb_count == vis_count:
                return

            if check_cancelled_callback:
                check_cancelled_callback()

            if status_callback:
                status_callback("Import umap/hdbscan...")

            # Lazy imports
            import umap
            import hdbscan
            from sklearn.decomposition import IncrementalPCA
            import numpy as np

            if check_cancelled_callback:
                check_cancelled_callback()

            if status_callback:
                status_callback("Optimizing data (Incremental PCA)...")

            # --- STEP 1: INCREMENTAL PCA (768 dims -> 50 dims) ---
            # This reduces memory usage by ~15x and speeds up UMAP distance calcs.

            n_components = 50
            batch_size = 2048
            ipca = IncrementalPCA(n_components=n_components)

            # Pass 1: Train PCA on the data stream without loading all to RAM
            conn = self._get_db_connection()
            cursor = conn.execute("SELECT embedding FROM embeddings")

            while True:
                if check_cancelled_callback:
                    check_cancelled_callback()

                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break

                # Convert bytes to numpy batch
                batch = np.vstack([self._reconstruct_embedding(r[0]).flatten() for r in rows])
                ipca.partial_fit(batch)

            cursor.close()

            # --- STEP 2: TRANSFORM & LOAD COMPRESSED DATA ---
            # Now we load the data, but it's only 50 floats per image instead of 768.
            # 60k images * 50 dims * 4 bytes ~= 12 MB RAM (Tiny!)

            if status_callback:
                status_callback("Projecting data...")

            compressed_data = []
            all_shas = []

            cursor = conn.execute("SELECT sha256, embedding FROM embeddings")
            while True:
                if check_cancelled_callback:
                    check_cancelled_callback()

                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break

                shas = [r[0] for r in rows]
                batch = np.vstack([self._reconstruct_embedding(r[1]).flatten() for r in rows])

                # Transform to 50 dims immediately
                batch_reduced = ipca.transform(batch)

                all_shas.extend(shas)
                compressed_data.append(batch_reduced)

            cursor.close()
            conn.close()

            # Consolidate into one array (Approx 12MB for 60k images)
            X_reduced = np.vstack(compressed_data)

            # --- STEP 3: UMAP (Standard Parallel Execution) ---

            if check_cancelled_callback:
                check_cancelled_callback()

            if status_callback:
                # With 50 dims, this is fast. With n_epochs=200, it's very fast.
                status_callback(f"Calculating layout for {len(X_reduced)} items...")

            reducer = umap.UMAP(
                n_neighbors=15,
                min_dist=0.1,
                n_components=2,
                n_jobs=-1,  # Use ALL cores
                n_epochs=200,  # <--- SPEED TWEAK: Reduced from default (usually 500)
                low_memory=False,  # Trade a tiny bit of RAM for speed
                # random_state=42 # REMOVED to allow parallelism
            )

            # Just fit everyone at once.
            # For <100k items with 50 dims, this is usually faster than transform().
            coords_2d = reducer.fit_transform(X_reduced)

            if check_cancelled_callback:
                check_cancelled_callback()

            # --- STEP 4: CLUSTERING ---
            if status_callback:
                status_callback("Clustering...")

            # We cluster on the 2D output. It's fast enough.
            cluster_labels = hdbscan.HDBSCAN(min_cluster_size=10, core_dist_n_jobs=-1).fit_predict(coords_2d)

            # --- STEP 5: SAVE ---
            if status_callback:
                status_callback("Saving visualization data...")

            vis_data = [(sh, float(c[0]), float(c[1]), int(l)) for sh, c, l in zip(all_shas, coords_2d, cluster_labels)]

            with self._get_db_connection() as conn:
                conn.execute("DELETE FROM visualization")
                conn.executemany(
                    "INSERT INTO visualization (sha256, coord_x, coord_y, cluster_id) VALUES (?, ?, ?, ?)", vis_data
                )

        except Exception as e:
            logger.error(f"Visualization update failed: {e}")
            # Optional: re-raise if you want the UI to show the error
            # raise e

    def _build_sha_to_path_map(self):
        self._sha_to_path_map_cache = defaultdict(list)
        with self._get_db_connection() as conn:
            for sha, path in conn.execute("SELECT sha256, filepath FROM filepaths").fetchall():
                self._sha_to_path_map_cache[sha].append(path)

    def _perform_search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[float, str]]:
        # --- OPTIMIZATION: Efficient Numpy Search ---
        if self._embedding_matrix is None or len(self._embedding_matrix) == 0:
            return []

        query_embedding = query_embedding.astype(self.embedder.embedding_dtype).reshape(1, -1)
        similarities = np.dot(self._embedding_matrix, query_embedding.T).flatten()

        if self._sha_to_path_map_cache is None:
            self._build_sha_to_path_map()

        if top_k != -1 and top_k < len(similarities):
            # Partial sort is faster than full sort
            indices = np.argpartition(similarities, -top_k)[-top_k:]
            indices = indices[np.argsort(similarities[indices])[::-1]]
        else:
            indices = np.argsort(similarities)[::-1]

        results = []
        for i in indices:
            sha = self._shas_in_order[i]
            paths = self._sha_to_path_map_cache.get(sha)
            if paths:
                results.append((float(similarities[i]), paths[0]))
        return results
        # --------------------------------------------

    def search_similar_images(self, image_path, top_k=-1):
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            return []
        return self._perform_search(self.embedder.embed_image(image), top_k)

    def search_by_text(self, text, top_k=-1):
        return self._perform_search(self.embedder.embed_text(text), top_k)

    def cancel_sync(self):
        self._cancel_flag.set()

    def get_all_unique_filepaths(self):
        with self._get_db_connection() as conn:
            return [r[0] for r in conn.execute("SELECT MIN(filepath) FROM filepaths GROUP BY sha256").fetchall()]

    def get_all_filepaths_with_mtime(self):
        with self._get_db_connection() as conn:
            return conn.execute("SELECT filepath, mtime FROM filepaths").fetchall()

    def get_all_embeddings_with_shas(self):
        with self._get_db_connection() as conn:
            return [
                {"sha256": sha, "embedding": self._reconstruct_embedding(eb).flatten()}
                for sha, eb in conn.execute("SELECT sha256, embedding FROM embeddings").fetchall()
            ]

    def get_visualization_data(self):
        with self._get_db_connection() as conn:
            return conn.execute(
                "SELECT v.coord_x, v.coord_y, v.cluster_id, f.filepath FROM visualization v JOIN (SELECT sha256, MIN(filepath) as filepath FROM filepaths GROUP BY sha256) f ON v.sha256 = f.sha256"
            ).fetchall()
