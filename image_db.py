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
from PIL import Image, ImageOps

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ml_core import ImageEmbedder

logger = logging.getLogger(__name__)

# --- Configuration ---
# Limit workers to prevent severe disk thrashing on mechanical HDDs
HASHING_WORKER_COUNT = min(4, os.cpu_count() or 4)
EMBEDDING_BATCH_SIZE = 64
PIPELINE_QUEUE_SIZE = 256
SQLITE_VARIABLE_LIMIT = 900  # Conservative limit; SQLite default max is 999
MAX_IMAGE_PIXELS = 80 * 1000 * 1000
PREPROCESS_TARGET_SIZE = (256, 256)  # Size for CLIP pre-processing
INVALID_FILE_SENTINEL = "__INVALID__"  # Sentinel SHA for files that fail hashing/decoding

_WORKER_KNOWN_SHAS = set()


def _init_worker(known_shas: set):
    """Initializes the global state for the ProcessPoolExecutor workers."""
    global _WORKER_KNOWN_SHAS
    _WORKER_KNOWN_SHAS = known_shas


def _targeted_hashing_and_resize_worker(filepath: Path, mtime: float) -> tuple[Path, str, float, bytes | None]:
    """
    Worker to perform the hash AND pre-resize the image to bytes.
    Optimized to read files into memory once to avoid double I/O penalties on slow drives,
    and skips image decoding entirely if the hash is already known.
    """
    try:
        ext = filepath.suffix.lower()

        # 1. Hashing & Buffering
        sha256_hash = hashlib.sha256()
        file_buffer = None

        try:
            file_size = filepath.stat().st_size
            # For standard images under 50MB, read entirely into memory ONCE to avoid double reads.
            if ext in (".jpg", ".jpeg", ".png", ".webp") and file_size < 50 * 1024 * 1024:
                with open(filepath, "rb") as f:
                    file_buffer = f.read()
                sha256_hash.update(file_buffer)
            else:
                # For videos or massive files, stream in 8MB chunks for better sequential HDD throughput
                buffer_size = 8 * 1024 * 1024
                with open(filepath, "rb") as f:
                    while chunk := f.read(buffer_size):
                        sha256_hash.update(chunk)
        except (IOError, OSError) as e:
            logger.warning(f"Could not read {filepath}: {e}")
            return filepath, INVALID_FILE_SENTINEL, mtime, None

        file_hash = sha256_hash.hexdigest()

        # FAST PATH: If the database already knows this hash, skip PIL decoding completely!
        if file_hash in _WORKER_KNOWN_SHAS:
            return filepath, file_hash, mtime, None

        # 2. Pre-processing (Resize in parallel process)
        img_bytes = None
        try:
            if ext in (".jpg", ".jpeg", ".png", ".webp", ".mp4"):
                img = None
                if ext == ".mp4":
                    import cv2

                    # Extract frame using OpenCV (max 6 seconds in to avoid long decode times)
                    cap = cv2.VideoCapture(str(filepath))
                    if cap.isOpened():
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        if fps <= 0:
                            fps = 30.0

                        target_frame = total_frames // 2
                        six_seconds_frames = int(fps * 6)

                        seek_frame = min(target_frame, six_seconds_frames)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, seek_frame))
                        ret, frame = cap.read()
                        if ret:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            img = Image.fromarray(frame)
                    cap.release()
                else:
                    # USE MEMORY BUFFER instead of opening from disk a second time
                    if file_buffer:
                        img = Image.open(io.BytesIO(file_buffer))
                    else:
                        img = Image.open(filepath)

                if img is not None and (img.width * img.height) <= MAX_IMAGE_PIXELS:
                    # Respect EXIF rotation before thumbnailing
                    img = ImageOps.exif_transpose(img)
                    img = img.convert("RGB")
                    img.thumbnail(PREPROCESS_TARGET_SIZE, Image.Resampling.LANCZOS)

                    out_buffer = io.BytesIO()
                    img.save(out_buffer, format="JPEG", quality=90)
                    img_bytes = out_buffer.getvalue()
                else:
                    # File exists but recognized format failed to decode
                    return filepath, INVALID_FILE_SENTINEL, mtime, None
        except Exception as e:
            logger.warning(f"Decoding failed for {filepath}: {e}")
            return filepath, INVALID_FILE_SENTINEL, mtime, None

        return filepath, file_hash, mtime, img_bytes

    except Exception as e:
        logger.warning(f"Unexpected error processing {filepath}: {e}")
        return filepath, INVALID_FILE_SENTINEL, mtime, None


class EmbeddingConsumerThread(threading.Thread):
    """
    Consumes (path, hash, mtime, bytes) tuples.
    Writes filepaths to DB. Generates embeddings for new hashes using the pre-resized bytes.
    """

    def __init__(
        self,
        db_path: str,
        work_queue: queue.Queue,
        embedder: "ImageEmbedder",
        cancel_flag: threading.Event,
        progress_callback=None,
        total_jobs=0,
    ):
        super().__init__()
        self.db_path = db_path
        self.work_queue = work_queue
        self.embedder = embedder
        self.cancel_flag = cancel_flag
        self.progress_callback = progress_callback
        self.total_jobs = total_jobs
        self.processed_count = 0
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
                except Exception as e:
                    if self.cancel_flag.is_set():
                        break
                    logger.error(f"Unhandled exception in consumer inner loop: {e}", exc_info=True)
                    break

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
            new_thumbnails_to_commit = []
            valid_new_shas = set()

            # Prepare images for embedding
            pil_images = []
            temp_valid_shas = []

            for sha, img_bytes in zip(shas, img_byte_list):
                # Embed if: Not in DB, Image data exists, Not already queued, Not sentinel
                if (
                    sha != INVALID_FILE_SENTINEL
                    and sha not in existing_shas_in_db
                    and img_bytes is not None
                    and sha not in temp_valid_shas
                ):
                    try:
                        # Fast load from memory
                        img = Image.open(io.BytesIO(img_bytes))
                        pil_images.append(img)
                        temp_valid_shas.append(sha)
                        # Store bytes for the database
                        new_thumbnails_to_commit.append((sha, img_bytes))
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

                # Commit the thumbnails
                if new_thumbnails_to_commit:
                    self.conn.executemany(
                        "INSERT OR IGNORE INTO thumbnails (sha256, image_data) VALUES (?, ?)", new_thumbnails_to_commit
                    )

                # POISON PILL FIX: Record ALL filepaths from the batch.
                # - Valid files: point to their real embedding
                # - Invalid files: point to the sentinel row (which always exists)
                # This ensures the DB knows about every file, preventing re-processing on next sync.
                filepath_data = [(str(fp), sha, mt) for fp, sha, mt in zip(filepaths, shas, mtimes)]

                if filepath_data:
                    self.conn.executemany(
                        "INSERT OR REPLACE INTO filepaths (filepath, sha256, mtime) VALUES (?, ?, ?)", filepath_data
                    )

            if self.progress_callback:
                self.processed_count += len(batch)
                self.progress_callback("processing", self.processed_count, self.total_jobs)

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
        self._sha_to_tags_cache = None  # NEW: Cache tags in RAM for instant searches
        self._cancel_flag = threading.Event()
        self._cache_lock = threading.Lock()
        self._load_embeddings_into_memory()

    def _get_db_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA synchronous = NORMAL;")
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

            # --- NEW THUMBNAILS TABLE ---
            conn.execute(
                "CREATE TABLE IF NOT EXISTS thumbnails (sha256 TEXT PRIMARY KEY, image_data BLOB NOT NULL, FOREIGN KEY (sha256) REFERENCES embeddings (sha256) ON DELETE CASCADE)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS visualization (sha256 TEXT PRIMARY KEY, coord_x REAL NOT NULL, coord_y REAL NOT NULL, cluster_id INTEGER NOT NULL, FOREIGN KEY (sha256) REFERENCES embeddings (sha256) ON DELETE CASCADE)"
            )
            conn.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

            # --- TAGS TABLE ---
            conn.execute("""CREATE TABLE IF NOT EXISTS tags (
                    sha256 TEXT NOT NULL,
                    tag_name TEXT NOT NULL,
                    PRIMARY KEY (sha256, tag_name),
                    FOREIGN KEY (sha256) REFERENCES embeddings (sha256) ON DELETE CASCADE
                )""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tags_sha256 ON tags(sha256)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tags_tag_name ON tags(tag_name)")
            # -----------------

            # --- POISON PILL: Insert sentinel for invalid files ---
            # This satisfies the FK constraint for files that fail hashing/decoding.
            # We use a minimal 1-byte blob as placeholder.
            conn.execute(
                "INSERT OR IGNORE INTO embeddings (sha256, embedding) VALUES (?, ?)",
                (INVALID_FILE_SENTINEL, b"\x00"),
            )

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

        # Fetch existing SHAs to pass to workers for the fast-path check
        with self._get_db_connection() as conn:
            existing_shas = {row[0] for row in conn.execute("SELECT sha256 FROM embeddings").fetchall()}

        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=HASHING_WORKER_COUNT, initializer=_init_worker, initargs=(existing_shas,)
        )
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
            try:
                executor.shutdown(wait=True, cancel_futures=True)
            except Exception as e:
                logger.warning(f"Executor took too long to shutdown: {e}")
                # Allow program to continue even if shutdown hangs

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
                if p.is_symlink():
                    continue
                if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".mp4"):
                    try:
                        disk_files[p.resolve().as_posix()] = p.stat().st_mtime
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
            status_callback(f"Hashing and Embedding {total_jobs} files...")
        work_queue = queue.Queue(maxsize=PIPELINE_QUEUE_SIZE)
        consumer = EmbeddingConsumerThread(
            self.db_path, work_queue, self.embedder, self._cancel_flag, progress_callback, total_jobs
        )
        consumer.start()

        # Sort paths alphabetically to roughly group files by directory.
        # This drastically reduces seek times on mechanical spinning hard drives.
        sorted_paths = sorted(list(paths_to_hash))
        job_args = [(Path(p), disk_files[p]) for p in sorted_paths]
        completed = 0
        try:
            # Force a tiny chunksize.
            # By default, Python divides the list into massive chunks, causing concurrent workers
            # to read from completely different alphabetical sections of the disk.
            # A chunksize of 4 forces workers to process adjacent files, respecting the sort order.
            results = executor.map(_targeted_hashing_and_resize_worker, *(zip(*job_args)), chunksize=4)
            for result in results:
                check_cancelled()
                if result:
                    while not self._cancel_flag.is_set():
                        try:
                            work_queue.put(result, timeout=1.0)
                            break
                        except queue.Full:
                            continue
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
        with self._cache_lock:
            self._sha_to_path_map_cache = None
            self._sha_to_tags_cache = None  # Invalidate tag cache on reload
        with self._get_db_connection() as conn:
            # Filter out the sentinel - its 1-byte blob cannot be reshaped to embedding dimensions
            rows = conn.execute(
                "SELECT sha256, embedding FROM embeddings WHERE sha256 != ?",
                (INVALID_FILE_SENTINEL,),
            ).fetchall()
        if not rows:
            self._shas_in_order, self._embedding_matrix = [], None
            return
        self._shas_in_order, blobs = zip(*rows)
        self._embedding_matrix = np.vstack([self._reconstruct_embedding(b) for b in blobs])

    def _cleanup_orphaned_embeddings(self):
        # --- OPTIMIZATION: Use NOT EXISTS and cleanup visualization too ---
        with self._get_db_connection() as conn:
            # POISON PILL FIX: Preserve the sentinel row even if no invalid files reference it.
            # The sentinel is needed for future invalid files to satisfy FK constraints.
            conn.execute(
                "DELETE FROM embeddings WHERE NOT EXISTS (SELECT 1 FROM filepaths WHERE filepaths.sha256 = embeddings.sha256) AND sha256 != ?",
                (INVALID_FILE_SENTINEL,),
            )
            # Also remove visualization entries that no longer have a corresponding embedding
            conn.execute("DELETE FROM visualization WHERE sha256 NOT IN (SELECT sha256 FROM embeddings)")
        # ------------------------------------------------------------------

    def _reconstruct_embedding(self, blob):
        # Defensive check: verify buffer length matches expected embedding size
        expected_bytes = np.dtype(self.embedder.embedding_dtype).itemsize * np.prod(self.embedder.embedding_shape)
        if len(blob) != expected_bytes:
            logger.warning(
                f"Malformed embedding blob detected (size {len(blob)}, expected {expected_bytes}). Returning zero vector."
            )
            return np.zeros(self.embedder.embedding_shape, dtype=self.embedder.embedding_dtype)
        return np.frombuffer(blob, dtype=self.embedder.embedding_dtype).reshape(self.embedder.embedding_shape)

    def ensure_visualization_data(self, status_callback=None, check_cancelled_callback=None):
        """
        Calculates visualization data using Incremental PCA and Landmark UMAP
        to optimize for speed and memory usage on large datasets.
        """
        try:
            # 1. Check if update is needed
            with self._get_db_connection() as conn:
                # Exclude the sentinel from the embedding count
                emb_count = conn.execute(
                    "SELECT COUNT(*) FROM embeddings WHERE sha256 != ?", (INVALID_FILE_SENTINEL,)
                ).fetchone()[0]
                vis_count = conn.execute("SELECT COUNT(*) FROM visualization").fetchone()[0]

                # Fast check: Is there ANY valid embedding missing from the visualization table?
                missing_vis = conn.execute(
                    "SELECT 1 FROM embeddings e WHERE e.sha256 != ? AND NOT EXISTS (SELECT 1 FROM visualization v WHERE v.sha256 = e.sha256) LIMIT 1",
                    (INVALID_FILE_SENTINEL,),
                ).fetchone()

            # If counts match AND no embeddings are missing, we are 100% up to date.
            if emb_count == 0 or (emb_count == vis_count and missing_vis is None):
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
            try:
                # Exclude sentinel - its 1-byte blob cannot be reshaped to embedding dimensions
                cursor = conn.execute("SELECT embedding FROM embeddings WHERE sha256 != ?", (INVALID_FILE_SENTINEL,))
                try:
                    while True:
                        if check_cancelled_callback:
                            check_cancelled_callback()

                        rows = cursor.fetchmany(batch_size)
                        if not rows:
                            break

                        # Convert bytes to numpy batch
                        batch = np.vstack([self._reconstruct_embedding(r[0]).flatten() for r in rows])
                        ipca.partial_fit(batch)
                finally:
                    cursor.close()
            finally:
                conn.close()

            # --- STEP 2: TRANSFORM & LOAD COMPRESSED DATA ---
            # Now we load the data, but it's only 50 floats per image instead of 768.
            # 60k images * 50 dims * 4 bytes ~= 12 MB RAM (Tiny!)

            if status_callback:
                status_callback("Projecting data...")

            compressed_data = []
            all_shas = []

            conn = self._get_db_connection()
            try:
                # Exclude sentinel - its 1-byte blob cannot be reshaped to embedding dimensions
                cursor = conn.execute(
                    "SELECT sha256, embedding FROM embeddings WHERE sha256 != ?",
                    (INVALID_FILE_SENTINEL,),
                )
                try:
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
                finally:
                    cursor.close()
            finally:
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
            if len(coords_2d) < 10:
                cluster_labels = np.zeros(len(coords_2d), dtype=int)
            else:
                cluster_labels = hdbscan.HDBSCAN(min_cluster_size=10, core_dist_n_jobs=-1).fit_predict(coords_2d)

            # --- STEP 5: SAVE ---
            if status_callback:
                status_callback("Saving visualization data...")

            vis_data = [(sh, float(c[0]), float(c[1]), int(l)) for sh, c, l in zip(all_shas, coords_2d, cluster_labels)]

            with self._get_db_connection() as conn:
                # The data is ready. Any InterruptedError raised *before* this point means
                # the old visualization data is safely retained without being deleted.
                # The DELETE and INSERT below are wrapped in an atomic transaction
                # by the sqlite3 connection's context manager.
                conn.execute("DELETE FROM visualization")
                conn.executemany(
                    "INSERT INTO visualization (sha256, coord_x, coord_y, cluster_id) VALUES (?, ?, ?, ?)", vis_data
                )

        except Exception as e:
            logger.error(f"Visualization update failed: {e}")
            raise e

    # Rename _build_sha_to_path_map to _build_memory_caches and load tags too
    def _build_memory_caches(self):
        self._sha_to_path_map_cache = defaultdict(list)
        self._sha_to_tags_cache = {}
        with self._get_db_connection() as conn:
            # 1. Build Filepath Cache
            for sha, path in conn.execute("SELECT sha256, filepath FROM filepaths").fetchall():
                self._sha_to_path_map_cache[sha].append(path)

            # 2. Build Tag Cache
            for sha, tags in conn.execute("SELECT sha256, GROUP_CONCAT(tag_name) FROM tags GROUP BY sha256").fetchall():
                self._sha_to_tags_cache[sha] = tags

    def _ensure_caches(self):
        with self._cache_lock:
            if self._sha_to_path_map_cache is None or self._sha_to_tags_cache is None:
                self._build_memory_caches()

    def _perform_search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[float, str, str]]:
        # --- OPTIMIZATION: Efficient Numpy Search ---
        if self._embedding_matrix is None or len(self._embedding_matrix) == 0:
            return []

        query_embedding = query_embedding.astype(self.embedder.embedding_dtype).reshape(1, -1)
        similarities = np.dot(self._embedding_matrix, query_embedding.T).flatten()

        # Build caches if they don't exist
        self._ensure_caches()

        if top_k != -1 and top_k < len(similarities):
            # Partial sort is faster than full sort
            indices = np.argpartition(similarities, -top_k)[-top_k:]
            indices = indices[np.argsort(similarities[indices])[::-1]]
        else:
            indices = np.argsort(similarities)[::-1]

        # Lookups are now instant RAM dictionary lookups. No SQLite execution needed!
        results = []
        for i in indices:
            sha = self._shas_in_order[i]
            paths = self._sha_to_path_map_cache.get(sha)
            if paths:
                tags = self._sha_to_tags_cache.get(sha, "")
                duplicate_count = len(paths)
                results.append((float(similarities[i]), paths[0], tags, duplicate_count))
        return results
        # --------------------------------------------

    def search_similar_images(self, image_path, top_k=-1):
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            return []
        return self._perform_search(self.embedder.embed_image(image), top_k)

    def search_by_text(self, text, top_k=-1):
        return self._perform_search(self.embedder.embed_text(text), top_k)

    def cancel_sync(self):
        self._cancel_flag.set()

    def get_all_unique_filepaths(self):
        """Used for Random Search. Entirely DB-free and instant."""
        self._ensure_caches()

        results = []
        for sha, paths in self._sha_to_path_map_cache.items():
            if paths and sha != INVALID_FILE_SENTINEL:
                tags = self._sha_to_tags_cache.get(sha, "")
                duplicate_count = len(paths)
                results.append((paths[0], tags, duplicate_count))
        return results

    def get_all_filepaths_with_mtime(self):
        """Used for Sort By Date. Fetches mtime, maps tags via RAM."""
        self._ensure_caches()

        # Fast query: no GROUP BY, no tag joins.
        with self._get_db_connection() as conn:
            rows = conn.execute(
                "SELECT filepath, sha256, mtime FROM filepaths WHERE sha256 != ?", (INVALID_FILE_SENTINEL,)
            ).fetchall()

        sha_to_mtime = {}
        for filepath, sha, mtime in rows:
            if sha not in sha_to_mtime or mtime > sha_to_mtime[sha]:
                sha_to_mtime[sha] = mtime

        results = []
        for sha, paths in self._sha_to_path_map_cache.items():
            if paths and sha != INVALID_FILE_SENTINEL:
                tags = self._sha_to_tags_cache.get(sha, "")
                duplicate_count = len(paths)
                mtime = sha_to_mtime.get(sha, 0.0)
                results.append((paths[0], mtime, tags, duplicate_count))
                
        return results

    def _get_sha256_for_filepaths(self, filepath_list: list[str]) -> list[str]:
        """Convert filepaths to their corresponding SHA256 hashes."""
        if not filepath_list:
            return []

        sha256_list = []
        with self._get_db_connection() as conn:
            for i in range(0, len(filepath_list), SQLITE_VARIABLE_LIMIT):
                chunk = filepath_list[i : i + SQLITE_VARIABLE_LIMIT]
                placeholders = ", ".join("?" for _ in chunk)
                cursor = conn.execute(f"SELECT sha256 FROM filepaths WHERE filepath IN ({placeholders})", chunk)
                sha256_list.extend(row[0] for row in cursor.fetchall())
        return sha256_list

    def toggle_tag(self, filepath_or_sha_list: list[str], tag_name: str = "marked"):
        if not filepath_or_sha_list:
            return

        sha256_list = self._get_sha256_for_filepaths(filepath_or_sha_list)
        if not sha256_list:
            return

        with self._get_db_connection() as conn:
            for i in range(0, len(sha256_list), SQLITE_VARIABLE_LIMIT):
                chunk = sha256_list[i : i + SQLITE_VARIABLE_LIMIT]
                placeholders = ", ".join("?" for _ in chunk)

                cursor = conn.execute(
                    f"SELECT sha256 FROM tags WHERE sha256 IN ({placeholders}) AND tag_name = ?", chunk + [tag_name]
                )
                tagged_shas = {row[0] for row in cursor.fetchall()}

                for sha in chunk:
                    if sha in tagged_shas:
                        conn.execute("DELETE FROM tags WHERE sha256 = ? AND tag_name = ?", (sha, tag_name))
                    else:
                        conn.execute("INSERT OR IGNORE INTO tags (sha256, tag_name) VALUES (?, ?)", (sha, tag_name))

        # INVALIDATE CACHE so it rebuilds on next search
        self._sha_to_tags_cache = None

    def untag_all(self, tag_name: str = "marked"):
        with self._get_db_connection() as conn:
            conn.execute("DELETE FROM tags WHERE tag_name = ?", (tag_name,))

        # INVALIDATE CACHE
        self._sha_to_tags_cache = None

    def delete_target_filepaths(self, filepath_list: list[str]):
        """Delete specific filepaths from the database without full sync."""
        if not filepath_list:
            return

        with self._get_db_connection() as conn:
            for i in range(0, len(filepath_list), SQLITE_VARIABLE_LIMIT):
                chunk = filepath_list[i : i + SQLITE_VARIABLE_LIMIT]
                placeholders = ", ".join("?" for _ in chunk)
                conn.execute(f"DELETE FROM filepaths WHERE filepath IN ({placeholders})", chunk)

        # Clean up orphaned embeddings after deletion
        self._cleanup_orphaned_embeddings()

        # Invalidate caches so deleted files disappear from subsequent searches/sorts
        self._sha_to_path_map_cache = None
        self._sha_to_tags_cache = None

    def get_all_embeddings_with_shas(self):
        with self._get_db_connection() as conn:
            return [
                {"sha256": sha, "embedding": self._reconstruct_embedding(eb).flatten()}
                for sha, eb in conn.execute(
                    "SELECT sha256, embedding FROM embeddings WHERE sha256 != ?", (INVALID_FILE_SENTINEL,)
                ).fetchall()
            ]

    def get_visualization_data(self):
        with self._get_db_connection() as conn:
            return conn.execute(
                "SELECT v.coord_x, v.coord_y, v.cluster_id, f.filepath FROM visualization v JOIN (SELECT sha256, MIN(filepath) as filepath FROM filepaths GROUP BY sha256) f ON v.sha256 = f.sha256"
            ).fetchall()

    def close(self):
        """
        Clean up database resources.
        Releases the in-memory embedding matrix and clears caches.
        """
        logger.info("Closing ImageDatabase and freeing resources...")
        self._embedding_matrix = None
        self._shas_in_order = []
        self._sha_to_path_map_cache = None
        self._sha_to_tags_cache = None  # Clear tag cache
        self._cancel_flag.set()
        logger.info("ImageDatabase closed.")
