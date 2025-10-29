# image_db.py

import hashlib
import logging
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
from PIL import Image
from scipy.spatial.distance import cosine
from tqdm import tqdm

# Local import from our new module
from ml_core import EMBEDDING_DTYPE, EMBEDDING_SHAPE, ImageEmbedder

logger = logging.getLogger(__name__)

# Define a constant for batch processing to control memory usage
PROCESSING_BATCH_SIZE = 64


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
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self._create_tables()

        # --- NEW: In-memory cache for fast searching ---
        self._shas_in_order: List[str] = []
        self._embedding_matrix: np.ndarray | None = None
        self._load_embeddings_into_memory()

    def _create_tables(self):
        """Creates the normalized database schema."""
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

    def _load_embeddings_into_memory(self):
        """Queries all embeddings from the DB and loads them into a NumPy matrix."""
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
        
        # Reconstruct the embeddings into a single large NumPy array
        embeddings_list = [self._reconstruct_embedding(row[1]) for row in rows]
        self._embedding_matrix = np.vstack(embeddings_list)
        
        logger.info(f"Loaded {len(self._shas_in_order)} embeddings into memory cache.")

    @staticmethod
    def _calculate_sha256(filepath: Path) -> str:
        sha256_hash = hashlib.sha256()
        try:
            with open(filepath, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except (IOError, FileNotFoundError):
            return ""

    def _reconstruct_embedding(self, embedding_blob: bytes) -> np.ndarray:
        return np.frombuffer(embedding_blob, dtype=EMBEDDING_DTYPE).reshape(EMBEDDING_SHAPE)

    def reconcile_database(self, configured_dirs: list[str]):
        """
        Reconciles the database using file modification times and processing
        new images in batches to ensure low memory usage.
        """
        logger.info("Starting database synchronization...")

        logger.info("Fetching current database state...")
        cursor = self.conn.cursor()
        cursor.execute("SELECT filepath, mtime, sha256 FROM filepaths")
        db_files = {Path(row[0]).resolve(): {"mtime": row[1], "sha256": row[2]} for row in cursor.fetchall()}

        logger.info("Scanning files on disk...")
        disk_files: Dict[Path, float] = {}
        for directory in configured_dirs:
            path = Path(directory)
            if not path.is_dir():
                logger.warning(f"Configured directory '{directory}' does not exist. Skipping.")
                continue
            image_paths = (p for p in path.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
            for p in image_paths:
                resolved_path = p.resolve()
                try:
                    disk_files[resolved_path] = resolved_path.stat().st_mtime
                except FileNotFoundError:
                    continue

        db_paths, disk_paths = set(db_files.keys()), set(disk_files.keys())
        added_paths = disk_paths - db_paths
        removed_paths = db_paths - disk_paths
        potential_modified_paths = disk_paths.intersection(db_paths)
        modified_paths = {p for p in potential_modified_paths if disk_files[p] > db_files[p]["mtime"]}
        
        paths_to_hash = added_paths.union(modified_paths)
        
        logger.info(f"Found {len(added_paths)} new, {len(removed_paths)} removed, and {len(modified_paths)} modified files.")

        with self.conn:
            if removed_paths:
                logger.info(f"Removing {len(removed_paths)} entries from database...")
                placeholders = ', '.join('?' for _ in removed_paths)
                self.conn.execute(f"DELETE FROM filepaths WHERE filepath IN ({placeholders})", [str(p) for p in removed_paths])

            if paths_to_hash:
                logger.info(f"Hashing and processing {len(paths_to_hash)} files...")
                
                hash_to_path_map = defaultdict(list)
                for path in tqdm(paths_to_hash, desc="Hashing files"):
                    sha = self._calculate_sha256(path)
                    if sha:
                        hash_to_path_map[sha].append(path)

                new_shas = set(hash_to_path_map.keys())
                if not new_shas:
                    shas_to_create = []
                else:
                    placeholders = ', '.join('?' for _ in new_shas)
                    cursor.execute(f"SELECT sha256 FROM embeddings WHERE sha256 IN ({placeholders})", list(new_shas))
                    existing_shas = {row[0] for row in cursor.fetchall()}
                    shas_to_create = list(new_shas - existing_shas)
                
                if shas_to_create:
                    logger.info(f"Found {len(shas_to_create)} new unique images to embed.")
                    
                    for i in tqdm(range(0, len(shas_to_create), PROCESSING_BATCH_SIZE), desc="Embedding new images"):
                        batch_shas = shas_to_create[i : i + PROCESSING_BATCH_SIZE]
                        batch_images_pil = []
                        
                        for sha in batch_shas:
                            path = hash_to_path_map[sha][0]
                            try:
                                batch_images_pil.append(Image.open(path).convert("RGB"))
                            except Exception as e:
                                logger.warning(f"Could not load image {path} for embedding. Skipping. Error: {e}")
                        
                        if batch_images_pil:
                            valid_shas_in_batch = batch_shas[:len(batch_images_pil)]
                            new_embeddings = self.embedder.embed_batch(batch_images_pil)
                            embedding_data = [(sha, emb.tobytes()) for sha, emb in zip(valid_shas_in_batch, new_embeddings)]
                            self.conn.executemany("INSERT INTO embeddings (sha256, embedding) VALUES (?, ?)", embedding_data)

                filepath_data = []
                for sha, paths in hash_to_path_map.items():
                    for path in paths:
                        mtime = disk_files[path]
                        filepath_data.append((str(path), sha, mtime))
                
                self.conn.executemany(
                    "INSERT OR REPLACE INTO filepaths (filepath, sha256, mtime) VALUES (?, ?, ?)",
                    filepath_data
                )
            
            logger.info("Cleaning up orphaned embeddings...")
            self.conn.execute("""
                DELETE FROM embeddings
                WHERE sha256 NOT IN (SELECT DISTINCT sha256 FROM filepaths)
            """)
        
        logger.info("Synchronization complete.")
        self._load_embeddings_into_memory()
        
    def get_all_embeddings_with_filepaths(self) -> List[Dict]:
        logger.info("Fetching data for visualization...")
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT e.embedding, f.filepath FROM embeddings e JOIN (
                SELECT sha256, MIN(filepath) as filepath FROM filepaths GROUP BY sha256
            ) f ON e.sha256 = f.sha256
        """
        )
        return [{"embedding": self._reconstruct_embedding(eb).flatten(), "filepath": fp} for eb, fp in cursor.fetchall()]

    def _perform_search(self, query_embedding: np.ndarray, top_k: int) -> List[tuple[float, str]]:
        """
        Performs a vectorized search against the in-memory embedding matrix.
        """
        if self._embedding_matrix is None or len(self._embedding_matrix) == 0:
            return []

        similarities = np.dot(query_embedding, self._embedding_matrix.T).flatten()

        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        
        top_results = sorted(
            [(similarities[i], self._shas_in_order[i]) for i in top_indices],
            key=lambda x: x[0],
            reverse=True
        )

        results, seen_sha256 = [], set()
        cursor = self.conn.cursor()
        for score, sha in top_results:
            if sha in seen_sha256: continue

            cursor.execute("SELECT filepath FROM filepaths WHERE sha256 = ?", (sha,))
            for row in cursor.fetchall():
                results.append((float(score), row[0]))
            seen_sha256.add(sha)
        
        return results[:top_k]

    def search_similar_images(self, image_path: str, top_k: int = 5):
        """
        Finds similar images to a given image path. The first result is always
        the query image itself with a perfect similarity score of 1.0.
        """
        query_path = Path(image_path)
        try:
            image = Image.open(query_path).convert("RGB")
        except FileNotFoundError:
            logger.error(f"Query image not found at '{image_path}'")
            return []

        query_embedding = self.embedder.embed_image(image)
        
        # Find the top k similar images from the database
        similar_images = self._perform_search(query_embedding, top_k)

        # Filter out the query image itself from the list of similar images,
        # as we will prepend it manually.
        query_path_resolved = query_path.resolve()
        other_results = [
            (score, path) for score, path in similar_images 
            if Path(path).resolve() != query_path_resolved
        ]
        
        # Prepend the original query image with a perfect score.
        final_results = [(1.0, image_path)] + other_results
        
        # Ensure the final list is no longer than top_k.
        return final_results[:top_k]

    def search_by_text(self, text_query: str, top_k: int = 5):
        """Finds images matching a text query using the fast in-memory search."""
        query_embedding = self.embedder.embed_text(text_query)
        return self._perform_search(query_embedding, top_k)

    def close(self):
        if self.conn:
            logger.info("Closing database connection.")
            self.conn.close()