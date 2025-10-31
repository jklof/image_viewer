import hashlib
import logging
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
from PIL import Image
from tqdm import tqdm

from ml_core import EMBEDDING_DTYPE, EMBEDDING_SHAPE, ImageEmbedder

logger = logging.getLogger(__name__)

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

        # This is now SAFE because this __init__ will be called from the worker thread.
        # No need for check_same_thread=False.
        self.conn = sqlite3.connect(self.db_path)
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
        logger.info("Starting database synchronization...")
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
                try:
                    disk_files[p.resolve()] = p.stat().st_mtime
                except FileNotFoundError:
                    continue

        db_paths, disk_paths = set(db_files.keys()), set(disk_files.keys())
        added_paths = disk_paths - db_paths
        removed_paths = db_paths - disk_paths
        potential_modified_paths = disk_paths.intersection(db_paths)
        modified_paths = {p for p in potential_modified_paths if disk_files[p] > db_files[p]["mtime"]}
        paths_to_hash = added_paths.union(modified_paths)

        logger.info(
            f"Found {len(added_paths)} new, {len(removed_paths)} removed, and {len(modified_paths)} modified files."
        )

        with self.conn:
            if removed_paths:
                placeholders = ", ".join("?" for _ in removed_paths)
                self.conn.execute(
                    f"DELETE FROM filepaths WHERE filepath IN ({placeholders})",
                    [str(p) for p in removed_paths],
                )

            if paths_to_hash:
                hash_to_path_map = defaultdict(list)
                for path in tqdm(paths_to_hash, desc="Hashing files"):
                    sha = self._calculate_sha256(path)
                    if sha:
                        hash_to_path_map[sha].append(path)

                new_shas = set(hash_to_path_map.keys())
                if new_shas:
                    placeholders = ", ".join("?" for _ in new_shas)
                    cursor.execute(
                        f"SELECT sha256 FROM embeddings WHERE sha256 IN ({placeholders})",
                        list(new_shas),
                    )
                    existing_shas = {row[0] for row in cursor.fetchall()}
                    shas_to_create = list(new_shas - existing_shas)

                    if shas_to_create:
                        logger.info(f"Found {len(shas_to_create)} new unique images to embed.")
                        for i in tqdm(
                            range(0, len(shas_to_create), PROCESSING_BATCH_SIZE),
                            desc="Embedding new images",
                        ):
                            batch_shas = shas_to_create[i : i + PROCESSING_BATCH_SIZE]
                            batch_images_pil, valid_shas_in_batch = [], []
                            for sha in batch_shas:
                                try:
                                    batch_images_pil.append(Image.open(hash_to_path_map[sha][0]).convert("RGB"))
                                    valid_shas_in_batch.append(sha)
                                except Exception as e:
                                    logger.warning(
                                        f"Could not load image {hash_to_path_map[sha][0]} for embedding. Skipping. Error: {e}"
                                    )
                            if batch_images_pil:
                                new_embeddings = self.embedder.embed_batch(batch_images_pil)
                                embedding_data = [
                                    (sha, emb.tobytes()) for sha, emb in zip(valid_shas_in_batch, new_embeddings)
                                ]
                                self.conn.executemany(
                                    "INSERT INTO embeddings (sha256, embedding) VALUES (?, ?)",
                                    embedding_data,
                                )

                filepath_data = [(str(p), sha, disk_files[p]) for sha, paths in hash_to_path_map.items() for p in paths]
                self.conn.executemany(
                    "INSERT OR REPLACE INTO filepaths (filepath, sha256, mtime) VALUES (?, ?, ?)",
                    filepath_data,
                )

            self.conn.execute("DELETE FROM embeddings WHERE sha256 NOT IN (SELECT DISTINCT sha256 FROM filepaths)")

        logger.info("Synchronization complete.")
        self._load_embeddings_into_memory()

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
