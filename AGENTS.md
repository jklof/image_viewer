# AGENTS.md

## Project Overview

**AI Image Explorer** is a local-first desktop application built with **PySide6** and **PyTorch (CLIP)**. It indexes image and video collections on local drives, computes normalized vector embeddings, stores metadata and thumbnails in SQLite, and enables:
1. **Multi-modal composite search:** Weighted text and image queries calculated via in-memory vector dot-products.
2. **2D embedding visualization:** Low-dimensional semantic clustering using Incremental PCA + UMAP + HDBSCAN rendered via `pyqtgraph`.
3. **Duplicate management & Tagging:** Content-addressed file management based on SHA-256 hashes.
4. **Media inspection:** Single-view inspection with video scrubbing (OpenCV) and ComfyUI/generation metadata extraction.

---

## Architectural Topology

The application relies on strict concurrency boundaries to keep the UI fluid and prevent CUDA/Qt event-loop conflicts.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        MAIN THREAD (Qt Event Loop)                     │
│  MainWindow  ──►  AppController  ──►  ImageResultModel / Virtual Views │
└──────┬────────────────────┬────────────────────┬───────────────────────┘
       │ queue.Queue        │ QSignals           │ QThreadPool (LIFO)
       ▼                    ▼                    ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────────────────┐
│  BackendWorker   │ │   SyncWorker     │ │       LoaderManager          │
│ (Python Thread)  │ │   (QThread)      │ │  Workers with thread-local   │
│                  │ │                  │ │  read-only SQLite URI conns  │
│ ImageEmbedder    │ │ Reconciles disk  │ └──────────────────────────────┘
│ In-Memory DB/RAM │ │ and DB records   │
└────────┬─────────┘ └────────┬─────────┘
         │                    │ ProcessPoolExecutor (spawn)
         │                    ▼
         │           ┌───────────────────────────────────────────┐
         │           │ _targeted_hashing_and_resize_worker       │
         │           │ Reads buffer once; hashes & pre-resizes   │
         │           └───────────────────────────────────────────┘
         ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     SQLite Database (WAL Mode)                         │
│       embeddings  |  filepaths  |  thumbnails  |  tags  |  vis         │
└────────────────────────────────────────────────────────────────────────┘
```

### Threading & Execution Responsibilities

1. **Main Thread (`main_window.py`, `virtual_model.py`):**
   - Handles all Qt rendering and user events.
   - Updates `ImageResultModel` optimistically (e.g., tag toggling).
   - Never touches heavy disk I/O, PIL decode, or PyTorch inference.

2. **Backend Thread (`backend.py`):**
   - Runs in a standard Python `threading.Thread` (not a `QThread`) consuming jobs from a standard `queue.Queue`.
   - Owns the primary in-memory `ImageDatabase` and `ImageEmbedder`.
   - Performs vector math (`np.dot`), fast tag queries, and DB writes triggered from the UI.
   - Communicates back to the controller via Qt signals (`BackendSignals`).

3. **Sync Worker (`sync_worker.py`):**
   - Runs on a `QThread` to reconcile disk state with SQLite.
   - Borrows the shared `ImageEmbedder` from the backend worker to avoid allocating duplicate model weights in VRAM.
   - Spawns a `ProcessPoolExecutor` with the `spawn` start method to perform parallel CPU hashing and thumbnail resizing.

4. **Thumbnail Workers (`loader_manager.py`):**
   - Managed by `QThreadPool`.
   - Uses **thread-local read-only SQLite connections** (`file:...mode=ro`) to load pre-rendered thumbnails from the database without blocking write transactions during sync.
   - Prioritizes requests via a LIFO task counter so visible items load first during fast scrolling.

---

## Data Model & Persistence (`image_db.py`)

The data architecture is **content-addressed (SHA-256 centric)**:

- `embeddings (sha256 PK, embedding BLOB)`: Canonical unit-norm float32 embedding vector per unique file content.
- `filepaths (filepath PK, sha256 FK, mtime)`: Maps multiple disk locations to a single content hash. Has `ON DELETE CASCADE`.
- `thumbnails (sha256 PK, image_data BLOB)`: JPEG-compressed pre-rendered thumbnails (256×256) stored directly in SQLite for fast retrieval.
- `tags (sha256, tag_name PK)`: Tags belong to the **content hash**, meaning duplicate files share tags automatically.
- `visualization (sha256 PK, coord_x, coord_y, cluster_id)`: Cached 2D coordinates and cluster assignments.

### In-Memory Search Layer
At startup and after sync, `ImageDatabase` loads all valid embeddings into RAM:
- `self._embedding_matrix`: 2D NumPy array (`N x D`).
- `self._shas_in_order`: List of SHAs aligned with matrix rows.
- `self._sha_to_path_map_cache` & `self._sha_to_tags_cache`: Dictionary lookups to eliminate SQLite queries during real-time search.

Search computes:
$$\text{similarities} = \mathbf{X} \cdot \mathbf{q}^T$$
and sorts via `np.argpartition` for top-$k$.

---

## Key Subsystems & Workflows

### 1. Ingestion Pipeline
1. **Discovery:** Scans configured directories. Skips any directory whose root is missing/unmounted (offline storage protection).
2. **Diffing:** Compares `(filepath, mtime)` against SQLite.
3. **Hashing & Resize (`ProcessPoolExecutor`):**
   - Files < 50 MB are buffered into memory once to prevent dual reads for hash + thumbnail.
   - If the SHA is already in the DB, image decoding is skipped entirely.
   - Corrupted or unreadable files receive the sentinel SHA `__INVALID__` so they are never rescanned.
4. **Batch Embedding:** Pre-resized byte buffers are sent to `EmbeddingConsumerThread`, batched (default 64), embedded by CLIP, and committed in transactions.

### 2. Multi-Modal Query Builder (`query_builder.py`)
- Text and image elements each have weights ranging from $-1.0$ to $+1.0$.
- Embeddings are calculated per item, weighted, summed, and normalized:
  $$\mathbf{v}_{\text{query}} = \frac{\sum w_i \mathbf{e}_i}{\left\|\sum w_i \mathbf{e}_i\right\|_2}$$
- Negative weights invert the vector direction, allowing negative semantic queries (e.g., `- "night"` or `- <unwanted image>`).

### 3. Dimensionality Reduction (`ensure_visualization_data`)
To scale 2D plotting without high RAM usage:
1. Streams embeddings from SQLite through **Incremental PCA** (768D $\to$ 50D).
2. Runs **UMAP** on the 50D data down to 2D.
3. Clusters 2D coordinates using **HDBSCAN**.
4. Renders interactive points using `pyqtgraph.ScatterPlotItem`.

### 4. VRAM Management (`ml_core.py`)
- `ImageEmbedder` initializes weights in CPU RAM.
- When an embedding is requested, `_wake_up()` moves the model to GPU (`cuda`/`mps`).
- A 30-second inactivity timer automatically offloads weights back to system RAM and invokes `torch.cuda.empty_cache()`.

---

## Core Invariants & Conventions

When modifying or extending the codebase, preserve these invariants:

1. **Multiprocessing Start Method:** Must remain `"spawn"`. PyTorch with CUDA or OpenMP cannot safely `fork`.
2. **Never Block the Main Thread:** Do not call `embed_text`, `embed_image`, heavy disk operations, or unbounded SQLite transactions from UI classes. Route through `AppController` and `backend_job_queue`.
3. **Thread Safety in `ml_core.py`:** All model evaluations and VRAM state transitions must occur within `ImageEmbedder.lock`.
4. **Foreign Key Integrity:** `PRAGMA foreign_keys = ON;` is enforced. Never insert into `filepaths`, `thumbnails`, or `tags` unless the corresponding `sha256` already exists in `embeddings` (or matches `__INVALID__`).
5. **Read-Only Connections in UI Workers:** Background thumbnail workers must connect to SQLite with URI mode `mode=ro` to avoid SQLite busy timeouts and write lock contention.
6. **Graceful Cancellation:** Long-running jobs (sync, UMAP) periodically poll `_cancel_flag` / `check_cancelled()`. Always handle `InterruptedError` cleanly.