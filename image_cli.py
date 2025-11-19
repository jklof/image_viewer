import argparse
import logging
import multiprocessing
from pathlib import Path
import sys
from tqdm import tqdm

from image_db import ImageDatabase
from ml_core import ImageEmbedder
from config_utils import get_scan_directories, get_db_path, get_model_id

# Setup basic logging for the CLI.
# We use a format that fits well with CLI output.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def handle_sync(db: ImageDatabase, args: argparse.Namespace):
    """Handles the database synchronization command."""
    logger.info(f"Loading directories to scan from '{args.config}'...")
    directories_to_scan = get_scan_directories(args.config)
    if not directories_to_scan:
        logger.warning("No directories were loaded from the config. Nothing to do.")
        return

    logger.info("Starting database synchronization. This may take a while...")

    # --- TQDM Progress Bar Setup ---
    pbar = None  # Keep a reference to the progress bar

    def cli_progress_callback(stage: str, current: int, total: int):
        nonlocal pbar
        if stage == "hashing":
            if pbar is None:
                pbar = tqdm(total=total, desc="Processing files", unit=" img", ncols=100)

            # TQDM update expects the delta, but our callback provides the absolute total.
            # Calculate the delta:
            delta = current - pbar.n
            if delta > 0:
                pbar.update(delta)

    def cli_status_callback(message: str):
        # Use tqdm.write to print messages without breaking the progress bar layout
        tqdm.write(f"-> {message}")

    try:
        db.reconcile_database(
            configured_dirs=directories_to_scan,
            progress_callback=cli_progress_callback,
            status_callback=cli_status_callback,
        )
        # Ensure pbar hits 100% visually if it exists
        if pbar:
            pbar.update(pbar.total - pbar.n)
        logger.info("Synchronization complete.")
    except KeyboardInterrupt:
        logger.warning("Synchronization interrupted by user.")
        db.cancel_sync()
    finally:
        if pbar:
            pbar.close()


def handle_search_image(db: ImageDatabase, args: argparse.Namespace):
    """Handles the search-by-image command."""
    query_path = Path(args.image_path)
    if not query_path.is_file():
        logger.error(f"Error: Image file '{args.image_path}' not found.")
        return

    logger.info(f"Searching for top {args.top_k} images similar to '{query_path.name}'...")
    results = db.search_similar_images(str(query_path), args.top_k)

    print(f"\n--- Top {len(results)} images similar to '{query_path.name}' ---")
    for score, path in results:
        print(f"Similarity: {score:.4f}\tPath: {path}")


def handle_search_text(db: ImageDatabase, args: argparse.Namespace):
    """Handles the search-by-text command."""
    logger.info(f"Searching for top {args.top_k} images matching query: '{args.query}'...")
    results = db.search_by_text(args.query, args.top_k)

    print(f"\n--- Top {len(results)} images matching '{args.query}' ---")
    for score, path in results:
        print(f"Relevance: {score:.4f}\tPath: {path}")


def main():
    """Main function to set up parser and handle command dispatch."""

    # --- CRITICAL: Set start method to 'spawn' ---
    # PyTorch + Multiprocessing on Linux defaults to 'fork', which is unsafe
    # when CUDA or OpenMP libraries are initialized. We must force 'spawn'.
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="An AI-powered command-line image management tool.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,  # Default to None to prioritize config file
        help="Path to the SQLite database file (overrides config.yml).",
    )
    parser.add_argument("--config", type=str, default="config.yml", help="Path to the config.yml file.")

    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force the application to use the CPU for all ML computations, ignoring the GPU.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- Sync command ---
    sync_parser = subparsers.add_parser("sync", help="Synchronize the database with configured directories.")
    sync_parser.set_defaults(func=handle_sync)

    # --- Search by Image command ---
    search_img_parser = subparsers.add_parser("search-image", help="Search for images similar to a given image.")
    search_img_parser.add_argument("image_path", type=str, help="Path to the query image.")
    search_img_parser.add_argument("--top-k", type=int, default=5, help="Number of similar images to find.")
    search_img_parser.set_defaults(func=handle_search_image)

    # --- Search by Text command ---
    search_text_parser = subparsers.add_parser("search-text", help="Search for images matching a text description.")
    search_text_parser.add_argument("query", type=str, help="The text description to search for.")
    search_text_parser.add_argument("--top-k", type=int, default=5, help="Number of matching images to find.")
    search_text_parser.set_defaults(func=handle_search_text)

    args = parser.parse_args()

    try:
        # Lazily instantiate the embedder only for commands that need it.
        if hasattr(args, "func"):
            logger.info("Initializing ML models...")

            # Load model ID from config
            model_id = get_model_id(args.config)
            embedder = ImageEmbedder(model_id=model_id, use_cpu_only=args.cpu_only)

            # Determine DB path
            db_path = args.db_path if args.db_path else get_db_path(args.config)

            logger.info(f"Opening database at '{db_path}'...")
            db = ImageDatabase(db_path=db_path, embedder=embedder)

            # Dispatch to the appropriate handler function
            args.func(db, args)
        else:
            parser.print_help()

    except ImageDatabase.ModelMismatchError as e:
        logger.error("--- DATABASE AND MODEL ARE INCOMPATIBLE ---")
        logger.error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        # Handle Ctrl+C cleanly at the top level
        print("\nOperation aborted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
