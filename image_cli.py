import argparse
import logging
from pathlib import Path
import sys
import yaml

from image_db import ImageDatabase
from ml_core import ImageEmbedder

# Setup basic logging for the CLI. Set to WARNING to keep the output clean,
# but allow module-level loggers (like ml_core) to print INFO if needed.
logging.basicConfig(format="%(name)s - %(levelname)s - %(message)s", level=logging.WARNING)
# Specifically set the logger for this script to INFO for our status messages.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_config(config_path="config.yml") -> list[str] | None:
    """Loads the list of directories to scan from the YAML configuration file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            if "directories" in config and isinstance(config["directories"], list):
                return config["directories"]
            else:
                logger.error(f"Error: Config file '{config_path}' is missing a 'directories' list or it is malformed.")
                return None
    except FileNotFoundError:
        logger.error(f"Error: Configuration file '{config_path}' not found.")
        logger.error("Please create it and add the directories you want to scan.")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file '{config_path}': {e}")
        return None


def handle_sync(db: ImageDatabase, args: argparse.Namespace):
    """Handles the database synchronization command."""
    logger.info(f"Loading directories to scan from '{args.config}'...")
    directories_to_scan = load_config(args.config)
    if directories_to_scan:
        logger.info("Starting database synchronization. This may take a while...")
        db.reconcile_database(directories_to_scan)
        logger.info("Synchronization complete.")
    else:
        logger.warning("No directories were loaded from the config. Nothing to do.")


def handle_search_image(db: ImageDatabase, args: argparse.Namespace):
    """Handles the search-by-image command."""
    query_path = Path(args.image_path)
    if not query_path.is_file():
        logger.error(f"Error: Image file '{args.image_path}' not found.")
        return

    logger.info(f"Searching for top {args.top_k} images similar to '{query_path.name}'...")
    results = db.search_similar_images(args.image_path, args.top_k)

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
    parser = argparse.ArgumentParser(
        description="An AI-powered command-line image management tool.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="images.db",
        help="Path to the SQLite database file.",
    )
    parser.add_argument("--config", type=str, default="config.yml", help="Path to the config.yml file.")

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

    embedder = None
    db = None

    try:
        # Lazily instantiate the embedder only for commands that need it.
        # This avoids loading the large ML model for commands like '--help'.
        if hasattr(args, "func"):
            logger.info("Initializing...")
            embedder = ImageEmbedder()

            logger.info(f"Opening database at '{args.db_path}'...")
            db = ImageDatabase(db_path=args.db_path, embedder=embedder)

            # Dispatch to the appropriate handler function
            args.func(db, args)
        else:
            parser.print_help()

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if db:
            logger.info("Closing database connection.")
            db.close()


if __name__ == "__main__":
    main()
