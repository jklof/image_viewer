# image_cli.py

import argparse
import logging
from pathlib import Path
import yaml

# Updated local imports
from image_db import ImageDatabase
from ml_core import ImageEmbedder

# Setup basic logging for the CLI
logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path="config.yml"):
    """Loads the directories to scan from the YAML configuration file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            if "directories" in config and isinstance(config["directories"], list):
                return config["directories"]
            else:
                print(f"Error: '{config_path}' is missing the 'directories' list.")
                return None
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        print("Please create it and add the directories you want to scan.")
        return None


def main():
    """Main function to handle command-line arguments."""
    parser = argparse.ArgumentParser(
        description="An AI-powered command-line image management tool.", formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--db-path", type=str, default="images.db", help="Path to the SQLite database file.")
    parser.add_argument("--config", type=str, default="config.yml", help="Path to the config.yml file.")
    parser.add_argument("--model-id", type=str, default="openai/clip-vit-base-patch32", help="The CLIP model to use.")

    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- Sync command ---
    subparsers.add_parser("sync", help="Synchronize the database with configured directories.")

    # --- Search by Image command ---
    search_img_parser = subparsers.add_parser("search-image", help="Search for images similar to a given image.")
    search_img_parser.add_argument("image_path", type=str, help="Path to the query image.")
    search_img_parser.add_argument("--top-k", type=int, default=5, help="Number of similar images to find.")

    # --- Search by Text command ---
    search_text_parser = subparsers.add_parser("search-text", help="Search for images matching a text description.")
    search_text_parser.add_argument("query", type=str, help="The text description to search for.")
    search_text_parser.add_argument("--top-k", type=int, default=5, help="Number of matching images to find.")

    # --- REMOVED: Visualize command ---

    args = parser.parse_args()
    
    # Lazily instantiate the embedder only for commands that need it.
    embedder = None
    if args.command in ["sync", "search-image", "search-text"]:
        embedder = ImageEmbedder()
        
    db = ImageDatabase(db_path=args.db_path, embedder=embedder)

    try:
        if args.command == "sync":
            directories_to_scan = load_config(args.config)
            if directories_to_scan:
                db.reconcile_database(directories_to_scan)

        elif args.command == "search-image":
            if not Path(args.image_path).is_file():
                print(f"Error: Image file '{args.image_path}' not found.")
            else:
                results = db.search_similar_images(args.image_path, args.top_k)
                print(f"\n--- Top {len(results)} images similar to '{Path(args.image_path).name}' ---")
                for score, path in results:
                    print(f"Similarity: {score:.4f}\tPath: {path}")

        elif args.command == "search-text":
            results = db.search_by_text(args.query, args.top_k)
            print(f"\n--- Top {len(results)} images matching '{args.query}' ---")
            for score, path in results:
                print(f"Relevance: {score:.4f}\tPath: {path}")
        

    finally:
        print("\nClosing database connection.")
        db.close()


if __name__ == "__main__":
    main()