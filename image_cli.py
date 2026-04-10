import argparse
import logging
from config_utils import get_scan_directories, get_db_path, get_model_id
from image_db import ImageDatabase
from ml_core import ImageEmbedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_sync():
    db_path = get_db_path()
    model_id = get_model_id()
    dirs = get_scan_directories()
    
    if not dirs:
        logger.error("No scan directories configured in config.yml. Please add directories to scan.")
        return
        
    logger.info(f"Loading embedding model '{model_id}'...")
    try:
        embedder = ImageEmbedder(model_id=model_id)
        db = ImageDatabase(db_path=db_path, embedder=embedder)
        
        logger.info(f"Starting database sync for {len(dirs)} directories...")
        db.reconcile_database(dirs, status_callback=lambda msg: logger.info(f"Status: {msg}"))
        logger.info("Sync complete.")
    except Exception as e:
        logger.error(f"Sync failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Viewer Command Line Interface")
    parser.add_argument("command", choices=["sync"], help="Command to execute (e.g. sync)")
    
    args = parser.parse_args()
    
    if args.command == "sync":
        run_sync()
