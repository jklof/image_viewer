import yaml
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "config.yml"
DEFAULT_DB_PATH = "images.db"
DEFAULT_DIRECTORIES = []  # Start with an empty list
DEFAULT_MODEL_ID = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"  # A good default (768 dim)

# The canonical structure of a default config file
DEFAULT_CONFIG = {
    "database_path": DEFAULT_DB_PATH,
    "directories": DEFAULT_DIRECTORIES,
    "model_id": DEFAULT_MODEL_ID,
}


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> Dict:
    """Loads the configuration from a YAML file, applying defaults for missing keys."""
    try:
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f)
            if not isinstance(user_config, dict):
                logger.warning(f"Config file '{config_path}' is malformed. Using a default.")
                return DEFAULT_CONFIG.copy()

            # Merge user config with defaults, ensuring all keys are present
            final_config = DEFAULT_CONFIG.copy()
            final_config.update(user_config)
            return final_config

    except FileNotFoundError:
        logger.info(f"Configuration file '{config_path}' not found. Creating a default one.")
        save_config(DEFAULT_CONFIG, config_path)
        return DEFAULT_CONFIG.copy()
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file '{config_path}': {e}")
        return DEFAULT_CONFIG.copy()


def save_config(config: Dict, config_path: str = DEFAULT_CONFIG_PATH):
    """Saves the configuration to a YAML file."""
    try:
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Configuration saved to '{config_path}'.")
    except IOError as e:
        logger.error(f"Error saving configuration to '{config_path}': {e}")


def get_scan_directories(config_path: str = DEFAULT_CONFIG_PATH) -> List[str]:
    """Convenience function to get only the list of directories to scan."""
    config = load_config(config_path)
    return config.get("directories", [])


def get_db_path(config_path: str = DEFAULT_CONFIG_PATH) -> str:
    """Convenience function to get the database path."""
    config = load_config(config_path)
    return config.get("database_path", DEFAULT_DB_PATH)


def get_model_id(config_path: str = DEFAULT_CONFIG_PATH) -> str:
    """Convenience function to get the Hugging Face model ID."""
    config = load_config(config_path)
    return config.get("model_id", DEFAULT_MODEL_ID)
