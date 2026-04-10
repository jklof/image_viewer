import yaml
import logging

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

class _ConfigStore:
    _cache: dict | None = None

    @classmethod
    def get(cls, config_path: str, force_reload=False) -> dict:
        if cls._cache is None or force_reload:
            cls._cache = cls._load_from_disk(config_path)
        return cls._cache.copy()

    @classmethod
    def set(cls, config: dict):
        cls._cache = config.copy()

    @classmethod
    def _load_from_disk(cls, config_path: str) -> dict:
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
            # Automatically create the default config if missing
            try:
                with open(config_path, "w") as f:
                    yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False, sort_keys=False)
            except IOError as e:
                logger.error(f"Could not initialize default config '{config_path}': {e}")
            return DEFAULT_CONFIG.copy()
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file '{config_path}': {e}")
            return DEFAULT_CONFIG.copy()


def load_config(config_path: str = DEFAULT_CONFIG_PATH, force_reload: bool = False) -> dict:
    """Loads the configuration from a YAML file, applying defaults for missing keys."""
    return _ConfigStore.get(config_path, force_reload)


def save_config(config: dict, config_path: str = DEFAULT_CONFIG_PATH):
    """Saves the configuration to a YAML file."""
    try:
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        _ConfigStore.set(config)
        logger.info(f"Configuration saved to '{config_path}'.")
    except IOError as e:
        logger.error(f"Error saving configuration to '{config_path}': {e}")

def get_scan_directories(config_path: str = DEFAULT_CONFIG_PATH) -> list[str]:
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
