"""Configuration loader utilities."""
import functools
from typing import Dict, Any

import yaml

from tacticalm.utils.paths import CONFIGS_DIR

@functools.lru_cache(maxsize=None)
def load_config(config_name: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        config_name: The name of the config file without extension (e.g., 'data').

    Returns:
        A dictionary containing the configuration data.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    config_path = CONFIGS_DIR / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
        
    return dict(config_data)

def get_data_config() -> Dict[str, Any]:
    """Get the data configuration."""
    return load_config("data")

def get_features_config() -> Dict[str, Any]:
    """Get the features configuration."""
    return load_config("features")

def get_training_config() -> Dict[str, Any]:
    """Get the training configuration."""
    return load_config("training")

def get_inference_config() -> Dict[str, Any]:
    """Get the inference configuration."""
    return load_config("inference")

def get_evaluation_config() -> Dict[str, Any]:
    """Get the evaluation configuration."""
    return load_config("evaluation")
