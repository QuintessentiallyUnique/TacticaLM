"""Path utilities for the TacticaLM project."""
import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = Path(os.getenv("TACTICALM_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EVALUATION_DIR = DATA_DIR / "evaluation"

MODELS_DIR = Path(os.getenv("TACTICALM_MODELS_DIR", PROJECT_ROOT / "models"))
ADAPTERS_DIR = MODELS_DIR / "adapters"

CONFIGS_DIR = PROJECT_ROOT / "configs"
REPORTS_DIR = PROJECT_ROOT / "reports"

def ensure_dir(path: Path) -> Path:
    """
    Ensure the given directory exists.

    Args:
        path: Path to the directory.

    Returns:
        The path to the created directory.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_match_data_path(match_id: int, data_type: str) -> Path:
    """
    Get the file path for a specific match and data type.

    Args:
        match_id: The ID of the match.
        data_type: The type of data (e.g., 'events', '360', 'lineups').

    Returns:
        The path to the match data file.
    """
    filename = f"{match_id}_{data_type}.json"
    return RAW_DIR / filename
