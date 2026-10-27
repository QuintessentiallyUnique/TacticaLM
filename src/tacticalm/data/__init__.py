"""Data pipeline modules for the TacticaLM project.

Handles downloading, loading, cleaning, and validating StatsBomb football event data.
"""

from .download import (
    download_competitions,
    download_matches,
    download_match_events,
    download_match_lineups,
    download_match_360,
    download_and_cache_match,
    download_competition_matches,
)
from .load import (
    load_competitions,
    load_matches,
    load_match_events,
    load_match_lineups,
    load_match_360,
    load_match_data,
)
from .clean import (
    clean_events,
    clean_lineups,
    generate_cleaning_report,
)
from .validate import (
    validate_match,
    EventLocation,
    MatchMetadata,
    ValidationResult,
)
from .split import (
    split_matches,
    save_splits,
    load_splits,
    verify_no_leakage,
)

__all__ = [
    "download_competitions",
    "download_matches",
    "download_match_events",
    "download_match_lineups",
    "download_match_360",
    "download_and_cache_match",
    "download_competition_matches",
    "load_competitions",
    "load_matches",
    "load_match_events",
    "load_match_lineups",
    "load_match_360",
    "load_match_data",
    "clean_events",
    "clean_lineups",
    "generate_cleaning_report",
    "validate_match",
    "EventLocation",
    "MatchMetadata",
    "ValidationResult",
    "split_matches",
    "save_splits",
    "load_splits",
    "verify_no_leakage",
]
