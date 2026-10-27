import json
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd

from tacticalm.utils.paths import RAW_DIR
from tacticalm.utils.logging import get_logger
from .download import (
    download_competitions,
    download_matches,
    download_and_cache_match
)

logger = get_logger(__name__)


def load_competitions() -> pd.DataFrame:
    """Load competitions, falling back to download if cache doesn't exist."""
    comps_path = RAW_DIR / "competitions" / "competitions.parquet"
    if comps_path.exists():
        logger.info(f"Loading competitions from {comps_path}")
        return pd.read_parquet(comps_path)
    
    logger.info("Competitions cache not found, downloading...")
    comps_path.parent.mkdir(parents=True, exist_ok=True)
    df = download_competitions()
    df.to_parquet(comps_path, index=False)
    return df


def load_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """Load matches, falling back to download if cache doesn't exist."""
    matches_path = RAW_DIR / "matches" / f"{competition_id}_{season_id}.parquet"
    if matches_path.exists():
        logger.info(f"Loading matches from {matches_path}")
        return pd.read_parquet(matches_path)
    
    logger.info(f"Matches cache not found for {competition_id}_{season_id}, downloading...")
    matches_path.parent.mkdir(parents=True, exist_ok=True)
    df = download_matches(competition_id, season_id)
    df.to_parquet(matches_path, index=False)
    return df


def load_match_events(match_id: int) -> pd.DataFrame:
    """Load events for a match from cached file."""
    events_path = RAW_DIR / "events" / f"{match_id}.parquet"
    if not events_path.exists():
        logger.warning(f"Events not found for match {match_id}. Triggering download.")
        download_and_cache_match(match_id)
        
    logger.info(f"Loading events for match {match_id}")
    return pd.read_parquet(events_path)


def load_match_lineups(match_id: int) -> Dict[str, pd.DataFrame]:
    """Load lineups for a match from cached file."""
    lineups_path = RAW_DIR / "lineups" / f"{match_id}.json"
    if not lineups_path.exists():
        logger.warning(f"Lineups not found for match {match_id}. Triggering download.")
        download_and_cache_match(match_id)
        
    logger.info(f"Loading lineups for match {match_id}")
    with open(lineups_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {team: pd.DataFrame(players) for team, players in data.items()}


def load_match_360(match_id: int) -> Optional[pd.DataFrame]:
    """Load 360 data for a match from cached file if available."""
    threesixty_path = RAW_DIR / "three_sixty" / f"{match_id}.parquet"
    if threesixty_path.exists():
        logger.info(f"Loading 360 data for match {match_id}")
        return pd.read_parquet(threesixty_path)
    
    logger.info(f"No cached 360 data found for match {match_id}.")
    return None


def load_match_data(match_id: int) -> Dict[str, Any]:
    """Load all data for a match into a single dictionary."""
    return {
        "events": load_match_events(match_id),
        "lineups": load_match_lineups(match_id),
        "three_sixty": load_match_360(match_id),
    }
