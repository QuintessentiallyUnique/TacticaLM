import json
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
from statsbombpy import sb

from tacticalm.utils.paths import RAW_DIR, ensure_dir
from tacticalm.utils.logging import get_logger

logger = get_logger(__name__)


def download_competitions() -> pd.DataFrame:
    """Fetch competitions from statsbombpy."""
    try:
        logger.info("Downloading competitions from StatsBomb")
        return sb.competitions()
    except Exception as e:
        logger.error(f"Failed to download competitions: {e}")
        raise


def download_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """Fetch matches for a specific competition and season."""
    try:
        logger.info(f"Downloading matches for competition {competition_id}, season {season_id}")
        return sb.matches(competition_id=competition_id, season_id=season_id)
    except Exception as e:
        logger.error(f"Failed to download matches: {e}")
        raise


def download_match_events(match_id: int) -> pd.DataFrame:
    """Fetch events for a specific match."""
    try:
        logger.info(f"Downloading events for match {match_id}")
        return sb.events(match_id=match_id)
    except Exception as e:
        logger.error(f"Failed to download events for match {match_id}: {e}")
        raise


def download_match_lineups(match_id: int) -> Dict[str, pd.DataFrame]:
    """Fetch lineups for a specific match."""
    try:
        logger.info(f"Downloading lineups for match {match_id}")
        return sb.lineups(match_id=match_id)
    except Exception as e:
        logger.error(f"Failed to download lineups for match {match_id}: {e}")
        raise


def download_match_360(match_id: int) -> Optional[pd.DataFrame]:
    """Fetch 360 data for a specific match. Returns None if unavailable."""
    try:
        logger.info(f"Downloading 360 data for match {match_id}")
        return sb.three_sixty(match_id=match_id)
    except Exception as e:
        logger.warning(f"360 data not available for match {match_id}: {e}")
        return None


def download_and_cache_match(match_id: int, force: bool = False) -> Dict[str, Path]:
    """
    Download all data for a match and save to data/raw/.
    Returns paths to saved files. Caches to avoid re-downloading.
    """
    out_paths = {}
    
    events_dir = RAW_DIR / "events"
    lineups_dir = RAW_DIR / "lineups"
    threesixty_dir = RAW_DIR / "three_sixty"
    
    ensure_dir(events_dir)
    ensure_dir(lineups_dir)
    ensure_dir(threesixty_dir)
    
    events_path = events_dir / f"{match_id}.parquet"
    lineups_path = lineups_dir / f"{match_id}.json"
    threesixty_path = threesixty_dir / f"{match_id}.parquet"
    
    # Events
    if force or not events_path.exists():
        events_df = download_match_events(match_id)
        events_df.to_parquet(events_path, index=False)
        logger.info(f"Saved events for match {match_id} to {events_path}")
    out_paths["events"] = events_path
    
    # Lineups
    if force or not lineups_path.exists():
        lineups = download_match_lineups(match_id)
        lineups_dict = {team: df.to_dict(orient="records") for team, df in lineups.items()}
        with open(lineups_path, "w", encoding="utf-8") as f:
            json.dump(lineups_dict, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved lineups for match {match_id} to {lineups_path}")
    out_paths["lineups"] = lineups_path
    
    # 360 data
    if force or not threesixty_path.exists():
        threesixty_df = download_match_360(match_id)
        if threesixty_df is not None:
            threesixty_df.to_parquet(threesixty_path, index=False)
            logger.info(f"Saved 360 data for match {match_id} to {threesixty_path}")
            out_paths["three_sixty"] = threesixty_path
        else:
            out_paths["three_sixty"] = None
    else:
        out_paths["three_sixty"] = threesixty_path
        
    return out_paths


def download_competition_matches(competition_id: int, season_id: int, max_matches: Optional[int] = None) -> List[int]:
    """Download all matches for a given competition and season, and cache them."""
    matches_df = download_matches(competition_id, season_id)
    match_ids = matches_df["match_id"].tolist()
    
    if max_matches is not None:
        match_ids = match_ids[:max_matches]
        
    logger.info(f"Found {len(match_ids)} matches for competition {competition_id}, season {season_id}")
    
    for match_id in match_ids:
        download_and_cache_match(match_id)
        
    return match_ids
