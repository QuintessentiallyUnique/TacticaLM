"""Data cleaning module for StatsBomb event data.

Handles deduplication, chronological ordering, missing value management,
coordinate validation, and period classification.

Missing-Value Rules (documented per spec section 16):
- Missing location: set to None. NOT replaced with [0,0] because [0,0] is a
  real pitch coordinate (top-left corner of the defending third).
- Missing pass_outcome (NaN): means the pass was COMPLETED. This is a StatsBomb
  convention. Do NOT fill with a string value.
- Missing shot_statsbomb_xg: set to None. NOT replaced with 0.0 because 0.0
  would conflate 'no shot' with 'a zero-probability shot'.
- Missing player: normal for team-level events (Half Start, Tactical Shift,
  Substitution framework events). Leave as NaN/None.
"""

from typing import Any, Dict, Tuple

import pandas as pd

from tacticalm.utils.logging import get_logger

logger = get_logger(__name__)


def clean_events(events: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean raw StatsBomb event data.

    Args:
        events: Raw events DataFrame from statsbombpy.

    Returns:
        Tuple of (cleaned DataFrame, cleaning report dict).
    """
    report: Dict[str, Any] = {
        "original_count": len(events),
        "removed_count": 0,
        "duplicates_removed": 0,
        "missing_locations": 0,
        "out_of_bounds_locations": 0,
        "missing_xg_on_shots": 0,
    }

    # 1. Remove duplicate events by ID
    if "id" in events.columns:
        dups = events.duplicated(subset=["id"])
        report["duplicates_removed"] = int(dups.sum())
        events = events[~dups].copy()
    else:
        events = events.copy()

    # 2. Sort chronologically using the index field (guaranteed sequential by StatsBomb)
    if "index" in events.columns:
        events = events.sort_values(by=["index"]).reset_index(drop=True)
    elif "period" in events.columns and "timestamp" in events.columns:
        events = events.sort_values(by=["period", "timestamp"]).reset_index(drop=True)

    # 3. Handle missing locations
    # Rationale: Team-level events (Half Start, Substitution) have no pitch location.
    # [0, 0] is the top-left corner of the pitch, a valid coordinate, so we must
    # NOT use it as a placeholder. Leave as NaN/None.
    if "location" in events.columns:
        missing_locs = events["location"].isna()
        report["missing_locations"] = int(missing_locs.sum())

        # Flag out-of-bounds coordinates
        def _check_bounds(loc: Any) -> bool:
            if isinstance(loc, list) and len(loc) >= 2:
                x, y = loc[0], loc[1]
                return not (0 <= x <= 120 and 0 <= y <= 80)
            return False

        out_of_bounds_mask = events["location"].apply(_check_bounds)
        report["out_of_bounds_locations"] = int(out_of_bounds_mask.sum())
        if report["out_of_bounds_locations"] > 0:
            logger.warning(
                f"Found {report['out_of_bounds_locations']} events with "
                f"out-of-bounds locations"
            )

    # 4. Handle pass_outcome
    # Rationale: In StatsBomb, a missing pass_outcome (NaN) means the pass was
    # COMPLETED. We intentionally leave it as NaN so downstream code can use
    # `pass_outcome.isna()` to identify completions. Do NOT fill with a string.
    # The feature engine's compute_passing() relies on this convention.

    # 5. Handle missing xG on shot events
    # Rationale: Missing xG means the value was not calculated or the event is
    # not a shot. Setting to 0.0 would conflate "no shot" with "zero-probability
    # shot". Leave as NaN/None.
    if "shot_statsbomb_xg" in events.columns:
        shot_mask = events["type"] == "Shot"
        missing_xg = events.loc[shot_mask, "shot_statsbomb_xg"].isna()
        report["missing_xg_on_shots"] = int(missing_xg.sum())
        if report["missing_xg_on_shots"] > 0:
            logger.warning(
                f"Found {report['missing_xg_on_shots']} shot events "
                f"with missing xG values"
            )

    # 6. Missing player
    # Rationale: Tactics and half starts don't have an associated player.
    # This is normal. We leave 'player' as NaN/None for such events.

    # 7. Add computed columns for period classification
    if "period" in events.columns:
        events["is_first_half"] = events["period"] == 1
        events["is_second_half"] = events["period"] == 2
        events["is_extra_time"] = events["period"].isin([3, 4])
        events["is_penalty_shootout"] = events["period"] == 5

    report["final_count"] = len(events)
    report["removed_count"] = report["original_count"] - report["final_count"]

    logger.info(
        f"Cleaning complete: {report['original_count']} -> {report['final_count']} events "
        f"({report['duplicates_removed']} duplicates removed)"
    )

    return events, report


def clean_lineups(
    lineups: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """Clean and process lineups data.

    Identifies starting XI players and adds an is_starter flag.

    Args:
        lineups: Dictionary of team names to lineup DataFrames.

    Returns:
        Cleaned lineup DataFrames with is_starter column.
    """
    cleaned_lineups = {}
    for team, df in lineups.items():
        if df.empty:
            cleaned_lineups[team] = df
            continue

        df = df.copy()

        # Determine starting XI based on positions metadata
        if "positions" in df.columns:
            def _is_starter(positions_list: Any) -> bool:
                if not isinstance(positions_list, list) or len(positions_list) == 0:
                    return False
                for pos in positions_list:
                    if isinstance(pos, dict) and pos.get("start_reason") == "Starting XI":
                        return True
                return False

            df["is_starter"] = df["positions"].apply(_is_starter)
        else:
            df["is_starter"] = False

        cleaned_lineups[team] = df

    return cleaned_lineups


def generate_cleaning_report(
    events_raw: pd.DataFrame, events_clean: pd.DataFrame
) -> Dict[str, Any]:
    """Generate a summary report comparing raw and cleaned events.

    Args:
        events_raw: Original raw events DataFrame.
        events_clean: Cleaned events DataFrame.

    Returns:
        Dictionary with cleaning statistics.
    """
    return {
        "raw_events": len(events_raw),
        "clean_events": len(events_clean),
        "removed_events": len(events_raw) - len(events_clean),
    }
