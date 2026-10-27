"""Data validation module using Pydantic models.

Validates StatsBomb match data: events, lineups, and metadata.
Uses statsbombpy's flattened column naming convention (e.g., 'type' not 'type.name',
'shot_statsbomb_xg' not 'shot.statsbomb_xg').
"""

from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field

from tacticalm.utils.logging import get_logger

logger = get_logger(__name__)


# --- Pydantic Schemas ---


class EventLocation(BaseModel):
    """Validated pitch coordinate."""

    x: float = Field(..., ge=0.0, le=120.0, description="X coordinate (0-120 yards)")
    y: float = Field(..., ge=0.0, le=80.0, description="Y coordinate (0-80 yards)")


class MatchMetadata(BaseModel):
    """Match-level metadata."""

    match_id: int
    competition: str = ""
    season: str = ""
    home_team: str
    away_team: str
    home_score: int = 0
    away_score: int = 0
    match_date: str = ""


class ValidationIssue(BaseModel):
    """A single validation error or warning."""

    field: str
    message: str
    severity: str = "error"  # 'error' or 'warning'
    count: int = 1


class ValidationResult(BaseModel):
    """Result of match data validation."""

    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    stats: Dict[str, Any] = Field(default_factory=dict)


# --- Validation Functions ---


def validate_coordinates(events: pd.DataFrame) -> List[str]:
    """Check that event locations fall within the pitch (0-120 x 0-80).

    Returns list of warning messages for out-of-bounds locations.
    """
    warnings: List[str] = []
    if "location" not in events.columns:
        return warnings

    out_of_bounds = 0
    for loc in events["location"].dropna():
        if isinstance(loc, list) and len(loc) >= 2:
            x, y = loc[0], loc[1]
            if not (0 <= x <= 120 and 0 <= y <= 80):
                out_of_bounds += 1

    if out_of_bounds > 0:
        warnings.append(f"Found {out_of_bounds} events with out-of-bounds locations")

    return warnings


def validate_chronological_order(events: pd.DataFrame) -> List[str]:
    """Check that events are in chronological order by index.

    Returns list of error messages.
    """
    errors: List[str] = []
    if "index" in events.columns:
        if not events["index"].is_monotonic_increasing:
            errors.append("Events are not in strictly increasing index order")
    return errors


def validate_shot_events(events: pd.DataFrame) -> List[str]:
    """Validate that shot events have required fields.

    Returns list of warning messages.
    """
    warnings: List[str] = []
    if "type" not in events.columns:
        return warnings

    shots = events[events["type"] == "Shot"]
    if shots.empty:
        return warnings

    # Check shot outcomes exist
    if "shot_outcome" not in events.columns:
        warnings.append("Shot outcome column missing entirely")
    else:
        missing_outcomes = shots["shot_outcome"].isna().sum()
        if missing_outcomes > 0:
            warnings.append(f"{missing_outcomes} shots missing outcome")

    # Check xG values
    if "shot_statsbomb_xg" not in events.columns:
        warnings.append("Shot xG column (shot_statsbomb_xg) missing entirely")
    else:
        missing_xg = shots["shot_statsbomb_xg"].isna().sum()
        if missing_xg > 0:
            warnings.append(f"{missing_xg} shots missing xG values")

    return warnings


def validate_pass_events(events: pd.DataFrame) -> List[str]:
    """Validate that pass events have required fields.

    Returns list of error/warning messages.
    """
    errors: List[str] = []
    if "type" not in events.columns:
        return errors

    passes = events[events["type"] == "Pass"]
    if passes.empty:
        return errors

    if "pass_end_location" not in events.columns:
        errors.append("Pass end location column missing entirely")

    return errors


def validate_periods(events: pd.DataFrame) -> List[str]:
    """Validate that event periods are valid (1-5).

    Returns list of error messages.
    """
    errors: List[str] = []
    if "period" not in events.columns:
        return errors

    valid_periods = {1, 2, 3, 4, 5}
    invalid = events[~events["period"].isin(valid_periods)]
    if len(invalid) > 0:
        errors.append(f"Found {len(invalid)} events with invalid periods")

    return errors


def validate_match(
    match_id: int,
    events: pd.DataFrame,
    lineups: Dict[str, pd.DataFrame],
    metadata: Dict[str, Any],
) -> ValidationResult:
    """Validate all match data.

    Checks: match_id consistency, teams exist, lineups present,
    chronological order, coordinate bounds, shot/pass fields, periods.

    Args:
        match_id: Expected match ID.
        events: Events DataFrame (statsbombpy flattened columns).
        lineups: Dict of team name -> lineup DataFrame.
        metadata: Match metadata dict with home_team, away_team, etc.

    Returns:
        ValidationResult with is_valid flag, errors, warnings, and stats.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # --- Statistics ---
    type_col = "type" if "type" in events.columns else None
    stats: Dict[str, Any] = {
        "event_count": len(events),
        "teams_in_events": (
            sorted(events["team"].dropna().unique().tolist())
            if "team" in events.columns
            else []
        ),
    }
    if type_col:
        stats["pass_count"] = int((events[type_col] == "Pass").sum())
        stats["shot_count"] = int((events[type_col] == "Shot").sum())

    # --- 1. Match ID ---
    if str(metadata.get("match_id")) != str(match_id):
        errors.append(
            f"Metadata match_id ({metadata.get('match_id')}) "
            f"does not match expected {match_id}"
        )

    # --- 2. Teams exist ---
    home_team = metadata.get("home_team")
    away_team = metadata.get("away_team")
    if not home_team or not away_team:
        errors.append("Missing home or away team in metadata")

    # --- 3. Lineups present ---
    if home_team and home_team not in lineups:
        errors.append(f"Missing lineup for home team: {home_team}")
    if away_team and away_team not in lineups:
        errors.append(f"Missing lineup for away team: {away_team}")

    # --- 4. Chronological order ---
    errors.extend(validate_chronological_order(events))

    # --- 5. Coordinate bounds ---
    warnings.extend(validate_coordinates(events))

    # --- 6. Shot and pass validation ---
    warnings.extend(validate_shot_events(events))
    errors.extend(validate_pass_events(events))

    # --- 7. Period validation ---
    errors.extend(validate_periods(events))

    is_valid = len(errors) == 0

    if is_valid:
        logger.info(f"Match {match_id} passed validation ({len(warnings)} warnings)")
    else:
        logger.error(
            f"Match {match_id} failed validation: {len(errors)} errors, "
            f"{len(warnings)} warnings"
        )

    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        stats=stats,
    )
