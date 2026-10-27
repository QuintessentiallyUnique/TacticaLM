"""Optional StatsBomb 360 spatial feature extraction.

The open-data pipeline often has no freeze-frame data, so every function in
this module is defensive: missing columns, missing frames, or malformed rows
return empty/default metrics instead of failing match processing.
"""

import math
from typing import Any

import numpy as np
import pandas as pd

_PITCH_LENGTH = 120.0
_PITCH_WIDTH = 80.0


def _as_location(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 2:
        return None
    try:
        return [float(value[0]), float(value[1])]
    except (TypeError, ValueError):
        return None


def _frame_players(frame: Any) -> list[dict[str, Any]]:
    if isinstance(frame, np.ndarray):
        frame = frame.tolist()
    if not isinstance(frame, list):
        return []
    return [player for player in frame if isinstance(player, dict)]


def _event_id_column(three_sixty: pd.DataFrame) -> str | None:
    for col in ("event_uuid", "id", "event_id"):
        if col in three_sixty.columns:
            return col
    return None


def _freeze_frame_by_event(three_sixty: pd.DataFrame) -> dict[str, Any]:
    id_col = _event_id_column(three_sixty)
    if id_col is None or "freeze_frame" not in three_sixty.columns:
        return {}
    return {
        str(row[id_col]): row["freeze_frame"]
        for _, row in three_sixty.iterrows()
        if pd.notna(row.get(id_col))
    }


def _nearest_distance(origin: list[float], players: list[dict[str, Any]]) -> float | None:
    distances = []
    for player in players:
        loc = _as_location(player.get("location"))
        if loc is not None:
            distances.append(math.dist(origin, loc))
    return min(distances) if distances else None


def _compactness(players: list[dict[str, Any]]) -> float | None:
    locations = [
        loc
        for player in players
        if (loc := _as_location(player.get("location"))) is not None
    ]
    if len(locations) < 2:
        return None

    distances = []
    for idx, loc_a in enumerate(locations[:-1]):
        for loc_b in locations[idx + 1 :]:
            distances.append(math.dist(loc_a, loc_b))
    return float(np.mean(distances)) if distances else None


def _pitch_control_share(
    teammates: list[dict[str, Any]],
    opponents: list[dict[str, Any]],
    grid_x: int = 12,
    grid_y: int = 8,
) -> float | None:
    teammate_locations = [
        loc
        for player in teammates
        if (loc := _as_location(player.get("location"))) is not None
    ]
    opponent_locations = [
        loc
        for player in opponents
        if (loc := _as_location(player.get("location"))) is not None
    ]
    if not teammate_locations or not opponent_locations:
        return None

    controlled = 0
    total = 0
    xs = np.linspace(0, _PITCH_LENGTH, grid_x)
    ys = np.linspace(0, _PITCH_WIDTH, grid_y)
    for x in xs:
        for y in ys:
            point = [float(x), float(y)]
            nearest_teammate = min(math.dist(point, loc) for loc in teammate_locations)
            nearest_opponent = min(math.dist(point, loc) for loc in opponent_locations)
            controlled += int(nearest_teammate <= nearest_opponent)
            total += 1

    return round(controlled / total * 100, 1) if total else None


def _line_breaking_pass(row: pd.Series, opponents: list[dict[str, Any]]) -> bool:
    if row.get("type") != "Pass" or pd.notna(row.get("pass_outcome")):
        return False

    start = _as_location(row.get("location"))
    end = _as_location(row.get("pass_end_location"))
    if start is None or end is None:
        return False
    if end[0] - start[0] < 10:
        return False

    x_min, x_max = sorted([start[0], end[0]])
    y_min, y_max = sorted([start[1], end[1]])
    corridor_y_min = y_min - 8
    corridor_y_max = y_max + 8

    for opponent in opponents:
        loc = _as_location(opponent.get("location"))
        if loc is None:
            continue
        if x_min < loc[0] < x_max and corridor_y_min <= loc[1] <= corridor_y_max:
            return True
    return False


def compute_360_team_metrics(
    events: pd.DataFrame,
    three_sixty: pd.DataFrame | None,
    team: str,
) -> dict[str, float | int | None]:
    """Compute team spatial metrics from StatsBomb 360 freeze frames."""
    defaults = {
        "line_breaking_passes": 0,
        "visible_teammate_count_avg": None,
        "visible_opponent_count_avg": None,
        "avg_nearest_opponent_distance": None,
        "pitch_control_share": None,
        "visible_opponent_compactness": None,
    }
    if three_sixty is None or three_sixty.empty or "id" not in events.columns:
        return defaults

    frames_by_event = _freeze_frame_by_event(three_sixty)
    if not frames_by_event:
        return defaults

    teammate_counts = []
    opponent_counts = []
    nearest_opponent_distances = []
    pitch_control_shares = []
    opponent_compactness = []
    line_breaking_passes = 0

    team_events = events[events["team"] == team]
    for _, row in team_events.iterrows():
        frame = _frame_players(frames_by_event.get(str(row.get("id"))))
        if not frame:
            continue

        teammates = [player for player in frame if player.get("teammate") is True]
        opponents = [player for player in frame if player.get("teammate") is False]
        teammate_counts.append(len(teammates))
        opponent_counts.append(len(opponents))

        origin = _as_location(row.get("location"))
        if origin is not None:
            nearest = _nearest_distance(origin, opponents)
            if nearest is not None:
                nearest_opponent_distances.append(nearest)

        control_share = _pitch_control_share(teammates, opponents)
        if control_share is not None:
            pitch_control_shares.append(control_share)

        compactness = _compactness(opponents)
        if compactness is not None:
            opponent_compactness.append(compactness)

        if _line_breaking_pass(row, opponents):
            line_breaking_passes += 1

    if not teammate_counts and not opponent_counts:
        return defaults

    return {
        "line_breaking_passes": line_breaking_passes,
        "visible_teammate_count_avg": round(float(np.mean(teammate_counts)), 1)
        if teammate_counts
        else None,
        "visible_opponent_count_avg": round(float(np.mean(opponent_counts)), 1)
        if opponent_counts
        else None,
        "avg_nearest_opponent_distance": round(float(np.mean(nearest_opponent_distances)), 1)
        if nearest_opponent_distances
        else None,
        "pitch_control_share": round(float(np.mean(pitch_control_shares)), 1)
        if pitch_control_shares
        else None,
        "visible_opponent_compactness": round(float(np.mean(opponent_compactness)), 1)
        if opponent_compactness
        else None,
    }
