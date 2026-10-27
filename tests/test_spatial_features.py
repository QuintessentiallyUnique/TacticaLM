"""Tests for optional 360 spatial metrics."""

import numpy as np
import pandas as pd

from tacticalm.features.spatial import compute_360_team_metrics


def test_compute_360_team_metrics_defaults_without_data():
    events = pd.DataFrame([{"id": "evt-1", "team": "Team A", "type": "Pass"}])

    metrics = compute_360_team_metrics(events, None, "Team A")

    assert metrics["line_breaking_passes"] == 0
    assert metrics["pitch_control_share"] is None


def test_compute_360_team_metrics_counts_line_breaking_pass():
    events = pd.DataFrame(
        [
            {
                "id": "evt-1",
                "team": "Team A",
                "type": "Pass",
                "location": [40, 40],
                "pass_end_location": [75, 40],
                "pass_outcome": np.nan,
            }
        ]
    )
    three_sixty = pd.DataFrame(
        [
            {
                "event_uuid": "evt-1",
                "freeze_frame": [
                    {"teammate": True, "location": [38, 38]},
                    {"teammate": True, "location": [45, 45]},
                    {"teammate": False, "location": [55, 40]},
                    {"teammate": False, "location": [60, 50]},
                ],
            }
        ]
    )

    metrics = compute_360_team_metrics(events, three_sixty, "Team A")

    assert metrics["line_breaking_passes"] == 1
    assert metrics["visible_teammate_count_avg"] == 2.0
    assert metrics["visible_opponent_count_avg"] == 2.0
    assert metrics["avg_nearest_opponent_distance"] == 15.0
    assert metrics["pitch_control_share"] is not None
