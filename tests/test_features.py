"""Tests for the feature engineering engine.

Uses synthetic match data with known correct values for every metric.
The synthetic DataFrame mimics statsbombpy's flattened column format.
"""

import uuid

import numpy as np
import pandas as pd
import pytest

from tacticalm.features.engine import FeatureEngine
from tacticalm.features.summary import build_match_summary, generate_evidence_statements


def create_synthetic_match() -> pd.DataFrame:
    """Create a minimal but complete set of synthetic events for testing.

    Known facts about this synthetic match:
    - Team A: 10 passes (8 completed, 2 incomplete), 2 progressive passes
    - Team B: 5 passes (4 completed, 1 incomplete)
    - Team A: 3 shots (xG: 0.4, 0.1, 0.05 -> total 0.55), 1 goal, 2 on target
    - Team B: 2 shots (xG: 0.3, 0.2 -> total 0.5), 0 goals
    - Team A: 1 Ball Recovery in attacking third (x=90) -> 1 high regain
    - Team A: 2 Pressure events
    - Team A: 1 Carry entering final third (70 -> 85)
    - Team A: 3 possession sequences, Team B: 2 possession sequences
    """
    rows = []
    idx = [0]  # Mutable counter for sequential index

    def make_event(
        type_name: str,
        team: str,
        loc: list,
        poss: int,
        poss_team: str,
        pattern: str = "Regular Play",
        **kwargs,
    ) -> dict:
        idx[0] += 1
        base = {
            "id": str(uuid.uuid4()),
            "index": idx[0],
            "period": 1,
            "timestamp": f"00:{idx[0]:02d}:00.000",
            "minute": idx[0],
            "second": 0,
            "type": type_name,
            "team": team,
            "player": f"Player_{team}_{idx[0]}",
            "location": loc,
            "possession": poss,
            "possession_team": poss_team,
            "play_pattern": pattern,
        }
        base.update(kwargs)
        return base

    # --- Team A Passes: 10 total (8 completed, 2 incomplete) ---
    # Progressive pass 1: [50,40] -> [70,40]
    #   dist to [120,40]: sqrt(70^2) = 70 -> sqrt(50^2) = 50, diff = 20
    rows.append(make_event(
        "Pass", "Team A", [50, 40], 1, "Team A",
        pass_end_location=[70, 40], pass_outcome=np.nan,
    ))
    # Progressive pass 2: [60,40] -> [100,40]
    #   dist to [120,40]: sqrt(60^2) = 60 -> sqrt(20^2) = 20, diff = 40
    rows.append(make_event(
        "Pass", "Team A", [60, 40], 1, "Team A",
        pass_end_location=[100, 40], pass_outcome=np.nan,
    ))
    # 6 more completed, non-progressive passes ([60,40] -> [62,40], diff about 2)
    for _ in range(6):
        rows.append(make_event(
            "Pass", "Team A", [60, 40], 1, "Team A",
            pass_end_location=[62, 40], pass_outcome=np.nan,
        ))
    # 2 incomplete passes
    for _ in range(2):
        rows.append(make_event(
            "Pass", "Team A", [60, 40], 1, "Team A",
            pass_end_location=[62, 40], pass_outcome="Incomplete",
        ))

    # --- Team B Passes: 5 total (4 completed, 1 incomplete) ---
    for _ in range(4):
        rows.append(make_event(
            "Pass", "Team B", [60, 40], 2, "Team B",
            pass_end_location=[62, 40], pass_outcome=np.nan,
        ))
    rows.append(make_event(
        "Pass", "Team B", [60, 40], 2, "Team B",
        pass_end_location=[62, 40], pass_outcome="Incomplete",
    ))

    # --- Team A Shots: 3 shots (xG: 0.4 Goal, 0.1 Saved, 0.05 Off T) ---
    rows.append(make_event(
        "Shot", "Team A", [110, 40], 1, "Team A",
        shot_statsbomb_xg=0.4, shot_outcome="Goal", shot_type="Open Play",
    ))
    rows.append(make_event(
        "Shot", "Team A", [100, 40], 1, "Team A",
        shot_statsbomb_xg=0.1, shot_outcome="Saved", shot_type="Open Play",
    ))
    rows.append(make_event(
        "Shot", "Team A", [95, 40], 3, "Team A",
        shot_statsbomb_xg=0.05, shot_outcome="Off T", shot_type="Open Play",
    ))

    # --- Team B Shots: 2 shots (xG: 0.3 Saved, 0.2 Off T) ---
    rows.append(make_event(
        "Shot", "Team B", [110, 40], 2, "Team B",
        shot_statsbomb_xg=0.3, shot_outcome="Saved", shot_type="Open Play",
    ))
    rows.append(make_event(
        "Shot", "Team B", [100, 40], 2, "Team B",
        shot_statsbomb_xg=0.2, shot_outcome="Off T", shot_type="Open Play",
    ))

    # --- Team A Ball Recovery in attacking third (x=90) ---
    rows.append(make_event(
        "Ball Recovery", "Team A", [90, 40], 3, "Team A",
    ))

    # --- Team A Pressure events ---
    rows.append(make_event(
        "Pressure", "Team A", [70, 40], 2, "Team B",
    ))
    rows.append(make_event(
        "Pressure", "Team A", [85, 40], 2, "Team B",
    ))

    # --- Team A Carry entering final third (70 -> 85) ---
    rows.append(make_event(
        "Carry", "Team A", [70, 40], 3, "Team A",
        carry_end_location=[85, 40],
    ))

    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_events():
    """Fixture providing synthetic match events."""
    return create_synthetic_match()


@pytest.fixture
def engine():
    """Fixture providing a FeatureEngine with default config."""
    # Use empty config to avoid needing the YAML file during tests
    return FeatureEngine(config={"features": {}})


# --- Possession Tests ---


def test_possession_calculation(synthetic_events, engine):
    """Team A possession % should match event count ratio."""
    features = engine.compute_possession(synthetic_events, "Team A")
    team_a_events = synthetic_events[
        synthetic_events["possession_team"] == "Team A"
    ]
    expected_pct = round(len(team_a_events) / len(synthetic_events) * 100, 1)
    assert features["possession_percentage"] == expected_pct


def test_possession_count(synthetic_events, engine):
    """Team A should have possession sequences 1 and 3."""
    features = engine.compute_possession(synthetic_events, "Team A")
    assert features["possession_count"] == 2  # possessions 1 and 3


# --- Passing Tests ---


def test_total_passes(synthetic_events, engine):
    """Team A: 10 passes total, 8 completed."""
    features = engine.compute_passing(synthetic_events, "Team A")
    assert features["total_passes"] == 10
    assert features["completed_passes"] == 8


def test_pass_completion_rate(synthetic_events, engine):
    """Team A: 8/10 = 80.0% completion rate."""
    features = engine.compute_passing(synthetic_events, "Team A")
    assert features["pass_completion_percentage"] == 80.0


def test_progressive_passes(synthetic_events, engine):
    """Team A has exactly 2 progressive passes (both move >= 10 yards closer)."""
    features = engine.compute_passing(synthetic_events, "Team A")
    assert features["progressive_passes"] == 2


def test_team_b_passes(synthetic_events, engine):
    """Team B: 5 passes total, 4 completed."""
    features = engine.compute_passing(synthetic_events, "Team B")
    assert features["total_passes"] == 5
    assert features["completed_passes"] == 4


# --- Shooting Tests ---


def test_shot_totals(synthetic_events, engine):
    """Team A: 3 shots, 1 goal, 2 on target (Goal + Saved)."""
    features = engine.compute_shooting(synthetic_events, "Team A")
    assert features["shots"] == 3
    assert features["goals"] == 1
    assert features["shots_on_target"] == 2


def test_expected_goals(synthetic_events, engine):
    """Team A: xG = 0.4 + 0.1 + 0.05 = 0.55."""
    features = engine.compute_shooting(synthetic_events, "Team A")
    assert abs(features["expected_goals"] - 0.55) < 1e-4


def test_high_xg_chances(synthetic_events, engine):
    """Team A: 1 high-xG chance (the 0.4 shot >= 0.3 threshold)."""
    features = engine.compute_shooting(synthetic_events, "Team A")
    assert features["high_xg_chances"] == 1


def test_team_b_xg(synthetic_events, engine):
    """Team B: xG = 0.3 + 0.2 = 0.5, 1 high-xG chance."""
    features = engine.compute_shooting(synthetic_events, "Team B")
    assert abs(features["expected_goals"] - 0.5) < 1e-4
    assert features["high_xg_chances"] == 1


# --- Progression Tests ---


def test_final_third_entries(synthetic_events, engine):
    """Team A: Carry [70->85] crosses x=80 = 1 final-third entry.
    Passes: [50->70] doesn't cross 80. [60->100] starts < 80 and ends >= 80 = 1.
    But [60->62] doesn't cross. Total from passes: 1 (the [60->100]).
    Total = 1 (pass) + 1 (carry) = 2."""
    features = engine.compute_progression(synthetic_events, "Team A")
    assert features["final_third_entries"] == 2


def test_progressive_carries(synthetic_events, engine):
    """Team A: Carry [70,40]->[85,40] moves ~15 yards closer to goal. Progressive."""
    features = engine.compute_progression(synthetic_events, "Team A")
    assert features["progressive_carries"] == 1


# --- Pressing Tests ---


def test_high_regains(synthetic_events, engine):
    """Team A: 1 Ball Recovery at x=90 (attacking third). 1 high regain."""
    features = engine.compute_pressing(synthetic_events, "Team A", "Team B")
    assert features["high_regains"] == 1


def test_pressure_count(synthetic_events, engine):
    """Team A: 2 pressure events total."""
    features = engine.compute_pressing(synthetic_events, "Team A", "Team B")
    assert features["pressures"] == 2


def test_pressures_by_zone(synthetic_events, engine):
    """Team A pressures: one at x=70 (middle), one at x=85 (attacking)."""
    features = engine.compute_pressing(synthetic_events, "Team A", "Team B")
    assert features["pressures_attacking_third"] == 1
    assert features["pressures_middle_third"] == 1


# --- Evidence Generation Tests ---


def test_evidence_generation(synthetic_events, engine):
    """Evidence statements should contain team name and key statistics."""
    summary = build_match_summary(
        match_id=1,
        events=synthetic_events,
        lineups={},
        match_metadata={"home_team": "Team A", "away_team": "Team B"},
    )
    statements = generate_evidence_statements(summary, "Team A")

    assert len(statements) > 10, "Should generate at least 10 evidence statements"
    assert any("Team A" in s for s in statements)
    assert any("80.0%" in s for s in statements)  # pass completion
    assert any("2 progressive passes" in s for s in statements)
    assert any("3 shots" in s for s in statements)
    assert any("0.55" in s for s in statements)  # xG


def test_evidence_no_unknown_team(synthetic_events, engine):
    """Evidence for a non-existent team should return empty list."""
    summary = build_match_summary(
        match_id=1,
        events=synthetic_events,
        lineups={},
        match_metadata={"home_team": "Team A", "away_team": "Team B"},
    )
    statements = generate_evidence_statements(summary, "Team C")
    assert statements == []


# --- Compute All Features Integration Test ---


def test_compute_all_features(synthetic_events, engine):
    """Integration test: compute_all_features returns nested dict for both teams."""
    features = engine.compute_all_features(
        synthetic_events,
        lineups={},
        match_metadata={"home_team": "Team A", "away_team": "Team B"},
    )
    assert "teams" in features
    assert "players" in features
    assert "Team A" in features["teams"]
    assert "Team B" in features["teams"]

    # Spot-check Team A features
    ta = features["teams"]["Team A"]
    assert ta["total_passes"] == 10
    assert ta["completed_passes"] == 8
    assert ta["shots"] == 3
    assert ta["high_regains"] == 1
