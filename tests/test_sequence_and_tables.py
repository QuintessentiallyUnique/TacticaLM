"""Tests for sequence extraction and table-aware dataset context."""

from tacticalm.dataset.generate_examples import format_summary_tables
from tacticalm.dataset.schemas import IntervalStats, MatchSummary, PossessionSequence, TeamStats
from tacticalm.features.engine import FeatureEngine
from tests.test_features import create_synthetic_match


def test_extract_high_value_sequences_describes_shot_possessions():
    events = create_synthetic_match()
    engine = FeatureEngine(config={"features": {}})

    sequences = engine.extract_high_value_sequences(events)

    assert sequences
    assert sequences[0]["expected_goals"] > 0
    assert "shot" in sequences[0]["sequence_text"].lower()


def test_format_summary_tables_includes_team_and_momentum_tables():
    summary = MatchSummary(
        match_id=123,
        home_team="Team A",
        away_team="Team B",
        team_stats={
            "Team A": TeamStats(possession_percentage=55.5, total_passes=500, shots=12),
            "Team B": TeamStats(possession_percentage=44.5, total_passes=400, shots=8),
        },
        momentum_timeline=[
            IntervalStats(
                team="Team A",
                start_minute=0,
                end_minute=14,
                possession_pct=60.0,
                expected_goals=0.2,
                shots=2,
            )
        ],
        high_value_sequences=[
            PossessionSequence(
                possession_id=7,
                team="Team A",
                start_minute=10,
                end_minute=11,
                outcome="shot",
                expected_goals=0.2,
                sequence_text="10' Player A passed -> 11' Player B shot",
            )
        ],
    )

    tables = format_summary_tables(summary, ["Team A"])

    assert any("| Team | Possession %" in table for table in tables)
    assert any("| Team A | 0-14 | 60.0 | 0.20 | 2 |" in table for table in tables)
    assert any("High-value sequence table" in table for table in tables)
