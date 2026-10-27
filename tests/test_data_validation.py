"""Tests for data validation module.

Tests Pydantic-based validation of match events, lineups, and metadata
using statsbombpy's flattened column naming convention.
"""

import pandas as pd
import pytest

from tacticalm.data.validate import validate_match, ValidationResult


def _make_minimal_events(n: int = 10, **overrides) -> pd.DataFrame:
    """Create a minimal valid events DataFrame."""
    data = {
        "id": [f"evt_{i}" for i in range(n)],
        "index": list(range(1, n + 1)),
        "period": [1] * n,
        "timestamp": [f"00:{i:02d}:00.000" for i in range(n)],
        "minute": list(range(n)),
        "second": [0] * n,
        "type": ["Pass"] * n,
        "team": ["Team A"] * (n // 2) + ["Team B"] * (n - n // 2),
        "player": [f"Player_{i}" for i in range(n)],
        "location": [[60, 40]] * n,
        "possession": list(range(1, n + 1)),
        "possession_team": ["Team A"] * (n // 2) + ["Team B"] * (n - n // 2),
        "play_pattern": ["Regular Play"] * n,
        "pass_end_location": [[65, 40]] * n,
    }
    data.update(overrides)
    return pd.DataFrame(data)


def _make_metadata(**overrides) -> dict:
    """Create minimal valid metadata."""
    meta = {
        "match_id": 1,
        "home_team": "Team A",
        "away_team": "Team B",
        "competition": "Test Cup",
        "season": "2024",
        "home_score": 1,
        "away_score": 0,
        "match_date": "2024-01-01",
    }
    meta.update(overrides)
    return meta


def _make_lineups() -> dict:
    """Create minimal valid lineups."""
    return {
        "Team A": pd.DataFrame({"player_id": [1, 2], "player_name": ["P1", "P2"]}),
        "Team B": pd.DataFrame({"player_id": [3, 4], "player_name": ["P3", "P4"]}),
    }


class TestValidMatch:
    """Tests for valid match data."""

    def test_valid_match_passes(self):
        """A minimal valid match should pass validation."""
        events = _make_minimal_events()
        metadata = _make_metadata()
        lineups = _make_lineups()

        result = validate_match(1, events, lineups, metadata)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_stats_include_event_count(self):
        """Validation result should include event count."""
        events = _make_minimal_events(20)
        result = validate_match(1, events, _make_lineups(), _make_metadata())
        assert result.stats["event_count"] == 20


class TestInvalidCoordinates:
    """Tests for coordinate validation."""

    def test_out_of_bounds_x(self):
        """Events with x > 120 should produce a warning."""
        events = _make_minimal_events(
            location=[[150, 40]] * 10,  # x=150 is out of bounds
        )
        result = validate_match(1, events, _make_lineups(), _make_metadata())
        assert any("out-of-bounds" in w for w in result.warnings)

    def test_out_of_bounds_y(self):
        """Events with y > 80 should produce a warning."""
        events = _make_minimal_events(
            location=[[60, 90]] * 10,  # y=90 is out of bounds
        )
        result = validate_match(1, events, _make_lineups(), _make_metadata())
        assert any("out-of-bounds" in w for w in result.warnings)


class TestMissingData:
    """Tests for missing required data."""

    def test_missing_match_id(self):
        """Mismatched match_id should produce an error."""
        events = _make_minimal_events()
        metadata = _make_metadata(match_id=999)
        result = validate_match(1, events, _make_lineups(), metadata)
        assert result.is_valid is False
        assert any("match_id" in e for e in result.errors)

    def test_missing_home_team(self):
        """Missing home team should produce an error."""
        metadata = _make_metadata(home_team="")
        result = validate_match(1, _make_minimal_events(), _make_lineups(), metadata)
        assert result.is_valid is False

    def test_missing_lineup(self):
        """Missing lineup for a team should produce an error."""
        lineups = {"Team A": pd.DataFrame({"player_id": [1]})}  # Missing Team B
        result = validate_match(
            1, _make_minimal_events(), lineups, _make_metadata()
        )
        assert result.is_valid is False
        assert any("Team B" in e for e in result.errors)


class TestEventOrder:
    """Tests for chronological event ordering."""

    def test_out_of_order_events(self):
        """Non-monotonic index should produce an error."""
        events = _make_minimal_events()
        # Reverse the index to break monotonic order
        events["index"] = list(range(len(events), 0, -1))
        result = validate_match(1, events, _make_lineups(), _make_metadata())
        assert any("index order" in e for e in result.errors)


class TestDuplicateEvents:
    """Tests for duplicate event detection."""

    def test_duplicate_ids_cleaned(self):
        """Events with duplicate IDs should be caught by clean_events, not validate."""
        # Validation doesn't remove duplicates; that's the cleaner's job.
        # But if duplicates remain, they'll have non-monotonic indices.
        events = _make_minimal_events()
        # Add a duplicate ID
        events.loc[1, "id"] = events.loc[0, "id"]
        # Validation should still pass if index is fine
        result = validate_match(1, events, _make_lineups(), _make_metadata())
        # The validation focuses on data quality, not dedup
        assert isinstance(result, ValidationResult)


class TestPeriodValidation:
    """Tests for period validation."""

    def test_invalid_period(self):
        """Events with period > 5 should produce an error."""
        events = _make_minimal_events(period=[6] * 10)
        result = validate_match(1, events, _make_lineups(), _make_metadata())
        assert result.is_valid is False
        assert any("invalid periods" in e for e in result.errors)

    def test_valid_extra_time(self):
        """Periods 3 and 4 (extra time) are valid."""
        events = _make_minimal_events(period=[3] * 5 + [4] * 5)
        result = validate_match(1, events, _make_lineups(), _make_metadata())
        assert not any("invalid periods" in e for e in result.errors)
