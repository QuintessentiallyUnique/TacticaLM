"""Tests for local summary retrieval."""

from tacticalm.dataset.schemas import MatchSummary, TeamStats
from tacticalm.retrieval.summaries import retrieve_relevant_summaries


def test_retrieve_relevant_summaries_from_processed_json(tmp_path):
    summary = MatchSummary(
        match_id=123,
        competition="Test Cup",
        season="2024",
        match_date="2024-01-01",
        home_team="Team A",
        away_team="Team B",
        score_home=2,
        score_away=1,
        team_stats={
            "Team A": TeamStats(pressures=35, high_regains=8, ppda=6.5),
            "Team B": TeamStats(pressures=10, high_regains=1, ppda=18.0),
        },
    )
    (tmp_path / "123_summary.json").write_text(summary.model_dump_json(), encoding="utf-8")

    results = retrieve_relevant_summaries(
        query="How intense was Team A pressing?",
        summaries_dir=tmp_path,
        team="Team A",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].summary.match_id == 123
    assert any("35 pressures" in statement for statement in results[0].evidence)
