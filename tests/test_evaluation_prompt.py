"""Regression tests for evaluation prompt formatting."""

from scripts.run_evaluation import format_prompt
from tacticalm.dataset.schemas import EvaluationExample, MatchContext


def test_format_prompt_includes_evidence():
    example = EvaluationExample(
        example_id="example-1",
        match_id=1,
        category="match_summary",
        question="How did Team A progress the ball?",
        context=MatchContext(
            competition="Test Cup",
            season="2024",
            home_team="Team A",
            away_team="Team B",
            score="1-0",
            match_date="2024-01-01",
        ),
        evidence=["Team A completed 12 progressive passes."],
    )

    prompt = format_prompt(example)

    assert "Evidence:" in prompt
    assert "- Team A completed 12 progressive passes." in prompt
    assert "Question: How did Team A progress the ball?" in prompt
