"""Dataset generation engine for TacticaLM."""

import json
import logging
import random
import uuid
from pathlib import Path

from tacticalm.dataset.schemas import (
    AnalysisCategory,
    EvaluationExample,
    MatchContext,
    MatchSummary,
    TrainingExample,
)
from tacticalm.dataset.prompts import (
    QUESTION_TEMPLATES,
    format_analysis_prompt,
    format_system_prompt,
)
from tacticalm.features.summary import generate_evidence_statements

logger = logging.getLogger(__name__)

def _get_random_question(category: AnalysisCategory, team: str, opponent: str, summary: MatchSummary) -> str:
    templates = QUESTION_TEMPLATES.get(category, [])
    if not templates:
        return f"Analyze {team} against {opponent}."
    
    template = random.choice(templates)
    return template.format(
        team=team,
        opponent=opponent,
        competition=summary.competition,
        match_date=summary.match_date
    )

def _get_prohibited_claims() -> list[str]:
    return [
        "Claims about the team's formation",
        "Claims about specific referee decisions (other than card counts)",
        "Claims comparing stats to the league average",
        "Claims about player intent, motivation, or psychology",
        "Claims based on video observations rather than stats"
    ]

def generate_reference_response(question: str, evidence: list[str], category: AnalysisCategory) -> str:
    """Generate a template-based reference response that uses the evidence.
    
    Args:
        question: The question asked.
        evidence: The evidence statements.
        category: The category of analysis.
        
    Returns:
        A reference response string.
    """
    if category == AnalysisCategory.LIMITATION_AWARE:
        return (
            "Based on the provided statistical evidence, there is insufficient data to answer this question. "
            "The data only includes tactical metrics (passes, shots, defensive actions, etc.) and does not contain "
            "information regarding formations, player intent, crowd atmosphere, or managerial instructions."
        )
    
    if not evidence:
        return "There is no statistical evidence available to answer this question."
        
    response = "Based on the provided statistical evidence:\n\n"
    for e in evidence:
        response += f"- {e}\n"
        
    return response


def format_summary_tables(summary: MatchSummary, teams: list[str] | None = None) -> list[str]:
    """Format compact Markdown tables for table-aware fine-tuning examples."""
    selected_teams = teams or [summary.home_team, summary.away_team]
    tables = []

    team_rows = [
        "| Team | Possession % | Passes | Progressive Passes | Shots | xG | Pressures | High Regains | PPDA |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for team in selected_teams:
        stats = summary.team_stats.get(team)
        if stats is None:
            continue
        team_rows.append(
            f"| {team} | {stats.possession_percentage:.1f} | {stats.total_passes} | "
            f"{stats.progressive_passes} | {stats.shots} | {stats.expected_goals:.2f} | "
            f"{stats.pressures} | {stats.high_regains} | {stats.ppda:.2f} |"
        )
    if len(team_rows) > 2:
        tables.append("Team statistical summary:\n" + "\n".join(team_rows))

    momentum_rows = [
        "| Team | Interval | Possession % | xG | Shots |",
        "|---|---|---:|---:|---:|",
    ]
    for interval in summary.momentum_timeline:
        if interval.team not in selected_teams:
            continue
        momentum_rows.append(
            f"| {interval.team} | {interval.start_minute}-{interval.end_minute} | "
            f"{interval.possession_pct:.1f} | {interval.expected_goals:.2f} | "
            f"{interval.shots} |"
        )
    if len(momentum_rows) > 2:
        tables.append("Momentum interval table:\n" + "\n".join(momentum_rows))

    sequence_rows = [
        "| Team | Possession | Minutes | Outcome | xG | Sequence |",
        "|---|---:|---|---|---:|---|",
    ]
    for sequence in summary.high_value_sequences[:5]:
        if sequence.team not in selected_teams:
            continue
        sequence_rows.append(
            f"| {sequence.team} | {sequence.possession_id} | "
            f"{sequence.start_minute}-{sequence.end_minute} | {sequence.outcome} | "
            f"{sequence.expected_goals:.2f} | {sequence.sequence_text} |"
        )
    if len(sequence_rows) > 2:
        tables.append("High-value sequence table:\n" + "\n".join(sequence_rows))

    return tables

def generate_eval_questions(match_summary: MatchSummary, num_per_category: int = 1) -> list[EvaluationExample]:
    """Generate evaluation questions for a match summary.
    
    Args:
        match_summary: The MatchSummary to generate questions for.
        num_per_category: Number of questions per category.
        
    Returns:
        List of EvaluationExample objects.
    """
    examples = []
    
    home_team = match_summary.home_team
    away_team = match_summary.away_team
    
    context = MatchContext(
        competition=match_summary.competition,
        season=match_summary.season,
        home_team=home_team,
        away_team=away_team,
        score=f"{match_summary.score_home}-{match_summary.score_away}",
        match_date=match_summary.match_date,
    )
    
    for category in AnalysisCategory:
        for _ in range(num_per_category):
            # Pick a team randomly as the focus
            team_focus = home_team if random.random() < 0.5 else away_team
            opponent = away_team if team_focus == home_team else home_team
            
            question = _get_random_question(category, team_focus, opponent, match_summary)
            evidence = generate_evidence_statements(match_summary, team_focus)
            
            # If team comparison, add opponent evidence
            if category == AnalysisCategory.TEAM_COMPARISON:
                evidence.extend(generate_evidence_statements(match_summary, opponent))
            
            # Use evidence as reference facts (in a real scenario, could filter to just relevant ones)
            reference_facts = evidence.copy()
            if category == AnalysisCategory.LIMITATION_AWARE:
                reference_facts = ["Acknowledge that there is insufficient data to answer."]
                
            prohibited_claims = _get_prohibited_claims()
            
            ref_answer = generate_reference_response(question, evidence, category)
            
            example = EvaluationExample(
                example_id=str(uuid.uuid4()),
                match_id=match_summary.match_id,
                category=category,
                question=question,
                context=context,
                evidence=evidence,
                reference_facts=reference_facts,
                prohibited_claims=prohibited_claims,
                reference_answer=ref_answer
            )
            examples.append(example)
            
    return examples

def _load_summaries(match_ids: list[int], summaries_dir: Path) -> list[MatchSummary]:
    summaries = []
    for match_id in match_ids:
        path = summaries_dir / f"{match_id}_summary.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                summaries.append(MatchSummary(**data))
        else:
            logger.warning(f"Summary for match {match_id} not found at {path}")
    return summaries

def generate_dev_eval_set(match_ids: list[int], summaries_dir: Path) -> list[EvaluationExample]:
    """Generate the development evaluation set.
    
    Args:
        match_ids: List of match IDs.
        summaries_dir: Directory containing match summaries.
        
    Returns:
        List of EvaluationExample objects.
    """
    summaries = _load_summaries(match_ids, summaries_dir)
    examples = []
    
    for summary in summaries:
        examples.extend(generate_eval_questions(summary, num_per_category=1))
        
    # Optional: limit to ~60-80 total questions if there are too many matches
    if len(examples) > 80:
        examples = random.sample(examples, 80)
        
    output_path = summaries_dir.parent.parent / "evaluation" / "dev" / "dev_eval.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([e.model_dump() for e in examples], f, indent=2)
        
    return examples

def generate_final_test_set(match_ids: list[int], summaries_dir: Path) -> list[EvaluationExample]:
    """Generate the final test evaluation set.
    
    Args:
        match_ids: List of match IDs.
        summaries_dir: Directory containing match summaries.
        
    Returns:
        List of EvaluationExample objects.
    """
    summaries = _load_summaries(match_ids, summaries_dir)
    examples = []
    
    for summary in summaries:
        examples.extend(generate_eval_questions(summary, num_per_category=1))
        
    output_path = summaries_dir.parent.parent / "evaluation" / "final_test" / "final_test.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([e.model_dump() for e in examples], f, indent=2)
        
    return examples

def generate_training_examples(
    match_ids: list[int],
    summaries_dir: Path,
    num_per_match: int = 8,
    include_tables: bool = True,
    table_probability: float = 0.35,
) -> list[TrainingExample]:
    """Generate training examples from match summaries.
    
    Args:
        match_ids: List of match IDs.
        summaries_dir: Directory containing match summaries.
        num_per_match: Number of examples to generate per match.
        include_tables: Whether to include Markdown tables in some examples.
        table_probability: Fraction of examples that receive table context.
        
    Returns:
        List of TrainingExample objects.
    """
    summaries = _load_summaries(match_ids, summaries_dir)
    examples = []
    
    for summary in summaries:
        home_team = summary.home_team
        away_team = summary.away_team
        
        context = MatchContext(
            competition=summary.competition,
            season=summary.season,
            home_team=home_team,
            away_team=away_team,
            score=f"{summary.score_home}-{summary.score_away}",
            match_date=summary.match_date,
        )
        
        categories = list(AnalysisCategory)
        
        for _ in range(num_per_match):
            # 5% chance of limitation aware
            if random.random() < 0.05:
                category = AnalysisCategory.LIMITATION_AWARE
            else:
                category = random.choice([c for c in categories if c != AnalysisCategory.LIMITATION_AWARE])
                
            team_focus = home_team if random.random() < 0.5 else away_team
            opponent = away_team if team_focus == home_team else home_team
            
            question = _get_random_question(category, team_focus, opponent, summary)
            evidence = generate_evidence_statements(summary, team_focus)
            
            if category == AnalysisCategory.TEAM_COMPARISON:
                evidence.extend(generate_evidence_statements(summary, opponent))

            context_tables = []
            should_include_tables = include_tables and random.random() < table_probability
            if should_include_tables:
                table_teams = [team_focus, opponent] if category == AnalysisCategory.TEAM_COMPARISON else [team_focus]
                context_tables = format_summary_tables(summary, table_teams)

            prompt_body = format_analysis_prompt(question, evidence, context.model_dump())
            if context_tables:
                prompt_body += (
                    "\n\nStructured Tables:\n"
                    + "\n\n".join(context_tables)
                    + "\n\nUse table cells only when they match the provided evidence."
                )

            instruction = format_system_prompt() + "\n\n" + prompt_body
            response = generate_reference_response(question, evidence, category)
            
            example = TrainingExample(
                example_id=str(uuid.uuid4()),
                match_id=summary.match_id,
                category=category,
                instruction=instruction,
                context=context,
                evidence=evidence,
                context_tables=context_tables,
                response=response
            )
            examples.append(example)
            
    return examples
