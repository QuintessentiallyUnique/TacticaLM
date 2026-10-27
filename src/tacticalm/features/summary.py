"""Match summary builder and evidence statement generator.

Orchestrates feature computation and wraps results in validated
Pydantic models. Also generates natural-language evidence statements
from computed metrics for use in model prompts.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from tacticalm.dataset.schemas import MatchSummary, PlayerStats, PossessionSequence, TeamStats

from .engine import FeatureEngine

logger = logging.getLogger(__name__)


def build_match_summary(
    match_id: int,
    events: pd.DataFrame,
    lineups: Dict,
    match_metadata: Dict,
    three_sixty: Optional[pd.DataFrame] = None,
) -> MatchSummary:
    """Compute all features and wrap in a validated MatchSummary.

    Args:
        match_id: Unique match identifier.
        events: Cleaned events DataFrame.
        lineups: Lineup data dict.
        match_metadata: Dict with competition, season, home_team, away_team, etc.
        three_sixty: Optional 360 freeze-frame DataFrame.

    Returns:
        Validated MatchSummary Pydantic model.

    Raises:
        ValueError: If feature computation fails.
    """
    engine = FeatureEngine()
    features = engine.compute_all_features(events, lineups, match_metadata, three_sixty)

    if not features:
        raise ValueError(f"Failed to compute features for match {match_id}")

    # Build TeamStats from computed features
    team_stats: Dict[str, TeamStats] = {}
    for team_name, stats_dict in features.get("teams", {}).items():
        # TeamStats has defaults for all fields, so missing keys are fine
        team_stats[team_name] = TeamStats(**stats_dict)

    # Build PlayerStats from computed features
    player_stats: Dict[str, List[PlayerStats]] = {}
    for team_name, p_list in features.get("players", {}).items():
        player_stats[team_name] = [PlayerStats(**p) for p in p_list]
        
    from tacticalm.dataset.schemas import GoalEvent, IntervalStats
    goals_timeline = [GoalEvent(**g) for g in features.get("goals_timeline", [])]
    momentum_timeline = [IntervalStats(**m) for m in features.get("momentum_timeline", [])]
    high_value_sequences = [
        PossessionSequence(**s) for s in features.get("high_value_sequences", [])
    ]

    summary = MatchSummary(
        match_id=match_id,
        competition=match_metadata.get("competition", "Unknown"),
        season=match_metadata.get("season", "Unknown"),
        match_date=match_metadata.get("match_date", "Unknown"),
        home_team=match_metadata.get("home_team", "Unknown"),
        away_team=match_metadata.get("away_team", "Unknown"),
        score_home=match_metadata.get("score_home", 0),
        score_away=match_metadata.get("score_away", 0),
        team_stats=team_stats,
        player_stats=player_stats,
        goals_timeline=goals_timeline,
        momentum_timeline=momentum_timeline,
        high_value_sequences=high_value_sequences,
        has_360_data=three_sixty is not None,
        metadata=match_metadata,
    )

    logger.info(f"Built match summary for match {match_id}")
    return summary


def generate_evidence_statements(
    summary: MatchSummary, team: str
) -> List[str]:
    """Convert numerical metrics into natural-language evidence statements.

    Generates deterministic factual statements from validated statistics.
    These statements form the evidence provided to the model in prompts.

    Args:
        summary: Validated MatchSummary.
        team: Team name to generate evidence for.

    Returns:
        List of 15-25 evidence statements.
    """
    if team not in summary.team_stats:
        logger.warning(f"Team '{team}' not found in summary")
        return []

    s = summary.team_stats[team]
    statements: List[str] = []

    # Possession
    statements.append(f"{team} had {s.possession_percentage:.1f}% possession.")
    if s.possession_count > 0:
        statements.append(
            f"{team} had {s.possession_count} possession sequences, "
            f"averaging {s.avg_actions_per_possession:.1f} actions per possession."
        )
    statements.append(f"{team}'s average spatial center of gravity during on-ball events was [x: {s.avg_x}, y: {s.avg_y}].")

    # Passing
    statements.append(
        f"{team} attempted {s.total_passes} passes with a "
        f"{s.pass_completion_percentage:.1f}% completion rate "
        f"({s.completed_passes} completed)."
    )
    statements.append(
        f"{team} completed {s.progressive_passes} progressive passes."
    )
    if s.through_balls > 0:
        statements.append(f"{team} played {s.through_balls} through balls.")
    if s.crosses > 0 or s.cutbacks > 0:
        statements.append(
            f"{team} attempted {s.crosses} crosses and {s.cutbacks} cutbacks."
        )

    # Shooting
    statements.append(
        f"{team} had {s.shots} shots, with {s.shots_on_target} on target."
    )
    statements.append(
        f"{team} generated {s.expected_goals:.2f} expected goals (xG)."
    )
    statements.append(f"{team} scored {s.goals} goal(s).")
    if s.high_xg_chances > 0:
        statements.append(
            f"{team} created {s.high_xg_chances} high-xG chance(s) "
            f"(xG >= 0.3 per shot)."
        )
    if s.shots_inside_box > 0 or s.shots_outside_box > 0:
        statements.append(
            f"{team} took {s.shots_inside_box} shots from inside the "
            f"penalty area and {s.shots_outside_box} from outside."
        )
    if s.open_play_xg > 0 or s.set_piece_xg > 0:
        statements.append(
            f"{team} generated {s.open_play_xg:.2f} xG from open play "
            f"and {s.set_piece_xg:.2f} from set pieces."
        )

    # Progression
    statements.append(
        f"{team} made {s.final_third_entries} entries into the final third."
    )
    statements.append(
        f"{team} entered the penalty area {s.penalty_area_entries} times."
    )
    if s.progressive_carries > 0:
        statements.append(
            f"{team} completed {s.progressive_carries} progressive carries."
        )

    # Pressing
    if s.pressures > 0:
        statements.append(
            f"{team} applied {s.pressures} pressures "
            f"({s.pressures_attacking_third} in the attacking third, "
            f"{s.pressures_middle_third} in the middle third)."
        )
    statements.append(
        f"{team} recorded {s.high_regains} high regains in the attacking third."
    )
    if s.ppda > 0:
        statements.append(
            f"{team} had a PPDA of {s.ppda:.2f} "
            f"(lower = more aggressive pressing)."
        )

    # Defending
    if s.tackles > 0:
        statements.append(
            f"{team} made {s.tackles} tackles "
            f"({s.tackles_won} won)."
        )
    if s.interceptions > 0:
        statements.append(f"{team} made {s.interceptions} interceptions.")
    if s.clearances > 0:
        statements.append(f"{team} made {s.clearances} clearances.")
    if s.blocks > 0:
        statements.append(f"{team} made {s.blocks} blocks.")

    # Transitions
    if s.counterattacks > 0:
        statements.append(
            f"{team} launched {s.counterattacks} counterattack(s)."
        )
    if s.turnovers > 0:
        statements.append(
            f"{team} lost possession {s.turnovers} times "
            f"({s.central_turnovers} centrally, {s.wide_turnovers} wide)."
        )

    # Set pieces
    if s.corners > 0:
        statements.append(f"{team} earned {s.corners} corner(s).")
    if s.fouls_committed > 0:
        statements.append(f"{team} committed {s.fouls_committed} foul(s).")

    # Opponent Summary
    opponent = summary.home_team if team == summary.away_team else summary.away_team
    if opponent in summary.team_stats:
        o = summary.team_stats[opponent]
        statements.append(f"In comparison, {opponent} had {o.possession_percentage:.1f}% possession.")
        statements.append(f"{opponent} generated {o.expected_goals:.2f} expected goals (xG).")
        statements.append(f"{opponent} took {o.shots} shots, with {o.shots_on_target} on target.")
        statements.append(f"{opponent} completed {o.progressive_passes} progressive passes.")

    # Match Timeline
    if summary.goals_timeline:
        timeline_str = "Goals in the match: " + ", ".join([f"{g.minute}' {g.player_name} ({g.team})" for g in summary.goals_timeline]) + "."
        statements.append(timeline_str)
    else:
        statements.append("There were no goals scored in the match.")

    # Momentum Timeline
    if summary.momentum_timeline:
        for m in summary.momentum_timeline:
            statements.append(f"Between {m.start_minute} and {m.end_minute} minutes, {m.team} had {m.possession_pct}% possession, generating {m.expected_goals:.2f} xG from {m.shots} shots.")

    # High-value shot sequences
    team_sequences = [
        seq for seq in summary.high_value_sequences if seq.team == team and seq.sequence_text
    ]
    for seq in team_sequences[:3]:
        statements.append(
            f"High-value possession {seq.possession_id} for {team} ended with "
            f"{seq.outcome} worth {seq.expected_goals:.2f} xG: {seq.sequence_text}."
        )

    # 360-derived spatial features
    if s.line_breaking_passes is not None and summary.has_360_data:
        statements.append(
            f"{team} completed {s.line_breaking_passes} line-breaking pass(es) "
            "based on available 360 freeze-frame data."
        )
    if s.pitch_control_share is not None and summary.has_360_data:
        statements.append(
            f"{team}'s average visible pitch-control share was "
            f"{s.pitch_control_share:.1f}% in available 360 freeze frames."
        )
    if s.avg_nearest_opponent_distance is not None and summary.has_360_data:
        statements.append(
            f"{team}'s ball carrier was on average "
            f"{s.avg_nearest_opponent_distance:.1f} yards from the nearest visible opponent."
        )

    # Player Stats
    for t in [team, opponent]:
        if t in summary.player_stats and summary.player_stats[t]:
            p_stats = summary.player_stats[t]
            # Filter players with < 15 mins to keep evidence concise
            active_players = [p for p in p_stats if p.minutes_played >= 15.0]
            if not active_players:
                active_players = p_stats
                
            for p in active_players:
                statements.append(f"{p.player_name} ({t}) played {int(p.minutes_played)} minutes, completed {p.pass_completion_percentage}% of passes ({p.progressive_passes} progressive), won {p.duels_won} duels, generated {p.expected_goals:.2f} xG, and had an average spatial position of [x: {p.avg_x}, y: {p.avg_y}].")

    return statements


def save_match_summary(summary: MatchSummary, output_dir: str | Path) -> Path:
    """Save a MatchSummary as a JSON file.

    Args:
        summary: Validated MatchSummary.
        output_dir: Directory to save to.

    Returns:
        Path to the saved JSON file.
    """
    output_path = Path(output_dir) / f"{summary.match_id}_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary.model_dump_json(indent=2))

    logger.info(f"Saved match summary to {output_path}")
    return output_path
