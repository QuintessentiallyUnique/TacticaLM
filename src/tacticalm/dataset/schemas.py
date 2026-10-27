"""Pydantic schemas for TacticaLM dataset and match representations.

Defines data models for:
- Team-level tactical statistics (TeamStats)
- Player-level statistics (PlayerStats)
- Match summaries (MatchSummary)
- Evaluation examples with reference facts and prohibited claims
- Training examples with instruction/evidence/response format
- Analysis categories matching the 10 categories from the project spec
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AnalysisCategory(str, Enum):
    """The 10 analysis categories defined in spec section 20."""

    MATCH_SUMMARY = "match_summary"
    BUILD_UP_ANALYSIS = "build_up_analysis"
    PRESSING_ANALYSIS = "pressing_analysis"
    CHANCE_CREATION = "chance_creation"
    DEFENSIVE_PERFORMANCE = "defensive_performance"
    TRANSITION_ANALYSIS = "transition_analysis"
    TEAM_COMPARISON = "team_comparison"
    PLAYER_CONTRIBUTION = "player_contribution"
    EVIDENCE_EXTRACTION = "evidence_extraction"
    LIMITATION_AWARE = "limitation_aware"


class ScoreInfo(BaseModel):
    """Match score."""

    home: int = 0
    away: int = 0


class MatchContext(BaseModel):
    """Context provided to the model with each analysis request."""

    competition: str = ""
    season: str = ""
    home_team: str = ""
    away_team: str = ""
    score: str = ""
    match_date: str = ""


class TeamStats(BaseModel):
    """All team-level tactical metrics computed by the feature engine.

    Fields are organized by metric category with defaults of 0/0.0 so that
    partial feature computation (e.g., event-only without 360 data) still
    produces a valid object.
    """

    # --- Possession ---
    possession_percentage: float = 0.0
    possession_count: int = 0
    avg_actions_per_possession: float = 0.0
    longest_possession_sequence: int = 0

    # --- Passing ---
    completed_passes: int = 0
    total_passes: int = 0
    pass_completion_percentage: float = 0.0
    progressive_passes: int = 0
    through_balls: int = 0
    crosses: int = 0
    cutbacks: int = 0

    # --- Shooting ---
    shots: int = 0
    shots_on_target: int = 0
    goals: int = 0
    expected_goals: float = 0.0
    open_play_xg: float = 0.0
    set_piece_xg: float = 0.0
    high_xg_chances: int = 0
    shots_inside_box: int = 0
    shots_outside_box: int = 0

    # --- Progression ---
    final_third_entries: int = 0
    penalty_area_entries: int = 0
    progressive_carries: int = 0

    # --- Pressing ---
    pressures: int = 0
    pressures_attacking_third: int = 0
    pressures_middle_third: int = 0
    pressures_defensive_third: int = 0
    high_regains: int = 0
    mid_regains: int = 0
    def_regains: int = 0
    ppda: float = 0.0

    # --- Defending ---
    tackles: int = 0
    tackles_won: int = 0
    interceptions: int = 0
    blocks: int = 0
    clearances: int = 0
    duels: int = 0
    aerial_duels: int = 0
    aerial_duels_won: int = 0

    # --- Transitions ---
    counterattacks: int = 0
    shots_after_regain: int = 0
    turnovers: int = 0
    central_turnovers: int = 0
    wide_turnovers: int = 0

    # --- Set Pieces ---
    corners: int = 0
    fouls_committed: int = 0
    fouls_won: int = 0
    yellow_cards: int = 0
    red_cards: int = 0

    # --- Spatial (Average positions from on-ball events) ---
    avg_x: float = 0.0
    avg_y: float = 0.0

    # --- 360-dependent (only computed when freeze-frame data available) ---
    line_breaking_passes: Optional[int] = None
    visible_teammate_count_avg: Optional[float] = None
    visible_opponent_count_avg: Optional[float] = None
    avg_nearest_opponent_distance: Optional[float] = None
    pitch_control_share: Optional[float] = None
    visible_opponent_compactness: Optional[float] = None


class GoalEvent(BaseModel):
    """A goal event in the match."""
    minute: int
    player_name: str
    team: str

class IntervalStats(BaseModel):
    """Momentum stats for a 15-minute interval."""

    team: str
    start_minute: int
    end_minute: int
    possession_pct: float = 0.0
    expected_goals: float = 0.0
    shots: int = 0


class PossessionSequence(BaseModel):
    """Chronological description of a high-value possession."""

    possession_id: int
    team: str
    start_minute: int = 0
    end_minute: int = 0
    event_count: int = 0
    outcome: str = ""
    expected_goals: float = 0.0
    sequence_text: str = ""


class PlayerStats(BaseModel):
    """Per-player tactical metrics."""

    player_name: str
    player_id: Optional[int] = None
    position: str = ""
    minutes_played: float = 0.0
    is_starter: bool = False

    # Passing
    passes_attempted: int = 0
    passes_completed: int = 0
    pass_completion_percentage: float = 0.0
    progressive_passes: int = 0
    key_passes: int = 0

    # Shooting
    goals: int = 0
    shots: int = 0
    expected_goals: float = 0.0

    # Carrying
    carries: int = 0
    progressive_carries: int = 0

    # Defensive
    recoveries: int = 0
    pressures: int = 0
    tackles: int = 0
    interceptions: int = 0
    duels: int = 0
    duels_won: int = 0

    # Touches
    touches: int = 0
    touches_attacking_third: int = 0
    touches_middle_third: int = 0
    touches_defensive_third: int = 0

    # Spatial
    avg_x: float = 0.0
    avg_y: float = 0.0


class MatchSummary(BaseModel):
    """Complete structured match representation for model input.

    Contains all computed features for both teams and all players,
    validated by Pydantic. This is the object that gets converted
    to evidence statements for the model prompt.
    """

    match_id: int
    competition: str = "Unknown"
    season: str = "Unknown"
    match_date: str = "Unknown"
    home_team: str
    away_team: str
    score_home: int = 0
    score_away: int = 0
    team_stats: Dict[str, TeamStats] = Field(default_factory=dict)
    player_stats: Dict[str, List[PlayerStats]] = Field(default_factory=dict)
    goals_timeline: List[GoalEvent] = Field(default_factory=list)
    momentum_timeline: List[IntervalStats] = Field(default_factory=list)
    high_value_sequences: List[PossessionSequence] = Field(default_factory=list)
    has_360_data: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationExample(BaseModel):
    """A single evaluation example with reference facts and prohibited claims.

    Used in both dev evaluation and final test sets. The reference_facts list
    contains claims that a correct answer should include. The prohibited_claims
    list contains claims that are NOT supported by the evidence and should NOT
    appear in a correct answer.
    """

    example_id: str
    match_id: int
    category: AnalysisCategory
    question: str
    context: MatchContext = Field(default_factory=MatchContext)
    evidence: List[str] = Field(default_factory=list)
    reference_facts: List[str] = Field(default_factory=list)
    prohibited_claims: List[str] = Field(default_factory=list)
    reference_answer: str = ""


class TrainingExample(BaseModel):
    """A single training example in instruction/evidence/response format.

    The instruction, context, and evidence form the model input.
    The response is the target output for supervised fine-tuning.
    """

    example_id: str
    match_id: int
    category: AnalysisCategory
    instruction: str
    context: MatchContext = Field(default_factory=MatchContext)
    evidence: List[str] = Field(default_factory=list)
    context_tables: List[str] = Field(default_factory=list)
    response: str = ""
