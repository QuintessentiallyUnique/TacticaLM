"""Tactical feature computation engine for StatsBomb event data.

Computes team-level and player-level metrics across all categories:
possession, passing, shooting, progression, pressing, defending,
transitions, and player stats.

All metrics operate on statsbombpy's flattened DataFrame columns
(e.g., 'type', 'pass_outcome', 'shot_statsbomb_xg', 'carry_end_location').

StatsBomb coordinate system:
- Pitch: 120 x 80 yards
- Events are standardized per-event: the acting team always attacks
  toward x=120. So attacking third = x >= 80 for ALL teams.
- Goal center: [120, 40]
"""

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from tacticalm.features.spatial import compute_360_team_metrics
from tacticalm.utils.config import get_features_config

logger = logging.getLogger(__name__)

# Default feature thresholds (overridden by config if available)
_DEFAULT_PROGRESSIVE_DISTANCE = 10.0  # yards closer to goal
_DEFAULT_HIGH_XG_THRESHOLD = 0.3
_DEFAULT_PPDA_ZONE_X_MIN = 48  # opponent's attacking 60%
_DEFAULT_SHOTS_AFTER_REGAIN_SECONDS = 10
_DEFAULT_MIN_MINUTES = 15
_GOAL_CENTER = [120.0, 40.0]


def _safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Safely get a column, returning an empty Series if missing."""
    if col in df.columns:
        return df[col]
    return pd.Series(dtype="object", index=df.index)


def _dist_to_goal(loc: list) -> float:
    """Euclidean distance from a pitch location to the goal center [120, 40]."""
    return math.dist(loc[:2], _GOAL_CENTER)


class FeatureEngine:
    """Computes tactical features and metrics from StatsBomb event data.

    All methods accept a full-match events DataFrame and filter by team
    internally. This ensures consistency and avoids pre-filtering bugs.

    Usage:
        engine = FeatureEngine()
        features = engine.compute_all_features(events, lineups, metadata)
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the feature engine.

        Args:
            config: Optional features config dict. If None, loads from
                configs/features.yaml.
        """
        try:
            self.config = config if config is not None else get_features_config()
        except FileNotFoundError:
            logger.warning("Features config not found, using defaults")
            self.config = {}

        features_cfg = self.config.get("features", {})
        prog_cfg = features_cfg.get("progressive_pass", {})
        self.progressive_distance = prog_cfg.get(
            "min_distance_reduction", _DEFAULT_PROGRESSIVE_DISTANCE
        )
        self.high_xg_threshold = features_cfg.get(
            "high_xg_threshold", _DEFAULT_HIGH_XG_THRESHOLD
        )
        pressing_cfg = features_cfg.get("pressing", {})
        self.ppda_zone_x_min = pressing_cfg.get(
            "ppda_defensive_zone_x_min", _DEFAULT_PPDA_ZONE_X_MIN
        )
        self.shots_after_regain_secs = pressing_cfg.get(
            "shots_after_regain_seconds", _DEFAULT_SHOTS_AFTER_REGAIN_SECONDS
        )
        players_cfg = features_cfg.get("players", {})
        self.min_minutes = players_cfg.get(
            "min_minutes_for_analysis", _DEFAULT_MIN_MINUTES
        )

    def compute_all_features(
        self,
        events: pd.DataFrame,
        lineups: Dict,
        match_metadata: Dict,
        three_sixty: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """Compute all tactical features for a match.

        Args:
            events: StatsBomb event data (flattened columns).
            lineups: Lineup data dict.
            match_metadata: Match metadata dict.
            three_sixty: Optional 360 freeze-frame data.

        Returns:
            Nested dict: {"teams": {team: stats}, "players": {team: [stats]}}
        """
        if events.empty:
            logger.warning("Empty events DataFrame provided")
            return {}

        teams = events["team"].dropna().unique().tolist()
        if len(teams) < 2:
            logger.warning(f"Found {len(teams)} teams, expected 2")
            return {}

        team_a, team_b = teams[0], teams[1]

        # --- Extract goals timeline ---
        goals_timeline = []
        shots = events[events["type"] == "Shot"]
        if "shot_outcome" in shots.columns:
            goal_shots = shots[shots["shot_outcome"] == "Goal"]
            if not goal_shots.empty and "minute" in goal_shots.columns and "second" in goal_shots.columns:
                goal_shots = goal_shots.sort_values(by=["minute", "second"])
            for _, row in goal_shots.iterrows():
                goals_timeline.append({
                    "minute": int(row.get("minute", 0)),
                    "player_name": str(row.get("player", "Unknown")),
                    "team": str(row.get("team", "Unknown"))
                })
                
        # --- Extract momentum timeline ---
        momentum_timeline = []
        if "minute" in events.columns:
            max_min = int(events["minute"].max()) if not events.empty else 90
            num_intervals = (max_min // 15) + 1
            for i in range(num_intervals):
                start_m = i * 15
                end_m = start_m + 14
                interval_events = events[(events["minute"] >= start_m) & (events["minute"] <= end_m)]
                
                for t in [team_a, team_b]:
                    t_events = interval_events[interval_events["team"] == t]
                    
                    # Possession pct
                    poss_pct = 0.0
                    if "possession_team" in interval_events.columns:
                        t_poss_events = t_events[t_events["possession_team"] == t]
                        total_poss_events = interval_events[interval_events["possession_team"].notna()]
                        if len(total_poss_events) > 0:
                            poss_pct = round(len(t_poss_events) / len(total_poss_events) * 100, 1)
                    
                    # xG
                    t_shots = t_events[t_events["type"] == "Shot"]
                    xg_col = _safe_col(t_shots, "shot_statsbomb_xg")
                    xg = float(xg_col.sum()) if not xg_col.empty else 0.0
                    
                    momentum_timeline.append({
                        "team": t,
                        "start_minute": start_m,
                        "end_minute": end_m,
                        "possession_pct": poss_pct,
                        "expected_goals": round(xg, 2),
                        "shots": len(t_shots)
                    })

        return {
            "teams": {
                team_a: self._compute_team_features(events, team_a, team_b, three_sixty),
                team_b: self._compute_team_features(events, team_b, team_a, three_sixty),
            },
            "players": {
                team_a: self.compute_player_stats(events, team_a, lineups),
                team_b: self.compute_player_stats(events, team_b, lineups),
            },
            "goals_timeline": goals_timeline,
            "momentum_timeline": momentum_timeline,
            "high_value_sequences": self.extract_high_value_sequences(events),
        }

    def _compute_team_features(
        self,
        events: pd.DataFrame,
        team: str,
        opponent: str,
        three_sixty: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """Compute all feature categories for one team."""
        features: Dict[str, Any] = {}
        features.update(self.compute_possession(events, team))
        features.update(self.compute_passing(events, team))
        features.update(self.compute_shooting(events, team))
        features.update(self.compute_progression(events, team))
        features.update(self.compute_pressing(events, team, opponent))
        features.update(self.compute_defending(events, team, opponent))
        features.update(self.compute_transitions(events, team, opponent))
        features.update(self.compute_set_pieces(events, team))
        
        # Spatial
        team_events = events[events["team"] == team]
        on_ball = team_events[team_events["type"].isin(["Pass", "Carry", "Shot", "Dribble", "Ball Receipt*", "Ball Recovery"])]
        x_list, y_list = [], []
        for _, row in on_ball.iterrows():
            loc = row.get("location")
            if isinstance(loc, (list, np.ndarray)) and len(loc) >= 2:
                x_list.append(float(loc[0]))
                y_list.append(float(loc[1]))
        features["avg_x"] = round(float(np.mean(x_list)), 1) if x_list else 0.0
        features["avg_y"] = round(float(np.mean(y_list)), 1) if y_list else 0.0
        features.update(compute_360_team_metrics(events, three_sixty, team))
        
        return features

    def extract_high_value_sequences(
        self,
        events: pd.DataFrame,
        max_sequences: int = 8,
    ) -> List[Dict[str, Any]]:
        """Extract shot-ending possessions as chronological natural language."""
        if events.empty or "possession" not in events.columns:
            return []

        sequences = []
        event_types_to_describe = {
            "Pass",
            "Carry",
            "Dribble",
            "Shot",
            "Pressure",
            "Ball Recovery",
            "Duel",
            "Interception",
            "Dispossessed",
            "Miscontrol",
        }

        for possession_id, poss_events in events.groupby("possession", sort=False):
            if pd.isna(possession_id):
                continue
            if "type" not in poss_events.columns or not (poss_events["type"] == "Shot").any():
                continue

            sort_cols = [
                col for col in ["period", "minute", "second", "index"] if col in poss_events.columns
            ]
            if sort_cols:
                poss_events = poss_events.sort_values(sort_cols)
            possession_team = self._sequence_team(poss_events)
            shots = poss_events[poss_events["type"] == "Shot"]
            xg_col = _safe_col(shots, "shot_statsbomb_xg")
            xg = float(xg_col.sum()) if not xg_col.empty else 0.0
            outcome = self._sequence_outcome(shots)

            described_events = []
            for _, row in poss_events.iterrows():
                if row.get("type") not in event_types_to_describe:
                    continue
                described = self._describe_sequence_event(row)
                if described:
                    described_events.append(described)
                if len(described_events) >= 12:
                    break

            if not described_events:
                continue

            sequences.append(
                {
                    "possession_id": int(possession_id),
                    "team": possession_team,
                    "start_minute": int(poss_events["minute"].min())
                    if "minute" in poss_events.columns
                    else 0,
                    "end_minute": int(poss_events["minute"].max())
                    if "minute" in poss_events.columns
                    else 0,
                    "event_count": int(len(poss_events)),
                    "outcome": outcome,
                    "expected_goals": round(xg, 4),
                    "sequence_text": " -> ".join(described_events),
                }
            )

        sequences.sort(key=lambda item: (item["expected_goals"], item["event_count"]), reverse=True)
        return sequences[:max_sequences]

    def _sequence_team(self, poss_events: pd.DataFrame) -> str:
        if "possession_team" in poss_events.columns:
            teams = poss_events["possession_team"].dropna().unique().tolist()
            if teams:
                return str(teams[0])
        teams = poss_events["team"].dropna().unique().tolist()
        return str(teams[0]) if teams else "Unknown"

    def _sequence_outcome(self, shots: pd.DataFrame) -> str:
        shot_outcome = _safe_col(shots, "shot_outcome")
        if (shot_outcome == "Goal").any():
            return "goal"
        if shot_outcome.isin({"Saved", "Saved To Post", "Saved to Post"}).any():
            return "shot on target"
        return "shot"

    def _describe_sequence_event(self, row: pd.Series) -> str:
        minute = int(row.get("minute", 0))
        player = row.get("player")
        player_name = str(player) if pd.notna(player) else "Unknown player"
        event_type = row.get("type")

        if event_type == "Pass":
            recipient = row.get("pass_recipient")
            if pd.notna(recipient):
                return f"{minute}' {player_name} passed to {recipient}"
            return f"{minute}' {player_name} attempted a pass"
        if event_type == "Carry":
            return f"{minute}' {player_name} carried the ball"
        if event_type == "Dribble":
            return f"{minute}' {player_name} dribbled"
        if event_type == "Shot":
            outcome = row.get("shot_outcome", "shot")
            xg = row.get("shot_statsbomb_xg", 0.0)
            xg_value = float(xg) if pd.notna(xg) else 0.0
            return f"{minute}' {player_name} shot ({outcome}, {xg_value:.2f} xG)"
        if event_type == "Pressure":
            return f"{minute}' {player_name} applied pressure"
        if event_type == "Ball Recovery":
            return f"{minute}' {player_name} recovered the ball"
        if event_type == "Duel":
            return f"{minute}' {player_name} contested a duel"
        if event_type == "Interception":
            return f"{minute}' {player_name} intercepted"
        if event_type in {"Dispossessed", "Miscontrol"}:
            return f"{minute}' {player_name} lost control"
        return ""

    # -------------------------------------------------------------------------
    # POSSESSION
    # -------------------------------------------------------------------------

    def compute_possession(self, events: pd.DataFrame, team: str) -> Dict:
        """Compute possession metrics.

        Possession % is calculated as the proportion of events belonging
        to possessions owned by this team, which approximates time-based
        possession from event data.

        Returns:
            possession_percentage, possession_count, avg_actions_per_possession,
            longest_possession_sequence
        """
        if "possession_team" not in events.columns:
            return {
                "possession_percentage": 0.0,
                "possession_count": 0,
                "avg_actions_per_possession": 0.0,
                "longest_possession_sequence": 0,
            }

        team_events = events[events["possession_team"] == team]
        total_events = len(events)
        team_event_count = len(team_events)

        possession_pct = (
            round(team_event_count / total_events * 100, 1) if total_events > 0 else 0.0
        )

        team_poss_ids = team_events["possession"].unique() if "possession" in events.columns else []
        poss_count = len(team_poss_ids)

        if poss_count > 0 and "possession" in events.columns:
            poss_sizes = team_events.groupby("possession").size()
            avg_actions = round(poss_sizes.mean(), 1)
            longest_seq = int(poss_sizes.max())
        else:
            avg_actions = 0.0
            longest_seq = 0

        return {
            "possession_percentage": possession_pct,
            "possession_count": poss_count,
            "avg_actions_per_possession": avg_actions,
            "longest_possession_sequence": longest_seq,
        }

    # -------------------------------------------------------------------------
    # PASSING
    # -------------------------------------------------------------------------

    def compute_passing(self, events: pd.DataFrame, team: str) -> Dict:
        """Compute passing metrics.

        Progressive pass rule: A completed forward pass that moves the ball
        at least {progressive_distance} yards closer to the goal center [120,40].

        Returns:
            total_passes, completed_passes, pass_completion_percentage,
            progressive_passes, through_balls, crosses, cutbacks
        """
        team_passes = events[(events["team"] == team) & (events["type"] == "Pass")]
        total = len(team_passes)

        # Completed passes: pass_outcome is NaN (StatsBomb convention)
        pass_outcome = _safe_col(team_passes, "pass_outcome")
        completed_mask = pass_outcome.isna()
        completed = int(completed_mask.sum())
        completion_pct = round(completed / total * 100, 1) if total > 0 else 0.0

        # Progressive passes (only completed)
        completed_passes = team_passes[completed_mask]
        progressive = 0
        for _, row in completed_passes.iterrows():
            start = row.get("location")
            end = row.get("pass_end_location")
            if self._is_progressive_pass(start, end):
                progressive += 1

        # Through balls
        pass_technique = _safe_col(team_passes, "pass_technique")
        through_balls = int((pass_technique == "Through Ball").sum())

        # Crosses
        pass_cross = _safe_col(team_passes, "pass_cross")
        crosses = int((pass_cross == True).sum())  # noqa: E712

        # Cutbacks
        pass_cut_back = _safe_col(team_passes, "pass_cut_back")
        cutbacks = int((pass_cut_back == True).sum())  # noqa: E712

        return {
            "total_passes": total,
            "completed_passes": completed,
            "pass_completion_percentage": completion_pct,
            "progressive_passes": progressive,
            "through_balls": through_balls,
            "crosses": crosses,
            "cutbacks": cutbacks,
        }

    def _is_progressive_pass(self, start_loc: Any, end_loc: Any) -> bool:
        """Check if a pass is progressive.

        A progressive pass moves the ball at least self.progressive_distance
        yards closer to the goal center [120, 40]. Only valid for completed
        passes with known start and end locations.
        """
        if (
            not isinstance(start_loc, (list, np.ndarray))
            or not isinstance(end_loc, (list, np.ndarray))
            or len(start_loc) < 2
            or len(end_loc) < 2
        ):
            return False

        start_dist = _dist_to_goal(start_loc)
        end_dist = _dist_to_goal(end_loc)
        return (start_dist - end_dist) >= self.progressive_distance

    def _is_progressive_carry(self, start_loc: Any, end_loc: Any) -> bool:
        """Check if a carry is progressive (same rule as progressive pass)."""
        return self._is_progressive_pass(start_loc, end_loc)

    # -------------------------------------------------------------------------
    # SHOOTING
    # -------------------------------------------------------------------------

    def compute_shooting(self, events: pd.DataFrame, team: str) -> Dict:
        """Compute shooting metrics.

        High-xG chances: shots with statsbomb_xg >= threshold (default 0.3).
        Open-play vs set-piece split by shot_type field.

        Returns:
            shots, shots_on_target, goals, expected_goals, open_play_xg,
            set_piece_xg, high_xg_chances, shots_inside_box, shots_outside_box
        """
        shots = events[(events["team"] == team) & (events["type"] == "Shot")]
        total = len(shots)

        # Outcomes
        shot_outcome = _safe_col(shots, "shot_outcome")
        goals = int((shot_outcome == "Goal").sum())
        on_target_outcomes = {"Goal", "Saved", "Saved To Post", "Saved to Post"}
        on_target = int(shot_outcome.isin(on_target_outcomes).sum())

        # xG
        xg_col = _safe_col(shots, "shot_statsbomb_xg")
        xg_total = float(xg_col.sum()) if not xg_col.empty else 0.0

        # Open-play vs set-piece xG
        shot_type = _safe_col(shots, "shot_type")
        set_piece_types = {"Free Kick", "Penalty", "Corner"}
        is_set_piece = shot_type.isin(set_piece_types)
        set_piece_xg = float(xg_col[is_set_piece].sum()) if not xg_col.empty else 0.0
        open_play_xg = round(xg_total - set_piece_xg, 4)

        # High-xG chances
        high_xg = int((xg_col >= self.high_xg_threshold).sum()) if not xg_col.empty else 0

        # Shots inside/outside penalty area (x >= 102, 18 <= y <= 62)
        inside_box = 0
        outside_box = 0
        for _, row in shots.iterrows():
            loc = row.get("location")
            if isinstance(loc, (list, np.ndarray)) and len(loc) >= 2:
                x, y = loc[0], loc[1]
                if x >= 102 and 18 <= y <= 62:
                    inside_box += 1
                else:
                    outside_box += 1

        return {
            "shots": total,
            "shots_on_target": on_target,
            "goals": goals,
            "expected_goals": round(xg_total, 4),
            "open_play_xg": round(open_play_xg, 4),
            "set_piece_xg": round(set_piece_xg, 4),
            "high_xg_chances": high_xg,
            "shots_inside_box": inside_box,
            "shots_outside_box": outside_box,
        }

    # -------------------------------------------------------------------------
    # PROGRESSION
    # -------------------------------------------------------------------------

    def compute_progression(self, events: pd.DataFrame, team: str) -> Dict:
        """Compute ball progression metrics.

        Final-third entry: a completed pass or carry crossing x=80
        (start < 80, end >= 80).
        Penalty-area entry: a completed pass or carry ending in the
        penalty area (x >= 102, 18 <= y <= 62) from outside it.

        Returns:
            final_third_entries, penalty_area_entries, progressive_carries
        """
        team_events = events[events["team"] == team]
        prog_events = team_events[team_events["type"].isin(["Pass", "Carry"])]

        final_third = 0
        penalty_area = 0
        prog_carries = 0

        for _, row in prog_events.iterrows():
            loc = row.get("location")
            if not isinstance(loc, (list, np.ndarray)) or len(loc) < 2:
                continue

            start_x, start_y = float(loc[0]), float(loc[1])
            is_pass = row["type"] == "Pass"

            # Skip incomplete passes
            if is_pass:
                outcome = row.get("pass_outcome")
                if pd.notna(outcome):
                    continue
                end_loc = row.get("pass_end_location")
            else:
                end_loc = row.get("carry_end_location")

            if not isinstance(end_loc, (list, np.ndarray)) or len(end_loc) < 2:
                continue

            end_x, end_y = float(end_loc[0]), float(end_loc[1])

            # Final-third entry: starts outside, ends inside
            if start_x < 80 and end_x >= 80:
                final_third += 1

            # Penalty-area entry: ends inside the box, started outside
            in_box_start = start_x >= 102 and 18 <= start_y <= 62
            in_box_end = end_x >= 102 and 18 <= end_y <= 62
            if in_box_end and not in_box_start:
                penalty_area += 1

            # Progressive carry
            if not is_pass and self._is_progressive_carry(loc, end_loc):
                prog_carries += 1

        return {
            "final_third_entries": final_third,
            "penalty_area_entries": penalty_area,
            "progressive_carries": prog_carries,
        }

    # -------------------------------------------------------------------------
    # PRESSING
    # -------------------------------------------------------------------------

    def compute_pressing(
        self, events: pd.DataFrame, team: str, opponent: str
    ) -> Dict:
        """Compute pressing metrics.

        High regain: Ball Recovery in the attacking third (x >= 80).
        PPDA: Opponent passes allowed / team defensive actions in the
        opponent's attacking 60% of the pitch.

        Returns:
            pressures, pressures by third, high/mid/def regains, ppda
        """
        team_events = events[events["team"] == team]

        # --- Pressure events ---
        pressures = team_events[team_events["type"] == "Pressure"]
        pressure_count = len(pressures)

        # Pressures by zone
        p_atk = p_mid = p_def = 0
        for _, row in pressures.iterrows():
            loc = row.get("location")
            if isinstance(loc, (list, np.ndarray)) and len(loc) >= 2:
                x = float(loc[0])
                if x >= 80:
                    p_atk += 1
                elif x >= 40:
                    p_mid += 1
                else:
                    p_def += 1

        # --- Ball recoveries by zone ---
        recoveries = team_events[team_events["type"] == "Ball Recovery"]
        high_regains = mid_regains = def_regains = 0
        for _, row in recoveries.iterrows():
            loc = row.get("location")
            if isinstance(loc, (list, np.ndarray)) and len(loc) >= 2:
                x = float(loc[0])
                if x >= 80:
                    high_regains += 1
                elif x >= 40:
                    mid_regains += 1
                else:
                    def_regains += 1

        # --- PPDA ---
        # Opponent passes in their defensive 40% = opponent's x <= 72 (from
        # their perspective, but since events are per-event standardized,
        # opponent passes at x <= 72 means in their own half + a bit more).
        # Actually PPDA = opponent passes allowed / defensive actions in the
        # opponent's attacking 60%. Opponent's attacking 60% from their view
        # is x >= 48 from their view. Since opponent events have their own
        # coordinate frame, we count passes by the opponent at x >= 48 (they
        # are progressing forward), and count our defensive actions against them.
        opp_passes = events[
            (events["team"] == opponent) & (events["type"] == "Pass")
        ]
        opp_passes_in_zone = 0
        for _, row in opp_passes.iterrows():
            loc = row.get("location")
            if isinstance(loc, (list, np.ndarray)) and len(loc) >= 2:
                if float(loc[0]) >= self.ppda_zone_x_min:
                    opp_passes_in_zone += 1

        # Defensive actions by the pressing team
        def_action_types = {"Tackle", "Interception", "Foul Committed"}
        def_actions = team_events[team_events["type"].isin(def_action_types)]
        def_in_zone = 0
        for _, row in def_actions.iterrows():
            loc = row.get("location")
            if isinstance(loc, (list, np.ndarray)) and len(loc) >= 2:
                # Our defensive actions in the opponent's half
                # When we perform an action, our x >= 48 means we're in
                # opponent territory
                if float(loc[0]) >= self.ppda_zone_x_min:
                    def_in_zone += 1

        ppda = round(opp_passes_in_zone / def_in_zone, 2) if def_in_zone > 0 else 0.0

        return {
            "pressures": pressure_count,
            "pressures_attacking_third": p_atk,
            "pressures_middle_third": p_mid,
            "pressures_defensive_third": p_def,
            "high_regains": high_regains,
            "mid_regains": mid_regains,
            "def_regains": def_regains,
            "ppda": ppda,
        }

    # -------------------------------------------------------------------------
    # DEFENDING
    # -------------------------------------------------------------------------

    def compute_defending(
        self, events: pd.DataFrame, team: str, opponent: str
    ) -> Dict:
        """Compute defensive metrics.

        Returns:
            tackles, tackles_won, interceptions, blocks, clearances,
            duels, aerial_duels, aerial_duels_won
        """
        team_events = events[events["team"] == team]

        # Tackles (from Duel events with duel_type containing 'Tackle')
        duels = team_events[team_events["type"] == "Duel"]
        duel_type = _safe_col(duels, "duel_type")
        duel_outcome = _safe_col(duels, "duel_outcome")

        tackle_mask = duel_type.str.contains("Tackle", na=False)
        tackles = int(tackle_mask.sum())
        tackles_won = int(
            (tackle_mask & duel_outcome.isin({"Won", "Success In Play", "Success Out"})).sum()
        )

        # Aerial duels
        aerial_mask = duel_type.str.contains("Aerial", na=False)
        aerial_duels = int(aerial_mask.sum())
        aerial_won = int(
            (aerial_mask & duel_outcome.isin({"Won", "Success In Play", "Success Out"})).sum()
        )

        # Total duels
        total_duels = len(duels)

        # Other defensive actions
        interceptions = int((team_events["type"] == "Interception").sum())
        blocks = int((team_events["type"] == "Block").sum())
        clearances = int((team_events["type"] == "Clearance").sum())

        return {
            "tackles": tackles,
            "tackles_won": tackles_won,
            "interceptions": interceptions,
            "blocks": blocks,
            "clearances": clearances,
            "duels": total_duels,
            "aerial_duels": aerial_duels,
            "aerial_duels_won": aerial_won,
        }

    # -------------------------------------------------------------------------
    # TRANSITIONS
    # -------------------------------------------------------------------------

    def compute_transitions(
        self, events: pd.DataFrame, team: str, opponent: str
    ) -> Dict:
        """Compute transition metrics.

        Counterattack: possession with play_pattern == 'From Counter'.
        Turnover: Dispossessed, Miscontrol, or Error events by the team.

        Returns:
            counterattacks, shots_after_regain, turnovers,
            central_turnovers, wide_turnovers
        """
        team_events = events[events["team"] == team]

        # Counterattacks (unique possessions with 'From Counter' pattern)
        play_pattern = _safe_col(team_events, "play_pattern")
        counter_mask = play_pattern == "From Counter"
        if "possession" in team_events.columns:
            counter_possessions = team_events.loc[counter_mask, "possession"].unique()
            counterattacks = len(counter_possessions)
        else:
            counterattacks = 0

        # Shots after regaining possession (within N seconds)
        recoveries = team_events[team_events["type"] == "Ball Recovery"]
        team_shots = team_events[team_events["type"] == "Shot"]
        shots_after = 0
        for _, rec in recoveries.iterrows():
            rec_min = rec.get("minute", 0)
            rec_sec = rec.get("second", 0)
            rec_time = rec_min * 60 + rec_sec
            for _, shot in team_shots.iterrows():
                shot_min = shot.get("minute", 0)
                shot_sec = shot.get("second", 0)
                shot_time = shot_min * 60 + shot_sec
                if 0 < (shot_time - rec_time) <= self.shots_after_regain_secs:
                    shots_after += 1
                    break  # Count each recovery at most once

        # Turnovers (Dispossessed + Miscontrol + Error events)
        turnover_types = {"Dispossessed", "Miscontrol", "Error"}
        turnover_events = team_events[team_events["type"].isin(turnover_types)]
        turnovers = len(turnover_events)

        central_to = wide_to = 0
        for _, row in turnover_events.iterrows():
            loc = row.get("location")
            if isinstance(loc, (list, np.ndarray)) and len(loc) >= 2:
                y = float(loc[1])
                # Central: roughly middle of pitch (y between 24 and 56)
                if 24 <= y <= 56:
                    central_to += 1
                else:
                    wide_to += 1

        return {
            "counterattacks": counterattacks,
            "shots_after_regain": shots_after,
            "turnovers": turnovers,
            "central_turnovers": central_to,
            "wide_turnovers": wide_to,
        }

    # -------------------------------------------------------------------------
    # SET PIECES
    # -------------------------------------------------------------------------

    def compute_set_pieces(self, events: pd.DataFrame, team: str) -> Dict:
        """Compute set-piece and discipline metrics.

        Returns:
            corners, fouls_committed, fouls_won, yellow_cards, red_cards
        """
        team_events = events[events["team"] == team]

        # Corners (passes with type 'Corner')
        team_passes = team_events[team_events["type"] == "Pass"]
        pass_type = _safe_col(team_passes, "pass_type")
        corners = int((pass_type == "Corner").sum())

        # Fouls
        fouls_committed = int((team_events["type"] == "Foul Committed").sum())
        fouls_won = int((team_events["type"] == "Foul Won").sum())

        # Cards (from Bad Behaviour or Foul Committed events)
        # In statsbombpy, cards appear in foul_committed or bad_behaviour columns
        yellow_cards = 0
        red_cards = 0
        card_cols = ["foul_committed_card", "bad_behaviour_card"]
        for col in card_cols:
            if col in team_events.columns:
                col_data = team_events[col].dropna()
                yellow_cards += int(col_data.str.contains("Yellow", na=False).sum())
                red_cards += int(
                    col_data.str.contains("Red|Second Yellow", na=False).sum()
                )

        return {
            "corners": corners,
            "fouls_committed": fouls_committed,
            "fouls_won": fouls_won,
            "yellow_cards": yellow_cards,
            "red_cards": red_cards,
        }

    # -------------------------------------------------------------------------
    # PLAYER STATS
    # -------------------------------------------------------------------------

    def compute_player_stats(
        self, events: pd.DataFrame, team: str, lineups: Dict
    ) -> List[Dict]:
        """Compute per-player metrics.

        Each player's stats include passing, shooting, carrying, and
        defensive metrics. Players with very few minutes are still included
        but flagged via the minutes_played field.

        Returns:
            List of player stat dicts.
        """
        team_events = events[events["team"] == team]
        players = team_events["player"].dropna().unique()

        player_stats: List[Dict] = []
        for player_name in players:
            p_events = team_events[team_events["player"] == player_name]

            # --- Minutes estimation ---
            if "minute" in p_events.columns and len(p_events) > 0:
                first_min = p_events["minute"].min()
                last_min = p_events["minute"].max()
                approx_minutes = float(last_min - first_min) if last_min > first_min else 1.0
            else:
                approx_minutes = 0.0

            # --- Passing ---
            p_passes = p_events[p_events["type"] == "Pass"]
            passes_attempted = len(p_passes)
            pass_outcome = _safe_col(p_passes, "pass_outcome")
            passes_completed = int(pass_outcome.isna().sum())
            pass_pct = (
                round(passes_completed / passes_attempted * 100, 1)
                if passes_attempted > 0
                else 0.0
            )

            # Progressive passes
            completed_passes = p_passes[pass_outcome.isna()]
            prog_passes = 0
            for _, row in completed_passes.iterrows():
                if self._is_progressive_pass(
                    row.get("location"), row.get("pass_end_location")
                ):
                    prog_passes += 1

            # Key passes (shot assists)
            shot_assist = _safe_col(p_passes, "pass_shot_assist")
            key_passes = int((shot_assist == True).sum())  # noqa: E712

            # --- Shooting ---
            p_shots = p_events[p_events["type"] == "Shot"]
            shots = len(p_shots)
            xg_col = _safe_col(p_shots, "shot_statsbomb_xg")
            xg = float(xg_col.sum()) if not xg_col.empty else 0.0
            shot_outcome = _safe_col(p_shots, "shot_outcome")
            goals = int((shot_outcome == "Goal").sum())

            # --- Carrying ---
            p_carries = p_events[p_events["type"] == "Carry"]
            carries = len(p_carries)
            prog_carries = 0
            for _, row in p_carries.iterrows():
                if self._is_progressive_carry(
                    row.get("location"), row.get("carry_end_location")
                ):
                    prog_carries += 1

            # --- Defensive ---
            recoveries = int((p_events["type"] == "Ball Recovery").sum())
            pressures = int((p_events["type"] == "Pressure").sum())
            
            p_duels_events = p_events[p_events["type"] == "Duel"]
            p_tackles = len(p_duels_events)  # Simplified to all duels originally
            interceptions = int((p_events["type"] == "Interception").sum())
            p_duels = p_tackles  # All duels
            
            duel_outcome = _safe_col(p_duels_events, "duel_outcome")
            p_duels_won = int(duel_outcome.isin({"Won", "Success In Play", "Success Out"}).sum())

            # --- Touches ---
            touches = len(p_events[p_events["type"].isin(
                ["Pass", "Shot", "Carry", "Dribble", "Ball Receipt*",
                 "Clearance", "Interception", "Block"]
            )])

            # Touches by zone
            t_atk = t_mid = t_def = 0
            for _, row in p_events.iterrows():
                loc = row.get("location")
                if isinstance(loc, (list, np.ndarray)) and len(loc) >= 2:
                    x = float(loc[0])
                    if x >= 80:
                        t_atk += 1
                    elif x >= 40:
                        t_mid += 1
                    else:
                        t_def += 1

            # Spatial
            on_ball = p_events[p_events["type"].isin(["Pass", "Carry", "Shot", "Dribble", "Ball Receipt*", "Ball Recovery"])]
            x_list, y_list = [], []
            for _, row in on_ball.iterrows():
                loc = row.get("location")
                if isinstance(loc, (list, np.ndarray)) and len(loc) >= 2:
                    x_list.append(float(loc[0]))
                    y_list.append(float(loc[1]))
            avg_x = round(float(np.mean(x_list)), 1) if x_list else 0.0
            avg_y = round(float(np.mean(y_list)), 1) if y_list else 0.0

            player_stats.append({
                "player_name": player_name,
                "minutes_played": approx_minutes,
                "passes_attempted": passes_attempted,
                "passes_completed": passes_completed,
                "pass_completion_percentage": pass_pct,
                "progressive_passes": prog_passes,
                "key_passes": key_passes,
                "goals": goals,
                "shots": shots,
                "expected_goals": round(xg, 4),
                "carries": carries,
                "progressive_carries": prog_carries,
                "recoveries": recoveries,
                "pressures": pressures,
                "tackles": p_tackles,
                "interceptions": interceptions,
                "duels": p_duels,
                "duels_won": p_duels_won,
                "touches": touches,
                "touches_attacking_third": t_atk,
                "touches_middle_third": t_mid,
                "touches_defensive_third": t_def,
                "avg_x": avg_x,
                "avg_y": avg_y,
            })

        return player_stats
