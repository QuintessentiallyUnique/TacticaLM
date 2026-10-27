"""Vertical slice test: download, clean, validate, compute features, and save one match."""

import sys

from tacticalm.data.download import download_and_cache_match
from tacticalm.data.load import load_match_events, load_match_lineups, load_match_360
from tacticalm.data.clean import clean_events
from tacticalm.data.validate import validate_match
from tacticalm.features.summary import (
    build_match_summary,
    generate_evidence_statements,
    save_match_summary,
)

# World Cup 2022 Final: Argentina vs France (match_id 3869685)
MATCH_ID = 3869685


def main():
    print("=== Step 1: Download and cache match data ===")
    paths = download_and_cache_match(MATCH_ID)
    for k, v in paths.items():
        print(f"  {k}: {v}")

    print("\n=== Step 2: Load match data ===")
    events = load_match_events(MATCH_ID)
    lineups = load_match_lineups(MATCH_ID)
    three_sixty = load_match_360(MATCH_ID)
    cols_preview = events.columns.tolist()[:10]
    print(f"  Events: {len(events)} rows, columns: {cols_preview}...")
    print(f"  Lineups: {list(lineups.keys())}")
    has_360 = "available" if three_sixty is not None else "not available"
    print(f"  360 data: {has_360}")

    print("\n=== Step 3: Clean events ===")
    events_clean, report = clean_events(events)
    print(f"  Cleaning report: {report}")

    print("\n=== Step 4: Validate ===")
    teams = events_clean["team"].dropna().unique().tolist()
    metadata = {
        "match_id": MATCH_ID,
        "home_team": teams[0] if len(teams) > 0 else "Unknown",
        "away_team": teams[1] if len(teams) > 1 else "Unknown",
        "competition": "FIFA World Cup",
        "season": "2022",
        "match_date": "2022-12-18",
        "score_home": 3,
        "score_away": 3,
    }
    result = validate_match(MATCH_ID, events_clean, lineups, metadata)
    print(f"  Valid: {result.is_valid}")
    print(f"  Errors: {result.errors}")
    print(f"  Warnings: {result.warnings}")
    print(f"  Stats: {result.stats}")

    print("\n=== Step 5: Compute features and build summary ===")
    summary = build_match_summary(
        match_id=MATCH_ID,
        events=events_clean,
        lineups=lineups,
        match_metadata=metadata,
        three_sixty=three_sixty,
    )
    print(f"  Teams: {list(summary.team_stats.keys())}")
    for team_name, stats in summary.team_stats.items():
        print(f"\n  --- {team_name} ---")
        print(f"    Possession: {stats.possession_percentage}%")
        print(
            f"    Passes: {stats.completed_passes}/{stats.total_passes} "
            f"({stats.pass_completion_percentage}%)"
        )
        print(f"    Progressive passes: {stats.progressive_passes}")
        print(f"    Shots: {stats.shots} (on target: {stats.shots_on_target})")
        print(f"    Goals: {stats.goals}")
        print(
            f"    xG: {stats.expected_goals:.2f} "
            f"(open-play: {stats.open_play_xg:.2f}, "
            f"set-piece: {stats.set_piece_xg:.2f})"
        )
        print(f"    High-xG chances: {stats.high_xg_chances}")
        print(f"    Final-third entries: {stats.final_third_entries}")
        print(f"    Penalty-area entries: {stats.penalty_area_entries}")
        print(f"    Pressures: {stats.pressures}")
        print(f"    High regains: {stats.high_regains}")
        print(f"    PPDA: {stats.ppda:.2f}")
        print(f"    Tackles: {stats.tackles} (won: {stats.tackles_won})")
        print(f"    Interceptions: {stats.interceptions}")
        print(f"    Counterattacks: {stats.counterattacks}")
        print(f"    Turnovers: {stats.turnovers}")

    print("\n=== Step 6: Generate evidence statements ===")
    for team_name in summary.team_stats.keys():
        statements = generate_evidence_statements(summary, team_name)
        print(f"\n  {team_name} ({len(statements)} statements):")
        for s in statements[:8]:
            print(f"    * {s}")
        if len(statements) > 8:
            print(f"    ... and {len(statements) - 8} more")

    print("\n=== Step 7: Save summary ===")
    output = save_match_summary(summary, "data/processed")
    print(f"  Saved to: {output}")

    print("\n=== VERTICAL SLICE COMPLETE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
