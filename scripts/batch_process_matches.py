"""Batch processing: download, clean, validate, compute features for multiple matches.

Usage:
    py -3.12 scripts/batch_process_matches.py
"""

import json
import sys
import time
from pathlib import Path

from tacticalm.data.download import download_and_cache_match
from tacticalm.data.load import load_match_events, load_match_lineups, load_match_360
from tacticalm.data.clean import clean_events
from tacticalm.data.validate import validate_match
from tacticalm.data.split import split_matches, save_splits, verify_no_leakage
from tacticalm.features.summary import build_match_summary, save_match_summary
from tacticalm.utils.paths import PROCESSED_DIR, ensure_dir

# StatsBomb Open Data: FIFA World Cup 2022
# Get match list from statsbombpy
from statsbombpy import sb


def get_wc2022_matches(max_matches: int = 64) -> list[dict]:
    """Fetch WC 2022 match list from StatsBomb."""
    matches = sb.matches(competition_id=43, season_id=106)
    records = matches.to_dict("records")
    if max_matches:
        records = records[:max_matches]
    return records


def process_one_match(match_record: dict) -> dict:
    """Process a single match through the full pipeline.

    Returns dict with processing results and status.
    """
    match_id = match_record["match_id"]
    home = match_record.get("home_team", "Unknown")
    away = match_record.get("away_team", "Unknown")
    result = {
        "match_id": match_id,
        "home_team": home,
        "away_team": away,
        "status": "unknown",
        "events_count": 0,
        "validation_errors": [],
        "validation_warnings": [],
    }

    try:
        # 1. Download
        download_and_cache_match(match_id)

        # 2. Load
        events = load_match_events(match_id)
        lineups = load_match_lineups(match_id)
        three_sixty = load_match_360(match_id)

        result["events_count"] = len(events)
        result["has_360"] = three_sixty is not None

        # 3. Clean
        events_clean, clean_report = clean_events(events)
        result["clean_report"] = clean_report

        # 4. Validate
        teams = events_clean["team"].dropna().unique().tolist()
        metadata = {
            "match_id": match_id,
            "home_team": teams[0] if len(teams) > 0 else home,
            "away_team": teams[1] if len(teams) > 1 else away,
            "competition": match_record.get("competition", "Unknown"),
            "season": match_record.get("season", "Unknown"),
            "match_date": str(match_record.get("match_date", "Unknown")),
            "score_home": int(match_record.get("home_score", 0)),
            "score_away": int(match_record.get("away_score", 0)),
        }
        val_result = validate_match(match_id, events_clean, lineups, metadata)
        result["validation_errors"] = val_result.errors
        result["validation_warnings"] = val_result.warnings
        result["is_valid"] = val_result.is_valid

        if not val_result.is_valid:
            result["status"] = "validation_failed"
            return result

        # 5. Compute features + save summary
        summary = build_match_summary(
            match_id=match_id,
            events=events_clean,
            lineups=lineups,
            match_metadata=metadata,
            three_sixty=three_sixty,
        )
        output_dir = ensure_dir(PROCESSED_DIR / "summaries")
        save_match_summary(summary, output_dir)
        result["status"] = "success"
        result["summary_path"] = str(output_dir / f"{match_id}_summary.json")

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


def main():
    print("=" * 70)
    print("TacticaLM - Batch Match Processing (FIFA World Cup 2022)")
    print("=" * 70)

    # Get matches
    print("\nFetching match list from StatsBomb...")
    match_records = get_wc2022_matches(max_matches=64)
    print(f"Found {len(match_records)} matches\n")

    # Process all
    results = []
    start = time.time()
    for i, match_rec in enumerate(match_records, 1):
        mid = match_rec["match_id"]
        home = match_rec.get("home_team", "?")
        away = match_rec.get("away_team", "?")
        print(f"[{i}/{len(match_records)}] Processing {home} vs {away} (ID: {mid})...", end=" ")

        r = process_one_match(match_rec)
        results.append(r)

        if r["status"] == "success":
            print(f"OK ({r['events_count']} events)")
        else:
            print(f"FAILED: {r.get('error', r.get('validation_errors', '?'))}")

    elapsed = time.time() - start

    # Summary
    success = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]
    print(f"\n{'=' * 70}")
    print(f"Processed {len(results)} matches in {elapsed:.1f}s")
    print(f"  Success: {len(success)}")
    print(f"  Failed:  {len(failed)}")
    if failed:
        for f in failed:
            print(f"    - {f['match_id']} ({f['home_team']} vs {f['away_team']}): {f['status']}")

    # Save processing report
    report_path = ensure_dir(PROCESSED_DIR) / "batch_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nReport saved to: {report_path}")

    # Three-way split
    valid_ids = [r["match_id"] for r in success]
    print(f"\n{'=' * 70}")
    print(f"Splitting {len(valid_ids)} matches into train/dev_eval/final_test...")

    splits = split_matches(
        valid_ids, seed=42, train_ratio=0.70, dev_eval_ratio=0.15, final_test_ratio=0.15
    )
    splits_dir = ensure_dir(PROCESSED_DIR / "splits")
    save_splits(splits, splits_dir)
    verify_no_leakage(splits)

    print(f"  Train:      {len(splits['train'])} matches")
    print(f"  Dev eval:   {len(splits['dev_eval'])} matches")
    print(f"  Final test: {len(splits['final_test'])} matches")
    print(f"  Splits saved to: {splits_dir}")
    print(f"\n{'=' * 70}")
    print("Batch processing complete.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
