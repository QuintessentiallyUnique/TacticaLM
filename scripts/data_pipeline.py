import argparse
import sys
import os
from statsbombpy import sb
from tacticalm.utils.paths import ensure_dir, PROCESSED_DIR

# Add current directory to path so we can import from batch_process_matches
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def process_single_match(match_id: int):
    print(f"Processing Match ID: {match_id}")
    from batch_process_matches import process_one_match
    
    match_record = {"match_id": match_id}
    try:
        result = process_one_match(match_record)
        if result["status"] == "success":
            print(f"Successfully processed match {match_id}. Summary saved to {result['summary_path']}")
        else:
            print(f"Failed to process match {match_id}: {result.get('error', result.get('validation_errors'))}")
    except Exception as e:
        print(f"Error processing match: {e}")

def process_competition_season(comp_id: int, season_id: int):
    print(f"Fetching matches for Competition {comp_id}, Season {season_id}...")
    try:
        matches = sb.matches(competition_id=comp_id, season_id=season_id)
        records = matches.to_dict("records")
        print(f"Found {len(records)} matches.")
        
        from batch_process_matches import process_one_match
        
        for i, match_rec in enumerate(records, 1):
            mid = match_rec["match_id"]
            home = match_rec.get("home_team", "?")
            away = match_rec.get("away_team", "?")
            print(f"[{i}/{len(records)}] Processing {home} vs {away} (ID: {mid})...", end=" ")
            
            r = process_one_match(match_rec)
            if r["status"] == "success":
                print(f"OK")
            else:
                print(f"FAILED: {r.get('error', r.get('validation_errors', '?'))}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TacticaLM Data Pipeline CLI")
    parser.add_argument("--match-id", type=int, help="Process a specific StatsBomb Match ID")
    parser.add_argument("--comp-id", type=int, help="Competition ID (requires --season-id)")
    parser.add_argument("--season-id", type=int, help="Season ID (requires --comp-id)")
    
    args = parser.parse_args()
    
    if args.match_id:
        process_single_match(args.match_id)
    elif args.comp_id and args.season_id:
        process_competition_season(args.comp_id, args.season_id)
    else:
        parser.print_help()
