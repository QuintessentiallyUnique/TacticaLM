from statsbombpy import sb

matches_found = []
comps = sb.competitions()
for _, row in comps.iterrows():
    comp_id = row['competition_id']
    season_id = row['season_id']
    try:
        matches = sb.matches(competition_id=comp_id, season_id=season_id)
        for _, m in matches.iterrows():
            if 'England' in [m['home_team'], m['away_team']] and 'Argentina' in [m['home_team'], m['away_team']]:
                matches_found.append(f"{m['home_team']} vs {m['away_team']} (Match ID: {m['match_id']}, Competition: {row['competition_name']} {row['season_name']})")
    except:
        pass

if not matches_found:
    print("No matches found.")
else:
    for match in matches_found:
        print(match)
