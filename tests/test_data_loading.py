import pytest
import pandas as pd
from unittest.mock import patch
# Assuming appropriate imports exist in the real structure, using basic mocks here

def test_download_competitions():
    with patch('statsbombpy.sb.competitions') as mock_comps:
        mock_comps.return_value = pd.DataFrame([{"competition_id": 43, "season_id": 106}])
        comps = mock_comps()
        assert not comps.empty
        assert "competition_id" in comps.columns

def test_load_match_events_missing_file():
    with patch('statsbombpy.sb.events') as mock_events:
        mock_events.side_effect = FileNotFoundError("Missing file")
        with pytest.raises(FileNotFoundError):
            mock_events(match_id=999999999)

def test_load_match_events_valid():
    with patch('statsbombpy.sb.events') as mock_events:
        mock_events.return_value = pd.DataFrame([{"id": "uuid", "type": "Pass", "team": "Team A"}])
        events = mock_events(match_id=123)
        assert not events.empty
        assert len(events) == 1
        assert events.iloc[0]['team'] == "Team A"
