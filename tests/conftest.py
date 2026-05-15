"""
Pytest configuration and shared fixtures for all tests.
"""
import pytest
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

import settings


@pytest.fixture
def sample_matches_df():
    """Create a realistic sample matches DataFrame for testing."""
    dates = pd.date_range("2024-01-01", periods=38, freq="W")
    teams = [
        "Manchester United FC",
        "Manchester City FC",
        "Liverpool FC",
        "Arsenal FC",
        "Chelsea FC",
        "Tottenham Hotspur FC",
    ]

    matches = []
    for i, date in enumerate(dates[:20]):  # First 20 matchdays
        for home_idx, home in enumerate(teams):
            for away_idx, away in enumerate(teams):
                if home_idx < away_idx:
                    matches.append(
                        {
                            "match_id": len(matches) + 1000,
                            "date": date,
                            "matchday": (i % 19) + 1,
                            "season": 2024,
                            "home_team": home,
                            "away_team": away,
                            "home_score": np.random.randint(0, 4),
                            "away_score": np.random.randint(0, 4),
                            "status": "FINISHED",
                            "attendance": np.random.randint(20000, 80000),
                        }
                    )

    return pd.DataFrame(matches)


@pytest.fixture
def sample_features_df():
    """Create a realistic sample features DataFrame for testing."""
    np.random.seed(42)
    n_rows = 100

    return pd.DataFrame(
        {
            "match_id": range(1000, 1000 + n_rows),
            "date": pd.date_range("2024-01-01", periods=n_rows, freq="D"),
            "home_team": [
                np.random.choice(list(settings.STADIUM_CAPACITY.keys())) for _ in range(n_rows)
            ],
            "away_team": [
                np.random.choice(list(settings.STADIUM_CAPACITY.keys())) for _ in range(n_rows)
            ],
            "season": np.random.choice([2023, 2024], n_rows),
            "matchday": np.random.randint(1, 39, n_rows),
            "home_win_rate_last5": np.random.uniform(0, 1, n_rows),
            "away_win_rate_last5": np.random.uniform(0, 1, n_rows),
            "home_goals_scored_avg": np.random.uniform(0.5, 3, n_rows),
            "away_goals_conceded_avg": np.random.uniform(0.5, 3, n_rows),
            "is_rivalry": np.random.choice([True, False], n_rows),
            "days_since_last_home_game": np.random.randint(1, 15, n_rows),
            "season_progress": np.random.uniform(0, 1, n_rows),
            "home_league_position": np.random.randint(1, 21, n_rows),
            "away_league_position": np.random.randint(1, 21, n_rows),
            "position_gap": np.random.randint(0, 20, n_rows),
            "is_weekend": np.random.choice([True, False], n_rows),
            "month": np.random.randint(1, 13, n_rows),
            "stadium_capacity": np.random.choice(list(settings.STADIUM_CAPACITY.values()), n_rows),
            "attendance_pct": np.random.uniform(0, 1, n_rows),
        }
    )
