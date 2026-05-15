"""Central configuration. All paths and constants live here."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
ROOT = Path(__file__).parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MLRUNS_DIR = ROOT / "mlruns"
NOTEBOOKS_DIR = ROOT / "notebooks"
DASHBOARD_DIR = ROOT / "dashboard"

# Ensure directories exist
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

# API
FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4"
API_RATE_LIMIT_SLEEP = 6  # seconds between requests (free tier: 10 req/min)

# Model
TARGET_COLUMN = "attendance_pct"
FEATURE_COLUMNS = [
    "home_win_rate_last5",
    "away_win_rate_last5",
    "home_goals_scored_avg",
    "away_goals_conceded_avg",
    "is_rivalry",
    "matchday",
    "days_since_last_home_game",
    "season_progress",
    "home_league_position",
    "away_league_position",
    "position_gap",
    "is_weekend",
    "month",
    "stadium_capacity",
]
TRAIN_SEASONS = [2020, 2021, 2022, 2023]
VALIDATION_SEASON = 2024
MAPE_THRESHOLD = 0.15  # Fail tests if MAPE exceeds 15%

# League config
DEFAULT_LEAGUE = "PL"  # Premier League
LEAGUES = {
    "PL": {"name": "Premier League", "teams": 20, "matchdays": 38},
    "BL1": {"name": "Bundesliga", "teams": 18, "matchdays": 34},
    "SA": {"name": "Serie A", "teams": 20, "matchdays": 38},
}

# Stadium capacities (Premier League 2024-25)
STADIUM_CAPACITY = {
    "Arsenal FC": 60704,
    "Aston Villa FC": 42657,
    "Brentford FC": 17250,
    "Brighton and Hove Albion FC": 31876,
    "Chelsea FC": 40341,
    "Crystal Palace FC": 25486,
    "Everton FC": 39572,
    "Fulham FC": 25700,
    "Ipswich Town FC": 30000,
    "Leicester City FC": 32312,
    "Liverpool FC": 61276,
    "Manchester City FC": 53400,
    "Manchester United FC": 74140,
    "Newcastle United FC": 52305,
    "Nottingham Forest FC": 30445,
    "Southampton FC": 32384,
    "Tottenham Hotspur FC": 62850,
    "West Ham United FC": 62500,
    "Wolverhampton Wanderers FC": 31750,
    "AFC Bournemouth": 11307,
}

# Top rivalries (high-attendance fixture bonus)
RIVALRIES = {
    frozenset({"Manchester United FC", "Manchester City FC"}),
    frozenset({"Arsenal FC", "Tottenham Hotspur FC"}),
    frozenset({"Liverpool FC", "Manchester United FC"}),
    frozenset({"Arsenal FC", "Chelsea FC"}),
    frozenset({"Liverpool FC", "Everton FC"}),
    frozenset({"Chelsea FC", "Tottenham Hotspur FC"}),
    frozenset({"Manchester City FC", "Liverpool FC"}),
    frozenset({"Arsenal FC", "Liverpool FC"}),
    frozenset({"Chelsea FC", "Manchester United FC"}),
}

# MLflow
MLFLOW_EXPERIMENT_NAME = "sports-attendance-forecast"
MODEL_NAME = "attendance_forecast_lgb"
