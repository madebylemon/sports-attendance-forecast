"""Shared utilities for the data pipeline: API client, logging, file I/O."""
import time
import logging
from typing import Optional, Dict, Any
from pathlib import Path

import requests
import pandas as pd
import pyarrow.parquet as pq

import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


def get_football_data_api_client() -> requests.Session:
    """Create a requests session with football-data.org auth header."""
    session = requests.Session()
    if not settings.FOOTBALL_DATA_API_KEY:
        raise ValueError(
            "FOOTBALL_DATA_API_KEY not set. "
            "Please set it in .env file. "
            "Get a free key at https://www.football-data.org/client/register"
        )
    session.headers.update({"X-Auth-Token": settings.FOOTBALL_DATA_API_KEY})
    return session


def fetch_matches(
    league: str = settings.DEFAULT_LEAGUE,
    season: Optional[int] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fetch matches from football-data.org API.

    Args:
        league: League code (e.g., 'PL' for Premier League)
        season: Season year (e.g., 2023 for 2023-24 season) — if None, uses current
        date_from: ISO date string to filter from (e.g., '2024-01-01')
        date_to: ISO date string to filter to (e.g., '2024-01-31')

    Returns:
        API response as dict
    """
    session = get_football_data_api_client()
    url = f"{settings.FOOTBALL_DATA_BASE_URL}/competitions/{league}/matches"

    params = {}
    if season is not None:
        params["season"] = season
    if date_from:
        params["dateFrom"] = date_from
    if date_to:
        params["dateTo"] = date_to

    logger.info(f"Fetching {url} with params {params}")
    time.sleep(settings.API_RATE_LIMIT_SLEEP)  # Rate limit: 10 req/min

    response = session.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def parse_matches_response(response: Dict[str, Any]) -> pd.DataFrame:
    """
    Parse football-data.org matches API response into DataFrame.

    Handles missing attendance values by using stadium capacity as a fallback.
    """
    matches = []
    for match in response.get("matches", []):
        home_team = match["homeTeam"]["name"]
        away_team = match["awayTeam"]["name"]

        # Attendance: use actual if available, else estimate from capacity
        attendance = match.get("attendance")
        if attendance is None:
            # Free tier limitation: estimate using stadium capacity
            capacity = settings.STADIUM_CAPACITY.get(home_team, 50000)
            # Assume typical 70% utilization for estimation
            attendance = int(capacity * 0.70)

        matches.append(
            {
                "match_id": match["id"],
                "date": pd.to_datetime(match["utcDate"]),
                "matchday": match["season"]["currentMatchday"],
                "season": match["season"]["id"] // 10000,  # e.g., 20242 → 2024
                "home_team": home_team,
                "away_team": away_team,
                "home_score": match["score"]["fullTime"]["home"],
                "away_score": match["score"]["fullTime"]["away"],
                "status": match["status"],
                "attendance": attendance,
            }
        )

    df = pd.DataFrame(matches)
    logger.info(f"Parsed {len(df)} matches from API response")
    return df


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="snappy")
    logger.info(f"Saved {len(df)} rows to {path}")


def load_parquet(path: Path) -> pd.DataFrame:
    """Load DataFrame from parquet file."""
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} rows from {path}")
    return df


def load_csv_matches(path: Path) -> pd.DataFrame:
    """Load matches CSV, ensuring proper date parsing and types."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def save_csv_matches(df: pd.DataFrame, path: Path) -> None:
    """Save matches to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved {len(df)} rows to {path}")
