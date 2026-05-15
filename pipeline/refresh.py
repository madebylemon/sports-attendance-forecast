"""
Daily incremental data refresh: fetch matches from the last 7 days and next 14 days.

Usage:
    python -m pipeline.refresh    # Refresh current season data
"""
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

import settings
from pipeline.utils import fetch_matches, parse_matches_response, load_csv_matches, save_csv_matches

logger = logging.getLogger(__name__)


def refresh_current_season(league: str = settings.DEFAULT_LEAGUE) -> pd.DataFrame:
    """
    Fetch recent and upcoming matches for the current season.
    Upserts into the current season CSV.

    Args:
        league: League code (e.g., 'PL')

    Returns:
        Updated DataFrame for current season
    """
    today = datetime.now().date()
    date_from = (today - timedelta(days=7)).isoformat()
    date_to = (today + timedelta(days=14)).isoformat()

    logger.info(f"Refreshing {league} matches from {date_from} to {date_to}")

    response = fetch_matches(league=league, date_from=date_from, date_to=date_to)
    new_matches = parse_matches_response(response)

    # Determine current season (assume ongoing season if between Aug and Jul)
    current_year = today.year
    current_season = current_year if today.month >= 8 else current_year - 1

    season_file = settings.DATA_RAW / f"matches_{current_season}_{current_season + 1}.csv"

    if season_file.exists():
        existing = load_csv_matches(season_file)
        # Upsert: remove any existing match_ids from new_matches, then concat
        upserted = pd.concat(
            [
                existing[~existing["match_id"].isin(new_matches["match_id"])],
                new_matches,
            ],
            ignore_index=True,
        )
        upserted = upserted.drop_duplicates(subset=["match_id"], keep="last")
        upserted = upserted.sort_values("date").reset_index(drop=True)
        logger.info(
            f"Upserted {len(new_matches)} matches into season file. "
            f"Total now: {len(upserted)} matches"
        )
    else:
        upserted = new_matches
        logger.info(f"Created new season file with {len(upserted)} matches")

    save_csv_matches(upserted, season_file)
    return upserted


def main():
    refresh_current_season()
    logger.info("✓ Daily refresh complete")


if __name__ == "__main__":
    main()
