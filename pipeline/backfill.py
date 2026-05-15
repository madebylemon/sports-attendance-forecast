"""
Backfill historical match data for the last 5 seasons.

Usage:
    python -m pipeline.backfill                    # Backfill PL, seasons 2020-2024
    python -m pipeline.backfill --league BL1       # Backfill Bundesliga
    python -m pipeline.backfill --seasons 2022,2023,2024  # Custom seasons
    python -m pipeline.backfill --force            # Re-download existing files
"""
import argparse
import logging
from pathlib import Path

import pandas as pd

import settings
from pipeline.utils import fetch_matches, parse_matches_response, save_csv_matches

logger = logging.getLogger(__name__)


def backfill_season(
    league: str, season: int, force: bool = False
) -> pd.DataFrame:
    """
    Fetch all matches for a given league and season.

    Args:
        league: League code (e.g., 'PL')
        season: Season year (e.g., 2023 for 2023-24)
        force: If True, re-download even if file exists

    Returns:
        DataFrame of matches
    """
    output_file = settings.DATA_RAW / f"matches_{season}_{season + 1}.csv"

    if output_file.exists() and not force:
        logger.info(f"File {output_file} exists. Skipping (use --force to re-download)")
        return pd.read_csv(output_file)

    logger.info(f"Fetching matches for {league} {season}-{season + 1}...")
    response = fetch_matches(league=league, season=season)
    df = parse_matches_response(response)

    save_csv_matches(df, output_file)
    logger.info(f"Saved {len(df)} matches to {output_file}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Backfill historical match data from football-data.org"
    )
    parser.add_argument(
        "--league",
        default=settings.DEFAULT_LEAGUE,
        help=f"League code (default: {settings.DEFAULT_LEAGUE})",
    )
    parser.add_argument(
        "--seasons",
        default="2020,2021,2022,2023,2024",
        help="Comma-separated season years (default: 2020,2021,2022,2023,2024)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they exist",
    )

    args = parser.parse_args()
    seasons = list(map(int, args.seasons.split(",")))

    logger.info(f"Backfilling {args.league} for seasons {seasons}")

    total_matches = 0
    for season in seasons:
        df = backfill_season(league=args.league, season=season, force=args.force)
        total_matches += len(df)

    logger.info(f"✓ Backfill complete: {total_matches} total matches across {len(seasons)} seasons")


if __name__ == "__main__":
    main()
