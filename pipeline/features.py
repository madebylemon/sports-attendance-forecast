"""
Feature engineering for attendance forecasting.

Constructs 14 hand-crafted features from match history and contextual data.
Every feature has documented sports-domain rationale.

Usage:
    python -m pipeline.features
"""
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

import settings
from pipeline.utils import load_csv_matches, save_parquet

logger = logging.getLogger(__name__)


def load_all_matches() -> pd.DataFrame:
    """Load all historical matches from raw CSV files."""
    raw_dir = settings.DATA_RAW
    csv_files = sorted(raw_dir.glob("matches_*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No match files found in {raw_dir}. "
            "Run 'python -m pipeline.backfill' first."
        )

    dfs = []
    for file in csv_files:
        df = load_csv_matches(file)
        dfs.append(df)
        logger.info(f"Loaded {len(df)} matches from {file.name}")

    all_matches = pd.concat(dfs, ignore_index=True)
    all_matches = all_matches.drop_duplicates(subset=["match_id"], keep="last")
    all_matches = all_matches.sort_values("date").reset_index(drop=True)

    logger.info(f"Total unique matches: {len(all_matches)}")
    return all_matches


def engineer_features(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer 14 features from match data.

    Args:
        matches: DataFrame with raw match data (from load_all_matches)

    Returns:
        DataFrame with engineered features, only for FINISHED matches (no data leakage)
    """
    features_list = []

    for idx, row in matches.iterrows():
        # Skip matches that haven't finished (no actual attendance to learn from)
        if row["status"] != "FINISHED":
            continue

        match_date = pd.to_datetime(row["date"])
        home_team = row["home_team"]
        away_team = row["away_team"]
        match_id = row["match_id"]
        season = row["season"]
        matchday = row["matchday"]

        # Only use matches before current match (no data leakage)
        history = matches[matches["date"] < match_date].copy()

        if len(history) < 1:
            # Not enough history; skip
            continue

        # Feature 1: Home team win rate in last 5 home games
        home_games = history[history["home_team"] == home_team].tail(5)
        if len(home_games) > 0:
            home_wins = (home_games["home_score"] > home_games["away_score"]).sum()
            home_win_rate_last5 = home_wins / len(home_games)
        else:
            home_win_rate_last5 = 0.5

        # Feature 2: Away team win rate in last 5 away games
        away_games = history[history["away_team"] == away_team].tail(5)
        if len(away_games) > 0:
            away_wins = (away_games["away_score"] > away_games["home_score"]).sum()
            away_win_rate_last5 = away_wins / len(away_games)
        else:
            away_win_rate_last5 = 0.5

        # Feature 3: Home team average goals scored (last 5 games, any role)
        home_all_games = history[
            (history["home_team"] == home_team) | (history["away_team"] == home_team)
        ].tail(5)
        if len(home_all_games) > 0:
            goals = []
            for _, g in home_all_games.iterrows():
                if g["home_team"] == home_team:
                    goals.append(g["home_score"])
                else:
                    goals.append(g["away_score"])
            home_goals_scored_avg = np.mean(goals) if goals else 1.5
        else:
            home_goals_scored_avg = 1.5

        # Feature 4: Away team average goals conceded (last 5 games, when home)
        away_home_games = history[history["home_team"] == away_team].tail(5)
        if len(away_home_games) > 0:
            goals_conceded = away_home_games["away_score"].values
            away_goals_conceded_avg = np.mean(goals_conceded)
        else:
            away_goals_conceded_avg = 1.5

        # Feature 5: Is this a rivalry? (high-attendance historical fixture)
        is_rivalry = frozenset({home_team, away_team}) in settings.RIVALRIES

        # Feature 6: Matchday (1-38 in league season) — higher late-season = potentially more drama
        season_progress = matchday / settings.LEAGUES[settings.DEFAULT_LEAGUE]["matchdays"]

        # Feature 7: Days since last home game for home team (rest/fatigue indicator)
        last_home_game = history[history["home_team"] == home_team]
        if len(last_home_game) > 0:
            last_date = last_home_game.iloc[-1]["date"]
            days_since_last_home = (match_date - pd.to_datetime(last_date)).days
        else:
            days_since_last_home = 7

        # Feature 8: Season progress (0.0 to 1.0)
        # (already computed above)

        # Feature 9: Home league position (estimated from win rate as proxy)
        home_position = max(1, 21 - int(home_win_rate_last5 * 20))

        # Feature 10: Away league position
        away_position = max(1, 21 - int(away_win_rate_last5 * 20))

        # Feature 11: Position gap (distance between teams in table)
        position_gap = abs(home_position - away_position)

        # Feature 12: Is weekend? (Sat/Sun matches typically have higher attendance)
        is_weekend = match_date.weekday() in [5, 6]  # 5=Sat, 6=Sun

        # Feature 13: Month (seasonal variation in attendance)
        month = match_date.month

        # Feature 14: Stadium capacity (home team's ground)
        stadium_capacity = settings.STADIUM_CAPACITY.get(home_team, 50000)

        # Target variable: attendance percentage
        attendance_pct = min(1.0, max(0.0, row["attendance"] / stadium_capacity))

        features_list.append(
            {
                "match_id": match_id,
                "date": match_date,
                "home_team": home_team,
                "away_team": away_team,
                "season": season,
                "matchday": matchday,
                "home_win_rate_last5": home_win_rate_last5,
                "away_win_rate_last5": away_win_rate_last5,
                "home_goals_scored_avg": home_goals_scored_avg,
                "away_goals_conceded_avg": away_goals_conceded_avg,
                "is_rivalry": is_rivalry,
                "matchday": matchday,
                "days_since_last_home_game": days_since_last_home,
                "season_progress": season_progress,
                "home_league_position": home_position,
                "away_league_position": away_position,
                "position_gap": position_gap,
                "is_weekend": is_weekend,
                "month": month,
                "stadium_capacity": stadium_capacity,
                "attendance_pct": attendance_pct,
            }
        )

    features_df = pd.DataFrame(features_list)
    logger.info(f"Engineered {len(features_df)} feature rows")
    return features_df


def main():
    logger.info("Loading matches...")
    matches = load_all_matches()

    logger.info("Engineering features...")
    features = engineer_features(matches)

    output_file = settings.DATA_PROCESSED / "features.parquet"
    save_parquet(features, output_file)

    logger.info(f"✓ Feature engineering complete: {len(features)} rows, {len(settings.FEATURE_COLUMNS)} features")


if __name__ == "__main__":
    main()
