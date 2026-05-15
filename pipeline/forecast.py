"""
Generate attendance forecasts for upcoming fixtures.

Usage:
    python -m pipeline.forecast
"""
import logging
from datetime import datetime

import pandas as pd
import numpy as np
import joblib

import settings
from pipeline.utils import load_csv_matches, save_parquet

logger = logging.getLogger(__name__)


def get_upcoming_fixtures(current_season_file: str) -> pd.DataFrame:
    """Get all SCHEDULED matches from the current season."""
    if not current_season_file.exists():
        logger.warning(f"Current season file not found: {current_season_file}")
        return pd.DataFrame()

    df = load_csv_matches(current_season_file)
    upcoming = df[df["status"] == "SCHEDULED"].copy()
    logger.info(f"Found {len(upcoming)} upcoming fixtures")
    return upcoming


def build_forecast_features(upcoming: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Build features for upcoming fixtures using latest team stats from history."""
    forecast_features = []

    for _, fixture in upcoming.iterrows():
        home_team = fixture["home_team"]
        away_team = fixture["away_team"]
        match_id = fixture["match_id"]
        match_date = pd.to_datetime(fixture["date"])
        matchday = fixture["matchday"]

        # Use all historical data (no date cutoff, since match hasn't happened yet)
        hist_home = history[history["home_team"] == home_team]
        hist_away = history[history["away_team"] == away_team]
        hist_all = history.copy()

        # Feature 1: Home team win rate (last 5 home games)
        last_5_home = hist_home.tail(5)
        if len(last_5_home) > 0:
            home_wins = (last_5_home["home_score"] > last_5_home["away_score"]).sum()
            home_win_rate = home_wins / len(last_5_home)
        else:
            home_win_rate = 0.5

        # Feature 2: Away team win rate (last 5 away games)
        last_5_away = hist_away.tail(5)
        if len(last_5_away) > 0:
            away_wins = (last_5_away["away_score"] > last_5_away["home_score"]).sum()
            away_win_rate = away_wins / len(last_5_away)
        else:
            away_win_rate = 0.5

        # Feature 3: Home goals scored average
        home_all = history[(history["home_team"] == home_team) | (history["away_team"] == home_team)].tail(5)
        if len(home_all) > 0:
            goals = []
            for _, g in home_all.iterrows():
                if g["home_team"] == home_team:
                    goals.append(g["home_score"])
                else:
                    goals.append(g["away_score"])
            home_goals_avg = np.mean(goals)
        else:
            home_goals_avg = 1.5

        # Feature 4: Away goals conceded average
        away_home = history[history["home_team"] == away_team].tail(5)
        if len(away_home) > 0:
            goals_conceded = away_home["away_score"].values
            away_goals_conceded = np.mean(goals_conceded)
        else:
            away_goals_conceded = 1.5

        # Feature 5: Is rivalry
        is_rivalry = frozenset({home_team, away_team}) in settings.RIVALRIES

        # Feature 6-8: Matchday, season progress, days since last home
        season_progress = matchday / settings.LEAGUES[settings.DEFAULT_LEAGUE]["matchdays"]
        last_home = history[history["home_team"] == home_team]
        if len(last_home) > 0:
            last_date = pd.to_datetime(last_home.iloc[-1]["date"])
            days_since = (match_date - last_date).days
        else:
            days_since = 7

        # Feature 9-11: Positions
        home_pos = max(1, 21 - int(home_win_rate * 20))
        away_pos = max(1, 21 - int(away_win_rate * 20))
        pos_gap = abs(home_pos - away_pos)

        # Feature 12-13: Weekend, month
        is_weekend = match_date.weekday() in [5, 6]
        month = match_date.month

        # Feature 14: Stadium capacity
        capacity = settings.STADIUM_CAPACITY.get(home_team, 50000)

        forecast_features.append(
            {
                "match_id": match_id,
                "date": match_date,
                "home_team": home_team,
                "away_team": away_team,
                "matchday": matchday,
                "home_win_rate_last5": home_win_rate,
                "away_win_rate_last5": away_win_rate,
                "home_goals_scored_avg": home_goals_avg,
                "away_goals_conceded_avg": away_goals_conceded,
                "is_rivalry": is_rivalry,
                "days_since_last_home_game": days_since,
                "season_progress": season_progress,
                "home_league_position": home_pos,
                "away_league_position": away_pos,
                "position_gap": pos_gap,
                "is_weekend": is_weekend,
                "month": month,
                "stadium_capacity": capacity,
            }
        )

    return pd.DataFrame(forecast_features)


def generate_forecasts():
    """Load model and generate predictions for upcoming fixtures."""
    model_path = settings.DATA_PROCESSED / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run 'python -m pipeline.train' first."
        )

    model = joblib.load(model_path)
    logger.info("Loaded model")

    # Get current season file
    today = datetime.now().date()
    current_year = today.year
    current_season = current_year if today.month >= 8 else current_year - 1
    season_file = settings.DATA_RAW / f"matches_{current_season}_{current_season + 1}.csv"

    if not season_file.exists():
        logger.warning(f"Season file not found: {season_file}")
        return pd.DataFrame()

    # Load all historical data (for building features)
    from pathlib import Path
    raw_dir = settings.DATA_RAW
    csv_files = sorted(raw_dir.glob("matches_*.csv"))
    dfs = [pd.read_csv(f) for f in csv_files]
    all_history = pd.concat(dfs, ignore_index=True)
    all_history = all_history.drop_duplicates(subset=["match_id"], keep="last")
    all_history["date"] = pd.to_datetime(all_history["date"])

    # Get upcoming fixtures
    upcoming = get_upcoming_fixtures(season_file)
    if len(upcoming) == 0:
        logger.warning("No upcoming fixtures found")
        return pd.DataFrame()

    # Build features
    forecast_df = build_forecast_features(upcoming, all_history)

    # Make predictions
    X_forecast = forecast_df[settings.FEATURE_COLUMNS]
    predictions = model.predict(X_forecast)

    # Simple prediction intervals: ±15% around prediction (can be improved with quantile regression)
    forecast_df["predicted_attendance_pct"] = predictions
    forecast_df["lower_bound_pct"] = np.maximum(0, predictions - 0.15)
    forecast_df["upper_bound_pct"] = np.minimum(1.0, predictions + 0.15)

    # Convert back to absolute attendance
    forecast_df["predicted_attendance"] = forecast_df["predicted_attendance_pct"] * forecast_df["stadium_capacity"]
    forecast_df["lower_bound"] = forecast_df["lower_bound_pct"] * forecast_df["stadium_capacity"]
    forecast_df["upper_bound"] = forecast_df["upper_bound_pct"] * forecast_df["stadium_capacity"]

    # Add metadata
    forecast_df["model_version"] = "lgb_v1"
    forecast_df["forecast_timestamp"] = datetime.now().isoformat()

    # Select columns for output
    output_df = forecast_df[
        [
            "match_id",
            "date",
            "home_team",
            "away_team",
            "matchday",
            "predicted_attendance",
            "lower_bound",
            "upper_bound",
            "predicted_attendance_pct",
            "model_version",
            "forecast_timestamp",
        ]
    ].copy()

    output_path = settings.DATA_PROCESSED / "forecasts.parquet"
    save_parquet(output_df, output_path)

    logger.info(f"✓ Generated forecasts for {len(output_df)} fixtures")
    return output_df


def main():
    forecasts = generate_forecasts()
    if len(forecasts) > 0:
        print("\nTop 5 forecasted fixtures (by predicted attendance):")
        top = forecasts.nlargest(5, "predicted_attendance")[
            ["date", "home_team", "away_team", "predicted_attendance", "upper_bound"]
        ]
        print(top.to_string(index=False))


if __name__ == "__main__":
    main()
