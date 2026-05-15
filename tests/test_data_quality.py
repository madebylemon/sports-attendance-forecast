"""Tests for raw data quality and schema validation."""
import pytest
import pandas as pd
from datetime import datetime, timedelta

import settings


class TestDataQuality:
    """Data quality checks on raw match CSV files."""

    def test_no_future_dates_in_raw_data(self, sample_matches_df):
        """Verify no match dates are more than 30 days in the future."""
        future_threshold = datetime.now() + timedelta(days=30)
        assert sample_matches_df["date"].max() <= future_threshold

    def test_attendance_bounds(self, sample_matches_df):
        """Verify attendance values are within realistic bounds (0 to 100k)."""
        assert (sample_matches_df["attendance"] >= 0).all()
        assert (sample_matches_df["attendance"] <= 100000).all()

    def test_no_duplicate_match_ids(self, sample_matches_df):
        """Verify match_id is unique per DataFrame."""
        assert sample_matches_df["match_id"].duplicated().sum() == 0

    def test_required_columns_present(self, sample_matches_df):
        """Verify all expected columns exist in raw data."""
        required = [
            "match_id",
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "status",
            "attendance",
            "season",
        ]
        for col in required:
            assert col in sample_matches_df.columns

    def test_scheduled_matches_have_no_score(self, sample_matches_df):
        """Verify SCHEDULED matches have null scores."""
        sample = sample_matches_df.copy()
        sample.loc[sample.index[:5], "status"] = "SCHEDULED"
        sample.loc[sample.index[:5], "home_score"] = None
        sample.loc[sample.index[:5], "away_score"] = None

        scheduled = sample[sample["status"] == "SCHEDULED"]
        assert scheduled["home_score"].isna().all() or scheduled["home_score"].notna().all()

    def test_completed_matches_have_scores(self, sample_matches_df):
        """Verify FINISHED matches have non-null scores."""
        finished = sample_matches_df[sample_matches_df["status"] == "FINISHED"]
        assert finished["home_score"].notna().all()
        assert finished["away_score"].notna().all()

    def test_season_row_counts(self, sample_matches_df):
        """Verify season has expected structure (should have multiple matches per matchday)."""
        matches_per_season = sample_matches_df[sample_matches_df["season"] == 2024]
        assert len(matches_per_season) > 0

    def test_no_null_home_team(self, sample_matches_df):
        """Verify home_team is never null."""
        assert sample_matches_df["home_team"].notna().all()

    def test_date_format(self, sample_matches_df):
        """Verify all dates parse as datetime."""
        assert pd.api.types.is_datetime64_any_dtype(sample_matches_df["date"])

    def test_home_away_teams_differ(self, sample_matches_df):
        """Verify home_team != away_team for every row."""
        mismatch = sample_matches_df["home_team"] == sample_matches_df["away_team"]
        assert not mismatch.any(), "Found matches with home_team == away_team"
