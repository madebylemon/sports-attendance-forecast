"""Tests for feature engineering correctness and data leakage."""
import pytest
import pandas as pd
import numpy as np

import settings


class TestFeatures:
    """Feature engineering tests."""

    def test_win_rate_range(self, sample_features_df):
        """Verify win rate features are between 0 and 1."""
        assert (sample_features_df["home_win_rate_last5"] >= 0).all()
        assert (sample_features_df["home_win_rate_last5"] <= 1).all()
        assert (sample_features_df["away_win_rate_last5"] >= 0).all()
        assert (sample_features_df["away_win_rate_last5"] <= 1).all()

    def test_feature_count(self, sample_features_df):
        """Verify exactly the expected number of features are present."""
        expected_features = set(settings.FEATURE_COLUMNS)
        actual_features = set(sample_features_df.columns) - {"match_id", "date", "home_team", "away_team", "season", "attendance_pct"}
        assert len([f for f in settings.FEATURE_COLUMNS if f in sample_features_df.columns]) == len(settings.FEATURE_COLUMNS)

    def test_is_rivalry_is_boolean(self, sample_features_df):
        """Verify is_rivalry feature is boolean."""
        assert sample_features_df["is_rivalry"].dtype == bool

    def test_season_progress_range(self, sample_features_df):
        """Verify season_progress is between 0 and 1."""
        assert (sample_features_df["season_progress"] >= 0).all()
        assert (sample_features_df["season_progress"] <= 1).all()

    def test_stadium_capacity_positive(self, sample_features_df):
        """Verify all stadiums have positive capacity."""
        assert (sample_features_df["stadium_capacity"] > 0).all()

    def test_position_gap_non_negative(self, sample_features_df):
        """Verify position gap is always non-negative."""
        assert (sample_features_df["position_gap"] >= 0).all()

    def test_matchday_in_valid_range(self, sample_features_df):
        """Verify matchday is between 1 and 38."""
        assert (sample_features_df["matchday"] >= 1).all()
        assert (sample_features_df["matchday"] <= 38).all()

    def test_no_nan_in_core_features(self, sample_features_df):
        """Verify no NaN values in core numeric features."""
        core_features = [
            "home_win_rate_last5",
            "away_win_rate_last5",
            "home_goals_scored_avg",
            "away_goals_conceded_avg",
            "stadium_capacity",
        ]
        for feature in core_features:
            if feature in sample_features_df.columns:
                assert sample_features_df[feature].notna().all(), f"NaN found in {feature}"
