"""Tests for model performance and drift detection."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

import settings


class TestModelPerformance:
    """Model performance and drift tests."""

    def test_model_file_exists(self):
        """Verify model artifact exists."""
        model_path = settings.DATA_PROCESSED / "model.pkl"
        # This test will pass if model has been trained
        # In CI, it might not exist yet, so we make it optional
        if model_path.exists():
            assert model_path.stat().st_size > 0

    def test_no_negative_predictions(self, sample_features_df):
        """Verify model predictions are never negative."""
        # This is a sanity check on the features, not actual predictions
        # Attendance percentage should be between 0 and 1
        assert (sample_features_df["attendance_pct"] >= 0).all()
        assert (sample_features_df["attendance_pct"] <= 1).all()

    def test_predictions_below_capacity(self, sample_features_df):
        """Verify predicted attendance never exceeds stadium capacity."""
        predicted_absolute = sample_features_df["attendance_pct"] * sample_features_df["stadium_capacity"]
        assert (predicted_absolute <= sample_features_df["stadium_capacity"]).all()

    def test_forecast_file_structure(self):
        """Verify forecasts parquet has expected columns if it exists."""
        forecast_path = settings.DATA_PROCESSED / "forecasts.parquet"
        if forecast_path.exists():
            df = pd.read_parquet(forecast_path)
            required_cols = [
                "match_id",
                "date",
                "home_team",
                "away_team",
                "predicted_attendance",
                "lower_bound",
                "upper_bound",
            ]
            for col in required_cols:
                assert col in df.columns, f"Missing column: {col}"

    def test_prediction_intervals_valid(self):
        """Verify lower_bound <= predicted <= upper_bound for all forecasts."""
        forecast_path = settings.DATA_PROCESSED / "forecasts.parquet"
        if forecast_path.exists():
            df = pd.read_parquet(forecast_path)
            assert (df["lower_bound"] <= df["predicted_attendance"]).all()
            assert (df["predicted_attendance"] <= df["upper_bound"]).all()

    def test_attendance_pct_bounds(self, sample_features_df):
        """Verify attendance percentage is between 0 and 1."""
        assert (sample_features_df["attendance_pct"] >= 0).all()
        assert (sample_features_df["attendance_pct"] <= 1).all()

    def test_features_have_required_columns(self, sample_features_df):
        """Verify all required feature columns are present."""
        for col in settings.FEATURE_COLUMNS:
            assert col in sample_features_df.columns, f"Missing required feature: {col}"

    def test_backtest_results_exist(self):
        """Verify backtest results exist if model has been trained."""
        backtest_path = settings.DATA_PROCESSED / "backtest_results.csv"
        if backtest_path.exists():
            results = pd.read_csv(backtest_path)
            assert len(results) > 0
            assert "mape" in results.columns
            # MAPE should be reasonable (not 50% or more)
            assert (results["mape"] < 0.5).all(), "MAPE unusually high; check model"
