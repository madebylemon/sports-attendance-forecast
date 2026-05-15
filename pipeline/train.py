"""
Train LightGBM model for attendance forecasting and log to MLflow.

Usage:
    python -m pipeline.train
"""
import logging
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import joblib

import settings
from pipeline.utils import load_parquet

logger = logging.getLogger(__name__)


def load_training_data() -> tuple:
    """Load features and split into train/val sets."""
    features_path = settings.DATA_PROCESSED / "features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(
            f"Features file not found: {features_path}. "
            "Run 'python -m pipeline.features' first."
        )

    df = load_parquet(features_path)

    # Split: train on 2020-2023, validate on 2024
    train_df = df[df["season"].isin(settings.TRAIN_SEASONS)].copy()
    val_df = df[df["season"] == settings.VALIDATION_SEASON].copy()

    logger.info(
        f"Train set: {len(train_df)} matches (seasons {settings.TRAIN_SEASONS})"
    )
    logger.info(f"Validation set: {len(val_df)} matches (season {settings.VALIDATION_SEASON})")

    return train_df, val_df


def train_model(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """
    Train LightGBM model and log to MLflow.

    Returns:
        (model, metrics_dict)
    """
    # MLflow setup
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
    mlflow.start_run()

    try:
        X_train = train_df[settings.FEATURE_COLUMNS].copy()
        y_train = train_df[settings.TARGET_COLUMN].copy()
        X_val = val_df[settings.FEATURE_COLUMNS].copy()
        y_val = val_df[settings.TARGET_COLUMN].copy()

        # Model hyperparameters
        params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbose": -1,
        }

        logger.info(f"Training LightGBM with params: {params}")

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="mae",
            callbacks=[lgb.log_evaluation(period=50)],
        )

        # Predictions
        y_pred_val = model.predict(X_val)

        # Metrics
        mape = mean_absolute_percentage_error(y_val, y_pred_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        mae = mean_absolute_error(y_val, y_pred_val)

        logger.info(f"Validation MAPE: {mape:.4f}")
        logger.info(f"Validation RMSE: {rmse:.4f}")
        logger.info(f"Validation MAE: {mae:.4f}")

        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("val_mape", mape)
        mlflow.log_metric("val_rmse", rmse)
        mlflow.log_metric("val_mae", mae)
        mlflow.log_metric("train_size", len(train_df))
        mlflow.log_metric("val_size", len(val_df))

        # Log feature importances
        importances = pd.DataFrame(
            {
                "feature": settings.FEATURE_COLUMNS,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        logger.info("Feature importances:")
        logger.info(importances.to_string(index=False))

        # Save artifacts
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        top_n = 12
        top_features = importances.head(top_n)
        ax.barh(range(len(top_features)), top_features["importance"])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features["feature"])
        ax.set_xlabel("Importance")
        ax.set_title("Top Feature Importances")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig("/tmp/feature_importance.png", dpi=100, bbox_inches="tight")
        mlflow.log_artifact("/tmp/feature_importance.png")
        plt.close()

        # Predictions vs actuals scatter
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_val, y_pred_val, alpha=0.5, s=20)
        ax.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], "r--", lw=2)
        ax.set_xlabel("Actual Attendance %")
        ax.set_ylabel("Predicted Attendance %")
        ax.set_title("Predictions vs Actual (Validation Set)")
        plt.tight_layout()
        plt.savefig("/tmp/predictions_vs_actuals.png", dpi=100, bbox_inches="tight")
        mlflow.log_artifact("/tmp/predictions_vs_actuals.png")
        plt.close()

        # Log model
        mlflow.lightgbm.log_model(model, "model")

        metrics = {"mape": mape, "rmse": rmse, "mae": mae}

        return model, metrics

    except Exception as e:
        mlflow.end_run(status="FAILED")
        raise e

    finally:
        mlflow.end_run()


def backtest(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling-origin backtest: train on seasons N, test on season N+1.

    Returns:
        DataFrame with backtest results per season
    """
    backtest_results = []

    for i, test_season in enumerate(settings.TRAIN_SEASONS[1:]):
        train_seasons = settings.TRAIN_SEASONS[: i + 1]
        train_df = features_df[features_df["season"].isin(train_seasons)]
        test_df = features_df[features_df["season"] == test_season]

        if len(test_df) == 0:
            continue

        X_train = train_df[settings.FEATURE_COLUMNS]
        y_train = train_df[settings.TARGET_COLUMN]
        X_test = test_df[settings.FEATURE_COLUMNS]
        y_test = test_df[settings.TARGET_COLUMN]

        model = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            random_state=42,
            verbose=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        backtest_results.append(
            {
                "train_seasons": str(train_seasons),
                "test_season": test_season,
                "test_size": len(test_df),
                "mape": mape,
            }
        )
        logger.info(f"Backtest: train on {train_seasons}, test on {test_season} → MAPE {mape:.4f}")

    return pd.DataFrame(backtest_results)


def main():
    logger.info("Loading training data...")
    train_df, val_df = load_training_data()

    logger.info("Training model...")
    model, metrics = train_model(train_df, val_df)

    # Save model
    model_path = settings.DATA_PROCESSED / "model.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Saved model to {model_path}")

    # Run backtest
    logger.info("Running backtest...")
    all_features = load_parquet(settings.DATA_PROCESSED / "features.parquet")
    backtest_results = backtest(all_features)
    backtest_results.to_csv(settings.DATA_PROCESSED / "backtest_results.csv", index=False)
    logger.info(f"Backtest results saved")

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Model: LightGBM Regressor")
    print(f"Train set: {len(train_df)} matches (seasons {settings.TRAIN_SEASONS})")
    print(f"Validation set: {len(val_df)} matches (season {settings.VALIDATION_SEASON})")
    print(f"Features: {len(settings.FEATURE_COLUMNS)}")
    print(f"\nValidation Metrics:")
    print(f"  MAPE: {metrics['mape']:.4f} ({metrics['mape']*100:.2f}%)")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"\nStatus: {'✓ PASS' if metrics['mape'] < settings.MAPE_THRESHOLD else '✗ FAIL'} (threshold: {settings.MAPE_THRESHOLD*100:.1f}%)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
