"""Luigi task for testing trained ML model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import luigi
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from wine_quality_epml.config.loader import load_config
from wine_quality_epml.config.schemas import ProjectConfig
from wine_quality_epml.experiments.tracking import write_csv, write_json
from wine_quality_epml.pipeline.train_model_task import TrainModelTask

logger = logging.getLogger(__name__)


class TestModelTask(luigi.Task):
    """Evaluate trained model on test split with Pydantic validation."""

    params_path = luigi.Parameter(default="params.yaml")

    def requires(self) -> TrainModelTask:
        """Depends on model training."""
        return TrainModelTask(params_path=self.params_path)

    def output(self) -> dict[str, luigi.LocalTarget]:
        """Defines output files."""
        config = self._load_config()
        paths = config.train.paths
        return {
            "metrics": luigi.LocalTarget(paths.test_metrics),
            "predictions": luigi.LocalTarget(paths.test_predictions),
        }

    def run(self) -> None:
        """Executes model testing with validated config."""
        config = self._load_config()
        train_cfg = config.train
        paths = train_cfg.paths

        logger.info("Testing model on test split")

        # Load model
        model_path = Path(paths.model)
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)

        # Load test data
        test_path = Path(paths.test)
        x_test, y_test = self._load_csv_dataset(
            test_path, target_col=train_cfg.target_col, id_col=train_cfg.id_col
        )

        logger.info(f"Test size: {len(y_test)}")

        # Make predictions
        y_pred = list(model.predict(x_test))

        # Compute metrics
        mse = float(mean_squared_error(y_test, y_pred))
        metrics = {
            "split": "test",
            "n": len(y_test),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "mse": mse,
            "rmse": float(mse**0.5),
            "r2": float(r2_score(y_test, y_pred)),
        }

        logger.info(f"Test RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")

        # Save metrics
        write_json(Path(self.output()["metrics"].path), metrics)

        # Save predictions
        rows: list[list[Any]] = []
        for y_true, y_hat in zip(y_test, y_pred, strict=False):
            rows.append(["test", y_true, y_hat])
        write_csv(
            Path(self.output()["predictions"].path),
            header=["split", "y_true", "y_pred"],
            rows=rows,
        )

        logger.info("✓ Model testing completed successfully")
        logger.info(
            f"Pipeline summary: Model={config.train.model_type}, "
            f"Test RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}"
        )

    def _load_config(self) -> ProjectConfig:
        """Load and validate configuration using Pydantic."""
        return load_config(str(self.params_path))

    def _load_csv_dataset(
        self, path: Path, *, target_col: str, id_col: str
    ) -> tuple[list[list[float]], list[float]]:
        """Load dataset from CSV."""
        import csv

        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        if not rows:
            raise ValueError(f"Dataset is empty: {path}")

        fieldnames = list(rows[0].keys())
        feature_cols = [c for c in fieldnames if c not in {target_col, id_col}]

        features: list[list[float]] = []
        target: list[float] = []
        for row in rows:
            target.append(float(row[target_col]))
            features.append([float(row[col]) for col in feature_cols])
        return features, target
