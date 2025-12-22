"""Luigi task for training ML model with Pydantic configuration."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import luigi
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from wine_quality_epml.config.loader import load_config
from wine_quality_epml.config.schemas import ProjectConfig
from wine_quality_epml.experiments.tracking import Timer, timer, write_csv, write_json
from wine_quality_epml.pipeline.split_data_task import SplitDataTask

logger = logging.getLogger(__name__)


class TrainModelTask(luigi.Task):
    """Train sklearn model using validated Pydantic configuration."""

    params_path = luigi.Parameter(default="params.yaml")

    def requires(self) -> SplitDataTask:
        """Depends on data splitting."""
        return SplitDataTask(params_path=self.params_path)

    def output(self) -> dict[str, luigi.LocalTarget]:
        """Defines output files."""
        config = self._load_config()
        paths = config.train.paths
        return {
            "model": luigi.LocalTarget(paths.model),
            "meta": luigi.LocalTarget(paths.meta),
            "metrics": luigi.LocalTarget(paths.train_eval_metrics),
            "predictions": luigi.LocalTarget(paths.predictions),
        }

    def run(self) -> None:
        """Executes model training with validated config."""
        config = self._load_config()
        train_cfg = config.train
        paths = train_cfg.paths

        logger.info(f"Training {train_cfg.model_type} model")
        logger.info(f"Seed: {train_cfg.seed}, Standardize: {train_cfg.standardize}")

        # Load data
        train_path = Path(paths.train)
        eval_path = Path(paths.eval)
        x_train, y_train, feature_names = self._load_csv_dataset(
            train_path, target_col=train_cfg.target_col, id_col=train_cfg.id_col
        )
        x_eval, y_eval, _ = self._load_csv_dataset(
            eval_path, target_col=train_cfg.target_col, id_col=train_cfg.id_col
        )

        logger.info(f"Train size: {len(y_train)}, Eval size: {len(y_eval)}")

        # Build model using Pydantic validated config
        model_type = train_cfg.model_type
        estimator = self._build_estimator(config)
        standardize = train_cfg.standardize and model_type not in {
            "rf",
            "extra_trees",
            "gbr",
            "hgb",
        }

        if standardize:
            logger.info("Building pipeline with StandardScaler")
            model: Any = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
        else:
            model = estimator

        # Train model
        logger.info("Starting model training...")
        with timer() as fit_timer:
            model.fit(x_train, y_train)
        logger.info(f"Training completed in {fit_timer.seconds:.2f}s")

        # Make predictions
        train_pred = list(model.predict(x_train))
        eval_pred = list(model.predict(x_eval))

        # Compute metrics
        train_metrics = self._compute_metrics(
            y_true=y_train, y_pred=train_pred, split="train", fit_timer=fit_timer
        )
        eval_metrics = self._compute_metrics(y_true=y_eval, y_pred=eval_pred, split="eval")

        logger.info(f"Train RMSE: {train_metrics['rmse']:.4f}, R²: {train_metrics['r2']:.4f}")
        logger.info(f"Eval RMSE: {eval_metrics['rmse']:.4f}, R²: {eval_metrics['r2']:.4f}")

        # Save model
        model_path = Path(self.output()["model"].path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")

        # Save metadata
        meta = {
            "model_type": model_type,
            "standardize": standardize,
            "target_col": train_cfg.target_col,
            "id_col": train_cfg.id_col,
            "feature_names": feature_names,
            "train_path": str(train_path),
            "eval_path": str(eval_path),
            "sklearn": {
                "estimator_class": estimator.__class__.__name__,
                "estimator_module": estimator.__class__.__module__,
            },
            "hyperparams": train_cfg.get_model_config().model_dump(),
        }
        meta_path = Path(self.output()["meta"].path)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        # Save metrics
        write_json(
            Path(self.output()["metrics"].path),
            {
                "model_type": model_type,
                "standardize": standardize,
                "train": train_metrics,
                "eval": eval_metrics,
            },
        )

        # Save predictions
        rows: list[list[Any]] = []
        for y_true, y_pred in zip(y_eval, eval_pred, strict=False):
            rows.append(["eval", y_true, y_pred])
        for y_true, y_pred in zip(y_train, train_pred, strict=False):
            rows.append(["train", y_true, y_pred])
        write_csv(
            Path(self.output()["predictions"].path),
            header=["split", "y_true", "y_pred"],
            rows=rows,
        )

        logger.info("✓ Model training completed successfully")

    def _load_config(self) -> ProjectConfig:
        """Load and validate configuration using Pydantic."""
        return load_config(str(self.params_path))

    def _load_csv_dataset(
        self, path: Path, *, target_col: str, id_col: str
    ) -> tuple[list[list[float]], list[float], list[str]]:
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
        return features, target, feature_cols

    def _build_estimator(self, config: ProjectConfig) -> Any:
        """Build estimator from validated Pydantic config."""
        model_type = config.train.model_type
        model_cfg = config.train.get_model_config()

        if model_type == "linear":
            return LinearRegression(**model_cfg.model_dump())
        if model_type == "ridge":
            return Ridge(**model_cfg.model_dump())
        if model_type == "lasso":
            return Lasso(**model_cfg.model_dump())
        if model_type == "elasticnet":
            return ElasticNet(**model_cfg.model_dump())
        if model_type == "svr":
            # SVR uses 'c' (lowercase) in our config but sklearn expects 'C' (uppercase)
            cfg_dict = model_cfg.model_dump()
            cfg_dict["C"] = cfg_dict.pop("c")
            return SVR(**cfg_dict)
        if model_type == "knn":
            return KNeighborsRegressor(**model_cfg.model_dump())
        if model_type == "rf":
            return RandomForestRegressor(**model_cfg.model_dump())
        if model_type == "extra_trees":
            return ExtraTreesRegressor(**model_cfg.model_dump())
        if model_type == "gbr":
            return GradientBoostingRegressor(**model_cfg.model_dump())
        if model_type == "hgb":
            return HistGradientBoostingRegressor(**model_cfg.model_dump())

        raise ValueError(f"Unsupported model_type={model_type!r}")

    def _compute_metrics(
        self,
        *,
        y_true: list[float],
        y_pred: list[float],
        split: str,
        fit_timer: Timer | None = None,
    ) -> dict[str, Any]:
        """Compute model metrics."""
        mse = float(mean_squared_error(y_true, y_pred))
        return {
            "split": split,
            "n": len(y_true),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "mse": mse,
            "rmse": float(mse**0.5),
            "r2": float(r2_score(y_true, y_pred)),
            **({"fit_seconds": float(fit_timer.seconds)} if fit_timer else {}),
        }
