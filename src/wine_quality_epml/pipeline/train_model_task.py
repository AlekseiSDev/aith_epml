"""Luigi task for training ML model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import luigi
from ruamel.yaml import YAML
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

from wine_quality_epml.experiments.tracking import Timer, timer, write_csv, write_json
from wine_quality_epml.pipeline.split_data_task import SplitDataTask


class TrainModelTask(luigi.Task):
    """Train a sklearn model based on params.yaml configuration."""

    params_path = luigi.Parameter(default="params.yaml")

    def requires(self) -> SplitDataTask:
        """Зависит от разделения данных."""
        params = self._load_params()
        split_cfg = params.get("split", {})
        return SplitDataTask(
            seed=split_cfg.get("seed", 42),
            test_size=split_cfg.get("test_size", 0.15),
            eval_size=split_cfg.get("eval_size", 0.15),
        )

    def output(self) -> dict[str, luigi.LocalTarget]:
        """Определяет выходные файлы задачи."""
        params = self._load_params()
        paths = params["train"]["paths"]
        return {
            "model": luigi.LocalTarget(paths["model"]),
            "meta": luigi.LocalTarget(paths["meta"]),
            "metrics": luigi.LocalTarget(paths["train_eval_metrics"]),
            "predictions": luigi.LocalTarget(paths["predictions"]),
        }

    def run(self) -> None:
        """Выполняет обучение модели."""
        params = self._load_params()
        train_cfg = params["train"]
        paths = train_cfg["paths"]
        target_col = str(train_cfg.get("target_col", "quality"))
        id_col = str(train_cfg.get("id_col", "Id"))

        # Загружаем данные
        train_path = Path(paths["train"])
        eval_path = Path(paths["eval"])
        x_train, y_train, feature_names = self._load_csv_dataset(
            train_path, target_col=target_col, id_col=id_col
        )
        x_eval, y_eval, _ = self._load_csv_dataset(eval_path, target_col=target_col, id_col=id_col)

        # Строим модель
        model_type, estimator = self._build_estimator(train_cfg)
        standardize = (
            bool(train_cfg.get("standardize", False)) and model_type not in self._tree_model_types()
        )

        if standardize:
            model: Any = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
        else:
            model = estimator

        # Обучаем модель
        with timer() as fit_timer:
            model.fit(x_train, y_train)

        # Делаем предсказания
        train_pred = list(model.predict(x_train))
        eval_pred = list(model.predict(x_eval))

        # Вычисляем метрики
        train_metrics = self._compute_metrics(
            y_true=y_train, y_pred=train_pred, split="train", fit_timer=fit_timer
        )
        eval_metrics = self._compute_metrics(y_true=y_eval, y_pred=eval_pred, split="eval")

        # Сохраняем модель
        model_path = Path(self.output()["model"].path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

        # Сохраняем метаданные
        meta = {
            "model_type": model_type,
            "standardize": standardize,
            "target_col": target_col,
            "id_col": id_col,
            "feature_names": feature_names,
            "train_path": str(train_path),
            "eval_path": str(eval_path),
            "sklearn": {
                "estimator_class": estimator.__class__.__name__,
                "estimator_module": estimator.__class__.__module__,
            },
            "hyperparams": dict(train_cfg.get(model_type, {})),
        }
        meta_path = Path(self.output()["meta"].path)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        # Сохраняем метрики
        write_json(
            Path(self.output()["metrics"].path),
            {
                "model_type": model_type,
                "standardize": standardize,
                "train": train_metrics,
                "eval": eval_metrics,
            },
        )

        # Сохраняем предсказания
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

    def _load_params(self) -> dict[str, Any]:
        """Loads parameters from YAML."""
        yaml = YAML(typ="safe")
        payload = yaml.load(Path(str(self.params_path)).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("params.yaml must contain a mapping at the root.")
        return payload

    def _load_csv_dataset(
        self, path: Path, *, target_col: str, id_col: str
    ) -> tuple[list[list[float]], list[float], list[str]]:
        """Загружает датасет из CSV."""
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

    def _tree_model_types(self) -> set[str]:
        """Возвращает набор древовидных моделей."""
        return {"rf", "extra_trees", "gbr", "hgb"}

    def _build_estimator(self, train_cfg: dict[str, Any]) -> tuple[str, Any]:
        """Создает estimator на основе конфигурации."""
        model_type = str(train_cfg["model_type"])

        if model_type == "linear":
            cfg = train_cfg.get("linear", {})
            return model_type, LinearRegression(**cfg)
        if model_type == "ridge":
            cfg = train_cfg.get("ridge", {})
            return model_type, Ridge(**cfg)
        if model_type == "lasso":
            cfg = train_cfg.get("lasso", {})
            return model_type, Lasso(**cfg)
        if model_type == "elasticnet":
            cfg = train_cfg.get("elasticnet", {})
            return model_type, ElasticNet(**cfg)
        if model_type == "svr":
            cfg = train_cfg.get("svr", {})
            normalized = {
                "C": cfg.get("c", 1.0),
                "epsilon": cfg.get("epsilon", 0.1),
                "kernel": cfg.get("kernel", "rbf"),
                "gamma": cfg.get("gamma", "scale"),
            }
            return model_type, SVR(**normalized)
        if model_type == "knn":
            cfg = train_cfg.get("knn", {})
            return model_type, KNeighborsRegressor(**cfg)
        if model_type == "rf":
            cfg = train_cfg.get("rf", {})
            return model_type, RandomForestRegressor(**cfg)
        if model_type == "extra_trees":
            cfg = train_cfg.get("extra_trees", {})
            return model_type, ExtraTreesRegressor(**cfg)
        if model_type == "gbr":
            cfg = train_cfg.get("gbr", {})
            return model_type, GradientBoostingRegressor(**cfg)
        if model_type == "hgb":
            cfg = train_cfg.get("hgb", {})
            return model_type, HistGradientBoostingRegressor(**cfg)

        raise ValueError(f"Unsupported train.model_type={model_type!r}")

    def _compute_metrics(
        self,
        *,
        y_true: list[float],
        y_pred: list[float],
        split: str,
        fit_timer: Timer | None = None,
    ) -> dict[str, Any]:
        """Вычисляет метрики модели."""
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
