"""Luigi task for testing trained ML model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import luigi
from ruamel.yaml import YAML
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from wine_quality_epml.experiments.tracking import write_csv, write_json
from wine_quality_epml.pipeline.train_model_task import TrainModelTask


class TestModelTask(luigi.Task):
    """Evaluate trained model on test split."""

    params_path = luigi.Parameter(default="params.yaml")

    def requires(self) -> TrainModelTask:
        """Зависит от обучения модели."""
        return TrainModelTask(params_path=self.params_path)

    def output(self) -> dict[str, luigi.LocalTarget]:
        """Определяет выходные файлы задачи."""
        params = self._load_params()
        paths = params["train"]["paths"]
        return {
            "metrics": luigi.LocalTarget(paths["test_metrics"]),
            "predictions": luigi.LocalTarget(paths["test_predictions"]),
        }

    def run(self) -> None:
        """Выполняет тестирование модели."""
        params = self._load_params()
        train_cfg = params["train"]
        paths = train_cfg["paths"]
        target_col = str(train_cfg.get("target_col", "quality"))
        id_col = str(train_cfg.get("id_col", "Id"))

        # Загружаем модель
        model_path = Path(paths["model"])
        model = joblib.load(model_path)

        # Загружаем тестовые данные
        test_path = Path(paths["test"])
        x_test, y_test = self._load_csv_dataset(test_path, target_col=target_col, id_col=id_col)

        # Делаем предсказания
        y_pred = list(model.predict(x_test))

        # Вычисляем метрики
        mse = float(mean_squared_error(y_test, y_pred))
        metrics = {
            "split": "test",
            "n": len(y_test),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "mse": mse,
            "rmse": float(mse**0.5),
            "r2": float(r2_score(y_test, y_pred)),
        }

        # Сохраняем метрики
        write_json(Path(self.output()["metrics"].path), metrics)

        # Сохраняем предсказания
        rows: list[list[Any]] = []
        for y_true, y_hat in zip(y_test, y_pred, strict=False):
            rows.append(["test", y_true, y_hat])
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
    ) -> tuple[list[list[float]], list[float]]:
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
        return features, target
