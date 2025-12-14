"""Evaluate the last trained model on the test split (for DVC experiment comparison)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Final

import joblib
from ruamel.yaml import YAML
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from wine_quality_epml.experiments.tracking import write_csv, write_json

PARAMS_PATH: Final[Path] = Path("params.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained model on the test split.")
    parser.add_argument("--params", default=PARAMS_PATH, help="Path to params.yaml.")
    return parser.parse_args()


def load_params(path: Path) -> dict[str, Any]:
    yaml = YAML(typ="safe")
    payload = yaml.load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("params.yaml must contain a mapping at the root.")
    return payload


def load_csv_dataset(
    path: Path, *, target_col: str, id_col: str
) -> tuple[list[list[float]], list[float]]:
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


def main() -> None:
    args = parse_args()
    params = load_params(Path(args.params))

    train_cfg = params["train"]
    paths = train_cfg["paths"]
    target_col = str(train_cfg.get("target_col", "quality"))
    id_col = str(train_cfg.get("id_col", "Id"))

    model_path = Path(paths["model"])
    test_path = Path(paths["test"])
    metrics_path = Path(paths["test_metrics"])
    predictions_path = Path(paths["test_predictions"])

    model = joblib.load(model_path)
    x_test, y_test = load_csv_dataset(test_path, target_col=target_col, id_col=id_col)
    y_pred = list(model.predict(x_test))

    mse = float(mean_squared_error(y_test, y_pred))
    metrics = {
        "split": "test",
        "n": len(y_test),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "mse": mse,
        "rmse": float(mse**0.5),
        "r2": float(r2_score(y_test, y_pred)),
    }
    write_json(metrics_path, metrics)

    rows: list[list[Any]] = []
    for y_true, y_hat in zip(y_test, y_pred, strict=False):
        rows.append(["test", y_true, y_hat])
    write_csv(predictions_path, header=["split", "y_true", "y_pred"], rows=rows)


if __name__ == "__main__":
    main()
