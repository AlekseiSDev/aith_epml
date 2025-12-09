"""Evaluate saved baseline models on a given dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Final, List, Mapping, Sequence

from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH: Final[Path] = Path("data/raw/WineQT.csv")
TARGET_COL: Final[str] = "quality"
ID_COL: Final[str] = "Id"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved baseline model on raw data.")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Путь к JSON модели (models/<tag>_model.json).",
    )
    parser.add_argument(
        "--metrics-path",
        default=None,
        help="Куда писать метрики (по умолчанию рядом с моделью, *_eval_metrics.json).",
    )
    parser.add_argument("--data-path", default=DATA_PATH, help="Путь к датасету для оценки.")
    parser.add_argument("--split-name", default="eval", help="Имя сплита для логирования.")
    return parser.parse_args()


def load_dataset(data_path: Path) -> tuple[List[List[float]], List[float], List[str]]:
    rows = list(csv.DictReader(data_path.open()))
    if not rows:
        raise ValueError("Dataset is empty.")

    fieldnames = rows[0].keys()
    feature_cols = [c for c in fieldnames if c not in {TARGET_COL, ID_COL}]
    X: List[List[float]] = []
    y: List[float] = []
    for row in rows:
        y.append(float(row[TARGET_COL]))
        X.append([float(row[col]) for col in feature_cols])

    return X, y, feature_cols


def predict(
    model: Mapping[str, Any], X: Sequence[Sequence[float]], feature_names: Sequence[str]
) -> list[float]:
    model_type = model.get("model_type")
    if model_type == "constant":
        value = float(model["value"])
        return [value] * len(X)
    if model_type == "linear":
        # Preserve feature order to match training
        weights = [model["weights"][name] for name in feature_names]
        bias = float(model["bias"])
        return [sum(w * f for w, f in zip(weights, row, strict=False)) + bias for row in X]
    raise ValueError(f"Unsupported model_type: {model_type}")


def compute_metrics(
    y_true: Sequence[float], y_pred: Sequence[float], meta: Mapping[str, Any]
) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "model_type": meta.get("model_type"),
        "tag": meta.get("tag"),
        "n": len(y_true),
        "mae": float(mae),
        "mse": float(mse),
        "split": meta.get("split"),
        "data_path": meta.get("data_path"),
    }


def evaluate_model_payload(model: Mapping[str, Any], data_path: Path, split_name: str) -> dict:
    X, y_true, feature_names = load_dataset(data_path)
    y_pred = predict(model, X, feature_names)
    meta = {
        **model,
        "data_path": str(data_path),
        "split": split_name,
    }
    return compute_metrics(y_true, y_pred, meta=meta)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    metrics_path = (
        Path(args.metrics_path)
        if args.metrics_path
        else model_path.with_name(f"{model_path.stem}_{args.split_name}_metrics.json")
    )

    model = json.loads(model_path.read_text())
    metrics = evaluate_model_payload(model, Path(args.data_path), split_name=args.split_name)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
