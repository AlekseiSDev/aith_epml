"""Train baseline models (constant or linear) and log JSON artifacts for DVC."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Final, List, Sequence

from evaluate_model import compute_metrics, evaluate_model_payload, predict
from sklearn.linear_model import LinearRegression

DATA_PATH: Final[Path] = Path("data/splits/train.csv")
EVAL_DATA_PATH: Final[Path] = Path("data/splits/eval.csv")
TARGET_COL: Final[str] = "quality"
ID_COL: Final[str] = "Id"


@dataclass
class Dataset:
    features: List[List[float]]
    target: List[float]
    feature_names: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline models for wine quality.")
    parser.add_argument(
        "--model-type",
        choices=["constant", "linear"],
        default="constant",
        help="Тип модели: константа или линейная регрессия.",
    )
    parser.add_argument(
        "--strategy",
        choices=["mean", "median"],
        default="mean",
        help="Как считать константу (для model-type=constant).",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Тэг для версии модели/метрик. Если не указан, используется model-type.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Кастомный путь для модели. По умолчанию models/<tag>_model.json.",
    )
    parser.add_argument(
        "--metrics-path",
        default=None,
        help=(
            "Кастомный путь для метрик (train+eval). "
            "По умолчанию models/<tag>_train_eval_metrics.json."
        ),
    )
    parser.add_argument(
        "--data-path",
        default=DATA_PATH,
        help="Путь к датасету для обучения (по умолчанию data/splits/train.csv).",
    )
    parser.add_argument(
        "--eval-data-path",
        default=EVAL_DATA_PATH,
        help="Путь к датасету для eval (по умолчанию data/splits/eval.csv).",
    )
    parser.add_argument(
        "--split-name",
        default="train",
        help="Имя сплита для логирования в метриках.",
    )
    parser.add_argument(
        "--eval-split-name",
        default="eval",
        help="Имя eval-сплита для логирования в метриках.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> tuple[List[List[float]], List[float], List[str]]:
    import csv

    rows = list(csv.DictReader(path.open()))
    if not rows:
        raise ValueError("Dataset is empty.")
    fieldnames = rows[0].keys()
    feature_cols = [c for c in fieldnames if c not in {TARGET_COL, ID_COL}]
    features: List[List[float]] = []
    target: List[float] = []
    for row in rows:
        target.append(float(row[TARGET_COL]))
        features.append([float(row[col]) for col in feature_cols])
    return features, target, feature_cols


def constant_value(values: Sequence[float], strategy: str) -> float:
    return (
        float(statistics.mean(values)) if strategy == "mean" else float(statistics.median(values))
    )


def fit_linear(dataset: Dataset) -> tuple[dict, float]:
    model = LinearRegression()
    model.fit(dataset.features, dataset.target)
    weights = {name: float(w) for name, w in zip(dataset.feature_names, model.coef_, strict=False)}
    bias = float(model.intercept_)
    return weights, bias


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    features, target, feature_names = load_dataset(data_path)

    tag = args.tag or args.model_type
    model_path = Path(args.model_path) if args.model_path else Path(f"models/{tag}_model.json")
    metrics_path = (
        Path(args.metrics_path)
        if args.metrics_path
        else Path(f"models/{tag}_train_eval_metrics.json")
    )

    if args.model_type == "constant":
        value = constant_value(target, args.strategy)
        model_payload = {
            "model_type": "constant",
            "strategy": args.strategy,
            "value": value,
            "tag": tag,
            "target": TARGET_COL,
            "split": args.split_name,
            "data_path": str(data_path),
        }
    else:
        dataset = Dataset(features=features, target=target, feature_names=feature_names)
        weights, bias = fit_linear(dataset)
        model_payload = {
            "model_type": "linear",
            "target": TARGET_COL,
            "feature_names": feature_names,
            "weights": weights,
            "bias": bias,
            "tag": tag,
            "solver": "sklearn.linear_model.LinearRegression",
            "split": args.split_name,
            "data_path": str(data_path),
        }

    train_pred = predict(model_payload, features, feature_names)
    train_meta = {
        "model_type": model_payload["model_type"],
        "tag": tag,
        "split": args.split_name,
        "data_path": str(data_path),
    }
    train_metrics = compute_metrics(target, train_pred, meta=train_meta)
    if args.model_type == "constant":
        train_metrics["strategy"] = args.strategy
    else:
        train_metrics["solver"] = "sklearn.linear_model.LinearRegression"

    eval_metrics = evaluate_model_payload(
        model=model_payload, data_path=Path(args.eval_data_path), split_name=args.eval_split_name
    )

    combined_metrics = {
        "tag": tag,
        "model_type": model_payload["model_type"],
        "train": train_metrics,
        "eval": eval_metrics,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(model_payload, indent=2) + "\n")
    metrics_path.write_text(json.dumps(combined_metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
