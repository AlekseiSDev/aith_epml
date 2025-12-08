"""Train a simple constant model and log metrics for DVC tracking."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Final

DATA_PATH: Final[Path] = Path("data/raw/WineQT.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline constant model (mean/median).")
    parser.add_argument(
        "--strategy",
        choices=["mean", "median"],
        default="mean",
        help="Как считать константу (mean или median).",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Тэг для версии модели/метрик. Если не указан, используется strategy.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Кастомный путь для модели. По умолчанию models/<tag>_model.json.",
    )
    parser.add_argument(
        "--metrics-path",
        default=None,
        help="Кастомный путь для метрик. По умолчанию models/<tag>_metrics.json.",
    )
    return parser.parse_args()


def load_target() -> list[float]:
    rows = list(csv.DictReader(DATA_PATH.open()))
    qualities = [float(row["quality"]) for row in rows if "quality" in row]
    if not qualities:
        raise ValueError("Dataset is empty or missing 'quality' column.")
    return qualities


def constant_value(values: list[float], strategy: str) -> float:
    if strategy == "mean":
        return sum(values) / len(values)
    mid = len(values) // 2
    sorted_vals = sorted(values)
    if len(values) % 2 == 1:
        return sorted_vals[mid]
    return 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid])


def compute_metrics(values: list[float], pred: float) -> dict:
    n = len(values)
    mse = sum((v - pred) ** 2 for v in values) / n
    mae = sum(abs(v - pred) for v in values) / n
    return {"n": n, "prediction": pred, "mse": mse, "mae": mae}


def main() -> None:
    args = parse_args()
    tag = args.tag or args.strategy
    model_path = Path(args.model_path) if args.model_path else Path(f"models/{tag}_model.json")
    metrics_path = (
        Path(args.metrics_path) if args.metrics_path else Path(f"models/{tag}_metrics.json")
    )

    values = load_target()
    pred = constant_value(values, args.strategy)
    metrics = compute_metrics(values, pred) | {"strategy": args.strategy, "tag": tag}

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    model_payload = {
        "model_type": "constant",
        "strategy": args.strategy,
        "value": pred,
        "n_samples": metrics["n"],
        "tag": tag,
    }
    model_path.write_text(json.dumps(model_payload, indent=2) + "\n")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
