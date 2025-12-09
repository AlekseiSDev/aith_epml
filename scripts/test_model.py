"""Evaluate a saved model on the test split using shared evaluation logic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

from evaluate_model import evaluate_model_payload

DEFAULT_TEST_PATH: Final[Path] = Path("data/splits/test.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved model on the test split (for reporting)."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Путь к JSON модели (например, models/linear_v1_model.json).",
    )
    parser.add_argument(
        "--data-path",
        default=DEFAULT_TEST_PATH,
        help="Путь к тестовому датасету (по умолчанию data/splits/test.csv).",
    )
    parser.add_argument(
        "--split-name",
        default="test",
        help="Имя сплита для логирования в метриках.",
    )
    parser.add_argument(
        "--metrics-path",
        default=None,
        help="Куда писать метрики (по умолчанию рядом с моделью, *_test_metrics.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    metrics_path = (
        Path(args.metrics_path)
        if args.metrics_path
        else model_path.with_name(f"{model_path.stem}_{args.split_name}_metrics.json")
    )

    model_payload = json.loads(model_path.read_text())
    metrics = evaluate_model_payload(
        model=model_payload, data_path=Path(args.data_path), split_name=args.split_name
    )

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
