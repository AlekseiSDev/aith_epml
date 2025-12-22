"""Luigi task for splitting raw dataset into train/eval/test CSVs."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import luigi
from sklearn.model_selection import train_test_split

from wine_quality_epml.config.loader import load_config
from wine_quality_epml.config.schemas import ProjectConfig

logger = logging.getLogger(__name__)


class SplitDataTask(luigi.Task):
    """Split raw dataset into train/eval/test splits with Pydantic validation."""

    raw_path = luigi.Parameter(default="data/raw/WineQT.csv")
    params_path = luigi.Parameter(default="params.yaml")

    def output(self) -> dict[str, luigi.LocalTarget]:
        """Defines output files for caching."""
        split_dir = Path("data/splits")
        return {
            "train": luigi.LocalTarget(str(split_dir / "train.csv")),
            "eval": luigi.LocalTarget(str(split_dir / "eval.csv")),
            "test": luigi.LocalTarget(str(split_dir / "test.csv")),
        }

    def run(self) -> None:
        """Executes data splitting using validated Pydantic config."""
        config = self._load_config()
        raw_path = Path(str(self.raw_path))

        logger.info(f"Loading raw data from {raw_path}")

        # Read source data
        with raw_path.open() as f:
            reader = csv.DictReader(f)
            raw_rows = list(reader)

        if not raw_rows:
            raise ValueError("Raw dataset is empty.")

        fieldnames = list(raw_rows[0].keys())
        logger.info(f"Dataset: {len(raw_rows)} rows, {len(fieldnames)} columns")

        # Split data using Pydantic validated config
        test_size = config.split.test_size
        eval_size = config.split.eval_size
        seed = config.split.seed

        logger.info(f"Splitting: test={test_size}, eval={eval_size}, seed={seed}")

        train_rows, temp_rows = train_test_split(
            raw_rows,
            test_size=test_size + eval_size,
            random_state=seed,
            shuffle=True,
        )
        relative_eval_size = eval_size / (test_size + eval_size)
        eval_rows, test_rows = train_test_split(
            temp_rows, test_size=relative_eval_size, random_state=seed, shuffle=True
        )

        logger.info(f"Sizes: train={len(train_rows)}, eval={len(eval_rows)}, test={len(test_rows)}")

        # Write results
        self._write_csv(self.output()["train"].path, fieldnames, train_rows)
        self._write_csv(self.output()["eval"].path, fieldnames, eval_rows)
        self._write_csv(self.output()["test"].path, fieldnames, test_rows)

        logger.info("✓ Data splitting completed")

    def _load_config(self) -> ProjectConfig:
        """Load and validate configuration using Pydantic."""
        return load_config(str(self.params_path))

    def _write_csv(self, path: str, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
        """Write CSV file."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
