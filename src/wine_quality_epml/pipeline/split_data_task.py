"""Luigi task for splitting raw dataset into train/eval/test CSVs."""

from __future__ import annotations

import csv
from pathlib import Path

import luigi
from sklearn.model_selection import train_test_split


class SplitDataTask(luigi.Task):
    """Split raw dataset into train/eval/test splits."""

    # Параметры задачи
    raw_path = luigi.Parameter(default="data/raw/WineQT.csv")
    split_dir = luigi.Parameter(default="data/splits")
    test_size = luigi.FloatParameter(default=0.15)
    eval_size = luigi.FloatParameter(default=0.15)
    seed = luigi.IntParameter(default=42)

    def output(self) -> dict[str, luigi.LocalTarget]:
        """Defines output files for caching."""
        split_dir = Path(str(self.split_dir))
        return {
            "train": luigi.LocalTarget(str(split_dir / "train.csv")),
            "eval": luigi.LocalTarget(str(split_dir / "eval.csv")),
            "test": luigi.LocalTarget(str(split_dir / "test.csv")),
        }

    def run(self) -> None:
        """Executes data splitting."""
        raw_path = Path(str(self.raw_path))

        # Читаем исходные данные
        with raw_path.open() as f:
            reader = csv.DictReader(f)
            raw_rows = list(reader)

        if not raw_rows:
            raise ValueError("Raw dataset is empty.")

        fieldnames = list(raw_rows[0].keys())

        # Split data
        train_rows, temp_rows = train_test_split(
            raw_rows,
            test_size=self.test_size + self.eval_size,
            random_state=self.seed,
            shuffle=True,
        )
        relative_eval_size = self.eval_size / (self.test_size + self.eval_size)
        eval_rows, test_rows = train_test_split(
            temp_rows, test_size=relative_eval_size, random_state=self.seed, shuffle=True
        )

        # Записываем результаты
        self._write_csv(self.output()["train"].path, fieldnames, train_rows)
        self._write_csv(self.output()["eval"].path, fieldnames, eval_rows)
        self._write_csv(self.output()["test"].path, fieldnames, test_rows)

    def _write_csv(self, path: str, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
        """Записывает CSV файл."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
