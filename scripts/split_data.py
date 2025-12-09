"""Split raw dataset into train/eval/test CSVs for versioning via DVC."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Final

from sklearn.model_selection import train_test_split

RAW_PATH: Final[Path] = Path("data/raw/WineQT.csv")
SPLIT_DIR: Final[Path] = Path("data/splits")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train/eval/test splits from raw data.")
    parser.add_argument("--raw-path", default=RAW_PATH, help="Путь к исходному CSV.")
    parser.add_argument(
        "--split-dir",
        default=SPLIT_DIR,
        help="Куда класть сплиты (train.csv, eval.csv, test.csv).",
    )
    parser.add_argument("--test-size", type=float, default=0.15, help="Доля теста.")
    parser.add_argument("--eval-size", type=float, default=0.15, help="Доля валидации.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    raw_rows = read_rows(Path(args.raw_path))
    if not raw_rows:
        raise ValueError("Raw dataset is empty.")

    fieldnames = list(raw_rows[0].keys())

    train_rows, temp_rows = train_test_split(
        raw_rows, test_size=args.test_size + args.eval_size, random_state=args.seed, shuffle=True
    )
    relative_eval_size = args.eval_size / (args.test_size + args.eval_size)
    eval_rows, test_rows = train_test_split(
        temp_rows, test_size=relative_eval_size, random_state=args.seed, shuffle=True
    )

    split_dir = Path(args.split_dir)
    write_rows(split_dir / "train.csv", fieldnames, train_rows)
    write_rows(split_dir / "eval.csv", fieldnames, eval_rows)
    write_rows(split_dir / "test.csv", fieldnames, test_rows)


if __name__ == "__main__":
    main()
