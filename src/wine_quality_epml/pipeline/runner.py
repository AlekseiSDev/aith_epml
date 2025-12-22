"""Main entry point for running Luigi pipeline."""

from __future__ import annotations

import argparse
import sys

import luigi

from wine_quality_epml.pipeline.test_model_task import TestModelTask


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Luigi ML pipeline")
    parser.add_argument(
        "--params",
        default="params.yaml",
        help="Path to params.yaml configuration file",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for Luigi",
    )
    parser.add_argument(
        "--local-scheduler",
        action="store_true",
        help="Use local scheduler instead of Luigi daemon",
    )
    return parser.parse_args()


def main() -> int:
    """Run the complete ML pipeline."""
    args = parse_args()

    # Запускаем полный пайплайн через TestModelTask (который зависит от всех остальных)
    success = luigi.build(
        [TestModelTask(params_path=args.params)],
        workers=args.workers,
        local_scheduler=args.local_scheduler,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
