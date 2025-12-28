"""ClearML pipeline for the wine quality workflow (split -> train -> test)."""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Final

from clearml.automation.controller import PipelineDecorator

PARAMS_PATH: Final[Path] = Path("params.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ClearML pipeline locally.")
    parser.add_argument("--params", default=PARAMS_PATH, help="Path to params.yaml.")
    parser.add_argument("--project", default="wine_quality_epml", help="ClearML project name.")
    parser.add_argument("--name", default="wine_quality_pipeline", help="Pipeline name.")
    return parser.parse_args()


@PipelineDecorator.component(return_values=["status"])
def split_data(params_path: str) -> str:
    from pathlib import Path as _Path

    from ruamel.yaml import YAML

    yaml = YAML(typ="safe")
    params = yaml.load(_Path(params_path).read_text(encoding="utf-8")) or {}
    split_cfg = params.get("split", {}) if isinstance(params, dict) else {}
    train_cfg = params.get("train", {}) if isinstance(params, dict) else {}
    paths = train_cfg.get("paths", {}) if isinstance(train_cfg, dict) else {}
    raw_path = paths.get("raw", "data/raw/WineQT.csv")
    split_dir = _Path(paths.get("train", "data/splits/train.csv")).parent
    args = [
        sys.executable,
        "scripts/split_data.py",
        "--raw-path",
        str(raw_path),
        "--split-dir",
        str(split_dir),
        "--test-size",
        str(split_cfg.get("test_size", 0.15)),
        "--eval-size",
        str(split_cfg.get("eval_size", 0.15)),
        "--seed",
        str(split_cfg.get("seed", 42)),
    ]
    subprocess.run(args, check=True)  # noqa: S603
    return "ok"


@PipelineDecorator.component(return_values=["model_path", "metrics_path"])
def train_model(params_path: str) -> tuple[str, str]:
    args = [sys.executable, "scripts/train_model.py", "--params", params_path]
    subprocess.run(args, check=True)  # noqa: S603
    from pathlib import Path as _Path

    from ruamel.yaml import YAML

    yaml = YAML(typ="safe")
    params = yaml.load(_Path(params_path).read_text(encoding="utf-8")) or {}
    paths = params["train"]["paths"]
    return str(paths["model"]), str(paths["train_eval_metrics"])


@PipelineDecorator.component(return_values=["test_metrics_path"])
def test_model(params_path: str) -> str:
    args = [sys.executable, "scripts/test_trained_model.py", "--params", params_path]
    subprocess.run(args, check=True)  # noqa: S603
    from pathlib import Path as _Path

    from ruamel.yaml import YAML

    yaml = YAML(typ="safe")
    params = yaml.load(_Path(params_path).read_text(encoding="utf-8")) or {}
    return str(params["train"]["paths"]["test_metrics"])


def build_pipeline(project: str, name: str, params_path: str) -> None:
    @PipelineDecorator.pipeline(name=name, project=project, version="1.0")
    def _pipeline() -> None:
        split_data(params_path)
        train_model(params_path)
        test_model(params_path)

    _pipeline()


def main() -> None:
    args = parse_args()
    PipelineDecorator.run_locally()
    build_pipeline(args.project, args.name, str(args.params))


if __name__ == "__main__":
    main()
