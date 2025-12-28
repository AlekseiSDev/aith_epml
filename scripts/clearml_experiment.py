"""Run a ClearML-tracked experiment with metrics, params, and model registration."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import joblib
from clearml import OutputModel, Task
from ruamel.yaml import YAML
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from wine_quality_epml.experiments.tracking import Timer, timer, write_csv, write_json

PARAMS_PATH: Final[Path] = Path("params.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and log experiment to ClearML.")
    parser.add_argument("--params", default=PARAMS_PATH, help="Path to params.yaml.")
    parser.add_argument("--project", default="wine_quality_epml", help="ClearML project name.")
    parser.add_argument("--task-name", default="train_eval", help="ClearML task name.")
    parser.add_argument("--model-name", default="wine_quality_model", help="Model registry name.")
    parser.add_argument(
        "--publish-model",
        action="store_true",
        help="Publish model to registry after upload.",
    )
    return parser.parse_args()


def load_params(path: Path) -> dict[str, Any]:
    yaml = YAML(typ="safe")
    payload = yaml.load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("params.yaml must contain a mapping at the root.")
    return payload


def load_csv_dataset(
    path: Path, *, target_col: str, id_col: str
) -> tuple[list[list[float]], list[float], list[str]]:
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
    return features, target, feature_cols


def _tree_model_types() -> set[str]:
    return {"rf", "extra_trees", "gbr", "hgb"}


def build_estimator(train_cfg: Mapping[str, Any]) -> tuple[str, Any]:
    model_type = str(train_cfg["model_type"])

    if model_type == "linear":
        cfg = train_cfg.get("linear", {})
        return model_type, LinearRegression(**cfg)
    if model_type == "ridge":
        cfg = train_cfg.get("ridge", {})
        return model_type, Ridge(**cfg)
    if model_type == "lasso":
        cfg = train_cfg.get("lasso", {})
        return model_type, Lasso(**cfg)
    if model_type == "elasticnet":
        cfg = train_cfg.get("elasticnet", {})
        return model_type, ElasticNet(**cfg)
    if model_type == "svr":
        cfg = train_cfg.get("svr", {})
        normalized = {
            "C": cfg.get("c", 1.0),
            "epsilon": cfg.get("epsilon", 0.1),
            "kernel": cfg.get("kernel", "rbf"),
            "gamma": cfg.get("gamma", "scale"),
        }
        return model_type, SVR(**normalized)
    if model_type == "knn":
        cfg = train_cfg.get("knn", {})
        return model_type, KNeighborsRegressor(**cfg)
    if model_type == "rf":
        cfg = train_cfg.get("rf", {})
        return model_type, RandomForestRegressor(**cfg)
    if model_type == "extra_trees":
        cfg = train_cfg.get("extra_trees", {})
        return model_type, ExtraTreesRegressor(**cfg)
    if model_type == "gbr":
        cfg = train_cfg.get("gbr", {})
        return model_type, GradientBoostingRegressor(**cfg)
    if model_type == "hgb":
        cfg = train_cfg.get("hgb", {})
        return model_type, HistGradientBoostingRegressor(**cfg)

    raise ValueError(f"Unsupported train.model_type={model_type!r}")


def compute_metrics(
    *, y_true: list[float], y_pred: list[float], split: str, fit_timer: Timer | None = None
) -> dict[str, Any]:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "split": split,
        "n": len(y_true),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(mse**0.5),
        "r2": float(r2_score(y_true, y_pred)),
        **({"fit_seconds": float(fit_timer.seconds)} if fit_timer else {}),
    }


def log_metrics_to_clearml(metrics: dict[str, Any], logger: Any) -> None:
    split = metrics.get("split", "unknown")
    for key in ("mae", "mse", "rmse", "r2", "fit_seconds"):
        if key in metrics:
            logger.report_scalar(title=key, series=split, value=float(metrics[key]), iteration=0)


def main() -> None:
    args = parse_args()
    params = load_params(Path(args.params))

    task = Task.init(
        project_name=args.project,
        task_name=args.task_name,
        auto_connect_frameworks={"scikit": True},
    )

    task.connect(params)
    logger = task.get_logger()

    train_cfg = params["train"]
    paths = train_cfg["paths"]
    target_col = str(train_cfg.get("target_col", "quality"))
    id_col = str(train_cfg.get("id_col", "Id"))

    train_path = Path(paths["train"])
    eval_path = Path(paths["eval"])
    test_path = Path(paths["test"])

    x_train, y_train, feature_names = load_csv_dataset(
        train_path, target_col=target_col, id_col=id_col
    )
    x_eval, y_eval, _ = load_csv_dataset(eval_path, target_col=target_col, id_col=id_col)
    x_test, y_test, _ = load_csv_dataset(test_path, target_col=target_col, id_col=id_col)

    model_type, estimator = build_estimator(train_cfg)

    standardize = (
        bool(train_cfg.get("standardize", False)) and model_type not in _tree_model_types()
    )
    if standardize:
        model: Any = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
    else:
        model = estimator

    with timer() as fit_timer:
        model.fit(x_train, y_train)

    train_pred = list(model.predict(x_train))
    eval_pred = list(model.predict(x_eval))
    test_pred = list(model.predict(x_test))

    train_metrics = compute_metrics(
        y_true=y_train, y_pred=train_pred, split="train", fit_timer=fit_timer
    )
    eval_metrics = compute_metrics(y_true=y_eval, y_pred=eval_pred, split="eval")
    test_metrics = compute_metrics(y_true=y_test, y_pred=test_pred, split="test")

    model_path = Path(paths["model"])
    meta_path = Path(paths["meta"])
    metrics_path = Path(paths["train_eval_metrics"])
    test_metrics_path = Path(paths["test_metrics"])
    predictions_path = Path(paths["predictions"])
    test_predictions_path = Path(paths["test_predictions"])

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    meta = {
        "model_type": model_type,
        "standardize": standardize,
        "target_col": target_col,
        "id_col": id_col,
        "feature_names": feature_names,
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "test_path": str(test_path),
        "sklearn": {
            "estimator_class": estimator.__class__.__name__,
            "estimator_module": estimator.__class__.__module__,
        },
        "hyperparams": dict(train_cfg.get(model_type, {})),
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    write_json(
        metrics_path,
        {
            "model_type": model_type,
            "standardize": standardize,
            "train": train_metrics,
            "eval": eval_metrics,
        },
    )
    write_json(test_metrics_path, test_metrics)

    rows: list[list[Any]] = []
    for y_true, y_pred in zip(y_eval, eval_pred, strict=False):
        rows.append(["eval", y_true, y_pred])
    for y_true, y_pred in zip(y_train, train_pred, strict=False):
        rows.append(["train", y_true, y_pred])
    write_csv(predictions_path, header=["split", "y_true", "y_pred"], rows=rows)

    test_rows: list[list[Any]] = []
    for y_true, y_pred in zip(y_test, test_pred, strict=False):
        test_rows.append(["test", y_true, y_pred])
    write_csv(test_predictions_path, header=["split", "y_true", "y_pred"], rows=test_rows)

    log_metrics_to_clearml(train_metrics, logger)
    log_metrics_to_clearml(eval_metrics, logger)
    log_metrics_to_clearml(test_metrics, logger)

    task.upload_artifact("train_eval_metrics", metrics_path)
    task.upload_artifact("test_metrics", test_metrics_path)
    task.upload_artifact("model_meta", meta_path)
    task.upload_artifact("predictions", predictions_path)
    task.upload_artifact("test_predictions", test_predictions_path)

    output_model = OutputModel(task=task, name=args.model_name, framework="scikit-learn")
    output_model.update_weights(weights_filename=str(model_path))
    output_model.set_metadata("meta", json.dumps(meta))
    if args.publish_model:
        output_model.publish()

    task.close()


if __name__ == "__main__":
    main()
