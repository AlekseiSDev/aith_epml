"""Pydantic schemas for configuration validation."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# ===== Split Configuration =====
class SplitConfig(BaseModel):
    """Configuration for data splitting."""

    seed: int = Field(default=42, ge=0, description="Random seed for reproducibility")
    test_size: float = Field(default=0.15, gt=0.0, lt=1.0, description="Test set size")
    eval_size: float = Field(default=0.15, gt=0.0, lt=1.0, description="Eval set size")

    @model_validator(mode="after")
    def validate_total_size(self) -> SplitConfig:
        """Validate that test_size + eval_size < 1.0."""
        if self.test_size + self.eval_size >= 1.0:
            raise ValueError(
                f"test_size + eval_size must be < 1.0, got {self.test_size + self.eval_size}"
            )
        return self


# ===== Paths Configuration =====
class PathsConfig(BaseModel):
    """Configuration for file paths."""

    train: str = Field(default="data/splits/train.csv")
    eval: str = Field(default="data/splits/eval.csv")
    test: str = Field(default="data/splits/test.csv")
    model: str = Field(default="models/exp_model.pkl")
    meta: str = Field(default="models/exp_model_meta.json")
    train_eval_metrics: str = Field(default="reports/train_eval_metrics.json")
    test_metrics: str = Field(default="reports/test_metrics.json")
    predictions: str = Field(default="reports/predictions.csv")
    test_predictions: str = Field(default="reports/test_predictions.csv")


# ===== Model Configurations =====
class LinearConfig(BaseModel):
    """Configuration for LinearRegression."""

    fit_intercept: bool = Field(default=True)


class RidgeConfig(BaseModel):
    """Configuration for Ridge regression."""

    alpha: float = Field(default=1.0, gt=0.0, description="Regularization strength")


class LassoConfig(BaseModel):
    """Configuration for Lasso regression."""

    alpha: float = Field(default=0.01, gt=0.0, description="Regularization strength")


class ElasticNetConfig(BaseModel):
    """Configuration for ElasticNet regression."""

    alpha: float = Field(default=0.01, gt=0.0, description="Regularization strength")
    l1_ratio: float = Field(default=0.5, ge=0.0, le=1.0, description="ElasticNet mixing parameter")


class SVRConfig(BaseModel):
    """Configuration for Support Vector Regression."""

    kernel: Literal["linear", "poly", "rbf", "sigmoid"] = Field(default="rbf")
    c: float = Field(default=1.0, gt=0.0, description="Regularization parameter", alias="C")
    epsilon: float = Field(default=0.1, ge=0.0, description="Epsilon parameter")
    gamma: Union[Literal["scale", "auto"], float] = Field(default="scale")


class KNNConfig(BaseModel):
    """Configuration for K-Nearest Neighbors."""

    n_neighbors: int = Field(default=5, ge=1, description="Number of neighbors")
    weights: Literal["uniform", "distance"] = Field(default="uniform")


class RandomForestConfig(BaseModel):
    """Configuration for Random Forest."""

    n_estimators: int = Field(default=100, ge=1, description="Number of trees")
    max_depth: int | None = Field(default=None, ge=1, description="Maximum tree depth")
    min_samples_split: int = Field(default=2, ge=2, description="Min samples to split")
    random_state: int | None = Field(default=42, ge=0)


class ExtraTreesConfig(BaseModel):
    """Configuration for Extra Trees."""

    n_estimators: int = Field(default=100, ge=1, description="Number of trees")
    max_depth: int | None = Field(default=None, ge=1, description="Maximum tree depth")
    min_samples_split: int = Field(default=2, ge=2, description="Min samples to split")
    random_state: int | None = Field(default=42, ge=0)


class GradientBoostingConfig(BaseModel):
    """Configuration for Gradient Boosting Regressor."""

    n_estimators: int = Field(default=100, ge=1, description="Number of boosting stages")
    learning_rate: float = Field(default=0.1, gt=0.0, description="Learning rate")
    max_depth: int = Field(default=3, ge=1, description="Maximum tree depth")
    random_state: int | None = Field(default=42, ge=0)


class HistGradientBoostingConfig(BaseModel):
    """Configuration for Histogram Gradient Boosting."""

    max_iter: int = Field(default=100, ge=1, description="Maximum number of iterations")
    learning_rate: float = Field(default=0.1, gt=0.0, description="Learning rate")
    max_depth: int | None = Field(default=None, ge=1, description="Maximum tree depth")
    random_state: int | None = Field(default=42, ge=0)


# Union type for all model configs
ModelConfigType = Annotated[
    Union[
        LinearConfig,
        RidgeConfig,
        LassoConfig,
        ElasticNetConfig,
        SVRConfig,
        KNNConfig,
        RandomForestConfig,
        ExtraTreesConfig,
        GradientBoostingConfig,
        HistGradientBoostingConfig,
    ],
    Field(discriminator=None),
]


# ===== Train Configuration =====
class TrainConfig(BaseModel):
    """Configuration for model training."""

    seed: int = Field(default=42, ge=0, description="Random seed")
    target_col: str = Field(default="quality", description="Target column name")
    id_col: str = Field(default="Id", description="ID column name")
    model_type: Literal[
        "linear",
        "ridge",
        "lasso",
        "elasticnet",
        "svr",
        "knn",
        "rf",
        "extra_trees",
        "gbr",
        "hgb",
    ] = Field(default="linear", description="Model type to train")
    standardize: bool = Field(
        default=False, description="Whether to standardize features (not for tree models)"
    )
    paths: PathsConfig = Field(default_factory=PathsConfig)

    # Model-specific configurations
    linear: LinearConfig = Field(default_factory=LinearConfig)
    ridge: RidgeConfig = Field(default_factory=RidgeConfig)
    lasso: LassoConfig = Field(default_factory=LassoConfig)
    elasticnet: ElasticNetConfig = Field(default_factory=ElasticNetConfig)
    svr: SVRConfig = Field(default_factory=SVRConfig)
    knn: KNNConfig = Field(default_factory=KNNConfig)
    rf: RandomForestConfig = Field(default_factory=RandomForestConfig)
    extra_trees: ExtraTreesConfig = Field(default_factory=ExtraTreesConfig)
    gbr: GradientBoostingConfig = Field(default_factory=GradientBoostingConfig)
    hgb: HistGradientBoostingConfig = Field(default_factory=HistGradientBoostingConfig)

    def get_model_config(self) -> Any:
        """Get configuration for the selected model type."""
        config_map = {
            "linear": self.linear,
            "ridge": self.ridge,
            "lasso": self.lasso,
            "elasticnet": self.elasticnet,
            "svr": self.svr,
            "knn": self.knn,
            "rf": self.rf,
            "extra_trees": self.extra_trees,
            "gbr": self.gbr,
            "hgb": self.hgb,
        }
        return config_map[self.model_type]


# ===== Project Configuration =====
class ProjectConfig(BaseModel):
    """Root configuration for the entire project."""

    split: SplitConfig = Field(default_factory=SplitConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)

    @field_validator("train")
    @classmethod
    def validate_standardize_for_tree_models(cls, v: TrainConfig) -> TrainConfig:
        """Warn if standardize is True for tree-based models."""
        tree_models = {"rf", "extra_trees", "gbr", "hgb"}
        if v.standardize and v.model_type in tree_models:
            # Note: This is a warning, not an error, so we just pass
            # In production, you might want to log a warning here
            pass
        return v
