"""Configuration loader with YAML support and validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from ruamel.yaml import YAML

from wine_quality_epml.config.schemas import ProjectConfig


def load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML file and return as dictionary.

    Args:
        path: Path to YAML file

    Returns:
        Dictionary with configuration data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If YAML is invalid
    """
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    yaml = YAML(typ="safe")
    content = yaml.load(path.read_text(encoding="utf-8"))

    if not isinstance(content, dict):
        raise ValueError(f"Configuration file must contain a mapping at root: {path}")

    return content


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """Merge multiple configuration dictionaries.

    Later configs override earlier ones.

    Args:
        *configs: Configuration dictionaries to merge

    Returns:
        Merged configuration dictionary
    """
    result: dict[str, Any] = {}

    for config in configs:
        for key, value in config.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                # Recursively merge nested dicts
                result[key] = merge_configs(result[key], value)
            else:
                result[key] = value

    return result


def apply_env_overrides(config: dict[str, Any], prefix: str = "WINE_QUALITY_") -> dict[str, Any]:
    """Apply environment variable overrides to configuration.

    Environment variables should be named like:
    WINE_QUALITY_TRAIN__MODEL_TYPE=ridge
    WINE_QUALITY_TRAIN__RIDGE__ALPHA=10.0

    Args:
        config: Configuration dictionary
        prefix: Environment variable prefix

    Returns:
        Configuration with environment overrides applied
    """
    result = config.copy()

    for env_key, env_value in os.environ.items():
        if not env_key.startswith(prefix):
            continue

        # Remove prefix and split by double underscore
        config_path = env_key[len(prefix) :].lower().split("__")

        # Navigate to the right place in config
        current = result
        for key in config_path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set the value (try to parse as number/bool if possible)
        final_key = config_path[-1]
        parsed_value: Any = env_value

        # Try to parse as bool
        if env_value.lower() in ("true", "false"):
            parsed_value = env_value.lower() == "true"
        # Try to parse as number
        else:
            try:
                if "." in env_value:
                    parsed_value = float(env_value)
                else:
                    parsed_value = int(env_value)
            except ValueError:
                pass

        current[final_key] = parsed_value

    return result


def load_config(
    path: str | Path,
    *,
    override_paths: list[str | Path] | None = None,
    apply_env: bool = True,
    env_prefix: str = "WINE_QUALITY_",
) -> ProjectConfig:
    """Load and validate configuration from YAML file(s).

    Args:
        path: Path to main configuration file
        override_paths: Optional list of override configuration files
        apply_env: Whether to apply environment variable overrides
        env_prefix: Prefix for environment variables

    Returns:
        Validated ProjectConfig instance

    Raises:
        FileNotFoundError: If configuration file not found
        ValidationError: If configuration is invalid
        ValueError: If YAML is malformed
    """
    main_path = Path(path)
    config_data = load_yaml(main_path)

    # Merge with override configs
    if override_paths:
        override_configs = [load_yaml(Path(p)) for p in override_paths]
        config_data = merge_configs(config_data, *override_configs)

    # Apply environment variable overrides
    if apply_env:
        config_data = apply_env_overrides(config_data, prefix=env_prefix)

    # Validate with Pydantic
    try:
        return ProjectConfig.model_validate(config_data)
    except ValidationError:
        # Re-raise with better formatting
        raise


def save_config(config: ProjectConfig, path: str | Path) -> None:
    """Save configuration to YAML file.

    Args:
        config: ProjectConfig instance to save
        path: Path where to save the configuration
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict and save as YAML
    config_dict = config.model_dump(mode="python", exclude_none=False)

    yaml = YAML()
    yaml.default_flow_style = False
    with output_path.open("w", encoding="utf-8") as f:
        yaml.dump(config_dict, f)
