# Домашка по инженерным практикам в ML — wine_quality_epml

Минимальная структура в стиле Cookiecutter Data Science, развернутая вручную под проект `wine_quality_epml`.

## Структура папок
- `data/` — данные (`raw/`, `processed/`, `interim/`, `external/`); исходник `WineQT.csv` лежит рядом.
- `notebooks/` — ноутбуки для EDA/экспериментов.
- `models/` — обученные модели и артефакты.
- `docs/` — документация и отчеты.
- `reports/figures/` — визуализации и графики.
- `src/wine_quality_epml/` — кодовая база проекта, пустые подпакеты `data/`, `features/`, `models/`, `visualization/` для будущих модулей.

## Почему так
Это упрощенный каркас Cookiecutter DS без генераторов: оставлены базовые примитивы (данные, ноутбуки, модели, доки) и скелет пакета под код.

## Качество кода
- Форматер/линтер: Ruff (`pyproject.toml`, включает security-правила `S`).
- Типизация: MyPy (`mypy.ini`).
- Хуки: `.pre-commit-config.yaml` (ruff fmt/lint, mypy).

Быстрые команды:
```bash
pre-commit install
pre-commit run --all-files
ruff format && ruff check
mypy src
```
