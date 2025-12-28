# Отчет по ДЗ: wine_quality_epml

## Структура проекта
- Каркас в стиле Cookiecutter DS: `data/`, `notebooks/`, `models/`, `docs/`, `reports/`, `src/wine_quality_epml/`.
- Исходные данные (`WineQT.csv`) остаются в `data/`.

## Качество кода
- Ruff: форматер и линтер с security-правилами `S` (`pyproject.toml`).
- MyPy: типизация (`mypy.ini`).
- pre-commit: хуки на Ruff (fmt+lint) и MyPy (`.pre-commit-config.yaml`).

## Управление зависимостями
- Менеджер: `uv`, зависимости и пины в `pyproject.toml`, лок-файл `uv.lock`.
- Dev-зависимости: `ruff`, `mypy`, `pre-commit` (см. `[tool.uv].dev-dependencies`).
- Виртуалка и установка:
  ```bash
  uv venv .venv
  source .venv/bin/activate
  uv sync --dev
  uv run pre-commit install
  ```

## Docker
- Базовый образ: `python:3.10-slim`, установка `uv`, сборка зависимостей из `pyproject.toml`/`uv.lock`.
- Команды: `docker build -t wine-quality-epml .`, затем `docker run --rm -it wine-quality-epml`.

## Git workflow
- Основные ветки: `master` (или `main`) и `develop`; рабочие фичи в `feature/<task>`, хотфиксы в `hotfix/<issue>`.
- Текущая работа велась в `master`; далее рекомендуется работать в ветках `feature/<task>` (например, `feature/hw_2`) с PR в `develop`, а затем в `master`.

## Скриншот
![Установка и коммит](screen.jpg)
