# Engineering Practices in ML — wine_quality_epml

Repo for Engineering Practices in ML course. Минимальная структура в стиле Cookiecutter Data Science, развернутая вручную под проект `wine_quality_epml`.

## Структура папок
- `data/raw/` — данные под DVC; исходник `WineQT.csv` лежит здесь.
- `notebooks/` — ноутбуки для EDA/экспериментов.
- `models/` — обученные модели и артефакты.
- `docs/` — документация и отчеты.
- `src/wine_quality_epml/` — кодовая база проекта, пустые подпакеты `data/`, `features/`, `models/`, `visualization/` для будущих модулей.

## Почему так
Это упрощенный каркас Cookiecutter DS без генераторов: оставлены базовые примитивы (данные, ноутбуки, модели, доки) и скелет пакета под код.

## Качество кода
- Форматер/линтер: Ruff (`pyproject.toml`, включает security-правила `S`).
- Типизация: MyPy (`mypy.ini`).
- Хуки: `.pre-commit-config.yaml` (ruff fmt/lint, mypy).

Быстрые команды:
```bash
uv run pre-commit install
uv run pre-commit run --all-files
ruff format && ruff check
mypy src
```

## Управление зависимостями (uv)
- Пакетный менеджер: `uv` (пины в `pyproject.toml`, лок-файл `uv.lock`).
- Основные зависимости пока пустые; дев-зависимости: `ruff`, `mypy`, `pre-commit` (см. `[tool.uv].dev-dependencies`).
- Виртуальное окружение (локально):
  ```bash
  uv venv .venv
  source .venv/bin/activate
  uv sync --dev
  ```
- Запуски через uv:
  ```bash
  uv run ruff format
  uv run ruff check
  uv run mypy src
  uv run pre-commit run --all-files
  ```

## Docker
- Сборка: `docker build -t wine-quality-epml .`
- Запуск контейнера с шеллом внутри: `docker run --rm -it wine-quality-epml`

## Git workflow
- Основная ветка: `master`; рабочая ветка разработки: `develop`.
- Фичи: `feature/<task>`, хотфиксы: `hotfix/<issue>`.
- Рабочий цикл: `feature/<task>` → PR в `develop` → слияние в `master`.
