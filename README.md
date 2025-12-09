# Engineering Practices in ML — wine_quality_epml

Repo for Engineering Practices in ML course. Минимальная структура в стиле Cookiecutter Data Science, развернутая вручную под проект `wine_quality_epml`.

## Структура папок
- `data/raw/` — данные под DVC; исходник `WineQT.csv` лежит здесь.
- `notebooks/` — ноутбуки для EDA/экспериментов.
- `models/` — обученные модели и артефакты.
- `docs/` — документация и отчеты.
- `reports/figures/` — визуализации и отчётные материалы (если понадобятся).
- `src/wine_quality_epml/` — кодовая база проекта, пустые подпакеты `data/`, `features/`, `models/`, `visualization/` для будущих модулей.

## Почему так
Это упрощенный каркас Cookiecutter DS без генераторов: оставлены базовые примитивы (данные, ноутбуки, модели, доки) и скелет пакета под код.

## Качество кода / pre-commit
Форматер/линтер: Ruff (`pyproject.toml`, с security-правилами `S`); типизация: MyPy (`mypy.ini`); хуки `.pre-commit-config.yaml` гоняют ruff fmt/lint и mypy по `src` и `scripts`.

```bash
uv run pre-commit install             # поставить хуки
uv run pre-commit run --all-files     # основной чек перед коммитом

# альтернатива по частям:
uv run ruff format && uv run ruff check
uv run mypy src scripts
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

## Версионирование данных
- Инструмент: DVC, remote `gs://aith_epml` (см. `.dvc/config`), локальный кеш: `export DVC_SITE_CACHE_DIR="$PWD/.dvc/site-cache"`.
- Забрать артефакты: `dvc pull` подтянет `data/raw/WineQT.csv` и сплиты.
- Обновить raw: заменить `data/raw/WineQT.csv` → `dvc add data/raw/WineQT.csv` → `dvc push` → в Git кладём только `.dvc`.
- Сплиты: `python scripts/split_data.py --test-size 0.15 --eval-size 0.15 --seed 42`, затем `dvc add data/splits/*.csv && dvc push`.
- Детали и команды: `docs/models_and_data_vcs.md`.

## Версионирование моделей
- Бейзлайн-тренировка: `python scripts/train_baseline.py --model-type {constant|linear} [--strategy mean|median] [--tag name]` использует сплиты из `data/splits/`.
- Артефакты: `models/<tag>_model.json` + `models/<tag>_metrics.json`; добавляем в DVC (`dvc add models/<...>.json && dvc push`), коммитим только `.dvc`.
- Оценка сохранённой модели: `python scripts/evaluate_model.py --model-path models/<tag>_model.json` → метрики `models/<tag>_model_eval_metrics.json`, также через `dvc add/push`.
- Сравнение версий: `dvc metrics diff --targets models/<tag>_metrics.json <rev>`; для полной цепочки используем `dvc repro` после правок `params.json`.
- Расширенное описание и примеры: `docs/models_and_data_vcs.md`.

## Docker
- Сборка: `docker build -t wine-quality-epml .`
- Запуск контейнера с шеллом внутри: `docker run --rm -it wine-quality-epml`

## Git workflow
- Основная ветка: `master`; рабочая ветка разработки: `develop`.
- Фичи: `feature/<task>`, хотфиксы: `hotfix/<issue>`.
- Рабочий цикл: `feature/<task>` → PR в `develop` → слияние в `master`.
