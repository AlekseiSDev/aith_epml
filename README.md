# Engineering Practices in ML — wine_quality_epml

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://AlekseiSDev.github.io/aith_epml/)

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

## Docker & reproducibility
```bash
# сборка и запуск (gcloud CLI уже внутри образа)
docker build -t wine-quality-epml .
docker run --rm -it --name wine-quality-epml -v ~/.config/gcloud:/root/.config/gcloud wine-quality-epml bash

# внутри контейнера: DVC уже установлен из uv.lock
export DVC_SITE_CACHE_DIR="$PWD/.dvc/site-cache"

# авторизация GCS (ADC): 
gcloud auth application-default login
# или задайте GOOGLE_APPLICATION_CREDENTIALS на JSON ключ; .dvc/config.local в gitignore

# dvc pull данных/моделей
dvc pull data/raw/WineQT.csv.dvc \
        data/splits/train.csv.dvc data/splits/eval.csv.dvc data/splits/test.csv.dvc \
        models/linear_v1_model.json.dvc models/linear_v1_train_eval_metrics.json.dvc

# тестовая оценка сохранённой модели
python scripts/test_model.py --model-path models/linear_v1_model.json --split-name test

# закрыть контейнер (если завис): 
docker stop wine-quality-epml
```

## ClearML (MLOps)
Сервер ClearML поднимается через Docker Compose:
```bash
docker compose -f configs/clearml_server.docker-compose.yml up -d
```

Compose основан на официальном ClearML Server и использует named volumes (без привязки к `/opt/clearml` на хосте).

Доступы:
- Web UI: `http://localhost:8080`
- API: `http://localhost:8008`
- Fileserver: `http://localhost:8081`

Остановка:
```bash
docker compose -f configs/clearml_server.docker-compose.yml down
```

Аутентификация/SDK (локально):
```bash
clearml-init
```

Прогон эксперимента (пример):
```bash
uv run python scripts/clearml_experiment.py --params configs/ridge_baseline.yaml --task-name ridge_baseline
```

Запуск пайплайна:
```bash
uv run python scripts/clearml_pipeline.py --params params.yaml
```

## Git workflow
- Основная ветка: `master`; рабочая ветка разработки: `develop`.
- Фичи: `feature/<task>`, хотфиксы: `hotfix/<issue>`.
- Рабочий цикл: `feature/<task>` → PR в `develop` → слияние в `master`.
