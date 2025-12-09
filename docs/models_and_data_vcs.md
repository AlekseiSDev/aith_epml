# DVC: данные и модели (GCS)

Краткий гайд для участников: как подтянуть данные/модели и как обновлять их через DVC. Предполагается знание Git.

## Установка и авторизация
1) DVC: `brew install dvc`
2) Google Cloud SDK: `brew install --cask google-cloud-sdk` и `source /usr/local/share/google-cloud-sdk/path.bash.inc`
3) Авторизация (ADC):
   ```bash
   CLOUDSDK_PYTHON=/usr/bin/python3 gcloud auth application-default login
   ```
   Альтернатива: сервисный аккаунт JSON (см. креды ниже).
4) Из-за ограничений macOS указывайте локальный кеш для DVC:
   ```bash
   export DVC_SITE_CACHE_DIR="$PWD/.dvc/site-cache"
   ```
5) Установить проектные зависимости (для sklearn): `uv sync` или `pip install -r requirements.txt` если используете pip-флоу.

## Забрать данные и модели
```bash
export DVC_SITE_CACHE_DIR="$PWD/.dvc/site-cache"
dvc pull   # подтянет data/raw/WineQT.csv и артефакты моделей из gs://aith_epml
```

## Обновить данные (raw → splits)
1) Положить новый raw-файл в `data/raw/` (например, заменить `WineQT.csv`).
2) Добавить в DVC и запушить:
   ```bash
   dvc add data/raw/WineQT.csv
   dvc push              # по умолчанию в gs://aith_epml
   ```
3) В Git коммитим только `.dvc` и конфиги, не сами данные.

## Сделать train/eval/test сплиты
```bash
export DVC_SITE_CACHE_DIR="$PWD/.dvc/site-cache"
python scripts/split_data.py --test-size 0.15 --eval-size 0.15 --seed 42
dvc add data/splits/train.csv data/splits/eval.csv data/splits/test.csv
   dvc push
   git add data/splits/*.dvc
   git commit -m "Add data splits"
   ```
По умолчанию сплиты кладутся в `data/splits/` (train/eval/test), seed=42.

## Обучить и зафиксировать модель (baseline)
- Скрипт: `python scripts/train_baseline.py --model-type {constant|linear} [--strategy mean|median] [--tag name]`
- Артефакты:
  - модель (параметры/веса): `models/<tag>_model.json`
  - метрики тренировки: `models/<tag>_metrics.json`
  где `<tag>` = model-type или переданный `--tag`.
 - Данные по умолчанию: `data/splits/train.csv`, в метриках логируется `split` и `data_path`.

Пример (две версии):
```bash
export DVC_SITE_CACHE_DIR="$PWD/.dvc/site-cache"
python scripts/train_baseline.py --model-type constant --strategy mean --tag const_mean_v1
python scripts/train_baseline.py --model-type linear --tag linear_v1
# Зафиксировать в DVC:
dvc add models/const_mean_v1_model.json models/const_mean_v1_metrics.json
  dvc add models/linear_v1_model.json models/linear_v1_metrics.json
  dvc push   # в gs://aith_epml
  git add models/*.dvc .dvc/config .dvcignore
  git commit -m "Add baseline models (mean/median)"
```
Метрики можно сравнивать между ревизиями Git через `dvc metrics diff --targets models/<tag>_metrics.json <rev>`.

## Оценка сохранённой модели
- Скрипт: `python scripts/evaluate_model.py --model-path models/<tag>_model.json [--metrics-path out.json]`
- По умолчанию берёт датасет `data/raw/WineQT.csv` (можно указать `--data-path data/splits/test.csv`) и пишет метрики в `models/<tag>_model_eval_metrics.json`, включая `split`/`data_path`.
- Результат можно добавить в DVC:
  ```bash
  python scripts/evaluate_model.py --model-path models/linear_v1_model.json
  dvc add models/linear_v1_model_eval_metrics.json
  dvc push
  git add models/*.dvc
  git commit -m "Eval linear_v1"
  ```

## Сравнение версий
- Меняйте `params.json` (например, strategy -> median), делайте `dvc repro` и коммит, чтобы зафиксировать полную цепочку (данные → сплиты → модель → метрики) в одной ревизии.
- Сравнить метрики между ревизиями:
  ```bash
  dvc metrics diff --targets models/metrics.json HEAD~1
  ```
  или по конкретным тегам/веткам Git.

## Быстрые принципы
- Данные и модели версионируем одинаково через DVC (`dvc add/push`), храним только `.dvc` и конфиги в Git.
- Сплиты и модели завязываем на фиксированные seeds/параметры (`params.json`), чтобы `dvc repro` воспроизводил пайплайн end-to-end.
- Метрики и артефакты моделей лежат в `models/` и так же проходят через DVC; дифф метрик делаем `dvc metrics diff`.

## Remotes и креды
- Default remote: `gcs-data` → `gs://aith_epml` (см. `.dvc/config`); резерв `local-data` → `.dvc/storage`.
- Креды не коммитим:
  - ADC (рекомендуется): `gcloud auth application-default login`.
  - JSON ключ: сохранить локально (например, `~/.config/gcloud/aith-epml.json`), подключить:
    ```bash
    dvc remote modify gcs-data credentialpath ~/.config/gcloud/aith-epml.json --local
    ```
    `.dvc/config.local` уже в gitignore.
