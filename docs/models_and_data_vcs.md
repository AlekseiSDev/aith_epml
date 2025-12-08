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

## Забрать данные и модели
```bash
export DVC_SITE_CACHE_DIR="$PWD/.dvc/site-cache"
dvc pull   # подтянет data/raw/WineQT.csv и артефакты моделей из gs://aith_epml
```

## Обновить данные
1) Положить новый raw-файл в `data/raw/` (например, заменить `WineQT.csv`).
2) Добавить в DVC и запушить:
   ```bash
   dvc add data/raw/WineQT.csv
   dvc push              # по умолчанию в gs://aith_epml
   ```
3) В Git коммитим только `.dvc` и конфиги, не сами данные.

## Обучить и зафиксировать модель (baseline, без DVC пайплайна)
- Скрипт: `python scripts/train_baseline.py --strategy {mean|median} [--tag name]`
- По умолчанию сохранит:
  - модель (параметры/веса): `models/<tag>_model.json`
  - метрики: `models/<tag>_metrics.json`
  где `<tag>` = strategy или переданный `--tag`.

Пример (две версии):
```bash
export DVC_SITE_CACHE_DIR="$PWD/.dvc/site-cache"
python scripts/train_baseline.py --strategy mean --tag mean_v1
python scripts/train_baseline.py --strategy median --tag median_v1
# Зафиксировать в DVC:
dvc add models/mean_v1_model.json models/mean_v1_metrics.json
dvc add models/median_v1_model.json models/median_v1_metrics.json
dvc push   # в gs://aith_epml
git add models/*.dvc .dvc/config .dvcignore
git commit -m "Add baseline models (mean/median)"
```
Метрики можно сравнивать между ревизиями Git через `dvc metrics diff --targets models/<tag>_metrics.json <rev>`.

## Сравнение версий
- Меняйте `params.json` (например, strategy -> median), делайте `dvc repro` и коммит.
- Сравнить метрики между ревизиями:
  ```bash
  dvc metrics diff --targets models/metrics.json HEAD~1
  ```
  или по конкретным тегам/веткам Git.

## Remotes и креды
- Default remote: `gcs-data` → `gs://aith_epml` (см. `.dvc/config`); резерв `local-data` → `.dvc/storage`.
- Креды не коммитим:
  - ADC (рекомендуется): `gcloud auth application-default login`.
  - JSON ключ: сохранить локально (например, `~/.config/gcloud/aith-epml.json`), подключить:
    ```bash
    dvc remote modify gcs-data credentialpath ~/.config/gcloud/aith-epml.json --local
    ```
    `.dvc/config.local` уже в gitignore.
