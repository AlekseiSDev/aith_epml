# Версионирование данных и моделей с DVC

Док для участников проекта: как получить данные и обновлять их с DVC. Предполагается знание Git.

## Быстрый старт (получить данные)
1) Установить зависимости:
   - DVC: `brew install dvc`
   - Google Cloud SDK: `brew install --cask google-cloud-sdk` и `source /usr/local/share/google-cloud-sdk/path.bash.inc`
2) Авторизоваться в GCP:
   ```bash
   CLOUDSDK_PYTHON=/usr/bin/python3 gcloud auth application-default login
   ```
   или использовать ключ сервисного аккаунта (см. ниже).
3) Из корня репо скачать данные:
   ```bash
   dvc pull   # default remote = gs://aith_epml
   ```
   Появится `data/raw/WineQT.csv`.

## Обновить данные (raw) или добавить модель
1) Положить новые/обновлённые файлы в `data/raw/` (для данных) или `models/` (для моделей).
2) Добавить в DVC:
   ```bash
   dvc add data/raw/WineQT.csv             # данные
   dvc add models/<model-file>             # модель/метрики
   ```
   Появятся файлы `*.dvc` рядом.
3) Отправить в хранилище:
   ```bash
   dvc push        # по умолчанию в gs://aith_epml
   # при желании локальный бэкап: dvc push -r local-data
   ```
4) Закоммитить в Git только метаданные:
   ```
   git add data/raw/WineQT.csv.dvc .dvc/config .dvcignore
   git add models/<model-file>.dvc         # если версия модели нужна
   git commit -m "Update data/model"
   ```
   Не коммитить сами данные/модели.

## Remotes и креды
- По умолчанию: `gcs-data` → `gs://aith_epml` (см. `.dvc/config`).
- Резерв: `local-data` → `.dvc/storage`.
- Креды (не в Git):
  - ADC: `gcloud auth application-default login` (рекомендуется).
  - Или JSON ключ: сохранить локально (например, `~/.config/gcloud/aith-epml.json`), затем:
    ```bash
    dvc remote modify gcs-data credentialpath ~/.config/gcloud/aith-epml.json --local
    ```
    Файл `.dvc/config.local` gitignore уже прикрыт.

## Примечания
- Не создавайте лишних каталогов: используем `data/raw/` и `models/` как основные точки.
- Если DVC просит PATH к gcloud, убедитесь, что `path.bash.inc` подключён или задайте `CLOUDSDK_PYTHON=/usr/bin/python3` при вызовах gcloud.
