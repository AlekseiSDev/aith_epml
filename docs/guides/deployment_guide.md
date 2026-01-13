# Руководство по развертыванию (Deployment Guide)

В данном руководстве описан процесс развертывания проекта Wine Quality EPML с использованием Docker и настройки окружения.

## 1. Использование Docker

Проект поставляется с `Dockerfile`, который содержит все необходимые зависимости, инструменты (gcloud, DVC) и настроенное Python-окружение.

### Сборка образа

Для сборки образа выполните следующую команду из корня проекта:

```bash
docker build -t wine-quality-epml .
```

### Запуск контейнера

При запуске рекомендуется пробрасывать конфигурацию `gcloud` для доступа к удаленному хранилищу DVC (Google Cloud Storage):

```bash
docker run --rm -it \
  --name wine-quality-epml \
  -v ~/.config/gcloud:/root/.config/gcloud \
  wine-quality-epml bash
```

### Настройка внутри контейнера

После входа в контейнер необходимо инициализировать доступ к данным:

1. **Авторизация в GCS (если не проброшены креды):**
   ```bash
   gcloud auth application-default login
   ```

2. **Получение данных через DVC:**
   ```bash
   export DVC_SITE_CACHE_DIR="$PWD/.dvc/site-cache"
   dvc pull
   ```

## 2. Локальное развертывание (без Docker)

Если вы предпочитаете развертывание напрямую в системе, используйте менеджер пакетов `uv`.

### Подготовка

1. **Установка uv:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Настройка окружения:**
   ```bash
   uv venv .venv
   source .venv/bin/activate
   uv sync --dev
   ```

## 3. Развертывание ClearML Server

Для отслеживания экспериментов в промышленном режиме или команде разверните сервер ClearML через Docker Compose:

```bash
docker compose -f configs/clearml_server.docker-compose.yml up -d
```

После запуска интерфейс будет доступен по адресу http://localhost:8080.
