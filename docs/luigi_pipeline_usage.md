# Luigi Pipeline для Wine Quality EPML

Автоматизированный ML пайплайн на основе Luigi для проекта wine_quality_epml.

## Установка зависимостей

```bash
uv sync
```

## Структура пайплайна

Пайплайн состоит из трех основных задач:

1. **SplitDataTask** - разделение данных на train/eval/test
2. **TrainModelTask** - обучение модели (зависит от SplitDataTask)
3. **TestModelTask** - тестирование модели (зависит от TrainModelTask)

## Запуск

### Базовый запуск (полный пайплайн)

```bash
# Через runner.py
uv run python -m wine_quality_epml.pipeline.runner --local-scheduler

# Напрямую через Luigi CLI
luigi --module wine_quality_epml.pipeline.test_model_task TestModelTask --local-scheduler
```

### Запуск с параметрами

```bash
# С указанием пути к конфигурации
uv run python -m wine_quality_epml.pipeline.runner --params params.yaml --local-scheduler

# С параллельным выполнением (несколько workers)
uv run python -m wine_quality_epml.pipeline.runner --workers 4 --local-scheduler
```

### Запуск отдельной задачи

```bash
# Только разделение данных
luigi --module wine_quality_epml.pipeline.split_data_task SplitDataTask --local-scheduler

# Только обучение (автоматически выполнит split, если нужно)
luigi --module wine_quality_epml.pipeline.train_model_task TrainModelTask --local-scheduler
```

## Кэширование

Luigi автоматически кэширует результаты выполнения задач. Если выходные файлы существуют, задача будет пропущена. Для принудительного перезапуска удалите соответствующие выходные файлы:

```bash
# Перезапустить обучение
rm -f models/exp_model.pkl models/exp_model_meta.json

# Перезапустить тестирование
rm -f reports/test_metrics.json reports/test_predictions.json

# Полный перезапуск
rm -f models/exp_model.pkl reports/test_metrics.json
```

## Конфигурация

Параметры пайплайна настраиваются в `params.yaml`:

- `split.*` - параметры разделения данных
- `train.model_type` - выбор алгоритма (ridge, lasso, gbr, rf, и т.д.)
- `train.<model_type>` - гиперпараметры конкретной модели
- `train.paths.*` - пути к выходным файлам

## Настройки Luigi

Глобальные настройки Luigi находятся в `luigi.cfg`:

- Порты для планировщика
- Таймауты и интервалы
- Коды возврата для различных сценариев

## Визуализация (опционально)

Для визуализации выполнения пайплайна можно запустить Luigi Central Scheduler:

```bash
luigid --port 8082
```

Затем запустить пайплайн без `--local-scheduler` и открыть http://localhost:8082 в браузере.
