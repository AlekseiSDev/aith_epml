# Pydantic: Памятка по использованию

## Что такое Pydantic?

**Pydantic** - это библиотека для валидации данных и управления настройками в Python, использующая аннотации типов.

Основная идея: вы описываете структуру данных через Python классы с типами, а Pydantic автоматически:
- Проверяет типы данных
- Приводит данные к нужным типам (если возможно)
- Валидирует значения по заданным правилам
- Выдает понятные ошибки при проблемах

## Ключевые возможности Pydantic

### 1. Автоматическая валидация типов

```python
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str
    age: int

# ✓ Работает
user = User(id=1, name="Alice", age=30)

# ✓ Автоматическое приведение типов
user = User(id="1", name="Alice", age="30")  # Строки → числа

# ✗ Ошибка валидации
user = User(id="abc", name="Alice", age=30)
# ValidationError: id должно быть int
```

### 2. Валидация с ограничениями (Field)

```python
from pydantic import BaseModel, Field

class Config(BaseModel):
    alpha: float = Field(gt=0.0, le=100.0, description="Должно быть 0 < alpha ≤ 100")
    n_estimators: int = Field(ge=1, description="Минимум 1")
    name: str = Field(min_length=1, max_length=50)

# ✓ Работает
config = Config(alpha=10.5, n_estimators=100, name="model")

# ✗ Ошибка: alpha должно быть > 0
config = Config(alpha=-5, n_estimators=100, name="model")

# ✗ Ошибка: n_estimators должно быть >= 1
config = Config(alpha=10, n_estimators=0, name="model")
```

**Доступные ограничения:**
- `gt` (greater than) - строго больше
- `ge` (greater or equal) - больше или равно
- `lt` (less than) - строго меньше
- `le` (less or equal) - меньше или равно
- `min_length`, `max_length` - для строк и списков
- `regex` - регулярное выражение для строк

### 3. Значения по умолчанию

```python
class TrainConfig(BaseModel):
    seed: int = 42
    learning_rate: float = 0.01
    model_type: str = "linear"

# Можно создать без параметров
config = TrainConfig()  # Все значения по умолчанию

# Или переопределить только нужные
config = TrainConfig(learning_rate=0.1)
```

### 4. Литералы (строгий перечень значений)

```python
from typing import Literal

class ModelConfig(BaseModel):
    model_type: Literal["linear", "ridge", "lasso", "rf"]
    kernel: Literal["rbf", "linear", "poly"] = "rbf"

# ✓ Работает
config = ModelConfig(model_type="ridge")

# ✗ Ошибка: недопустимое значение
config = ModelConfig(model_type="invalid")
# ValidationError: model_type должно быть 'linear', 'ridge', 'lasso' или 'rf'
```

### 5. Вложенные модели

```python
class Address(BaseModel):
    street: str
    city: str

class User(BaseModel):
    name: str
    address: Address

# Создание с вложенными данными
user = User(
    name="Alice",
    address={"street": "Main St", "city": "NYC"}  # Dict автоматически → Address
)

# Доступ к вложенным полям
print(user.address.city)  # NYC
```

### 6. Кастомные валидаторы

```python
from pydantic import BaseModel, field_validator, model_validator

class SplitConfig(BaseModel):
    test_size: float = Field(gt=0, lt=1)
    eval_size: float = Field(gt=0, lt=1)

    @model_validator(mode='after')
    def check_total_size(self) -> 'SplitConfig':
        """Проверяем, что test + eval < 1"""
        if self.test_size + self.eval_size >= 1.0:
            raise ValueError(f"test + eval должно быть < 1.0")
        return self

# ✓ Работает
config = SplitConfig(test_size=0.2, eval_size=0.2)

# ✗ Ошибка: 0.5 + 0.6 >= 1.0
config = SplitConfig(test_size=0.5, eval_size=0.6)
```

### 7. Загрузка из dict/JSON/YAML

```python
# Из словаря
data = {"alpha": 10.0, "n_estimators": 100}
config = Config.model_validate(data)

# Из JSON строки
json_str = '{"alpha": 10.0, "n_estimators": 100}'
config = Config.model_validate_json(json_str)

# Из YAML (с помощью ruamel.yaml или pyyaml)
import yaml
yaml_data = yaml.safe_load(yaml_file.read_text())
config = Config.model_validate(yaml_data)
```

### 8. Сериализация обратно в dict/JSON

```python
config = Config(alpha=10.0, n_estimators=100)

# В словарь
config_dict = config.model_dump()  # {'alpha': 10.0, 'n_estimators': 100}

# В JSON строку
config_json = config.model_dump_json()

# С исключением полей
config_dict = config.model_dump(exclude={'n_estimators'})
```

### 9. Настройка модели (Config)

```python
class MyModel(BaseModel):
    model_config = {
        'extra': 'forbid',  # Запретить неизвестные поля
        'frozen': True,     # Сделать иммутабельным
        'validate_assignment': True,  # Валидация при изменении
    }
    
    value: int

# ✗ Ошибка: лишние поля запрещены
model = MyModel(value=10, unknown=20)

# ✗ Ошибка: frozen = True
model = MyModel(value=10)
model.value = 20  # Нельзя изменить
```

## Применение в нашем проекте Wine Quality EPML

### Проблемы до Pydantic

**Без Pydantic:**
```python
# Загрузка конфигурации напрямую из YAML
config = yaml.load('params.yaml')

# Проблемы:
# 1. Нет валидации - ошибки обнаруживаются во время выполнения
# 2. Нет автодополнения в IDE
# 3. Опечатки в ключах → KeyError в рантайме
# 4. Неправильные типы → ошибки при обучении модели
# 5. Невалидные значения (alpha=-5) проходят незамеченными

alpha = config['train']['ridge']['alpha']  # Может не существовать
```

**С Pydantic:**
```python
from wine_quality_epml.config.loader import load_config

# Загрузка с валидацией
config = load_config('params.yaml')

# Преимущества:
# 1. ✓ Все ошибки ловятся при загрузке, не в рантайме
# 2. ✓ IDE знает структуру и дает автодополнение
# 3. ✓ Опечатки невозможны - config.train.model_tipe → ошибка
# 4. ✓ Типы гарантированы - alpha всегда float
# 5. ✓ Валидация значений - alpha=-5 → ValidationError

alpha = config.train.ridge.alpha  # Всегда существует и валидно
```

### Структура конфигураций в проекте

```
ProjectConfig (корень)
├── split: SplitConfig
│   ├── seed: int (≥ 0)
│   ├── test_size: float (0 < x < 1)
│   └── eval_size: float (0 < x < 1)
│       └── validator: test + eval < 1.0
└── train: TrainConfig
    ├── seed: int (≥ 0)
    ├── target_col: str
    ├── model_type: Literal[10 вариантов]
    ├── standardize: bool
    ├── paths: PathsConfig (9 путей)
    └── model configs:
        ├── ridge: RidgeConfig (alpha > 0)
        ├── lasso: LassoConfig (alpha > 0)
        ├── gbr: GradientBoostingConfig
        │   ├── n_estimators ≥ 1
        │   ├── learning_rate > 0
        │   └── max_depth ≥ 1
        └── ... (еще 7 моделей)
```

### Примеры использования в проекте

#### 1. Безопасная загрузка конфигурации

```python
from wine_quality_epml.config.loader import load_config
from pydantic import ValidationError

try:
    config = load_config('params.yaml')
    print(f"✓ Конфигурация валидна")
    print(f"  Модель: {config.train.model_type}")
    print(f"  Seed: {config.train.seed}")
except ValidationError as e:
    print(f"✗ Ошибка в конфигурации:")
    for error in e.errors():
        loc = ' -> '.join(str(x) for x in error['loc'])
        print(f"  • {loc}: {error['msg']}")
    exit(1)
```

#### 2. Доступ к параметрам с автодополнением

```python
# IDE знает все поля и их типы
config = load_config('params.yaml')

# Автодополнение работает на всех уровнях
model_type = config.train.model_type  # str
seed = config.train.seed  # int
alpha = config.train.ridge.alpha  # float

# Получение конфига текущей модели
model_cfg = config.train.get_model_config()
if config.train.model_type == "ridge":
    print(f"Ridge alpha: {model_cfg.alpha}")
```

#### 3. Композиция конфигураций

```python
# Базовая конфигурация + переопределения
config = load_config(
    'params.yaml',
    override_paths=['configs/ridge_baseline.yaml']
)

# Значения из ridge_baseline.yaml переопределяют params.yaml
# При этом все валидируется!
```

#### 4. Environment variables

```bash
# Переопределить через переменные окружения
export WINE_QUALITY_TRAIN__MODEL_TYPE=lasso
export WINE_QUALITY_TRAIN__LASSO__ALPHA=0.5

python -m wine_quality_epml.pipeline.runner
```

```python
# В коде это автоматически применяется
config = load_config('params.yaml', apply_env=True)
# config.train.model_type будет 'lasso'
# config.train.lasso.alpha будет 0.5
```

#### 5. Валидация в Luigi Tasks

```python
from wine_quality_epml.config.loader import load_config

class TrainModelTask(luigi.Task):
    params_path = luigi.Parameter(default="params.yaml")

    def run(self):
        # Загрузка с валидацией
        config = load_config(self.params_path)
        
        # Гарантируется:
        # - model_type существует и валиден
        # - параметры модели корректны
        # - пути существуют
        
        model_type = config.train.model_type
        model_cfg = config.train.get_model_config()
        
        # Безопасно использовать
        estimator = build_estimator(model_cfg)
```

### Что защищает Pydantic в нашем проекте?

#### 1. Защита от опечаток
```yaml
# params.yaml с опечаткой
train:
  model_tipe: ridge  # Опечатка!
```
→ ValidationError: model_type is required

#### 2. Защита от невалидных значений
```yaml
train:
  ridge:
    alpha: -5  # Отрицательное значение!
```
→ ValidationError: alpha must be > 0

#### 3. Защита от несовместимых параметров
```yaml
split:
  test_size: 0.6
  eval_size: 0.5  # test + eval = 1.1 > 1.0!
```
→ ValidationError: test_size + eval_size must be < 1.0

#### 4. Защита от неизвестных моделей
```yaml
train:
  model_type: deep_neural_network  # Не поддерживается!
```
→ ValidationError: model_type должен быть одним из: linear, ridge, lasso, ...

#### 5. Защита от пропущенных полей
```yaml
train:
  # target_col пропущен!
  model_type: ridge
```
→ ValidationError: target_col is required

## Преимущества для разработки

### 1. Раннее обнаружение ошибок
Ошибки в конфигурации обнаруживаются **до запуска обучения**, а не через 2 часа когда модель уже обучается.

### 2. Самодокументирующийся код
```python
class RidgeConfig(BaseModel):
    """Configuration for Ridge regression."""
    
    alpha: float = Field(
        default=1.0,
        gt=0.0,
        description="Regularization strength, must be positive"
    )
```

Код сам объясняет что нужно и какие ограничения есть.

### 3. IDE поддержка
- Автодополнение всех полей
- Подсказки типов
- Рефакторинг работает корректно
- Ctrl+Click переходит к определению

### 4. Легкое тестирование
```python
def test_invalid_config():
    with pytest.raises(ValidationError):
        ProjectConfig(
            split=SplitConfig(seed=-5)  # Должно быть ≥ 0
        )
```

### 5. Версионирование конфигураций
Можно легко сохранить валидированную конфигурацию:
```python
config = load_config('params.yaml')
save_config(config, 'configs/experiment_001.yaml')
```

## Когда использовать Pydantic?

✅ **Используйте Pydantic когда:**
- Загружаете конфигурацию из файлов (YAML, JSON, TOML)
- Валидируете данные от пользователя
- Описываете API схемы (FastAPI)
- Нужна строгая типизация в Python
- Хотите раннее обнаружение ошибок

❌ **Не нужен Pydantic когда:**
- Простые скрипты без конфигурации
- Данные уже валидированы на уровне базы данных
- Производительность критична (хотя Pydantic довольно быстр)

## Полезные ссылки

- [Официальная документация Pydantic](https://docs.pydantic.dev/)
- [Pydantic V2 Migration Guide](https://docs.pydantic.dev/latest/migration/)
- [Validators and Computed Fields](https://docs.pydantic.dev/latest/concepts/validators/)

## Итого

**Pydantic в нашем проекте решает:**
1. ✅ Валидация конфигурации до запуска пайплайна
2. ✅ Автодополнение и проверка типов в IDE
3. ✅ Защита от опечаток и невалидных значений
4. ✅ Самодокументирующиеся схемы конфигураций
5. ✅ Легкая композиция и переопределение конфигураций
6. ✅ Environment variables support
7. ✅ Понятные сообщения об ошибках

Вместо загрузки "сырых" dict из YAML мы получаем **типизированные, валидированные объекты** с гарантиями корректности еще до запуска обучения.
