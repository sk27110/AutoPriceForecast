
# Car Price Prediction API

API-сервис для обучения и предсказания цен автомобилей на основе пользовательских или предобученных моделей линейной регрессии (LinearRegression, Ridge, Lasso).


## 🚀 Запуск

```bash
uvicorn main:app --reload
```

## 📁 Структура проекта

```
service/
├── backend/
│   ├── api/               # FastAPI endpoints
│   ├── core/              # Конфигурации и логгирование
│   ├── datasets/          # CSV-датасеты (например, processed_data.csv)
│   ├── models/            # Модели, схемы
│   ├── preprocessing/     # Пользовательские трансформеры
│   ├── saved_models/      # Сохранённые модели (.pkl)
│   └── service/           # Обучение моделей
└── main.py                # Точка входа
```

## 📊 Обработка признаков

- Числовые признаки масштабируются с помощью `StandardScaler`
- Категориальные признаки проходят через `TitleExtractor`, затем `OneHotEncoder`

## 🧠 Поддерживаемые модели

- LinearRegression
- Ridge
- Lasso

## 🧪 Обучение моделей

Доступны эндпоинты:
- `POST /fit_linearregression`
- `POST /fit_ridge`
- `POST /fit_lasso`

Перед обучением модель идентифицируется по `model_id`, и обучение выполняется в фоне.

## 🔮 Предсказания

- `POST /predict-one`: предсказание по одному автомобилю
- `POST /predict-multiple`: пакетное предсказание по CSV-файлу

## 🧰 Управление моделями

- `GET /models`: список всех моделей
- `POST /set`: установить активную модель
- `GET /get_dataset`: получить датасет

## ♻️ Работа с предобученными моделями

- `GET /pretrained/scan`: сканировать сохранённые модели
- `POST /pretrained/load`: загрузить модель из файла
- `POST /pretrained/activate`: активировать модель

## 📄 Пример запроса на предсказание

```json
POST /predict-one
{
  "title": "Toyota Camry",
  "year": 2015,
  "mileage": 90000,
  "transmission": "automatic",
  "body_type": "sedan",
  "drive_type": "front",
  "color": "black",
  "engine_capacity": 2.5,
  "engine_power": 181,
  "fuel_type": "gasoline",
  "travel_distance": 15.5
}
```

## 📓 Логирование

Все события логируются в `logs/app.log`. Также вывод доступен в консоли.

## 🧩 Зависимости

- `FastAPI`
- `scikit-learn`
- `pandas`
- `joblib`
