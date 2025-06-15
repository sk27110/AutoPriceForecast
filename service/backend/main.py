import os
import joblib
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import FastAPI
import pandas as pd
from sklearn.model_selection import train_test_split

from service.backend.core.config import MODEL_STORAGE, DATASET_DIR, logger
from service.backend.api.router import router as model_router
from service.backend.models.training import initialize as training_init
from service.backend.api.router import initialize as router_init


app = FastAPI(title="API для предсказания цен на автомобили")

MODELS: Dict[str, Dict[str, Any]] = {}
ACTIVE_MODEL_ID: Optional[str] = None

try:
    DATASET = pd.read_csv(DATASET_DIR / "processed_data.csv")
    DATA = DATASET.drop("price", axis=1)
    TARGET = DATASET["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        DATA, TARGET, random_state=42, shuffle=True
    )
except Exception as err:
    raise RuntimeError(f"Ошибка загрузки данных: {str(err)}") from err


for model_file in MODEL_STORAGE.glob("*.pkl"):
    model_id = model_file.stem
    try:
        MODELS[model_id] = joblib.load(model_file)
        logger.info("Загружена модель: %s", model_id)
    except Exception as err:  # pylint: disable=broad-except
        logger.error("Ошибка загрузки модели %s: %s",
                    model_id, str(err), exc_info=True)


router_init(MODELS, ACTIVE_MODEL_ID, DATASET)
training_init(MODELS, X_train, X_test, y_train, y_test)

app.include_router(model_router)

for model_file in MODEL_STORAGE.glob("*.pkl"):
    model_id = model_file.stem
    try:
        MODELS[model_id] = joblib.load(model_file)
        logger.info("Загружена модель: %s", model_id)
    except Exception as err:  # pylint: disable=broad-except
        logger.error("Ошибка загрузки модели %s: %s",
                     model_id, str(err), exc_info=True)
        