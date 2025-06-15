import time
import joblib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Dict, Any
from sklearn.metrics import mean_squared_error, r2_score

from service.backend.core.config import logger, MODEL_STORAGE
from service.backend.models.schemas import TrainParams, ModelType, ModelMetrics
from service.backend.models.utils import build_pipeline

MODELS = {}
X_train, X_test, y_train, y_test = None, None, None, None

def initialize(models_dict, x_train, x_test, y_train_data, y_test_data):
    global MODELS, X_train, X_test, y_train, y_test
    MODELS = models_dict
    X_train, X_test = x_train, x_test
    y_train, y_test = y_train_data, y_test_data

def train_job(
    params: TrainParams,
    unique_model_id: str,
    model_type: ModelType
) -> None:
    """Фоновая задача для обучения модели"""
    pipeline = None
    metrics = ModelMetrics().dict()
    start_time = time.time()

    try:
        logger.info("Начало обучения модели %s (%s)",
                    unique_model_id, model_type.model_type)
        pipeline = build_pipeline(params, model_type)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(pipeline.fit, X_train, y_train)
            try:
                future.result(timeout=10)
                train_time = time.time() - start_time
                y_pred = pipeline.predict(X_test)

                metrics.update({
                    "r2_score": r2_score(y_test, y_pred),
                    "mse": mean_squared_error(y_test, y_pred),
                    "train_time": float(train_time),
                })
                logger.info("Модель %s обучена. Метрики: %s",
                            unique_model_id, metrics)

            except FutureTimeoutError:
                logger.warning("Таймаут обучения модели %s",
                               unique_model_id)

    except Exception as e:  # pylint: disable=broad-except
        logger.error("Ошибка обучения модели %s: %s",
                     unique_model_id, str(e), exc_info=True)
    finally:
        if pipeline is not None:
            MODELS[unique_model_id] = {"pipeline":
                                       pipeline, "metrics": metrics}
            try:
                joblib.dump(MODELS[unique_model_id],
                            MODEL_STORAGE / f"{unique_model_id}.pkl")
                logger.info("Модель %s сохранена", unique_model_id)
            except Exception as e:  # pylint: disable=broad-except
                logger.error("Ошибка сохранения модели %s: %s",
                             unique_model_id, str(e), exc_info=True)
