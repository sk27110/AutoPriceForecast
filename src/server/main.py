import os
import logging
import time
import io
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Annotated, Optional, Dict, List, Any
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError)
import joblib
import pandas as pd
import numpy as np
from fastapi import (
    FastAPI,
    HTTPException,
    BackgroundTasks,
    Query,
    UploadFile,
    File,
)
from pydantic import BaseModel, Field
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


MODEL_STORAGE = Path("saved_models")
DATASET_DIR = Path("datasets")
MODEL_STORAGE.mkdir(exist_ok=True)
os.makedirs("logs", exist_ok=True)


try:
    DATASET = pd.read_csv(DATASET_DIR / "processed_data.csv")
    DATA = DATASET.drop("price", axis=1)
    TARGET = DATASET["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        DATA, TARGET, random_state=42, shuffle=True
    )
except Exception as err:
    raise RuntimeError(f"Ошибка загрузки данных: {str(err)}") from err


LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("app")
logger.setLevel(logging.INFO)

file_handler = RotatingFileHandler(
    filename=LOG_DIR / "app.log",
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8",
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


numerical_cols = [
    "year",
    "mileage",
    "engine_capacity",
    "engine_power",
    "travel_distance",
]
categorical_cols = [
    "title",
    "transmission",
    "body_type",
    "drive_type",
    "color",
    "fuel_type",
]


class TitleExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, column: str):
        """Инициализация с указанием имени столбца"""
        self.column = column

    def fit(self, _X: pd.DataFrame,
            _y: Optional[pd.Series] = None) -> 'TitleExtractor':
        """Требуемый метод fit (не выполняет действий)"""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Объединение первых слов в названии автомобиля"""
        X = X.copy()
        split_title = X[self.column].str.split()
        X[self.column] = np.where(
            split_title.str.len() >= 2,
            split_title.str[0] + split_title.str[1],
            split_title.str[0]
        )
        return X


app = FastAPI(title="API для предсказания цен на автомобили")


MODELS: Dict[str, Dict[str, Any]] = {}
ACTIVE_MODEL_ID: Optional[str] = None


for model_file in MODEL_STORAGE.glob("*.pkl"):
    model_id = model_file.stem
    try:
        MODELS[model_id] = joblib.load(model_file)
        logger.info("Загружена модель: %s", model_id)
    except Exception as err:  # pylint: disable=broad-except
        logger.error("Ошибка загрузки модели %s: %s",
                     model_id, str(err), exc_info=True)


class TrainParams(BaseModel):
    """Базовый класс параметров обучения"""


class LinearRegressionTrainParams(TrainParams):
    """Параметры для линейной регрессии"""
    fit_intercept: bool = Field(default=True,
                                description="Использовать intercept")
    n_jobs: Optional[int] = Field(default=None, description="Количество ядер")
    copy_X: bool = Field(default=True, description="Копировать данные")


class RidgeTrainParams(TrainParams):
    """Параметры для Ridge регрессии"""
    alpha: float = Field(default=1.0, description="Коэффициент регуляризации")
    fit_intercept: bool = Field(default=True,
                                description="Использовать intercept")
    max_iter: Optional[int] = Field(default=None,
                                    description="Максимум итераций")
    solver: str = Field(default="auto",
                        description="Алгоритм оптимизации")
    tol: float = Field(default=0.0001,
                       description="Точность остановки")


class LassoTrainParams(TrainParams):
    """Параметры для Lasso регрессии"""
    alpha: float = Field(default=1.0,
                         description="Коэффициент регуляризации")
    fit_intercept: bool = Field(default=True,
                                description="Использовать intercept")
    max_iter: Optional[int] = Field(default=1000,
                                    description="Максимум итераций")
    selection: str = Field(default="cyclic",
                           description="Метод выбора признаков")
    tol: float = Field(default=0.0001,
                       description="Точность остановки")


class ModelType(BaseModel):
    """Тип модели для обучения"""
    model_type: str = Field(default="LinearRegression",
                            description="Тип модели")


class ModelID(BaseModel):
    """Идентификатор модели"""
    id: str = Field(default="new_model", description="Уникальный ID модели")


class PredictOneInput(BaseModel):
    """Данные для единичного предсказания"""
    title: str = Field(..., description="Название автомобиля")
    year: int = Field(..., description="Год выпуска")
    mileage: float = Field(..., description="Пробег")
    transmission: str = Field(..., description="Тип коробки передач")
    body_type: str = Field(..., description="Тип кузова")
    drive_type: str = Field(..., description="Привод")
    color: str = Field(..., description="Цвет")
    engine_capacity: float = Field(..., description="Объем двигателя")
    engine_power: float = Field(..., description="Мощность двигателя")
    fuel_type: str = Field(..., description="Тип топлива")
    travel_distance: float = Field(..., description="Расстояние поездки")


class ModelMetrics(BaseModel):
    """Метрики качества модели"""
    r2_score: Optional[float] = Field(None, description="R-квадрат")
    mse: Optional[float] = Field(None, description="Среднеквадратичная ошибка")
    train_time: Optional[float] = Field(None, description="Время обучения")


class ModelInfo(BaseModel):
    """Информация о модели"""
    id: str = Field(..., description="Идентификатор модели")
    model_type: str = Field(..., description="Тип модели")
    hyperparameters: dict = Field(..., description="Параметры модели")
    is_active: bool = Field(..., description="Активна ли модель")
    metrics: ModelMetrics = Field(...,
                                  description="Метрики модели")
    fit_intercept: Optional[bool] = Field(None,
                                          description="""Используется
                                          ли intercept""")


def build_pipeline(
    params: TrainParams,
    model_type: ModelType
) -> Pipeline:
    """Сборка pipeline для обработки данных и обучения модели"""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("title_extractor", TitleExtractor(column="title")),
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    column_trans = ColumnTransformer([
        ("cat", categorical_transformer, categorical_cols),
        ("num", numeric_transformer, numerical_cols),
    ], remainder="drop")

    model_cls_map = {
        "LinearRegression": LinearRegression,
        "Ridge": Ridge,
        "Lasso": Lasso,
    }

    model_cls = model_cls_map[model_type.model_type]
    model_params = params.model_dump()
    model = model_cls(**model_params)

    return Pipeline([
        ("preprocessor", column_trans),
        ("classifier", model),
    ])


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


@app.post("/fit_linearregression", response_model=ModelInfo)
def train_linear_model(
    params: LinearRegressionTrainParams,
    model_id_param: ModelID,
    background_tasks: BackgroundTasks,
) -> ModelInfo:
    """Обучение модели линейной регрессии"""
    model_type = ModelType(model_type="LinearRegression")
    unique_model_id = model_id_param.id

    if unique_model_id in MODELS:
        logger.warning("Модель %s уже существует",
                       unique_model_id)
        raise HTTPException(400,
                            detail=f"Модель {unique_model_id} уже существует")

    logger.info("Старт обучения LinearRegression для модели %s",
                unique_model_id)
    background_tasks.add_task(train_job, params,
                              unique_model_id, model_type)

    return ModelInfo(
        id=unique_model_id,
        model_type=model_type.model_type,
        hyperparameters=params.dict(),
        is_active=False,
        metrics=ModelMetrics(),
        fit_intercept=params.fit_intercept,
    )


@app.post("/fit_ridge", response_model=ModelInfo)
def train_ridge_model(
    params: RidgeTrainParams,
    model_id_param: ModelID,
    background_tasks: BackgroundTasks,
) -> ModelInfo:
    """Обучение Ridge регрессии"""
    model_type = ModelType(model_type="Ridge")
    unique_model_id = model_id_param.id

    if unique_model_id in MODELS:
        logger.warning("Модель %s уже существует",
                       unique_model_id)
        raise HTTPException(400,
                            detail=f"Модель {unique_model_id} уже существует")

    logger.info("Старт обучения Ridge для модели %s",
                unique_model_id)
    background_tasks.add_task(train_job, params,
                              unique_model_id, model_type)

    return ModelInfo(
        id=unique_model_id,
        model_type=model_type.model_type,
        hyperparameters=params.dict(),
        is_active=False,
        metrics=ModelMetrics(),
        fit_intercept=params.fit_intercept,
    )


@app.post("/fit_lasso", response_model=ModelInfo)
def train_lasso_model(
    params: LassoTrainParams,
    model_id_param: ModelID,
    background_tasks: BackgroundTasks,
) -> ModelInfo:
    """Обучение Lasso регрессии"""
    model_type = ModelType(model_type="Lasso")
    unique_model_id = model_id_param.id

    if unique_model_id in MODELS:
        logger.warning("Модель %s уже существует", unique_model_id)
        raise HTTPException(400,
                            detail=f"Модель {unique_model_id} уже существует")

    logger.info("Старт обучения Lasso для модели %s",
                unique_model_id)
    background_tasks.add_task(train_job, params,
                              unique_model_id, model_type)

    return ModelInfo(
        id=unique_model_id,
        model_type=model_type.model_type,
        hyperparameters=params.dict(),
        is_active=False,
        metrics=ModelMetrics(),
        fit_intercept=params.fit_intercept,
    )


@app.post("/predict-one")
async def predict_one(input_data: PredictOneInput) -> Dict[str, float]:
    """Предсказание цены для одного автомобиля"""
    if ACTIVE_MODEL_ID is None:
        logger.error("Попытка предсказания без активной модели")
        raise HTTPException(404, detail="Нет активной модели")

    model_data = MODELS.get(ACTIVE_MODEL_ID)
    if not model_data or "pipeline" not in model_data:
        logger.error("Активная модель %s не найдена", ACTIVE_MODEL_ID)
        raise HTTPException(404, detail="Активная модель не найдена")

    try:
        input_df = pd.DataFrame([input_data.dict()])
        pipeline = model_data["pipeline"]
        prediction = pipeline.predict(input_df)[0]
        logger.info("Успешное предсказание для модели %s", ACTIVE_MODEL_ID)
        return {"prediction": float(prediction)}

    except Exception as e:  # pylint: disable=broad-except
        logger.error("Ошибка предсказания: %s", str(e), exc_info=True)
        raise HTTPException(500,
                            detail=f"Ошибка предсказания: {str(e)}") from e


@app.post("/predict-multiple")
async def predict_multiple(
    file: UploadFile = File(..., description="CSV файл с данными")
) -> Dict[str, List[float]]:
    """Пакетное предсказание цен из CSV файла"""
    if ACTIVE_MODEL_ID is None:
        logger.error("Пакетное предсказание без активной модели")
        raise HTTPException(404, detail="Нет активной модели")

    model_data = MODELS.get(ACTIVE_MODEL_ID)
    if not model_data or "pipeline" not in model_data:
        logger.error("Активная модель %s не найдена", ACTIVE_MODEL_ID)
        raise HTTPException(404, detail="Активная модель не найдена")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        required_columns = set(numerical_cols + categorical_cols)
        missing = required_columns - set(df.columns)
        if missing:
            logger.warning("Отсутствуют колонки: %s", missing)
            raise HTTPException(400, detail=f"Отсутствуют колонки: {missing}")

        pipeline = model_data["pipeline"]
        predictions = pipeline.predict(df).tolist()
        logger.info("Успешно обработано %s записей", len(df))
        return {"predictions": [float(p) for p in predictions]}

    except Exception as e:  # pylint: disable=broad-except
        logger.error("Ошибка пакетного предсказания: %s",
                     str(e), exc_info=True)
        raise HTTPException(500,
                            detail=f"Ошибка предсказания: {str(e)}") from e


@app.get("/models", response_model=List[ModelInfo])
def list_models() -> List[ModelInfo]:
    """Получение списка всех доступных моделей"""
    logger.info("Запрос списка моделей")
    return [ModelInfo(**get_model_info_dict(model_id)) for model_id in MODELS]


@app.post("/set")
def set_active_model(
    unique_model_id: Annotated[str,
                               Query(description="ID модели для активации")]
) -> Dict[str, str]:
    """Установка активной модели"""
    if unique_model_id not in MODELS:
        logger.warning("Попытка установки неизвестной модели: %s",
                       unique_model_id)
        raise HTTPException(404, detail="Модель не найдена")

    global ACTIVE_MODEL_ID  # pylint: disable=global-statement
    ACTIVE_MODEL_ID = unique_model_id
    logger.info("Активная модель установлена: %s", unique_model_id)
    return {"message": f"Модель {unique_model_id} активирована"}


@app.get("/get_dataset")
async def get_dataset() -> List[Dict[str, Any]]:
    """Получение полного набора данных в формате JSON"""
    try:
        logger.info("Запрос на получение датасета")
        return DATASET.to_dict(orient="records")
    except Exception as e:  # pylint: disable=broad-except
        logger.error(
            "Ошибка доступа к датасету: %s",
            str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="Ошибка загрузки данных"
        ) from e


def get_model_info_dict(unique_model_id: str) -> dict:
    """Генерация информации о модели для ответа API

    Args:
        model_id: Идентификатор модели для получения информации

    Returns:
        Словарь с информацией о модели в формате для API
    """
    if unique_model_id not in MODELS:
        logger.error("Модель %s не найдена", unique_model_id)
        raise HTTPException(
            status_code=404,
            detail=f"Модель {unique_model_id} не найдена"
        )

    model_data = MODELS[unique_model_id]
    pipeline = model_data.get("pipeline")
    metrics = model_data.get("metrics", {})

    # Если pipeline не загружен
    if pipeline is None:
        return {
            "id": unique_model_id,
            "model_type": "unknown",
            "hyperparameters": {},
            "is_active": unique_model_id == ACTIVE_MODEL_ID,
            "metrics": metrics,
            "fit_intercept": None,
        }

    try:
        classifier = pipeline.named_steps["classifier"]
        return {
            "id": unique_model_id,
            "model_type": type(classifier).__name__,
            "hyperparameters": classifier.get_params(),
            "is_active": unique_model_id == ACTIVE_MODEL_ID,
            "metrics": metrics,
            "fit_intercept": classifier.get_params().get(
                "fit_intercept", None),
        }
    except KeyError as e:
        logger.warning(
            "Ошибка получения классификатора для модели %s: %s",
            unique_model_id,
            str(e)
        )
        return {
            "id": unique_model_id,
            "model_type": "invalid",
            "hyperparameters": {},
            "is_active": False,
            "metrics": metrics,
            "fit_intercept": None,
        }
