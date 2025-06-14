from typing import List, Dict, Any, Optional, Annotated
import pandas as pd
import io
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, UploadFile, File
from fastapi.responses import JSONResponse

from src.server.models.schemas import (
    LinearRegressionTrainParams,
    RidgeTrainParams,
    LassoTrainParams,
    ModelType,
    ModelID,
    ModelInfo,
    ModelMetrics,
    PredictOneInput
)
from src.server.core.config import logger, numerical_cols, categorical_cols
from src.server.models.utils import get_model_info_dict

router = APIRouter()

MODELS: Dict[str, Dict[str, Any]] = {}
ACTIVE_MODEL_ID: Optional[str] = None
DATASET = None

def initialize(models_dict, active_model, dataset):
    global MODELS, ACTIVE_MODEL_ID, DATASET
    MODELS = models_dict
    ACTIVE_MODEL_ID = active_model
    DATASET = dataset


@router.post("/fit_linearregression", response_model=ModelInfo)
def train_linear_model(
    params: LinearRegressionTrainParams,
    model_id_param: ModelID,
    background_tasks: BackgroundTasks,
) -> ModelInfo:
    """Обучение модели линейной регрессии"""
    from src.server.models.training import train_job
    
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


@router.post("/fit_ridge", response_model=ModelInfo)
def train_ridge_model(
    params: RidgeTrainParams,
    model_id_param: ModelID,
    background_tasks: BackgroundTasks,
) -> ModelInfo:
    """Обучение Ridge регрессии"""
    from src.server.models.training import train_job
    
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


@router.post("/fit_lasso", response_model=ModelInfo)
def train_lasso_model(
    params: LassoTrainParams,
    model_id_param: ModelID,
    background_tasks: BackgroundTasks,
) -> ModelInfo:
    """Обучение Lasso регрессии"""
    from src.server.models.training import train_job
    
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


@router.post("/predict-one")
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


@router.post("/predict-multiple")
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


@router.get("/models", response_model=List[ModelInfo])
def list_models() -> List[ModelInfo]:
    """Получение списка всех доступных моделей"""
    logger.info("Запрос списка моделей")
    return [ModelInfo(**get_model_info_dict(model_id, MODELS, ACTIVE_MODEL_ID)) for model_id in MODELS]


@router.post("/set")
# def set_active_model(
#     unique_model_id: str = Query(..., description="ID модели для активации")
# ) -> Dict[str, str]:
#     """Установка активной модели"""
#     global ACTIVE_MODEL_ID  # pylint: disable=global-statement
    
#     if unique_model_id not in MODELS:
#         logger.warning("Попытка установки неизвестной модели: %s",
#                        unique_model_id)
#         raise HTTPException(404, detail="Модель не найдена")

#     ACTIVE_MODEL_ID = unique_model_id
#     logger.info("Активная модель установлена: %s", unique_model_id)
#     return {"message": f"Модель {unique_model_id} активирована"}

@router.post("/set")
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



@router.get("/get_dataset")
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
