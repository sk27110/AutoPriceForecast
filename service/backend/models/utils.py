from typing import Dict, Any, Optional
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from service.backend.core.config import categorical_cols, numerical_cols
from service.backend.preprocessing.feature_transformers import TitleExtractor
from service.backend.models.schemas import TrainParams, ModelType

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


def get_model_info_dict(unique_model_id: str, MODELS: Dict[str, Dict[str, Any]], ACTIVE_MODEL_ID: Optional[str]) -> dict:
    """Генерация информации о модели для ответа API

    Args:
        model_id: Идентификатор модели для получения информации

    Returns:
        Словарь с информацией о модели в формате для API
    """
    from fastapi import HTTPException
    from src.server.core.config import logger

    if unique_model_id not in MODELS:
        logger.error("Модель %s не найдена", unique_model_id)
        raise HTTPException(
            status_code=404,
            detail=f"Модель {unique_model_id} не найдена"
        )

    model_data = MODELS[unique_model_id]
    pipeline = model_data.get("pipeline")
    metrics = model_data.get("metrics", {})

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
