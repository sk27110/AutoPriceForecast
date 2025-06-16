from pydantic import BaseModel, Field
from typing import Optional


class TrainParams(BaseModel):
    """Базовый класс параметров обучения"""


class LinearRegressionTrainParams(TrainParams):
    """Параметры для линейной регрессии"""

    fit_intercept: bool = Field(default=True, description="Использовать intercept")
    n_jobs: Optional[int] = Field(default=None, description="Количество ядер")
    copy_X: bool = Field(default=True, description="Копировать данные")


class RidgeTrainParams(TrainParams):
    """Параметры для Ridge регрессии"""

    alpha: float = Field(default=1.0, description="Коэффициент регуляризации")
    fit_intercept: bool = Field(default=True, description="Использовать intercept")
    max_iter: Optional[int] = Field(default=None, description="Максимум итераций")
    solver: str = Field(default="auto", description="Алгоритм оптимизации")
    tol: float = Field(default=0.0001, description="Точность остановки")


class LassoTrainParams(TrainParams):
    """Параметры для Lasso регрессии"""

    alpha: float = Field(default=1.0, description="Коэффициент регуляризации")
    fit_intercept: bool = Field(default=True, description="Использовать intercept")
    max_iter: Optional[int] = Field(default=1000, description="Максимум итераций")
    selection: str = Field(default="cyclic", description="Метод выбора признаков")
    tol: float = Field(default=0.0001, description="Точность остановки")


class ModelType(BaseModel):
    """Тип модели для обучения"""

    model_type: str = Field(default="LinearRegression", description="Тип модели")


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
    metrics: ModelMetrics = Field(..., description="Метрики модели")
    fit_intercept: Optional[bool] = Field(
        None,
        description="""Используется
                                          ли intercept""",
    )
    is_pretrained: bool = Field(default=False, description="Предобученная модель")


class PretrainedModelInfo(BaseModel):
    """Информация о предобученной модели"""
    
    filename: str = Field(..., description="Имя файла модели")
    model_id: str = Field(..., description="Идентификатор модели")
    file_size: int = Field(..., description="Размер файла в байтах")
    is_loaded: bool = Field(..., description="Загружена ли модель в память")


class LoadPretrainedRequest(BaseModel):
    """Запрос на активацию предобученной модели"""
    
    filename: str = Field(..., description="Имя файла модели для активации")
