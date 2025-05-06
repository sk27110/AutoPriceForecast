from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Annotated, Optional, Dict, List
from uuid import uuid4
from pathlib import Path
import os
import logging
from logging.handlers import RotatingFileHandler
import joblib
import time
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, UploadFile, File
import io
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso



MODEL_STORAGE = Path("saved_models")
DATASET_DIR = Path("datasets")
MODEL_STORAGE.mkdir(exist_ok=True)
os.makedirs("logs", exist_ok=True)


DATASET = pd.read_csv(DATASET_DIR / "processed_data.csv")
X = DATASET.drop('price', axis=1)
y = DATASET['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, shuffle=True)


numerical_cols = ['year', 'mileage', 'engine_capacity', 'engine_power', 'travel_distance']
categorical_cols = ['title', 'transmission', 'body_type', 'drive_type', 'color', 'fuel_type']


class TitleExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if len(X[self.column].str.split())>=2:
          X[self.column] = X[self.column].str.split().str[0] + X[self.column].str.split().str[1]
        if len(X[self.column].str.split())==1:
          X[self.column] = X[self.column].str.split().str[0]
        return X


app = FastAPI(title="Linear Model API")


MODELS: Dict[str, Pipeline] = {}
ACTIVE_MODEL_ID: Optional[str] = None


class TrainParams(BaseModel):
    None


class LinearRegressionTrainParams(TrainParams):
    fit_intercept: bool = Field(default=True, description="Indicates whether to fit the intercept term.")
    n_jobs: Optional[int] = Field(default=None, description="Number of processors to use for computation. If None, all available processors are used.")
    copy_X: bool = Field(default=True, description="If True, the input data X will be copied. If False, they may be modified in place.")


class RidgeTrainParams(TrainParams):
    alpha: float = Field(default=1.0, description="Regularization strength; must be a positive float.")
    fit_intercept: bool = Field(default=True, description="Whether to fit the intercept.")
    max_iter: Optional[int] = Field(default=None, description="Maximum number of iterations for optimization.")
    solver: str = Field(default='auto', description="Solver to use for optimization.")
    tol: float = Field(default=0.0001, description="Tolerance for the optimization; smaller values lead to more precise convergence.")


class LassoTrainParams(TrainParams):
    alpha: float = Field(default=1.0, description="Regularization strength; must be a positive float.")
    fit_intercept: bool = Field(default=True, description="Whether to fit the intercept.")
    max_iter: Optional[int] = Field(default=1000, description="Maximum number of iterations for optimization.")
    selection: str = Field(default='cyclic', description="If 'cyclic', the updates will be done in a cyclic fashion.")
    tol: float = Field(default=0.0001, description="Tolerance for the optimization; smaller values lead to more precise convergence.")



class ModelType(BaseModel):
    model_type: str = Field(default="LinearRegression", description="Type of linear model")




class ModelID(BaseModel):
    id: str = Field("new_model")

class PredictInput(BaseModel):
    data: List[Dict[str, float]]


class ModelInfo(BaseModel):
    id: str
    fit_intercept: bool
    is_active: bool



def build_pipeline(params: TrainParams, model_type: ModelType) -> Pipeline:
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('title_extractor', TitleExtractor(column='title')),
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    column_trans = ColumnTransformer(
        [
            ('cat', categorical_transformer, categorical_cols),
            ('num', numeric_transformer, numerical_cols)
        ],
        remainder='drop'
    )
    model_cls_map = {
        "LinearRegression": LinearRegression,
        "Ridge": Ridge,
        "Lasso": Lasso
    }

    model_cls = model_cls_map[model_type.model_type]
    # Преобразование Pydantic модели в словарь
    model_params = params.model_dump()
    model = model_cls(**model_params)
    
    pipeline = Pipeline(steps=[
        ('preprocessor', column_trans),
        ('classifier', model)
    ])
    return pipeline  # Возвращаем созданный пайплайн


@app.post("/fit_linear", response_model=ModelInfo)
def train_linear_model(params: LinearRegressionTrainParams, id: ModelID, background_tasks: BackgroundTasks):
    model_type = ModelType(model_type="LinearRegression")
    def train_job(params: LinearRegressionTrainParams, model_id: str, model_type):
        start = time.time()
        # try:
        pipeline = build_pipeline(params, model_type=model_type)
        pipeline.fit(X_train, y_train)
        if time.time() - start <= 10:
            MODELS[model_id] = pipeline
            joblib.dump(pipeline, MODEL_STORAGE / f"{model_id}.pkl")
        # except Exception as e:
        #     print("Warning")


    model_id = id.id
    background_tasks.add_task(train_job, params, model_id, model_type)

    return ModelInfo(
        id=model_id,
        fit_intercept=params.fit_intercept,
        is_active=False
    )



@app.post("/fit_ridge", response_model=ModelInfo)
def train_ridge_model(params: RidgeTrainParams, id: ModelID, background_tasks: BackgroundTasks):
    model_type = ModelType(model_type="Ridge")
    def train_job(params: RidgeTrainParams, model_id: str, model_type):
        start = time.time()
        # try:
        pipeline = build_pipeline(params, model_type=model_type)
        pipeline.fit(X_train, y_train)
        if time.time() - start <= 10:
            MODELS[model_id] = pipeline
            joblib.dump(pipeline, MODEL_STORAGE / f"{model_id}.pkl")
        # except Exception as e:
        #     print("Warning")


    model_id = id.id
    background_tasks.add_task(train_job, params, model_id, model_type)

    return ModelInfo(
        id=model_id,
        fit_intercept=params.fit_intercept,
        is_active=False
    )



@app.post("/fit_lasso", response_model=ModelInfo)
def train_lasso_model(params: LassoTrainParams, id: ModelID, background_tasks: BackgroundTasks):
    model_type = ModelType(model_type="Lasso")
    def train_job(params: LassoTrainParams, model_id: str, model_type):
        start = time.time()
        # try:
        pipeline = build_pipeline(params, model_type=model_type)
        pipeline.fit(X_train, y_train)
        MODELS[model_id] = pipeline
        joblib.dump(pipeline, MODEL_STORAGE / f"{model_id}.pkl")
        # except Exception as e:
        #     print("Warning")


    model_id = id.id
    background_tasks.add_task(train_job, params, model_id, model_type)

    return ModelInfo(
        id=model_id,
        fit_intercept=params.fit_intercept,
        is_active=False
    )



@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if ACTIVE_MODEL_ID is None:
        raise HTTPException(status_code=404, detail="No active model")
    model = MODELS[ACTIVE_MODEL_ID]

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")

    required_columns = numerical_cols + categorical_cols
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing columns: {missing}"
        )

    try:
        preds = model.predict(df)
        return {"predictions": preds.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/models", response_model=List[ModelInfo])
def list_models():
    return [
        ModelInfo(
            id=model_id,
            fit_intercept=model.named_steps["classifier"].fit_intercept,
            is_active=(model_id == ACTIVE_MODEL_ID)
        ) for model_id, model in MODELS.items()
    ]


@app.post("/set")
def set_active_model(model_id: Annotated[str, Query()]):
    if model_id not in MODELS:
        raise HTTPException(status_code=404, detail="Model not found")
    global ACTIVE_MODEL_ID
    ACTIVE_MODEL_ID = model_id
    return {"message": f"Model {model_id} set as active"}