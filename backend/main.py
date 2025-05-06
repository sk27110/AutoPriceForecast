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


df = X_train[:1000]
df.to_csv('dataset.csv', index=False)


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
    id: str
    fit_intercept: bool = Field(default=True)
    normalize: bool = Field(default=False) 
    n_jobs: Optional[int] = Field(default=None)


class PredictInput(BaseModel):
    data: List[Dict[str, float]]


class ModelInfo(BaseModel):
    id: str
    fit_intercept: bool
    normalize: bool
    is_active: bool



def build_pipeline(params: TrainParams) -> Pipeline:
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

    model = LinearRegression(
        fit_intercept=params.fit_intercept,
        n_jobs=params.n_jobs
    )

    pipeline = Pipeline(steps=[('preprocessor', column_trans),
                      ('classifier', model)])
    return pipeline


@app.post("/fit", response_model=ModelInfo)
def train_model(params: TrainParams, background_tasks: BackgroundTasks):
    def train_job(params: TrainParams, model_id: str):
        start = time.time()
        try:
            pipeline = build_pipeline(params)
            print("true")
            pipeline.fit(X_train, y_train)
            y_train_pred = pipeline.predict(X_train)
            print('r2_score на трейне', r2_score(y_train, y_train_pred))
            y_test_pred = pipeline.predict(X_test)
            print('r2_score на трейне', r2_score(y_test, y_test_pred))
            if time.time() - start <= 10:
                MODELS[model_id] = pipeline
                joblib.dump(pipeline, MODEL_STORAGE / f"{model_id}.pkl")
        except Exception as e:
            print("Warning")


    model_id = params.id
    background_tasks.add_task(train_job, params, model_id)

    return ModelInfo(
        id=model_id,
        fit_intercept=params.fit_intercept,
        normalize=params.normalize,
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
        print(True)
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
            normalize=False,
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