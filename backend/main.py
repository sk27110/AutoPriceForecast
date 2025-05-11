from fastapi import (FastAPI, HTTPException, BackgroundTasks, Query,
                     UploadFile, File)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Annotated, Optional, Dict, List, Tuple, Any
from uuid import uuid4
from pathlib import Path
import os
import logging
from logging.handlers import RotatingFileHandler
import joblib
import time
import pandas as pd
import io
from concurrent.futures import ThreadPoolExecutor, TimeoutError

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
DATASET = pd.read_csv(DATASET_DIR / "processed_data.csv")
X = DATASET.drop('price', axis=1)
y = DATASET['price']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, shuffle=True
)

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Configure logging
formatter = logging.Formatter(
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger("app")
logger.setLevel(logging.INFO)

file_handler = RotatingFileHandler(
    filename=LOG_DIR / 'app.log',
    maxBytes=10*1024*1024,
    backupCount=5,
    encoding='utf-8'
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

numerical_cols = ['year', 'mileage', 'engine_capacity',
                  'engine_power', 'travel_distance']
categorical_cols = ['title', 'transmission', 'body_type',
                    'drive_type', 'color', 'fuel_type']


class TitleExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if len(X[self.column].str.split()) >= 2:
            X[self.column] = (X[self.column].str.split().str[0] +
                              X[self.column].str.split().str[1])

        if len(X[self.column].str.split()) == 1:
            X[self.column] = X[self.column].str.split().str[0]
        return X


app = FastAPI(title="Linear Model API")

MODELS: Dict[str, Dict[str, Any]] = {}
ACTIVE_MODEL_ID: Optional[str] = None

# Load existing models at startup
for model_file in MODEL_STORAGE.glob("*.pkl"):
    model_id = model_file.stem
    try:
        model_data = joblib.load(model_file)
        MODELS[model_id] = model_data
        logger.info(f"Successfully loaded model: {model_id}")
    except Exception as e:
        logger.error(f"Error loading model {model_id}: {str(e)}", exc_info=True)


class TrainParams(BaseModel):
    pass


class LinearRegressionTrainParams(TrainParams):
    fit_intercept: bool = Field(default=True)
    n_jobs: Optional[int] = Field(default=None)
    copy_X: bool = Field(default=True)


class RidgeTrainParams(TrainParams):
    alpha: float = Field(default=1.0)
    fit_intercept: bool = Field(default=True)
    max_iter: Optional[int] = Field(default=None)
    solver: str = Field(default='auto')
    tol: float = Field(default=0.0001)


class LassoTrainParams(TrainParams):
    alpha: float = Field(default=1.0)
    fit_intercept: bool = Field(default=True)
    max_iter: Optional[int] = Field(default=1000)
    selection: str = Field(default='cyclic')
    tol: float = Field(default=0.0001)


class ModelType(BaseModel):
    model_type: str = Field(default="LinearRegression")


class ModelID(BaseModel):
    id: str = Field(default="new_model")


class PredictOneInput(BaseModel):
    title: str
    year: int
    mileage: float
    transmission: str
    body_type: str
    drive_type: str
    color: str
    engine_capacity: float
    engine_power: float
    fuel_type: str
    travel_distance: float


class ModelMetrics(BaseModel):
    r2_score: Optional[float] = None
    mse: Optional[float] = None
    train_time: Optional[float] = None


class ModelInfo(BaseModel):
    id: str
    model_type: str
    hyperparameters: dict
    is_active: bool
    metrics: ModelMetrics
    fit_intercept: Optional[bool] = None


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
    model_params = params.model_dump()
    model = model_cls(**model_params)
    pipeline = Pipeline(steps=[
        ('preprocessor', column_trans),
        ('classifier', model)
    ])
    return pipeline


def train_job(params: TrainParams, model_id: str, model_type):
    pipeline = None
    metrics = ModelMetrics().dict()
    start_time = time.time()

    try:
        logger.info(f"Starting training for model {model_id} ({model_type.model_type})")
        pipeline = build_pipeline(params, model_type=model_type)
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(pipeline.fit, X_train, y_train)
            try:
                future.result(timeout=10)
                train_time = time.time() - start_time
                y_pred = pipeline.predict(X_test)
                metrics = {
                    'r2_score': r2_score(y_test, y_pred),
                    'mse': mean_squared_error(y_test, y_pred),
                    'train_time': float(train_time),
                }
                logger.info(f"Model {model_id} trained successfully. Metrics: {metrics}")
            except TimeoutError:
                logger.warning(f"Training interrupted for model {model_id} (timeout)")
    except Exception as e:
        logger.error(f"Training failed for model {model_id}: {str(e)}", exc_info=True)
    finally:
        if pipeline is not None:
            MODELS[model_id] = {
                'pipeline': pipeline,
                'metrics': metrics
            }
            try:
                joblib.dump(MODELS[model_id], MODEL_STORAGE / f"{model_id}.pkl")
                logger.info(f"Model {model_id} saved successfully")
            except Exception as e:
                logger.error(f"Failed to save model {model_id}: {str(e)}", exc_info=True)


@app.post("/fit_linearregression", response_model=ModelInfo)
def train_linear_model(
    params: LinearRegressionTrainParams,
    id: ModelID,
    background_tasks: BackgroundTasks
):
    model_type = ModelType(model_type="LinearRegression")
    model_id = id.id

    if model_id in MODELS:
        logger.warning(f"Model creation conflict: {model_id} already exists")
        raise HTTPException(status_code=400, detail=f"Model '{model_id}' already exists")
    
    logger.info(f"Starting LinearRegression training for model {model_id}")
    background_tasks.add_task(train_job, params, model_id, model_type)
    
    return ModelInfo(
        id=model_id,
        model_type=model_type.model_type,
        hyperparameters=params.dict(),
        is_active=False,
        metrics=ModelMetrics(),
        fit_intercept=params.fit_intercept
    )


@app.post("/fit_ridge", response_model=ModelInfo)
def train_ridge_model(
    params: RidgeTrainParams,
    id: ModelID,
    background_tasks: BackgroundTasks
):
    model_type = ModelType(model_type="Ridge")
    model_id = id.id

    if model_id in MODELS:
        logger.warning(f"Model creation conflict: {model_id} already exists")
        raise HTTPException(status_code=400, detail=f"Model '{model_id}' already exists")

    logger.info(f"Starting Ridge training for model {model_id}")
    background_tasks.add_task(train_job, params, model_id, model_type)

    return ModelInfo(
        id=model_id,
        model_type=model_type.model_type,
        hyperparameters=params.dict(),
        is_active=False,
        metrics=ModelMetrics(),
        fit_intercept=params.fit_intercept
    )


@app.post("/fit_lasso", response_model=ModelInfo)
def train_lasso_model(
    params: LassoTrainParams,
    id: ModelID,
    background_tasks: BackgroundTasks
):
    model_type = ModelType(model_type="Lasso")
    model_id = id.id

    if model_id in MODELS:
        logger.warning(f"Model creation conflict: {model_id} already exists")
        raise HTTPException(status_code=400, detail=f"Model '{model_id}' already exists")
    
    logger.info(f"Starting Lasso training for model {model_id}")
    background_tasks.add_task(train_job, params, model_id, model_type)

    return ModelInfo(
        id=model_id,
        model_type=model_type.model_type,
        hyperparameters=params.dict(),
        is_active=False,
        metrics=ModelMetrics(),
        fit_intercept=params.fit_intercept
    )


@app.post("/predict-one")
async def predict_one(input_data: PredictOneInput):
    if ACTIVE_MODEL_ID is None:
        logger.error("Prediction attempt with no active model")
        raise HTTPException(status_code=404, detail="No active model")
    
    model_data = MODELS.get(ACTIVE_MODEL_ID)
    if model_data is None or 'pipeline' not in model_data:
        logger.error(f"Active model {ACTIVE_MODEL_ID} not found")
        raise HTTPException(status_code=404, detail="Active model not found")

    try:
        input_dict = input_data.dict()
        df = pd.DataFrame([input_dict])
        logger.info(f"Prediction request for model {ACTIVE_MODEL_ID}")
        
        pipeline = model_data['pipeline']
        preds = pipeline.predict(df)
        
        logger.info(f"Prediction successful for model {ACTIVE_MODEL_ID}")
        return {"prediction": float(preds[0])}
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict-multiple")
async def predict_multiple(file: UploadFile = File(...)):
    if ACTIVE_MODEL_ID is None:
        logger.error("Batch prediction attempt with no active model")
        raise HTTPException(status_code=404, detail="No active model")
    
    model_data = MODELS.get(ACTIVE_MODEL_ID)
    if model_data is None or 'pipeline' not in model_data:
        logger.error(f"Active model {ACTIVE_MODEL_ID} not found")
        raise HTTPException(status_code=404, detail="Active model not found")

    try:
        logger.info(f"Batch prediction request for model {ACTIVE_MODEL_ID}")
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        required_columns = numerical_cols + categorical_cols
        missing = set(required_columns) - set(df.columns)
        if missing:
            logger.warning(f"Missing columns in batch prediction: {missing}")
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")
        
        pipeline = model_data['pipeline']
        predictions = pipeline.predict(df)
        
        logger.info(f"Batch prediction completed for {len(df)} records")
        return {"predictions": [float(p) for p in predictions]}
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/models", response_model=List[ModelInfo])
def list_models():
    logger.info("Listing all available models")
    return [ModelInfo(**get_model_info_dict(model_id)) for model_id in MODELS.keys()]


@app.post("/set")
def set_active_model(model_id: Annotated[str, Query()]):
    if model_id not in MODELS:
        logger.warning(f"Attempt to set unknown model: {model_id}")
        raise HTTPException(status_code=404, detail="Model not found")
    
    global ACTIVE_MODEL_ID
    ACTIVE_MODEL_ID = model_id
    logger.info(f"Active model set to: {model_id}")
    return {"message": f"Model {model_id} set as active"}


@app.get("/get_dataset")
async def get_dataset() -> List[Dict[str, Any]]:
    try:
        logger.info("Dataset download request")
        return DATASET.to_dict(orient='records')
    except Exception as e:
        logger.error(f"Dataset access failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def get_model_info_dict(model_id: str) -> dict:
    if model_id not in MODELS:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_data = MODELS[model_id]
    pipeline = model_data.get('pipeline')
    metrics = model_data.get('metrics', {})
    
    if pipeline is None:
        return {
            "id": model_id,
            "model_type": "unknown",
            "hyperparameters": {},
            "is_active": model_id == ACTIVE_MODEL_ID,
            "metrics": metrics,
            "fit_intercept": None
        }
    
    classifier = pipeline.named_steps["classifier"]
    return {
        "id": model_id,
        "model_type": type(classifier).__name__,
        "hyperparameters": classifier.get_params(),
        "is_active": model_id == ACTIVE_MODEL_ID,
        "metrics": metrics,
        "fit_intercept": classifier.get_params().get('fit_intercept', None)
    }
