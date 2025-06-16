import os
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler

MODEL_STORAGE = Path("service/backend/saved_models")
DATASET_DIR = Path("service/backend/datasets")
MODEL_STORAGE.mkdir(exist_ok=True)
os.makedirs("logs", exist_ok=True)

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
