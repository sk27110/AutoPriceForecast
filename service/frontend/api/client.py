import requests
import streamlit as st
from config import BASE_API_URL


def get_models():
    """Получает список моделей из API."""
    try:
        response = requests.get(f"{BASE_API_URL}/models", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка при получении моделей: {str(e)}")
        return []


def activate_model(model_id: str) -> bool:
    """Активирует модель по ID."""
    try:
        response = requests.post(f"{BASE_API_URL}/set?unique_model_id={model_id}")
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def train_model(model_type: str, model_name: str, params: dict) -> dict:
    """Отправляет запрос на обучение модели."""
    endpoint = {
        "LinearRegression": "/fit_linearregression",
        "Ridge": "/fit_ridge",
        "Lasso": "/fit_lasso",
    }[model_type]

    request_body = {"params": params, "model_id_param": {"id": model_name.strip()}}

    response = requests.post(f"{BASE_API_URL}{endpoint}", json=request_body, timeout=30)

    return {
        "status_code": response.status_code,
        "data": response.json() if response.status_code == 200 else None,
        "error": response.text if response.status_code != 200 else None,
    }


def predict_single(data: dict) -> dict:
    """Выполняет единичное прогнозирование."""
    response = requests.post(f"{BASE_API_URL}/predict-one", json=data, timeout=10)
    response.raise_for_status()
    return response.json()


def predict_multiple(file_data) -> dict:
    """Выполняет пакетное прогнозирование."""
    response = requests.post(
        f"{BASE_API_URL}/predict-multiple", files={"file": file_data}, timeout=30
    )
    response.raise_for_status()
    return response.json()


def scan_pretrained_models():
    """Сканирует предобученные модели."""
    try:
        response = requests.get(f"{BASE_API_URL}/pretrained/scan", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка при сканировании предобученных моделей: {str(e)}")
        return []


def load_pretrained_model(filename: str) -> dict:
    """Загружает предобученную модель."""
    try:
        response = requests.post(
            f"{BASE_API_URL}/pretrained/load",
            json={"filename": filename},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def activate_pretrained_model(filename: str) -> dict:
    """Активирует предобученную модель."""
    try:
        response = requests.post(
            f"{BASE_API_URL}/pretrained/activate",
            json={"filename": filename},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
