from typing import Optional
import streamlit as st
import requests
import pandas as pd
from config import BASE_API_URL


def load_data_from_api() -> Optional[pd.DataFrame]:
    """Загружает данные автомобилей из API и возвращает DataFrame."""
    try:
        response = requests.get(f"{BASE_API_URL}/get_dataset", timeout=10)
        response.raise_for_status()
        return pd.DataFrame(response.json())
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка при загрузке данных: {str(e)}")
        return None
