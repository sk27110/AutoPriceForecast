import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional


BASE_API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Car Price Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_data_from_api() -> Optional[pd.DataFrame]:
    try:
        response = requests.get(f"{BASE_API_URL}/get_dataset", timeout=10)
        if response.status_code == 200:
            return pd.DataFrame(response.json())
        st.error(f"Ошибка сервера: {response.status_code}")
        return None
    except Exception as e:
        st.error(f"Ошибка соединения: {str(e)}")
        return None

def show_data_analysis(df: pd.DataFrame):
    with st.expander("🔍 Просмотр данных", expanded=True):
        st.dataframe(df.head())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Распределение цен")
        try:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(
                df['price'], 
                bins=30, 
                kde=True, 
                ax=ax, 
                binrange=(0, 10_000_000)
            )
            ax.set_xlim(0, 10_000_000)
            ax.set_xticks(range(0, 11_000_000, 1_000_000))
            ax.set_xticklabels([f"{x//1_000_000}M" for x in ax.get_xticks()])
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Ошибка построения графика цен: {str(e)}")
    
    with col2:
        st.subheader("Корреляция признаков")
        try:
            numeric_cols = ['year', 'mileage', 'engine_capacity', 
                          'engine_power', 'travel_distance', 'price']
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.heatmap(df[numeric_cols].corr(), annot=True, ax=ax)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Ошибка построения матрицы корреляции: {str(e)}")

    st.subheader("Анализ категориальных признаков")
    try:
        categorical_cols = ['transmission', 'body_type',
                          'drive_type', 'color', 'fuel_type']
        categorical = st.selectbox("Выберите признак", categorical_cols)
        fig, ax = plt.subplots(figsize=(10, 4))
        df[categorical].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Ошибка анализа категориальных признаков: {str(e)}")

def main():
    st.sidebar.title("Навигация")
    page = st.sidebar.radio("Выберите раздел", ["Аналитика данных"])

    if page == "Аналитика данных":
        st.title("📊 Анализ данных автомобилей")
        
        with st.spinner("Загрузка данных..."):
            df = load_data_from_api()
        
        if df is not None:
            show_data_analysis(df)

if name == "main":
    main()