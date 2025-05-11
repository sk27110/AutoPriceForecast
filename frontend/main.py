import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
import uuid

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
    page = st.sidebar.radio("Выберите раздел", ["Аналитика данных", "Управление моделями", "Обучение моделей"])

    if page == "Аналитика данных":
        st.title("📊 Анализ данных автомобилей")
        
        with st.spinner("Загрузка данных..."):
            df = load_data_from_api()
        
        if df is not None:
            show_data_analysis(df)

    elif page == "Управление моделями":
        st.title("⚙️ Управление моделями")
        
        try:
            response = requests.get(f"{BASE_API_URL}/models")
            models = response.json()
            
            active_model = next((m for m in models if m['is_active']), None)
            if active_model:
                st.success(f"Активная модель: {active_model['id']}")
            
            for model in models:
                with st.expander(f"{model['model_type']} ({model['id']})"):
                    cols = st.columns([1,3])
                    with cols[0]:
                        if st.button(f"Активировать {model['id']}"):
                            requests.post(
                                f"{BASE_API_URL}/set?model_id={model['id']}"
                            )
                            st.rerun()
                        
                        st.metric("R²", model['metrics']['r2_score'] or "N/A")
                        st.metric("MSE", model['metrics']['mse'] or "N/A")
                    
                    with cols[1]:
                        st.write("**Параметры:**")
                        st.json(model['hyperparameters'])
                        
                        st.write("**Метрики:**")
                        st.json(model['metrics'])
        
        except Exception as e:
            st.error(f"Ошибка получения моделей: {str(e)}")


    elif page == "Обучение моделей":
        st.title("🎓 Обучение моделей")
        
        model_type = st.selectbox(
            "Тип модели",
            ["LinearRegression", "Ridge", "Lasso"]
        )
        
        params = {}
        with st.form("model_params"):
            if model_type == "LinearRegression":
                params['fit_intercept'] = st.checkbox("fit_intercept", True)
                params['n_jobs'] = st.number_input("n_jobs", value=-1)
                params['copy_X'] = st.checkbox("copy_X", True)
            
            elif model_type == "Ridge":
                params['alpha'] = st.number_input("alpha", 1.0)
                params['fit_intercept'] = st.checkbox("fit_intercept", True)
                params['solver'] = st.selectbox(
                    "solver", 
                    ["auto", "svd", "cholesky", "lsqr", "sag"]
                )
                params['tol'] = st.number_input("tol", 0.0001)
            
            elif model_type == "Lasso":
                params['alpha'] = st.number_input("alpha", 1.0)
                params['fit_intercept'] = st.checkbox("fit_intercept", True)
                params['selection'] = st.selectbox("selection", ["cyclic", "random"])
                params['tol'] = st.number_input("tol", 0.0001)
            
            model_id = str(uuid.uuid4())[:8]
            submitted = st.form_submit_button("Начать обучение")
            
            if submitted:
                endpoint = {
                    "LinearRegression": "/fit_linearregression",
                    "Ridge": "/fit_ridge",
                    "Lasso": "/fit_lasso"
                }[model_type]
                
                try:
                    response = requests.post(
                        f"{BASE_API_URL}{endpoint}",
                        json={
                            "params": params,
                            "id": {"id": model_id}
                        }
                    )
                    
                    if response.status_code == 200:
                        st.success(f"Модель {model_id} начала обучение!")
                        st.session_state.training_model = model_id
                    else:
                        st.error(f"Ошибка: {response.json()['detail']}")
                
                except Exception as e:
                    st.error(f"Ошибка соединения: {str(e)}")

if __name__ == "__main__":
    main()