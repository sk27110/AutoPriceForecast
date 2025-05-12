"""
Streamlit-приложение для прогнозирования стоимости автомобилей.
Включает анализ данных, управление моделями и прогнозирование.
Обновлено для совместимости с новым API бэкенда.
"""

from typing import Optional
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Конфигурация страницы
BASE_API_URL = "http://localhost:8000"
st.set_page_config(
    page_title="Car Price Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_data_from_api() -> Optional[pd.DataFrame]:
    """Загружает данные автомобилей из API и возвращает DataFrame."""
    try:
        response = requests.get(f"{BASE_API_URL}/get_dataset", timeout=10)
        response.raise_for_status()
        return pd.DataFrame(response.json())
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка при загрузке данных: {str(e)}")
        return None

def show_data_analysis(df: pd.DataFrame) -> None:
    """Отображает раздел анализа данных."""
    st.title("📊 Анализ данных автомобилей")
    
    with st.expander("🔍 Просмотр данных", expanded=True):
        st.dataframe(df.head())

    # Визуализация распределения цен
    try:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Распределение цен")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(
                df['price'],
                bins=30,
                kde=True,
                ax=ax,
                binrange=(0, 10_000_000))
            ax.set_xlim(0, 10_000_000)
            ax.set_xticks(range(0, 11_000_000, 1_000_000))
            ax.set_xticklabels([f"{x//1_000_000}M" for x in ax.get_xticks()])
            st.pyplot(fig)

        with col2:
            st.subheader("Корреляция признаков")
            numeric_cols = ['year', 'mileage', 'engine_capacity',
                            'engine_power', 'travel_distance', 'price']
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.heatmap(df[numeric_cols].corr(), annot=True, ax=ax)
            st.pyplot(fig)
    except KeyError as e:
        st.error(f"Отсутствует колонка в данных: {str(e)}")
    except Exception as e:
        st.error(f"Ошибка визуализации: {str(e)}")

    try:
        st.subheader("Анализ категориальных признаков")
        categorical_cols = ['transmission', 'body_type',
                           'drive_type', 'color', 'fuel_type']
        categorical = st.selectbox("Выберите признак", categorical_cols)
        fig, ax = plt.subplots(figsize=(10, 4))
        df[categorical].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)
    except KeyError as e:
        st.error(f"Отсутствует категориальная колонка: {str(e)}")
    except Exception as e:
        st.error(f"Ошибка анализа категориальных признаков: {str(e)}")

def show_model_management() -> None:
    """Отображает раздел управления моделями."""
    st.title("⚙️ Управление моделями")
    
    try:
        response = requests.get(f"{BASE_API_URL}/models", timeout=10)
        response.raise_for_status()
        models = response.json()
        
        active_model = next((m for m in models if m['is_active']), None)
        if active_model:
            st.success(f"Активная модель: {active_model['id']}")
            
        for model in models:
            with st.expander(f"{model['model_type']} ({model['id']})"):
                cols = st.columns([1, 3])
                with cols[0]:
                    if st.button(f"Активировать {model['id']}"):
                        response = requests.post(
                            f"{BASE_API_URL}/set?unique_model_id={model['id']}"
                        )
                        if response.status_code == 200:
                            st.rerun()
                    st.metric("R²", model['metrics']['r2_score'] or "N/A")
                    st.metric("MSE", model['metrics']['mse'] or "N/A")
                with cols[1]:
                    st.write("**Параметры:**")
                    st.json(model['hyperparameters'])
                    st.write("**Метрики:**")
                    st.json(model['metrics'])
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка при получении моделей: {str(e)}")

def show_model_training() -> None:
    """Отображает раздел обучения моделей."""
    st.title("🎓 Обучение моделей")
    
    model_type = st.selectbox(
        "Тип модели",
        ["LinearRegression", "Ridge", "Lasso"]
    )
    
    with st.form("model_params"):
        model_name = st.text_input(
            "Название модели",
            help="Укажите уникальное имя для вашей модели",
            max_chars=30
        )
        
        params = {}
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
        
        if st.form_submit_button("Начать обучение"):
            if not model_name.strip():
                st.error("Пожалуйста, укажите название модели")
            else:
                try:
                    endpoint = {
                        "LinearRegression": "/fit_linearregression",
                        "Ridge": "/fit_ridge",
                        "Lasso": "/fit_lasso"
                    }[model_type]
                    
                    response = requests.post(
                        f"{BASE_API_URL}{endpoint}",
                        json={"params": params, "model_id_param": model_name.strip()},
                        timeout=30
                    )
                    response.raise_for_status()
                    st.success(f"Модель '{model_name}' начала обучение!")
                except requests.exceptions.RequestException as e:
                    st.error(f"Ошибка при обучении модели: {str(e)}")

def show_prediction() -> None:
    """Отображает раздел прогнозирования."""
    st.title("🔮 Прогнозирование цены")
    prediction_type = st.radio("Тип прогноза", ["Единичный", "Пакетный"])
    
    if prediction_type == "Единичный":
        with st.form("single_prediction"):
            col1, col2 = st.columns(2)
            data = {}
            with col1:
                st.subheader("Характеристики автомобиля")
                data["title"] = st.text_input("Марка")
                data["year"] = st.number_input("Год выпуска", 1900, 2023)
                data["mileage"] = st.number_input("Пробег (км)", 0)
                data["engine_volume"] = st.number_input("Объем двигателя (л)", 0.0, step=0.1)
                data["engine_horsepower"] = st.number_input("Мощность двигателя (л.с.)", 0)
            
            with col2:
                st.subheader("Дополнительные параметры")
                data["transmission"] = st.selectbox(
                    "Коробка передач", 
                    ["механика", "автомат", "робот", "вариатор"]
                )
                data["body_type"] = st.selectbox("Тип кузова", [
                    "универсал 5 дв.", "внедорожник 5 дв.", "седан", "минивэн",
                    "хэтчбек 5 дв.", "купе", "кабриолет", "внедорожник 3 дв.",
                    "хэтчбек 3 дв.", "пикап двойная кабина", "седан-хардтоп", "тарга",
                    "компактвэн", "лифтбек", "внедорожник открытый", "родстер",
                    "купе-хардтоп", "фургон", "пикап одинарная кабина", "микровэн",
                    "универсал 3 дв.", "седан 2 дв.", "пикап полуторная кабина",
                    "спидстер", "лимузин", "хэтчбек 4 дв.", "универсал", "фастбек"
                ])
                data["drive_type"] = st.selectbox("Привод", ["передний", "полный", "задний"])
                data["color"] = st.text_input("Цвет")
                data["fuel_type"] = st.selectbox(
                    "Тип топлива", 
                    ["Бензин", "Дизель", "Газ", "Гибрид", "Электро"]
                )
                data["travel_distance"] = 0
                if data["fuel_type"] == "Электро":
                    data["travel_distance"] = st.number_input("Дальность хода (км)", 0)
            
            if st.form_submit_button("Прогнозировать"):
                if data["fuel_type"] == "Электро" and data["travel_distance"] <= 0:
                    st.error("Для электромобилей укажите дальность хода")
                else:
                    try:
                        response = requests.post(
                            f"{BASE_API_URL}/predict-one",
                            json=data,
                            timeout=10
                        )
                        response.raise_for_status()
                        prediction = response.json()['prediction']
                        st.success(f"Прогнозируемая цена: {prediction:,.2f} руб.")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Ошибка при прогнозировании: {str(e)}")
    
    else:
        uploaded_file = st.file_uploader("CSV файл с данными", type="csv")
        if uploaded_file:
            try:
                response = requests.post(
                    f"{BASE_API_URL}/predict-multiple",
                    files={"file": uploaded_file.getvalue()},
                    timeout=30
                )
                response.raise_for_status()
                
                predictions = response.json()["predictions"]
                df = pd.read_csv(uploaded_file)
                df["Прогноз цены"] = predictions
                
                st.subheader("Результаты прогнозирования")
                st.dataframe(df.style.format({"Прогноз цены": "{:,.2f}"}))
                
                csv = df.to_csv(index=False).encode()
                st.download_button(
                    "Скачать результаты",
                    csv,
                    "predictions.csv",
                    "text/csv"
                )
            except requests.exceptions.RequestException as e:
                st.error(f"Ошибка при пакетном прогнозировании: {str(e)}")

def main() -> None:
    """Основная функция приложения."""
    st.sidebar.title("Навигация")
    page = st.sidebar.radio("Выберите раздел", [
        "Аналитика данных", 
        "Управление моделями",
        "Обучение моделей", 
        "Прогнозирование"
    ])

    if page == "Аналитика данных":
        df = load_data_from_api()
        if df is not None:
            show_data_analysis(df)
    
    elif page == "Управление моделями":
        show_model_management()
    
    elif page == "Обучение моделей":
        show_model_training()
    
    elif page == "Прогнозирование":
        show_prediction()

if __name__ == "__main__":
    main()