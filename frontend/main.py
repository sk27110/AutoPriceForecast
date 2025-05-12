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


def main():
    st.sidebar.title("Навигация")
    page = st.sidebar.radio("Выберите раздел", ["Аналитика данных", "Управление моделями",
                                                "Обучение моделей", "Прогнозирование"])
    if page == "Аналитика данных":
        st.title("📊 Анализ данных автомобилей")
        with st.spinner("Загрузка данных..."):
            df = load_data_from_api()
        if df is not None:
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
                    cols = st.columns([1, 3])
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
            # Поле для ввода названия модели
            model_name = st.text_input(
                "Название модели",
                help="Укажите уникальное имя для вашей модели",
                max_chars=30
            )
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
            submitted = st.form_submit_button("Начать обучение")
            if submitted:
                if not model_name.strip():
                    st.error("Пожалуйста, укажите название модели")
                else:
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
                                "id": {"id": model_name.strip()}
                            }
                        )
                        if response.status_code == 200:
                            st.success(f"Модель '{model_name}' начала обучение!")
                            st.session_state.training_model = model_name
                        else:
                            st.error(f"Ошибка: {response.json()['detail']}")
                    except Exception as e:
                        st.error(f"Ошибка соединения: {str(e)}")

    elif page == "Прогнозирование":
        st.title("🔮 Прогнозирование цены")
        prediction_type = st.radio("Тип прогноза", ["Единичный", "Пакетный"])
        if prediction_type == "Единичный":
            with st.form("single_prediction"):
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Характеристики автомобиля")
                    title = st.text_input("Марка")
                    year = st.number_input("Год выпуска", min_value=1900, max_value=2023)
                    mileage = st.number_input("Пробег (км)", min_value=0)
                    engine_capacity = st.number_input("Объем двигателя (л)", min_value=0.0, step=0.1)
                    engine_power = st.number_input("Мощность двигателя (л.с.)", min_value=0)
                with col2:
                    st.subheader("Дополнительные параметры")
                    transmission = st.selectbox("Коробка передач", ["механика", "автомат", "робот", "вариатор"])
                    body_type = st.selectbox("Тип кузова", [
                        "универсал 5 дв.", "внедорожник 5 дв.", "седан", "минивэн",
                        "хэтчбек 5 дв.", "купе", "кабриолет", "внедорожник 3 дв.",
                        "хэтчбек 3 дв.", "пикап двойная кабина", "седан-хардтоп", "тарга",
                        "компактвэн", "лифтбек", "внедорожник открытый", "родстер",
                        "купе-хардтоп", "фургон", "пикап одинарная кабина", "микровэн",
                        "универсал 3 дв.", "седан 2 дв.", "пикап полуторная кабина",
                        "спидстер", "лимузин", "хэтчбек 4 дв.", "универсал", "фастбек"
                    ])
                    drive_type = st.selectbox("Привод", ["передний", "полный", "задний"])
                    color = st.text_input("Цвет")
                    fuel_type = st.selectbox("Тип топлива", ["Бензин", "Дизель", "Газ", "Гибрид", "Электро"])
                    travel_distance = 0
                    if fuel_type == "Электро":
                        travel_distance = st.number_input("Дальность хода (км)", min_value=0)
                if st.form_submit_button("Прогнозировать"):
                    error = False
                    if fuel_type == "Электро" and travel_distance <= 0:
                        st.error("Для электромобилей укажите дальность хода")
                        error = True
                    if not error:
                        data = {
                            "title": title,
                            "year": year,
                            "mileage": mileage,
                            "transmission": transmission,
                            "body_type": body_type,
                            "drive_type": drive_type,
                            "color": color,
                            "engine_capacity": engine_capacity,
                            "engine_power": engine_power,
                            "fuel_type": fuel_type,
                            "travel_distance": travel_distance
                        }
                        try:
                            response = requests.post(
                                f"{BASE_API_URL}/predict-one",
                                json=data
                            )
                            if response.status_code == 200:
                                prediction = response.json()['prediction']
                                st.success(f"Прогнозируемая цена: {prediction:,.2f} руб.")
                            else:
                                st.error(f"Ошибка прогнозирования: {response.text}")
                        except Exception as e:
                            st.error(f"Ошибка соединения: {str(e)}")
        else:
            uploaded_file = st.file_uploader("CSV файл с данными", type="csv")
            if uploaded_file:
                try:
                    response = requests.post(
                        f"{BASE_API_URL}/predict-multiple",
                        files={"file": uploaded_file.getvalue()}
                    )
                    if response.status_code == 200:
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
                    else:
                        st.error("Ошибка обработки файла")
                except Exception as e:
                    st.error(f"Ошибка: {str(e)}")

if __name__ == "__main__":
    main()