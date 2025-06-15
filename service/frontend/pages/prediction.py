import streamlit as st
import pandas as pd
import requests
from service.frontend.api.client import predict_single, predict_multiple


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
                data["engine_capacity"] = st.number_input(
                    "Объем двигателя (л)", 0.0, step=0.1
                )
                data["engine_power"] = st.number_input("Мощность двигателя (л.с.)", 0)

            with col2:
                st.subheader("Дополнительные параметры")
                data["transmission"] = st.selectbox(
                    "Коробка передач", ["механика", "автомат", "робот", "вариатор"]
                )
                data["body_type"] = st.selectbox(
                    "Тип кузова",
                    [
                        "универсал 5 дв.",
                        "внедорожник 5 дв.",
                        "седан",
                        "минивэн",
                        "хэтчбек 5 дв.",
                        "купе",
                        "кабриолет",
                        "внедорожник 3 дв.",
                        "хэтчбек 3 дв.",
                        "пикап двойная кабина",
                        "седан-хардтоп",
                        "тарга",
                        "компактвэн",
                        "лифтбек",
                        "внедорожник открытый",
                        "родстер",
                        "купе-хардтоп",
                        "фургон",
                        "пикап одинарная кабина",
                        "микровэн",
                        "универсал 3 дв.",
                        "седан 2 дв.",
                        "пикап полуторная кабина",
                        "спидстер",
                        "лимузин",
                        "хэтчбек 4 дв.",
                        "универсал",
                        "фастбек",
                    ],
                )
                data["drive_type"] = st.selectbox(
                    "Привод", ["передний", "полный", "задний"]
                )
                data["color"] = st.text_input("Цвет")
                data["fuel_type"] = st.selectbox(
                    "Тип топлива", ["Бензин", "Дизель", "Газ", "Гибрид", "Электро"]
                )
                data["travel_distance"] = 0
                if data["fuel_type"] == "Электро":
                    data["travel_distance"] = st.number_input("Дальность хода (км)", 0)

            if st.form_submit_button("Прогнозировать"):
                if data["fuel_type"] == "Электро" and data["travel_distance"] <= 0:
                    st.error("Для электромобилей укажите дальность хода")
                else:
                    try:
                        prediction = predict_single(data)["prediction"]
                        st.success(f"Прогнозируемая цена: {prediction:,.2f} руб.")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Ошибка при прогнозировании: {str(e)}")
                        if hasattr(e, "response") and e.response.status_code == 422:
                            st.error("Проверьте правильность введенных данных")
    else:
        uploaded_file = st.file_uploader("CSV файл с данными", type="csv")
        if uploaded_file:
            try:
                predictions = predict_multiple(uploaded_file.getvalue())["predictions"]
                df = pd.read_csv(uploaded_file)
                df["Прогноз цены"] = predictions

                st.subheader("Результаты прогнозирования")
                st.dataframe(df.style.format({"Прогноз цены": "{:,.2f}"}))

                csv = df.to_csv(index=False).encode()
                st.download_button(
                    "Скачать результаты", csv, "predictions.csv", "text/csv"
                )
            except requests.exceptions.RequestException as e:
                st.error(f"Ошибка при пакетном прогнозировании: {str(e)}")
