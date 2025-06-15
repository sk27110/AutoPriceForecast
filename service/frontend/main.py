import streamlit as st
from config import STREAMLIT_CONFIG
from utils.data_loader import load_data_from_api
from service.frontend.pages.data_analysis import show_data_analysis
from service.frontend.pages.model_management import show_model_management
from service.frontend.pages.model_training import show_model_training
from service.frontend.pages.prediction import show_prediction


def main() -> None:
    """Основная функция приложения."""
    st.set_page_config(**STREAMLIT_CONFIG)
    
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
