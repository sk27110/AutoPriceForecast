import streamlit as st
import requests
from api.client import train_model


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
            params['alpha'] = st.number_input("alpha", value=1.0)
            params['fit_intercept'] = st.checkbox("fit_intercept", True)
            params['solver'] = st.selectbox(
                "solver",
                ["auto", "svd", "cholesky", "lsqr", "sag"]
            )
            params['tol'] = st.number_input("tol", value=0.0001)
            params['max_iter'] = st.number_input("max_iter", value=500)
    
        elif model_type == "Lasso":
            params['alpha'] = st.number_input("alpha", value=1.0)
            params['fit_intercept'] = st.checkbox("fit_intercept", True)
            params['selection'] = st.selectbox("selection", ["cyclic", "random"])
            params['tol'] = st.number_input("tol", value=0.0001)
            params['max_iter'] = st.number_input("max_iter", value=500)
        
        if st.form_submit_button("Начать обучение"):
            if not model_name.strip():
                st.error("Пожалуйста, укажите название модели")
            else:
                try:
                    result = train_model(model_type, model_name, params)
                    
                    if result["status_code"] == 200:
                        st.success(f"Модель '{model_name}' начала обучение!")
                        st.json(result["data"])
                    else:
                        st.error(f"Ошибка {result['status_code']}: {result['error']}")
                        
                except requests.exceptions.RequestException as e:
                    st.error(f"Ошибка соединения: {str(e)}")
                except KeyError as e:
                    st.error(f"Неизвестный тип модели: {str(e)}")
