import streamlit as st
from service.frontend.api.client import get_models, activate_model


def show_model_management() -> None:
    """Отображает раздел управления моделями."""
    st.title("⚙️ Управление моделями")

    models = get_models()
    if not models:
        return

    active_model = next((m for m in models if m["is_active"]), None)
    if active_model:
        st.success(f"Активная модель: {active_model['id']}")

    for model in models:
        with st.expander(f"{model['model_type']} ({model['id']})"):
            cols = st.columns([1, 3])
            with cols[0]:
                if st.button(f"Активировать {model['id']}"):
                    if activate_model(model["id"]):
                        st.rerun()
                st.metric("R²", model["metrics"]["r2_score"] or "N/A")
                st.metric("MSE", model["metrics"]["mse"] or "N/A")
            with cols[1]:
                st.write("**Параметры:**")
                st.json(model["hyperparameters"])
                st.write("**Метрики:**")
                st.json(model["metrics"])
