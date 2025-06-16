import streamlit as st
from service.frontend.api.client import get_models, activate_model


def show_model_management() -> None:
    """Отображает раздел управления моделями."""
    st.title("⚙️ Управление моделями")

    models = get_models()
    if not models:
        st.warning("Нет доступных моделей")
        st.info("Обучите новую модель в разделе 'Обучение моделей' или загрузите предобученную модель в разделе 'Предобученные модели'")
        return

    active_model = next((m for m in models if m["is_active"]), None)
    if active_model:
        model_type_emoji = "🤖" if active_model.get("is_pretrained", False) else "🧠"
        st.success(f"Активная модель: {model_type_emoji} {active_model['id']}")

    pretrained_models = [m for m in models if m.get("is_pretrained", False)]
    trained_models = [m for m in models if not m.get("is_pretrained", False)]

    if pretrained_models:
        st.subheader("🤖 Предобученные модели")
        for model in pretrained_models:
            title = f"🤖 {model['model_type']} ({model['id']})"
            if model["is_active"]:
                title += " ✅"
            
            with st.expander(title):
                cols = st.columns([1, 3])
                with cols[0]:
                    if st.button(f"Активировать {model['id']}", key=f"pretrained_{model['id']}"):
                        if activate_model(model["id"]):
                            st.rerun()
                    st.metric("R²", model["metrics"]["r2_score"] or "N/A")
                    st.metric("MSE", model["metrics"]["mse"] or "N/A")
                    st.write("**Тип:** Предобученная")
                with cols[1]:
                    st.write("**Параметры:**")
                    st.json(model["hyperparameters"])
                    st.write("**Метрики:**")
                    st.json(model["metrics"])

    if trained_models:
        st.subheader("🧠 Обученные модели")
        for model in trained_models:
            title = f"🧠 {model['model_type']} ({model['id']})"
            if model["is_active"]:
                title += " ✅"
            
            with st.expander(title):
                cols = st.columns([1, 3])
                with cols[0]:
                    if st.button(f"Активировать {model['id']}", key=f"trained_{model['id']}"):
                        if activate_model(model["id"]):
                            st.rerun()
                    st.metric("R²", model["metrics"]["r2_score"] or "N/A")
                    st.metric("MSE", model["metrics"]["mse"] or "N/A")
                    st.write("**Тип:** Обученная")
                with cols[1]:
                    st.write("**Параметры:**")
                    st.json(model["hyperparameters"])
                    st.write("**Метрики:**")
                    st.json(model["metrics"])
