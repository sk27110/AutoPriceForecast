import streamlit as st
from api.client import scan_pretrained_models, load_pretrained_model, activate_pretrained_model


def show_pretrained_management() -> None:
    """Отображает раздел управления предобученными моделями."""
    st.title("🔧 Управление предобученными моделями")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔄 Обновить список"):
            st.rerun()
    
    pretrained_models = scan_pretrained_models()
    
    if not pretrained_models:
        st.warning("В директории saved_models не найдено предобученных моделей (.pkl файлов)")
        return
    
    st.success(f"Найдено {len(pretrained_models)} предобученных моделей")
    
    for model in pretrained_models:
        with st.expander(f"📁 {model['filename']} ({model['file_size']} байт)"):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**ID модели:** {model['model_id']}")
                st.write(f"**Размер файла:** {model['file_size']:,} байт")
                st.write(f"**Статус:** {'✅ Загружена' if model['is_loaded'] else '⏳ Не загружена'}")
            
            with col2:
                if st.button(f"📥 Загрузить", key=f"load_{model['filename']}"):
                    with st.spinner("Загрузка модели..."):
                        result = load_pretrained_model(model['filename'])
                        if 'error' in result:
                            st.error(f"Ошибка загрузки: {result['error']}")
                        else:
                            st.success(f"Модель загружена как {result['model_id']}")
                            st.rerun()
            
            with col3:
                if st.button(f"⚡ Активировать", key=f"activate_{model['filename']}"):
                    with st.spinner("Активация модели..."):
                        result = activate_pretrained_model(model['filename'])
                        if 'error' in result:
                            st.error(f"Ошибка активации: {result['error']}")
                        else:
                            st.success(f"Модель {result['model_id']} активирована!")
                            st.rerun()
