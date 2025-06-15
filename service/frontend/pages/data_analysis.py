import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def show_data_analysis(df: pd.DataFrame) -> None:
    """Отображает раздел анализа данных."""
    st.title("📊 Анализ данных автомобилей")
    
    with st.expander("🔍 Просмотр данных", expanded=True):
        st.dataframe(df.head())

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
