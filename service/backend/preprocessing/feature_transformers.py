import pandas as pd
import numpy as np
from typing import Optional
from sklearn.base import BaseEstimator, TransformerMixin


class TitleExtractor(BaseEstimator, TransformerMixin):
    """Преобразователь для объединения первых слов в названии автомобиля"""

    def __init__(self, column: str):
        """Инициализация с указанием имени столбца"""
        self.column = column

    def fit(self, _X: pd.DataFrame, _y: Optional[pd.Series] = None) -> "TitleExtractor":
        """Требуемый метод fit (не выполняет действий)"""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Объединение первых слов в названии автомобиля"""
        X = X.copy()
        split_title = X[self.column].str.split()
        X[self.column] = np.where(
            split_title.str.len() >= 2,
            split_title.str[0] + split_title.str[1],
            split_title.str[0],
        )
        return X
