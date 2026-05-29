import numpy as np
import pandas as pd

from src.models.base_model import BaseModel


class Baseline(BaseModel):
    """
    Output prediction as previous day RV, i.e RV_t = RV_{t-1}
    """

    def __init__(self):
        super().__init__(name="Baseline")
        self.FEATURE_COLS = ['RV_d']

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.is_trained = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X['RV_d'].values
