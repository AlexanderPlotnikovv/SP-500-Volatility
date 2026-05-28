import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from src.models.base_model import BaseModel


class XGBoost(BaseModel):
    """
    XGBoost model to predict RV.
    """

    def __init__(self):
        super().__init__(name="XGBoost")
        self.model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction.")
        return self.model.predict(X)
