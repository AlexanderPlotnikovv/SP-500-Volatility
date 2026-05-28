import pandas as pd
from sklearn.linear_model import LinearRegression

from src.models.base_model import BaseModel


class HarRv(BaseModel):
    """
    HAR-RV (Heterogeneous Autoregressive model of Realized Volatility).
    RV_t = α + β_d * RV_{t-1} + β_w * RV̄_{t-5,t-1} + β_m * RV̄_{t-22,t-1} + ε
    """

    def __init__(self):
        super().__init__(name="HAR-RV")
        self.model = LinearRegression()

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction.")

        return self.model.predict(X)

    def get_coefficients(self) -> dict:
        """Return the coefficients of the model."""
        assert self.is_trained, "Call fit() before."
        return {
            "alpha": self.model.intercept_,
            "beta_d": self.model.coef_[0],
            "beta_w": self.model.coef_[1],
            "beta_m": self.model.coef_[2],
        }
