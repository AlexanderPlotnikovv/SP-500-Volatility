from abc import ABC, abstractmethod
import json
import numpy as np
import pandas as pd
from pathlib import Path
import config


class BaseModel(ABC):
    """
    Abstract base class for all models implemented in project.
    """

    def __init__(self, name: str):
        self.name = name
        self.is_trained = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit the model to the training data.

        Args:
            X (DataFrame): Training features.
            y (Series): Training target variable.
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict using the trained model.
        Args:
            X (DataFrame): Test features.
        Returns:
            DataFrame: Predictions.
        """
        pass

    def window_expand_fit(self, X: pd.DataFrame, y: pd.Series, window_size: int = config.MIN_TRAIN_SIZE) -> pd.Series:
        """
        Fit the model using a window expansion approach.

        Args:
            X (DataFrame): Training features.
            y (Series): Training target variable.
            window_size (int): Minimal size of training window.
        Returns:
            Series: Predictions for each time step.
        """
        predictions = np.full(len(X), np.nan)

        for time in range(window_size, len(X)):
            X_train = X.iloc[:time]
            y_train = y.iloc[:time]
            X_test = X.iloc[[time]]

            self.fit(X_train, y_train)

            y_pred = self.predict(X_test)[0]
            predictions[time] = y_pred

            if time % 500 == 0:
                print(f"[{self.name}] walk-forward: {time}/{len(X)}")

        return pd.Series(predictions, index=y.index, name=f"{self.name}_pred")

    def save_predictions(self, dates: pd.DatetimeIndex, predictions: np.ndarray, values: np.ndarray,
                         metrics: dict) -> dict:
        """
        Save the predictions in outputs/predictions/{name}.json.
        Args:
            dates (DatetimeIndex): Dates corresponding to predictions.
            predictions (ndarray): Predicted values.
            values (ndarray): Actual values.
            metrics (dict): Evaluation metrics to be saved.
        Return:
            Dictionary of metrics saved for this model.
        """

        result = {
            "model": self.name,
            "dates": dates,
            "predictions": predictions,
            "values": values,
            "metrics": metrics
        }

        output_path = Path(config.OUTPUTS_DIR) / f"{self.name}.json"
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"[{self.name}] Saved: {output_path}")
