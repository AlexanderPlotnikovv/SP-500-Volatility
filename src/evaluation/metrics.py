import numpy as np
import pandas as pd


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate mean squared error
    Args:
        y_true: true values
        y_pred: predicted values
    Returns:
        Mean squared error
    """
    return float(np.mean((y_true - y_pred) ** 2))


def qlike(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate QLIKE loss
    Args:
        y_true: true values
        y_pred: predicted values
    Returns:
        QLIKE loss
    """
    mask = (y_pred > 0) & (y_true > 0)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    return float(np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1))


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """
    Compute evaluation metrics for volatility forecasting.
    Args:
        y_true: true values
        y_pred: predicted values
    Returns:
        Dictionary with MSE and QLIKE metrics.
    """
    mask = ~np.isnan(y_pred)
    y_true = np.array(y_true)[mask]
    y_pred = np.array(y_pred)[mask]

    return {
        "MSE": mse(y_true, y_pred),
        "QLIKE": qlike(y_true, y_pred),
        "predictions_count": int(np.sum(mask))
    }
