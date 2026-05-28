import pandas as pd
import numpy as np
import config


def compute_log_returns(df: pd.DataFrame) -> pd.Series:
    """
    Compute log returns from a dataframe.

        Args:
            df (dataframe): Dataframe containing stock data with 'Close' price.
        Returns:
            Series of log returns.
    """
    return np.log(df.Close / df.Close.shift(1))


def compute_rv(returns: pd.Series) -> pd.Series:
    """
    Compute realized volatility (RV) from log returns.
        Args:
            returns (Series): Series of log returns.
        Returns:
            Series of realized volatility.
    """
    return returns ** 2


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features from a S&P500 dataframe.
    Features:
        RV_d - Realized volatility previous day
        RV_w - Realized volatility for 5 days
        RV_md - Realized volatility for 22 days
    Target:
        y - Realized volatility of current day

    Args:
        df (dataframe): Dataframe containing S&P500 data with 'Close' price.
    Returns:
        Dataframe with features and target.
    """

    returns = compute_log_returns(df)
    rv = compute_rv(returns)

    features = pd.DataFrame(index=df.index, columns=['RV_d', 'RV_w', 'RV_m', 'y'])
    features['RV_d'] = rv.shift(1)
    features["RV_w"] = rv.shift(1).rolling(config.WEEKLY_WINDOW).mean()
    features["RV_m"] = rv.shift(1).rolling(config.MONTHLY_WINDOW).mean()

    features["y"] = rv
    features = features.dropna()
    return features


def save_features(features: pd.DataFrame) -> None:
    """
    Save feature table in data/processed/features.csv.

    Args:
        features: DataFrame from compute_features()
    """
    path = config.DATA_PROCESSED_DIR / "features.csv"
    features.to_csv(path)
    print(f"[features] Saved: {path} ({len(features)} lines)")


def load_features() -> pd.DataFrame:
    """
        Load feature table from data/processed/features.csv.

        Returns:
            DataFrame с признаками и таргетом
    """
    path = config.DATA_PROCESSED_DIR / "features.csv"
    if not path.exists():
        raise FileNotFoundError(
            "features.csv doesn't exist in data/processed/features.csv"
        )

    features = pd.read_csv(path, index_col='Date', parse_dates=True)
    print(f"[features] Loaded: {path} ({len(features)} lines)")
    return features


if __name__ == "__main__":
    from loader import download_data

    raw_data = download_data()
    features = compute_features(raw_data)
    save_features(features)

    print(f"Example of feature data: {features.head()}")
    print(f"Period: {features.index[0].date()} — {features.index[-1].date()}")
    print(f"Statistics:\n{features.describe()}")
