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


def merge_sentiment(features: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    """
    Merge market features with NLP sentiment scores by date.
    Missing days (weekends, holidays) filled with forward fill.

    Args:
        features:  DataFrame from build_features() with RV_d, RV_w, RV_m, y
        sentiment: DataFrame from sentiment.py with sentiment_mean, sentiment_std, news_count

    Returns:
        DataFrame with all columns merged by date
    """

    sentiment = sentiment.set_index("date")
    sentiment.index.name = "Date"
    df = features.join(sentiment, how="left")

    nlp_cols = ["sentiment_mean", "sentiment_std", "news_count"]
    df[nlp_cols] = df[nlp_cols].ffill()

    df = df.dropna()
    return df


def save_features_nlp(df: pd.DataFrame) -> None:
    """
    Save merged features to data/processed/features_nlp.csv.

    Args:
        df: DataFrame with merged features.
    """
    path = config.DATA_PROCESSED_DIR / "features_nlp.csv"
    df.to_csv(path)
    print(f"[features] Saved: {path} ({len(df)} rows)")


def load_features_nlp() -> pd.DataFrame:
    """
    Load merged features from data/processed/features_nlp.csv.

    Returns:
        DataFrame with merged features.
    """
    path = config.DATA_PROCESSED_DIR / "features_nlp.csv"
    if not path.exists():
        raise FileNotFoundError("features_nlp.csv not found — run merge_sentiment() first")
    return pd.read_csv(path, index_col="Date", parse_dates=True)


if __name__ == "__main__":
    from loader import download_data

    raw_data = download_data()
    features = compute_features(raw_data)
    save_features(features)

    print(f"Example of feature data: {features.head()}")
    print(f"Period: {features.index[0].date()} — {features.index[-1].date()}")
    print(f"Statistics:\n{features.describe()}")
