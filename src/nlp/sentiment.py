import pandas as pd
import numpy as np
from transformers import pipeline
import torch
import config

from src.nlp.fetcher import load_headlines


def load_model():
    """
    Load the FinBERT model for financial sentiment analysis.
    Uses MPS (Apple Silicon) or CUDA if available, otherwise CPU.

     Returns:
        HuggingFace pipeline for text classification
    """
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"[sentiment] Loading FinBERT on {device}...")
    model = pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        device=device,
        top_k=None,
    )
    print(f"[sentiment] FinBERT loaded.")
    return model


def get_scores(
        model,
        headlines: pd.Series,
        batch_size: int = 64,
) -> np.ndarray:
    """
    Run FinBERT on a dataframe of headlines.
    Scores = P(positive) - P(negative)

    Args:
        model:      FinBERT pipeline with top_k=None
        headlines:  Series of headline strings
        batch_size: number of headlines per batch

    Returns:
        numpy array of sentiment scores per headline
    """
    texts = headlines.tolist()
    scores = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        results = model(batch, truncation=True, max_length=512)

        for res in results:
            res_dict = {item["label"]: item["score"] for item in res}
            score = res_dict.get("positive", 0.0) - res_dict.get("negative", 0.0)
            scores.append(score)

        if i % 10000 == 0:
            print(f"[sentiment] Processed {i}/{len(texts)} headlines...")

    return np.array(scores)


def aggregate_scores(
        df: pd.DataFrame,
        scores: np.ndarray,
) -> pd.DataFrame:
    """
    Aggregate sentiment scores by date.

    Args:
        df:     DataFrame with column: date
        scores: sentiment scores per headline

    Returns:
        DataFrame with columns: date, sentiment_mean, sentiment_std, news_count
    """

    df = df.copy()
    df["score"] = scores

    agg = df.groupby("date")["score"].agg(
        sentiment_mean="mean",
        sentiment_std="std",
        news_count="count",
    ).reset_index()

    agg["sentiment_std"] = agg["sentiment_std"].fillna(0.0)
    return agg


def save_sentiment_scores(
        df: pd.DataFrame,
) -> None:
    """
    Save sentiment scores to data/processed/sentiment.csv.

    Args:
        df: DataFrame with columns: date, sentiment_mean, sentiment_std, news_count
    """
    path = config.DATA_PROCESSED_DIR / "sentiment.csv"
    df.to_csv(path, index=False)
    print(f"[sentiment] Saved: {path} ({len(df)} rows)")


def load_sentiment_scores() -> pd.DataFrame:
    """
    Load sentiment scores from data/processed/sentiment.csv.

    Returns:
        DataFrame with columns: date, sentiment_mean, sentiment_std, news_count
    """

    path = config.DATA_PROCESSED_DIR / "sentiment.csv"
    if not path.exists():
        raise FileNotFoundError(
            "sentiment.csv not found — run sentiment.py first"
        )
    df = pd.read_csv(path, parse_dates=["date"])
    return df


if __name__ == "__main__":
    sentiment_path = config.DATA_PROCESSED_DIR / "sentiment.csv"
    if sentiment_path.exists():
        print(f"[sentiment] Cache found: {sentiment_path}, skipping inference.")
        sentiment_scores = load_sentiment_scores()
    else:
        headlines = load_headlines()

        model = load_model()

        scores = get_scores(model, headlines["headline"])

        sentiment_scores = aggregate_scores(headlines, scores)

        save_sentiment_scores(sentiment_scores)

    print(sentiment_scores.head(10))
    print(f"\nPeriod: {sentiment_scores['date'].min().date()} — {sentiment_scores['date'].max().date()}")
    print(f"Days with sentiment: {len(sentiment_scores)}")
