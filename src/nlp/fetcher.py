import pandas as pd
import config


def download_headlines() -> str:
    """
    Download financial headlines dataset from HuggingFace Hub.
    Returns path to downloaded CSV file.
    """
    from huggingface_hub import hf_hub_download

    cache_path = config.DATA_RAW_DIR / "sp500_headlines.csv"
    if cache_path.exists():
        print("[fetcher] Headlines cache found, skipping download.")
        return str(cache_path)

    print("[fetcher] Downloading headlines from HuggingFace...")
    path = hf_hub_download(
        repo_id="AlexPlotnikovTech/sp500-headlines",
        filename="sp500_headlines.csv",
        repo_type="dataset",
        local_dir=str(config.DATA_RAW_DIR),
    )
    print(f"[fetcher] Downloaded to {path}")
    return path


def load_headlines(
        path: str = None,
        start: str = config.START_DATE,
        end: str = config.END_DATE,
) -> pd.DataFrame:
    """
    Load and clean financial news headlines from raw CSV.

    Args:
        path:  path to raw CSV file, defaults to data/raw/sp500_headlines.csv
        start: start date filter in YYYY-MM-DD format
        end:   end date filter in YYYY-MM-DD format

    Returns:
        DataFrame with columns: date (datetime), headline (str)
    """
    if path is None:
        path = config.DATA_RAW_DIR / "sp500_headlines.csv"

    print(f"[fetcher] Loading headlines from {path}...")
    df = pd.read_csv(path, usecols=["Title", "Date"])

    df = df.rename(columns={"Title": "headline", "Date": "date"})

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "headline"])

    df = df[df["date"].between(start, end)]
    df["date"] = df["date"].dt.normalize()

    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    print(f"[fetcher] Loaded {len(df)} headlines "
          f"from {df['date'].min().date()} to {df['date'].max().date()}")

    return df


def save_headlines(df: pd.DataFrame) -> None:
    """
    Save cleaned headlines to data/processed/headlines.csv.

    Args:
        df: DataFrame with columns: date, headline
    """
    path = config.DATA_PROCESSED_DIR / "headlines.csv"
    df.to_csv(path, index=False)
    print(f"[fetcher] Saved: {path} ({len(df)} rows)")


if __name__ == "__main__":
    download_headlines()
    df = load_headlines()
    print(df.head())
    print(f"\nHeadlines per year:")
    print(df.groupby(df["date"].dt.year).size())
    save_headlines(df)
