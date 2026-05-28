import yfinance as yf
import pandas as pd
import config


def download_data(
        ticker: str = config.TICKER,
        start_date: str = config.START_DATE,
        end_date: str = config.END_DATE,
        force_redownload: bool = False
) -> pd.DataFrame:
    """
    Download historical stock data using yfinance.

    Args:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date for data in 'YYYY-MM-DD' format.
        end_date (str): End date for data in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: DataFrame containing historical stock data.
    """

    cache_path = config.DATA_RAW_DIR / f"{ticker.replace('^', '')}.csv"
    if cache_path.exists() and not force_redownload:
        print(f"[loader] Read from cache: {cache_path}")
        df = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
        return df

    print(f"[loader] Download {ticker} from {start_date} to {end_date}...")
    df = yf.download(ticker, start_date, end_date, auto_adjust=True)

    if df.empty:
        raise Exception("[loader] Download failed")

    df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index.name = "Date"
    df.to_csv(cache_path)
    print(f"[loader] {ticker} data saved: {cache_path}")

    return df


if __name__ == "__main__":
    ticker = config.TICKER
    df = download_data(ticker=ticker)
    print(f"Example of {ticker} data: {df.head()}")
    print(f"Period from {df.index[0].date()} to {df.index[-1].date()}")
