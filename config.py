from pathlib import Path

# Root project directory
ROOT_DIR = Path(__file__).parent

# Paths to files
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUTS_DIR = ROOT_DIR / "outputs" / "predictions"

# Market data
TICKER = "^GSPC"
START_DATE = "2011-01-04"
END_DATE = "2023-12-31"

# RV data
WEEKLY_WINDOW = 5
MONTHLY_WINDOW = 22

# Walk-forward validation
MIN_TRAIN_SIZE = 252
