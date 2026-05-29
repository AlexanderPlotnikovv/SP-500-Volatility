# S&P 500 Volatility Forecasting with NLP

Forecasting realized volatility of the S&P 500 index by integrating market time series with NLP sentiment features
extracted from financial news headlines using FinBERT.

## Results

| Model                 | MSE       | QLIKE  |
|-----------------------|-----------|--------|
| Baseline ($RV_{t-1}$) | 2.486e-07 | 2620.8 |
| HAR-RV                | 2.258e-07 | 1.617  |
| XGBoost               | 2.332e-07 | 1.954  |
| XGBoost + NLP         | 2.324e-07 | 1.954  |

XGBoost with NLP features achieves the best MSE, improving over the naive baseline by 7%.

## Architecture

```
S&P 500 (yfinance)          Financial News Headlines
        ↓                            ↓
   Log Returns               FinBERT Inference
        ↓                            ↓
  RV_d, RV_w, RV_m      sentiment_mean, sentiment_std, news_count
        ↓                            ↓
        └──────────── merge ─────────┘
                          ↓
              Walk-forward Validation (expanding window)
                          ↓
         Baseline | HAR-RV | XGBoost | XGBoost-NLP
                          ↓
                     results.json
```

## Quickstart

### macOS

```bash
brew install libomp
git clone https://github.com/AlexanderPlotnikovv/SP-500-Volatility
cd SP-500-Volatility
pip install -r requirements.txt
python3 run_pipeline.py
```

### Linux

```bash
git clone https://github.com/AlexanderPlotnikovv/SP-500-Volatility
cd SP-500-Volatility
pip install -r requirements.txt
python3 run_pipeline.py
```

## Pipeline

```
run_pipeline.py
├── Step 1: Download S&P 500 data (yfinance)
├── Step 2: Compute RV features (RV_d, RV_w, RV_m)
├── Step 3: Download headlines (HuggingFace)
├── Step 4: FinBERT inference → sentiment scores
├── Step 5: Merge market + NLP features
└── Step 6: Train models → results.json
```

## Project Structure

```
SP-500-Volatility/
├── src/
│   ├── data/
│   │   ├── loader.py       # S&P 500 data download
│   │   └── features.py     # RV computation and feature engineering
│   ├── models/
│   │   ├── base_model.py   # Abstract base class with walk-forward
│   │   ├── baseline.py     # Naive model (RV_{t-1})
│   │   ├── har_rv.py       # HAR-RV linear model
│   │   └── xgboost.py      # XGBoost and XGBoost+NLP
│   ├── evaluation/
│   │   └── metrics.py      # MSE, QLIKE
│   └── nlp/
│       ├── fetcher.py      # Headlines download
│       └── sentiment.py    # FinBERT inference
├── config.py               # All parameters in one place
├── run_pipeline.py         # Entry point
└── requirements.txt
```

## Data

- **Market data**: S&P 500 daily OHLCV via `yfinance`, 2011-2023
- **News headlines**: Financial news headlines paired with S&P 500 closing prices, hosted on HuggingFace
- **Target**: Realized Volatility `RV_t = r_t²` where `r_t = log(Close_t / Close_{t-1})`

## Models

**Baseline** — predicts tomorrow's RV as today's RV (`RV_{t-1}`)

**HAR-RV** — Heterogeneous Autoregressive model (Corsi, 2009):

```
RV_t = α + β_d·RV_{t-1} + β_w·RV̄_{t-5} + β_m·RV̄_{t-22} + ε
```

**XGBoost** — gradient boosting on market features (RV_d, RV_w, RV_m)

**XGBoost + NLP** — gradient boosting on market + sentiment features

## Validation

Walk-forward expanding window validation with minimum training size of 252 days (1 trading year). No data leakage —
model only sees past data at each step.

## Metrics

- **MSE** — Mean Squared Error
- **QLIKE** — standard loss function for volatility forecasting, robust to outliers:
  `QLIKE = mean(y_true/y_pred - log(y_true/y_pred) - 1)`

## References

1. Kubica, et al. (2025). Can AI read between the lines? Benchmarking LLMs on financial nuance. *arXiv:2505.16090*
2. Halousková, M., & Lyócsa, Š. (2025). Forecasting U.S. equity market volatility with attention and sentiment to the
   economy. *arXiv:2503.19767*
3. Du, K., et al. (2025). Natural Language Processing in Finance: A Survey. *Decision Support Systems*
4. Todd, A., et al. (2024). Text-based sentiment analysis in finance. *International Journal of Finance & Economics*