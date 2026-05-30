import numpy as np
import json
from pathlib import Path
import pandas as pd

import config
from src.data.loader import download_data
from src.data.features import compute_features, save_features, load_features
from src.models.baseline import Baseline
from src.models.har_rv import HarRv
from src.models.xgboost import XGBoost, XGBoostNLP
from src.evaluation.metrics import compute_metrics
from src.data.features import merge_sentiment, save_features_nlp, load_features_nlp
from src.nlp.fetcher import download_headlines, load_headlines
from src.nlp.sentiment import load_sentiment_scores, load_model, get_scores, aggregate_scores, save_sentiment_scores

MODELS = [
    Baseline(),
    HarRv(),
    XGBoost(),
    XGBoostNLP(),
]

TARGET_COL = "y"

if __name__ == "__main__":
    config.ensure_dirs()

    # Step 1: Download raw data and compute features
    raw_data = download_data()
    features = compute_features(raw_data)
    save_features(features)
    features = load_features()

    # Step 2: Download headlines and compute sentiment
    sentiment_path = config.DATA_PROCESSED_DIR / "sentiment.csv"
    if sentiment_path.exists():
        print("[pipeline] Sentiment cache found, skipping inference.")
    else:
        download_headlines()
        headlines = load_headlines()
        model_nlp = load_model()
        scores = get_scores(model_nlp, headlines["headline"])
        sentiment = aggregate_scores(headlines, scores)
        save_sentiment_scores(sentiment)

    # Step 3: Load sentiment features
    sentiment_scores = load_sentiment_scores()
    features_nlp = merge_sentiment(features, sentiment_scores)
    save_features_nlp(features_nlp)
    features_nlp = load_features_nlp()

    # Step 4: Train models and make predictions
    results = {}
    for model in MODELS:
        print(f"Training {model.name}...")

        df = features_nlp if model.USE_NLP else features
        X = df[model.FEATURE_COLS]
        y = df[TARGET_COL]

        predictions = model.window_expand_fit(X, y)
        mask = ~np.isnan(predictions)
        preds_clean = np.array(predictions)[mask]
        y_clean = np.array(y)[mask]
        dates_clean = y.index[mask]

        metrics = compute_metrics(y_clean, preds_clean)
        results[model.name] = {
            "metrics": metrics,
            "dates": [str(d.date()) for d in dates_clean],
            "predictions": preds_clean.tolist(),
            "actuals": y_clean.tolist(),
        }

    # Step 5: Save results
    output_path = config.OUTPUTS_DIR / "results.json"
    output = {
        "period": {
            "start": config.START_DATE,
            "end": config.END_DATE,
        },
        "models": results
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)
    print(f"Results saved to {output_path}")
