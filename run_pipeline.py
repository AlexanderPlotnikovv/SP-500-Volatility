import numpy as np
import json
from pathlib import Path

import config
from src.data.loader import download_data
from src.data.features import compute_features, save_features, load_features
from src.models.har_rv import HarRv, Baseline
from src.evaluation.metrics import compute_metrics

MODELS = [
    Baseline(),
    HarRv(),
]

FEATURE_COLS = ["RV_d", "RV_w", "RV_m"]
TARGET_COL = "y"

if __name__ == "__main__":
    # Step 1: Download raw data and compute features
    raw_data = download_data()
    features = compute_features(raw_data)
    save_features(features)

    # Step 2: Load features and prepare training data
    features = load_features()
    X = features[FEATURE_COLS]
    y = features[TARGET_COL]

    # Step 3: Train models and make predictions
    results = {}
    for model in MODELS:
        print(f"Training {model.name}...")
        predictions = model.window_expand_fit(X, y)
        metrics = compute_metrics(y, predictions)
        results[model.name] = metrics

    # Step 4: Save results
    output_path = config.OUTPUTS_DIR / "results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_path}")
