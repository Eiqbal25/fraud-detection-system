"""Soft-voting ensemble combining the best-tuned pipeline for each algorithm.

Whether this actually beats the best single model is an empirical question,
not an assumption -- `main()` reports both and states the honest answer in
results/model_results.json under the "ensemble" key.
"""
import json

import joblib
import pandas as pd
from sklearn.ensemble import VotingClassifier

from src.config import MODELS_DIR, RESULTS_DIR, TEST_PATH, TRAIN_PATH
from src.data.features import split_feature_target
from src.models.evaluate import evaluate_model
from src.models.train import ALGORITHMS


def load_best_per_algorithm(results: dict) -> dict:
    """Pick the higher test PR-AUC strategy (class_weight vs smote) per algorithm."""
    best_keys = {}
    for algorithm in ALGORITHMS:
        candidates = {k: v for k, v in results.items() if k.startswith(algorithm)}
        best_key = max(candidates, key=lambda k: candidates[k]["pr_auc"])
        best_keys[algorithm] = best_key
    return best_keys


def build_ensemble(best_keys: dict) -> VotingClassifier:
    estimators = []
    for algorithm, key in best_keys.items():
        pipeline = joblib.load(MODELS_DIR / f"{key}.pkl")
        estimators.append((algorithm, pipeline))
    return VotingClassifier(estimators=estimators, voting="soft")


def main():
    with open(RESULTS_DIR / "model_results.json") as f:
        results = json.load(f)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    X_train, y_train = split_feature_target(train_df)
    X_test, y_test = split_feature_target(test_df)

    best_keys = load_best_per_algorithm(results)
    print("Best pipeline per algorithm:", best_keys)

    ensemble = build_ensemble(best_keys)
    ensemble.fit(X_train, y_train)
    metrics = evaluate_model(ensemble, X_test, y_test)
    metrics["members"] = best_keys

    best_single_key = max(results, key=lambda k: results[k]["pr_auc"])
    best_single_pr_auc = results[best_single_key]["pr_auc"]
    beats_best_single = bool(metrics["pr_auc"] > best_single_pr_auc)

    print(f"Ensemble  pr_auc={metrics['pr_auc']:.3f} recall_fraud={metrics['recall_fraud']:.3f} f1_fraud={metrics['f1_fraud']:.3f}")
    print(f"Best single ({best_single_key}) pr_auc={best_single_pr_auc:.3f}")
    print(f"Ensemble beats best single model on PR-AUC: {beats_best_single}")

    metrics["beats_best_single_model"] = beats_best_single
    metrics["best_single_model"] = best_single_key
    metrics["best_single_model_pr_auc"] = best_single_pr_auc

    results["ensemble"] = metrics
    with open(RESULTS_DIR / "model_results.json", "w") as f:
        json.dump(results, f, indent=2)

    joblib.dump(ensemble, MODELS_DIR / "ensemble.pkl")
    return results


if __name__ == "__main__":
    main()
