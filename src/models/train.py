"""Train Logistic Regression, Random Forest, and XGBoost under two imbalance
strategies (class weighting vs. SMOTE), tune each with RandomizedSearchCV,
evaluate on the held-out test set, and persist the fitted pipelines.

Each saved artifact is a full sklearn/imblearn Pipeline: preprocessing ->
(SMOTE, train-time only) -> classifier. Downstream consumers (API, dashboard,
SHAP) can feed it the engineered-but-unencoded feature DataFrame directly.
"""
import json

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import MODELS_DIR, RANDOM_SEED, RESULTS_DIR, TEST_PATH, TRAIN_PATH
from src.data.features import build_preprocessor, split_feature_target
from src.models.evaluate import evaluate_model
from src.models.tune import tune_model

ALGORITHMS = ["logistic_regression", "random_forest", "xgboost"]
STRATEGIES = ["class_weight", "smote"]


def make_classifier(algorithm: str, strategy: str, scale_pos_weight: float):
    if algorithm == "logistic_regression":
        if strategy == "class_weight":
            return LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED)
        return LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)

    if algorithm == "random_forest":
        if strategy == "class_weight":
            return RandomForestClassifier(class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)
        return RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)

    if algorithm == "xgboost":
        if strategy == "class_weight":
            return XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=RANDOM_SEED, eval_metric="logloss")
        return XGBClassifier(random_state=RANDOM_SEED, eval_metric="logloss")

    raise ValueError(algorithm)


def build_pipeline(algorithm: str, strategy: str, scale_pos_weight: float):
    clf = make_classifier(algorithm, strategy, scale_pos_weight)
    if strategy == "smote":
        return ImbPipeline([
            ("preprocess", build_preprocessor()),
            ("smote", SMOTE(random_state=RANDOM_SEED)),
            ("clf", clf),
        ])
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("clf", clf),
    ])


def train_all(X_train, y_train, X_test, y_test):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    results = {}
    fitted = {}

    for algorithm in ALGORITHMS:
        for strategy in STRATEGIES:
            key = f"{algorithm}__{strategy}"
            pipeline = build_pipeline(algorithm, strategy, scale_pos_weight)
            best_pipeline, best_params, best_cv_score = tune_model(pipeline, algorithm, X_train, y_train)
            metrics = evaluate_model(best_pipeline, X_test, y_test)
            metrics["best_params"] = {k: (v if not hasattr(v, "item") else v.item()) for k, v in best_params.items()}
            metrics["cv_pr_auc"] = best_cv_score

            results[key] = metrics
            fitted[key] = best_pipeline

            joblib.dump(best_pipeline, MODELS_DIR / f"{key}.pkl")
            print(f"[{key}] cv_pr_auc={best_cv_score:.3f} test_pr_auc={metrics['pr_auc']:.3f} "
                  f"recall_fraud={metrics['recall_fraud']:.3f} f1_fraud={metrics['f1_fraud']:.3f}")

    return results, fitted, scale_pos_weight


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    X_train, y_train = split_feature_target(train_df)
    X_test, y_test = split_feature_target(test_df)

    results, fitted, scale_pos_weight = train_all(X_train, y_train, X_test, y_test)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "model_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved all pipelines to", MODELS_DIR)
    print("Saved raw metrics to", RESULTS_DIR / "model_results.json")


if __name__ == "__main__":
    main()
