"""SHAP explainability for the final deployed model.

Works on the pipeline's inner classifier + the preprocessor's transformed
matrix (SHAP's TreeExplainer needs numeric arrays, not raw claim fields).
`explain_instance` re-exposes a single claim's top contributing features in
terms of the ORIGINAL, human-readable engineered feature names, for the API.
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import MODELS_DIR, RESULTS_DIR, SHAP_PLOTS_DIR, TEST_PATH
from src.data.features import get_preprocessed_feature_names, split_feature_target


def load_final_pipeline():
    return joblib.load(MODELS_DIR / "final_model.pkl")


def build_explainer(pipeline, X_background: pd.DataFrame):
    preprocessor = pipeline.named_steps["preprocess"]
    clf = pipeline.named_steps["clf"]
    Xt_background = preprocessor.transform(X_background)
    explainer = shap.TreeExplainer(clf)
    return explainer, preprocessor


def global_feature_importance(pipeline, X: pd.DataFrame, out_path=None):
    explainer, preprocessor = build_explainer(pipeline, X)
    Xt = preprocessor.transform(X)
    feature_names = get_preprocessed_feature_names(preprocessor)

    shap_values = explainer.shap_values(Xt)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive (fraud) class

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    if out_path:
        fig, ax = plt.subplots(figsize=(8, 8))
        shap.summary_plot(shap_values, Xt, feature_names=feature_names, plot_type="bar", show=False, max_display=15)
        fig = plt.gcf()
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    return importance, shap_values, feature_names


def explain_instance(pipeline, X_single_row: pd.DataFrame, top_n: int = 5) -> list:
    """Top contributing features for one claim, as [(feature, shap_value), ...]."""
    preprocessor = pipeline.named_steps["preprocess"]
    clf = pipeline.named_steps["clf"]
    feature_names = get_preprocessed_feature_names(preprocessor)

    Xt = preprocessor.transform(X_single_row)
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(Xt)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    row_values = shap_values[0]
    order = np.argsort(-np.abs(row_values))[:top_n]
    return [{"feature": feature_names[i], "shap_value": float(row_values[i])} for i in order]


def main():
    pipeline = load_final_pipeline()
    test_df = pd.read_csv(TEST_PATH)
    X_test, y_test = split_feature_target(test_df)

    SHAP_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    importance, shap_values, feature_names = global_feature_importance(
        pipeline, X_test, out_path=SHAP_PLOTS_DIR / "global_importance.png"
    )
    importance.to_csv(RESULTS_DIR / "shap_global_importance.csv", index=False)
    print("Top 10 global features driving fraud predictions:")
    print(importance.head(10).to_string(index=False))

    # Local explanation for one genuinely fraudulent claim in the test set, as an example.
    fraud_idx = y_test[y_test == 1].index[0]
    local_row = X_test.loc[[fraud_idx]]
    local_explanation = explain_instance(pipeline, local_row, top_n=5)
    print(f"\nLocal explanation for test claim index {fraud_idx} (actual fraud={y_test.loc[fraud_idx]}):")
    for item in local_explanation:
        print(f"  {item['feature']}: {item['shap_value']:+.4f}")

    with open(RESULTS_DIR / "shap_local_example.json", "w") as f:
        json.dump({"test_index": int(fraud_idx), "top_features": local_explanation}, f, indent=2)


if __name__ == "__main__":
    main()
