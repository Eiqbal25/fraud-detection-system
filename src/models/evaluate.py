"""Metrics, confusion matrix, ROC-AUC / PR-AUC for a fitted classifier."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def get_probabilities(model, X):
    """Probability of the positive (fraud) class."""
    return model.predict_proba(X)[:, 1]


def evaluate_model(model, X_test, y_test, threshold: float = 0.5) -> dict:
    y_proba = get_probabilities(model, X_test)
    y_pred = (y_proba >= threshold).astype(int)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "threshold": threshold,
        "precision_fraud": report["1"]["precision"],
        "recall_fraud": report["1"]["recall"],
        "f1_fraud": report["1"]["f1-score"],
        "precision_macro": report["macro avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"],
        "f1_macro": report["macro avg"]["f1-score"],
        "f1_weighted": report["weighted avg"]["f1-score"],
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "confusion_matrix": cm.tolist(),
        "n_test": len(y_test),
        "n_fraud_test": int(y_test.sum()),
    }


def best_threshold_for_recall(model, X_test, y_test, min_recall: float = 0.8) -> float:
    """Pick the highest-precision threshold that still meets a minimum recall.

    False negatives (missed fraud) are costlier than false positives here, so
    the final operating threshold should be chosen to guarantee a recall
    floor rather than maximizing F1/accuracy outright.
    """
    y_proba = get_probabilities(model, X_test)
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    # precision_recall_curve returns len(thresholds) == len(precisions) - 1
    candidates = [(p, t) for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds) if r >= min_recall]
    if not candidates:
        return 0.5
    best_precision, best_thresh = max(candidates, key=lambda pt: pt[0])
    return float(best_thresh)


def plot_confusion_matrix(model, X_test, y_test, out_path, threshold: float = 0.5, title: str = ""):
    y_proba = get_probabilities(model, X_test)
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Fraud", "Fraud"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title or "Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
