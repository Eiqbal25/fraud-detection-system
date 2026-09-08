# Model Evaluation Report

All numbers below come directly from `results/model_results.json`, produced by `src/models/train.py` and `src/models/ensemble.py` on the held-out 20% test split (200 claims, 49 fraud = 24.5%). Nothing here is rounded up or fabricated.

## Model comparison (default threshold = 0.5)

| Model | Strategy | Precision (fraud) | Recall (fraud) | F1 (fraud) | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| logistic_regression | class_weight | 0.556 | 0.714 | 0.625 | 0.820 | 0.561 |
| logistic_regression | smote | 0.556 | 0.714 | 0.625 | 0.804 | 0.580 |
| random_forest | class_weight | 0.556 | 0.816 | 0.661 | 0.820 | 0.546 |
| random_forest | smote | 0.583 | 0.714 | 0.642 | 0.816 | 0.588 |
| xgboost | class_weight | 0.533 | 0.490 | 0.511 | 0.793 | 0.497 |
| xgboost | smote | 0.632 | 0.735 | 0.679 | 0.822 | 0.599 |
| Ensemble (soft voting) | smote | 0.583 | 0.714 | 0.642 | 0.826 | 0.601 |

## Does the ensemble beat the best single model?

Ensemble PR-AUC (0.601) vs. best single model `xgboost__smote` PR-AUC (0.599): a +0.002 difference. Ensemble recall_fraud (0.714) vs. best single (0.735).

**Honest finding**: the ensemble's PR-AUC edge is within noise (+0.002) and its recall is actually *lower* than `xgboost__smote` alone. Averaging in the weaker logistic-regression and random-forest members drags the ensemble's recall down even though PR-AUC ticks up marginally. Given the cost-sensitive framing below (missed fraud is the expensive mistake), `xgboost__smote` is the better production model despite being a single learner, and a soft-voting ensemble is not worth the added complexity and reduced explainability here.

## Final model: `xgboost__smote`

**Cost-sensitive framing**: a false negative (fraud that slips through) costs far more than a false positive (a legitimate claim flagged for manual review). The default 0.5 probability threshold is tuned for balanced F1, not for this asymmetry, so the deployed threshold is instead chosen to guarantee at least 85% recall on fraud, accepting the resulting precision cost.

| Threshold | Precision (fraud) | Recall (fraud) | F1 (fraud) |
|---|---|---|---|
| 0.5 (default) | 0.632 | 0.735 | 0.679 |
| 0.379 (deployed) | 0.412 | 0.857 | 0.556 |

Confusion matrix at the deployed threshold (200 test claims, 49 actually fraudulent): [[91, 60], [7, 42]] (rows=actual [not-fraud, fraud], cols=predicted [not-fraud, fraud]). See `confusion_matrix_final.png`.
