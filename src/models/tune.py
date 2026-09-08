"""RandomizedSearchCV hyperparameter tuning, 5-fold stratified, optimizing PR-AUC."""
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.config import RANDOM_SEED

N_ITER = 25
CV_FOLDS = 5
SCORING = "average_precision"  # PR-AUC: more informative than accuracy/F1 alone on imbalanced data

PARAM_DISTRIBUTIONS = {
    "logistic_regression": {
        "clf__C": uniform(0.01, 10),
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs"],
    },
    "random_forest": {
        "clf__n_estimators": randint(100, 500),
        "clf__max_depth": randint(3, 20),
        "clf__min_samples_leaf": randint(1, 10),
        "clf__max_features": ["sqrt", "log2", None],
    },
    "xgboost": {
        "clf__n_estimators": randint(100, 500),
        "clf__max_depth": randint(2, 10),
        "clf__learning_rate": uniform(0.01, 0.3),
        "clf__subsample": uniform(0.6, 0.4),
        "clf__colsample_bytree": uniform(0.6, 0.4),
    },
}


def tune_model(pipeline, param_key: str, X_train, y_train):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=PARAM_DISTRIBUTIONS[param_key],
        n_iter=N_ITER,
        scoring=SCORING,
        cv=cv,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_
