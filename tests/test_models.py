import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.config import MODELS_DIR
from src.data.features import build_preprocessor, split_feature_target
from sklearn.pipeline import Pipeline

TRAINED_MODEL_FILES = [
    "logistic_regression__class_weight.pkl",
    "logistic_regression__smote.pkl",
    "random_forest__class_weight.pkl",
    "random_forest__smote.pkl",
    "xgboost__class_weight.pkl",
    "xgboost__smote.pkl",
    "final_model.pkl",
]


@pytest.fixture
def quick_pipeline():
    """A minimal, fast-fitting pipeline for mechanics tests (not tuned)."""
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=1000, random_state=0)),
    ])


class TestPipelineMechanics:
    def test_fit_predict_roundtrip(self, quick_pipeline, train_test_split_dfs):
        train_df, test_df = train_test_split_dfs
        X_train, y_train = split_feature_target(train_df)
        X_test, _ = split_feature_target(test_df)

        quick_pipeline.fit(X_train, y_train)
        preds = quick_pipeline.predict(X_test)
        assert len(preds) == len(X_test)
        assert set(np.unique(preds)) <= {0, 1}

    def test_predict_proba_in_valid_range(self, quick_pipeline, train_test_split_dfs):
        train_df, test_df = train_test_split_dfs
        X_train, y_train = split_feature_target(train_df)
        X_test, _ = split_feature_target(test_df)

        quick_pipeline.fit(X_train, y_train)
        proba = quick_pipeline.predict_proba(X_test)

        assert proba.shape == (len(X_test), 2)
        assert (proba >= 0).all() and (proba <= 1).all()
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-6)

    def test_predict_and_predict_proba_agree(self, quick_pipeline, train_test_split_dfs):
        train_df, test_df = train_test_split_dfs
        X_train, y_train = split_feature_target(train_df)
        X_test, _ = split_feature_target(test_df)

        quick_pipeline.fit(X_train, y_train)
        preds = quick_pipeline.predict(X_test)
        proba = quick_pipeline.predict_proba(X_test)[:, 1]
        derived_preds = (proba >= 0.5).astype(int)

        np.testing.assert_array_equal(preds, derived_preds)


class TestTrainedArtifacts:
    @pytest.mark.parametrize("filename", TRAINED_MODEL_FILES)
    def test_saved_model_loads_and_predicts(self, filename, train_test_split_dfs):
        path = MODELS_DIR / filename
        if not path.exists():
            pytest.skip(f"{filename} not trained yet -- run `python -m src.models.train` first")

        import joblib

        _, test_df = train_test_split_dfs
        X_test, y_test = split_feature_target(test_df)

        model = joblib.load(path)
        proba = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

        assert proba.shape[0] == len(X_test)
        assert (proba >= 0).all() and (proba <= 1).all()
        assert set(np.unique(preds)) <= {0, 1}
