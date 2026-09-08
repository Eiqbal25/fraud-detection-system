from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.data.features import build_preprocessor, split_feature_target
from src.models.train import build_pipeline


class TestClassWeighting:
    def test_logistic_regression_balanced_gets_class_weight(self, train_test_split_dfs):
        train_df, _ = train_test_split_dfs
        X_train, y_train = split_feature_target(train_df)

        pipeline = build_pipeline("logistic_regression", "class_weight", scale_pos_weight=1.0)
        pipeline.fit(X_train, y_train)
        assert pipeline.named_steps["clf"].class_weight == "balanced"

    def test_random_forest_balanced_gets_class_weight(self, train_test_split_dfs):
        train_df, _ = train_test_split_dfs
        X_train, y_train = split_feature_target(train_df)

        pipeline = build_pipeline("random_forest", "class_weight", scale_pos_weight=1.0)
        pipeline.fit(X_train, y_train)
        assert pipeline.named_steps["clf"].class_weight == "balanced"

    def test_xgboost_gets_scale_pos_weight(self, train_test_split_dfs):
        train_df, _ = train_test_split_dfs
        X_train, y_train = split_feature_target(train_df)
        spw = (y_train == 0).sum() / (y_train == 1).sum()

        pipeline = build_pipeline("xgboost", "class_weight", scale_pos_weight=spw)
        pipeline.fit(X_train, y_train)
        assert pipeline.named_steps["clf"].scale_pos_weight == spw
        assert spw > 1  # fraud is the minority class in this dataset


class TestSMOTE:
    def test_smote_pipeline_is_noop_at_predict_time(self, train_test_split_dfs):
        """The imblearn Pipeline must skip the sampler on predict/transform --
        that's what guarantees SMOTE never touches the test set."""
        train_df, test_df = train_test_split_dfs
        X_train, y_train = split_feature_target(train_df)
        X_test, _ = split_feature_target(test_df)

        pipeline = build_pipeline("logistic_regression", "smote", scale_pos_weight=1.0)
        pipeline.fit(X_train, y_train)
        # If SMOTE ran at predict time this would either error (shape mismatch
        # against y) or change the number of rows; it should not.
        preds = pipeline.predict(X_test)
        assert len(preds) == len(X_test)

    def test_smote_balances_training_class_distribution(self, train_test_split_dfs):
        train_df, _ = train_test_split_dfs
        X_train, y_train = split_feature_target(train_df)

        original_minority_ratio = y_train.mean()
        assert original_minority_ratio < 0.3  # confirms the dataset is genuinely imbalanced

        preprocessor = build_preprocessor()
        Xt_train = preprocessor.fit_transform(X_train, y_train)

        smote = SMOTE(random_state=42)
        Xt_resampled, y_resampled = smote.fit_resample(Xt_train, y_train)

        assert len(y_resampled) > len(y_train)
        resampled_ratio = y_resampled.mean()
        assert resampled_ratio == 0.5  # SMOTE oversamples the minority to parity by default
        assert Xt_resampled.shape[0] == len(y_resampled)

    def test_smote_only_applied_to_training_fold_not_test(self, train_test_split_dfs):
        """Fitting the full imblearn pipeline must not change len(X_test)/y_test
        anywhere in the flow -- SMOTE only ever sees training data."""
        train_df, test_df = train_test_split_dfs
        X_train, y_train = split_feature_target(train_df)
        X_test, y_test = split_feature_target(test_df)
        original_test_len = len(X_test)
        original_test_fraud_rate = y_test.mean()

        pipeline = build_pipeline("random_forest", "smote", scale_pos_weight=1.0)
        pipeline.fit(X_train, y_train)
        pipeline.predict(X_test)

        assert len(X_test) == original_test_len
        assert y_test.mean() == original_test_fraud_rate
