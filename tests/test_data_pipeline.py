import numpy as np
import pandas as pd
import pytest

from src.config import TARGET_COL
from src.data.features import build_preprocessor, split_feature_target


class TestLoadAndClean:
    def test_target_is_binary(self, raw_df):
        assert set(raw_df[TARGET_COL].unique()) <= {0, 1}

    def test_no_question_mark_tokens_remain(self, raw_df):
        obj_cols = raw_df.select_dtypes(include="object").columns
        for col in obj_cols:
            assert "?" not in raw_df[col].values

    def test_no_duplicate_rows_after_clean(self, engineered_df):
        assert engineered_df.duplicated().sum() == 0

    def test_no_negative_ages(self, engineered_df):
        assert (engineered_df["age"] >= 0).all()

    def test_id_like_and_date_cols_dropped(self, engineered_df):
        for col in ["policy_number", "insured_zip", "incident_location", "policy_bind_date", "incident_date"]:
            assert col not in engineered_df.columns


class TestSplit:
    def test_split_shapes(self, train_test_split_dfs, engineered_df):
        train_df, test_df = train_test_split_dfs
        assert len(train_df) + len(test_df) == len(engineered_df)
        assert len(test_df) == pytest.approx(len(engineered_df) * 0.2, abs=1)

    def test_split_preserves_class_ratio(self, train_test_split_dfs):
        train_df, test_df = train_test_split_dfs
        train_rate = train_df[TARGET_COL].mean()
        test_rate = test_df[TARGET_COL].mean()
        assert abs(train_rate - test_rate) < 0.05

    def test_no_row_overlap_between_splits(self, train_test_split_dfs):
        train_df, test_df = train_test_split_dfs
        # Reconstruct a comparable key since indices are reset on both sides.
        train_keys = set(map(tuple, train_df.astype(str).values.tolist()))
        test_keys = set(map(tuple, test_df.astype(str).values.tolist()))
        assert train_keys.isdisjoint(test_keys)


class TestNoLeakage:
    def test_scaler_stats_come_from_train_only(self, train_test_split_dfs):
        train_df, test_df = train_test_split_dfs
        X_train, y_train = split_feature_target(train_df)
        X_test, _ = split_feature_target(test_df)

        pre = build_preprocessor()
        pre.fit(X_train, y_train)
        scaler = pre.named_transformers_["numeric"].named_steps["scale"]
        expected_mean = X_train["age"].mean()

        from src.data.features import NUMERIC_COLS
        age_pos = NUMERIC_COLS.index("age")
        assert scaler.mean_[age_pos] == pytest.approx(expected_mean, rel=1e-6)

        # The train-only mean must differ from the full-dataset mean; if it
        # didn't, this test wouldn't be able to detect an accidental fit on
        # the combined data (a real leakage bug) in the assertion above.
        combined_X = pd.concat([X_train, X_test], ignore_index=True)
        combined_mean = combined_X["age"].mean()
        assert combined_mean != pytest.approx(expected_mean, rel=1e-6)

    def test_transform_does_not_refit(self, train_test_split_dfs):
        train_df, test_df = train_test_split_dfs
        X_train, y_train = split_feature_target(train_df)
        X_test, _ = split_feature_target(test_df)

        pre = build_preprocessor()
        pre.fit(X_train, y_train)
        scaler_before = pre.named_transformers_["numeric"].named_steps["scale"].mean_.copy()
        pre.transform(X_test)
        scaler_after = pre.named_transformers_["numeric"].named_steps["scale"].mean_
        np.testing.assert_array_equal(scaler_before, scaler_after)

    def test_frequency_encoder_unseen_category_maps_to_zero(self, train_test_split_dfs):
        train_df, test_df = train_test_split_dfs
        X_train, y_train = split_feature_target(train_df)

        pre = build_preprocessor()
        pre.fit(X_train, y_train)
        freq_encoder = pre.named_transformers_["frequency"].named_steps["encode"]

        unseen = pd.DataFrame({"auto_make": ["TotallyMadeUpBrand"] * 1, "auto_model": ["x"], "insured_occupation": ["x"], "insured_hobbies": ["x"]})
        # Only test the column actually fit against for a genuinely unseen value.
        col = "auto_make"
        col_idx = list(X_train[["insured_occupation", "insured_hobbies", "auto_make", "auto_model"]].columns).index(col)
        result = freq_encoder.transform(unseen[["insured_occupation", "insured_hobbies", "auto_make", "auto_model"]])
        assert result[0, col_idx] == 0.0


class TestShapesAndTypes:
    def test_preprocessor_output_shape(self, train_test_split_dfs):
        train_df, test_df = train_test_split_dfs
        X_train, y_train = split_feature_target(train_df)
        X_test, _ = split_feature_target(test_df)

        pre = build_preprocessor()
        Xt_train = pre.fit_transform(X_train, y_train)
        Xt_test = pre.transform(X_test)

        assert Xt_train.shape[0] == len(train_df)
        assert Xt_test.shape[0] == len(test_df)
        assert Xt_train.shape[1] == Xt_test.shape[1]

    def test_preprocessor_output_has_no_nulls(self, train_test_split_dfs):
        train_df, test_df = train_test_split_dfs
        X_train, y_train = split_feature_target(train_df)
        X_test, _ = split_feature_target(test_df)

        pre = build_preprocessor()
        Xt_train = pre.fit_transform(X_train, y_train)
        Xt_test = pre.transform(X_test)

        assert not np.isnan(Xt_train).any()
        assert not np.isnan(Xt_test).any()

    def test_preprocessor_output_is_numeric(self, train_test_split_dfs):
        train_df, test_df = train_test_split_dfs
        X_train, y_train = split_feature_target(train_df)

        pre = build_preprocessor()
        Xt_train = pre.fit_transform(X_train, y_train)
        assert np.issubdtype(Xt_train.dtype, np.floating)
