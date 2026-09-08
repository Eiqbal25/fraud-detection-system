import numpy as np
import pandas as pd
import pytest

from src.data.features import FrequencyEncoder, engineer_features


class TestEngineerFeatures:
    def test_days_to_incident_is_correct(self, sample_raw_rows):
        out = engineer_features(sample_raw_rows)
        assert out["days_to_incident"].tolist() == [10, 10]

    def test_claim_to_premium_ratio(self, sample_raw_rows):
        out = engineer_features(sample_raw_rows)
        assert out.loc[0, "claim_to_premium_ratio"] == pytest.approx(1000.0 / 500.0)

    def test_zero_premium_produces_nan_not_inf(self, sample_raw_rows):
        out = engineer_features(sample_raw_rows)
        assert np.isnan(out.loc[1, "claim_to_premium_ratio"])

    def test_zero_vehicles_produces_nan_not_inf(self, sample_raw_rows):
        out = engineer_features(sample_raw_rows)
        assert np.isnan(out.loc[1, "claim_per_vehicle"])

    def test_claim_component_ratios_sum_near_one(self, sample_raw_rows):
        out = engineer_features(sample_raw_rows)
        total_ratio = out.loc[0, "injury_claim_ratio"] + out.loc[0, "property_claim_ratio"] + out.loc[0, "vehicle_claim_ratio"]
        assert total_ratio == pytest.approx(1.0)

    def test_id_and_date_columns_dropped(self, sample_raw_rows):
        out = engineer_features(sample_raw_rows)
        for col in ["policy_number", "insured_zip", "incident_location", "policy_bind_date", "incident_date"]:
            assert col not in out.columns


class TestFrequencyEncoder:
    def test_fit_computes_relative_frequency(self):
        X = pd.DataFrame({"cat": ["a", "a", "a", "b"]})
        enc = FrequencyEncoder().fit(X)
        assert enc.freq_maps_[0]["a"] == pytest.approx(0.75)
        assert enc.freq_maps_[0]["b"] == pytest.approx(0.25)

    def test_transform_maps_known_categories(self):
        X = pd.DataFrame({"cat": ["a", "a", "b"]})
        enc = FrequencyEncoder().fit(X)
        result = enc.transform(pd.DataFrame({"cat": ["a", "b"]}))
        np.testing.assert_allclose(result[:, 0], [2 / 3, 1 / 3])

    def test_transform_maps_unseen_category_to_zero(self):
        X = pd.DataFrame({"cat": ["a", "a", "b"]})
        enc = FrequencyEncoder().fit(X)
        result = enc.transform(pd.DataFrame({"cat": ["never_seen"]}))
        assert result[0, 0] == 0.0

    def test_does_not_mutate_fit_data(self):
        X = pd.DataFrame({"cat": ["a", "b"]})
        X_copy = X.copy()
        FrequencyEncoder().fit(X)
        pd.testing.assert_frame_equal(X, X_copy)
