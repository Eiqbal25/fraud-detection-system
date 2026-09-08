"""Feature engineering: row-wise derived features + a fit-on-train preprocessor.

Two distinct stages, kept separate on purpose:
  1. `engineer_features` — pure row-wise arithmetic (ratios, date deltas).
     No statistics are fit, so it's safe to run before the train/test split.
  2. `build_preprocessor` — returns an unfit sklearn ColumnTransformer
     (imputation + scaling + encoding). Callers MUST `.fit` this on the
     training split only, then `.transform` both splits, to avoid leakage.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import ID_LIKE_COLS, TARGET_COL

DATE_COLS = ["policy_bind_date", "incident_date"]

NUMERIC_COLS = [
    "months_as_customer",
    "age",
    "policy_deductable",
    "policy_annual_premium",
    "umbrella_limit",
    "capital-gains",
    "capital-loss",
    "incident_hour_of_the_day",
    "number_of_vehicles_involved",
    "bodily_injuries",
    "witnesses",
    "total_claim_amount",
    "injury_claim",
    "property_claim",
    "vehicle_claim",
    "auto_year",
    # engineered
    "days_to_incident",
    "claim_to_premium_ratio",
    "injury_claim_ratio",
    "property_claim_ratio",
    "vehicle_claim_ratio",
    "claim_per_vehicle",
]

ONEHOT_COLS = [
    "policy_state",
    "policy_csl",
    "insured_sex",
    "insured_education_level",
    "insured_relationship",
    "incident_type",
    "collision_type",
    "incident_severity",
    "authorities_contacted",
    "incident_state",
    "incident_city",
    "property_damage",
    "police_report_available",
]

FREQUENCY_COLS = [
    "insured_occupation",
    "insured_hobbies",
    "auto_make",
    "auto_model",
]


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encodes each category as its relative frequency in the fit data.

    Categories unseen at fit time (e.g. appearing only in the test split)
    map to 0.0 rather than raising, so this is safe to use with strict
    train-only fitting.
    """

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.freq_maps_ = [X[col].value_counts(normalize=True).to_dict() for col in X.columns]
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        out = np.zeros(X.shape, dtype=float)
        for i, col in enumerate(X.columns):
            out[:, i] = X[col].map(self.freq_maps_[i]).fillna(0.0).to_numpy()
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_freq" for c in (input_features or [])])


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add row-wise derived features and drop identifier/raw-date columns."""
    df = df.copy()

    df["days_to_incident"] = (df["incident_date"] - df["policy_bind_date"]).dt.days

    # Guard against divide-by-zero on the (rare) zero-premium/zero-vehicle rows.
    df["claim_to_premium_ratio"] = df["total_claim_amount"] / df["policy_annual_premium"].replace(0, np.nan)
    df["injury_claim_ratio"] = df["injury_claim"] / df["total_claim_amount"].replace(0, np.nan)
    df["property_claim_ratio"] = df["property_claim"] / df["total_claim_amount"].replace(0, np.nan)
    df["vehicle_claim_ratio"] = df["vehicle_claim"] / df["total_claim_amount"].replace(0, np.nan)
    df["claim_per_vehicle"] = df["total_claim_amount"] / df["number_of_vehicles_involved"].replace(0, np.nan)

    drop_cols = [c for c in ID_LIKE_COLS + DATE_COLS if c in df.columns]
    df = df.drop(columns=drop_cols)

    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    onehot_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])

    frequency_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", FrequencyEncoder()),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_COLS),
        ("onehot", onehot_pipeline, ONEHOT_COLS),
        ("frequency", frequency_pipeline, FREQUENCY_COLS),
    ])


def get_preprocessed_feature_names(preprocessor: ColumnTransformer) -> list:
    """Human-readable output feature names, valid only after `preprocessor.fit`."""
    names = []
    names.extend(NUMERIC_COLS)
    onehot_encoder = preprocessor.named_transformers_["onehot"].named_steps["encode"]
    names.extend(onehot_encoder.get_feature_names_out(ONEHOT_COLS).tolist())
    names.extend([f"{c}_freq" for c in FREQUENCY_COLS])
    return names


def split_feature_target(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y
