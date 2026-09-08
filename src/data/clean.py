"""Structural cleaning: dedupe and fix impossible values.

Deliberately does NOT impute using dataset-wide statistics (median/mode) --
that fitting step is left to `features.py`'s ColumnTransformer, which is
fit on the training split only. Doing median/mode imputation here, before
the train/test split, would leak test-set information into training.
"""
import numpy as np
import pandas as pd

DATE_COLS = ["policy_bind_date", "incident_date"]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates()
    n_dupes = before - len(df)
    if n_dupes:
        df = df.reset_index(drop=True)

    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Ages can't be negative or implausibly large for a policyholder.
    if "age" in df.columns:
        df.loc[(df["age"] < 0) | (df["age"] > 100), "age"] = np.nan

    # umbrella_limit is a coverage cap; it cannot be negative.
    if "umbrella_limit" in df.columns:
        df.loc[df["umbrella_limit"] < 0, "umbrella_limit"] = np.nan

    # An incident can't occur before the policy was bound.
    if set(DATE_COLS).issubset(df.columns):
        impossible = df["incident_date"] < df["policy_bind_date"]
        df.loc[impossible, "incident_date"] = pd.NaT

    return df
