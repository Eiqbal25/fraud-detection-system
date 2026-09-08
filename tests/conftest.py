import pandas as pd
import pytest

from src.data.clean import clean_data
from src.data.features import engineer_features
from src.data.load import load_raw_data
from src.data.split import make_split


@pytest.fixture(scope="session")
def raw_df():
    return load_raw_data()


@pytest.fixture(scope="session")
def engineered_df(raw_df):
    df = clean_data(raw_df)
    df = engineer_features(df)
    return df


@pytest.fixture(scope="session")
def train_test_split_dfs(engineered_df):
    return make_split(engineered_df)


@pytest.fixture
def sample_raw_rows():
    """A tiny synthetic frame with known values, for arithmetic/unit checks."""
    return pd.DataFrame({
        "policy_bind_date": pd.to_datetime(["2020-01-01", "2020-06-01"]),
        "incident_date": pd.to_datetime(["2020-01-11", "2020-06-11"]),
        "total_claim_amount": [1000.0, 500.0],
        "policy_annual_premium": [500.0, 0.0],
        "injury_claim": [200.0, 100.0],
        "property_claim": [300.0, 100.0],
        "vehicle_claim": [500.0, 300.0],
        "number_of_vehicles_involved": [2, 0],
        "policy_number": [1, 2],
        "insured_zip": [11111, 22222],
        "incident_location": ["a st", "b st"],
    })
