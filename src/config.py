"""Paths, constants, and random seed shared across the pipeline."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "insurance_claims.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"

MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
SHAP_PLOTS_DIR = RESULTS_DIR / "shap_plots"

TARGET_COL = "fraud_reported"
TARGET_POSITIVE = "Y"

RANDOM_SEED = 42
TEST_SIZE = 0.2

# Columns that are pure identifiers/leakage risks, dropped during feature engineering.
ID_LIKE_COLS = [
    "policy_number",
    "insured_zip",
    "incident_location",
]

# String tokens used in the raw export in place of a real missing value.
MISSING_TOKENS = ["?"]
