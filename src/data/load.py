"""Load the raw insurance claims CSV into a clean-ish DataFrame."""
import pandas as pd

from src.config import RAW_DATA_PATH, TARGET_COL, TARGET_POSITIVE, MISSING_TOKENS


def load_raw_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV, normalize missing-value tokens, and binarize the target."""
    df = pd.read_csv(path, na_values=MISSING_TOKENS)

    # Drop a trailing unnamed/empty column if the source export includes one.
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
    df = df.drop(columns=unnamed_cols)

    if df[TARGET_COL].dtype == object:
        df[TARGET_COL] = (df[TARGET_COL] == TARGET_POSITIVE).astype(int)

    return df
