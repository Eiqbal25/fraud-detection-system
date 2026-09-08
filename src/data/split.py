"""Stratified train/test split and persistence to data/processed/."""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DIR, RANDOM_SEED, TARGET_COL, TEST_PATH, TEST_SIZE, TRAIN_PATH


def make_split(df: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_SEED):
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COL],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_split(train_df: pd.DataFrame, test_df: pd.DataFrame):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)


def build_processed_dataset():
    """Full pipeline: load raw -> clean -> engineer -> stratified split -> save."""
    from src.data.clean import clean_data
    from src.data.features import engineer_features
    from src.data.load import load_raw_data

    df = load_raw_data()
    df = clean_data(df)
    df = engineer_features(df)
    train_df, test_df = make_split(df)
    save_split(train_df, test_df)
    return train_df, test_df


if __name__ == "__main__":
    train_df, test_df = build_processed_dataset()
    print(f"train: {train_df.shape}, test: {test_df.shape}")
    print(f"train fraud rate: {train_df['fraud_reported'].mean():.3f}")
    print(f"test fraud rate: {test_df['fraud_reported'].mean():.3f}")
