import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import Config


def prepare_data(df: pd.DataFrame, config: Config):
    df = df.dropna().copy()

    # Target Variable
    y = df["Revenue"].astype(int)

    # Drop PageValues - leakage proxy (avg page value including purchase pages)
    # Not available at prediction time for non-purchasing sessions
    features_df = df.drop(columns=["Revenue", "PageValues"], errors="ignore")

    # One-hot encoding
    X = pd.get_dummies(features_df, drop_first=True)

    # Three-way split: train / val / test
    test_size = config.get("data.test_size", 0.2)
    val_size = config.get("data.val_size", 0.1)
    random_state = config.get("data.random_state", 42)
    stratify = config.get("data.stratify", True)
    stratify_col = y if stratify else None

    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_col
    )

    # Second split: train vs val (from train+val)
    val_size_adjusted = val_size / (1 - test_size)  # Adjust val size relative to train+val
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size_adjusted, random_state=random_state, stratify=y_trainval
    )

    # Fit scaler ONLY on training data to prevent leakage
    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_val = pd.DataFrame(
        scaler.transform(X_val), columns=X_val.columns, index=X_val.index
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    return X_train, X_val, X_test, y_train, y_val, y_test