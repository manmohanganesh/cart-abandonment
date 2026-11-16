import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import pandas as pd

from src.config import load_config
from src.data.features import basic_feature_engineering

config = load_config()


def load_model():
    model_dir = Path(config.get("paths.model_dir", "models"))
    model_file = config.get("paths.model_file", "model.pkl")
    columns_file = config.get("paths.columns_file", "columns.pkl")

    model = joblib.load(model_dir / model_file)
    columns = joblib.load(model_dir / columns_file)
    return model, columns


def preprocess_input(input_df, columns):
    # Feature Engineering
    df = basic_feature_engineering(input_df, config)

    # Encoding
    df = pd.get_dummies(df, drop_first=True)

    # Align Columns
    df = df.reindex(columns=columns, fill_value=0)

    return df


def predict(model, columns, input_df):
    processed_df = preprocess_input(input_df, columns)
    prediction = model.predict_proba(processed_df)[0][1]
    return prediction, processed_df
