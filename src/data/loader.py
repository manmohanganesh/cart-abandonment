from pathlib import Path

import pandas as pd

from src.config import Config


def load_raw_data(config: Config) -> pd.DataFrame:
    raw_path = config.get("data.raw_path", "data/raw_data/online_shoppers_intention.csv")

    path = Path(raw_path)

    if not path.exists():
        raise FileNotFoundError(f"Raw data not found at: {raw_path}")

    df = pd.read_csv(path)
    return df