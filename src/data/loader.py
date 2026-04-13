import pandas as pd
from pathlib import Path

def load_raw_data(config:dict) ->pd.DataFrame:
    raw_path = config['data']['raw_path']

    path=Path(raw_path)

    if not path.exists():
        raise FileNotFoundError(f"Raw data not found at:{raw_path}")
    
    df=pd.read_csv(path)
    return df