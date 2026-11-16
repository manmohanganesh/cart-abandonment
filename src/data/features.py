import numpy as np
import pandas as pd

from src.config import Config


def basic_feature_engineering(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    df = df.copy()

    # Total pages viewed
    df["Total_pages"] = (
        df["Administrative"]
        + df["Informational"]
        + df["ProductRelated"]
    )

    # Total time spent
    df["total_duration"] = (
        df["Administrative_Duration"]
        + df["Informational_Duration"]
        + df["ProductRelated_Duration"]
    )

    df["engagement_score"] = df["Total_pages"] * df["total_duration"]

    df["avg_time_per_page"] = df["total_duration"] / df["Total_pages"].replace(0, 1)

    low_eng_pages = config.get("data.feature_engineering.low_engagement_pages_threshold", 3)
    low_eng_duration = config.get("data.feature_engineering.low_engagement_duration_threshold", 60)
    df["is_low_engagement"] = (
        (df["Total_pages"] < low_eng_pages) & (df["total_duration"] < low_eng_duration)
    ).astype(int)

    # Session stage (early vs mid vs late session based on pages viewed)
    bins = config.get("data.feature_engineering.session_stage_bins", [-1, 3, 10, 999999])
    labels = config.get("data.feature_engineering.session_stage_labels", ["early", "mid", "late"])
    df["session_stage"] = pd.cut(
        df["Total_pages"],
        bins=bins,
        labels=labels,
    )
    return df