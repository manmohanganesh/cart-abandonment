from pathlib import Path

import joblib
from xgboost import XGBClassifier

from src.config import Config


def train_model(X_train, y_train, config: Config):
    xgb_config = config.get("model.xgboost", {})

    scale_pos_weight = xgb_config.get("scale_pos_weight")
    if scale_pos_weight is None:
        # Auto-compute from class ratio
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    model = XGBClassifier(
        n_estimators=xgb_config.get("n_estimators", 100),
        max_depth=xgb_config.get("max_depth", 5),
        learning_rate=xgb_config.get("learning_rate", 0.1),
        subsample=xgb_config.get("subsample", 0.8),
        colsample_bytree=xgb_config.get("colsample_bytree", 0.8),
        random_state=xgb_config.get("random_state", 42),
        eval_metric=xgb_config.get("eval_metric", "logloss"),
        scale_pos_weight=scale_pos_weight,
    )

    model.fit(X_train, y_train)

    # SAVE MODEL
    model_dir = Path(config.get("paths.model_dir", "models"))
    model_dir.mkdir(parents=True, exist_ok=True)

    model_file = config.get("paths.model_file", "model.pkl")
    columns_file = config.get("paths.columns_file", "columns.pkl")

    joblib.dump(model, model_dir / model_file)
    joblib.dump(X_train.columns, model_dir / columns_file)

    return model