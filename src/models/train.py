import joblib
from xgboost import XGBClassifier

def train_model(X_train,y_train):
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
    )

    model.fit(X_train,y_train)

    # SAVE MODEL
    joblib.dump(model, "models/model.pkl")
    joblib.dump(X_train.columns, "models/columns.pkl")

    return model