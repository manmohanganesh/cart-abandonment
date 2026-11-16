import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_data(df: pd.DataFrame):
    df = df.dropna().copy()  # Drops NaNs across all columns (including target & categoricals) upfront

    # Target Variable
    y = df['Revenue'].astype(int)

    # Dropping target from features & one-hot encoding
    X = pd.get_dummies(df.drop(columns=['Revenue']), drop_first=True)

    # Train-test split (now stratified on exact surviving rows)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit scaler ONLY on training data to prevent leakage
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    return X_train, X_test, y_train, y_test