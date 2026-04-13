import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_data(df: pd.DataFrame):
    df= df.copy()

    #Target Variable
    df['Revenue']=df['Revenue'].astype(int)

    y=df['Revenue']

    #Dropping target from features
    X=df.drop(columns=['Revenue'])

    #Handle categorical variables
    X=pd.get_dummies(X,drop_first=True)

    #Train-test split
    X_train,X_test,y_train,y_test=train_test_split(
        X,y,test_size=0.2,random_state=42,stratify=y
    )

    X_train=X_train.dropna()
    y_train=y_train[X_train.index]
    X_test=X_test.dropna()
    y_test=y_test[X_test.index]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train,X_test,y_train,y_test