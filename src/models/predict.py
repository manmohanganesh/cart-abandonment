import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import joblib
from src.data.features import basic_feature_engineering

def load_model():
    model = joblib.load("models/model.pkl")
    columns = joblib.load("models/columns.pkl")
    return model,columns

def preprocess_input(input_df,columns):
    #Feature Engineering
    df = basic_feature_engineering(input_df)

    #Encoding
    df=pd.get_dummies(df,drop_first=True)

    #Align Columns
    df=df.reindex(columns=columns,fill_value=0)

    return df

def predict(model,columns,input_df):
    processed_df = preprocess_input(input_df,columns)
    prediction=model.predict_proba(processed_df)[0][1]
    return prediction,processed_df
