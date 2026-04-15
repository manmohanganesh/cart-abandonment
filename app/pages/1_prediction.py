import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import streamlit as st

from src.models.predict import load_model, predict

@st.cache_resource
def load_model_cached():
    return load_model()

model, columns = load_model_cached()

st.title("Prediction Dashboard")

def create_input_df():
    return pd.DataFrame({
        "Administrative": [administrative],
        "Informational": [informational],
        "ProductRelated": [product_related],
        "Administrative_Duration": [administrative_duration],
        "Informational_Duration": [informational_duration],
        "ProductRelated_Duration": [product_related_duration],
        "BounceRates": [bounce_rates],
        "ExitRates": [exit_rates],
        "PageValues": [page_values],
        "price_sensitivity": [price_sensitivity],
    })

# Inputs
administrative = st.slider("Administrative Pages", 0, 20, 2)
informational = st.slider("Informational Pages", 0, 20, 1)
product_related = st.slider("Product Related Pages", 0, 50, 10)

administrative_duration = st.slider("Administrative Duration", 0, 500, 50)
informational_duration = st.slider("Informational Duration", 0, 500, 30)
product_related_duration = st.slider("Product Related Duration", 0, 2000, 300)

bounce_rates = st.slider("Bounce Rate", 0.0, 1.0, 0.2)
exit_rates = st.slider("Exit Rate", 0.0, 1.0, 0.3)

page_values = st.number_input("Page Value", 0.0, 100.0, 10.0)
price_sensitivity = st.slider("Price Sensitivity", 0.0, 1.0, 0.5)

discount = st.slider("Simulated Discount (%)", 0, 50, 0)

if st.button("Predict"):
    input_df = create_input_df()

    original_pred, _ = predict(model, columns, input_df)

    simulated_df = input_df.copy()
    simulated_df["price_sensitivity"] *= (1 - discount / 100)
    simulated_pred, _ = predict(model, columns, simulated_df)

    st.metric("Abandonment Probability", f"{original_pred:.2f}")

    st.markdown("### Impact")
    st.write(f"Before: {original_pred:.2f}")
    st.write(f"After: {simulated_pred:.2f}")