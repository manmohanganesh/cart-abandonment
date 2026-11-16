import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import streamlit as st

from src.config import load_config
from src.models.predict import load_model, predict
from src.services import get_config_cached, get_model_cached

config = get_config_cached()
model, columns = get_model_cached()

st.title("Prediction Dashboard")

# Inputs
pred_defaults = config.get("streamlit.prediction_defaults", {})
slider_ranges = config.get("streamlit.slider_ranges", {})

administrative = st.slider(
    "Administrative Pages",
    slider_ranges.get("administrative", [0, 20])[0],
    slider_ranges.get("administrative", [0, 20])[1],
    pred_defaults.get("administrative", 2),
)
informational = st.slider(
    "Informational Pages",
    slider_ranges.get("informational", [0, 20])[0],
    slider_ranges.get("informational", [0, 20])[1],
    pred_defaults.get("informational", 1),
)
product_related = st.slider(
    "Product Related Pages",
    slider_ranges.get("product_related", [0, 50])[0],
    slider_ranges.get("product_related", [0, 50])[1],
    pred_defaults.get("product_related", 10),
)

administrative_duration = st.slider(
    "Administrative Duration",
    slider_ranges.get("administrative_duration", [0, 500])[0],
    slider_ranges.get("administrative_duration", [0, 500])[1],
    pred_defaults.get("administrative_duration", 50),
)
informational_duration = st.slider(
    "Informational Duration",
    slider_ranges.get("informational_duration", [0, 500])[0],
    slider_ranges.get("informational_duration", [0, 500])[1],
    pred_defaults.get("informational_duration", 30),
)
product_related_duration = st.slider(
    "Product Related Duration",
    slider_ranges.get("product_related_duration", [0, 2000])[0],
    slider_ranges.get("product_related_duration", [0, 2000])[1],
    pred_defaults.get("product_related_duration", 300),
)

bounce_rates = st.slider(
    "Bounce Rate",
    slider_ranges.get("bounce_rates", [0.0, 1.0])[0],
    slider_ranges.get("bounce_rates", [0.0, 1.0])[1],
    pred_defaults.get("bounce_rates", 0.2),
)
exit_rates = st.slider(
    "Exit Rate",
    slider_ranges.get("exit_rates", [0.0, 1.0])[0],
    slider_ranges.get("exit_rates", [0.0, 1.0])[1],
    pred_defaults.get("exit_rates", 0.3),
)

discount = st.slider(
    "Simulated Discount (%)",
    slider_ranges.get("discount", [0, 50])[0],
    slider_ranges.get("discount", [0, 50])[1],
    pred_defaults.get("discount", 0),
)


def create_input_df():
    return pd.DataFrame(
        {
            "Administrative": [administrative],
            "Informational": [informational],
            "ProductRelated": [product_related],
            "Administrative_Duration": [administrative_duration],
            "Informational_Duration": [informational_duration],
            "ProductRelated_Duration": [product_related_duration],
            "BounceRates": [bounce_rates],
            "ExitRates": [exit_rates],
        }
    )


if st.button("Predict"):
    input_df = create_input_df()

    original_pred, _ = predict(model, columns, input_df)

    st.markdown("### Abandonment Probability")
    st.write(f"{100 * original_pred:.2f}%")
    
    st.info("💡 Note: PageValues removed (leakage). Discount simulation via A/B test "
            "(`python -m src.main`).")
