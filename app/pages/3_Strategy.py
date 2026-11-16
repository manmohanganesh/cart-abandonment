import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st

from src.config import load_config
from src.env import load_env
from src.recovery.generator import generate_message_llm
from src.services import get_config_cached

load_env()
config = get_config_cached()

st.title("Strategy & Intervention")

st.markdown("Enter key behavioral signals to generate personalized recovery strategies:")

col1, col2 = st.columns(2)
with col1:
    bounce_rate = st.slider("Bounce Rate", 0.0, 1.0, 0.3, 0.05)
    exit_rate = st.slider("Exit Rate", 0.0, 1.0, 0.3, 0.05)
with col2:
    product_pages = st.slider("Product Pages Viewed", 0, 50, 10)
    product_duration = st.slider("Product Duration (sec)", 0, 2000, 300)

shap_dict = {
    "BounceRates": bounce_rate,
    "ExitRates": exit_rate,
    "ProductRelated": float(product_pages),
    "ProductRelated_Duration": float(product_duration),
    "is_low_engagement": 1.0 if (product_pages < 3 and product_duration < 60) else 0.0,
}

if st.button("Generate Strategy"):
    message = generate_message_llm(shap_dict, config)
    st.success(message)