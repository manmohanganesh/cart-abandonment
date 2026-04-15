import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st

from src.config import load_config
from src.env import load_env
from src.recovery.generator import generate_message_llm

load_env()
config = load_config()

st.title("Strategy & Intervention")

price_sensitivity = st.slider("Price Sensitivity", 0.0, 1.0, 0.7)

shap_dict = {
    "price_sensitivity": price_sensitivity,
    "BounceRates": 0.3,
}

if st.button("Generate Strategy"):
    message = generate_message_llm(shap_dict, config)
    st.success(message)