import streamlit as st

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import load_config

config = load_config()

page_config = config.get("streamlit.page_config", {})
st.set_page_config(
    page_title=page_config.get("page_title", "Cart Abandonment AI"),
    layout=page_config.get("layout", "wide"),
    initial_sidebar_state=page_config.get("initial_sidebar_state", "expanded"),
)

st.title("🛒 Cart Abandonment Intelligence System")

st.markdown(
    """
### Multi-layer AI system:
- Prediction
- Explainability + Diagnosis
- Strategy + Intervention

Use the sidebar to navigate.
"""
)