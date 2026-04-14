import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import streamlit as st

from src.config import load_config
from src.env import load_env
from src.models.explain import compute_shap_values, get_shap_dict, plot_shap_bar
from src.models.predict import load_model, predict
from src.recovery.generator import generate_message_llm

# --- Setup ---
load_env()
config = load_config()

@st.cache_resource
def load_model_cached():
    return load_model()

model, columns = load_model_cached()

# --- Styling ---
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}

h1, h2, h3 {
    color: #FAFAFA;
}

.block-container {
    padding-top: 2rem;
}

.card {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0px 0px 15px rgba(0,0,0,0.4);
}

.stMetric {
    background-color: #1E1E1E;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
# 🛒 Cart Abandonment Intelligence  
### Predict → Explain → Simulate → Act  
Turn user behavior into actionable decisions using ML + AI
""")

# --- Input Creator ---
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

# --- Layout ---
col1, col2 = st.columns(2)

# ================= INPUT =================
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📥 User Session Input")

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

    st.markdown("### 🧪 What-if Simulation")
    discount = st.slider("Simulated Discount (%)", 0, 50, 0)

    st.markdown('</div>', unsafe_allow_html=True)

# ================= OUTPUT =================
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Prediction & Insights")

    if st.button("Predict"):

        input_df = create_input_df()

        # --- Original prediction ---
        original_pred, processed_df = predict(model, columns, input_df)

        # --- Simulation ---
        simulated_df = input_df.copy()
        if discount > 0:
            simulated_df["price_sensitivity"] *= (1 - discount / 100)

        simulated_pred, _ = predict(model, columns, simulated_df)

        # --- Prediction Section ---
        st.markdown("### 📊 Prediction")

        colA, colB = st.columns([2,1])

        with colA:
            st.metric("Abandonment Probability", f"{original_pred:.2f}")
            st.progress(float(original_pred))

        with colB:
            confidence = abs(original_pred - 0.5) * 2
            st.metric("Confidence", f"{confidence:.2f}")

        # --- Risk ---
        if original_pred > 0.7:
            st.error("⚠️ High risk of abandonment")
        elif original_pred > 0.4:
            st.warning("⚠️ Medium risk of abandonment")
        else:
            st.success("✅ Low risk of abandonment")

        # --- Decision ---
        st.markdown("### 🧠 Decision")
        if original_pred > 0.3:
            st.success("🎯 Intervention recommended")
        else:
            st.info("No intervention needed")

        st.divider()

        # --- Impact ---
        st.markdown("### 📈 Intervention Impact")

        colX, colY = st.columns(2)

        with colX:
            st.metric("Before", f"{original_pred:.2f}")

        with colY:
            st.metric("After", f"{simulated_pred:.2f}")

        delta = original_pred - simulated_pred
        if delta > 0:
            st.success(f"Improvement: {delta:.2f}")
        else:
            st.warning("Intervention not effective")

        st.divider()

        # --- SHAP ---
        st.markdown("### 🔍 Why is the user likely to abandon?")

        shap_values = compute_shap_values(model, processed_df)
        shap_dict = get_shap_dict(shap_values, processed_df)

        fig = plot_shap_bar(shap_values)
        st.pyplot(fig)

        st.markdown("### 🔍 Key Drivers")
        for feature, value in shap_dict.items():
            direction = "↑ increases abandonment" if value > 0 else "↓ reduces abandonment"
            st.write(f"**{feature}**: {direction}")

        # --- Segment Insight ---
        if shap_dict.get("price_sensitivity", 0) > 0:
            st.info("💡 User is price-sensitive")

        st.divider()

        # --- Recommendation ---
        message = generate_message_llm(shap_dict, config)

        st.markdown("""
        <div class="card">
        <h4>💡 Recommended Action</h4>
        <p>{}</p>
        </div>
        """.format(message), unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)