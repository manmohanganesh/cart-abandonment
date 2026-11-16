import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import streamlit as st

from src.config import load_config
from src.models.explain import compute_shap_values, get_shap_dict, plot_shap_bar
from src.models.predict import predict
from src.rag.chain import generate_rag_answer
from src.services import get_config_cached, get_model_cached, get_vectorstore_cached, get_rag_context

config = get_config_cached()
model, columns = get_model_cached()
vectorstore = get_vectorstore_cached()

st.title("🔍 Explainability & Diagnosis")

pred_defaults = config.get("streamlit.prediction_defaults", {})

input_df = pd.DataFrame(
    {
        "Administrative": [pred_defaults.get("administrative", 2)],
        "Informational": [pred_defaults.get("informational", 1)],
        "ProductRelated": [pred_defaults.get("product_related", 10)],
        "Administrative_Duration": [pred_defaults.get("administrative_duration", 50)],
        "Informational_Duration": [pred_defaults.get("informational_duration", 30)],
        "ProductRelated_Duration": [pred_defaults.get("product_related_duration", 300)],
        "BounceRates": [pred_defaults.get("bounce_rates", 0.2)],
        "ExitRates": [pred_defaults.get("exit_rates", 0.3)],
    }
)

if st.button("Explain"):
    pred, processed_df = predict(model, columns, input_df)
    top_n = config.get("model.evaluation.shap_top_n", 5)
    shap_values = compute_shap_values(model, processed_df)
    shap_dict = get_shap_dict(shap_values, processed_df, top_n=top_n)

    st.markdown(f"**Abandonment Probability: {100 * pred:.2f}%**")
    st.pyplot(plot_shap_bar(shap_values))

    # RAG - dynamic query based on top SHAP feature
    top_feature = list(shap_dict.keys())[0] if shap_dict else "engagement"
    query = f"Why do users abandon due to {top_feature}?"
    k = config.get("rag.retriever.k", 3)
    context = get_rag_context(query, k=k)
    answer = generate_rag_answer(context, query, config)

    st.markdown("### Diagnosis")
    st.info(answer)
    
    # Show top SHAP features
    st.markdown("### Top Behavioral Drivers")
    for feature, value in shap_dict.items():
        direction = "↑ increases risk" if value > 0 else "↓ decreases risk"
        st.write(f"- **{feature}**: {value:.4f} ({direction})")