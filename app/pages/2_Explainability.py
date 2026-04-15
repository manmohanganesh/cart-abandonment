import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import streamlit as st

from src.models.predict import load_model, predict
from src.models.explain import compute_shap_values, get_shap_dict, plot_shap_bar
from src.rag.ingest import load_documents, create_vector_store
from src.rag.retriever import retrieve_context
from src.rag.chain import generate_rag_answer

@st.cache_resource
def setup():
    model, columns = load_model()
    docs = load_documents("data/external/reviews.txt")
    vectorstore = create_vector_store(docs)
    return model, columns, vectorstore

model, columns, vectorstore = setup()

st.title("🔍 Explainability & Diagnosis")

# Minimal input
price_sensitivity = st.slider("Price Sensitivity", 0.0, 1.0, 0.5)

input_df = pd.DataFrame({
    "Administrative": [2],
    "Informational": [1],
    "ProductRelated": [10],
    "Administrative_Duration": [50],
    "Informational_Duration": [30],
    "ProductRelated_Duration": [300],
    "BounceRates": [0.2],
    "ExitRates": [0.3],
    "PageValues": [10],
    "price_sensitivity": [price_sensitivity],
})

if st.button("Explain"):

    pred, processed_df = predict(model, columns, input_df)

    shap_values = compute_shap_values(model, processed_df)
    shap_dict = get_shap_dict(shap_values, processed_df)

    st.pyplot(plot_shap_bar(shap_values))

    # RAG
    query = "Why do users abandon due to price sensitivity?"
    context = retrieve_context(vectorstore, query)
    answer = generate_rag_answer(context, query)

    st.markdown("### Diagnosis")
    st.info(answer)