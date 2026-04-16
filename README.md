# Cart Abandonment Intelligence System

## Overview

Most e-commerce systems handle cart abandonment with generic reminders like “You left something in your cart.”

This project goes beyond that by building a decision intelligence system that:

* Predicts abandonment before it happens
* Explains why each user is likely to abandon
* Diagnoses behavioral patterns using RAG
* Simulates intervention strategies
* Generates personalized recovery messages

---

## Key Idea

Don’t just predict abandonment — understand it and act on it intelligently.

---

## System Architecture

```
User Session Data
→ Feature Engineering
→ ML Model (XGBoost)
→ SHAP Explainability (Why THIS user)
→ RAG Diagnosis (Why USERS LIKE THIS)
→ Decision Layer
→ What-if Simulation
→ LLM Personalization
→ Multi-page Streamlit Dashboard
```

---

## Features

### 1. Predictive Modeling

* XGBoost classifier trained on session behavior
* Threshold tuning focused on business impact
* Optimized for high recall to minimize lost conversions

---

### 2. Explainability (SHAP)

* Global and local feature importance
* Per-user reasoning
* Identifies drivers like:

  * Price sensitivity
  * Engagement levels
  * Exit behavior

---

### 3. Diagnosis with RAG (Retrieval-Augmented Generation)

* Built a RAG pipeline over simulated user feedback (reviews, complaints)
* Uses embeddings and vector search (ChromaDB)
* Retrieves relevant behavioral insights

Example:

* “Why do users with high price sensitivity abandon?”

Combines:

* SHAP → individual reasoning
* RAG → population-level reasoning

---

### 4. Decision Intelligence Layer

* Converts predictions into actionable decisions
* Determines whether to intervene and urgency level

---

### 5. What-if Simulation Engine

* Simulates interventions such as discounts
* Compares before vs after predictions
* Estimates effectiveness of actions

---

### 6. Intervention Impact Estimation

* Quantifies improvement after intervention
* Helps avoid unnecessary or ineffective actions

---

### 7. LLM-Powered Personalization

* Uses Gemini API to generate recovery messages
* Context-aware messaging based on SHAP insights
* Adapts strategy based on user behavior

---

### 8. Multi-Page Interactive Dashboard

Organized into three dashboards:

Prediction Dashboard

* User session input
* Abandonment probability
* What-if simulation

Explainability & Diagnosis

* SHAP-based reasoning
* RAG-powered insights

Strategy & Intervention

* LLM-generated personalized actions

---

## Example Output

* Prediction: 0.72 (High risk)
* Key Drivers: Price sensitivity, high bounce rate
* RAG Insight: Users find prices high and discounts unattractive
* Simulation: Discount reduces risk to 0.58
* Action: Trigger intervention
* Message: Personalized recovery strategy

---

## Tech Stack

* Python
* XGBoost
* SHAP
* LangChain
* ChromaDB
* HuggingFace Embeddings
* Gemini API
* Pandas / Scikit-learn
* Streamlit

---

## Key Learnings

* ML must align with business objectives, not just accuracy
* Explainability is critical for trust and actionability
* Prediction alone is insufficient — systems must decide and act
* Combining ML, RAG, and LLM enables context-aware intelligence
* Simulation helps estimate real-world impact before execution

---

## How to Run

```
git clone <repo>
cd cart-abandonment-system

pip install -r requirements.txt

# Add API keys
touch .env
# GEMINI_API_KEY=your_key
# HF_TOKEN=optional

python -m src.main
streamlit run app/streamlit_app.py
```

---

## Project Structure

```
cart-abandonment-system/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── src/
│   ├── data/
│   ├── models/
│   ├── rag/
│   └── recovery/
│
├── app/
│   ├── streamlit_app.py
│   └── pages/
│       ├── 1_Prediction.py
│       ├── 2_Explainability.py
│       └── 3_Strategy.py
│
├── configs/
├── tests/
├── requirements.txt
└── README.md
```

---

## Resume Impact

Built an end-to-end cart abandonment intelligence system combining XGBoost, SHAP explainability, RAG-based behavioral diagnosis, and LLM-driven personalization, enabling prediction, reasoning, simulation, and actionable intervention.

---

## Interview Pitch

Most systems treat cart abandonment as a notification problem. This system approaches it as a decision intelligence problem. It predicts abandonment using XGBoost, explains why using SHAP, diagnoses behavioral patterns using RAG, simulates interventions like discounts, and generates personalized recovery messages using an LLM. The focus is on moving from prediction to action through context-aware decisions tailored to each user.

---

## Future Improvements

* Real-time data pipeline integration
* A/B testing for intervention strategies
* Reinforcement learning for optimal decision-making
* Deployment (AWS / GCP)
* API layer for production use

---

## Final Thought

This project demonstrates the shift from:

Prediction → Decision Intelligence → Actionable AI Systems
