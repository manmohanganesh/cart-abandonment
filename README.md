# 🛒 Cart Abandonment Intelligence System

## 🚀 Overview

Most e-commerce platforms handle cart abandonment with generic reminders like *“You left something in your cart.”*

This project rethinks the problem as a **decision intelligence system** that:

* Predicts abandonment **before it happens**
* Explains **why** a user is likely to abandon
* Simulates **intervention strategies**
* Generates **personalized recovery messages**

---

## 🧠 Core Idea

> Move from *prediction* → *explanation* → *action*

Instead of sending generic notifications, this system enables **context-aware, data-driven interventions**.

---

## 🧱 System Architecture

```
User Behavior Data
→ Feature Engineering
→ ML Model (XGBoost)
→ SHAP Explainability
→ Decision Layer
→ Intervention Simulation
→ LLM Personalization
→ Streamlit Dashboard
```

---

## ⚙️ Features

### 🔮 Predictive Modeling

* XGBoost classifier trained on session-level behavior data
* Threshold tuning (0.3) based on business trade-offs
* Focus on **recall** to minimize missed high-intent users

---

### 🔍 Explainability (SHAP)

* Global + local explanations
* Identifies key drivers per user:

  * Price sensitivity
  * Engagement metrics
  * Exit behavior

---

### 🧠 Decision Intelligence Layer

* Converts predictions into **actionable decisions**
* Determines whether intervention is required

---

### 🧪 What-if Simulation

* Simulate interventions (e.g., discounts)
* Compare:

  * Before vs After prediction
* Estimate effectiveness of actions

---

### 📈 Intervention Impact Estimation

* Quantifies expected improvement
* Avoids unnecessary or ineffective interventions

---

### 🤖 LLM-Based Personalization

* Uses Gemini API to generate recovery messages
* Context-aware messaging based on SHAP insights
* Supports multiple strategies:

  * Discount-based
  * Urgency-based
  * Value-based

---

### 🎯 Confidence Scoring

* Measures model certainty
* Enables fallback strategies for uncertain predictions

---

### 🎨 Interactive Dashboard

* Built using Streamlit
* Card-based UI for clean visualization
* Displays:

  * Prediction
  * Explanation
  * Simulation
  * Recommended action

---

## 📊 Example Output

* **Prediction:** 0.72 (High abandonment risk)
* **Key Drivers:** Price sensitivity, low engagement
* **Simulation:** Discount reduces risk to 0.58
* **Decision:** Trigger intervention
* **Output:** Personalized recovery message

---

## 🧪 Tech Stack

* Python
* XGBoost
* SHAP
* Streamlit
* Google Gemini API (LLM)
* Pandas, Scikit-learn

---

## 🚀 How to Run

```bash
git clone <https://github.com/manmohanganesh/cart-abandonment.git>
cd cart-abandonment-system

pip install -r requirements.txt
```

Create a `.env` file:

```
GEMINI_API_KEY=your_api_key_here
```

Run the pipeline:

```bash
python -m src.main
```

Launch the app:

```bash
streamlit run app/streamlit_app.py
```

---

## 🧠 Key Learnings

* ML systems must align with **business objectives**, not just accuracy
* Explainability enables **trust and actionable insights**
* Prediction alone is insufficient — systems must **decide and act**
* Combining ML + LLM unlocks **intelligent automation**

---

## 🔮 Future Improvements

* Real-time streaming inference
* A/B testing for intervention strategies
* Reinforcement learning for optimal decision policies
* Cloud deployment (AWS / GCP)

---

## 💼 Author

Built as a portfolio project to demonstrate **end-to-end ML system design**, combining prediction, explainability, and AI-driven personalization.
