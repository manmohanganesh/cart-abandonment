# Cart Abandonment Intelligence System

## Overview

End-to-end ML system for e-commerce cart abandonment: **predict → explain → diagnose → intervene**.

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Prediction** | XGBoost (calibrated) | P(Revenue=1) from session behavior |
| **Explainability** | SHAP | Per-session feature contributions |
| **Diagnosis** | RAG (ChromaDB + Groq) | Natural-language driver explanations |
| **Intervention** | A/B test simulation | Discount/Urgency/Value strategy uplift |
| **Personalization** | LLM (Groq) | Context-aware recovery messages |
| **Dashboard** | Streamlit | 3-page interactive UI |

---

## Key Results (2026-11-16)

| Metric | Value |
|--------|-------|
| **AUC-ROC (uncalibrated)** | 0.7695 |
| **AUC-ROC (calibrated)** | 0.7117 |
| **Best threshold (val F1)** | 0.30 (F1=0.914) |
| **Test accuracy @ 0.30** | 82.24% |
| **Top SHAP drivers** | ExitRates (+0.45), Month_Mar (-0.28), Month_Nov (-0.25) |

**A/B Test Winner:** Discount 25% → **+$17.61/session uplift** (vs. Urgency +$17.04, Value -$0.05)

> **Full metrics, limitations, fairness checks, and known bugs:** [`model_card.md`](model_card.md)

---

## System Architecture

```
User Session Data
→ Feature Engineering (7 engineered features, NO simulated/leaky features)
→ XGBoost + Isotonic Calibration
→ SHAP Explainer (global + local)
→ RAG over User Reviews (ChromaDB persisted)
→ A/B Test Simulation (3 strategies × intensity levels)
→ LLM Recovery Message Generator (Groq)
→ Streamlit Dashboard (3 pages)
```

---

## ML Correctness Fixes Applied

- ❌ **Removed simulated features:** `scroll_depth` (was random), `price_sensitivity` (was PageValues proxy)
- ❌ **Removed leakage:** `PageValues` (post-purchase signal) dropped from features
- ✅ **Class imbalance:** `scale_pos_weight=auto` (~5.3:1)
- ✅ **Proper splits:** 70% train / 10% val / 20% test (stratified)
- ✅ **Threshold on validation:** F1-optimized, not test-set tuned
- ✅ **Calibration:** Isotonic regression on validation set
- ⚠️ **Known issue:** Model miscalibrated (predicts 73.7% conversion vs 15.5% actual) — needs more data / better calibration

---

## Quickstart

```bash
git clone <repo>
cd cart-abandonment-system

# Install dependencies
pip install -r requirements.txt

# Configure API keys (copy .env.example → .env)
cp .env.example .env
# Edit .env: GROQ_API_KEY=your_key

# Run training + evaluation + A/B simulation
python -m src.main

# Launch dashboard
streamlit run app/streamlit_app.py
```

**Pages:**
1. **Prediction** — session inputs → abandonment probability
2. **Explainability** — SHAP bar plot + RAG diagnosis
3. **Strategy** — behavioral inputs → LLM recovery messages

---

## Project Structure

```
cart-abandonment-system/
├── app/
│   ├── streamlit_app.py
│   └── pages/
│       ├── 1_Prediction.py
│       ├── 2_Explainability.py
│       └── 3_Strategy.py
├── configs/
│   └── config.yaml          # All hyperparams, paths, prompts
├── data/
│   ├── raw_data/
│   ├── external/reviews.txt
│   └── vectorstore/chroma_db/
├── models/
│   ├── model.pkl
│   └── columns.pkl
├── src/
│   ├── config.py            # Config class with dot notation
│   ├── services.py          # Shared singletons (model, vectorstore)
│   ├── data/
│   │   ├── loader.py
│   │   ├── features.py      # 7 engineered features
│   │   └── preprocess.py    # 70/10/20 split + scaling
│   ├── models/
│   │   ├── train.py         # XGBoost + class imbalance
│   │   ├── predict.py
│   │   ├── evaluate.py      # AUC, calibration, thresholds
│   │   └── explain.py       # SHAP
│   ├── rag/
│   │   ├── ingest.py        # ChromaDB (persisted)
│   │   ├── retriever.py
│   │   └── chain.py         # Groq LLM
│   ├── recovery/
│   │   └── generator.py     # Rule-based + LLM messages
│   ├── simulation/
│   │   └── ab_test.py       # Discount/Urgency/Value uplift
│   ├── main.py              # End-to-end pipeline
│   └── logger.py / env.py
├── model_card.md            # Model card (metrics, limits, fairness)
├── requirements.txt
└── README.md
```

---

## Tech Stack

- **ML:** XGBoost, scikit-learn, SHAP, isotonic calibration
- **RAG:** LangChain, ChromaDB (persisted), sentence-transformers
- **LLM:** Groq API (`openai/gpt-oss-120b`)
- **Dashboard:** Streamlit
- **Config:** YAML + typed `Config` class

---

## What This Demonstrates

- **ML Engineering:** Proper splits, calibration, leakage prevention, class imbalance
- **Explainability:** SHAP global/local + RAG population-level diagnosis
- **Decision Intelligence:** A/B simulation with revenue estimation
- **Production Thinking:** Config-driven, persisted vectorstore, shared services, model card
- **Honest Documentation:** Known bugs (miscalibration, simulation limits) in model card

---

## Future Work

- [ ] Better calibration (more data / Platt scaling comparison)
- [ ] Causal uplift modeling (not heuristic simulation)
- [ ] Real user feedback corpus (replace synthetic reviews)
- [ ] FastAPI service layer for production serving
- [ ] Drift detection & automated retraining pipeline
- [ ] A/B test framework for live intervention experiments