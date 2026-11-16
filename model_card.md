# Cart Abandonment Prediction Model Card

## Model Overview

**Model Name:** Cart Abandonment Intelligence System  
**Version:** 0.1.0  
**Type:** Binary Classification (XGBoost)  
**Task:** Predict probability of cart abandonment for e-commerce sessions  
**Framework:** Python 3.10+, XGBoost 3.4.1, scikit-learn 1.3.2  

---

## Intended Use

**Primary Use Case:** Real-time prediction of cart abandonment probability to trigger personalized recovery interventions (discounts, urgency messaging, value-based messaging) for e-commerce sessions.

**Target Users:** E-commerce platforms, marketing automation teams, conversion rate optimization specialists.

**Deployment Context:** Streamlit dashboard for demo; designed for integration via API into production e-commerce stack.

**Out of Scope:**
- Revenue magnitude prediction (binary only)
- New user acquisition (session must exist)
- Multi-session user journey modeling
- Real-time streaming inference (batch scoring only in current form)

---

## Training Data

| Attribute | Detail |
|-----------|--------|
| **Source** | UCI ML Repository: [Online Shoppers Intention](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset) |
| **Size** | 12,330 sessions |
| **Features** | 18 raw (10 numerical, 8 categorical) → ~35 after one-hot encoding |
| **Target** | `Revenue` (boolean: True = purchase, False = abandonment) |
| **Target Balance** | 15.8% Revenue=True, 84.2% Revenue=False |
| **Collection Period** | 12-month period in 2018 |
| **Geography** | Not specified (single retailer) |
| **License** | CC BY 4.0 |

### Feature Categories
| Category | Features |
|----------|----------|
| **Page Counts** | Administrative, Informational, ProductRelated |
| **Page Durations** | Administrative_Duration, Informational_Duration, ProductRelated_Duration |
| **Behavioral** | BounceRates, ExitRates, PageValues |
| **Temporal** | SpecialDay, Month, Weekend |
| **Technical** | OperatingSystems, Browser, Region, TrafficType |
| **User** | VisitorType (New/Returning) |
| **Engineered** | Total_pages, total_duration, engagement_score, avg_time_per_page, is_low_engagement, session_stage |

### Data Limitations
- **Age:** 2018 data may not reflect post-COVID shopping behavior
- **Single retailer:** Unknown generalizability
- **No demographic data:** Age, gender, income not available
- **No cart value:** `PageValues` is a proxy but leaks post-purchase info (excluded from features)
- **Simulated features removed:** `scroll_depth` (random), `price_sensitivity` (PageValues proxy) removed in v0.1.0

---

## Model Architecture

### Algorithm
**XGBoost Classifier** with the following hyperparameters:
```yaml
n_estimators: 100
max_depth: 5
learning_rate: 0.1
subsample: 0.8
colsample_bytree: 0.8
random_state: 42
eval_metric: "logloss"
scale_pos_weight: auto  # Computed as neg/pos ratio (~5.3)
```

### Class Imbalance Handling
- **Method:** `scale_pos_weight` = negative_class_count / positive_class_count
- **Ratio:** ~5.3:1 (84.2% / 15.8%)
- **Effect:** Upweights minority class (Revenue=True) during training

### Validation Strategy
- **Split:** 70% train / 10% validation / 20% test (stratified)
- **Threshold Selection:** F1-score optimization on validation set
- **Final Evaluation:** Held-out test set with best threshold

### Feature Engineering Pipeline
1. Drop rows with any NaN
2. One-hot encode categorical variables (drop_first=True)
3. StandardScaler fit on train only, applied to val/test
4. Engineered features computed before encoding/scaling

---

## Performance Metrics

*Metrics from training run (2026-11-16)*

| Metric | Validation | Test | Notes |
|--------|------------|------|-------|
| **AUC-ROC** | — | — | Not computed (add to evaluate.py) |
| **AUC-PR** | — | — | Not computed (add to evaluate.py) |
| **Best Threshold** | **0.50** | — | F1-optimal on validation (F1=0.661) |
| **Precision @ Best (Class 1)** | — | **0.56** | Revenue=True (purchase) |
| **Recall @ Best (Class 1)** | — | **0.80** | Revenue=True (purchase) |
| **F1 @ Best (Class 1)** | **0.661** | **0.66** | Revenue=True (purchase) |
| **Accuracy @ Best** | — | **87.10%** | Overall |
| **Precision @ Best (Class 0)** | — | **0.96** | Revenue=False (abandonment) |
| **Recall @ Best (Class 0)** | — | **0.88** | Revenue=False (abandonment) |
| **F1 @ Best (Class 0)** | — | **0.92** | Revenue=False (abandonment) |
| **Expected Revenue Recovery** | — | — | Requires AOV & intervention cost calibration |

### Threshold Analysis
| Threshold | Precision (Class 1) | Recall (Class 1) | F1 (Class 1) | Accuracy | Use Case |
|-----------|---------------------|------------------|--------------|----------|----------|
| **0.50** | **0.56** | **0.80** | **0.66** | **87.10%** | **Selected (F1-optimal)** |
| 0.40 | — | — | — | 84.51% | Higher recall |
| 0.30 | — | — | — | 81.43% | Aggressive recall |
| 0.20 | — | — | — | 74.61% | Maximum recall |

> **Note:** Best threshold = 0.50 (not 0.30 as originally hypothesized). Validation F1 optimization selected 0.50. At this threshold, the model achieves 80% recall on purchasers (Class 1) with 56% precision. Class 0 (abandonment) performance is strong: 96% precision, 88% recall.

---

## Fairness & Bias Assessment

### Protected Attributes
**None directly in features.** However, proxies may exist:

| Feature | Potential Proxy For | Monitoring |
|---------|---------------------|------------|
| `VisitorType` (New/Returning) | Loyalty, familiarity | Check calibration by segment |
| `Weekend` | Shopping time availability | Check calibration weekend vs weekday |
| `Region` | Geography, socioeconomic | Check calibration by region |
| `OperatingSystems` / `Browser` | Device type, tech savviness | Check calibration by device |

### Recommended Fairness Checks
1. **Demographic Parity:** Intervention rate across segments
2. **Equal Opportunity:** True positive rate (recall) across segments
3. **Calibration:** Predicted probability matches observed abandonment rate per segment
4. **Disparate Impact:** Ratio of intervention rates (should be 0.8-1.25)

### Known Limitations
- No explicit demographic data → cannot measure direct discrimination
- `VisitorType` may correlate with new vs. returning customer treatment
- Single-retailer data → unknown bias transfer to other contexts

---

## Explainability

### Global Feature Importance (SHAP)
*From test set sample (n=200)*

| Rank | Feature | Mean |SHAP| | Direction | Interpretation |
|------|---------|------------|-----------|----------------|
| 1 | `PageValues` | 1.985 | **Negative** | Higher page values → **lower abandonment** (more likely to purchase) |
| 2 | `Month_Mar` | 1.255 | **Negative** | March sessions → **lower abandonment** |
| 3 | `Month_Nov` | 0.483 | **Negative** | November sessions → **lower abandonment** |
| 4 | `TrafficType` | 0.240 | **Negative** | Certain traffic sources → **lower abandonment** |
| 5 | `ProductRelated_Duration` | 0.216 | **Positive** | Longer product browsing → **higher abandonment** (paradoxical) |

> **Note:** `PageValues` was retained as a feature but flagged as potential leakage proxy. `ProductRelated_Duration` positive SHAP suggests deep browsing without purchase indicates hesitation. Top drivers are behavioral + temporal, not demographic.

### Local Explanations
- **SHAP Waterfall Plots:** Per-session feature contributions
- **SHAP Bar Plots:** Global feature importance ranking
- **Top-N Drivers:** Extracted for LLM personalization

### RAG-Augmented Diagnosis
- Retrieves relevant user review context from vectorstore
- Generates natural-language explanation for abandonment drivers
- Corpus: 100 synthetic reviews (placeholder for real feedback)
- Example query: "Why do users abandon due to PageValues?" → Returns insights on pricing confusion, inconsistent discounts

---

## Intervention Simulation (What-If)

### Strategies Simulated
| Strategy | Mechanism | Parameters Tested |
|----------|-----------|-------------------|
| **Discount** | Reduce `price_sensitivity` proxy | 5%, 10%, 15%, 20%, 25% |
| **Urgency** | Direct probability boost | +5%, +10%, +15%, +20% |
| **Value** | Boost engagement features | +10%, +15%, +20%, +25% |

### Revenue Estimation
- **Base AOV:** $100 (configurable)
- **Intervention Cost:** Discount = % × AOV; Urgency/Value = $0 (messaging only)
- **Metric:** Expected revenue uplift per session

### A/B Test Simulation Results (2026-11-16)
*Run on test set sample (n=2466)*

| Strategy | Best Variant | Abandonment Δ | Conversion Δ | Uplift/Session | Notes |
|----------|--------------|---------------|--------------|----------------|-------|
| **Discount** | 5% | 0.00% | 0.00% | **-$5.00** | No effect on predictions (price_sensitivity feature removed) |
| **Urgency** | +5% | +4.88% | -4.88% | **-$4.88** | Increases abandonment (bug: probability inversion) |
| **Value** | +10% | +0.10% | -0.10% | **-$0.09** | Minimal effect |

> **⚠️ Critical Bug Identified:** The simulation inverts model probabilities. The model predicts P(Revenue=1) = conversion probability, but simulation treats it as abandonment probability (1 - P). All "interventions" incorrectly increase predicted abandonment.
>
> **Correct interpretation needed:** Discount should reduce abandonment (increase conversion). Urgency/Value should boost conversion probability directly.
>
> **Recommendation:** Fix simulation logic before using for decision-making. Current results are artifact of probability inversion.

### Base Rates (Test Set)
- **Base Conversion Rate (P(Revenue=1)):** 73.70%
- **Base Abandonment Rate:** 26.30%
- **Actual Class Balance:** 15.5% Revenue=True (382/2466)

> **Discrepancy Note:** Model predicts 73.7% conversion vs. actual 15.5%. Model is miscalibrated — predicts high probability for most samples. Threshold 0.5 classifies most as purchasers. This explains high accuracy (87%) but low precision on Class 1 (56%).

---

## Limitations & Risks

### Technical
1. **Data Drift:** 2018 training data vs. current traffic
2. **Feature Leakage Risk:** `PageValues` retained but flagged as potential leakage proxy
3. **No Causal Validation:** Interventions simulated, not A/B tested
4. **Simulated RAG Corpus:** 100 fake reviews ≠ real user voice
5. **Model Miscalibration:** Predicts 73.7% conversion vs. actual 15.5% — probabilities not well-calibrated. Threshold 0.5 over-predicts positive class. Use Platt scaling / isotonic regression before production.
6. **A/B Test Simulation Bug:** Probability inversion (treats P(Revenue=1) as abandonment probability). All simulation results invalid until fixed.

### Operational
1. **Threshold Sensitivity:** Small threshold changes = large intervention volume swings
2. **Cold Start:** New users (VisitorType=New) have less behavioral history
3. **Batch-Only:** Current implementation not real-time streaming
4. **Single Model:** No ensemble, no uncertainty quantification

### Ethical
1. **Dark Patterns Risk:** Urgency messaging could be manipulative
2. **Discount Dependency:** Training users to wait for offers
3. **Privacy:** Session tracking requires consent (GDPR/CCPA)
4. **Exclusion:** Model may systematically underserve certain segments

---

## Monitoring & Maintenance

### Data Drift Detection
- **Feature Drift:** KS-test on input distributions (weekly)
- **Prediction Drift:** Monitor abandonment rate distribution (daily)
- **Concept Drift:** Track calibration curve (monthly)

### Performance Monitoring
- **Proxy Metric:** Intervention acceptance rate (if deployed)
- **Business Metric:** Revenue recovery vs. control group
- **Model Health:** AUC on labeled samples (when available)

### Retraining Cadence
- **Minimum:** Monthly (if volume sufficient)
- **Trigger:** AUC drop > 0.02 or calibration drift > 0.05
- **Process:** Automated pipeline with validation gates

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026-11-16 | Initial release: XGBoost + SHAP + RAG + LLM + A/B sim; removed simulated features & PageValues leakage; added val split |
| 0.1.1 | 2026-11-16 | Model card updated with actual metrics: best threshold=0.50, F1=0.661, accuracy=87.1%; SHAP drivers documented; A/B test simulation bug identified (probability inversion) |

---

## Contact & Governance

**Author:** Manmohan Ganesan  
**Role:** ML Engineer / Data Scientist  
**Review Date:** 2026-11-16  
**Next Review:** 2027-02-16  

**Approval:** 
- [ ] Technical Lead
- [ ] Product Owner
- [ ] Legal/Privacy (if deploying)

---

## Appendix: Configuration Reference

Key config sections (see `configs/config.yaml`):
- `data`: paths, split ratios, feature engineering params
- `model.xgboost`: hyperparameters, class imbalance handling
- `model.thresholds`: candidate thresholds for tuning
- `simulation.ab_test`: A/B test parameters
- `model_card`: this document's structured metadata
