# ASTRA M1 Model: Threshold Optimization and Executive Summary

This report presents the analysis and recommendation for optimizing thresholds on the ASTRA M1 Closure Prediction Model to reduce operator alert fatigue while ensuring critical event capture.

---

## 1. Event Causes Contributing Most to False Positives
The baseline model generated a high rate of False Positives (**890 FP** on the test set). 

A per-cause breakdown of False Positives from the test split reveals that a few classes dominate the false alarm rate:
1. **vehicle_breakdown**: Generated **442 FP** (49.7% of total FP). 
2. **water_logging**: Generated **146 FP** (16.4% of total FP).
3. **pot_holes**: Generated **91 FP** (10.2% of total FP).
4. **tree_fall**: Generated **67 FP** (7.5% of total FP).
5. **others**: Generated **64 FP** (7.2% of total FP).

### Top FP Contributors Summary
Combined, `vehicle_breakdown`, `water_logging`, and `pot_holes` account for **76.3%** of all false positive alerts. Alert fatigue is primarily driven by these low-prevalence/high-volume categories where the local threshold was set extremely low to force high recall.

---

## 2. Per-Cause Prevalence & Evaluation Metrics
Below is the test split breakdown of closure prevalence, recall, precision, and false positive rates (FPR) under the starting model parameters:

| Event Cause | Group | Total Samples | True Positives | FPR | Recall | Precision | Prevalence % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **vehicle_breakdown** | Low | 899 | 27 | 50.7% | 66.67% | 3.91% | 3.00% |
| **water_logging** | Low | 213 | 16 | 74.1% | 93.75% | 9.32% | 7.51% |
| **tree_fall** | Medium | 126 | 44 | 81.7% | 95.45% | 38.53% | 34.92% |
| **pot_holes** | Low | 116 | 3 | 80.5% | 66.67% | 2.15% | 2.59% |
| **others** | Low | 89 | 7 | 78.0% | 100.00% | 9.86% | 7.87% |
| **construction** | Medium | 56 | 13 | 44.2% | 100.00% | 40.62% | 23.21% |
| **accident** | Low | 43 | 3 | 55.0% | 66.67% | 8.33% | 6.98% |
| **road_conditions** | Medium | 35 | 5 | 50.0% | 40.00% | 11.76% | 14.29% |
| **congestion** | Very Low | 27 | 0 | 59.3% | 0.00% | 0.00% | 0.00% |
| **vip_movement** | High | 16 | 16 | 0.0% | 100.00% | 100.00% | 100.00% |
| **public_event** | High | 9 | 6 | 100.0% | 100.00% | 66.67% | 66.67% |
| **procession** | High | 8 | 2 | 50.0% | 100.00% | 40.00% | 25.00% |
| **protest** | High | 1 | 0 | 100.0% | 0.00% | 0.00% | 0.00% |
| **debris** | Very Low | 1 | 0 | 100.0% | 0.00% | 0.00% | 0.00% |

---

## 3. Comparison Table: Current vs. Optimized Models

| Metric | Current Model (Baseline) | Optimized Global Threshold (0.01) | Optimized Cause-Group Tiers (Score-Max) | Optimized Cause-Group Tiers (FP-Min) |
| :--- | :--- | :--- | :--- | :--- |
| **True Negatives (TN)** | 607 | 214 | 217 | 639 |
| **False Positives (FP)** | 890 | 1283 | 1280 | 858 |
| **False Negatives (FN)** | 17 | 2 | 2 | 18 |
| **True Positives (TP)** | 125 | 140 | 140 | 124 |
| **Overall Recall** | 88.03% | 98.59% | 98.59% | 87.32% |
| **Overall Precision** | 12.32% | 9.84% | 9.86% | 12.63% |
| **F1 Score** | 0.2161 | 0.1789 | 0.1793 | 0.2206 |
| **F2 Score** | 0.3948 | 0.3516 | 0.3521 | 0.4000 |
| **PR-AUC** | 0.3829 | 0.3829 | 0.3829 | 0.3829 |
| **FP Reduction %** | Baseline | **-44.16%** | **-43.82%** | **3.60%** |
| **FN Increase %** | Baseline | **-88.24%** | **-88.24%** | **5.88%** |

---

## 4. Final Recommendation & Deployment Threshold Strategy

### Analysis of Threshold Tuning Constraints
To qualify as a successful threshold-only optimization under the project requirements:
1. Recall must remain **>= 85%**.
2. False Positives must be reduced by **>= 40%** (i.e. FP <= 534).

Looking at the optimization results:
* The **Optimal Global Threshold (0.01)** yields **FP = 1283** (-44.16% reduction) with a Recall of **98.59%**.
* The **Optimal Cause-Group Thresholds (Score-Max)** yield **FP = 1280** (-43.82% reduction) with a Recall of **98.59%**.
* The **Optimal Cause-Group Thresholds (FP-Min)** yield **FP = 858** (3.60% reduction) with a Recall of **87.32%**.

### Recommendation:
> [!WARNING]
> **Threshold Tuning Limitation Identified:** None of the threshold-only optimization strategies can successfully reduce False Positives by **>= 40%** (target FP <= 534) while maintaining the operational recall floor of **>= 85%**.
> 
> Specifically, the best possible FP reduction that maintains a Recall >= 85% on the test split is **3.60%** (yielding **858 FP** and **87.32% Recall** using cause-group thresholds of High=0.23, Med=0.10, Low=0.05).
> 
> Raising the global threshold to any value greater than or equal to `0.07` drops the test Recall to `77.46%`, which violates the `85%` recall floor. This indicates that true positive and false positive predictions are highly overlapped between probabilities `0.01` and `0.07`.
> 
> Therefore, we recommend **not deploying a threshold-only change** as the final solution. Instead, the model needs to be retrained with advanced improvements.

### Recommended Advanced Model Improvements:
1. **Cost-Sensitive Learning / Sample Weighting:** During training, apply a custom sample weight to the positive class (e.g. 5x or 10x weight to road closure events) or modify the loss function to explicitly penalize false negatives more heavily than false positives. This will force the classifier to learn a cleaner boundary, separating positive and negative instances with a larger probability gap.
2. **Feature Engineering:** Build interaction features such as `cause_prev_closures` (historical closure rate for the event cause), `corridor_closure_density`, or NLP embeddings from text fields like `incident_description` to provide stronger predictive signals.
3. **Calibrated Cause-Specific Modeling:** Train separate models or sub-classifiers for high-risk cause categories (like public events or construction) to model their closure conditions independently.

### backend/main.py Integration (If threshold-only solution is selected):
If a sub-optimal threshold-only solution must be deployed, modify the backend startup threshold dictionary in `main.py` using the FP-Min Cause-Group configuration:
```python
# Update thresholds in src/astra_ml/api/main.py:
cause_thresholds = {
    "vip_movement": 0.23,
    "public_event": 0.23,
    "protest": 0.23,
    "procession": 0.23,
    "construction": 0.10,
    "tree_fall": 0.10,
    "road_conditions": 0.10,
    "vehicle_breakdown": 0.05,
    "accident": 0.05,
    "pot_holes": 0.05,
    "water_logging": 0.05,
    "debris": 0.05,
    "congestion": 0.05,
}
```
