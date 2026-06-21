# M1 — Closure-Necessity Classifier — Evaluation Report

_Generated: 2026-06-20 18:39 UTC_

## Split Strategy
Time-based split: Train < Feb 15 2024, Val Feb 15–Mar 15, Test > Mar 15 2024. No random splitting — this is temporal event data with trends.

## Summary Comparison Table (Test Split)
| Experiment | Recall | Precision | F2 | F1 | PR-AUC | Review % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline (Global Uncalibrated) | 0.8873 | 0.1520 | 0.4510 | 0.2595 | 0.4448 | 50.58% |
| Step 1: Cause-Conditional Thresholds | 0.8803 | 0.1123 | 0.3718 | 0.1992 | 0.4448 | 67.91% |
| Step 2: Calibrated + Cause Thresholds | 0.9225 | 0.1145 | 0.3826 | 0.2037 | 0.3471 | 69.80% |
| Step 3: PR-AUC HPO + Cal + Cause Thresholds | 0.8873 | 0.1165 | 0.3818 | 0.2059 | 0.3362 | 66.02% |
| Step 4: Variant A (with interactions) | 0.9225 | 0.1139 | 0.3813 | 0.2028 | 0.3430 | 70.16% |
| Step 5: Variant B (no interactions) [WINNER] | 0.8944 | 0.1192 | 0.3889 | 0.2104 | 0.3528 | 64.98% |
| Step 6: Label downweighting | 0.9085 | 0.1114 | 0.3737 | 0.1985 | 0.3380 | 70.65% |

*Note: F2-score is the primary optimization metric.*

## Cause-Conditional Thresholding & Review Rate Isolation (Step 1)
To satisfy the recall floor of >= 0.85 per cause group on the validation split, thresholds were adjusted per group:
- **High-Closure**: 0.5298 (val positives: 9)
- **Medium-Closure**: 0.4238 (val positives: 78)
- **Low-Closure**: 0.0903 (val positives: 66)
- **Very-Low-Closure**: 0.0698 (val positives: 4)
- **Global Fallback**: 0.0617 (val positives: 4)

### Review-Rate Isolation (Test Split)
When moving from global to cause-conditional thresholding:
- **Net new review events flagged from `vehicle_breakdown`**: 181
- **Net new review events flagged from other cause groups**: 103
- **Change Contribution**: `vehicle_breakdown` contributed **63.7%** of the net flagged volume change, while all other groups combined contributed **36.3%**.

*Insight*: Enforcing local recall floors on low-base-rate groups requires setting very low thresholds (e.g., 0.0903 for low-closure). This dramatically increases false positives in high-volume classes like `vehicle_breakdown`, causing a spike in the overall review rate.

## Per-Cause Confusion Matrix (Winner Model - Variant B Calibrated + Cause Thresholds)
| Cause | Group | TN | FP | FN | TP | Recall | Precision | Total |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Debris | very_low_closure | 0 | 1 | 0 | 0 | 0.00% | 0.00% | 1 |
| accident | low_closure | 7 | 33 | 0 | 3 | 100.00% | 8.33% | 43 |
| congestion | very_low_closure | 5 | 22 | 0 | 0 | 0.00% | 0.00% | 27 |
| construction | medium_closure | 21 | 22 | 2 | 11 | 84.62% | 33.33% | 56 |
| others | low_closure | 6 | 76 | 0 | 7 | 100.00% | 8.43% | 89 |
| pot_holes | global_fallback | 27 | 86 | 2 | 1 | 33.33% | 1.15% | 116 |
| procession | high_closure | 2 | 4 | 0 | 2 | 100.00% | 33.33% | 8 |
| protest | high_closure | 0 | 1 | 0 | 0 | 0.00% | 0.00% | 1 |
| public_event | high_closure | 1 | 2 | 0 | 6 | 100.00% | 75.00% | 9 |
| road_conditions | medium_closure | 15 | 15 | 3 | 2 | 40.00% | 11.76% | 35 |
| tree_fall | medium_closure | 13 | 69 | 2 | 42 | 95.45% | 37.84% | 126 |
| vehicle_breakdown | low_closure | 281 | 591 | 5 | 22 | 81.48% | 3.59% | 899 |
| vip_movement | high_closure | 0 | 0 | 0 | 16 | 100.00% | 100.00% | 16 |
| water_logging | low_closure | 21 | 176 | 1 | 15 | 93.75% | 7.85% | 213 |

## Accuracy Caveat
> [!IMPORTANT]
> At an 8.3% base closure rate in the test dataset, accuracy is a highly misleading metric. A trivial "always-negative" classifier would score ~91.7% accuracy with 0% recall. In an operational context where a missed closure event is the costliest failure mode, this is unacceptable. Therefore, we optimize for the F2-score (weighting recall twice as much as precision) and require recall to stay above 0.85, referencing accuracy solely as a baseline check.

## Known Limitations
- Target encoding uses smoothed group means, not full K-fold LOO, which may introduce slight leakage for small groups.
- The operational threshold favors recall ≥ 0.85 — precision is traded for coverage, since a missed closure is more costly than a false alarm.
- Per-cause closure rates vary enormously (2.4% for pot_holes to 80% for vip_movement) — the model's accuracy is not uniform across causes.
- Very rare event types (vip_movement n=20, protest n=15) have wide confidence intervals; M3 addresses this with text-embedding transfer.
- `pot_holes` exhibits no discriminative signal (within-cause ROC-AUC of 0.4749) and has been routed to a global fallback.
