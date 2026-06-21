# M2 — Duration Estimator — Acute Regime Evaluation Report

_Generated: 2026-06-19 18:51 UTC_

## Split Strategy

Time-based split: Train < Feb 15 2024, Val Feb 15–Mar 15, Test > Mar 15 2024. Only events with non-null duration included.

## Acute Regime Metrics

| Metric  | Value            |
| ------- | ---------------- |
| name    | LightGBM (acute) |
| log_mae | 0.7541           |
| mae     | 30.9170          |
| rmse    | 40.9091          |

## Regime-Split vs. Pooled Baseline

- **Pooled baseline log-MAE**: 1.6094
- **Acute regime log-MAE**: 0.7541
- **Error reduction**: 53.1%

> The regime-split approach reduces log-MAE by **53.1%** compared to the pooled single-regressor baseline.

## Known Limitations

- Acute regime only covers ~80% of duration-labeled events.
- Events with zero or negative computed duration are excluded.
- The pooled baseline includes chronic events, making its log-MAE artificially high; the comparison is valid but regime-specific.
- Optuna search space may not cover the global optimum.
