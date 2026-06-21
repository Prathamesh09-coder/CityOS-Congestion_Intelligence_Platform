# M2 — Duration Estimator (Survival) — Chronic Regime Evaluation Report

_Generated: 2026-06-19 18:51 UTC_

## Split Strategy

Time-based split: Train+Val (< Mar 15 2024) combined due to small chronic sample, Test (> Mar 15 2024). Right-censored events included in training.

## Chronic Regime Metrics

| Metric             | Value  |
| ------------------ | ------ |
| name               | GBST   |
| c_index            | 0.5410 |
| log_mae_uncensored | 1.0094 |

## Known Limitations

- Chronic regime has fewer labeled events than acute.
- Right-censored events are included via survival analysis, preventing the bias toward shorter durations that dropping them would cause.
- C-index measures ranking accuracy, not calibration — a high C-index doesn't guarantee well-calibrated survival curves.
- CoxNet may fail on very sparse data; GBST is more robust.
- log-MAE is computed only on uncensored test events, which may not be representative of all chronic events.
