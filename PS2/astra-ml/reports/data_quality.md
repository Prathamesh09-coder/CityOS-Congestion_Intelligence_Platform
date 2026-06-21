# Data Quality Report — ASTRA ML Pipeline

_Generated: 2026-06-19 15:36 UTC_

## Geo-Imputation Coverage

Junction and zone fields had high null rates in the raw ASTRAM data. Geo-imputation using lat/long coordinates (which are 100% populated) was applied to recover these values.

| Field    | Before (null %) | After (null %) | Records Recovered |
| -------- | --------------- | -------------- | ----------------- |
| junction | 69.3%           | 0.0%           | 5663              |
| zone     | 57.9%           | 0.0%           | 4729              |

## Duration Source Resolution

Event duration was computed by coalescing `closed_datetime` and `resolved_datetime` (preferring whichever is non-null). This recovered additional duration-labeled records beyond `closed_datetime` alone.

| Duration Source   | Count | Percentage |
| ----------------- | ----- | ---------- |
| closed_datetime   | 3141  | 38.4%      |
| resolved_datetime | 68    | 0.8%       |
| none              | 4964  | 60.7%      |

## Implications

- The coalesced duration source recovers records that would otherwise be right-censored in the chronic-regime survival model (M2).
- Geo-imputed junction/zone values carry `zone_imputed=True` / `junction_imputed=True` flags, allowing downstream models to distinguish original from imputed values.
- The `duration_source` column makes the datetime resolution auditable per-record.
