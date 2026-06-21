"""Report generation utilities — writes markdown/JSON evaluation reports.

No plotting, no UI, no Streamlit — pure text/table output to reports/ directory.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    """Format a markdown table from headers and rows."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    sep_line = "| " + " | ".join("-" * w for w in widths) + " |"
    data_lines = [
        "| " + " | ".join(cell.ljust(w) for cell, w in zip(row, widths)) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *data_lines])


def write_classification_report(
    report_path: str | Path,
    model_name: str,
    metrics_dict: dict[str, Any],
    baseline_auc: float | None = None,
    split_strategy: str = "time-based",
    known_limitations: str = "",
    comparison_runs: list[dict[str, Any]] | None = None,
) -> Path:
    """Write a markdown classification evaluation report.

    Args:
        report_path: Output path for the .md file.
        model_name: Name of the model being evaluated.
        metrics_dict: Dict of metric_name → value.
        baseline_auc: Expected baseline AUC for comparison.
        split_strategy: Description of train/val/test split.
        known_limitations: Free-text known limitations section.
        comparison_runs: Optional list of dicts with 'name' and metric keys for comparison table.

    Returns:
        Path to the written report.
    """
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# {model_name} — Evaluation Report",
        f"\n_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_\n",
        "## Split Strategy",
        f"\n{split_strategy}\n",
        "## Metrics",
        "",
    ]

    # Main metrics table
    headers = ["Metric", "Value"]
    rows = [[k, f"{v:.4f}" if isinstance(v, float) else str(v)] for k, v in metrics_dict.items()]
    lines.append(_format_table(headers, rows))

    if baseline_auc is not None:
        lines.extend([
            "\n## Baseline Comparison",
            f"\n- **Expected baseline AUC**: {baseline_auc:.3f}",
            f"- **Achieved AUC**: {metrics_dict.get('roc_auc', 'N/A')}",
        ])
        if "roc_auc" in metrics_dict:
            delta = metrics_dict["roc_auc"] - baseline_auc
            direction = "improvement" if delta > 0 else "regression"
            lines.append(f"- **Delta**: {delta:+.4f} ({direction})")

    if comparison_runs:
        lines.extend(["\n## Model Comparison\n"])
        comp_headers = ["Model"] + [k for k in comparison_runs[0] if k != "name"]
        comp_rows = []
        for run in comparison_runs:
            row = [run.get("name", "unnamed")]
            for k in comp_headers[1:]:
                v = run.get(k, "N/A")
                row.append(f"{v:.4f}" if isinstance(v, float) else str(v))
            comp_rows.append(row)
        lines.append(_format_table(comp_headers, comp_rows))

    if known_limitations:
        lines.extend(["\n## Known Limitations\n", known_limitations])

    report_text = "\n".join(lines) + "\n"
    report_path.write_text(report_text)
    return report_path


def write_regression_report(
    report_path: str | Path,
    model_name: str,
    regime: str,
    metrics_dict: dict[str, Any],
    pooled_baseline: dict[str, Any] | None = None,
    split_strategy: str = "time-based",
    known_limitations: str = "",
) -> Path:
    """Write a markdown regression evaluation report.

    Args:
        report_path: Output path for the .md file.
        model_name: Name of the model being evaluated.
        regime: Duration regime ("acute" or "chronic").
        metrics_dict: Dict of metric_name → value.
        pooled_baseline: Optional metrics from the pooled (non-split) baseline.
        split_strategy: Description of train/val/test split.
        known_limitations: Free-text known limitations section.

    Returns:
        Path to the written report.
    """
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# {model_name} — {regime.title()} Regime Evaluation Report",
        f"\n_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_\n",
        "## Split Strategy",
        f"\n{split_strategy}\n",
        f"## {regime.title()} Regime Metrics\n",
    ]

    headers = ["Metric", "Value"]
    rows = [[k, f"{v:.4f}" if isinstance(v, float) else str(v)] for k, v in metrics_dict.items()]
    lines.append(_format_table(headers, rows))

    if pooled_baseline:
        lines.extend(["\n## Regime-Split vs. Pooled Baseline\n"])
        pooled_log_mae = pooled_baseline.get("log_mae", None)
        regime_log_mae = metrics_dict.get("log_mae", None)
        if pooled_log_mae is not None and regime_log_mae is not None:
            improvement = (pooled_log_mae - regime_log_mae) / pooled_log_mae * 100
            lines.extend([
                f"- **Pooled baseline log-MAE**: {pooled_log_mae:.4f}",
                f"- **{regime.title()} regime log-MAE**: {regime_log_mae:.4f}",
                f"- **Error reduction**: {improvement:.1f}%",
                "",
                f"> The regime-split approach reduces log-MAE by **{improvement:.1f}%** "
                f"compared to the pooled single-regressor baseline.",
            ])

    if known_limitations:
        lines.extend(["\n## Known Limitations\n", known_limitations])

    report_text = "\n".join(lines) + "\n"
    report_path.write_text(report_text)
    return report_path


def write_data_quality_report(
    report_path: str | Path,
    imputation_stats: dict[str, Any],
    duration_source_stats: dict[str, Any],
) -> Path:
    """Write the data quality report documenting imputation and duration source resolution.

    Args:
        report_path: Output path for the .md file.
        imputation_stats: Dict with before/after null rates for junction/zone.
        duration_source_stats: Dict with counts for each duration source.

    Returns:
        Path to the written report.
    """
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Data Quality Report — ASTRA ML Pipeline",
        f"\n_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_\n",
        "## Geo-Imputation Coverage\n",
        "Junction and zone fields had high null rates in the raw ASTRAM data. "
        "Geo-imputation using lat/long coordinates (which are 100% populated) "
        "was applied to recover these values.\n",
    ]

    headers = ["Field", "Before (null %)", "After (null %)", "Records Recovered"]
    rows = []
    for field_name in ["junction", "zone"]:
        before = imputation_stats.get(f"{field_name}_null_before", 0)
        after = imputation_stats.get(f"{field_name}_null_after", 0)
        total = imputation_stats.get("total_records", 1)
        recovered = before - after
        rows.append([
            field_name,
            f"{before / total * 100:.1f}%",
            f"{after / total * 100:.1f}%",
            str(recovered),
        ])
    lines.append(_format_table(headers, rows))

    lines.extend([
        "\n## Duration Source Resolution\n",
        "Event duration was computed by coalescing `closed_datetime` and "
        "`resolved_datetime` (preferring whichever is non-null). "
        "This recovered additional duration-labeled records beyond `closed_datetime` alone.\n",
    ])

    ds_headers = ["Duration Source", "Count", "Percentage"]
    ds_rows = []
    total = duration_source_stats.get("total", 1)
    for source in ["closed_datetime", "resolved_datetime", "none"]:
        count = duration_source_stats.get(source, 0)
        ds_rows.append([source, str(count), f"{count / total * 100:.1f}%"])
    lines.append(_format_table(ds_headers, ds_rows))

    lines.extend([
        "\n## Implications\n",
        "- The coalesced duration source recovers records that would otherwise "
        "be right-censored in the chronic-regime survival model (M2).",
        "- Geo-imputed junction/zone values carry `zone_imputed=True` / "
        "`junction_imputed=True` flags, allowing downstream models to distinguish "
        "original from imputed values.",
        "- The `duration_source` column makes the datetime resolution auditable "
        "per-record.",
    ])

    report_text = "\n".join(lines) + "\n"
    report_path.write_text(report_text)

    # Also write machine-readable JSON alongside
    json_path = report_path.with_suffix(".json")
    json_path.write_text(json.dumps({
        "imputation": imputation_stats,
        "duration_source": duration_source_stats,
    }, indent=2))

    return report_path
