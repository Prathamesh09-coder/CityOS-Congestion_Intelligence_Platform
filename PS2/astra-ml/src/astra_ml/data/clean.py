"""Data cleaning — deduplication, missingness indicators, duration computation, regime labeling.

Key additions per the approved plan:
- Duration source coalescing: prefers non-null of closed_datetime/resolved_datetime
- duration_source column for auditability
- Missingness indicator columns
- Duration regime labeling (acute vs chronic)
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Load data config from configs/data.yaml."""
    cfg = OmegaConf.load("configs/data.yaml")
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def clean(config: dict | None = None) -> pl.DataFrame:
    """Clean the interim Parquet: dedup, missingness indicators, duration, regime labels.

    Args:
        config: Optional config dict. Loaded from configs/data.yaml if None.

    Returns:
        The cleaned DataFrame.
    """
    if config is None:
        config = load_config()

    interim_path = Path(config["paths"]["interim_parquet"])
    cleaned_path = Path(config["paths"]["cleaned_parquet"])
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Reading interim Parquet: %s", interim_path)
    df = pl.read_parquet(interim_path)
    logger.info("Loaded %d rows", df.height)

    # === Deduplication ===
    n_before = df.height
    df = df.unique(subset=["event_id"], keep="first")
    n_dupes = n_before - df.height
    logger.info("Deduplicated: removed %d duplicate event_ids", n_dupes)

    # === Missingness indicators ===
    missingness_cols = {
        "junction": "junction_missing",
        "zone": "zone_missing",
        "closed_datetime": "closed_datetime_missing",
        "vehicle_type": "vehicle_type_missing",
        "assigned_to_police_id": "assigned_to_police_missing",
    }
    for source_col, indicator_col in missingness_cols.items():
        if source_col in df.columns:
            df = df.with_columns(
                pl.col(source_col).is_null().alias(indicator_col)
            )
            null_count = df.filter(pl.col(indicator_col)).height
            logger.info(
                "  %s null rate: %d/%d (%.1f%%)",
                source_col,
                null_count,
                df.height,
                null_count / df.height * 100,
            )

    # === Duration source resolution ===
    # Check null rates independently
    closed_null = df.filter(pl.col("closed_datetime").is_null()).height if "closed_datetime" in df.columns else df.height
    resolved_null = df.filter(pl.col("resolved_datetime").is_null()).height if "resolved_datetime" in df.columns else df.height
    logger.info("Duration source null rates:")
    logger.info("  closed_datetime null: %d/%d (%.1f%%)", closed_null, df.height, closed_null / df.height * 100)
    logger.info("  resolved_datetime null: %d/%d (%.1f%%)", resolved_null, df.height, resolved_null / df.height * 100)

    # Coalesce: prefer whichever is non-null
    has_closed = "closed_datetime" in df.columns
    has_resolved = "resolved_datetime" in df.columns

    if has_closed and has_resolved:
        df = df.with_columns(
            # Coalesced end datetime
            pl.coalesce(["closed_datetime", "resolved_datetime"]).alias("end_datetime_coalesced"),
            # Duration source tracking
            pl.when(pl.col("closed_datetime").is_not_null())
            .then(pl.lit("closed_datetime"))
            .when(pl.col("resolved_datetime").is_not_null())
            .then(pl.lit("resolved_datetime"))
            .otherwise(pl.lit("none"))
            .alias("duration_source"),
        )
    elif has_closed:
        df = df.with_columns(
            pl.col("closed_datetime").alias("end_datetime_coalesced"),
            pl.when(pl.col("closed_datetime").is_not_null())
            .then(pl.lit("closed_datetime"))
            .otherwise(pl.lit("none"))
            .alias("duration_source"),
        )
    else:
        df = df.with_columns(
            pl.lit(None).cast(pl.Datetime).alias("end_datetime_coalesced"),
            pl.lit("none").alias("duration_source"),
        )

    # Log duration source distribution
    if "duration_source" in df.columns:
        source_counts = df.group_by("duration_source").len().sort("duration_source")
        logger.info("Duration source distribution:")
        for row in source_counts.iter_rows(named=True):
            logger.info(
                "  %s: %d (%.1f%%)",
                row["duration_source"],
                row["len"],
                row["len"] / df.height * 100,
            )

    # === Duration computation (from coalesced end datetime) ===
    if "reported_datetime" in df.columns:
        df = df.with_columns(
            pl.when(
                pl.col("end_datetime_coalesced").is_not_null()
                & pl.col("reported_datetime").is_not_null()
            )
            .then(
                (pl.col("end_datetime_coalesced") - pl.col("reported_datetime"))
                .dt.total_minutes()
            )
            .otherwise(pl.lit(None))
            .cast(pl.Float64)
            .alias("duration_minutes")
        )

        # Filter out negative durations (data quality issues)
        neg_dur = df.filter(
            pl.col("duration_minutes").is_not_null() & (pl.col("duration_minutes") < 0)
        ).height
        if neg_dur > 0:
            logger.warning("Found %d records with negative duration — setting to null", neg_dur)
            df = df.with_columns(
                pl.when(pl.col("duration_minutes") < 0)
                .then(pl.lit(None))
                .otherwise(pl.col("duration_minutes"))
                .alias("duration_minutes")
            )

        duration_available = df.filter(pl.col("duration_minutes").is_not_null()).height
        logger.info(
            "Duration computed for %d/%d records (%.1f%%)",
            duration_available,
            df.height,
            duration_available / df.height * 100,
        )

    # === Duration regime labeling ===
    acute_causes = config.get("duration_regimes", {}).get("acute", [])
    chronic_causes = config.get("duration_regimes", {}).get("chronic", [])

    if "event_cause" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("event_cause").is_in(acute_causes))
            .then(pl.lit("acute"))
            .when(pl.col("event_cause").is_in(chronic_causes))
            .then(pl.lit("chronic"))
            .otherwise(pl.lit("other"))
            .alias("duration_regime")
        )

        regime_counts = df.group_by("duration_regime").len().sort("duration_regime")
        logger.info("Duration regime distribution:")
        for row in regime_counts.iter_rows(named=True):
            logger.info("  %s: %d", row["duration_regime"], row["len"])

    # === Censoring flag for survival analysis ===
    # Right-censor if end_datetime_coalesced is null (event hadn't resolved at data cutoff)
    df = df.with_columns(
        pl.col("end_datetime_coalesced").is_not_null().alias("event_observed")
    )

    # Duration in days for survival model
    df = df.with_columns(
        pl.when(pl.col("duration_minutes").is_not_null())
        .then(pl.col("duration_minutes") / (60.0 * 24.0))
        .otherwise(pl.lit(None))
        .alias("duration_days")
    )

    # For censored events, compute time from start to data cutoff as the censored duration
    data_cutoff = config.get("splits", {}).get("train_end", "2024-04-08")
    # Use the last date in dataset as actual cutoff
    if "reported_datetime" in df.columns:
        cutoff_dt = pl.Series([data_cutoff]).str.to_datetime("%Y-%m-%d", time_zone="UTC")[0]
        df = df.with_columns(
            pl.when(
                pl.col("event_observed").not_()
                & pl.col("reported_datetime").is_not_null()
            )
            .then(
                (pl.lit(cutoff_dt) - pl.col("reported_datetime"))
                .dt.total_minutes()
                / (60.0 * 24.0)
            )
            .otherwise(pl.col("duration_days"))
            .alias("duration_days")
        )

    # Write cleaned Parquet
    df.write_parquet(cleaned_path)
    logger.info("Wrote cleaned Parquet: %s (%d rows)", cleaned_path, df.height)

    return df


if __name__ == "__main__":
    clean()
