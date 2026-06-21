"""Data ingestion — reads raw CSV and produces a typed interim Parquet file.

Handles:
- Windows line endings (\r\n)
- Literal "NULL" strings → proper nulls
- Column renaming to canonical names
- Datetime parsing with timezone handling (UTC → IST)
- Boolean casting for requires_road_closure
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# IST offset for Bengaluru
IST_OFFSET = "5h30m"


def load_config() -> dict:
    """Load data config from configs/data.yaml."""
    cfg = OmegaConf.load("configs/data.yaml")
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def ingest(config: dict | None = None) -> pl.DataFrame:
    """Ingest raw CSV and produce typed interim Parquet.

    Args:
        config: Optional config dict. Loaded from configs/data.yaml if None.

    Returns:
        The ingested DataFrame.
    """
    if config is None:
        config = load_config()

    raw_path = Path(config["paths"]["raw_csv"])
    interim_path = Path(config["paths"]["interim_parquet"])
    interim_path.parent.mkdir(parents=True, exist_ok=True)

    null_values = config.get("null_values", ["NULL", ""])
    column_rename = config.get("column_rename", {})

    logger.info("Reading raw CSV: %s", raw_path)

    # Read CSV with polars, handling null values
    df = pl.read_csv(
        raw_path,
        null_values=null_values,
        try_parse_dates=False,  # We'll parse dates manually for control
        truncate_ragged_lines=True,
    )

    # Strip \r from column names and string values (Windows line endings)
    df = df.rename({col: col.strip().replace("\r", "") for col in df.columns})

    logger.info("Raw CSV shape: %d rows × %d columns", df.height, df.width)

    # Rename columns to canonical names
    rename_map = {k: v for k, v in column_rename.items() if k in df.columns}
    df = df.rename(rename_map)
    logger.info("Renamed columns: %s", rename_map)

    # Cast requires_road_closure from string TRUE/FALSE to boolean
    if "requires_road_closure" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("requires_road_closure").cast(pl.Utf8).str.to_uppercase() == "TRUE")
            .then(pl.lit(True))
            .otherwise(pl.lit(False))
            .alias("requires_road_closure")
        )

    # Parse datetime columns
    datetime_cols = [
        "reported_datetime",
        "end_datetime",
        "modified_datetime",
        "created_date",
        "closed_datetime",
        "resolved_datetime",
    ]
    for col_name in datetime_cols:
        if col_name in df.columns:
            df = df.with_columns(
                pl.col(col_name)
                .str.replace(r"\+00$", "+00:00")  # Fix timezone format
                .str.to_datetime(
                    format=None,  # Let polars infer
                    strict=False,
                    time_zone="UTC",
                )
                .alias(col_name)
            )

    # Cast planned_flag to a clean binary
    if "planned_flag" in df.columns:
        df = df.with_columns(
            pl.col("planned_flag")
            .cast(pl.Utf8)
            .str.to_lowercase()
            .str.strip_chars()
            .alias("planned_flag")
        )

    # Cast numeric columns
    numeric_cols = ["latitude", "longitude", "endlatitude", "endlongitude"]
    for col_name in numeric_cols:
        if col_name in df.columns:
            df = df.with_columns(pl.col(col_name).cast(pl.Float64, strict=False))

    # Log summary stats
    logger.info("Target distribution (requires_road_closure):")
    if "requires_road_closure" in df.columns:
        closure_counts = df.group_by("requires_road_closure").len().sort("requires_road_closure")
        for row in closure_counts.iter_rows(named=True):
            logger.info(
                "  %s: %d (%.1f%%)",
                row["requires_road_closure"],
                row["len"],
                row["len"] / df.height * 100,
            )

    if "planned_flag" in df.columns:
        logger.info("Event type distribution:")
        type_counts = df.group_by("planned_flag").len().sort("planned_flag")
        for row in type_counts.iter_rows(named=True):
            logger.info("  %s: %d", row["planned_flag"], row["len"])

    # Write interim Parquet
    df.write_parquet(interim_path)
    logger.info("Wrote interim Parquet: %s (%d rows)", interim_path, df.height)

    return df


if __name__ == "__main__":
    ingest()
