"""Time-based train/val/test splitting.

Uses temporal boundaries (not random) to prevent future information leakage:
- Train: events before Feb 15, 2024
- Validation: Feb 15 – Mar 15, 2024
- Test: after Mar 15, 2024
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


def create_splits(config: dict | None = None) -> pl.DataFrame:
    """Add a 'split' column (train/val/test) based on temporal boundaries.

    Args:
        config: Optional config dict. Loaded from configs/data.yaml if None.

    Returns:
        DataFrame with split column added.
    """
    if config is None:
        config = load_config()

    input_path = Path(config["paths"]["featured_parquet"])
    output_path = Path(config["paths"]["splits_parquet"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Reading featured Parquet: %s", input_path)
    df = pl.read_parquet(input_path)

    # Parse split boundaries
    split_config = config.get("splits", {})
    train_end = split_config.get("train_end", "2024-02-15")
    val_end = split_config.get("val_end", "2024-03-15")

    # Convert to datetime for comparison
    train_end_dt = pl.Series([train_end]).str.to_datetime("%Y-%m-%d", time_zone="UTC")[0]
    val_end_dt = pl.Series([val_end]).str.to_datetime("%Y-%m-%d", time_zone="UTC")[0]

    if "reported_datetime" not in df.columns:
        logger.error("No reported_datetime column — cannot create temporal splits")
        df = df.with_columns(pl.lit("train").alias("split"))
        df.write_parquet(output_path)
        return df

    # Assign splits
    df = df.with_columns(
        pl.when(pl.col("reported_datetime") < train_end_dt)
        .then(pl.lit("train"))
        .when(pl.col("reported_datetime") < val_end_dt)
        .then(pl.lit("val"))
        .otherwise(pl.lit("test"))
        .alias("split")
    )

    # Log split distribution
    split_counts = df.group_by("split").len().sort("split")
    logger.info("Temporal split distribution:")
    for row in split_counts.iter_rows(named=True):
        logger.info(
            "  %s: %d records (%.1f%%)",
            row["split"],
            row["len"],
            row["len"] / df.height * 100,
        )

    # Log target distribution per split (for imbalance awareness)
    if "requires_road_closure" in df.columns:
        logger.info("Target (requires_road_closure) distribution per split:")
        for split_name in ["train", "val", "test"]:
            split_df = df.filter(pl.col("split") == split_name)
            if split_df.height > 0:
                pos_rate = float(split_df["requires_road_closure"].cast(pl.Float64).mean() or 0)
                logger.info("  %s: %.1f%% positive", split_name, pos_rate * 100)

    # Validate no temporal leakage in target-encoded features
    # The target encoding should use only training data statistics
    logger.info("Temporal split validation: checking for leakage indicators...")
    if "cause_closure_rate" in df.columns:
        train_rate = df.filter(pl.col("split") == "train")["cause_closure_rate"].mean()
        test_rate = df.filter(pl.col("split") == "test")["cause_closure_rate"].mean()
        if train_rate is not None and test_rate is not None:
            logger.info(
                "  cause_closure_rate — train mean: %.4f, test mean: %.4f (should be similar but not identical)",
                train_rate,
                test_rate,
            )

    # Write final output
    df.write_parquet(output_path)
    logger.info("Wrote splits Parquet: %s (%d rows)", output_path, df.height)

    return df


if __name__ == "__main__":
    create_splits()
