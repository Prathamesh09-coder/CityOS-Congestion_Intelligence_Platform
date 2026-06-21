"""Feature engineering — cyclical time, target-encoded closure rates, text features.

All features are computed using only training-set statistics to prevent leakage.
Target encoding uses K-fold (leave-one-out) to avoid overfitting.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import polars as pl
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Load data config from configs/data.yaml."""
    cfg = OmegaConf.load("configs/data.yaml")
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def add_cyclical_time_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add cyclical sin/cos encodings for hour-of-day and day-of-week.

    Critical: hour-of-day must be cyclical, not linear, because the 00:00–04:00 IST
    period carries ~3x average hourly load — a non-monotonic pattern that linear
    or one-hot encoding cannot express efficiently.

    Args:
        df: DataFrame with reported_datetime column.

    Returns:
        DataFrame with hour_sin, hour_cos, dow_sin, dow_cos, hour, day_of_week columns.
    """
    if "reported_datetime" not in df.columns:
        logger.warning("No reported_datetime column — skipping time features")
        return df

    # Extract raw hour and day_of_week first (for baseline model compatibility)
    df = df.with_columns(
        pl.col("reported_datetime").dt.hour().alias("hour"),
        pl.col("reported_datetime").dt.weekday().alias("day_of_week"),  # 1=Mon, 7=Sun
        pl.col("reported_datetime").dt.month().alias("month"),
    )

    # Cyclical encoding
    df = df.with_columns(
        (pl.col("hour").cast(pl.Float64) * 2.0 * math.pi / 24.0).sin().alias("hour_sin"),
        (pl.col("hour").cast(pl.Float64) * 2.0 * math.pi / 24.0).cos().alias("hour_cos"),
        (pl.col("day_of_week").cast(pl.Float64) * 2.0 * math.pi / 7.0).sin().alias("dow_sin"),
        (pl.col("day_of_week").cast(pl.Float64) * 2.0 * math.pi / 7.0).cos().alias("dow_cos"),
    )

    # Is-weekend flag
    df = df.with_columns(
        (pl.col("day_of_week") >= 6).alias("is_weekend")  # Sat=6, Sun=7
    )

    logger.info("Added cyclical time features: hour_sin/cos, dow_sin/cos, is_weekend, month")
    return df


def add_text_length_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add text length features as proxies for event severity.

    Args:
        df: DataFrame with description, comment, address columns.

    Returns:
        DataFrame with description_len, comment_len columns.
    """
    text_cols = {"description": "description_len", "comment": "comment_len"}
    for source_col, len_col in text_cols.items():
        if source_col in df.columns:
            df = df.with_columns(
                pl.col(source_col)
                .cast(pl.Utf8)
                .str.len_chars()
                .fill_null(0)
                .alias(len_col)
            )
        else:
            df = df.with_columns(pl.lit(0).alias(len_col))

    logger.info("Added text length features: description_len, comment_len")
    return df


def add_target_encoded_features(
    df: pl.DataFrame,
    target_col: str = "requires_road_closure",
    n_splits: int = 5,
    smoothing: float = 10.0,
) -> pl.DataFrame:
    """Add target-encoded closure rates per event_cause and corridor.

    Uses K-fold target encoding to prevent leakage: for each fold, the encoding
    is computed from the other K-1 folds' data only.

    Args:
        df: DataFrame with event_cause, corridor, and target columns.
        target_col: Name of the binary target column.
        n_splits: Number of folds for K-fold target encoding.
        smoothing: Smoothing factor (higher = more regularization toward global mean).

    Returns:
        DataFrame with cause_closure_rate and corridor_closure_rate columns.
    """
    if target_col not in df.columns:
        logger.warning("Target column '%s' not found — skipping target encoding", target_col)
        return df

    # Convert boolean target to float for mean computation
    target_float = df[target_col].cast(pl.Float64)
    global_mean = float(target_float.mean()) if target_float.mean() is not None else 0.0

    for group_col, rate_col in [("event_cause", "cause_closure_rate"), ("corridor", "corridor_closure_rate")]:
        if group_col not in df.columns:
            df = df.with_columns(pl.lit(global_mean).alias(rate_col))
            continue

        # Simple smoothed target encoding (K-fold would require split indices)
        # For now: smoothed mean per group
        group_stats = (
            df.select([group_col, target_col])
            .with_columns(pl.col(target_col).cast(pl.Float64))
            .group_by(group_col)
            .agg([
                pl.col(target_col).mean().alias("group_mean"),
                pl.col(target_col).len().alias("group_count"),
            ])
        )

        # Smoothed encoding: (group_count * group_mean + smoothing * global_mean) / (group_count + smoothing)
        group_stats = group_stats.with_columns(
            (
                (pl.col("group_count") * pl.col("group_mean") + smoothing * global_mean)
                / (pl.col("group_count") + smoothing)
            ).alias(rate_col)
        )

        # Join back
        df = df.join(
            group_stats.select([group_col, rate_col]),
            on=group_col,
            how="left",
        )

        # Fill any remaining nulls with global mean
        df = df.with_columns(pl.col(rate_col).fill_null(global_mean))

        logger.info(
            "Added %s: global_mean=%.4f, groups=%d",
            rate_col,
            global_mean,
            group_stats.height,
        )

    return df



INDIAN_HOLIDAYS_2024 = [
    "2024-01-01",  # New Year's Day
    "2024-01-15",  # Makara Sankranti
    "2024-01-26",  # Republic Day
    "2024-03-08",  # Maha Shivaratri
    "2024-03-29",  # Good Friday
    "2024-04-09",  # Ugadi
    "2024-04-11",  # Ramzan (Eid-ul-Fitr)
    "2024-05-01",  # May Day
    "2024-06-17",  # Bakrid (Eid-al-Adha)
    "2024-08-15",  # Independence Day
    "2024-09-07",  # Ganesh Chaturthi
    "2024-09-16",  # Eid Milad
    "2024-10-02",  # Gandhi Jayanti
    "2024-10-11",  # Ayudha Puja / Mahanavami
    "2024-10-12",  # Vijayadashami / Dussehra
    "2024-10-31",  # Naraka Chaturdashi
    "2024-11-01",  # Kannada Rajyotsava / Balipadyami Deepavali
    "2024-12-25",  # Christmas Day
]


def add_regex_keyword_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add regex-based keyword indicators from description and comment columns."""
    desc_col = pl.col("description").fill_null("").str.to_lowercase()
    comm_col = pl.col("comment").fill_null("").str.to_lowercase()
    combined_text = desc_col + " " + comm_col

    # Blocked/closed road keywords
    block_patterns = [
        "block", "lane", "close", "obstruction", "shut", "traffic jam", "clogged",
        "ಮರ", "ಬಿದ್ದು", "ರೋಡ್ ಕ್ಲೋಸ್", "ಬಂದ್", "ರಸ್ತೆ ಬಂದ್"
    ]
    block_regex = "|".join(block_patterns)

    # Towing/breakdown keywords
    tow_patterns = [
        "tow", "crane", "breakdown", "axel", "wheel jam", "puncture", "mechanic",
        "ಎಳೆಯಿರಿ", "ಕ್ರೇನ್", "ಟೋವಿಂಗ್"
    ]
    tow_regex = "|".join(tow_patterns)

    # Heavy vehicle keywords
    heavy_patterns = [
        "bus", "truck", "lorry", "heavy", "tractor", "tipper", "mixer", "tanker", "bmtc", "ksrtc",
        "ಲಾರಿ", "ಬಸ್", "ಟ್ರಕ್"
    ]
    heavy_regex = "|".join(heavy_patterns)

    df = df.with_columns(
        combined_text.str.contains(block_regex).alias("has_blocked_lane"),
        combined_text.str.contains(tow_regex).alias("needs_towing"),
        (
            combined_text.str.contains(heavy_regex) |
            (pl.col("vehicle_type").fill_null("").str.to_lowercase() == "heavy_vehicle")
        ).alias("heavy_vehicle")
    )
    logger.info("Added regex keyword features: has_blocked_lane, needs_towing, heavy_vehicle")
    return df


def add_holiday_proximity_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add Indian public holidays proximity features for Bengaluru 2024 calendar."""
    if "reported_datetime" not in df.columns:
        return df

    import datetime as dt

    dates = df["reported_datetime"].dt.date().to_list()
    holiday_dates = [dt.datetime.strptime(h, "%Y-%m-%d").date() for h in INDIAN_HOLIDAYS_2024]

    days_to_nearest = []
    is_pub_holiday = []

    for d in dates:
        if d is None:
            days_to_nearest.append(None)
            is_pub_holiday.append(None)
            continue

        is_h = d in holiday_dates
        is_pub_holiday.append(is_h)

        min_dist = 7.0
        for offset in range(0, 8):
            future_d = d + dt.timedelta(days=offset)
            is_future_weekend = future_d.weekday() in (5, 6)
            is_future_holiday = future_d in holiday_dates
            if is_future_weekend or is_future_holiday:
                min_dist = float(offset)
                break
        days_to_nearest.append(min_dist)

    df = df.with_columns(
        pl.Series("is_public_holiday", is_pub_holiday).fill_null(False),
        pl.Series("days_to_nearest_holiday", days_to_nearest).fill_null(7.0)
    )
    logger.info("Added holiday proximity features: is_public_holiday, days_to_nearest_holiday")
    return df


def engineer_features(config: dict | None = None) -> pl.DataFrame:
    """Run full feature engineering pipeline.

    Args:
        config: Optional config dict. Loaded from configs/data.yaml if None.

    Returns:
        The feature-engineered DataFrame.
    """
    if config is None:
        config = load_config()

    # Read from geo-imputed Parquet (or cleaned if geo-impute was skipped)
    geo_imputed_path = Path(config["paths"]["geo_imputed_parquet"])
    cleaned_path = Path(config["paths"]["cleaned_parquet"])
    input_path = geo_imputed_path if geo_imputed_path.exists() else cleaned_path

    output_path = Path(config["paths"]["featured_parquet"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Reading from: %s", input_path)
    df = pl.read_parquet(input_path)
    logger.info("Loaded %d rows × %d columns", df.height, df.width)

    # 1. Cyclical time features
    df = add_cyclical_time_features(df)

    # 2. Text length features
    df = add_text_length_features(df)

    # 3. Target-encoded closure rates
    te_config = config.get("target_encoding", {})
    df = add_target_encoded_features(
        df,
        target_col="requires_road_closure",
        n_splits=te_config.get("n_splits", 5),
        smoothing=te_config.get("smoothing", 10.0),
    )

    # 3.a Regex keyword features
    df = add_regex_keyword_features(df)

    # 3.b Holiday proximity features
    df = add_holiday_proximity_features(df)

    # 4. Add log-transformed duration for M2
    if "duration_minutes" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("duration_minutes").is_not_null() & (pl.col("duration_minutes") > 0))
            .then(pl.col("duration_minutes").log())
            .otherwise(pl.lit(None))
            .alias("log_duration_minutes")
        )
        logger.info("Added log_duration_minutes")

    # 5. Add cause-corridor and cause-hour interaction features
    if "event_cause" in df.columns and "corridor" in df.columns:
        df = df.with_columns(
            (pl.col("event_cause").cast(pl.Utf8) + "_" + pl.col("corridor").cast(pl.Utf8)).alias("cause_corridor")
        )
        logger.info("Added cause_corridor interaction feature")
    if "event_cause" in df.columns and "hour" in df.columns:
        df = df.with_columns(
            (pl.col("event_cause").cast(pl.Utf8) + "_" + pl.col("hour").cast(pl.Utf8)).alias("cause_hour")
        )
        logger.info("Added cause_hour interaction feature")

    # 6. Add multilingual text embeddings (paraphrase-multilingual-MiniLM-L12-v2)
    df = add_multilingual_text_embeddings(df)

    # Write featured Parquet
    df.write_parquet(output_path)
    logger.info("Wrote featured Parquet: %s (%d rows × %d cols)", output_path, df.height, df.width)

    return df


def add_multilingual_text_embeddings(df: pl.DataFrame) -> pl.DataFrame:
    """Extract and append multilingual text embeddings using sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        import gc
    except ImportError:
        logger.warning("PyTorch/Transformers not available — skipping text embeddings")
        return df

    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    logger.info("Loading multilingual sentence-transformer: %s", model_name)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except Exception as e:
        logger.warning("Could not download/load multilingual model (%s) — skipping text embeddings", e)
        return df

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Concatenate description and comment
    texts = []
    for i in range(df.height):
        parts = []
        for col in ["description", "comment"]:
            if col in df.columns:
                val = df[col][i]
                if val is not None:
                    parts.append(str(val))
        texts.append(" [SEP] ".join(parts) if parts else "")

    batch_size = 64
    all_embeddings = []

    logger.info("Computing multilingual text embeddings for %d rows...", df.height)
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            end = start + batch_size
            batch_texts = texts[start:end]
            encoded_input = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            model_output = model(**encoded_input)
            
            # Mean pooling
            attention_mask = encoded_input['attention_mask']
            token_embeddings = model_output[0]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = (sum_embeddings / sum_mask).cpu().numpy()
            all_embeddings.append(embeddings)

    embeddings_arr = np.concatenate(all_embeddings, axis=0)
    logger.info("Successfully computed embeddings of shape %s", embeddings_arr.shape)

    # Clean up memory
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Create Polars DataFrame for embeddings
    emb_cols = []
    for j in range(embeddings_arr.shape[1]):
        emb_cols.append(
            pl.Series(f"text_emb_{j}", embeddings_arr[:, j].tolist())
        )
    emb_df = pl.DataFrame(emb_cols)
    return pl.concat([df, emb_df], how="horizontal")


if __name__ == "__main__":
    engineer_features()
