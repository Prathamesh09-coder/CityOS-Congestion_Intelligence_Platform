"""Geo-imputation for junction and zone fields using lat/long coordinates.

Two strategies:
- Zone: Approximate spatial clustering using existing labeled points (KNN)
  (or point-in-polygon against boundary shapefile if available).
- Junction: Snap lat/long to nearest OSM road graph node.

Adds `zone_imputed` / `junction_imputed` boolean flags distinct from missingness indicators.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl
from omegaconf import OmegaConf
from sklearn.neighbors import KNeighborsClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Load data config from configs/data.yaml."""
    cfg = OmegaConf.load("configs/data.yaml")
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def _impute_zone_by_clustering(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Impute missing zone values using KNN on lat/long from labeled rows.

    This is an approximation, not ground truth — uses spatial proximity
    to existing zone labels as a proxy for zone boundaries.

    Args:
        df: DataFrame with latitude, longitude, and zone columns.

    Returns:
        DataFrame with zone filled and zone_imputed flag added.
    """
    # Separate labeled vs unlabeled rows
    has_zone = df.filter(
        pl.col("zone").is_not_null()
        & pl.col("latitude").is_not_null()
        & pl.col("longitude").is_not_null()
    )
    needs_zone = df.filter(pl.col("zone").is_null())

    if has_zone.height == 0 or needs_zone.height == 0:
        logger.info("No zone imputation needed or no labeled data available")
        return df.with_columns(pl.lit(False).alias("zone_imputed"))

    logger.info(
        "Zone imputation: %d labeled rows → imputing %d rows",
        has_zone.height,
        needs_zone.height,
    )

    # Train KNN classifier on labeled data
    X_train = has_zone.select(["latitude", "longitude"]).to_numpy()
    y_train = has_zone["zone"].to_list()

    # Use k=5 neighbors, weighted by distance
    knn = KNeighborsClassifier(n_neighbors=min(5, has_zone.height), weights="distance")
    knn.fit(X_train, y_train)

    # Predict for unlabeled rows
    X_pred = needs_zone.select(["latitude", "longitude"]).to_numpy()
    if X_pred.shape[0] > 0 and not np.any(np.isnan(X_pred)):
        predicted_zones = knn.predict(X_pred)
    else:
        predicted_zones = [None] * needs_zone.height

    # Build the imputed column and flag
    # Create a new column that coalesces original zone with predicted
    zone_values = df["zone"].to_list()
    imputed_flags = [False] * df.height

    # Map back using row indices
    null_idx = 0
    for i in range(df.height):
        if zone_values[i] is None:
            if null_idx < len(predicted_zones) and predicted_zones[null_idx] is not None:
                zone_values[i] = predicted_zones[null_idx]
                imputed_flags[i] = True
            null_idx += 1

    df = df.with_columns(
        pl.Series("zone", zone_values),
        pl.Series("zone_imputed", imputed_flags),
    )

    return df


def _impute_junction_by_graph_snap(
    df: pl.DataFrame,
    config: dict,
) -> pl.DataFrame:
    """Impute missing junction values by snapping lat/long to nearest OSM graph node.

    Args:
        df: DataFrame with latitude, longitude, and junction columns.
        config: Data config dict.

    Returns:
        DataFrame with junction filled and junction_imputed flag added.
    """
    try:
        import osmnx as ox
        from astra_ml.data.road_graph import get_or_build_road_graph
    except ImportError:
        logger.warning(
            "osmnx not available — skipping junction graph-snap imputation. "
            "Install with: uv sync --extra deep"
        )
        return df.with_columns(pl.lit(False).alias("junction_imputed"))

    needs_junction = df.filter(
        pl.col("junction").is_null()
        & pl.col("latitude").is_not_null()
        & pl.col("longitude").is_not_null()
    )

    if needs_junction.height == 0:
        logger.info("No junction imputation needed")
        return df.with_columns(pl.lit(False).alias("junction_imputed"))

    logger.info("Junction imputation: snapping %d events to nearest graph nodes", needs_junction.height)

    # Get road graph
    graph = get_or_build_road_graph(config)

    # Snap each point to nearest node
    lats = needs_junction["latitude"].to_list()
    lons = needs_junction["longitude"].to_list()

    nearest_nodes = ox.distance.nearest_nodes(graph, lons, lats)

    # Use node ID as junction identifier (since we don't have junction names in graph)
    node_names = [str(n) for n in nearest_nodes]

    # Build result columns
    junction_values = df["junction"].to_list()
    imputed_flags = [False] * df.height

    null_idx = 0
    for i in range(df.height):
        if junction_values[i] is None:
            lat_val = df["latitude"][i]
            lon_val = df["longitude"][i]
            if lat_val is not None and lon_val is not None:
                if null_idx < len(node_names):
                    junction_values[i] = f"osm_node_{node_names[null_idx]}"
                    imputed_flags[i] = True
                null_idx += 1

    df = df.with_columns(
        pl.Series("junction", junction_values),
        pl.Series("junction_imputed", imputed_flags),
    )

    return df


def geo_impute(config: dict | None = None) -> pl.DataFrame:
    """Run geo-imputation pipeline for zone and junction.

    Args:
        config: Optional config dict. Loaded from configs/data.yaml if None.

    Returns:
        DataFrame with imputed zone/junction and imputation flags.
    """
    if config is None:
        config = load_config()

    geo_config = config.get("geo_impute", {})
    if not geo_config.get("enabled", True):
        logger.info("Geo-imputation disabled in config — skipping")
        cleaned_path = Path(config["paths"]["cleaned_parquet"])
        df = pl.read_parquet(cleaned_path)
        df = df.with_columns(
            pl.lit(False).alias("zone_imputed"),
            pl.lit(False).alias("junction_imputed"),
        )
        output_path = Path(config["paths"]["geo_imputed_parquet"])
        df.write_parquet(output_path)
        return df

    cleaned_path = Path(config["paths"]["cleaned_parquet"])
    output_path = Path(config["paths"]["geo_imputed_parquet"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Reading cleaned Parquet: %s", cleaned_path)
    df = pl.read_parquet(cleaned_path)

    # Log before stats
    zone_null_before = df.filter(pl.col("zone").is_null()).height if "zone" in df.columns else 0
    junction_null_before = df.filter(pl.col("junction").is_null()).height if "junction" in df.columns else 0
    logger.info("BEFORE imputation:")
    logger.info("  zone null: %d/%d (%.1f%%)", zone_null_before, df.height, zone_null_before / df.height * 100)
    logger.info("  junction null: %d/%d (%.1f%%)", junction_null_before, df.height, junction_null_before / df.height * 100)

    # Impute zone
    boundary_source = geo_config.get("zone_boundary_source", "approximate_clustering")
    if boundary_source == "approximate_clustering":
        logger.info("Using approximate spatial clustering for zone imputation")
        df = _impute_zone_by_clustering(df)
    else:
        # TODO: Implement point-in-polygon with shapefile
        logger.warning("Shapefile-based zone imputation not yet implemented — using clustering")
        df = _impute_zone_by_clustering(df)

    # Impute junction
    df = _impute_junction_by_graph_snap(df, config)

    # Log after stats
    zone_null_after = df.filter(pl.col("zone").is_null()).height if "zone" in df.columns else 0
    junction_null_after = df.filter(pl.col("junction").is_null()).height if "junction" in df.columns else 0
    logger.info("AFTER imputation:")
    logger.info("  zone null: %d/%d (%.1f%%)", zone_null_after, df.height, zone_null_after / df.height * 100)
    logger.info("  junction null: %d/%d (%.1f%%)", junction_null_after, df.height, junction_null_after / df.height * 100)
    logger.info("  zone records recovered: %d", zone_null_before - zone_null_after)
    logger.info("  junction records recovered: %d", junction_null_before - junction_null_after)

    # Write output
    df.write_parquet(output_path)
    logger.info("Wrote geo-imputed Parquet: %s", output_path)

    # Generate data quality report
    from astra_ml.eval.reports import write_data_quality_report

    # Get duration source stats
    duration_stats: dict = {"total": df.height}
    if "duration_source" in df.columns:
        for row in df.group_by("duration_source").len().iter_rows(named=True):
            duration_stats[row["duration_source"]] = row["len"]

    write_data_quality_report(
        report_path="reports/data_quality.md",
        imputation_stats={
            "total_records": df.height,
            "junction_null_before": junction_null_before,
            "junction_null_after": junction_null_after,
            "zone_null_before": zone_null_before,
            "zone_null_after": zone_null_after,
        },
        duration_source_stats=duration_stats,
    )
    logger.info("Wrote data quality report: reports/data_quality.md")

    return df


if __name__ == "__main__":
    geo_impute()
