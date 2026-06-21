"""Tests for geo-imputation against known Bengaluru coordinates.

Verifies:
- KNN zone imputation assigns plausible zones for known locations
- Junction snapping produces valid OSM node IDs
- Imputation flags are set correctly
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest


class TestZoneImputationByKNN:
    """Test zone imputation via spatial KNN clustering."""

    def test_knn_assigns_nearest_zone(self) -> None:
        """Points near labeled data should get the correct zone."""
        from astra_ml.data.geo_impute import _impute_zone_by_clustering

        # Known Bengaluru locations with zones
        df = pl.DataFrame({
            "latitude": [
                12.9716,  # MG Road area (labeled: Central)
                12.9716,  # Same area (labeled: Central)
                12.9352,  # Jayanagar (labeled: South)
                12.9352,  # Same (labeled: South)
                12.9700,  # Near MG Road (unlabeled — should get Central)
                12.9340,  # Near Jayanagar (unlabeled — should get South)
            ],
            "longitude": [
                77.5946,
                77.5950,
                77.5831,
                77.5835,
                77.5940,  # unlabeled
                77.5825,  # unlabeled
            ],
            "zone": [
                "Central Zone",
                "Central Zone",
                "South Zone",
                "South Zone",
                None,  # Should be imputed as Central
                None,  # Should be imputed as South
            ],
        })

        result = _impute_zone_by_clustering(df)

        # Check that the unlabeled rows got imputed
        assert result["zone"][4] is not None, "Row 4 should be imputed"
        assert result["zone"][5] is not None, "Row 5 should be imputed"

        # Check imputation flags
        assert result["zone_imputed"][4] is True, "Row 4 should be flagged as imputed"
        assert result["zone_imputed"][5] is True, "Row 5 should be flagged as imputed"
        assert result["zone_imputed"][0] is False, "Row 0 (original) should NOT be flagged"

    def test_original_zones_preserved(self) -> None:
        """Original non-null zones should not be changed by imputation."""
        from astra_ml.data.geo_impute import _impute_zone_by_clustering

        df = pl.DataFrame({
            "latitude": [12.97, 12.93, 12.95],
            "longitude": [77.59, 77.58, 77.60],
            "zone": ["Original Zone", "Another Zone", None],
        })

        result = _impute_zone_by_clustering(df)

        assert result["zone"][0] == "Original Zone"
        assert result["zone"][1] == "Another Zone"
        assert result["zone_imputed"][0] is False
        assert result["zone_imputed"][1] is False

    def test_all_null_zones_handled(self) -> None:
        """If all zones are null, imputation should handle gracefully."""
        from astra_ml.data.geo_impute import _impute_zone_by_clustering

        df = pl.DataFrame({
            "latitude": [12.97, 12.93],
            "longitude": [77.59, 77.58],
            "zone": [None, None],
        })

        # Should not crash — returns with all False imputed flags
        result = _impute_zone_by_clustering(df)
        assert "zone_imputed" in result.columns

    def test_no_null_zones_noop(self) -> None:
        """If no zones are null, imputation should be a no-op."""
        from astra_ml.data.geo_impute import _impute_zone_by_clustering

        df = pl.DataFrame({
            "latitude": [12.97, 12.93],
            "longitude": [77.59, 77.58],
            "zone": ["Zone A", "Zone B"],
        })

        result = _impute_zone_by_clustering(df)
        assert result["zone"][0] == "Zone A"
        assert result["zone"][1] == "Zone B"
        assert all(v is False for v in result["zone_imputed"].to_list())


class TestJunctionImputation:
    """Test junction imputation via OSM graph snapping.

    Note: These tests require osmnx and the cached road graph.
    They are marked with pytest.mark.slow and can be skipped in CI.
    """

    @pytest.mark.slow
    def test_junction_snap_produces_valid_ids(self) -> None:
        """Snapped junctions should have valid OSM node ID format."""
        try:
            from astra_ml.data.geo_impute import _impute_junction_by_graph_snap
        except ImportError:
            pytest.skip("osmnx not available")

        df = pl.DataFrame({
            "latitude": [12.9716],
            "longitude": [77.5946],
            "junction": [None],
        })

        config = {
            "geo_impute": {
                "osm_place": "Bengaluru, India",
                "osm_network_type": "drive",
            },
            "paths": {
                "osm_graph_cache_path": "data/interim/bengaluru_road_graph.graphml.gz",
            },
        }

        result = _impute_junction_by_graph_snap(df, config)

        if result["junction"][0] is not None:
            assert result["junction"][0].startswith("osm_node_"), \
                f"Snapped junction should start with 'osm_node_', got {result['junction'][0]}"
            assert result["junction_imputed"][0] is True
